"""
SAM2-Plus Streaming Video Predictor
====================================
Extends SAM2VideoPredictor_Plus with an online, frame-by-frame tracking API
that avoids loading the entire video into memory.

Architecture:
    FrameSource -> FramePreprocessor -> StreamingSAM2State
                -> TrackerController -> OcclusionAndDriftMonitor
                -> Renderer -> VideoSink

Key classes:
    RollingFrameStore           – bounded ring buffer for preprocessed frames
    OcclusionAndDriftMonitor    – state machine: TRACKING/UNCERTAIN/OCCLUDED/LOST
    SAM2StreamingVideoPredictor – subclass that adds the streaming API

Usage (Phase 1 – one frame at a time, no pruning)::

    predictor = SAM2StreamingVideoPredictor.from_pretrained(...)
    inference_state = predictor.init_stream_state(first_frame)
    predictor.add_new_points_or_box(inference_state, frame_idx=0, obj_id=1,
                                    box=bbox_xyxy)
    for frame in source:
        idx = predictor.append_frame(inference_state, frame)
        out = predictor.track_next_frame(inference_state, idx, obj_id=1)
        render(out)

Usage (Phase 2 – with rolling buffer and pruning)::

    predictor.prune_stream_state(inference_state,
                                 keep_from_frame_idx=idx - memory_window)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports so the module is importable even when torch/cv2 are absent
# (useful for unit-testing the pure-Python logic without a GPU environment).
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tracking state enum
# ---------------------------------------------------------------------------


class TrackingState(Enum):
    """High-level state produced by OcclusionAndDriftMonitor."""

    TRACKING = auto()
    UNCERTAIN = auto()
    OCCLUDED = auto()
    LOST = auto()
    NEEDS_USER_CORRECTION = auto()


# ---------------------------------------------------------------------------
# Rolling frame store
# ---------------------------------------------------------------------------


class RollingFrameStore:
    """Bounded ring buffer of preprocessed frame tensors.

    Stores up to *frame_buffer_size* recent frames for rendering and
    re-use as SAM2 conditioning.  Conditioning (keyframe) slots are
    pinned and not evicted by the normal LRU policy.

    Args:
        frame_buffer_size: Maximum number of frames kept in the store.
        keyframe_interval: Minimum gap (in frames) between promoted
            conditioning keyframes.
        max_conditioning_frames: Upper limit on pinned conditioning slots.
        max_non_conditioning_frames: Upper limit on recent non-pinned slots.
    """

    def __init__(
        self,
        frame_buffer_size: int = 128,
        keyframe_interval: int = 20,
        max_conditioning_frames: int = 16,
        max_non_conditioning_frames: int = 96,
    ) -> None:
        self.frame_buffer_size = frame_buffer_size
        self.keyframe_interval = keyframe_interval
        self.max_conditioning_frames = max_conditioning_frames
        self.max_non_conditioning_frames = max_non_conditioning_frames

        # frame_idx -> preprocessed tensor (C×H×W, float32, [0,1])
        self._frames: OrderedDict[int, Optional[object]] = OrderedDict()
        # set of pinned conditioning frame indices
        self._conditioning_set: set = set()
        self._last_keyframe_idx: int = -1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, frame_idx: int, tensor) -> None:
        """Store a preprocessed frame tensor."""
        self._frames[frame_idx] = tensor
        self._evict_if_needed()

    def get(self, frame_idx: int):
        """Return the tensor for *frame_idx*, or None if evicted."""
        return self._frames.get(frame_idx)

    def pin_conditioning(self, frame_idx: int) -> None:
        """Pin *frame_idx* as a conditioning keyframe (not evicted by LRU)."""
        self._conditioning_set.add(frame_idx)
        self._last_keyframe_idx = frame_idx
        self._evict_conditioning_if_needed()

    def should_promote_keyframe(
        self, frame_idx: int, confidence: float, is_occluded: bool
    ) -> bool:
        """Return True when *frame_idx* qualifies as a new keyframe.

        Criteria (all must hold):
        - Not occluded.
        - Confidence above 0.5.
        - At least *keyframe_interval* frames after the last keyframe.
        """
        if is_occluded:
            return False
        if confidence < 0.5:
            return False
        if frame_idx - self._last_keyframe_idx < self.keyframe_interval:
            return False
        return True

    def drop_frame(self, frame_idx: int) -> None:
        """Replace the raw tensor with None (frees GPU/CPU memory)."""
        if frame_idx in self._frames:
            self._frames[frame_idx] = None

    def prune_before(self, keep_from: int) -> List[int]:
        """Drop all non-conditioning frames with index < *keep_from*.

        Returns the list of evicted frame indices.
        """
        evicted: List[int] = []
        for idx in list(self._frames.keys()):
            if idx < keep_from and idx not in self._conditioning_set:
                self._frames[idx] = None
                evicted.append(idx)
        return evicted

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    @property
    def conditioning_frames(self) -> List[int]:
        return sorted(self._conditioning_set)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        # Only count non-None non-conditioning frames against the limit so that
        # already-evicted entries don't inflate the count.
        non_cond = [
            k for k in self._frames
            if k not in self._conditioning_set and self._frames[k] is not None
        ]
        while len(non_cond) > self.max_non_conditioning_frames:
            oldest = non_cond.pop(0)
            self._frames[oldest] = None

    def _evict_conditioning_if_needed(self) -> None:
        live_cond = [k for k in self._conditioning_set if self._frames.get(k) is not None]
        while len(live_cond) > self.max_conditioning_frames:
            oldest = min(live_cond)
            live_cond.remove(oldest)
            self._conditioning_set.discard(oldest)
            self._frames[oldest] = None


# ---------------------------------------------------------------------------
# Occlusion and drift monitor
# ---------------------------------------------------------------------------


@dataclass
class FrameMetrics:
    """Scalar quality indicators computed from one SAM2 output frame."""

    frame_idx: int
    mask_area: float = 0.0
    bbox_area: float = 0.0
    bbox_aspect: float = 1.0
    object_score: float = 0.0
    logit_strength: float = 0.0
    iou_with_prev: float = 1.0
    center_distance: float = 0.0


class OcclusionAndDriftMonitor:
    """Finite-state machine that tracks object visibility quality.

    Transitions::

        TRACKING  <-> UNCERTAIN  <-> OCCLUDED  -> LOST
                                               -> NEEDS_USER_CORRECTION

    Args:
        score_threshold: Object score below which frame is *uncertain*.
        occluded_threshold: Object score below which frame is *occluded*.
        lost_grace_frames: Consecutive occluded frames before LOST.
        area_change_threshold: Relative area change flagging drift.
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        occluded_threshold: float = 0.1,
        lost_grace_frames: int = 30,
        area_change_threshold: float = 3.0,
    ) -> None:
        self.score_threshold = score_threshold
        self.occluded_threshold = occluded_threshold
        self.lost_grace_frames = lost_grace_frames
        self.area_change_threshold = area_change_threshold

        self.state: TrackingState = TrackingState.TRACKING
        self._occluded_count: int = 0
        self._prev_metrics: Optional[FrameMetrics] = None
        self.last_good_metrics: Optional[FrameMetrics] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, metrics: FrameMetrics) -> TrackingState:
        """Ingest *metrics* and return the new state."""
        new_state = self._compute_state(metrics)
        self._transition(new_state, metrics)
        return self.state

    def reset(self) -> None:
        """Reset after a successful user correction."""
        self.state = TrackingState.TRACKING
        self._occluded_count = 0

    def request_correction(self) -> None:
        """External call when the operator wants to intervene."""
        self.state = TrackingState.NEEDS_USER_CORRECTION

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_state(self, m: FrameMetrics) -> TrackingState:
        if m.object_score < self.occluded_threshold:
            return TrackingState.OCCLUDED
        if m.object_score < self.score_threshold:
            return TrackingState.UNCERTAIN
        # Check for sudden area explosion (possible drift)
        if self._prev_metrics is not None and self._prev_metrics.bbox_area > 0:
            ratio = m.bbox_area / max(self._prev_metrics.bbox_area, 1.0)
            if ratio > self.area_change_threshold or ratio < 1.0 / self.area_change_threshold:
                return TrackingState.UNCERTAIN
        return TrackingState.TRACKING

    def _transition(self, new_state: TrackingState, metrics: FrameMetrics) -> None:
        if new_state == TrackingState.OCCLUDED:
            self._occluded_count += 1
            if self._occluded_count >= self.lost_grace_frames:
                self.state = TrackingState.LOST
            else:
                self.state = TrackingState.OCCLUDED
        else:
            self._occluded_count = 0
            self.state = new_state
            if new_state == TrackingState.TRACKING:
                self.last_good_metrics = metrics

        self._prev_metrics = metrics


# ---------------------------------------------------------------------------
# Frame packet (output from FrameSource)
# ---------------------------------------------------------------------------


@dataclass
class FramePacket:
    """Minimal wrapper for one raw camera/file frame."""

    frame_id: int
    timestamp: float
    bgr: np.ndarray


# ---------------------------------------------------------------------------
# SAM2StreamingVideoPredictor
# ---------------------------------------------------------------------------


class SAM2StreamingVideoPredictor:
    """Online, frame-by-frame SAM2-Plus tracker.

    This class **wraps** an existing SAM2 predictor instance (via
    :meth:`from_predictor`) and delegates all standard SAM2 calls to it, while
    adding a streaming API for one-frame-at-a-time tracking.  It is designed
    to be compatible with ``SAM2VideoPredictor_Plus`` but does not require
    inheriting from it, so the file can be loaded even when SAM2 is not
    installed.

    When used for real tracking, create a base predictor via the SAM2 factory
    and wrap it::

        from sam2.build_sam import build_sam2_video_predictor
        base = build_sam2_video_predictor(config, checkpoint, device=device)
        predictor = SAM2StreamingVideoPredictor.from_predictor(base)

    The streaming API consists of four methods:

    - :meth:`init_stream_state`  – bootstrap an inference_state from frame 0
    - :meth:`append_frame`       – add the next frame tensor and return its idx
    - :meth:`track_next_frame`   – run one propagation step for the new frame
    - :meth:`prune_stream_state` – evict old frames/outputs to bound memory

    Implementation note on internal SAM2 APIs
    ------------------------------------------
    :meth:`_build_state_from_predictor` calls the private ``_init_state``
    method on the wrapped predictor because SAM2 does not expose a public API
    for bootstrapping state from a pre-loaded image list.  If the SAM2 API
    changes, the fallback path (stub state) is used automatically.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_predictor(cls, predictor) -> "SAM2StreamingVideoPredictor":
        """Wrap an existing SAM2 predictor instance.

        The returned object delegates all standard SAM2 calls to *predictor*
        while adding the streaming API.
        """
        instance = cls.__new__(cls)
        instance._predictor = predictor
        instance._image_size = getattr(predictor, "image_size", 1024)
        return instance

    def __init__(self, image_size: int = 1024) -> None:
        """Stand-alone constructor (no GPU / SAM2 required)."""
        self._predictor = None
        self._image_size = image_size

    # ------------------------------------------------------------------
    # Core streaming API
    # ------------------------------------------------------------------

    def init_stream_state(
        self,
        first_frame: np.ndarray,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict:
        """Bootstrap an *inference_state* from a single BGR frame.

        If a real SAM2 predictor is attached, this calls the underlying
        ``_init_state_from_image_list`` helper with just the first frame.
        Otherwise it builds a minimal stub state suitable for unit testing.

        Args:
            first_frame: BGR image as a ``np.ndarray`` (H×W×3, uint8).
            offload_video_to_cpu: If True, keep frame tensors on CPU.
            offload_state_to_cpu: If True, keep state tensors on CPU.

        Returns:
            A mutable ``inference_state`` dict that can be passed to all
            other streaming methods.
        """
        tensor = self._preprocess_frame(first_frame)

        if self._predictor is not None:
            state = self._build_state_from_predictor(
                tensor,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )
        else:
            state = self._build_stub_state(tensor)

        state["streaming"] = True
        state["stream_frame_offset"] = 0
        state["images"] = [tensor]
        state["num_frames"] = 1

        return state

    def append_frame(
        self,
        inference_state: dict,
        frame: np.ndarray,
    ) -> int:
        """Append one preprocessed frame and return its *state frame index*.

        The returned index is what subsequent calls to :meth:`track_next_frame`
        and :meth:`prune_stream_state` expect.

        Args:
            inference_state: Mutable state produced by :meth:`init_stream_state`.
            frame: Next BGR frame as a ``np.ndarray`` (H×W×3, uint8).

        Returns:
            The zero-based index of the appended frame within *inference_state*.
        """
        if not inference_state.get("streaming"):
            raise ValueError(
                "inference_state was not created by init_stream_state(). "
                "Use init_stream_state() for streaming mode."
            )

        tensor = self._preprocess_frame(frame)
        inference_state["images"].append(tensor)
        inference_state["num_frames"] += 1

        # Invalidate any cached image feature for this slot if it existed
        frame_idx = inference_state["num_frames"] - 1
        cached = inference_state.get("cached_features", {})
        cached.pop(frame_idx, None)

        logger.debug("Appended frame %d (total=%d)", frame_idx, inference_state["num_frames"])
        return frame_idx

    def track_next_frame(
        self,
        inference_state: dict,
        frame_idx: int,
        obj_id: int,
    ) -> dict:
        """Run one propagation step for a single appended frame.

        This is equivalent to calling ``propagate_in_video()`` with
        ``processing_order=[frame_idx]``, but without requiring future frames
        to exist.

        Args:
            inference_state: Streaming state.
            frame_idx: Index returned by :meth:`append_frame`.
            obj_id: Object identifier used when the initial prompt was added.

        Returns:
            A dict with keys ``masks``, ``boxes``, ``scores``, ``frame_idx``,
            ``obj_id``.  ``masks`` is a list of ``np.ndarray`` (H×W, bool).
            ``boxes`` is a list of ``[x1,y1,x2,y2]`` arrays.  ``scores`` is
            a list of floats.
        """
        if self._predictor is not None:
            return self._track_with_predictor(inference_state, frame_idx, obj_id)

        # Stub behaviour for unit testing / no-GPU environments
        return self._track_stub(inference_state, frame_idx, obj_id)

    def prune_stream_state(
        self,
        inference_state: dict,
        keep_from_frame_idx: int,
        keep_conditioning: bool = True,
    ) -> None:
        """Evict old frame tensors and output dicts to bound memory.

        Frames with index < *keep_from_frame_idx* are dropped from
        ``inference_state["images"]`` (replaced with ``None``) and from
        the non-conditioning output dicts.

        Args:
            inference_state: Streaming state (modified in-place).
            keep_from_frame_idx: Drop everything before this index.
            keep_conditioning: If True, conditioning (user-prompted) frame
                outputs are never dropped, regardless of their index.
        """
        if not inference_state.get("streaming"):
            raise ValueError(
                "prune_stream_state() called on a non-streaming state."
            )

        images: list = inference_state.get("images", [])
        # Replace old raw tensors with None to release GPU/CPU memory
        for i in range(min(keep_from_frame_idx, len(images))):
            if images[i] is not None:
                images[i] = None
                logger.debug("Pruned raw frame tensor at index %d", i)

        # Prune per-object output dicts
        output_dict = inference_state.get("output_dict", {})
        self._prune_output_dict(
            output_dict, keep_from_frame_idx, keep_conditioning
        )

        output_dict_per_obj = inference_state.get("output_dict_per_obj", {})
        for obj_output in output_dict_per_obj.values():
            self._prune_output_dict(
                obj_output, keep_from_frame_idx, keep_conditioning
            )

        # Prune frames_tracked_per_obj metadata
        frames_tracked = inference_state.get("frames_tracked_per_obj", {})
        for obj_id, tracked_set in frames_tracked.items():
            if isinstance(tracked_set, set):
                frames_tracked[obj_id] = {
                    f for f in tracked_set if f >= keep_from_frame_idx
                }

        # Prune cached features for old frames
        cached = inference_state.get("cached_features", {})
        for idx in list(cached.keys()):
            if idx < keep_from_frame_idx:
                del cached[idx]
                logger.debug("Pruned cached features at index %d", idx)

        logger.debug(
            "Pruned stream state: keep_from=%d, images_len=%d",
            keep_from_frame_idx,
            len(images),
        )

    # ------------------------------------------------------------------
    # Convenience wrappers that delegate to the wrapped predictor
    # ------------------------------------------------------------------

    def add_new_points_or_box(self, inference_state, **kwargs):
        """Delegate to the underlying predictor."""
        if self._predictor is None:
            raise RuntimeError("No predictor attached. Use from_predictor().")
        return self._predictor.add_new_points_or_box(inference_state, **kwargs)

    def add_new_mask(self, inference_state, **kwargs):
        """Delegate to the underlying predictor."""
        if self._predictor is None:
            raise RuntimeError("No predictor attached. Use from_predictor().")
        return self._predictor.add_new_mask(inference_state, **kwargs)

    def reset_state(self, inference_state: dict) -> None:
        """Clear all tracking state for a fresh prompt."""
        if self._predictor is not None:
            self._predictor.reset_state(inference_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_frame(self, bgr: np.ndarray):
        """Convert a BGR uint8 frame to a model-ready float tensor.

        When torch is available, returns a CPU float32 tensor (3×H×W).
        Otherwise, returns the numpy array (unit-tests / no-GPU path).
        """
        if not _CV2_AVAILABLE or not _TORCH_AVAILABLE:
            # Fallback: just return the numpy array
            return bgr

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Resize long edge to model image size
        h, w = rgb.shape[:2]
        scale = self._image_size / max(h, w)
        if abs(scale - 1.0) > 1e-3:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(rgb)
        # Return early for numpy arrays (test stubs) or any object that lacks
        # torch-tensor methods, to avoid AttributeError in non-GPU environments.
        if isinstance(tensor, np.ndarray) or not hasattr(tensor, "permute"):
            return tensor
        return tensor.permute(2, 0, 1).float().div(255.0)

    def _build_stub_state(self, first_tensor) -> dict:
        """Build a minimal inference_state for unit-testing (no GPU needed)."""
        return {
            "images": [],
            "num_frames": 0,
            "cached_features": {},
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            "output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
            "obj_ids": [],
            "obj_id_to_idx": {},
            "obj_idx_to_id": {},
        }

    def _build_state_from_predictor(
        self,
        first_tensor,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
    ) -> dict:
        """Use the wrapped predictor to build state from a single frame."""
        # Call the internal SAM2 init helper with a one-element list
        try:
            state = self._predictor._init_state(
                video_path=None,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
                async_loading_frames=False,
                _images=[first_tensor],
                _num_frames=1,
            )
        except TypeError:
            # Older SAM2 API: fall back to stub and warn
            logger.warning(
                "SAM2 predictor does not support _images kwarg in _init_state; "
                "using stub state."
            )
            state = self._build_stub_state(first_tensor)
        return state

    @staticmethod
    def _prune_output_dict(
        output_dict: dict,
        keep_from: int,
        keep_conditioning: bool,
    ) -> None:
        """Remove entries from a SAM2 output dict that are before *keep_from*."""
        non_cond: dict = output_dict.get("non_cond_frame_outputs", {})
        for idx in [k for k in non_cond if k < keep_from]:
            del non_cond[idx]

        if not keep_conditioning:
            cond: dict = output_dict.get("cond_frame_outputs", {})
            for idx in [k for k in cond if k < keep_from]:
                del cond[idx]

    def _track_with_predictor(
        self,
        inference_state: dict,
        frame_idx: int,
        obj_id: int,
    ) -> dict:
        """Run one propagation step using the real SAM2 predictor."""
        results: Dict[int, dict] = {}

        # Use propagate_in_video with a single-frame processing order
        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in (
                self._predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=1,
                    reverse=False,
                )
            ):
                for oi, oid in enumerate(out_obj_ids):
                    mask = (out_mask_logits[oi] > 0.0).squeeze().cpu().numpy()
                    score = float(out_mask_logits[oi].max().cpu())
                    boxes = _mask_to_box(mask)
                    results[oid] = {
                        "mask": mask,
                        "box": boxes,
                        "score": score,
                    }
        except Exception as exc:
            logger.error("propagate_in_video failed at frame %d: %s", frame_idx, exc)
            raise

        obj_result = results.get(obj_id, {"mask": None, "box": None, "score": 0.0})
        return {
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "masks": [obj_result["mask"]],
            "boxes": [obj_result["box"]],
            "scores": [obj_result["score"]],
        }

    @staticmethod
    def _track_stub(
        inference_state: dict,
        frame_idx: int,
        obj_id: int,
    ) -> dict:
        """Return a zeroed result for unit testing without a GPU."""
        images = inference_state.get("images", [])
        if frame_idx < len(images) and images[frame_idx] is not None:
            img = images[frame_idx]
            if _TORCH_AVAILABLE and hasattr(img, "shape"):
                h, w = img.shape[-2], img.shape[-1]
            elif isinstance(img, np.ndarray):
                h, w = img.shape[:2]
            else:
                h, w = 256, 256
        else:
            h, w = 256, 256

        mask = np.zeros((h, w), dtype=bool)
        return {
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "masks": [mask],
            "boxes": [None],
            "scores": [0.0],
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    """Convert a boolean mask to ``[x1, y1, x2, y2]``, or None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def compute_frame_metrics(
    frame_idx: int,
    mask: Optional[np.ndarray],
    box: Optional[np.ndarray],
    score: float,
    prev_box: Optional[np.ndarray] = None,
) -> FrameMetrics:
    """Derive :class:`FrameMetrics` from raw SAM2 outputs.

    Args:
        frame_idx: Current frame index.
        mask: Boolean mask (H×W) or None.
        box: Bounding box [x1,y1,x2,y2] or None.
        score: Object score logit (higher is better).
        prev_box: Previous frame bounding box for IoU / drift estimation.

    Returns:
        Populated :class:`FrameMetrics`.
    """
    m = FrameMetrics(frame_idx=frame_idx, object_score=score, logit_strength=score)

    if mask is not None:
        m.mask_area = float(mask.sum())

    if box is not None:
        w = float(box[2] - box[0])
        h = float(box[3] - box[1])
        m.bbox_area = w * h
        m.bbox_aspect = (w / h) if h > 0 else 1.0

        if prev_box is not None:
            cx_cur = (box[0] + box[2]) / 2.0
            cy_cur = (box[1] + box[3]) / 2.0
            cx_prev = (prev_box[0] + prev_box[2]) / 2.0
            cy_prev = (prev_box[1] + prev_box[3]) / 2.0
            m.center_distance = float(
                np.sqrt((cx_cur - cx_prev) ** 2 + (cy_cur - cy_prev) ** 2)
            )
            m.iou_with_prev = _box_iou(box, prev_box)

    return m


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two ``[x1,y1,x2,y2]`` boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def render_frame(
    bgr: np.ndarray,
    mask: Optional[np.ndarray],
    box: Optional[np.ndarray],
    frame_idx: int,
    state: TrackingState,
    score: float,
    mask_color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay mask, bbox, frame-id, and state label on *bgr*.

    Returns a new ``np.ndarray`` (H×W×3, uint8).  No-op when cv2 is absent.
    """
    out = bgr.copy()

    if not _CV2_AVAILABLE:
        return out

    if mask is not None:
        overlay = out.copy()
        overlay[mask] = mask_color
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    if box is not None:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = (0, 255, 0) if state == TrackingState.TRACKING else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    label = f"f={frame_idx} {state.name} s={score:.2f}"
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return out
