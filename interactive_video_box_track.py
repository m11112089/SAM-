#!/usr/bin/env python3
"""
interactive_video_box_track.py – Interactive box-prompt SAM2-Plus tracker
==========================================================================
Extends the streaming tracker with an interactive bounding-box UI.
The user draws a box on the first frame (or supplies one via ``--box``),
and the tracker follows the object with continuous SAM2 memory.

Phase-2 rolling-buffer pruning is exposed through four new flags:

``--frame-buffer-size``
    Maximum number of raw frame tensors kept in the rolling buffer.
``--memory-window``
    Non-conditioning SAM2 output frames retained per pruning pass.
``--prune-every``
    Pruning is triggered every N processed frames.
``--print-prune-stats``
    Print a one-line memory summary after each pruning pass.

Usage::

    # Interactive box selection on a video file
    python interactive_video_box_track.py \\
        --source path/to/video.mp4 \\
        --checkpoint sam2_checkpoints/sam2_hiera_large.pt \\
        --config sam2_configs/sam2_hiera_l.yaml \\
        --output output.mp4 \\
        --frame-buffer-size 128 \\
        --memory-window 64 \\
        --prune-every 30 \\
        --print-prune-stats

    # Non-interactive: supply box directly
    python interactive_video_box_track.py \\
        --source path/to/video.mp4 \\
        --checkpoint sam2_checkpoints/sam2_hiera_large.pt \\
        --config sam2_configs/sam2_hiera_l.yaml \\
        --box 100 200 400 500 \\
        --no-display

All options from ``streaming_video_track.py`` are also available here.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Argument parser (superset of streaming_video_track.py)
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive box-prompt SAM2-Plus streaming tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- source / model ----
    p.add_argument("--source", required=True, help="Video file, webcam index, or RTSP URL")
    p.add_argument("--checkpoint", required=True, help="SAM2 model checkpoint (.pt)")
    p.add_argument("--config", required=True, help="SAM2 model config (.yaml)")
    # ---- prompt ----
    p.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Initial bounding box (skips interactive selection)",
    )
    p.add_argument("--obj-id", type=int, default=1, help="Object ID")
    # ---- output ----
    p.add_argument("--output", default=None, help="Output MP4 path")
    p.add_argument("--device", default=None, help="Torch device (cuda/cpu/mps)")
    p.add_argument("--resize", type=int, default=720, help="Long-edge resize limit (pixels)")
    # ---- Phase-2 rolling buffer / pruning ----
    p.add_argument(
        "--frame-buffer-size",
        type=int,
        default=128,
        help="Rolling frame buffer: max raw frame tensors to keep",
    )
    p.add_argument(
        "--memory-window",
        type=int,
        default=96,
        help="Number of non-conditioning SAM2 output frames retained per prune",
    )
    p.add_argument(
        "--prune-every",
        type=int,
        default=30,
        help="Trigger prune_stream_state() every N frames (0 = disabled)",
    )
    p.add_argument(
        "--print-prune-stats",
        action="store_true",
        help="Print memory stats after each pruning pass",
    )
    # ---- misc ----
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = unlimited)",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress all OpenCV windows",
    )
    return p


# ---------------------------------------------------------------------------
# Interactive box selector
# ---------------------------------------------------------------------------


class BoxSelector:
    """Allows the user to draw a bounding box on an OpenCV window.

    Usage::

        selector = BoxSelector(frame)
        box = selector.select()   # blocks until the user draws a box
    """

    _WINDOW = "Select object – drag to draw box, then press ENTER or SPACE"

    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame.copy()
        self._drawing = False
        self._start: Optional[Tuple[int, int]] = None
        self._end: Optional[Tuple[int, int]] = None
        self._confirmed = False

    def select(self) -> Optional[List[float]]:
        """Block until the user confirms a box.  Returns [x1,y1,x2,y2] or None."""
        if not _CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for interactive box selection.")

        cv2.namedWindow(self._WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._WINDOW, self._on_mouse)
        cv2.imshow(self._WINDOW, self._frame)

        while not self._confirmed:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):  # ENTER or SPACE
                if self._start and self._end:
                    self._confirmed = True
            elif key == 27:  # ESC – abort
                cv2.destroyWindow(self._WINDOW)
                return None

        cv2.destroyWindow(self._WINDOW)

        if self._start is None or self._end is None:
            return None

        x1 = min(self._start[0], self._end[0])
        y1 = min(self._start[1], self._end[1])
        x2 = max(self._start[0], self._end[0])
        y2 = max(self._start[1], self._end[1])
        return [float(x1), float(y1), float(x2), float(y2)]

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start = (x, y)
            self._end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._end = (x, y)
            display = self._frame.copy()
            cv2.rectangle(display, self._start, self._end, (0, 255, 0), 2)
            cv2.imshow(self._WINDOW, display)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._end = (x, y)
            display = self._frame.copy()
            cv2.rectangle(display, self._start, self._end, (0, 255, 0), 2)
            cv2.imshow(self._WINDOW, display)


# ---------------------------------------------------------------------------
# Helpers (shared with streaming_video_track.py)
# ---------------------------------------------------------------------------


def _auto_device() -> str:
    if _TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _open_source(source: str):
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for video input.")
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    return cap


def _scale_box(box: list, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
    sh, sw = src_hw
    dh, dw = dst_hw
    sx = dw / sw
    sy = dh / sh
    return np.array(
        [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy],
        dtype=np.float32,
    )


def _resize_frame(frame: np.ndarray, long_edge: int) -> np.ndarray:
    if not _CV2_AVAILABLE:
        return frame
    h, w = frame.shape[:2]
    scale = long_edge / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return frame
    return cv2.resize(
        frame,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_LINEAR,
    )


def _print_mem_stats(inference_state: dict, frame_idx: int) -> None:
    images = inference_state.get("images", [])
    non_none = sum(1 for x in images if x is not None)
    output = inference_state.get("output_dict", {})
    n_cond = len(output.get("cond_frame_outputs", {}))
    n_non_cond = len(output.get("non_cond_frame_outputs", {}))
    print(
        f"[prune stats] frame={frame_idx} "
        f"raw_frames_in_mem={non_none}/{len(images)} "
        f"cond_outputs={n_cond} non_cond_outputs={n_non_cond}"
    )


# ---------------------------------------------------------------------------
# Main interactive tracking loop
# ---------------------------------------------------------------------------


def run_interactive_tracker(args: argparse.Namespace) -> None:  # noqa: C901
    """Entry point for the interactive tracking pipeline."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        logger.error(
            "SAM2 is not installed.  Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything-2.git"
        )
        raise SystemExit(1) from exc

    from sam2_plus.streaming_video_predictor import (
        OcclusionAndDriftMonitor,
        RollingFrameStore,
        SAM2StreamingVideoPredictor,
        TrackingState,
        compute_frame_metrics,
        render_frame,
    )

    device = args.device or _auto_device()
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Build predictor with rolling frame store
    # ------------------------------------------------------------------
    base_predictor = build_sam2_video_predictor(
        args.config, args.checkpoint, device=device
    )
    predictor = SAM2StreamingVideoPredictor.from_predictor(base_predictor)
    predictor._image_size = args.resize

    frame_store = RollingFrameStore(
        frame_buffer_size=args.frame_buffer_size,
        max_non_conditioning_frames=min(args.memory_window, args.frame_buffer_size),
    )

    # ------------------------------------------------------------------
    # Open source and read first frame
    # ------------------------------------------------------------------
    cap = _open_source(args.source)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info("Source: %dx%d @ %.1f fps", src_w, src_h, src_fps)

    ret, first_bgr = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame.")

    first_bgr = _resize_frame(first_bgr, args.resize)
    dst_h, dst_w = first_bgr.shape[:2]

    # ------------------------------------------------------------------
    # Obtain initial bounding box (interactive or from CLI)
    # ------------------------------------------------------------------
    if args.box is not None:
        box = _scale_box(args.box, (src_h, src_w), (dst_h, dst_w))
    elif not args.no_display:
        logger.info("Please draw a bounding box around the object to track.")
        selector = BoxSelector(first_bgr)
        raw_box = selector.select()
        if raw_box is None:
            logger.error("No box selected. Aborting.")
            cap.release()
            return
        box = np.array(raw_box, dtype=np.float32)
    else:
        raise ValueError("Either --box or an interactive display is required.")

    logger.info("Initial box: %s", box.tolist())

    # ------------------------------------------------------------------
    # Init streaming state and add the first-frame prompt
    # ------------------------------------------------------------------
    inference_state = predictor.init_stream_state(first_bgr)
    frame_store.add(0, inference_state["images"][0])

    with torch.inference_mode():
        _, _, _ = predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=args.obj_id,
            box=box,
        )

    # Frame 0 is a conditioning frame: pin it in the store
    frame_store.pin_conditioning(0)

    # ------------------------------------------------------------------
    # Output sink
    # ------------------------------------------------------------------
    writer: Optional[cv2.VideoWriter] = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (dst_w, dst_h))
        if not writer.isOpened():
            logger.warning("Could not open VideoWriter for %s", args.output)
            writer = None

    # ------------------------------------------------------------------
    # Tracking helpers
    # ------------------------------------------------------------------
    monitor = OcclusionAndDriftMonitor()
    last_box: Optional[np.ndarray] = None
    frame_count = 0
    import time

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    logger.info("Starting interactive tracking loop…")

    try:
        while True:
            if args.max_frames > 0 and frame_count >= args.max_frames:
                logger.info("Reached max-frames limit (%d).", args.max_frames)
                break

            ret, bgr = cap.read()
            if not ret:
                logger.info("End of stream.")
                break

            bgr = _resize_frame(bgr, args.resize)

            # ---- Append frame ----
            with torch.inference_mode():
                frame_idx = predictor.append_frame(inference_state, bgr)

            frame_store.add(frame_idx, inference_state["images"][frame_idx])

            # ---- One-step propagation ----
            with torch.inference_mode():
                result = predictor.track_next_frame(
                    inference_state, frame_idx, obj_id=args.obj_id
                )

            mask = result["masks"][0]
            box_out = result["boxes"][0]
            score = result["scores"][0]

            # ---- Occlusion monitor ----
            metrics = compute_frame_metrics(
                frame_idx, mask, box_out, score, prev_box=last_box
            )
            track_state = monitor.update(metrics)

            if track_state == TrackingState.TRACKING and box_out is not None:
                last_box = box_out
                if frame_store.should_promote_keyframe(
                    frame_idx, confidence=score, is_occluded=False
                ):
                    frame_store.pin_conditioning(frame_idx)
                    logger.debug("Promoted frame %d as keyframe.", frame_idx)

            # ---- Render ----
            rendered = render_frame(bgr, mask, box_out, frame_idx, track_state, score)

            if writer is not None:
                writer.write(rendered)

            if not args.no_display:
                cv2.imshow("SAM2 Interactive Tracking", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("User quit.")
                    break
                # 'c' – request user correction
                if key == ord("c"):
                    logger.info("User correction requested at frame %d.", frame_idx)
                    monitor.request_correction()

            # ---- Prune ----
            if (
                args.prune_every > 0
                and frame_idx > 0
                and frame_idx % args.prune_every == 0
            ):
                keep_from = max(0, frame_idx - args.memory_window)
                predictor.prune_stream_state(
                    inference_state,
                    keep_from_frame_idx=keep_from,
                    keep_conditioning=True,
                )
                frame_store.prune_before(keep_from)
                if args.print_prune_stats:
                    _print_mem_stats(inference_state, frame_idx)

            frame_count += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Done. Processed %d frames in %.1f s (%.1f fps).",
        frame_count,
        elapsed,
        fps,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_interactive_tracker(args)


if __name__ == "__main__":
    main()
