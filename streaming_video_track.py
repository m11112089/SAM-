#!/usr/bin/env python3
"""
streaming_video_track.py – Streaming SAM2-Plus video tracker
=============================================================
Reads frames from a video file, webcam, or RTSP stream one at a time and
tracks a user-specified object using SAM2-Plus with a continuous memory state.

Usage examples::

    # Track using a bounding box prompt on frame 0 of a video file
    python streaming_video_track.py \\
        --source path/to/video.mp4 \\
        --checkpoint sam2_checkpoints/sam2_hiera_large.pt \\
        --config sam2_configs/sam2_hiera_l.yaml \\
        --box 100 200 400 500 \\
        --output output.mp4

    # Track from a webcam (device index 0)
    python streaming_video_track.py \\
        --source 0 \\
        --checkpoint sam2_checkpoints/sam2_hiera_large.pt \\
        --config sam2_configs/sam2_hiera_l.yaml \\
        --box 100 200 400 500

    # Track from an RTSP stream
    python streaming_video_track.py \\
        --source rtsp://user:pass@192.168.1.10:554/stream \\
        --checkpoint sam2_checkpoints/sam2_hiera_large.pt \\
        --config sam2_configs/sam2_hiera_l.yaml \\
        --box 100 200 400 500

Options
-------
--source        Path to video file, webcam device index, or RTSP URL.
--checkpoint    Path to SAM2 model checkpoint (.pt).
--config        Path to SAM2 model config (.yaml).
--box           Initial bounding box x1 y1 x2 y2 (in source-frame pixels).
--obj-id        Object ID to use for tracking (default: 1).
--output        Path for the rendered output MP4.  Omit to show live window.
--device        Torch device: cuda, cpu, mps (default: auto-detect).
--resize        Resize long edge of each frame to this value (default: 720).
--frame-buffer-size  Number of raw frames kept in the rolling buffer (default: 128).
--memory-window      How many frames of SAM2 output to retain (default: 96).
--prune-every        Prune memory every N frames (default: 30).
--print-prune-stats  Print memory usage after each pruning pass.
--max-frames         Stop after this many frames (0 = unlimited).
--no-display         Suppress the live preview window.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports (fail gracefully for import-time checks)
# ---------------------------------------------------------------------------

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
# Argument parser
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SAM2-Plus streaming video tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True, help="Video file, webcam index, or RTSP URL")
    p.add_argument("--checkpoint", required=True, help="SAM2 model checkpoint (.pt)")
    p.add_argument("--config", required=True, help="SAM2 model config (.yaml)")
    p.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="Initial bounding box in source-frame pixels",
    )
    p.add_argument("--obj-id", type=int, default=1, help="Object ID")
    p.add_argument("--output", default=None, help="Output MP4 path (None = live preview)")
    p.add_argument("--device", default=None, help="Torch device (cuda/cpu/mps)")
    p.add_argument("--resize", type=int, default=720, help="Long-edge resize limit (pixels)")
    p.add_argument(
        "--frame-buffer-size",
        type=int,
        default=128,
        help="Rolling frame buffer size",
    )
    p.add_argument(
        "--memory-window",
        type=int,
        default=96,
        help="Number of non-conditioning SAM2 output frames to retain",
    )
    p.add_argument(
        "--prune-every",
        type=int,
        default=30,
        help="Run prune_stream_state() every N frames",
    )
    p.add_argument(
        "--print-prune-stats",
        action="store_true",
        help="Print memory usage after each pruning pass",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = unlimited)",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the live OpenCV preview window",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_device() -> str:
    if _TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _open_source(source: str):
    """Return an OpenCV VideoCapture for the given source string or int."""
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
    """Scale a [x1,y1,x2,y2] box from *src_hw* to *dst_hw* resolution."""
    sh, sw = src_hw
    dh, dw = dst_hw
    sx = dw / sw
    sy = dh / sh
    return np.array(
        [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy],
        dtype=np.float32,
    )


def _resize_frame(frame: np.ndarray, long_edge: int) -> np.ndarray:
    """Resize *frame* so that its long edge equals *long_edge*."""
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
# Main tracking loop
# ---------------------------------------------------------------------------


def run_streaming_tracker(args: argparse.Namespace) -> None:  # noqa: C901
    """Entry point for the streaming tracker pipeline."""
    # ------------------------------------------------------------------
    # Imports that require the full SAM2 installation
    # ------------------------------------------------------------------
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
        SAM2StreamingVideoPredictor,
        TrackingState,
        compute_frame_metrics,
        render_frame,
    )

    device = args.device or _auto_device()
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Build predictor
    # ------------------------------------------------------------------
    base_predictor = build_sam2_video_predictor(
        args.config, args.checkpoint, device=device
    )
    predictor = SAM2StreamingVideoPredictor.from_predictor(base_predictor)
    predictor._image_size = args.resize

    # ------------------------------------------------------------------
    # Open source
    # ------------------------------------------------------------------
    cap = _open_source(args.source)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info("Source: %dx%d @ %.1f fps", src_w, src_h, src_fps)

    # ------------------------------------------------------------------
    # Read and resize the first frame, init streaming state
    # ------------------------------------------------------------------
    ret, first_bgr = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame.")

    first_bgr = _resize_frame(first_bgr, args.resize)
    dst_h, dst_w = first_bgr.shape[:2]

    inference_state = predictor.init_stream_state(first_bgr)

    # Scale box from source resolution to resized resolution
    box = _scale_box(args.box, (src_h, src_w), (dst_h, dst_w))

    # Add the initial prompt (box on frame 0)
    with torch.inference_mode():
        _, _, _ = predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=args.obj_id,
            box=box,
        )

    # ------------------------------------------------------------------
    # Open output sink (if requested)
    # ------------------------------------------------------------------
    writer: Optional[cv2.VideoWriter] = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (dst_w, dst_h))
        if not writer.isOpened():
            logger.warning("Could not open VideoWriter for %s", args.output)
            writer = None

    # ------------------------------------------------------------------
    # Tracking state helpers
    # ------------------------------------------------------------------
    monitor = OcclusionAndDriftMonitor()
    last_box: Optional[np.ndarray] = None
    frame_count = 0

    # ------------------------------------------------------------------
    # Tracking loop
    # ------------------------------------------------------------------
    logger.info("Starting streaming tracking loop…")
    t_start = time.perf_counter()

    try:
        while True:
            if args.max_frames > 0 and frame_count >= args.max_frames:
                logger.info("Reached max-frames limit (%d).", args.max_frames)
                break

            ret, bgr = cap.read()
            if not ret:
                logger.info("End of stream / source exhausted.")
                break

            bgr = _resize_frame(bgr, args.resize)

            # ---- Append frame to streaming state ----
            with torch.inference_mode():
                frame_idx = predictor.append_frame(inference_state, bgr)

            # ---- Run one-step propagation ----
            with torch.inference_mode():
                result = predictor.track_next_frame(
                    inference_state, frame_idx, obj_id=args.obj_id
                )

            mask = result["masks"][0]
            box_out = result["boxes"][0]
            score = result["scores"][0]

            # ---- Update occlusion monitor ----
            metrics = compute_frame_metrics(
                frame_idx, mask, box_out, score, prev_box=last_box
            )
            track_state = monitor.update(metrics)

            if track_state == TrackingState.TRACKING and box_out is not None:
                last_box = box_out

            # ---- Render ----
            rendered = render_frame(bgr, mask, box_out, frame_idx, track_state, score)

            if writer is not None:
                writer.write(rendered)

            if not args.no_display:
                cv2.imshow("SAM2 Streaming", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    logger.info("User quit.")
                    break

            # ---- Prune every N frames ----
            if args.prune_every > 0 and frame_idx % args.prune_every == 0 and frame_idx > 0:
                keep_from = max(0, frame_idx - args.memory_window)
                predictor.prune_stream_state(
                    inference_state,
                    keep_from_frame_idx=keep_from,
                    keep_conditioning=True,
                )
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
    run_streaming_tracker(args)


if __name__ == "__main__":
    main()
