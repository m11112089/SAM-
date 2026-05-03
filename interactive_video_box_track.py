import argparse
from contextlib import nullcontext
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


WINDOW_NAME = "SAM2-Plus tracker - click top-left and bottom-right"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Track one object in a video with SAM2-Plus. The first positional "
            "argument is the input video. Click the target's top-left and "
            "bottom-right corners on the first frame."
        )
    )
    parser.add_argument("input_video", help="Input video path, e.g. input.mp4")
    parser.add_argument(
        "--output",
        help="Output video path. Default: <input_stem>_sam2pp_tracked.mp4",
    )
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml",
        help="SAM2-Plus config file relative to this repo.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/SAM2-Plus/checkpoint_phase123.pt",
        help="SAM2-Plus checkpoint path relative to this repo.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=180,
        help="Maximum frames to track. Use 0 to process the whole video. Default: 180 for CPU memory safety.",
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=720,
        help="Resize extracted frames so the long edge is at most this value. Use 0 to keep original size. Default: 720.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every Nth frame. Default: 1.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help=(
            "Deprecated; phase-1 streaming keeps one continuous state and ignores this option."
        ),
    )
    parser.add_argument(
        "--frame-buffer-size",
        type=int,
        default=16,
        help="Number of recent normalized frames to keep in streaming state. Default: 16.",
    )
    parser.add_argument(
        "--memory-window",
        type=int,
        default=96,
        help="Number of recent non-conditioning memory outputs to keep. Default: 96.",
    )
    parser.add_argument(
        "--prune-every",
        type=int,
        default=10,
        help="Run streaming state pruning every N tracked frames. Use 0 to disable. Default: 10.",
    )
    parser.add_argument(
        "--print-prune-stats",
        action="store_true",
        help="Print pruning statistics whenever pruning runs.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="Device for SAM2-Plus inference. Default: cpu.",
    )
    parser.add_argument("--no-display", action="store_true", help="Write video without playback.")
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Reduce GPU memory usage at the cost of speed.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        help="Store tracking state on CPU. Useful for GPU memory, slower.",
    )
    return parser.parse_args()


def maybe_wslpath(path):
    if os.name != "posix":
        return path
    if len(path) >= 3 and path[1:3] == ":\\":
        try:
            return subprocess.check_output(["wslpath", "-u", path], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            return path
    return path


def resolve_existing_path(path, repo_dir):
    path = maybe_wslpath(path)
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    candidate = (Path.cwd() / expanded).resolve()
    if candidate.exists():
        return candidate
    return (repo_dir / expanded).resolve()


def resolve_checkpoint(path, repo_dir):
    checkpoint = resolve_existing_path(path, repo_dir)
    if checkpoint.exists():
        return checkpoint
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint}\n"
        "Download it with:\n"
        "  source .venv/bin/activate\n"
        "  hf download MCG-NJU/SAM2-Plus --local-dir checkpoints/SAM2-Plus\n"
        "or:\n"
        "  bash install_sam2pp_wsl.sh --checkpoint-only"
    )


def read_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Cannot read the first frame: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame, fps, width, height, frame_count


def resize_for_processing(frame, long_edge):
    if long_edge <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = min(long_edge / max(h, w), 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def fit_for_display(frame, max_w=1280, max_h=800):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return frame.copy(), scale
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale


def select_box(first_frame):
    points = []
    display_frame, scale = fit_for_display(first_frame)

    def redraw():
        canvas = display_frame.copy()
        if len(points) == 1:
            cv2.circle(canvas, points[0], 5, (0, 255, 255), -1)
        elif len(points) >= 2:
            cv2.rectangle(canvas, points[0], points[1], (0, 255, 255), 2)
        cv2.putText(
            canvas,
            "Click top-left, then bottom-right. r: reset, Esc/q: quit",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, canvas)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    while len(points) < 2:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(WINDOW_NAME)
            raise RuntimeError("Selection cancelled")
        if key == ord("r"):
            points.clear()
            redraw()

    cv2.destroyWindow(WINDOW_NAME)
    p1 = np.array(points[0], dtype=np.float32) / scale
    p2 = np.array(points[1], dtype=np.float32) / scale
    x1, y1 = np.minimum(p1, p2)
    x2, y2 = np.maximum(p1, p2)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Invalid box selection")
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.45):
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
    mask = mask.astype(bool)
    result = frame.copy()
    color_layer = np.zeros_like(frame)
    color_layer[:] = color
    result[mask] = cv2.addWeighted(frame, 1.0 - alpha, color_layer, alpha, 0)[mask]
    return result


def draw_result(frame, mask, box_xyxy, label):
    out = overlay_mask(frame, mask)
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        h, w = frame.shape[:2]
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return out


def mask_to_box(mask):
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def valid_box(box, width, height):
    if box is None:
        return None
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).reshape(-1)[:4]
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def update_box_from_prediction(pred_box, mask, width, height, previous_box):
    box = valid_box(pred_box, width, height)
    if box is not None:
        return box
    box = valid_box(mask_to_box(mask), width, height)
    if box is not None:
        return box
    return previous_box


def resolve_device(device_arg):
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if args.frame_buffer_size < 1:
        raise ValueError("--frame-buffer-size must be >= 1")
    if args.memory_window < 1:
        raise ValueError("--memory-window must be >= 1")
    if args.prune_every < 0:
        raise ValueError("--prune-every must be >= 0")
    repo_dir = Path(__file__).resolve().parent
    os.chdir(repo_dir)

    input_video = resolve_existing_path(args.input_video, repo_dir)
    device = resolve_device(args.device)
    first_frame_raw, fps, source_width, source_height, frame_count = read_video_info(input_video)
    output_fps = fps / args.frame_step
    if output_fps <= 0:
        output_fps = fps

    print(
        "[INFO] Phase-1 streaming preprocessing: "
        f"max_frames={args.max_frames if args.max_frames > 0 else 'all'}, "
        f"resize_long_edge={args.resize_long_edge if args.resize_long_edge > 0 else 'original'}, "
        f"frame_step={args.frame_step}"
    )

    output_path = Path(maybe_wslpath(args.output)).expanduser() if args.output else None
    if output_path is None:
        output_path = input_video.with_name(f"{input_video.stem}_sam2pp_tracked.mp4")
    elif not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading SAM2-Plus on {device}...")
    from sam2_plus.build_sam import build_sam2_streaming_video_predictor_plus

    predictor = build_sam2_streaming_video_predictor_plus(
        config_file=args.config,
        ckpt_path=str(resolve_checkpoint(args.checkpoint, repo_dir)),
        device=device,
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=false"],
        task="box",
    )

    first_frame = resize_for_processing(first_frame_raw, args.resize_long_edge)
    height, width = first_frame.shape[:2]
    print(
        f"[INFO] Streaming at {width}x{height} from source "
        f"{source_width}x{source_height}, total source frames={frame_count}"
    )

    box_xyxy = select_box(first_frame)
    print(f"[INFO] Selected box xyxy: {box_xyxy.tolist()}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video for writing: {output_path}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot reopen video: {input_video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    if not args.no_display:
        cv2.namedWindow("SAM2-Plus tracking", cv2.WINDOW_NORMAL)

    print("[INFO] Tracking with one continuous streaming state...")
    try:
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            state = predictor.init_stream_state(
                cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                offload_video_to_cpu=True if device == "cpu" else args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                box=box_xyxy,
            )

            delay_ms = max(int(1000 / output_fps), 1)
            frame_idx, obj_ids, mask_logits, pred_boxes, _scores = predictor.track_next_frame(
                state, 0
            )
            mask = (mask_logits[0].squeeze().detach().cpu().numpy() > args.score_thresh)
            pred_box = None
            if pred_boxes is not None:
                pred_box = pred_boxes[0].detach().cpu().numpy().reshape(-1)[:4]
            current_box = update_box_from_prediction(pred_box, mask, width, height, box_xyxy)
            vis = draw_result(first_frame, mask, current_box, "frame 0")
            writer.write(vis)
            if not args.no_display:
                cv2.imshow("SAM2-Plus tracking", vis)
                cv2.waitKey(delay_ms)
            if args.prune_every == 1:
                stats = predictor.prune_stream_state(
                    state,
                    frame_idx,
                    frame_buffer_size=args.frame_buffer_size,
                    memory_window=args.memory_window,
                )
                if args.print_prune_stats:
                    print(f"[INFO] prune stats: {stats}")

            kept_count = 1
            source_idx = 1
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if source_idx % args.frame_step != 0:
                    source_idx += 1
                    continue

                frame = resize_for_processing(frame, args.resize_long_edge)
                stream_idx = predictor.append_frame(
                    state, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                frame_idx, obj_ids, mask_logits, pred_boxes, _scores = predictor.track_next_frame(
                    state, stream_idx
                )
                mask = (
                    mask_logits[0].squeeze().detach().cpu().numpy() > args.score_thresh
                )
                pred_box = None
                if pred_boxes is not None:
                    pred_box = pred_boxes[0].detach().cpu().numpy().reshape(-1)[:4]

                current_box = update_box_from_prediction(
                    pred_box, mask, width, height, current_box
                )
                vis = draw_result(frame, mask, current_box, f"frame {kept_count}")
                writer.write(vis)

                if not args.no_display:
                    cv2.imshow("SAM2-Plus tracking", vis)
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key in (27, ord("q")):
                        print("[INFO] Stopped by user")
                        break

                kept_count += 1
                source_idx += 1
                if args.prune_every and kept_count % args.prune_every == 0:
                    stats = predictor.prune_stream_state(
                        state,
                        stream_idx,
                        frame_buffer_size=args.frame_buffer_size,
                        memory_window=args.memory_window,
                    )
                    if args.print_prune_stats:
                        print(f"[INFO] prune stats: {stats}")
                if args.max_frames > 0 and kept_count >= args.max_frames:
                    break
    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"[OK] Saved tracked video: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
