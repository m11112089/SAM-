import cv2
import numpy as np


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.45):
    if mask is None:
        return frame.copy()
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
    mask = mask.astype(bool)
    out = frame.copy()
    color_layer = np.zeros_like(frame)
    color_layer[:] = color
    blended = cv2.addWeighted(frame, 1.0 - alpha, color_layer, alpha, 0)
    out[mask] = blended[mask]
    return out


def draw_box(frame, box_xyxy, color, label=None, thickness=2):
    if box_xyxy is None:
        return frame
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)[:4]
    h, w = frame.shape[:2]
    pt1 = (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
    pt2 = (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
    cv2.rectangle(frame, pt1, pt2, color, thickness)
    if label:
        cv2.putText(
            frame,
            label,
            (pt1[0], max(18, pt1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def render_hybrid_frame(frame, refined, candidate=None, action=None):
    out = overlay_mask(frame, refined.mask if refined else None)
    if candidate is not None:
        draw_box(out, candidate.bbox_xyxy, (255, 128, 0), f"NvDCF {candidate.track_id}")
    if refined is not None:
        draw_box(out, refined.bbox_xyxy, (0, 255, 255), "SAM2++")
        label = refined.state.value
        if action is not None and action.type.value != "NONE":
            label = f"{label} / {action.type.value}"
        cv2.putText(
            out,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out
