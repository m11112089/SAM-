import numpy as np


def as_box(box):
    if box is None:
        return None
    arr = np.asarray(box, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return None
    x1, y1, x2, y2 = arr[:4]
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_area(box):
    box = as_box(box)
    if box is None:
        return 0.0
    return float((box[2] - box[0]) * (box[3] - box[1]))


def box_iou(a, b):
    a = as_box(a)
    b = as_box(b)
    if a is None or b is None:
        return 0.0

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, float(ix2 - ix1))
    ih = max(0.0, float(iy2 - iy1))
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def box_center(box):
    box = as_box(box)
    if box is None:
        return None
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def center_distance(a, b):
    ca = box_center(a)
    cb = box_center(b)
    if ca is None or cb is None:
        return float("inf")
    return float(np.linalg.norm(ca - cb))


def mask_to_box(mask):
    if mask is None:
        return None
    mask = np.asarray(mask).astype(bool)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def mask_area(mask):
    if mask is None:
        return 0.0
    return float(np.asarray(mask).astype(bool).sum())
