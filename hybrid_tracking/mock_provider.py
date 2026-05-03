from collections import deque
from pathlib import Path

import numpy as np

from hybrid_tracking.contracts import TrackCandidate


class MockNvDCFProvider:
    """Mock NvDCF candidate source for file-video prototyping.

    It can replay boxes from CSV-like text or synthesize a stable candidate from
    the latest SAM refined bbox. This keeps Phase A independent of DeepStream.
    """

    def __init__(
        self,
        initial_box_xyxy=None,
        track_id=1,
        class_id=0,
        tracker_confidence=1.0,
        csv_path=None,
    ):
        self.track_id = track_id
        self.class_id = class_id
        self.tracker_confidence = tracker_confidence
        self.last_box_xyxy = (
            np.asarray(initial_box_xyxy, dtype=np.float32) if initial_box_xyxy is not None else None
        )
        self.rows = deque(_load_rows(csv_path)) if csv_path else None

    def update_from_refined(self, bbox_xyxy):
        if bbox_xyxy is not None:
            self.last_box_xyxy = np.asarray(bbox_xyxy, dtype=np.float32)

    def get_candidate(self, frame_id):
        if self.rows is not None:
            while self.rows and self.rows[0][0] < frame_id:
                self.rows.popleft()
            if self.rows and self.rows[0][0] == frame_id:
                row = self.rows[0]
                return TrackCandidate(
                    frame_id=frame_id,
                    track_id=int(row[1]),
                    bbox_xyxy=np.array(row[2:6], dtype=np.float32),
                    class_id=int(row[6]) if len(row) > 6 else self.class_id,
                    tracker_confidence=float(row[7]) if len(row) > 7 else self.tracker_confidence,
                    detector_confidence=float(row[8]) if len(row) > 8 else None,
                )

        if self.last_box_xyxy is None:
            return None
        return TrackCandidate(
            frame_id=frame_id,
            track_id=self.track_id,
            bbox_xyxy=self.last_box_xyxy.copy(),
            class_id=self.class_id,
            tracker_confidence=self.tracker_confidence,
            detector_confidence=None,
            age=frame_id,
            time_since_update=0,
        )


def _load_rows(csv_path):
    path = Path(csv_path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [p.strip() for p in stripped.replace("\t", ",").split(",")]
        rows.append([float(p) for p in parts])
    return rows
