from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class TargetState(str, Enum):
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    REACQUIRED = "REACQUIRED"
    NEEDS_USER = "NEEDS_USER"


class RePromptActionType(str, Enum):
    NONE = "NONE"
    ADD_BOX_PROMPT = "ADD_BOX_PROMPT"
    RESET_SAM = "RESET_SAM"
    REQUEST_USER = "REQUEST_USER"


@dataclass
class Detection:
    frame_id: int
    bbox_xyxy: np.ndarray
    class_id: int
    confidence: float


@dataclass
class TrackCandidate:
    frame_id: int
    track_id: int
    bbox_xyxy: np.ndarray
    class_id: int
    tracker_confidence: float
    detector_confidence: Optional[float] = None
    age: int = 0
    time_since_update: int = 0


@dataclass
class SelectedTarget:
    active_track_id: Optional[int] = None
    class_id: Optional[int] = None
    last_good_box_xyxy: Optional[np.ndarray] = None
    last_good_mask: Optional[np.ndarray] = None
    last_good_embedding: Optional[np.ndarray] = None
    lost_frames: int = 0


@dataclass
class RefinedTarget:
    frame_id: int
    track_id: Optional[int]
    mask: Optional[np.ndarray]
    bbox_xyxy: Optional[np.ndarray]
    sam_score: Optional[float]
    nvdcf_bbox_xyxy: Optional[np.ndarray]
    state: TargetState = TargetState.TRACKING


@dataclass
class RePromptAction:
    type: RePromptActionType
    bbox_xyxy: Optional[np.ndarray] = None
    reason: str = ""
