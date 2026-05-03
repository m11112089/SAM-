from dataclasses import dataclass, field

from hybrid_tracking.contracts import RePromptAction, RePromptActionType, TargetState
from hybrid_tracking.geometry import box_iou


@dataclass
class RePromptConfig:
    candidate_min_confidence: float = 0.5
    correction_iou_below: float = 0.35
    reset_after_lost_frames: int = 10


@dataclass
class RePromptController:
    config: RePromptConfig = field(default_factory=RePromptConfig)

    def decide(self, refined, candidate, selected):
        if candidate is None:
            if refined.state in (TargetState.LOST, TargetState.NEEDS_USER):
                return RePromptAction(
                    RePromptActionType.REQUEST_USER,
                    reason="target lost and no NvDCF candidate is available",
                )
            return RePromptAction(RePromptActionType.NONE, reason="no candidate")

        if candidate.tracker_confidence < self.config.candidate_min_confidence:
            if refined.state == TargetState.LOST:
                return RePromptAction(
                    RePromptActionType.REQUEST_USER,
                    reason="target lost and candidate confidence is low",
                )
            return RePromptAction(
                RePromptActionType.NONE,
                reason="candidate confidence below threshold",
            )

        if refined.state == TargetState.LOST:
            if selected.lost_frames >= self.config.reset_after_lost_frames:
                return RePromptAction(
                    RePromptActionType.RESET_SAM,
                    bbox_xyxy=candidate.bbox_xyxy.copy(),
                    reason="lost target reacquired by candidate",
                )
            return RePromptAction(
                RePromptActionType.ADD_BOX_PROMPT,
                bbox_xyxy=candidate.bbox_xyxy.copy(),
                reason="lost target but within reset grace window",
            )

        if refined.state in (TargetState.UNCERTAIN, TargetState.OCCLUDED):
            iou = box_iou(refined.bbox_xyxy, candidate.bbox_xyxy)
            if iou < self.config.correction_iou_below:
                return RePromptAction(
                    RePromptActionType.ADD_BOX_PROMPT,
                    bbox_xyxy=candidate.bbox_xyxy.copy(),
                    reason="SAM output diverged from NvDCF candidate",
                )

        return RePromptAction(RePromptActionType.NONE, reason="SAM output accepted")
