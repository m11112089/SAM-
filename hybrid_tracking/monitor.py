from dataclasses import dataclass, field
from statistics import median

from hybrid_tracking.contracts import RefinedTarget, SelectedTarget, TargetState
from hybrid_tracking.geometry import box_iou, center_distance, mask_area


@dataclass
class MonitorConfig:
    min_sam_score: float = 0.0
    min_iou_with_candidate: float = 0.15
    max_center_jump_ratio: float = 0.25
    min_mask_area_ratio: float = 0.2
    max_mask_area_ratio: float = 5.0
    occluded_after_uncertain: int = 5
    lost_after_uncertain: int = 20
    recent_area_window: int = 20


@dataclass
class OcclusionLostMonitor:
    config: MonitorConfig = field(default_factory=MonitorConfig)
    recent_mask_areas: list = field(default_factory=list)
    uncertain_frames: int = 0

    def update(self, refined, candidate, selected, frame_shape):
        state = self.classify(refined, candidate, selected, frame_shape)
        refined.state = state

        if state in (TargetState.TRACKING, TargetState.REACQUIRED):
            self.uncertain_frames = 0
            selected.lost_frames = 0
            if refined.bbox_xyxy is not None:
                selected.last_good_box_xyxy = refined.bbox_xyxy.copy()
            if refined.mask is not None:
                selected.last_good_mask = refined.mask.copy()
                self._remember_area(mask_area(refined.mask))
            if candidate is not None:
                selected.active_track_id = candidate.track_id
                selected.class_id = candidate.class_id
        elif state == TargetState.LOST:
            selected.lost_frames += 1
        else:
            self.uncertain_frames += 1

        return state

    def classify(self, refined, candidate, selected, frame_shape):
        if refined is None or refined.bbox_xyxy is None or refined.mask is None:
            return self._degraded_state(candidate)

        if refined.sam_score is not None and refined.sam_score < self.config.min_sam_score:
            return self._degraded_state(candidate)

        area = mask_area(refined.mask)
        if area <= 0:
            return self._degraded_state(candidate)

        if self.recent_mask_areas:
            med_area = max(1.0, float(median(self.recent_mask_areas)))
            area_ratio = area / med_area
            if (
                area_ratio < self.config.min_mask_area_ratio
                or area_ratio > self.config.max_mask_area_ratio
            ):
                return self._degraded_state(candidate)

        if candidate is not None:
            iou = box_iou(refined.bbox_xyxy, candidate.bbox_xyxy)
            if iou < self.config.min_iou_with_candidate:
                return TargetState.UNCERTAIN

        if selected.last_good_box_xyxy is not None:
            height, width = frame_shape[:2]
            diag = max(1.0, (height * height + width * width) ** 0.5)
            jump = center_distance(refined.bbox_xyxy, selected.last_good_box_xyxy)
            if jump / diag > self.config.max_center_jump_ratio:
                return TargetState.UNCERTAIN

        return TargetState.TRACKING

    def _degraded_state(self, candidate):
        next_uncertain_frames = self.uncertain_frames + 1
        if next_uncertain_frames >= self.config.lost_after_uncertain:
            return TargetState.LOST
        if next_uncertain_frames >= self.config.occluded_after_uncertain:
            return TargetState.OCCLUDED if candidate is not None else TargetState.LOST
        return TargetState.UNCERTAIN

    def _remember_area(self, area):
        self.recent_mask_areas.append(area)
        max_len = self.config.recent_area_window
        if len(self.recent_mask_areas) > max_len:
            self.recent_mask_areas = self.recent_mask_areas[-max_len:]
