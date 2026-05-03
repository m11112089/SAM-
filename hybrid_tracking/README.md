# Hybrid Tracking Interfaces

Phase A provides pure Python interfaces for a future NvDCF + SAM2-Plus
pipeline. These modules do not depend on DeepStream.

## Modules

- `contracts.py`: dataclasses and state/action enums.
- `mock_provider.py`: mock NvDCF-style `TrackCandidate` source.
- `monitor.py`: classifies SAM2-Plus output as tracking, uncertain, occluded, or lost.
- `reprompt.py`: decides whether SAM2-Plus should continue, receive a correction bbox, reset, or ask for user input.
- `renderer.py`: draws SAM2-Plus mask/bbox and NvDCF candidate bbox.

## Minimal Flow

```python
from hybrid_tracking import (
    MockNvDCFProvider,
    OcclusionLostMonitor,
    RefinedTarget,
    RePromptController,
    SelectedTarget,
)

selected = SelectedTarget()
provider = MockNvDCFProvider(initial_box_xyxy=[100, 100, 180, 180])
monitor = OcclusionLostMonitor()
reprompt = RePromptController()

candidate = provider.get_candidate(frame_id=0)
refined = RefinedTarget(
    frame_id=0,
    track_id=candidate.track_id,
    mask=mask,
    bbox_xyxy=sam_box,
    sam_score=None,
    nvdcf_bbox_xyxy=candidate.bbox_xyxy,
)

state = monitor.update(refined, candidate, selected, frame_shape=frame.shape)
action = reprompt.decide(refined, candidate, selected)
```

The next implementation step is to route `interactive_video_box_track.py`
through these interfaces, then replace `MockNvDCFProvider` with a DeepStream
metadata adapter.
