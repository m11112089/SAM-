# Hybrid NvDCF + SAM2-Plus Streaming Tracker

This design uses NvDCF for real-time MOT, ID management, and candidate boxes,
while SAM2-Plus performs mask refinement for a selected target. When the target
is occluded or lost, NvDCF, detector boxes, and optional re-id are used to
recover and re-prompt SAM2-Plus.

## Architecture

```text
Video Source
  -> Detector
  -> NvDCF Tracker
  -> Track Adapter
  -> Target Selector / ID Manager
  -> SAM2-Plus Streaming Refiner
  -> Occlusion / Lost Monitor
  -> Re-Prompt Controller
  -> Renderer / Video Sink
```

## Responsibility Split

### Detector

Provides object detections:

```python
Detection(
    frame_id: int,
    bbox_xyxy: np.ndarray,
    class_id: int,
    confidence: float,
)
```

Examples:

- DeepStream primary GIE
- YOLO
- domain-specific detector

### NvDCF Tracker

Provides stable MOT tracks:

```python
TrackCandidate(
    frame_id: int,
    track_id: int,
    bbox_xyxy: np.ndarray,
    class_id: int,
    tracker_confidence: float,
    detector_confidence: float | None,
    age: int,
    time_since_update: int,
)
```

NvDCF owns:

- multi-object association
- short-term bbox tracking
- track id continuity
- candidate bbox during detector gaps

### Track Adapter

Normalizes DeepStream/NvDCF metadata into Python-side `TrackCandidate` objects.

The adapter should be the only DeepStream-specific layer. SAM2-Plus code should
not depend directly on DeepStream structs.

### Target Selector / ID Manager

Maps a selected target to NvDCF track ids.

Selection modes:

- user selects initial bbox on first frame
- closest NvDCF track to user bbox becomes target id
- explicit track id from external UI
- detector class + region rule

State:

```python
SelectedTarget(
    active_track_id: int | None,
    class_id: int | None,
    last_good_box_xyxy: np.ndarray | None,
    last_good_mask: np.ndarray | None,
    last_good_embedding: np.ndarray | None,
)
```

### SAM2-Plus Streaming Refiner

Consumes the selected target bbox as prompt or re-prompt and outputs:

```python
SAMRefinedOutput(
    frame_id: int,
    mask: np.ndarray,
    bbox_xyxy: np.ndarray,
    object_score: float | None,
    state: str,
)
```

SAM2-Plus owns:

- pixel-level mask
- refined bbox from mask/model box
- prompt memory for the selected object
- short-term continuity across shape changes and partial occlusion

### Occlusion / Lost Monitor

Compares SAM2-Plus output and NvDCF candidate:

Signals:

- SAM object score
- mask area ratio
- bbox area jump
- bbox center jump
- IoU between SAM bbox and NvDCF bbox
- NvDCF tracker confidence
- detector update availability
- consecutive frames without reliable mask

States:

```text
TRACKING
UNCERTAIN
OCCLUDED
LOST
REACQUIRED
NEEDS_USER
```

### Re-Prompt Controller

Decides when SAM2-Plus should receive a new prompt.

Inputs:

- current SAM confidence
- selected NvDCF candidate
- detector candidate
- optional re-id match
- target state

Outputs:

- no prompt; continue SAM memory
- add correction bbox on current frame
- reset SAM stream state from candidate bbox
- request user correction

## Main Runtime Loop

```python
for frame in source:
    detections = detector(frame)
    tracks = nvdcf.update(frame, detections)
    target_candidate = id_manager.select(tracks, detections)

    if sam_state is None:
        sam_state = sam.init_stream_state(frame)
        sam.add_box_prompt(target_candidate.bbox_xyxy)
        sam_output = sam.track_current_frame()
    else:
        frame_idx = sam.append_frame(frame)
        sam_output = sam.track_next_frame(frame_idx)

    target_state = monitor.update(
        sam_output=sam_output,
        track_candidate=target_candidate,
    )

    action = reprompt_controller.decide(target_state, target_candidate)
    if action.type == "ADD_BOX_PROMPT":
        sam.add_box_prompt(frame_idx, action.bbox_xyxy)
    elif action.type == "RESET_SAM":
        sam_state = sam.init_stream_state(frame)
        sam.add_box_prompt(action.bbox_xyxy)

    sam.prune_stream_state(...)
    renderer.write(frame, sam_output, target_candidate, target_state)
```

## Re-Prompt Policy

### Continue SAM Only

Use when:

- SAM score is good
- mask area is plausible
- SAM bbox and NvDCF bbox overlap enough
- no large sudden jump

Recommended:

```text
sam_iou_with_nvdcf >= 0.3
mask_area_ratio within [0.2, 5.0] of recent median
center_jump < 0.25 * frame_diagonal
```

### Add Correction BBox

Use when:

- SAM is drifting
- NvDCF confidence is still good
- NvDCF bbox overlaps recent target trajectory
- detector class matches selected target

Action:

```python
predictor.add_new_points_or_box(
    inference_state=sam_state,
    frame_idx=current_frame_idx,
    obj_id=target_obj_id,
    box=nvdcf_bbox_xyxy,
)
```

This keeps SAM state and adds a correction prompt.

### Do Not Update SAM Memory

Use during likely occlusion:

- SAM mask area collapses
- object score drops
- NvDCF predicts through occlusion but detector is absent

Phase 3 should add a dedicated method to suppress memory promotion for these
frames. Until then, mark output uncertain and avoid re-prompting from weak boxes.

### Reset SAM From Candidate

Use when:

- SAM is lost for many frames
- NvDCF or detector reacquires target confidently
- optional re-id confirms match

This loses old SAM memory but prevents drift from dominating.

### Request User Correction

Use when:

- SAM lost
- NvDCF confidence is low
- detector has multiple ambiguous candidates
- re-id match is weak or unavailable

## Re-ID Layer

NvDCF is not full long-term re-id. Add optional embedding matching when target
can leave and re-enter.

Gallery:

```python
TargetEmbeddingGallery(
    track_id: int,
    embeddings: list[np.ndarray],
    last_update_frame_id: int,
)
```

Update gallery only from high-confidence frames:

- SAM state `TRACKING`
- detector present
- mask area stable
- no occlusion flag

Candidate match:

```text
same class
embedding cosine similarity >= threshold
motion/region gate optional
```

If matched, bind new NvDCF `track_id` to selected target and re-prompt SAM.

## Data Contracts

```python
@dataclass
class TrackCandidate:
    frame_id: int
    track_id: int
    bbox_xyxy: np.ndarray
    class_id: int
    tracker_confidence: float
    detector_confidence: float | None = None
    age: int = 0
    time_since_update: int = 0


@dataclass
class RefinedTarget:
    frame_id: int
    track_id: int | None
    mask: np.ndarray | None
    bbox_xyxy: np.ndarray | None
    sam_score: float | None
    nvdcf_bbox_xyxy: np.ndarray | None
    state: str
```

## Implementation Plan

### Phase A: Python Interfaces

Create:

```text
hybrid_tracking/contracts.py
hybrid_tracking/monitor.py
hybrid_tracking/reprompt.py
hybrid_tracking/renderer.py
```

Use a mock `TrackCandidateProvider` first, so SAM2-Plus logic can be tested
without DeepStream.

Implementation status:

- `hybrid_tracking/contracts.py`
- `hybrid_tracking/mock_provider.py`
- `hybrid_tracking/monitor.py`
- `hybrid_tracking/reprompt.py`
- `hybrid_tracking/renderer.py`
- `hybrid_tracking/README.md`

These modules are pure Python interfaces and do not import DeepStream or
SAM2-Plus. They can be smoke-tested independently.

### Phase B: File-Video Prototype

Use:

- OpenCV video source
- optional detector or manually selected target bbox
- mock NvDCF candidate from last bbox or an external txt/csv track file
- SAM2 streaming refiner

Output:

- rendered MP4 with SAM mask, SAM bbox, NvDCF bbox, and state label

### Phase C: DeepStream Adapter

Implement adapter that consumes DeepStream metadata:

- frame number
- object id
- bbox
- class id
- tracker confidence
- detector confidence if available

Keep this adapter separate from SAM code.

### Phase D: Re-ID Recovery

Add embedding extraction from target crop or mask crop and candidate matching.

## Practical Defaults

For GPU:

```text
SAM resize_long_edge = 720 or 1080
SAM frame_buffer_size = 32
SAM memory_window = 128
SAM prune_every = 10
NvDCF detector interval = 1 to 5 depending on detector cost
lost_grace_frames = 30
reid_similarity_threshold = 0.75
```

For CPU:

```text
SAM resize_long_edge = 480
SAM frame_buffer_size = 4 to 8
SAM memory_window = 32 to 64
SAM prune_every = 5
Run SAM only on selected target, not every object
```

## Key Tradeoffs

- NvDCF gives real-time MOT and ID management but only bbox.
- SAM2-Plus gives high-quality masks but is expensive.
- Correction prompts from NvDCF improve recovery but can inject drift if NvDCF
  is wrong.
- Resetting SAM recovers from long loss but discards SAM memory.
- Re-id reduces ID switches but adds another model and thresholds.

## Recommended Next Step

Implement Phase A with mock candidates first. Then wire the existing
`interactive_video_box_track.py` SAM streaming output through the monitor and
reprompt controller before integrating DeepStream/NvDCF.
