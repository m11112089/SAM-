# Streaming SAM2-Plus Tracking Architecture

This design targets camera, RTSP, or long-video tracking where frames arrive
incrementally and the tracker must avoid loading the whole video into memory.
The goal is to keep tracking memory continuous while bounding RAM/VRAM usage.

## Problem

SAM2-Plus video tracking is currently organized around:

1. `predictor.init_state(video_path=...)`
2. add an initial prompt with `add_new_points_or_box(...)`
3. run `predictor.propagate_in_video(inference_state)`

This works well when the full frame sequence is known up front. It is not a
native live-streaming API because `inference_state["images"]`,
`inference_state["num_frames"]`, cached outputs, and memory dictionaries are
initialized around a fixed sequence.

The external chunk workaround reduces memory, but each chunk reinitializes from
the last bbox. That breaks model memory continuity and weakens occlusion and
re-identification.

## Goals

- Accept one frame at a time from video file, camera, or RTSP.
- Keep SAM2-Plus tracking memory continuous across frames.
- Write rendered mask and bbox output immediately.
- Bound RAM/VRAM with a rolling frame buffer and memory pruning.
- Support occlusion without blindly reinitializing every chunk from bbox.
- Allow operator correction when confidence drops or drift is detected.

## Non-Goals

- Full multi-camera tracking.
- Cross-camera re-identification.
- Training-time changes to SAM2-Plus.
- Guaranteed long-term re-id after very long disappearance. This design
  preserves SAM2 memory better than chunking, but SAM2-Plus is still not a
  dedicated global re-id tracker.

## High-Level Components

```text
FrameSource
  -> FramePreprocessor
  -> StreamingSAM2State
  -> TrackerController
  -> OcclusionAndDriftMonitor
  -> Renderer
  -> VideoSink
```

## Component Responsibilities

### FrameSource

Reads frames from:

- video file via OpenCV
- webcam via OpenCV
- RTSP stream via OpenCV or ffmpeg pipe

Outputs:

```python
FramePacket(
    frame_id: int,
    timestamp: float,
    bgr: np.ndarray,
)
```

### FramePreprocessor

Applies deterministic transforms before inference:

- resize long edge to configured limit
- optional frame skipping
- BGR to RGB conversion if needed by loader path
- stable frame-id mapping between source frame and model frame

The same processed frame is used for:

- SAM2 input
- mask/bbox rendering
- output video size

### StreamingSAM2State

Owns a modified SAM2-Plus `inference_state`.

Core additions:

```python
class StreamingSAM2State:
    predictor: SAM2VideoPredictor_Plus
    inference_state: dict
    frame_ring: RollingFrameStore
    frame_id_to_state_idx: dict[int, int]
    state_idx_to_frame_id: dict[int, int]
    last_good_box_xyxy: np.ndarray | None
    last_good_mask: np.ndarray | None
    object_id: int
```

This object hides the difference between "fixed video" and "streaming video".

### RollingFrameStore

Stores only recent preprocessed frames.

Policy:

- keep last `N` raw/preprocessed frames for rendering and correction
- keep selected keyframes longer when they are used as conditioning memory
- drop frames that are no longer needed by SAM2 memory or output rendering

Recommended starting values:

```text
frame_buffer_size = 64 to 256 frames
keyframe_interval = 15 to 30 frames
max_conditioning_frames = 8 to 32
max_non_conditioning_frames = 32 to 128
```

### TrackerController

Implements the online tracking loop:

1. receive next frame
2. append it into `inference_state`
3. run SAM2 inference for the new frame
4. update memory dictionaries
5. render mask and bbox
6. prune old frame and memory entries

### OcclusionAndDriftMonitor

Consumes SAM2 outputs:

- mask area
- bbox area and aspect ratio
- object score logits
- mask confidence/logit strength
- IoU or center-distance against previous frame

States:

```text
TRACKING
UNCERTAIN
OCCLUDED
LOST
NEEDS_USER_CORRECTION
```

Behavior:

- `TRACKING`: update `last_good_box` and `last_good_mask`
- `UNCERTAIN`: keep memory, render warning, avoid aggressive bbox updates
- `OCCLUDED`: keep predicting but do not overwrite last good identity memory
- `LOST`: pause automatic memory updates or request user correction
- `NEEDS_USER_CORRECTION`: accept new click/box/mask prompt

### Renderer

Overlays:

- translucent mask
- bbox
- frame id
- confidence/state label

Writes each frame immediately to `VideoSink`.

### VideoSink

Writes MP4 or streams frames to a display. It does not wait for all masks.

## Required Predictor Changes

The clean implementation is to subclass `SAM2VideoPredictor_Plus` with an
online API.

### New API

```python
class SAM2StreamingVideoPredictor(SAM2VideoPredictor_Plus):
    def init_stream_state(
        self,
        first_frame: np.ndarray,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict:
        ...

    def append_frame(
        self,
        inference_state: dict,
        frame: np.ndarray,
    ) -> int:
        """Append one processed frame and return its state frame index."""
        ...

    def track_next_frame(
        self,
        inference_state: dict,
        frame_idx: int,
        obj_id: int,
    ):
        """Run one-step propagation for a single appended frame."""
        ...

    def prune_stream_state(
        self,
        inference_state: dict,
        keep_from_frame_idx: int,
        keep_conditioning: bool = True,
    ) -> None:
        ...
```

### Why a Subclass

Keeping this in a subclass avoids destabilizing benchmark and training code.
Existing fixed-video behavior stays unchanged.

## Inference State Changes

The current state assumes fixed length:

```python
inference_state["images"]
inference_state["num_frames"]
inference_state["cached_features"]
inference_state["output_dict_per_obj"]
inference_state["frames_tracked_per_obj"]
```

Streaming requires:

```python
inference_state["images"]              # appendable list or frame store
inference_state["num_frames"]          # increments per appended frame
inference_state["streaming"] = True
inference_state["stream_frame_offset"] # maps pruned state indices if needed
```

For CPU/GPU memory control, `images` should not be a single stacked tensor for
streaming. It should be an appendable lazy store returning one normalized tensor
at a time.

## One-Frame Propagation Strategy

Instead of calling `propagate_in_video()` over a range, online tracking should
reuse the same internal logic for one frame:

```text
append frame k
for each object:
  if frame k has user prompt:
    use cond_frame_outputs
  else:
    call _run_single_frame_inference(... frame_idx=k ...)
store non_cond_frame_outputs[k]
resize mask/box to video resolution
return frame output
```

This is essentially `propagate_in_video()` with `processing_order = [k]`, but
without requiring all future frames to exist.

## Memory Pruning

Without pruning, a stream will grow forever.

### Frame Pruning

Keep raw frames for:

- current rendering
- recent correction UI
- recent memory context

Drop raw frame tensors older than `frame_buffer_size`, unless they are selected
conditioning keyframes.

### Output Pruning

For each object:

```python
obj_output_dict["non_cond_frame_outputs"]
obj_output_dict["cond_frame_outputs"]
```

Keep:

- all user prompt frames
- recent non-conditioning outputs
- sampled keyframes with high confidence

Drop:

- old low-value non-conditioning outputs
- cached image features for old frames

### Keyframe Selection

Promote a frame to conditioning/keyframe when:

- confidence is high
- mask area is stable
- bbox is not clipped
- frame is at least `keyframe_interval` after previous keyframe
- object is not occluded

This helps preserve identity while bounding memory.

## Occlusion Handling

During occlusion:

- do not reinitialize from low-confidence bbox
- keep last good bbox and mask separately
- continue one-step propagation for a configurable grace period
- suppress memory promotion for uncertain frames
- render predicted output as uncertain

If object score and mask confidence stay poor for too long:

- mark `LOST`
- optionally run external detector/re-id
- request user correction box/click

## Re-ID Options

SAM2 memory helps short-term occlusion but is not a global re-id system.

For stronger re-id, add an optional sidecar:

```text
last_good_crop -> embedding model -> gallery
new detections/candidates -> embedding match -> reinitialize SAM2 prompt
```

Possible detectors:

- YOLO for generic object proposals
- domain-specific detector
- background motion proposal

Possible embeddings:

- CLIP/DINOv2/ReID model
- color histogram fallback for simple scenes

## Recommended Implementation Phases

### Phase 1: One-Step Predictor

- Add `SAM2StreamingVideoPredictor` subclass.
- Implement `init_stream_state()`.
- Implement `append_frame()`.
- Implement `track_next_frame()` by adapting `propagate_in_video()` logic for one frame.
- No pruning yet.

Validation:

- Compare output against fixed-video predictor on a short clip.
- Masks and boxes should be similar frame-by-frame.

### Phase 2: Rolling Buffer

- Replace full stacked image tensor with appendable lazy frame store.
- Add `prune_stream_state()`.
- Keep recent outputs and selected conditioning frames.

Validation:

- Run 1000+ frames without monotonic memory growth.

Implementation status:

- `sam2_plus/streaming_video_predictor.py` now includes phase-2 pruning.
- Old frame tensors are replaced with `None` without reindexing frame ids.
- Conditioning outputs are kept by default as identity anchors.
- Old non-conditioning outputs and `frames_tracked` metadata are pruned by a
  configurable memory window.
- `interactive_video_box_track.py` exposes `--frame-buffer-size`,
  `--memory-window`, `--prune-every`, and `--print-prune-stats`.

### Phase 3: Occlusion State Machine

- Add confidence metrics.
- Add `TRACKING/UNCERTAIN/OCCLUDED/LOST` states.
- Stop promoting uncertain frames to memory.

Validation:

- Test clips with temporary occlusion.

### Phase 4: External Re-ID

- Add optional detector/embedding recovery.
- Only reinitialize SAM2 when match confidence is high.

Validation:

- Object leaves/re-enters scene.

## Practical Defaults

For GPU:

```text
resize_long_edge = 720 or 1080
frame_buffer_size = 128
max_non_conditioning_frames = 96
max_conditioning_frames = 16
keyframe_interval = 20
occlusion_grace_frames = 30
```

For CPU:

```text
resize_long_edge = 480 or 720
frame_buffer_size = 32
max_non_conditioning_frames = 16
max_conditioning_frames = 4
keyframe_interval = 15
occlusion_grace_frames = 10
```

## Risks

- Internal SAM2-Plus methods are not a stable public API.
- Pruning memory too aggressively can cause identity drift.
- Keeping too much memory can OOM long streams.
- Re-id needs a detector/embedding model; SAM2 alone is not enough for robust
  long disappearance recovery.

## Recommended Next Step

Implement Phase 1 in a new file:

```text
sam2_plus/streaming_video_predictor.py
```

Then add a runner:

```text
streaming_video_track.py
```

This keeps the current fixed-video predictor untouched while enabling a real
append-frame tracking path.
