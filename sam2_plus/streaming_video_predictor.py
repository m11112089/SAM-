from collections import OrderedDict

import numpy as np
import torch

from sam2_plus.sam2_video_predictor import SAM2VideoPredictor_Plus


class StreamingFrameStore:
    """Append-only normalized frame store for streaming inference.

    Frame indices are stable. Pruning replaces old entries with None instead of
    compacting the list, so SAM2 temporal indices remain valid.
    """

    def __init__(
        self,
        image_size,
        compute_device,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.compute_device = compute_device
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        self.frames = []
        self.video_height = None
        self.video_width = None

    def append(self, frame_rgb):
        if not isinstance(frame_rgb, np.ndarray):
            raise TypeError("frame must be a numpy ndarray")
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("frame must have shape HxWx3")
        if frame_rgb.dtype != np.uint8:
            raise ValueError("frame must be uint8")

        height, width = frame_rgb.shape[:2]
        if self.video_height is None:
            self.video_height = height
            self.video_width = width
        elif (height, width) != (self.video_height, self.video_width):
            raise ValueError(
                "streaming frames must keep a stable resolution; "
                f"got {(width, height)}, expected {(self.video_width, self.video_height)}"
            )

        resized = _resize_rgb(frame_rgb, self.image_size)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor -= self.img_mean
        tensor /= self.img_std
        if not self.offload_video_to_cpu:
            tensor = tensor.to(self.compute_device, non_blocking=True)

        self.frames.append(tensor)
        return len(self.frames) - 1

    def __getitem__(self, index):
        frame = self.frames[index]
        if frame is None:
            raise RuntimeError(f"stream frame {index} was pruned")
        return frame

    def __len__(self):
        return len(self.frames)

    def prune_before(self, keep_from_frame_idx, keep_indices=None):
        keep_indices = set(keep_indices or [])
        pruned = 0
        for idx in range(min(keep_from_frame_idx, len(self.frames))):
            if idx in keep_indices:
                continue
            if self.frames[idx] is not None:
                self.frames[idx] = None
                pruned += 1
        return pruned

    def live_count(self):
        return sum(frame is not None for frame in self.frames)


def _resize_rgb(frame_rgb, image_size):
    from PIL import Image

    return np.array(Image.fromarray(frame_rgb).resize((image_size, image_size)))


class SAM2StreamingVideoPredictor(SAM2VideoPredictor_Plus):
    """Streaming predictor.

    This subclass keeps the fixed-video predictor untouched while adding an
    append-frame API plus phase-2 rolling pruning.
    """

    @torch.inference_mode()
    def init_stream_state(
        self,
        first_frame,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        compute_device = self.device
        images = StreamingFrameStore(
            image_size=self.image_size,
            compute_device=compute_device,
            offload_video_to_cpu=offload_video_to_cpu,
        )
        images.append(first_frame)

        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        inference_state["streaming"] = True
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["video_height"] = images.video_height
        inference_state["video_width"] = images.video_width
        inference_state["device"] = compute_device
        inference_state["storage_device"] = (
            torch.device("cpu") if offload_state_to_cpu else compute_device
        )
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}

        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    @torch.inference_mode()
    def append_frame(self, inference_state, frame):
        if not inference_state.get("streaming", False):
            raise ValueError("append_frame requires a stream state from init_stream_state")
        frame_idx = inference_state["images"].append(frame)
        inference_state["num_frames"] = len(inference_state["images"])
        inference_state["video_height"] = inference_state["images"].video_height
        inference_state["video_width"] = inference_state["images"].video_width
        return frame_idx

    @torch.inference_mode()
    def track_next_frame(self, inference_state, frame_idx, reverse=False):
        """Track exactly one frame already present in the stream state."""
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        batch_size = self._get_obj_num(inference_state)
        pred_masks_per_obj = [None] * batch_size
        pred_boxes_xyxy_norm_per_obj = [None] * batch_size
        pred_object_score_logits_per_obj = [None] * batch_size

        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if frame_idx in obj_output_dict["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = obj_output_dict[storage_key][frame_idx]
                device = inference_state["device"]
                pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                pred_boxes_xyxy_norm = (
                    current_out["pred_boxes_xyxy_norm"].to(device, non_blocking=True)
                    if self.task == "box"
                    else None
                )
                pred_object_score_logits = current_out["object_score_logits"].to(
                    device, non_blocking=True
                )
                if self.clear_non_cond_mem_around_input:
                    self._clear_obj_non_cond_mem_around_input(
                        inference_state, frame_idx, obj_idx
                    )
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks, pred_boxes_xyxy_norm = (
                    self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                )
                pred_object_score_logits = current_out["object_score_logits"]
                obj_output_dict[storage_key][frame_idx] = current_out

            inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                "reverse": reverse
            }
            pred_masks_per_obj[obj_idx] = pred_masks
            pred_boxes_xyxy_norm_per_obj[obj_idx] = pred_boxes_xyxy_norm
            pred_object_score_logits_per_obj[obj_idx] = pred_object_score_logits

        if len(pred_masks_per_obj) > 1:
            all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            all_pred_boxes_xyxy_norm = (
                torch.cat(pred_boxes_xyxy_norm_per_obj, dim=0)
                if self.task == "box"
                else None
            )
            all_pred_object_score_logits = torch.cat(
                pred_object_score_logits_per_obj, dim=0
            )
        else:
            all_pred_masks = pred_masks_per_obj[0]
            all_pred_boxes_xyxy_norm = (
                pred_boxes_xyxy_norm_per_obj[0] if self.task == "box" else None
            )
            all_pred_object_score_logits = pred_object_score_logits_per_obj[0]

        _, video_res_masks, video_res_boxes_xyxy = self._get_orig_video_res_output(
            inference_state, all_pred_masks, all_pred_boxes_xyxy_norm
        )
        return (
            frame_idx,
            obj_ids,
            video_res_masks,
            video_res_boxes_xyxy,
            all_pred_object_score_logits,
        )

    @torch.inference_mode()
    def prune_stream_state(
        self,
        inference_state,
        current_frame_idx,
        frame_buffer_size=16,
        memory_window=96,
        keep_conditioning=True,
    ):
        """Prune old streaming state without reindexing frames.

        `memory_window` controls old non-conditioning outputs. Conditioning
        frames are kept by default because they carry user prompts and identity
        anchors. `frame_buffer_size` controls normalized input tensors only; old
        memory outputs can remain even after their raw frame tensors are pruned.
        """
        if not inference_state.get("streaming", False):
            raise ValueError("prune_stream_state requires a streaming inference state")
        if frame_buffer_size < 1:
            raise ValueError("frame_buffer_size must be >= 1")
        if memory_window < 1:
            raise ValueError("memory_window must be >= 1")

        frame_keep_from = max(0, current_frame_idx - frame_buffer_size + 1)
        memory_keep_from = max(0, current_frame_idx - memory_window + 1)

        keep_frame_indices = set()
        if keep_conditioning:
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                keep_frame_indices.update(obj_output_dict["cond_frame_outputs"].keys())

        pruned_frames = inference_state["images"].prune_before(
            frame_keep_from, keep_indices=keep_frame_indices
        )

        cached_features = inference_state["cached_features"]
        pruned_cached_features = 0
        for idx in list(cached_features.keys()):
            if idx < frame_keep_from and idx not in keep_frame_indices:
                cached_features.pop(idx, None)
                pruned_cached_features += 1

        pruned_non_cond = 0
        pruned_tracked = 0
        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            non_cond = obj_output_dict["non_cond_frame_outputs"]
            for idx in list(non_cond.keys()):
                if idx < memory_keep_from:
                    non_cond.pop(idx, None)
                    pruned_non_cond += 1

            frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
            cond_frames = obj_output_dict["cond_frame_outputs"]
            for idx in list(frames_tracked.keys()):
                if idx < memory_keep_from and idx not in cond_frames:
                    frames_tracked.pop(idx, None)
                    pruned_tracked += 1

        stats = {
            "frame_keep_from": frame_keep_from,
            "memory_keep_from": memory_keep_from,
            "pruned_frames": pruned_frames,
            "live_frames": inference_state["images"].live_count(),
            "pruned_cached_features": pruned_cached_features,
            "pruned_non_cond_outputs": pruned_non_cond,
            "pruned_frames_tracked": pruned_tracked,
        }
        inference_state["last_prune_stats"] = stats
        return stats
