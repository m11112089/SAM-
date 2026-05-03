"""
Tests for sam2_plus.streaming_video_predictor
=============================================
These tests exercise the pure-Python logic of the streaming predictor,
RollingFrameStore, and OcclusionAndDriftMonitor without requiring a GPU,
PyTorch, or OpenCV.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Stub out torch and cv2 so the module loads in a minimal environment
# ---------------------------------------------------------------------------

def _make_torch_stub():
    """Return a minimal 'torch' stub that satisfies the import in the module."""
    torch = types.ModuleType("torch")
    torch.cuda = MagicMock()
    torch.cuda.is_available = lambda: False
    torch.backends = MagicMock()
    torch.backends.mps = MagicMock()
    torch.backends.mps.is_available = lambda: False
    torch.inference_mode = MagicMock(return_value=MagicMock(
        __enter__=lambda s, *a: None,
        __exit__=lambda s, *a: None,
    ))
    # Make torch.from_numpy a no-op that just returns the input
    torch.from_numpy = lambda x: x
    return torch


def _make_cv2_stub():
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2RGB = 4
    cv2.INTER_LINEAR = 1
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.cvtColor = lambda img, *args: img
    cv2.resize = lambda img, size, **kwargs: img
    cv2.rectangle = lambda img, *args, **kwargs: None
    cv2.putText = lambda img, *args, **kwargs: None
    cv2.addWeighted = lambda src1, alpha, src2, beta, gamma, dst=None: src2.copy()
    return cv2


# Inject stubs before importing the module under test
sys.modules.setdefault("torch", _make_torch_stub())
sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
sys.modules.setdefault("torch.nn.functional", types.ModuleType("torch.nn.functional"))
sys.modules.setdefault("cv2", _make_cv2_stub())

from sam2_plus.streaming_video_predictor import (  # noqa: E402
    FrameMetrics,
    OcclusionAndDriftMonitor,
    RollingFrameStore,
    SAM2StreamingVideoPredictor,
    TrackingState,
    _box_iou,
    _mask_to_box,
    compute_frame_metrics,
    render_frame,
)


# ===========================================================================
# RollingFrameStore tests
# ===========================================================================


class TestRollingFrameStore(unittest.TestCase):

    def _make_frame(self, value: float = 1.0):
        return np.full((3, 64, 64), value, dtype=np.float32)

    def test_add_and_get(self):
        store = RollingFrameStore()
        f = self._make_frame()
        store.add(0, f)
        retrieved = store.get(0)
        np.testing.assert_array_equal(retrieved, f)

    def test_missing_returns_none(self):
        store = RollingFrameStore()
        self.assertIsNone(store.get(99))

    def test_num_frames(self):
        store = RollingFrameStore()
        for i in range(5):
            store.add(i, self._make_frame())
        self.assertEqual(store.num_frames, 5)

    def test_drop_frame(self):
        store = RollingFrameStore()
        store.add(0, self._make_frame())
        store.drop_frame(0)
        self.assertIsNone(store.get(0))

    def test_pin_conditioning(self):
        store = RollingFrameStore(max_non_conditioning_frames=2)
        # Pin frame 0 before adding more frames so eviction cannot remove it
        store.add(0, self._make_frame())
        store.pin_conditioning(0)
        for i in range(1, 5):
            store.add(i, self._make_frame(float(i)))
        # Pinned frame should survive eviction of non-conditioning frames
        self.assertIsNotNone(store.get(0))

    def test_eviction_of_non_conditioning(self):
        """Frames beyond max_non_conditioning_frames are set to None."""
        store = RollingFrameStore(
            max_non_conditioning_frames=3,
            max_conditioning_frames=16,
        )
        for i in range(10):
            store.add(i, self._make_frame(float(i)))
        # The oldest non-pinned frames should have been evicted
        none_count = sum(1 for i in range(10) if store.get(i) is None)
        self.assertGreaterEqual(none_count, 7)  # 10 - 3 = 7 evicted

    def test_prune_before(self):
        store = RollingFrameStore()
        for i in range(10):
            store.add(i, self._make_frame())
        evicted = store.prune_before(5)
        self.assertEqual(sorted(evicted), [0, 1, 2, 3, 4])
        for i in range(5):
            self.assertIsNone(store.get(i))
        for i in range(5, 10):
            self.assertIsNotNone(store.get(i))

    def test_prune_before_skips_conditioning(self):
        store = RollingFrameStore()
        for i in range(10):
            store.add(i, self._make_frame())
        store.pin_conditioning(2)
        evicted = store.prune_before(5)
        self.assertNotIn(2, evicted)
        self.assertIsNotNone(store.get(2))

    def test_should_promote_keyframe_happy_path(self):
        store = RollingFrameStore(keyframe_interval=5)
        store._last_keyframe_idx = 0
        self.assertTrue(store.should_promote_keyframe(6, confidence=0.9, is_occluded=False))

    def test_should_not_promote_when_occluded(self):
        store = RollingFrameStore(keyframe_interval=5)
        store._last_keyframe_idx = 0
        self.assertFalse(store.should_promote_keyframe(6, confidence=0.9, is_occluded=True))

    def test_should_not_promote_low_confidence(self):
        store = RollingFrameStore(keyframe_interval=5)
        store._last_keyframe_idx = 0
        self.assertFalse(store.should_promote_keyframe(6, confidence=0.3, is_occluded=False))

    def test_should_not_promote_too_soon(self):
        store = RollingFrameStore(keyframe_interval=20)
        store._last_keyframe_idx = 0
        self.assertFalse(store.should_promote_keyframe(10, confidence=0.9, is_occluded=False))

    def test_conditioning_frames_list(self):
        store = RollingFrameStore()
        for i in range(5):
            store.add(i, self._make_frame())
        store.pin_conditioning(1)
        store.pin_conditioning(3)
        self.assertEqual(store.conditioning_frames, [1, 3])

    def test_max_conditioning_frames_evicts_oldest(self):
        store = RollingFrameStore(max_conditioning_frames=2)
        for i in range(5):
            store.add(i, self._make_frame())
            store.pin_conditioning(i)
        # Only 2 conditioning frames should remain
        self.assertLessEqual(len(store.conditioning_frames), 2)


# ===========================================================================
# OcclusionAndDriftMonitor tests
# ===========================================================================


class TestOcclusionAndDriftMonitor(unittest.TestCase):

    def _metrics(self, frame_idx=0, object_score=0.9, bbox_area=100.0):
        return FrameMetrics(
            frame_idx=frame_idx,
            mask_area=500.0,
            bbox_area=bbox_area,
            bbox_aspect=1.0,
            object_score=object_score,
            logit_strength=object_score,
            iou_with_prev=1.0,
            center_distance=0.0,
        )

    def test_tracking_state_on_good_frame(self):
        mon = OcclusionAndDriftMonitor()
        state = mon.update(self._metrics(object_score=0.9))
        self.assertEqual(state, TrackingState.TRACKING)

    def test_uncertain_on_medium_score(self):
        mon = OcclusionAndDriftMonitor(score_threshold=0.5, occluded_threshold=0.1)
        state = mon.update(self._metrics(object_score=0.3))
        self.assertEqual(state, TrackingState.UNCERTAIN)

    def test_occluded_on_low_score(self):
        mon = OcclusionAndDriftMonitor(score_threshold=0.5, occluded_threshold=0.1)
        state = mon.update(self._metrics(object_score=0.05))
        self.assertEqual(state, TrackingState.OCCLUDED)

    def test_lost_after_grace_period(self):
        mon = OcclusionAndDriftMonitor(lost_grace_frames=3)
        for i in range(4):
            state = mon.update(self._metrics(frame_idx=i, object_score=0.0))
        self.assertEqual(state, TrackingState.LOST)

    def test_recovery_from_occlusion(self):
        mon = OcclusionAndDriftMonitor(lost_grace_frames=10)
        # First occluded frame
        state_1 = mon.update(self._metrics(frame_idx=0, object_score=0.0))
        self.assertEqual(state_1, TrackingState.OCCLUDED)
        # Second occluded frame – still OCCLUDED (grace period not exhausted)
        state_2 = mon.update(self._metrics(frame_idx=1, object_score=0.0))
        self.assertEqual(state_2, TrackingState.OCCLUDED)
        # Then tracking recovers
        state = mon.update(self._metrics(frame_idx=2, object_score=0.9))
        self.assertEqual(state, TrackingState.TRACKING)
        self.assertEqual(mon._occluded_count, 0)

    def test_reset_clears_occlusion_count(self):
        mon = OcclusionAndDriftMonitor()
        for i in range(5):
            mon.update(self._metrics(frame_idx=i, object_score=0.0))
        mon.reset()
        self.assertEqual(mon.state, TrackingState.TRACKING)
        self.assertEqual(mon._occluded_count, 0)

    def test_request_correction(self):
        mon = OcclusionAndDriftMonitor()
        mon.update(self._metrics(object_score=0.9))
        mon.request_correction()
        self.assertEqual(mon.state, TrackingState.NEEDS_USER_CORRECTION)

    def test_last_good_metrics_updated_on_tracking(self):
        mon = OcclusionAndDriftMonitor()
        m = self._metrics(object_score=0.9)
        mon.update(m)
        self.assertIs(mon.last_good_metrics, m)

    def test_last_good_metrics_not_updated_on_uncertain(self):
        mon = OcclusionAndDriftMonitor()
        good = self._metrics(frame_idx=0, object_score=0.9)
        mon.update(good)
        bad = self._metrics(frame_idx=1, object_score=0.3)
        mon.update(bad)
        # last_good_metrics should still point to the good frame
        self.assertIs(mon.last_good_metrics, good)

    def test_drift_detected_on_area_explosion(self):
        mon = OcclusionAndDriftMonitor(area_change_threshold=3.0)
        mon.update(self._metrics(frame_idx=0, bbox_area=100.0))
        state = mon.update(self._metrics(frame_idx=1, bbox_area=1000.0))
        self.assertEqual(state, TrackingState.UNCERTAIN)


# ===========================================================================
# SAM2StreamingVideoPredictor tests (stub / no-GPU)
# ===========================================================================


class TestSAM2StreamingVideoPredictorStub(unittest.TestCase):

    def _make_frame(self, h=64, w=64):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def setUp(self):
        self.predictor = SAM2StreamingVideoPredictor(image_size=64)

    def test_init_stream_state_has_required_keys(self):
        state = self.predictor.init_stream_state(self._make_frame())
        self.assertIn("streaming", state)
        self.assertTrue(state["streaming"])
        self.assertIn("num_frames", state)
        self.assertIn("images", state)
        self.assertEqual(state["num_frames"], 1)
        self.assertEqual(len(state["images"]), 1)

    def test_append_frame_increments_num_frames(self):
        state = self.predictor.init_stream_state(self._make_frame())
        idx = self.predictor.append_frame(state, self._make_frame())
        self.assertEqual(idx, 1)
        self.assertEqual(state["num_frames"], 2)

    def test_append_multiple_frames(self):
        state = self.predictor.init_stream_state(self._make_frame())
        for i in range(9):
            idx = self.predictor.append_frame(state, self._make_frame())
        self.assertEqual(idx, 9)
        self.assertEqual(state["num_frames"], 10)
        self.assertEqual(len(state["images"]), 10)

    def test_append_frame_rejects_non_streaming_state(self):
        state = {"num_frames": 1, "images": [None]}  # no streaming=True
        with self.assertRaises(ValueError):
            self.predictor.append_frame(state, self._make_frame())

    def test_track_next_frame_returns_expected_keys(self):
        state = self.predictor.init_stream_state(self._make_frame())
        idx = self.predictor.append_frame(state, self._make_frame())
        result = self.predictor.track_next_frame(state, idx, obj_id=1)
        self.assertIn("masks", result)
        self.assertIn("boxes", result)
        self.assertIn("scores", result)
        self.assertIn("frame_idx", result)
        self.assertIn("obj_id", result)
        self.assertEqual(result["frame_idx"], idx)
        self.assertEqual(result["obj_id"], 1)

    def test_track_stub_mask_shape(self):
        frame = self._make_frame(h=32, w=48)
        state = self.predictor.init_stream_state(frame)
        idx = self.predictor.append_frame(state, frame)
        result = self.predictor.track_next_frame(state, idx, obj_id=1)
        mask = result["masks"][0]
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, bool)

    def test_prune_stream_state_nullifies_old_images(self):
        state = self.predictor.init_stream_state(self._make_frame())
        for _ in range(9):
            self.predictor.append_frame(state, self._make_frame())
        self.predictor.prune_stream_state(state, keep_from_frame_idx=5)
        # Frames 0-4 should be None
        for i in range(5):
            self.assertIsNone(state["images"][i])
        # Frames 5-9 should be intact (not None)
        for i in range(5, 10):
            self.assertIsNotNone(state["images"][i])

    def test_prune_stream_state_rejects_non_streaming(self):
        with self.assertRaises(ValueError):
            self.predictor.prune_stream_state({"num_frames": 1}, keep_from_frame_idx=0)

    def test_prune_stream_state_prunes_non_cond_outputs(self):
        state = self.predictor.init_stream_state(self._make_frame())
        # Manually add some fake non-conditioning outputs
        non_cond = state["output_dict"]["non_cond_frame_outputs"]
        for i in range(10):
            non_cond[i] = {"fake": True}
        self.predictor.prune_stream_state(state, keep_from_frame_idx=5)
        for i in range(5):
            self.assertNotIn(i, non_cond)
        for i in range(5, 10):
            self.assertIn(i, non_cond)

    def test_prune_keeps_conditioning_outputs_by_default(self):
        state = self.predictor.init_stream_state(self._make_frame())
        cond = state["output_dict"]["cond_frame_outputs"]
        cond[0] = {"user_prompt": True}
        self.predictor.prune_stream_state(state, keep_from_frame_idx=5)
        self.assertIn(0, cond)

    def test_prune_drops_conditioning_when_requested(self):
        state = self.predictor.init_stream_state(self._make_frame())
        cond = state["output_dict"]["cond_frame_outputs"]
        cond[0] = {"user_prompt": True}
        self.predictor.prune_stream_state(
            state, keep_from_frame_idx=5, keep_conditioning=False
        )
        self.assertNotIn(0, cond)

    def test_prune_evicts_cached_features(self):
        state = self.predictor.init_stream_state(self._make_frame())
        state["cached_features"] = {i: {"feat": True} for i in range(10)}
        self.predictor.prune_stream_state(state, keep_from_frame_idx=5)
        for i in range(5):
            self.assertNotIn(i, state["cached_features"])
        for i in range(5, 10):
            self.assertIn(i, state["cached_features"])

    def test_prune_updates_frames_tracked_per_obj(self):
        state = self.predictor.init_stream_state(self._make_frame())
        state["frames_tracked_per_obj"] = {1: set(range(10))}
        self.predictor.prune_stream_state(state, keep_from_frame_idx=5)
        remaining = state["frames_tracked_per_obj"][1]
        self.assertEqual(remaining, {5, 6, 7, 8, 9})

    def test_stream_frame_offset_in_state(self):
        state = self.predictor.init_stream_state(self._make_frame())
        self.assertIn("stream_frame_offset", state)
        self.assertEqual(state["stream_frame_offset"], 0)


# ===========================================================================
# Utility helper tests
# ===========================================================================


class TestUtilHelpers(unittest.TestCase):

    def test_mask_to_box_empty(self):
        mask = np.zeros((64, 64), dtype=bool)
        self.assertIsNone(_mask_to_box(mask))

    def test_mask_to_box_full(self):
        mask = np.ones((64, 64), dtype=bool)
        box = _mask_to_box(mask)
        np.testing.assert_array_equal(box, [0, 0, 63, 63])

    def test_mask_to_box_single_pixel(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10, 20] = True
        box = _mask_to_box(mask)
        np.testing.assert_array_equal(box, [20, 10, 20, 10])

    def test_box_iou_identical(self):
        box = np.array([0, 0, 10, 10], dtype=np.float32)
        self.assertAlmostEqual(_box_iou(box, box), 1.0)

    def test_box_iou_no_overlap(self):
        a = np.array([0, 0, 5, 5], dtype=np.float32)
        b = np.array([10, 10, 20, 20], dtype=np.float32)
        self.assertAlmostEqual(_box_iou(a, b), 0.0)

    def test_box_iou_partial(self):
        a = np.array([0, 0, 4, 4], dtype=np.float32)
        b = np.array([2, 2, 6, 6], dtype=np.float32)
        iou = _box_iou(a, b)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_compute_frame_metrics_no_box(self):
        m = compute_frame_metrics(0, mask=None, box=None, score=0.5)
        self.assertEqual(m.frame_idx, 0)
        self.assertAlmostEqual(m.object_score, 0.5)
        self.assertAlmostEqual(m.mask_area, 0.0)

    def test_compute_frame_metrics_with_box(self):
        mask = np.ones((10, 10), dtype=bool)
        box = np.array([0, 0, 10, 10], dtype=np.float32)
        m = compute_frame_metrics(5, mask, box, score=0.8)
        self.assertEqual(m.frame_idx, 5)
        self.assertAlmostEqual(m.mask_area, 100.0)
        self.assertAlmostEqual(m.bbox_area, 100.0)

    def test_compute_frame_metrics_center_distance(self):
        box_prev = np.array([0, 0, 10, 10], dtype=np.float32)
        box_curr = np.array([10, 10, 20, 20], dtype=np.float32)
        m = compute_frame_metrics(1, mask=None, box=box_curr, score=0.9, prev_box=box_prev)
        # Centers: prev=(5,5), curr=(15,15) → distance = sqrt((15-5)^2 + (15-5)^2)
        expected = np.sqrt((15.0 - 5.0) ** 2 + (15.0 - 5.0) ** 2)
        self.assertAlmostEqual(m.center_distance, expected, places=4)

    def test_render_frame_returns_ndarray(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)
        box = np.array([5, 5, 20, 20], dtype=np.float32)
        # cv2 is stubbed so this should not raise
        out = render_frame(bgr, mask, box, 0, TrackingState.TRACKING, 0.9)
        self.assertIsInstance(out, np.ndarray)

    def test_render_frame_no_mask_no_box(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        out = render_frame(bgr, None, None, 0, TrackingState.LOST, 0.0)
        self.assertIsInstance(out, np.ndarray)


# ===========================================================================
# Argument parser tests (CLI modules)
# ===========================================================================


class TestStreamingCLIParser(unittest.TestCase):
    """Verify that streaming_video_track.py exposes the required flags."""

    def setUp(self):
        # Stub out heavy imports in the CLI module
        for mod in ("sam2", "sam2.build_sam"):
            sys.modules.setdefault(mod, types.ModuleType(mod))

    def test_required_flags_present(self):
        import streaming_video_track as svt

        p = svt.build_arg_parser()
        args = p.parse_args([
            "--source", "video.mp4",
            "--checkpoint", "ckpt.pt",
            "--config", "cfg.yaml",
            "--box", "10", "20", "30", "40",
        ])
        self.assertEqual(args.source, "video.mp4")
        self.assertEqual(args.box, [10.0, 20.0, 30.0, 40.0])
        # Phase-2 flags with defaults
        self.assertEqual(args.frame_buffer_size, 128)
        self.assertEqual(args.memory_window, 96)
        self.assertEqual(args.prune_every, 30)
        self.assertFalse(args.print_prune_stats)

    def test_prune_flags_accepted(self):
        import streaming_video_track as svt

        p = svt.build_arg_parser()
        args = p.parse_args([
            "--source", "v.mp4",
            "--checkpoint", "c.pt",
            "--config", "g.yaml",
            "--box", "0", "0", "100", "100",
            "--frame-buffer-size", "64",
            "--memory-window", "32",
            "--prune-every", "15",
            "--print-prune-stats",
        ])
        self.assertEqual(args.frame_buffer_size, 64)
        self.assertEqual(args.memory_window, 32)
        self.assertEqual(args.prune_every, 15)
        self.assertTrue(args.print_prune_stats)


class TestInteractiveCLIParser(unittest.TestCase):
    """Verify that interactive_video_box_track.py exposes the required flags."""

    def setUp(self):
        for mod in ("sam2", "sam2.build_sam"):
            sys.modules.setdefault(mod, types.ModuleType(mod))

    def test_required_flags_present(self):
        import interactive_video_box_track as ivbt

        p = ivbt.build_arg_parser()
        args = p.parse_args([
            "--source", "video.mp4",
            "--checkpoint", "ckpt.pt",
            "--config", "cfg.yaml",
        ])
        self.assertEqual(args.frame_buffer_size, 128)
        self.assertEqual(args.memory_window, 96)
        self.assertEqual(args.prune_every, 30)
        self.assertFalse(args.print_prune_stats)

    def test_prune_flags_customised(self):
        import interactive_video_box_track as ivbt

        p = ivbt.build_arg_parser()
        args = p.parse_args([
            "--source", "v.mp4",
            "--checkpoint", "c.pt",
            "--config", "g.yaml",
            "--frame-buffer-size", "256",
            "--memory-window", "128",
            "--prune-every", "50",
            "--print-prune-stats",
            "--no-display",
        ])
        self.assertEqual(args.frame_buffer_size, 256)
        self.assertEqual(args.memory_window, 128)
        self.assertEqual(args.prune_every, 50)
        self.assertTrue(args.print_prune_stats)
        self.assertTrue(args.no_display)


if __name__ == "__main__":
    unittest.main()
