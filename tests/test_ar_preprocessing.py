import os

import numpy as np
import pytest
import torch
from mmengine import Config

from precision_track.models import ActionRecognitionPreprocessor

ROOT = "./tests/"


class MockDataSample:
    def __init__(self, pred_track_instances, img_id):
        self.pred_track_instances = pred_track_instances
        self.img_id = img_id

    def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.fixture
def metainfo():
    return Config.fromfile(os.path.join(ROOT, "configs/action_recognition.py"))["metainfo"]


@pytest.fixture
def preprocessor_config():
    """Default configuration for ActionRecognitionPreprocessor"""
    return {
        "embd_size": 256,
        "block_size": 8,
        "max_size": 10,
        "kpts_conf_thr": 0.5,
        "device": "cpu",
        "velocity_encoder": {"type": "BaseVelocityEncoder"},
        "with_vels": True,
        "with_kpts": True,
        "with_kpt_vels": False,
        "with_actions": True,
    }


@pytest.fixture
def preprocessor(metainfo, preprocessor_config):
    """Create a ActionRecognitionPreprocessor instance with mocked dependencies"""
    preprocessor_config["metainfo"] = metainfo
    return ActionRecognitionPreprocessor(**preprocessor_config)


def create_track_instance(num_instances, embd_size=256, num_keypoints=17, ids=None, labels=None, include_actions=True, feature_pattern=None):
    """Helper to create pred_track_instances dict

    Args:
        feature_pattern: If provided, use this as features. Should be shape (num_instances, embd_size)
                        This allows creating deterministic features for testing.
    """
    if ids is None:
        ids = np.arange(num_instances)
    if labels is None:
        labels = np.zeros(num_instances, dtype=np.int64)

    if feature_pattern is not None:
        features = feature_pattern
    else:
        features = np.random.randn(num_instances, embd_size).astype(np.float32)

    instance = {
        "instances_id": ids,
        "labels": labels,
        "features": features,
        "velocities": np.random.randn(num_instances, 2).astype(np.float32),
        "keypoints": np.random.randn(num_instances, num_keypoints, 2).astype(np.float32),
        "keypoint_scores": np.random.rand(num_instances, num_keypoints).astype(np.float32),
    }

    if include_actions:
        instance["actions"] = np.random.randint(0, 10, size=(num_instances, 1)).astype(np.float32)

    return instance


def create_simple_feature_pattern(instance_id, frame_id, embd_size=256):
    """Create a simple, predictable feature pattern for testing.

    Pattern: Each feature vector is filled with (instance_id * 100 + frame_id)
    This makes it easy to verify which instance and frame a feature belongs to.

    Example:
        Instance 0, Frame 3 -> all features = 3.0
        Instance 1, Frame 5 -> all features = 105.0
        Instance 2, Frame 7 -> all features = 207.0
    """
    value = float(instance_id * 100 + frame_id)
    return np.full((1, embd_size), value, dtype=np.float32)


def verify_block_pattern(block_features, idx, expected_pattern, block_size):
    """Verify that the block contains the expected pattern.

    Args:
        block_features: The block tensor to check
        idx: The index in the block to check
        expected_pattern: List of expected values in chronological order
        block_size: Size of the ring buffer
    """

    assert len(expected_pattern) == block_size, "Length pattern != block size"

    # Read the block in chronological order
    for i in range(block_size):
        values = block_features[idx, i, ...]
        assert torch.all(values == expected_pattern[i]), f"{values[0].item()} != expected_pattern[i]"


class TestEasyCase:
    def test_continuous_tracking_single_instance(self, preprocessor):
        block_size = preprocessor._block_size
        embd_size = preprocessor._embd_size

        expected_values = torch.zeros(block_size)
        valid_context = False

        # Track the same instance (id=0, label=0) through all frames
        for frame_id in range(100):
            features = create_simple_feature_pattern(0, frame_id, embd_size)

            if frame_id >= block_size:
                expected_values[:-1] = expected_values[1:].clone()
                expected_values[-1] = float(frame_id)
                valid_context = True
            else:
                expected_values[frame_id] = float(frame_id)

            track_instance = create_track_instance(num_instances=1, embd_size=embd_size, ids=np.array([0]), labels=np.array([0]), feature_pattern=features)

            data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=frame_id)}

            out = preprocessor.forward(data)

            assert "features" in out
            assert out["features"].shape[0] == 1  # One active instance
            assert out["features"].shape[1] == block_size
            assert out["features"].shape[2] == embd_size

            idx = preprocessor.ids2idx.get("0-0")
            verify_block_pattern(out["features"], idx, expected_values, block_size)
            out_valid_context = data["data_samples"].pred_track_instances["valid_action_recognition_context"]
            assert out_valid_context.item() == valid_context

        # Verify the instance is still tracked
        assert preprocessor.ids2idx.size() == 1
        assert preprocessor.ids2idx.has("0-0")

    def test_continuous_tracking_multiple_instances(self, preprocessor):
        block_size = preprocessor._block_size
        embd_size = preprocessor._embd_size
        num_instances = 3

        expected_values = {0: torch.zeros(block_size), 1: torch.zeros(block_size), 2: torch.zeros(block_size)}
        valid_contexts = [False, False, False]

        for frame_id in range(100):
            features_list = []
            for inst_id in range(num_instances):
                features = create_simple_feature_pattern(inst_id, frame_id, embd_size)
                features_list.append(features)

                new_val = features[0][0].item()
                if frame_id >= block_size:
                    expected_values[inst_id][:-1] = expected_values[inst_id][1:].clone()
                    expected_values[inst_id][-1] = new_val
                    valid_contexts[inst_id] = True
                else:
                    expected_values[inst_id][frame_id] = new_val

            all_features = np.concatenate(features_list, axis=0)

            track_instance = create_track_instance(
                num_instances=num_instances, embd_size=embd_size, ids=np.array([0, 1, 2]), labels=np.array([0, 0, 1]), feature_pattern=all_features
            )

            data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=frame_id)}

            out = preprocessor.forward(data)

            assert out["features"].shape[0] == num_instances
            assert out["features"].shape[1] == block_size

            for inst_id in preprocessor.ids2idx.id2idx.keys():
                idx = preprocessor.ids2idx.get(inst_id)
                verify_block_pattern(out["features"], idx, expected_values[idx], block_size)
                out_valid_context = data["data_samples"].pred_track_instances["valid_action_recognition_context"][idx]
                assert out_valid_context.item() == valid_contexts[idx]

        assert preprocessor.ids2idx.size() == num_instances


class TestHardCase:
    def test_disappearing_instances(self, preprocessor):
        block_size = preprocessor._block_size
        embd_size = preprocessor._embd_size

        expected_values = {0: torch.zeros(block_size), 1: torch.zeros(block_size), 2: torch.zeros(block_size)}
        dissapearing_frames = [2, 8, 50]
        labels = np.array([0, 0, 1])
        ids = np.array([0, 1, 0])
        all_instances = np.array([0, 1, 2])

        for frame_id in range(58):
            features_list = []
            still_tracked_mask = np.ones_like(ids, dtype=bool)
            for inst_id in all_instances:
                features = create_simple_feature_pattern(inst_id, frame_id, embd_size)
                features_list.append(features)

                diss_frame = dissapearing_frames[inst_id]
                if frame_id < diss_frame:
                    new_val = features[0][0].item()
                else:
                    still_tracked_mask[inst_id] = 0
                    new_val = float(0)
                if frame_id >= block_size:
                    expected_values[inst_id][:-1] = expected_values[inst_id][1:].clone()
                    expected_values[inst_id][-1] = new_val
                else:
                    expected_values[inst_id][frame_id] = new_val

            all_features = np.concatenate(features_list, axis=0)
            nb_tracked_inst = still_tracked_mask.sum()
            tracked_labels = labels[still_tracked_mask]
            tracked_ids = ids[still_tracked_mask]
            tracked_insts = all_instances[still_tracked_mask]

            track_instance = create_track_instance(
                num_instances=nb_tracked_inst,
                embd_size=embd_size,
                ids=tracked_ids,
                labels=tracked_labels,
                feature_pattern=all_features[still_tracked_mask],
            )

            data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=frame_id)}

            out = preprocessor.forward(data)

            assert out["features"].shape[0] == nb_tracked_inst
            assert out["poses"].shape[0] == nb_tracked_inst
            assert out["dynamics"].shape[0] == nb_tracked_inst
            assert out["actions"].shape[0] == nb_tracked_inst

            for i, tracked_label, tracked_id in zip(tracked_insts, tracked_labels, tracked_ids):
                unique_id = f"{tracked_label}-{tracked_id}"
                assert preprocessor.ids2idx.has(unique_id)
                idx = np.where(
                    (tracked_label == data["data_samples"].pred_track_instances["labels"])
                    & (tracked_id == data["data_samples"].pred_track_instances["instances_id"])
                )[0]
                verify_block_pattern(out["features"], idx, expected_values[i], block_size)

            for i, lost_label, lost_id in zip(all_instances[~still_tracked_mask], labels[~still_tracked_mask], ids[~still_tracked_mask]):
                unique_id = f"{lost_label}-{lost_id}"
                diss_frame = dissapearing_frames[i]
                if diss_frame + block_size - 1 <= frame_id:
                    assert not preprocessor.ids2idx.has(unique_id)
                    verify_block_pattern(preprocessor.block_features, i, torch.zeros(block_size), block_size)
                else:
                    assert preprocessor.ids2idx.has(unique_id)
                    assert not np.any(
                        (lost_label == data["data_samples"].pred_track_instances["labels"])
                        & (lost_id == data["data_samples"].pred_track_instances["instances_id"])
                    )

        block_sum = preprocessor.block_features.abs().sum().item()
        assert block_sum == 0.0, f"Blocks should be zeroed after deletion, sum={block_sum}"

    def test_appearing_instances(self, preprocessor):
        block_size = preprocessor._block_size
        embd_size = preprocessor._embd_size

        expected_values = {0: torch.zeros(block_size), 1: torch.zeros(block_size), 2: torch.zeros(block_size)}
        apearing_frames = [1, 5, 15]
        labels = np.array([0, 0, 0])
        ids = np.array([0, 1, 2])
        all_instances = np.array([0, 1, 2])
        valid_contexts = [False, False, False]

        for frame_id in range(58):
            features_list = []
            tracked_mask = np.zeros_like(ids, dtype=bool)
            for inst_id in all_instances:
                features = create_simple_feature_pattern(inst_id, frame_id, embd_size)
                features_list.append(features)

                app_frame = apearing_frames[inst_id]
                if frame_id < app_frame:
                    new_val = float(0)
                else:
                    tracked_mask[inst_id] = 1
                    new_val = features[0][0].item()
                    block_frame_id = frame_id - app_frame
                    if block_frame_id >= block_size:
                        expected_values[inst_id][:-1] = expected_values[inst_id][1:].clone()
                        expected_values[inst_id][-1] = new_val
                        valid_contexts[inst_id] = True
                    else:
                        expected_values[inst_id][block_frame_id] = new_val

            all_features = np.concatenate(features_list, axis=0)
            nb_tracked_inst = tracked_mask.sum()
            tracked_labels = labels[tracked_mask]
            tracked_ids = ids[tracked_mask]
            tracked_insts = all_instances[tracked_mask]

            track_instance = create_track_instance(
                num_instances=nb_tracked_inst,
                embd_size=embd_size,
                ids=tracked_ids,
                labels=tracked_labels,
                feature_pattern=all_features[tracked_mask],
            )

            data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=frame_id)}

            out = preprocessor.forward(data)

            assert out["features"].shape[0] == nb_tracked_inst
            assert out["poses"].shape[0] == nb_tracked_inst
            assert out["dynamics"].shape[0] == nb_tracked_inst
            assert out["actions"].shape[0] == nb_tracked_inst

            for i, tracked_label, tracked_id in zip(tracked_insts, tracked_labels, tracked_ids):
                unique_id = f"{tracked_label}-{tracked_id}"
                assert preprocessor.ids2idx.has(unique_id)
                idx = np.where(
                    (tracked_label == data["data_samples"].pred_track_instances["labels"])
                    & (tracked_id == data["data_samples"].pred_track_instances["instances_id"])
                )[0].item()
                verify_block_pattern(out["features"], idx, expected_values[i], block_size)
                out_valid_context = data["data_samples"].pred_track_instances["valid_action_recognition_context"][idx]
                assert out_valid_context.item() == valid_contexts[idx]

            for i, not_tracked_yet_label, not_tracked_yet_id in zip(all_instances[~tracked_mask], labels[~tracked_mask], ids[~tracked_mask]):
                unique_id = f"{not_tracked_yet_label}-{not_tracked_yet_id}"
                assert not preprocessor.ids2idx.has(unique_id)
                verify_block_pattern(preprocessor.block_features, i, torch.zeros(block_size), block_size)
                assert not np.any(
                    (not_tracked_yet_label == data["data_samples"].pred_track_instances["labels"])
                    & (not_tracked_yet_id == data["data_samples"].pred_track_instances["instances_id"])
                )

    def test_intermittent_tracking(self, preprocessor):
        block_size = preprocessor._block_size
        embd_size = preprocessor._embd_size

        expected_values = {0: torch.zeros(block_size), 1: torch.zeros(block_size), 2: torch.zeros(block_size), 3: torch.zeros(block_size)}
        apearing_frames = [1, 5, 15, 25]
        dissapearing_frames = [2, 13, 16, 26]
        reapearing_frames = [5, 21, 35, 27]
        labels = np.array([0, 0, 0, 1])
        ids = np.array([0, 1, 2, 0])
        all_instances = np.array([0, 1, 2, 3])
        valid_contexts = [False, False, False, False]

        for frame_id in range(100):
            features_list = []
            tracked_mask = np.zeros_like(ids, dtype=bool)
            for inst_id in all_instances:
                features = create_simple_feature_pattern(inst_id, frame_id, embd_size)
                features_list.append(features)

                app_frame = apearing_frames[inst_id]
                diss_frame = dissapearing_frames[inst_id]
                rea_frame = reapearing_frames[inst_id]

                if frame_id < app_frame or diss_frame <= frame_id < rea_frame:
                    new_val = float(0)
                else:
                    tracked_mask[inst_id] = 1
                    new_val = features[0][0].item()
                    block_frame_id = frame_id - app_frame if frame_id < rea_frame else frame_id - rea_frame
                    if block_frame_id >= block_size:
                        expected_values[inst_id][:-1] = expected_values[inst_id][1:].clone()
                        expected_values[inst_id][-1] = new_val
                        valid_contexts[inst_id] = True
                    else:
                        expected_values[inst_id][block_frame_id] = new_val
                        valid_contexts[inst_id] = False

            all_features = np.concatenate(features_list, axis=0)
            nb_tracked_inst = tracked_mask.sum()
            tracked_labels = labels[tracked_mask]
            tracked_ids = ids[tracked_mask]
            tracked_insts = all_instances[tracked_mask]

            track_instance = create_track_instance(
                num_instances=nb_tracked_inst,
                embd_size=embd_size,
                ids=tracked_ids,
                labels=tracked_labels,
                feature_pattern=all_features[tracked_mask],
            )

            data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=frame_id)}

            out = preprocessor.forward(data)

            assert out["features"].shape[0] == nb_tracked_inst
            assert out["poses"].shape[0] == nb_tracked_inst
            assert out["dynamics"].shape[0] == nb_tracked_inst
            assert out["actions"].shape[0] == nb_tracked_inst

            out_valid_context = np.where(data["data_samples"].pred_track_instances["valid_action_recognition_context"] == 1)[0]

            if 9 <= frame_id:
                assert np.isin(0, out_valid_context)
            else:
                assert not np.isin(0, out_valid_context)

            if 29 <= frame_id:
                assert np.isin(1, out_valid_context)
            else:
                assert not np.isin(1, out_valid_context)

            if 33 <= frame_id < 35:
                assert np.isin(2, out_valid_context)
            elif 35 <= frame_id:
                if frame_id < 43:
                    assert not np.isin(2, out_valid_context)
                assert np.isin(3, out_valid_context)
            else:
                assert not np.isin(3, out_valid_context)

            if (1 <= frame_id < 2) or (5 <= frame_id):
                assert preprocessor.ids2idx.has("0-0")
                idx = preprocessor.ids2idx.get("0-0")
                assert idx == 0
            if (5 <= frame_id < 13) or (21 <= frame_id):
                assert preprocessor.ids2idx.has("0-1")
                idx = preprocessor.ids2idx.get("0-1")
                assert idx == 1
            if 15 <= frame_id < 16:
                assert preprocessor.ids2idx.has("0-2")
                idx = preprocessor.ids2idx.get("0-2")
                assert idx == 2
            if 35 == frame_id:
                assert preprocessor.ids2idx.has("0-2")
                idx = preprocessor.ids2idx.get("0-2")
                assert idx == 3
            if (25 <= frame_id < 26) or (27 <= frame_id):
                assert preprocessor.ids2idx.has("1-0")
                idx = preprocessor.ids2idx.get("1-0")
                assert idx == 2

            for i, tracked_label, tracked_id in zip(tracked_insts, tracked_labels, tracked_ids):
                unique_id = f"{tracked_label}-{tracked_id}"
                assert preprocessor.ids2idx.has(unique_id)
                idx = np.where(
                    (tracked_label == data["data_samples"].pred_track_instances["labels"])
                    & (tracked_id == data["data_samples"].pred_track_instances["instances_id"])
                )[0].item()
                out_valid_context = data["data_samples"].pred_track_instances["valid_action_recognition_context"][idx]
                if ((9 <= frame_id < 13) and idx == 0) or ((33 <= frame_id < 35) and idx == 2):
                    assert out_valid_context.item() != valid_contexts[idx]  # Should still run on the 1st appearance context, since it did not die
                    if frame_id == 33:
                        verify_block_pattern(out["features"], idx, [0.0, 327, 328, 329, 330, 331, 332, 333], block_size)
                    if frame_id == 34:
                        verify_block_pattern(out["features"], idx, [327, 328, 329, 330, 331, 332, 333, 334], block_size)
                    if frame_id == 35:
                        verify_block_pattern(out["features"], idx, [328, 329, 330, 331, 332, 333, 335], block_size)
                else:
                    assert out_valid_context.item() == valid_contexts[idx], f"{frame_id}-{idx}"
                if valid_contexts[idx]:
                    verify_block_pattern(out["features"], idx, expected_values[i], block_size)

        assert preprocessor.ids2idx.has("0-0")
        idx = preprocessor.ids2idx.get("0-0")
        assert idx == 0

        assert preprocessor.ids2idx.has("0-1")
        idx = preprocessor.ids2idx.get("0-1")
        assert idx == 1

        assert preprocessor.ids2idx.has("0-2")
        idx = preprocessor.ids2idx.get("0-2")
        assert idx == 3

        assert preprocessor.ids2idx.has("1-0")
        idx = preprocessor.ids2idx.get("1-0")
        assert idx == 2


class TestEdgeCases:
    def test_max_capacity(self, preprocessor):
        max_size = preprocessor._max_size
        embd_size = preprocessor._embd_size

        track_instance = create_track_instance(num_instances=max_size, embd_size=embd_size, ids=np.arange(max_size), labels=np.zeros(max_size, dtype=np.int64))

        data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=0)}

        preprocessor.forward(data)
        assert preprocessor.ids2idx.size() == max_size

        track_instance = create_track_instance(num_instances=1, embd_size=embd_size, ids=np.array([999]), labels=np.array([1]))

        data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=1)}

        with pytest.raises(RuntimeError, match="at capacity"):
            preprocessor.forward(data)

    def test_empty_frames(self, preprocessor):
        embd_size = preprocessor._embd_size

        for frame_id in range(5):
            track_instance = create_track_instance(num_instances=0, embd_size=embd_size, ids=np.array([]), labels=np.array([]))

            data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=frame_id)}

            out = preprocessor.forward(data)

            assert isinstance(out, dict)

    def test_id_reuse_after_expiration(self, preprocessor):
        block_size = preprocessor._block_size
        embd_size = preprocessor._embd_size

        track_instance = create_track_instance(num_instances=1, embd_size=embd_size, ids=np.array([0]), labels=np.array([0]))

        data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=0)}

        preprocessor.forward(data)
        first_idx = preprocessor.ids2idx.get("0-0")

        for frame_id in range(1, block_size + 1):
            track_instance = create_track_instance(num_instances=0, embd_size=embd_size, ids=np.array([]), labels=np.array([]))

            data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=frame_id)}

            preprocessor.forward(data)

        assert not preprocessor.ids2idx.has("0-0")

        track_instance = create_track_instance(num_instances=1, embd_size=embd_size, ids=np.array([999]), labels=np.array([1]))

        data = {"data_samples": MockDataSample(pred_track_instances=track_instance, img_id=block_size + 1)}

        preprocessor.forward(data)
        second_idx = preprocessor.ids2idx.get("1-999")

        assert second_idx == first_idx


if __name__ == "__main__":
    pytest.main(["-x", os.path.realpath(__file__), "-v", "-s"])
