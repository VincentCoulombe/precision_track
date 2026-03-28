import json
import multiprocessing as mp
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from mmengine import Config
from mmengine.structures import InstanceData

from precision_track import PipelinedTracker, Tracker
from precision_track.registry import DATASETS, MODELS, OUTPUTS
from precision_track.utils import VideoReader, cuda_available, load_user_configs

ROOT = "./tests/"

TOOLS_DIR = "./tools"


@pytest.fixture
def config():
    return os.path.join(ROOT, "configs/action_recognition.py")


@pytest.fixture
def user_configs():
    return os.path.join(ROOT, "../configs/user_configs.yaml")


@pytest.fixture
def training_config():
    return os.path.join(ROOT, "configs/training_action_recognition.py")


@pytest.fixture
def testing_config():
    return os.path.join(ROOT, "configs/testing_action_recognition.py")


@pytest.fixture
def deployed_checkpoints():
    ckpt_names = [
        "model_mart_DEPLOYED.pth",
        "model_mart_DEPLOYED_NVIDIAGeForceRTX3090_FP16.engine",
        "model_mart_DEPLOYED.onnx",
    ]
    return dict(
        checkpoint_names=ckpt_names,
        found=[False for _ in ckpt_names],
    )


def set_user_configs(with_action_recognition: bool, user_configs, system_config, deploying_directory="", metadata_file=""):
    with open(user_configs, "r") as f:
        user_configs = yaml.safe_load(f)
    user_configs["booleans"]["with_action_recognition"] = with_action_recognition
    if deploying_directory:
        user_configs["training"]["deploying_directory"] = deploying_directory + "/"
    if metadata_file:
        user_configs["training"]["metainfo"] = metadata_file
    load_user_configs(user_configs, system_config)


@pytest.mark.timeout(10 * 60)
@pytest.mark.parametrize(
    "checkpoints",
    [
        (
            os.path.join(ROOT, "configs/mart.pth"),
            os.path.join(ROOT, "configs/mart_DEPLOYED.pth"),
            os.path.join(ROOT, "configs/mart_DEPLOYED.onnx"),
            os.path.join(ROOT, "configs/mart_DEPLOYED_NVIDIAGeForceRTX3090_FP16.engine"),
        )
    ],
)
def test_inference(checkpoints, config):
    config = Config.fromfile(config)
    analyzer = config["analyzer"]

    for checkpoint in checkpoints:
        if checkpoint.endswith(".engine") and not cuda_available():
            continue
        if checkpoint.endswith(".onnx") and not cuda_available():
            checkpoint = os.path.splitext(checkpoint)[0] + "_cpu.onnx"
        analyzer["runtime"]["checkpoint"] = checkpoint
        for i, pipelined in enumerate([True, False]):
            video = VideoReader(os.path.join(ROOT, "../assets/20mice_sanity_check.avi"))
            config["outputs"] = [
                dict(
                    type="CsvActions",
                    path=os.path.join(ROOT, f"work_dir/actions{i}.csv"),
                    instance_data="pred_track_instances",
                    metainfo=config["metainfo"],
                    precision=64,
                )
            ]
            if checkpoint.endswith(".pth") and not pipelined:
                config["outputs"].append(
                    dict(
                        type="CsvActions",
                        path=os.path.join(ROOT, "work_dir/reference.csv"),
                        instance_data="pred_track_instances",
                        metainfo=config["metainfo"],
                        precision=64,
                    )
                )
            if pipelined:
                mp.set_start_method("spawn", force=True)
                tracker = PipelinedTracker(
                    detector=config.get("detector"),
                    assigner=config.get("assigner"),
                    validator=config.get("validator"),
                    analyzer=config.get("analyzer"),
                    outputs=config.get("outputs"),
                    expected_resolution=(video.resolution[1], video.resolution[0], 3),
                    batch_size=config.get("batch_size"),
                    verbose=True,
                )
                tracker(video=video)
            else:
                tracker = Tracker(
                    detector=config.get("detector"),
                    assigner=config.get("assigner"),
                    validator=config.get("validator"),
                    analyzer=config.get("analyzer"),
                    outputs=config.get("outputs"),
                    batch_size=config.get("batch_size"),
                    verbose=True,
                )
                tracker(video=video)

        df0 = pd.read_csv(os.path.join(ROOT, "work_dir/actions0.csv"))
        assert not df0.empty
        df1 = pd.read_csv(os.path.join(ROOT, "work_dir/actions1.csv"))
        assert df0.equals(df1)

        if os.path.exists(os.path.join(ROOT, "work_dir/reference.csv")):
            dv_ref = pd.read_csv(os.path.join(ROOT, "work_dir/reference.csv")).values

        dv0 = df0.values
        for i in range(len(df0)):
            # If the predicted action differs, the softmax scores need to at least be close.
            if not np.all(np.equal(dv0[i, :-1], dv_ref[i, :-1])):
                assert np.isclose(dv0[i, -1], dv_ref[i, -1], atol=1e-2), f"{dv0[i, :]} != {dv_ref[i, :]}."


@pytest.mark.parametrize(
    "predict_inputs, loss_sequence_input",
    [
        (
            [
                os.path.join(ROOT, "work_dir/1079.pth"),
                os.path.join(ROOT, "work_dir/1080.pth"),
                os.path.join(ROOT, "work_dir/1081.pth"),
                os.path.join(ROOT, "work_dir/1082.pth"),
            ],
            os.path.join(ROOT, "work_dir/1079-1082_id17.pth"),
        ),
    ],
)
def test_training_preprocessing(predict_inputs, loss_sequence_input, config):
    ar_preprocessing = MODELS.build(Config.fromfile(config)["training_data_preprocessor"])
    ar_preprocessing.block_size = 4

    inf_ar_pre_cfg = Config.fromfile(config)["analyzer"]["data_preprocessor"]
    inf_ar_pre_cfg.block_size = 3
    inf_ar_preprocessing = MODELS.build(inf_ar_pre_cfg)

    map_location = "cuda" if cuda_available() else "cpu"

    for predict_input in predict_inputs:
        loaded_ds = torch.load(predict_input, weights_only=False, map_location=torch.device(map_location))
        predict_output = inf_ar_preprocessing(loaded_ds)

    loss_sequence_input = torch.load(loss_sequence_input, weights_only=False, map_location=torch.device(map_location))
    for ds in loss_sequence_input["data_samples"]:
        ds.pred_track_instances.velocities = ds.pred_track_instances.dynamics
    loss_output = ar_preprocessing.loss(loss_sequence_input)

    assert torch.allclose(loss_output["features"][-1][1:].to(torch.float16), predict_output["features"][-1].to(torch.float16))
    assert torch.allclose(loss_output["poses"][-1][1:].to(torch.float16), predict_output["poses"][-1].to(torch.float16))
    assert torch.allclose(loss_output["dynamics"][-1][1:].to(torch.float16), predict_output["dynamics"][-1].to(torch.float16))

    loss_sequence_input["inputs"][0] = loss_sequence_input["inputs"][0].view(1, 4, 128)

    pred_track_instances = InstanceData()

    pred_track_instances.velocities = loss_sequence_input["data_samples"][0].pred_track_instances.dynamics.view(1, 4, 2)
    pred_track_instances.kpts = loss_sequence_input["data_samples"][0].pred_track_instances.kpts.view(1, 4, 8, 2)
    pred_track_instances.kpt_vis = loss_sequence_input["data_samples"][0].pred_track_instances.kpt_vis.view(1, 4, 8)
    loss_sequence_input["data_samples"][0].pred_track_instances = pred_track_instances
    sequence_output = ar_preprocessing.sequence(loss_sequence_input)

    assert torch.allclose(sequence_output["features"][-1][1:].to(torch.float16), predict_output["features"][-1].to(torch.float16))
    assert torch.allclose(sequence_output["poses"][-1][1:].to(torch.float16), predict_output["poses"][-1].to(torch.float16))
    assert torch.allclose(sequence_output["dynamics"][-1][1:].to(torch.float16), predict_output["dynamics"][-1].to(torch.float16))


def test_action_recognition_dataset(training_config, user_configs):

    set_user_configs(True, user_configs, training_config, deploying_directory=os.path.join(ROOT, "configs"), metadata_file="./configs/metadata/mice.py")
    try:
        train_config = Config.fromfile(training_config)
        val_data_prefix = train_config.val_dataloader["dataset"]["data_prefix"]
        data_root = train_config.val_dataloader["dataset"]["data_root"]
        train_config.val_dataloader["dataset"]["data_root"] = os.path.abspath(os.path.join(TOOLS_DIR, data_root))

        bboxes_gt_path = val_data_prefix["bboxes_gt_paths"][0]
        kpts_gt_path = val_data_prefix["keypoints_gt_paths"][0]
        actions_gt_path = val_data_prefix["actions_gt_paths"][0]
        sequence_path = val_data_prefix["sequences"][0]

        if os.path.isfile(bboxes_gt_path) and os.path.isfile(kpts_gt_path) and os.path.isfile(actions_gt_path) and os.path.isfile(sequence_path):
            train_config.val_dataloader["dataset"]["data_prefix"] = dict(
                sequences=[sequence_path],
                keypoints_gt_paths=[kpts_gt_path],
                bboxes_gt_paths=[bboxes_gt_path],
                actions_gt_paths=[actions_gt_path],
            )

            reference_bboxes = OUTPUTS.build(
                {
                    "type": train_config.val_dataloader["dataset"]["bboxes_gt_format"],
                    "path": bboxes_gt_path,
                }
            )
            reference_bboxes.read()
            reference_actions = OUTPUTS.build(
                {
                    "type": train_config.val_dataloader["dataset"]["actions_gt_format"],
                    "path": kpts_gt_path,
                }
            )
            reference_actions.read()

            def run_asserts(ar_dataset, reference_bboxes, reference_actions):
                assert len(ar_dataset.data_list) > 0, "Dataset data_list should not be empty"
                for seq_idx, sequence in enumerate(ar_dataset.data_list):
                    assert len(sequence) > 0, f"Sequence {seq_idx} should not be empty"
                    for data_sample in sequence:
                        frame_id = data_sample.img_id
                        frame_reference_actions = np.array(reference_actions[frame_id])
                        frame_reference_bboxes = np.array(reference_bboxes[frame_id])

                        assert np.allclose(frame_reference_actions[:, :3].astype(int), frame_reference_bboxes[:, :3].astype(int))

                        gt_ids = frame_reference_bboxes[:, 2]
                        gt_bboxes = frame_reference_bboxes[:, 3:]
                        gt_actions = frame_reference_actions[:, 3]

                        ds_pti = data_sample.pred_track_instances
                        ds_gt = data_sample.gt_instance_labels

                        for ds_id, ds_bboxe, ds_action in zip(ds_pti.instances_id, ds_pti.bboxes, gt_bboxes, ds_gt.action_labels):
                            assert np.isin(ds_id, gt_ids)
                            ds_id_idx = np.where(ds_id == gt_ids)[0][0]
                            gt_id_bboxes = gt_bboxes[ds_id_idx]
                            assert np.allclose(
                                gt_id_bboxes, ds_bboxe, atol=50
                            ), f"The ground truth bboxe ({gt_id_bboxes.tolist()}) is significantly different than the data_sample bboxe ({ds_bboxe.tolist()})."
                            assert gt_actions[ds_id_idx].item() == ar_dataset.label_to_action_map[ds_action.item()]

                        assert data_sample.pred_track_instances.features.shape[1] == ar_dataset.n_feats
                        assert data_sample.pred_track_instances.dynamics.shape[1] == ar_dataset.n_velocities

                assert len(ar_dataset.action_to_sequence_map) > 0, "action_to_sequence_map should not be empty"

                for action, sequence_list in ar_dataset.action_to_sequence_map.items():
                    action_label = ar_dataset.label_to_action_map[action]
                    for seq in sequence_list:
                        seq_idx, _, frame_id, inst_id = seq
                        inst_id = int(inst_id)
                        frame_reference_actions = np.array(reference_actions[frame_id])
                        ref_frame_id = frame_reference_actions[:, 0].astype(int)[0]
                        assert frame_id == ref_frame_id
                        ref_ids = frame_reference_actions[:, 2].astype(int)
                        ref_action = frame_reference_actions[:, 3]
                        assert np.isin(inst_id, ref_ids)
                        inst_id_idx = np.where(inst_id == ref_ids)[0][0]
                        assert action_label == ref_action[inst_id_idx].item()

                        assert isinstance(seq_idx, (int, np.integer)), f"seq_idx should be int, got {type(seq_idx)}"
                        assert isinstance(frame_id, (int, np.integer)), f"frame_id should be int, got {type(frame_id)}"
                        assert frame_id >= ar_dataset.block_size, f"frame_id {frame_id} should be >= block_size {ar_dataset.block_size}"

                num_iterations = 200
                for iteration in range(num_iterations):
                    data = ar_dataset.prepare_data(iteration)

                    assert "inputs" in data, "prepare_data output missing 'inputs'"
                    assert "data_samples" in data, "prepare_data output missing 'data_samples'"

                    inputs = data["inputs"]
                    assert inputs.shape == (
                        ar_dataset.block_size,
                        ar_dataset.n_feats,
                    ), f"Iteration {iteration}: inputs shape {inputs.shape} != expected {(ar_dataset.block_size, ar_dataset.n_feats)}"

                    data_samples = data["data_samples"]
                    assert hasattr(data_samples, "pred_track_instances"), "data_samples missing pred_track_instances"
                    assert hasattr(data_samples, "gt_instance_labels"), "data_samples missing gt_instance_labels"
                    assert data_samples.pred_track_instances.kpts.shape == (
                        ar_dataset.block_size,
                        ar_dataset.n_kpts,
                        2,
                    ), f"Iteration {iteration}: kpts shape mismatch"
                    assert data_samples.pred_track_instances.kpt_vis.shape == (
                        ar_dataset.block_size,
                        ar_dataset.n_kpts,
                    ), f"Iteration {iteration}: kpt_vis shape mismatch"
                    assert data_samples.pred_track_instances.velocities.shape == (
                        ar_dataset.block_size,
                        ar_dataset.n_velocities,
                    ), f"Iteration {iteration}: velocities shape mismatch"
                    assert (
                        data_samples.gt_instance_labels.action_labels.shape[0] == ar_dataset.block_size
                    ), f"Iteration {iteration}: action_labels shape mismatch"

                    frame_id = int(data_samples.img_id)
                    inst_id = int(data_samples.instance_id)

                    for i in reversed(range(ar_dataset.block_size)):
                        block_frame_id = frame_id - i
                        frame_reference_actions = np.array(reference_actions[block_frame_id])
                        gt_ids = frame_reference_actions[:, 2].astype(int)
                        gt_actions = frame_reference_actions[:, 3]
                        prepared_action = data_samples.gt_instance_labels.action_labels[ar_dataset.block_size - i - 1].item()
                        if np.isin(inst_id, gt_ids):
                            if int(prepared_action) == int(ar_dataset._ignore_idx):
                                inst_id = int(inst_id)
                                assert inst_id in ar_dataset.missed_gt
                                assert block_frame_id in ar_dataset.missed_gt[inst_id]
                                assert not np.isin(inst_id, ar_dataset.data_list[data_samples.seq_id][block_frame_id].pred_track_instances.instances_id)
                            else:
                                inst_id_idx = np.where(inst_id == gt_ids)[0][0]
                                assert ar_dataset.label_to_action_map[prepared_action] == gt_actions[inst_id_idx].item()
                        else:
                            assert int(prepared_action) == int(ar_dataset._ignore_idx)

            # Reloading without augments
            ar_dataset = DATASETS.build(train_config.val_dataloader["dataset"])
            run_asserts(ar_dataset, reference_bboxes, reference_actions)
            ar_dataset.reset()
            run_asserts(ar_dataset, reference_bboxes, reference_actions)

            # Reloading with augments
            ar_dataset = DATASETS.build(train_config.val_dataloader["dataset"])
            train_config.val_dataloader["dataset"]["pipeline"] = train_config.augmented_pipeline
            run_asserts(ar_dataset, reference_bboxes, reference_actions)
            ar_dataset.reset()
            run_asserts(ar_dataset, reference_bboxes, reference_actions)

    finally:
        set_user_configs(False, user_configs, training_config, deploying_directory="../checkpoints/mice/", metadata_file="../configs/metadata/mice.py")


def test_testing(testing_config, user_configs):
    set_user_configs(True, user_configs, testing_config, deploying_directory=os.path.join(ROOT, "configs"))
    try:
        test_config = Config.fromfile(testing_config)

        ar_data_root = Path(test_config.action_recognition_data_root)
        ar_data_root = str(Path(*ar_data_root.parts[1:]))

        test_sequences = os.path.join(ar_data_root, test_config.test_sequences)
        test_bboxes_gt_paths = os.path.join(ar_data_root, test_config.test_bboxes_gt_paths)
        test_keypoints_gt_paths = os.path.join(ar_data_root, test_config.test_keypoints_gt_paths)
        test_actions_gt_paths = os.path.join(ar_data_root, test_config.test_actions_gt_paths)

        if (
            os.path.isdir(test_sequences)
            and os.path.isdir(test_bboxes_gt_paths)
            and os.path.isdir(test_keypoints_gt_paths)
            and os.path.isdir(test_actions_gt_paths)
            and torch.cuda.is_available()
        ):

            result = subprocess.run(
                ["python", "./test_action_recognition.py", f"--config={os.path.abspath(testing_config)}"],
                capture_output=True,
                text=True,
                cwd=TOOLS_DIR,
            )
            assert result.returncode == 0, f"Training failed with: {result.stderr}"

            with open(os.path.abspath(os.path.join(ROOT, test_config.work_dir, "best_ActionRecognition_f1.json")), "r") as f:
                obtained = json.load(f)

            with open(os.path.abspath(os.path.join(ROOT, "work_dir", "bare_minimum_ActionRecognition_f1.json")), "r") as f:
                bare_minimum = json.load(f)

            for metric in obtained:
                obtained_metric = obtained[metric]
                bare_minimum_metric = bare_minimum[metric]
                assert obtained_metric >= bare_minimum_metric, f"Metric: {metric}, Obtained: {obtained_metric:.4f}, Bare minimum: {bare_minimum_metric:.4f}."
    finally:
        set_user_configs(False, user_configs, testing_config, deploying_directory="../checkpoints/mice/")


if __name__ == "__main__":
    pytest.main(["-x", os.path.realpath(__file__), "-s"])
