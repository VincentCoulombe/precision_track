import argparse
import logging
import os
from collections import defaultdict

from mmengine.logging import MMLogger
from test_action_recognition import main as test_ar_main
from train_detection import deploy, get_device, load_config, parse_device_id, str2bool

from precision_track import AssociationStep, Runner
from precision_track.deploy.to_onnx import mart_to_onnx
from precision_track.deploy.to_tensorrt import to_tensorrt
from precision_track.models.backends import DetectionBackend
from precision_track.registry import TASK_UTILS
from precision_track.utils import find_checkpoint_hook, load_user_configs, parse_pose_metainfo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str2bool, default=True, help="True to test the trained model, False otherwise")
    parser.add_argument("--deploy", type=str2bool, default=True, help="True to deploy the trained model, False otherwise")
    parser.add_argument("--config", type=str, default="../configs/tasks/training_action_recognition.py", help="Path to the training config")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


_ANALYZER_KEYS = {
    "mart_runtime_config": "analyzer",
    "gmart_runtime_config": "gmart_analyzer",
}


def deploy_to_onnx_trt(
    runtime_config: str,
    input_names: str,
    deploy_cfg: dict,
    logger: MMLogger,
    tracking_config: dict,
    deployed_path: str,
    device: str,
):
    onnx_config_key = runtime_config.replace("_runtime_config", "_onnx_config")
    analyzer_key = _ANALYZER_KEYS[runtime_config]

    if deploy_cfg[runtime_config]["type"] in ["onnxruntime", "tensorrt"]:
        ir_config = deploy_cfg[onnx_config_key]
        ir_save_file = ir_config["save_file"]
        logger.info(f"Deploying {ir_save_file} to ONNX.")
        detector = DetectionBackend(**tracking_config.detector)
        assigner = AssociationStep(**tracking_config.assigner)
        mart_to_onnx(
            assigner(detector([deploy_cfg["img"]], [0])[0]),
            deploy_cfg[runtime_config]["paths"]["directory"],
            ir_save_file,
            deploy_cfg,
            deployed_path,
            device=device,
            onnx_config_key=onnx_config_key,
            analyzer_key=analyzer_key,
        )

    if deploy_cfg[runtime_config]["type"] == "tensorrt":
        logger.info(f"Optimizing {ir_save_file} to TensorRT.")

        common_params = deploy_cfg[runtime_config]["common_config"]

        input_shape_cfgs = deploy_cfg[analyzer_key]["runtime"]["input_shapes"]
        input_shapes = []
        for input_shape in input_shape_cfgs:
            input_shapes.append(TASK_UTILS.build(input_shape))

        input_names = deploy_cfg[input_names]

        num_subjects = deploy_cfg.get("num_subjects")
        if isinstance(num_subjects, dict):
            num_subjects = max([int(v) for v in num_subjects.values()])
        elif isinstance(num_subjects, int):
            num_subjects = num_subjects
        else:
            num_subjects = 1
        max_subjects = 10 * num_subjects

        formatted_input_shapes = defaultdict(dict)
        for k, input_shape in zip(input_names, input_shapes):
            if getattr(input_shape, "is_pairwise", False):
                formatted_input_shapes[k]["min_shape"] = [1, 1] + list(input_shape.shape)
                formatted_input_shapes[k]["opt_shape"] = [num_subjects, num_subjects] + list(input_shape.shape)
                formatted_input_shapes[k]["max_shape"] = [max_subjects, max_subjects] + list(input_shape.shape)
            else:
                formatted_input_shapes[k]["min_shape"] = [1] + list(input_shape.shape)
                formatted_input_shapes[k]["opt_shape"] = [num_subjects] + list(input_shape.shape)
                formatted_input_shapes[k]["max_shape"] = [max_subjects] + list(input_shape.shape)

        to_tensorrt(
            os.path.join(deploy_cfg[runtime_config]["paths"]["directory"], ir_save_file),
            input_shapes=formatted_input_shapes,
            log_level=None,
            half_precision=common_params.get("half_precision", False),
            max_workspace_size=common_params.get("max_workspace_size", 0),
            device_id=parse_device_id(device),
        )


def main(args):
    logger = MMLogger.get_instance("mmengine", log_level=logging.INFO, file_mode="w")
    system_configs_path = args.config
    user_system_configs_path = "../configs/user_configs.yaml"
    load_user_configs(user_system_configs_path, system_configs_path, dynamic_ar_flag=True)

    runner = Runner(system_configs_path, args.launcher, mode="train")
    runner()
    checkpoint_hook = find_checkpoint_hook(runner)
    best_ckpt_path = str(checkpoint_hook.best_ckpt_path)

    assert os.path.isfile(best_ckpt_path), f"The current best training checkpoint ({best_ckpt_path}) does not exists. "
    "This is either because you deleted it manually or because the training run stopped before a validation step took place."

    deploy_cfg = load_config("../configs/tasks/deploying.py")
    deployed_path = deploy(deploy_cfg, "mart_runtime_config", best_ckpt_path, logger)
    tracking_config = load_config(deploy_cfg.tracking_cfg)
    tracking_config.load_from = deployed_path

    device = deploy_cfg["device"]
    if device == "auto":
        device = get_device()

    args.config = "../configs/tasks/testing_action_recognition.py"
    if args.test:
        test_ar_main(args=args)

    if args.deploy:
        deploy_to_onnx_trt(
            "mart_runtime_config",
            "action_recognition_input_names",
            deploy_cfg,
            logger,
            tracking_config,
            deployed_path,
            device,
        )

    if deploy_cfg.with_group_action_recognition:
        metainfo = deploy_cfg["metainfo"]
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        n_group_classes = len(metainfo.get("social_actions", [])) + 1  # Account for added null class
        assert n_group_classes, f"'with_group_action_recognition', but no social_actions are defined in the '{metainfo}' metadata file."
        system_configs_path = "../configs/tasks/training_group_action_recognition.py"
        load_user_configs(user_system_configs_path, system_configs_path, dynamic_ar_flag=True)
        runner = Runner(system_configs_path, args.launcher, mode="train")
        runner()
        checkpoint_hook = find_checkpoint_hook(runner)
        best_ckpt_path = str(checkpoint_hook.best_ckpt_path)
        assert os.path.isfile(best_ckpt_path), f"The current best training checkpoint ({best_ckpt_path}) does not exists. "
        "This is either because you deleted it manually or because the training run stopped before a validation step took place."
        deployed_path = deploy(deploy_cfg, "gmart_runtime_config", best_ckpt_path, logger)
        tracking_config = load_config(deploy_cfg.tracking_cfg)
        tracking_config.load_from = deployed_path

        args.config = "../configs/tasks/testing_group_action_recognition.py"
        if args.test:
            test_ar_main(args=args)

        if args.deploy:
            deploy_to_onnx_trt(
                "gmart_runtime_config",
                "gar_input_names",
                deploy_cfg,
                logger,
                tracking_config,
                deployed_path,
                device,
            )


if __name__ == "__main__":
    main(parse_args())
