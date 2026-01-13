import argparse
import logging
import os
from collections import defaultdict

import yaml
from mmengine.logging import MMLogger
from test_action_recognition import main as test_ar_main
from train_detection import deploy, get_device, load_config, parse_device_id, str2bool

from precision_track import AssociationStep, Runner
from precision_track.registry import TASK_UTILS

from precision_track.deploy.to_onnx import mart_to_onnx
from precision_track.deploy.to_tensorrt import to_tensorrt
from precision_track.models.backends import DetectionBackend
from precision_track.utils import load_user_configs


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


def main(args):
    logger = MMLogger.get_instance("mmengine", log_level=logging.INFO, file_mode="w")
    system_configs_path = args.config
    with open("../configs/user_configs.yaml", "r") as f:
        user_configs = yaml.safe_load(f)
    user_configs["booleans"]["with_action_recognition"] = True
    load_user_configs(user_configs, system_configs_path)

    # runner = Runner(system_configs_path, args.launcher, mode="train")
    # runner()

    deploy_cfg = load_config("../configs/tasks/deploying.py")
    deployed_path = deploy(deploy_cfg, "mart_runtime_config", deploy_cfg["mart_testing_checkpoint"], logger)
    tracking_config = load_config(deploy_cfg.tracking_cfg)
    tracking_config.load_from = deployed_path

    device = deploy_cfg["device"]
    if device == "auto":
        device = get_device()

    args.config = "../configs/tasks/testing_action_recognition.py"
    # if args.test:
    #     test_ar_main(args=args)

    if args.deploy:
        if deploy_cfg["mart_runtime_config"]["type"] in ["onnxruntime", "tensorrt"]:
            ir_config = deploy_cfg["mart_onnx_config"]
            ir_save_file = ir_config["save_file"]
            logger.info(f"Deploying {ir_save_file} to ONNX.")
            detector = DetectionBackend(**tracking_config.detector)
            assigner = AssociationStep(**tracking_config.assigner)
            mart_to_onnx(
                assigner(detector([deploy_cfg["img"]], [0])[0]),
                deploy_cfg["mart_runtime_config"]["paths"]["directory"],
                ir_save_file,
                deploy_cfg,
                deployed_path,
                device=device,
            )

        if deploy_cfg["mart_runtime_config"]["type"] == "tensorrt":
            logger.info(f"Optimizing {ir_save_file} to TensorRT.")

            common_params = deploy_cfg["mart_runtime_config"]["common_config"]

            input_shape_cfgs = deploy_cfg["analyzer"]["runtime"]["input_shapes"]
            input_shapes = []
            for input_shape in input_shape_cfgs:
                input_shapes.append(TASK_UTILS.build(input_shape))

            input_names = deploy_cfg["action_recognition_input_names"]

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
                formatted_input_shapes[k]["min_shape"] = [1] + list(input_shape.shape)
                formatted_input_shapes[k]["opt_shape"] = [num_subjects] + list(input_shape.shape)
                formatted_input_shapes[k]["max_shape"] = [max_subjects] + list(input_shape.shape)

            to_tensorrt(
                os.path.join(deploy_cfg["mart_runtime_config"]["paths"]["directory"], ir_save_file),
                input_shapes=[dict(input_shapes=formatted_input_shapes)],
                log_level=None,
                half_precision=common_params.get("half_precision", False),
                max_workspace_size=common_params.get("max_workspace_size", 0),
                device_id=parse_device_id(device),
            )


if __name__ == "__main__":
    main(parse_args())
