import yaml
from collections import defaultdict
import os
import logging
from mmengine.logging import MMLogger
import argparse

from precision_track import Runner, AssociationStep
from precision_track.utils import load_user_configs
from precision_track.deploy.to_onnx import mart_to_onnx
from precision_track.deploy.to_tensorrt import to_tensorrt
from precision_track.models.backends import DetectionBackend

from train_detection import str2bool, deploy, load_config, parse_device_id, get_device
from test_action_recognition import main as test_ar_main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str2bool, default=True, help="True to test the trained model, False otherwise")
    parser.add_argument("--deploy", type=str2bool, default=True, help="True to deploy the trained model, False otherwise")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main(args):
    logger = MMLogger.get_instance("mmengine", log_level=logging.INFO, file_mode="w")
    system_configs_path = "../configs/tasks/training_action_recognition.py"
    with open("../configs/user_configs.yaml", "r") as f:
        user_configs = yaml.safe_load(f)
    load_user_configs(user_configs, system_configs_path)
    runner = Runner(system_configs_path, args.launcher, mode="train")
    runner()

    deploy_cfg = load_config("../configs/tasks/deploying.py")
    deployed_path = deploy(deploy_cfg, "mart_runtime_config", deploy_cfg["mart_testing_checkpoint"], logger)
    tracking_config = load_config(deploy_cfg.tracking_cfg)
    tracking_config.load_from = deployed_path

    device = deploy_cfg["device"]
    if device == "auto":
        device = get_device()

    if args.test:
        test_ar_main(args=args)

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
            model_params = deploy_cfg["mart_runtime_config"]["model_inputs"]

            input_shapes = deploy_cfg["analyzer"]["runtime"]["input_shapes"]
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

            final_params = common_params
            final_params.update(model_params)

            to_tensorrt(
                os.path.join(deploy_cfg["mart_runtime_config"]["paths"]["directory"], ir_save_file),
                input_shapes=final_params["input_shapes"],
                log_level=None,
                half_precision=final_params.get("half_precision", False),
                max_workspace_size=final_params.get("max_workspace_size", 0),
                device_id=parse_device_id(device),
            )


if __name__ == "__main__":
    main(parse_args())
