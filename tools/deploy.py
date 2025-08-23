import argparse
import logging
import os

import mmengine
from mmengine import Config
from mmengine.logging import MMLogger

from precision_track import AssociationStep, Runner
from precision_track.deploy.to_onnx import mart_to_onnx, to_onnx
from precision_track.deploy.to_tensorrt import to_tensorrt
from precision_track.models.backends import DetectionBackend
from precision_track.models.optimization.thresholds_search import StitchingHyperparamsGridSearch, ThresholdsGridSearch
from precision_track.utils import (
    deploy_weights,
    get_common_config,
    get_device,
    get_ir_config,
    get_model_inputs,
    load_calibration,
    load_config,
    load_hyperparameter_dict,
    parse_device_id,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to backends.")
    parser.add_argument("optimize_hyperparams", type=str2bool, nargs="?", default=True, help="True to optimize the hyperparameters, False otherwise")
    args = parser.parse_args()
    return args


def deploy(deploy_cfg: Config, runtime_cfg_key: str, ckpt_path: str, logger: MMLogger):
    paths = deploy_cfg[runtime_cfg_key]["paths"]
    deployed_name = paths["deployed_name"]
    work_dir = paths["directory"]
    mmengine.mkdir_or_exist(os.path.abspath(work_dir))
    deployed_path = os.path.join(work_dir, deployed_name)

    logger.info(f"Deploying {ckpt_path} weights.")
    deploy_weights(ckpt_path, deployed_path)
    return deployed_path


def main(args):
    logger = MMLogger.get_instance("mmengine", log_level=logging.INFO, file_mode="w")

    deploy_cfg = load_config("../configs/tasks/deploying.py")
    deployed_path = deploy(deploy_cfg, "runtime_config", deploy_cfg["testing_checkpoint"], logger)
    deploy_cfg["model"]["checkpoint"] = deployed_path

    device = deploy_cfg["device"]
    if device == "auto":
        device = get_device()

    half_precision = deploy_cfg.get("half_precision", False)
    if half_precision and device == "cpu":
        logger.warning("Will not perform the half-precision (FP16) conversion on cpu. Reverting back to FP32.")
        deploy_cfg["half_precision"] = False
        deploy_cfg["runtime_config"]["common_config"]["half_precision"] = False
    precision = "FP16" if half_precision else "FP32"
    logger.info(f"Deploying on device: {device} with precision: {precision}.")

    if args.optimize_hyperparams:
        runner = Runner(deploy_cfg, "none", mode="calibrate")
        logger.info(f"Calibrating the network on {deploy_cfg.test_dataloader.dataset.ann_file}.")
        metrics = runner()
        load_calibration(deployed_path, metrics)

    if deploy_cfg["runtime_config"]["type"] in ["onnxruntime", "tensorrt"]:
        ir_config = get_ir_config(deploy_cfg)
        ir_save_file = ir_config["save_file"]
        logger.info(f"Deploying {ir_save_file} to ONNX.")
        to_onnx(
            deploy_cfg["img"],
            deploy_cfg["runtime_config"]["paths"]["directory"],
            ir_save_file,
            deploy_cfg,
            deployed_path,
            device=device,
        )

    if deploy_cfg["runtime_config"]["type"] == "tensorrt":
        logger.info(f"Optimizing {ir_save_file} to TensorRT.")

        common_params = get_common_config(deploy_cfg)
        model_params = get_model_inputs(deploy_cfg)[0]

        final_params = common_params
        final_params.update(model_params)

        to_tensorrt(
            os.path.join(deploy_cfg["runtime_config"]["paths"]["directory"], ir_save_file),
            input_shapes=final_params["input_shapes"],
            log_level=None,
            half_precision=final_params.get("half_precision", False),
            max_workspace_size=final_params.get("max_workspace_size", 0),
            device_id=parse_device_id(device),
        )

    tracking_config = load_config(deploy_cfg.tracking_cfg)
    tracking_config.load_from = deployed_path

    if args.optimize_hyperparams:
        logger.info(f"Searching for optimal tracking thresholds...")
        search_results = ThresholdsGridSearch(
            tracking_config=tracking_config,
            video_paths=deploy_cfg.videos,
            gt_paths=deploy_cfg.gt_paths,
            metadata_path=deploy_cfg.metainfo,
            output_path=os.path.join(deploy_cfg["runtime_config"]["paths"]["directory"], "tracking_predictions.csv"),
            save_path=os.path.join(deploy_cfg["runtime_config"]["paths"]["directory"], "thresholds_grid_search_results.csv"),
            low_thr_range=deploy_cfg.get("low_thr_range"),
            high_thr_range=deploy_cfg.get("high_thr_range"),
            init_thr_range=deploy_cfg.get("init_thr_range"),
        )()
        load_hyperparameter_dict(
            deployed_path,
            "tracking_thresholds",
            dict(
                low_thr=search_results.loc[0, "low_thr"],
                conf_thr=search_results.loc[0, "high_thr"],
                init_thr=search_results.loc[0, "init_thr"],
            ),
        )
        if "stitching_algorithm" in tracking_config:
            search_results = StitchingHyperparamsGridSearch(
                tracking_config=tracking_config,
                video_paths=deploy_cfg.videos,
                gt_paths=deploy_cfg.gt_paths,
                metadata_path=deploy_cfg.metainfo,
                bboxes_path=os.path.join(deploy_cfg["runtime_config"]["paths"]["directory"], "tracking_predictions.csv"),
                search_zones_path=os.path.join(deploy_cfg["runtime_config"]["paths"]["directory"], "search_zones.csv"),
                save_path=os.path.join(deploy_cfg["runtime_config"]["paths"]["directory"], "stitching_hyperparameter_search.csv"),
                match_thr_range=deploy_cfg.get("match_thr_range"),
                beta_range=deploy_cfg.get("beta_range"),
                eps_range=deploy_cfg.get("eps_range"),
            )()
            load_hyperparameter_dict(
                deployed_path,
                "stitching_hyperparams",
                dict(
                    beta=search_results.loc[0, "beta"],
                    match_thr=search_results.loc[0, "match_thr"],
                    eps=search_results.loc[0, "eps"],
                ),
            )

    if deploy_cfg["with_action_recognition"] and deploy_cfg.get("analyzer", None) is not None:  # MART
        deployed_path = deploy(deploy_cfg, "mart_runtime_config", deploy_cfg["mart_testing_checkpoint"], logger)
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
