import argparse
import logging
import os
import shutil
import yaml
import mmengine
from mmengine import Config
from mmengine.logging import MMLogger

from precision_track import Runner
from precision_track.deploy.to_onnx import to_onnx
from precision_track.deploy.to_tensorrt import to_tensorrt
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
    load_user_configs,
    check_if_mot_dataset_is_ok,
    assert_coco_dataset_directory,
    resize_coco_dataset,
)

from test_detection import main as test_detection_main


if "DYNAMO_CACHE_SIZE_LIMIT" in os.environ:
    import torch._dynamo

    cache_size_limit = int(os.environ["DYNAMO_CACHE_SIZE_LIMIT"])
    torch._dynamo.config.cache_size_limit = cache_size_limit


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str2bool, default=True, help="True to test the trained model, False otherwise")
    parser.add_argument("--format_dataset", type=str2bool, default=True, help="True to format the training dataset, False otherwise")
    parser.add_argument("--calibrate", type=str2bool, default=True, help="True to calibrate the trained model, False otherwise")
    parser.add_argument("--deploy", type=str2bool, default=True, help="True to deploy the trained model, False otherwise")
    parser.add_argument("--optimize_hyperparams", type=str2bool, default=True, help="True to optimize the hyperparameters, False otherwise")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
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
    system_configs_path = "../configs/tasks/training_detection.py"
    with open("../configs/user_configs.yaml", "r") as f:
        user_configs = yaml.safe_load(f)
    load_user_configs(user_configs, system_configs_path)

    training_config = load_config(system_configs_path)
    data_root = training_config["data_root"]
    input_size = training_config["input_size"]
    assert_coco_dataset_directory(data_root)

    if args.format_dataset:
        normed_data_root = os.path.normpath(data_root)
        data_root_c = os.path.basename(normed_data_root)
        data_root_c = f"{data_root_c}_{input_size[0]}x{input_size[1]}/"
        data_root_p = os.path.dirname(normed_data_root)
        formatted_dataset_data_root = os.path.join(data_root_p, data_root_c)
        formatted_dataset_cfg = dict(training=dict(data_root=formatted_dataset_data_root))

        logger.info(f"Auto-formatting your COCO-style dataset saved at: {data_root}. The formatted dataset will be saved at: {formatted_dataset_data_root}.")
        if os.path.isdir(formatted_dataset_data_root):
            shutil.rmtree(formatted_dataset_data_root)
        for ann_name in ["train", "val"]:
            resize_coco_dataset(data_root, formatted_dataset_data_root, ann_name=f"{ann_name}.json")
        load_user_configs(formatted_dataset_cfg, system_configs_path)

    runner = Runner(system_configs_path, args.launcher, mode="train")
    runner()

    deploy_cfg = load_config("../configs/tasks/deploying.py")
    deployed_path = deploy(deploy_cfg, "runtime_config", os.path.join(training_config.work_dir, f"epoch_{training_config.num_epochs}.pth"), logger)
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

    if args.test:
        test_detection_main(args=args)

    if args.calibrate:
        runner = Runner(deploy_cfg, "none", mode="calibrate")
        logger.info(f"Calibrating the network on {deploy_cfg.test_dataloader.dataset.ann_file}.")
        metrics = runner()
        load_calibration(deployed_path, metrics)

    if args.deploy:
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
        load_user_configs(dict(training=dict(data_root=data_root)), system_configs_path)
        deploy_cfg = load_config("../configs/tasks/deploying.py")
        testing_tracking_data_root = deploy_cfg.testing_tracking_data_root
        mot_dataset_ok, feedback = check_if_mot_dataset_is_ok(testing_tracking_data_root)

        if mot_dataset_ok:

            video_paths = os.path.join(os.path.normpath(testing_tracking_data_root), "videos")
            gt_paths = os.path.join(os.path.normpath(testing_tracking_data_root), "bboxes")

            logger.info(f"Searching for optimal tracking hyperparameters...")
            search_results = ThresholdsGridSearch(
                tracking_config=tracking_config,
                video_paths=video_paths,
                gt_paths=gt_paths,
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
                    video_paths=video_paths,
                    gt_paths=gt_paths,
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
        else:
            logger.warning(
                f"To be able to search for the optimal hyperparameters, you will need a correctly formatted MOT dataset. Since: {feedback}, your MOT dataset is not correctly formatted."
            )


if __name__ == "__main__":
    main(parse_args())
