# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import ctypes
import glob
import json
import logging
import os
import re
import sys
from enum import Enum
from typing import Any, List, Optional, Union

import onnx
import torch
from mmengine import Config
from mmengine.logging import MMLogger
from onnxconverter_common import float16

from precision_track.registry import CODEBASE

from .cuda import get_device, get_trt


def onnx_is_fp16(nn):
    for tensor in nn.graph.initializer:
        if tensor.data_type != onnx.TensorProto.FLOAT16:
            return False
    return True


def onnx_to_fp16(checkpoint, logger):
    assert os.path.isfile(checkpoint), f"The checkpoint path: {checkpoint}, does not lead to a valid file."
    nn = onnx.load(checkpoint)
    if not onnx_is_fp16(nn):
        if isinstance(logger, MMLogger):
            logger.info(f"Converting {checkpoint} to FP16...")
        nn = float16.convert_float_to_float16(nn)
        onnx.save(nn, checkpoint)


def deploy_weights(in_file, out_file):
    checkpoint = torch.load(in_file, map_location="cpu", weights_only=False)

    if "optimizer" in checkpoint:
        del checkpoint["optimizer"]
    if "message_hub" in checkpoint:
        del checkpoint["message_hub"]
    if "ema_state_dict" in checkpoint:
        del checkpoint["ema_state_dict"]

    for key in list(checkpoint["state_dict"]):
        if key.startswith("data_preprocessor"):
            checkpoint["state_dict"].pop(key)
        elif "priors_base_sizes" in key:
            checkpoint["state_dict"].pop(key)
        elif "grid_offset" in key:
            checkpoint["state_dict"].pop(key)
        elif "prior_inds" in key:
            checkpoint["state_dict"].pop(key)

    torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)


def load_calibration(file, calibration_metrics):
    ckpt_dir = os.path.dirname(file)
    hyperparameters_path = os.path.join(ckpt_dir, "hyperparameters.json")

    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    data["calibrated_temperature"] = calibration_metrics.get("calibrated_temperature", None)
    with open(hyperparameters_path, "w") as f:
        json.dump(data, f, indent=4)


def load_hyperparameter_dict(file, name: str, hyperparams: dict):
    assert isinstance(hyperparams, dict)
    assert isinstance(name, str)
    ckpt_dir = os.path.dirname(file)
    hyperparameters_path = os.path.join(ckpt_dir, "hyperparameters.json")

    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    data[name] = hyperparams
    with open(hyperparameters_path, "w") as f:
        json.dump(data, f, indent=4)
    return hyperparameters_path


def freeze_model_part(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


class AdvancedEnum(Enum):
    """Define an enumeration class."""

    @classmethod
    def get(cls, value):
        """Get the key through a value."""
        for k in cls:
            if k.value == value:
                return k

        raise KeyError(f'Cannot get key by value "{value}" of {cls}')


class Task(AdvancedEnum):
    """Define task enumerations."""

    PRECISION_TRACK_DETECTION = "PrecisionTrackDetection"
    MART_ACTION_RECOGNITION = "MARTActionRecognition"


class Codebase(AdvancedEnum):
    """Define codebase enumerations."""

    PRECISION_TRACK = "precision_track"


class Runtime(AdvancedEnum):
    """Define codebase enumerations."""

    TensorRTRuntime = "engine"
    ONNXRuntime = "onnx"
    PytorchRuntime = "pth"


def get_best_available_runtime(directory: str):
    priorities = [b.value for b in Runtime]
    found_ext = {}
    for file in os.listdir(directory):
        found_ext[os.path.splitext(file)[1][1:]] = os.path.join(directory, file)
    for priority in priorities:
        if priority in found_ext:
            return found_ext[priority]
    raise ValueError(f"The directory: {directory} contains none of the supported runtimes: {priorities}.")


def get_codebase_config(deploy_cfg: Union[str, Config]) -> dict:
    """Get the codebase_config from the config.

    Args:
        deploy_cfg (str | Config): The path or content of config.

    Returns:
        Dict : codebase config dict.
    """
    deploy_cfg = load_config(deploy_cfg)
    codebase_config = deploy_cfg.get("codebase_config", {})
    return codebase_config


def load_config(cfg) -> List[Config]:
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    if not isinstance(cfg, (dict, Config)):
        raise TypeError("deploy_cfg must be a filename or Config object, " f"but got {type(cfg)}")
    return cfg


def get_codebase(deploy_cfg: Union[str, Config]) -> Codebase:
    """Get the codebase from the config.

    Args:
        deploy_cfg (str | Config): The path or content of config.

    Returns:
        Codebase : An enumeration denotes the codebase type.
    """
    codebase_config = get_codebase_config(deploy_cfg)
    assert "type" in codebase_config, "The codebase config of deploy config" 'requires a "type" field'
    codebase = codebase_config["type"]
    return Codebase.get(codebase)


def get_runtime_config(deploy_cfg: Union[str, Config]) -> dict:
    """Get the runtime_config from the config.

    Args:
        deploy_cfg (str | Config): The path or content of config.

    Returns:
        Dict : runtime config dict.
    """
    deploy_cfg = load_config(deploy_cfg)
    runtime_config = deploy_cfg.get("runtime_config", {})
    return runtime_config


def parse_device_id(device: str) -> Optional[int]:
    """Parse device index from a string.

    Args:
        device (str): The typical style of string specifying device,
            e.g.: 'cuda:0', 'cpu'.

    Returns:
        Optional[int]: The return value depends on the type of device.
            If device is 'cuda': cuda device index, defaults to `0`.
            If device is 'cpu': `-1`.
            Otherwise, `None` will be returned.
    """
    if device == "auto":
        device = get_device()
    if device == "cpu":
        return -1
    if "cuda" in device:
        return parse_cuda_device_id(device)
    return None


def parse_cuda_device_id(device: str) -> int:
    """Parse cuda device index from a string.

    Args:
        device (str): The typical style of string specifying cuda device,
            e.g.: 'cuda:0'.

    Returns:
        int: The parsed device id, defaults to `0`.
    """
    match_result = re.match("([^:]+)(:[0-9]+)?$", device)
    assert match_result is not None, f"Can not parse device {device}."
    assert match_result.group(1).lower() == "cuda", "Not cuda device."

    device_id = 0 if match_result.lastindex == 1 else int(match_result.group(2)[1:])

    return device_id


def check_runtime_device(runtime: str, device: str):
    device_id = parse_device_id(device)
    mismatch = dict(
        tensorrt=lambda id: id == -1,
        openvino=lambda id: id > -1,
    )
    if runtime in mismatch and mismatch[runtime](device_id):
        raise ValueError(f"{device} is invalid for the runtime {runtime}")


def build_task_processor(deploy_cfg: Config, device: str):
    """Build a task processor to manage the deployment pipeline.

    Args:
        model_cfg (str | Config): Model config file.
        deploy_cfg (str | Config): Deployment config file.
        device (str): A string specifying device type.

    Returns:
        BaseTask: A task processor.
    """
    runtime = get_runtime_config(deploy_cfg).get("type")
    if runtime is None:
        raise ValueError("No defined runtime in the deployment configuration.")
    check_runtime_device(runtime, device=device)
    codebase_name = get_codebase(deploy_cfg).value
    codebase = CODEBASE.build({"type": codebase_name})
    return codebase.build_task_processor(deploy_cfg["model"], deploy_cfg, device)


def get_task_type(deploy_cfg: Union[str, Config]) -> Task:
    """Get the task type of the algorithm.

    Args:
        deploy_cfg (str | Config): The path or content of config.

    Returns:
        Task : An enumeration denotes the task type.
    """

    codebase_config = get_codebase_config(deploy_cfg)
    assert "task" in codebase_config, "The codebase config of deploy config" 'requires a "task" field'
    task = codebase_config["task"]
    return Task.get(task)


def save(engine: Any, path: str) -> None:
    """Serialize TensorRT engine to disk.

    Args:
        engine (Any): TensorRT engine to be serialized.
        path (str): The absolute disk path to write the engine.
    """
    trt = get_trt()
    with open(path, mode="wb") as f:
        if isinstance(engine, trt.ICudaEngine):
            engine = engine.serialize()
        f.write(bytearray(engine))


def load(path: str, allocator: Optional[Any] = None):
    """Deserialize TensorRT engine from disk.

    Args:
        path (str): The disk path to read the engine.
        allocator (Any): gpu allocator

    Returns:
        tensorrt.ICudaEngine: The TensorRT engine loaded from disk.
    """
    trt = get_trt()
    load_tensorrt_plugin()
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        if allocator is not None:
            runtime.gpu_allocator = allocator
        with open(path, mode="rb") as f:
            engine_bytes = f.read()
        trt.init_libnvinfer_plugins(logger, namespace="")
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine


def search_cuda_version() -> str:
    """try cmd to get cuda version, then try `torch.cuda`
    Returns:
        str: cuda version, for example 10.2
    """

    version = None

    pattern = re.compile(r"[0-9]+\.[0-9]+")
    platform = sys.platform.lower()

    def cmd_result(txt: str):
        cmd = os.popen(txt)
        return cmd.read().rstrip().lstrip()

    if platform == "linux" or platform == "darwin" or platform == "freebsd":  # noqa E501
        version = cmd_result(" nvcc --version | grep  release | awk '{print $5}' | awk -F , '{print $1}' ")  # noqa E501
        if version is None or pattern.match(version) is None:
            version = cmd_result(" nvidia-smi  | grep CUDA | awk '{print $9}' ")

    elif platform == "win32" or platform == "cygwin":
        # nvcc_release = "Cuda compilation tools, release 10.2, V10.2.89"
        nvcc_release = cmd_result(' nvcc --version | find "release" ')
        if nvcc_release is not None:
            result = pattern.findall(nvcc_release)
            if len(result) > 0:
                version = result[0]

        if version is None or pattern.match(version) is None:
            # nvidia_smi = "| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |" # noqa E501
            nvidia_smi = cmd_result(' nvidia-smi | find "CUDA Version" ')
            result = pattern.findall(nvidia_smi)
            if len(result) > 2:
                version = result[2]

    if version is None or pattern.match(version) is None:
        try:
            import torch

            version = torch.version.cuda
        except Exception:
            pass

    return version


def get_file_path(prefix, candidates) -> str:
    """Search for file in candidates.

    Args:
        prefix (str): Prefix of the paths.
        candidates (str): Candidate paths
    Returns:
        str: file path or '' if not found
    """
    for candidate in candidates:
        wildcard = os.path.abspath(os.path.join(prefix, candidate))
        paths = glob.glob(wildcard)
        if paths:
            lib_path = paths[0]
            return lib_path
    return ""


def get_trt_log_level():
    """Get tensorrt log level from root logger.

    Returns:
        level (tensorrt.Logger.Severity):
        Logging level of tensorrt.Logger.
    """
    trt = get_trt()
    logger = MMLogger.get_instance("mmengine", log_level=logging.INFO, file_mode="w")
    level = logger.level
    trt_log_level = trt.Logger.INFO
    if level == logging.ERROR:
        trt_log_level = trt.Logger.ERROR
    elif level == logging.WARNING:
        trt_log_level = trt.Logger.WARNING
    elif level == logging.DEBUG:
        trt_log_level = trt.Logger.VERBOSE
    return trt_log_level


def get_ops_path() -> str:
    """Get path of the TensorRT plugin library.

    Returns:
        str: A path of the TensorRT plugin library.
    """
    return ""


def load_tensorrt_plugin() -> bool:
    """Load TensorRT plugins library.

    Returns:
        bool: True if TensorRT plugin library is successfully loaded.
    """
    # TODO depreciated. No longer using custom plugins.
    lib_path = get_ops_path()
    success = False
    logger = MMLogger.get_instance("mmengine", log_level=logging.INFO, file_mode="w")
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        logger.info(f"Successfully loaded tensorrt plugins from {lib_path}")
        success = True
    return success


def get_ir_config(deploy_cfg: Union[str, Config]) -> dict:
    """Get the IR parameters in export() from config.

    Args:
        deploy_cfg (str | Config): The path or content of config.

    Returns:
        Dict: The config dictionary of IR parameters
    """

    deploy_cfg = load_config(deploy_cfg)
    ir_config = deploy_cfg.get("ir_config", None)
    if ir_config is None:
        # TODO: deprecate in future
        ir_config = deploy_cfg.get("onnx_config", {})
    return ir_config


def get_dynamic_axes(deploy_cfg: Union[str, Config], axes_names: List[str] = None) -> dict:
    """Get model dynamic axes from config.

    Args:
        deploy_cfg (str | Config): The path or content of config.
        axes_names (List[str]): List with names for dynamic axes.

    Returns:
        Dict[str, Union[List[int], Dict[int, str]]]:
            Dictionary with dynamic axes.
    """
    deploy_cfg = load_config(deploy_cfg)
    ir_config = get_ir_config(deploy_cfg)

    # TODO onnx will be deprecated in the future
    onnx_config = deploy_cfg.get("onnx_config", None)
    if onnx_config is None and ir_config == {}:
        raise KeyError("Field 'onnx_config' was not found in 'deploy_cfg'.")
    dynamic_axes = ir_config.get("dynamic_axes", None)
    if dynamic_axes and not isinstance(dynamic_axes, dict):
        if axes_names is None:
            axes_names = []
            input_names = ir_config.get("input_names", None)
            if input_names:
                axes_names += input_names
            output_names = ir_config.get("output_names", None)
            if output_names:
                axes_names += output_names
            if not axes_names:
                raise KeyError("No names were found to define dynamic axes.")
        dynamic_axes = dict(zip(axes_names, dynamic_axes))
    return dynamic_axes


def get_common_config(deploy_cfg: Union[str, Config]) -> dict:
    """Get common parameters from config.

    Args:
        deploy_cfg (str | Config): The path or content of config.

    Returns:
        dict: A dict of common parameters for a model.
    """
    runtime_config = deploy_cfg["runtime_config"]
    model_params = runtime_config.get("common_config", dict())
    return model_params


def get_model_inputs(deploy_cfg: Union[str, Config]) -> List[dict]:
    """Get model input parameters from config.

    Args:
        deploy_cfg (str | Config): The path or content of config.

    Returns:
        list[dict]: A list of dict containing input parameters for a model.
    """
    runtime_config = deploy_cfg["runtime_config"]
    model_params = runtime_config.get("model_inputs", [])
    return model_params


PYTHON_CKPT_EXTS = [".pt", ".pth"]
ONNX_CKPT_EXTS = [".onnx"]
TRT_CKPT_EXTS = [".engine"]


def get_runtime_type(checkpoint: Optional[str] = None):
    if checkpoint is None:
        return "PytorchRuntime"
    elif isinstance(checkpoint, str) and os.path.exists(checkpoint) and os.path.isfile(checkpoint):
        _, ext = os.path.splitext(checkpoint)
        if ext in PYTHON_CKPT_EXTS:
            return "PytorchRuntime"
        elif ext in ONNX_CKPT_EXTS:
            return "ONNXRuntime"
        elif ext in TRT_CKPT_EXTS:
            return "TensorRTRuntime"
        else:
            raise ValueError(
                (
                    f"The runtime's checkpoint's extension ({ext}) is not supported."
                    "The supported checkpoint extensions are {PYTHON_CKPT_EXTS+ONNX_CKPT_EXTS+TRT_CKPT_EXTS}."
                )
            )
    else:
        raise ValueError(f"The runtime's checkpoint: {os.path.abspath(checkpoint)} is not a valid checkpoint path.")
