import logging
import os
from typing import Dict, Sequence

import torch
from mmengine.logging import MMLogger

from precision_track.utils import get_trt, get_trt_log_level, load_tensorrt_plugin, trt_available


def to_tensorrt(
    onnx_path: str,
    input_shapes: Dict[str, Sequence[int]],
    max_workspace_size: int = 1 << 30,
    half_precision: bool = False,
    device_id: int = 0,
    log_level=None,
    **kwargs,
):
    """Create a tensorrt engine from ONNX.

    Args:
        onnx_path (str): Input onnx model to convert from.
        input_shapes (Dict[str, Sequence[int]]): The min/opt/max shape of
            each input.
        max_workspace_size (int): To set max workspace size of TensorRT engine.
            some tactics and layers need large workspace. Defaults to `0`.
        half_precision (bool): Specifying whether to enable fp16 mode.
            Defaults to `False`.
        device_id (int): Choice the device to create engine. Defaults to `0`.
        log_level (trt.Logger.Severity): The log level of TensorRT. Defaults to
            `trt.Logger.ERROR`.
    """
    logger = MMLogger.get_instance("mmengine", log_level=logging.INFO, file_mode="w")
    if not trt_available():
        logger.warning("Skipped TensorRT deployment since the TensorRT module is not available in the execution environment.")
        return

    trt = get_trt()
    if log_level is None:
        log_level = get_trt_log_level()
    torch.cuda.set_device(device_id)

    load_tensorrt_plugin()

    # build a tensorrt logger
    trt_logger = trt.Logger(log_level)

    # create builder and network
    builder = trt.Builder(trt_logger)

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    assert os.path.exists(onnx_path), f"{onnx_path} does not exist."
    parser = trt.OnnxParser(network, trt_logger)
    parse_valid = parser.parse_from_file(onnx_path)

    if not parse_valid:
        error_msgs = ""
        for error in range(parser.num_errors):
            error_msgs += f"{parser.get_error(error)}\n"
        raise RuntimeError(f"Failed to parse onnx, {error_msgs}")

    config = builder.create_builder_config()
    config.builder_optimization_level = 5
    config.avg_timing_iterations = 8
    config.max_aux_streams = 7

    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    else:
        config.max_workspace_size = max_workspace_size

    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param["min_shape"]
        opt_shape = param["opt_shape"]
        max_shape = param["max_shape"]
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    if config.add_optimization_profile(profile) < 0:
        logger.warning(f"Invalid optimization profile {profile}.")

    precision = "FP32"
    if half_precision:
        if not getattr(builder, "platform_has_fast_fp16", True):
            logger.warning("Platform does not has fast native fp16.")
        config.set_flag(trt.BuilderFlag.FP16)
        precision = "FP16"
    logger.info(f"The TensorRT engine's precision will be of: {precision}.")

    # create engine
    if hasattr(builder, "build_serialized_network"):
        engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)

    assert engine is not None, "Failed to create TensorRT engine"

    # save engine
    gpu_name = torch.cuda.get_device_name(device_id).replace(" ", "")
    file_path, _ = os.path.splitext(onnx_path)
    output_path = f"{file_path}_{gpu_name}_{precision}.engine"
    logger.info(f"Saving {output_path}")

    with open(output_path, mode="wb") as f:
        if isinstance(engine, trt.ICudaEngine):
            engine = engine.serialize()
        f.write(bytearray(engine))
