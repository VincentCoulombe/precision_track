import os.path as osp
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine.device import get_device
from mmengine.logging import MMLogger

from precision_track.utils import CODEBASE, build_task_processor, check_runtime_device, get_codebase, get_dynamic_axes, get_ir_config, onnx_to_fp16

from .rewriters import RewriterContext


def to_onnx(
    img: Union[str, np.ndarray, torch.Tensor],
    work_dir: str,
    save_file: str,
    deploy_cfg: mmengine.Config,
    model_checkpoint: Optional[str] = None,
    device: str = "auto",
):
    if device == "auto":
        device = get_device()

    deploy_cfg = deepcopy(deploy_cfg)
    deploy_cfg["runtime_config"]["type"] = "onnx"
    task_processor = build_task_processor(deploy_cfg, device)

    torch_model = task_processor.build_pytorch_model(model_checkpoint)
    data = task_processor.create_input([img])
    model_inputs = data["inputs"]
    data_samples = data["data_samples"]
    input_metas = {"data_samples": data_samples, "mode": "predict"}

    # export to onnx
    context_info = dict()
    context_info["deploy_cfg"] = deploy_cfg
    output_prefix = osp.join(work_dir, osp.splitext(osp.basename(save_file))[0])

    onnx_cfg = get_ir_config(deploy_cfg)
    opset_version = onnx_cfg.get("opset_version", 11)

    input_names = onnx_cfg["input_names"]
    output_names = onnx_cfg["output_names"]
    axis_names = input_names + output_names
    dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)
    verbose = not onnx_cfg.get("strip_doc_string", True) or onnx_cfg.get("verbose", False)
    keep_initializers_as_inputs = onnx_cfg.get("keep_initializers_as_inputs", True)

    export(
        torch_model,
        model_inputs,
        input_metas=input_metas,
        output_path_prefix=output_prefix,
        input_names=input_names,
        output_names=output_names,
        context_info=context_info,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
    )
    checkpoint = output_prefix + ".onnx"
    half_precision = deploy_cfg.get("half_precision", False)
    if half_precision:
        onnx_to_fp16(checkpoint, MMLogger.get_current_instance())


def mart_to_onnx(
    tracking_output: dict,
    work_dir: str,
    save_file: str,
    deploy_cfg: mmengine.Config,
    model_checkpoint: Optional[str] = None,
    device: str = "auto",
):
    if device == "auto":
        device = get_device()

    check_runtime_device("onnx", device=device)
    codebase_name = get_codebase(deploy_cfg).value
    codebase = CODEBASE.build({"type": codebase_name})
    mart_cfg = deploy_cfg["analyzer"]["runtime"]["model"]
    mart_cfg["data_preprocessor"] = deploy_cfg["analyzer"]["data_preprocessor"]
    task_processor = codebase.build_task_processor(mart_cfg, deploy_cfg, device, is_mart=True)

    torch_model = task_processor.build_pytorch_model(model_checkpoint)
    data = task_processor.create_input(tracking_output)
    model_inputs = (data["features"], data["poses"], data["dynamics"])
    data_samples = data["data_samples"]
    input_metas = {"data_samples": data_samples, "mode": "predict"}

    # export to onnx
    context_info = dict()
    context_info["deploy_cfg"] = deploy_cfg
    output_prefix = osp.join(work_dir, osp.splitext(osp.basename(save_file))[0])

    onnx_cfg = deploy_cfg["mart_onnx_config"]
    opset_version = onnx_cfg.get("opset_version", 11)

    input_names = onnx_cfg["input_names"]
    output_names = onnx_cfg["output_names"]
    dynamic_axes = deploy_cfg["mart_dynamic_axes"]
    verbose = not onnx_cfg.get("strip_doc_string", True) or onnx_cfg.get("verbose", False)
    keep_initializers_as_inputs = onnx_cfg.get("keep_initializers_as_inputs", True)

    export(
        torch_model,
        model_inputs,
        input_metas=input_metas,
        output_path_prefix=output_prefix,
        input_names=input_names,
        output_names=output_names,
        context_info=context_info,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
    )
    checkpoint = output_prefix + ".onnx"
    half_precision = deploy_cfg.get("half_precision", False)
    if half_precision:
        onnx_to_fp16(checkpoint, MMLogger.get_current_instance())


def export(
    model: torch.nn.Module,
    args: Union[torch.Tensor, Tuple, Dict],
    output_path_prefix: str,
    input_metas: Optional[Dict] = None,
    context_info: Dict = dict(),
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    opset_version: int = 11,
    dynamic_axes: Optional[Dict] = None,
    verbose: bool = False,
    keep_initializers_as_inputs: Optional[bool] = None,
):
    output_path = output_path_prefix + ".onnx"
    runtime = "onnxruntime"

    def _add_or_update(cfg: dict, key: str, val: Any):
        if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
            cfg[key].update(val)
        else:
            cfg[key] = val

    context_info = deepcopy(context_info)
    deploy_cfg = context_info.pop("deploy_cfg", dict())
    ir_config = dict(
        type="onnx",
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
    )
    _add_or_update(deploy_cfg, "ir_config", ir_config)
    runtime_config = dict(type=runtime)
    _add_or_update(deploy_cfg, "runtime_config", runtime_config)

    context_info["cfg"] = deploy_cfg
    context_info["ir"] = "onnx"
    if "runtime" not in context_info:
        context_info["runtime"] = runtime
    if "opset" not in context_info:
        context_info["opset"] = opset_version
    patched_model = model
    with RewriterContext(**context_info), torch.no_grad():
        # patch input_metas
        if input_metas is not None:
            assert isinstance(input_metas, dict), f"Expect input_metas type is dict, get {type(input_metas)}."
            model_forward = patched_model.forward

            def wrap_forward(forward):

                def wrapper(*arg, **kwargs):
                    return forward(*arg, **kwargs)

                return wrapper

            patched_model.forward = wrap_forward(patched_model.forward)
            # Wrap le forward pour toujours y passer **input_metas comme si cétait des constantes
            patched_model.forward = partial(patched_model.forward, **input_metas)
        # force to export on cpu
        patched_model = patched_model.cpu()
        if isinstance(args, torch.Tensor):
            args = args.cpu()
        elif isinstance(args, (tuple, list)):
            args = tuple([_.cpu() for _ in args])
        else:
            raise RuntimeError(f"Not supported args: {args}")

        torch.onnx.export(
            patched_model,
            args,
            output_path,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            verbose=verbose,
            training=torch.onnx.TrainingMode.EVAL,
        )

        if input_metas is not None:
            patched_model.forward = model_forward
