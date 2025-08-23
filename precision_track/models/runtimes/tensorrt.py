import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine.logging import print_log
from packaging import version

from precision_track.registry import RUNTIMES
from precision_track.utils import PoseDataSample, get_pycuda, get_trt, load_tensorrt_plugin

from .base import InferenceOnlyRuntime


@RUNTIMES.register_module()
class TensorRTRuntime(InferenceOnlyRuntime):
    def __init__(
        self,
        output_names: Union[str, List[str]],
        checkpoint: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_id: Optional[int] = 0,
        half_precision: Optional[bool] = True,
        verbose: Optional[bool] = True,
        **kwargs,
    ) -> None:
        trt = get_trt()  # lazy import, raises with clear msg if missing
        self.trt = trt
        self._TORCH_DTYPE = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        self.cuda_drv = get_pycuda()
        if isinstance(output_names, str):
            output_names = [output_names]
        self.output_names_cfg = output_names
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.device_id = device_id
        super(TensorRTRuntime, self).__init__(
            checkpoint=checkpoint,
            device=device,
            half_precision=half_precision,
            verbose=verbose,
            **kwargs,
        )
        self._assert_runtime()

        self._buffer = dict()

    def _assert_runtime(self):
        """Load the TensorRT engine."""
        assert "cuda" in self.device, f"TensorRT runtime can only be leveraged on cuda, not {self.device}."
        load_tensorrt_plugin()
        with self.trt.Runtime(self.trt_logger) as runtime:
            with open(self.checkpoint, "rb") as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                raise RuntimeError("Failed to deserialize the engine.")
            self.engine = engine
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                "Failed to build the engine's context. the most probable reason is that the system's do not have enough VRAM to run the model in inference."
            )

        self.main_stream = torch.cuda.Stream(device=self.device)
        self.default_stream = torch.cuda.default_stream(device=self.device)
        self.aux_streams = [torch.cuda.Stream(device=self.device) for _ in range(self.engine.num_aux_streams)]
        if self.aux_streams:
            self.context.set_aux_streams([s.cuda_stream for s in self.aux_streams])

        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.NONE:
                print_log(
                    f"The Tensor {name} from the {self.checkpoint} TensorRT engine is neither an input nor an output.",
                    logger="current",
                    level=logging.WARNING,
                )
            else:
                self.output_names.append(name)

        assert set(self.output_names) == set(self.output_names_cfg), f"{set(self.output_names)} != {set(self.output_names_cfg)}"

        precision = "FP16" if self.nn_is_fp16() else "FP32"
        if self.half_precision and precision != "FP16":
            print_log(
                f"Can not infer with half precision as the {self.checkpoint} TensoRRT engine is compile with {precision} precision.",
                logger="current",
                level=logging.WARNING,
            )
            self.half_precision = False
            self._initialize_type()

        self.input_profiles = {n: self.engine.get_tensor_profile_shape(n, 0) for n in self.input_names}
        self.dtypes = {n: self._TORCH_DTYPE[self.engine.get_tensor_dtype(n)] for n in self.engine}
        self._running_batch_size = -1
        self._outputs = tuple()

        self.log_runtime(f"Inference backend set to: TensorRT: {version.parse(self.trt.__version__)}")

    def _stage_to_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_cuda:
            self.gpu_tensor = tensor
        else:
            np_dtype = np.dtype(str(tensor.numpy().dtype))
            h_pinned = self.cuda_drv.pagelocked_empty(tensor.numel(), np_dtype)
            np.copyto(h_pinned, tensor.flatten().numpy())
            self.gpu_tensor = torch.empty_like(tensor, device=self.device)
            self.main_stream.cuda_stream.synchronize()
            self.cuda_drv.memcpy_htod_async(int(self.gpu_tensor.data_ptr()), h_pinned, self.main_stream.cuda_stream)
        return self.gpu_tensor.contiguous()

    def _set_input(self, name: str, tensor: torch.Tensor) -> None:
        prof_min, _, prof_max = self.input_profiles[name]
        assert prof_min[0] <= tensor.size(0) <= prof_max[0], f"Batch {tensor.size(0)} outside [{prof_min[0]}, {prof_max[0]}] for {name}"

        self.context.set_input_shape(name, tuple(tensor.shape))
        self.context.set_tensor_address(name, int(tensor.data_ptr()))

    def _alloc_output(self, name: str) -> torch.Tensor:
        shape = tuple(self.context.get_tensor_shape(name))
        out = torch.empty(shape, dtype=self.dtypes[name], device=self.device)
        self.context.set_tensor_address(name, int(out.data_ptr()))
        return out

    def nn_is_fp16(self):
        return "FP16" in self.checkpoint

    def predict(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor]], data_samples: List["PoseDataSample"]) -> Tuple[torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            current_batch_size = inputs.size(0)
            inputs = (inputs,)
        elif isinstance(inputs, tuple):
            current_batch_size = inputs[0].size(0)
            assert all(i.size(0) == current_batch_size for i in inputs), "All the inputs tensors must have the same batch size."
        else:
            raise ValueError(f"inputs must be a torch.Tensor or a tuple. Received: {type(inputs)}.")

        if current_batch_size != self._running_batch_size:
            for name, tensor in zip(self.input_names, inputs):
                tensor = self._stage_to_gpu(tensor).to(self.dtypes[name])
                self._set_input(name, tensor)
                self._buffer[name] = tensor
            self.context.infer_shapes()
            self._outputs = tuple(self._alloc_output(n) for n in self.output_names)

        self.main_stream.wait_stream(self.default_stream)
        self.context.execute_async_v3(stream_handle=self.main_stream.cuda_stream)
        self.main_stream.synchronize()

        return self._outputs
