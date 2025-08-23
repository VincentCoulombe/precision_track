import os
from typing import List, Tuple, Union

import numpy as np
import onnxruntime as ort
import pkg_resources
import torch
from mmengine import Config

from precision_track.registry import RUNTIMES, TASK_UTILS
from precision_track.utils import PoseDataSample, onnx_to_fp16, parse_device_id

from .base import InferenceOnlyRuntime


@RUNTIMES.register_module()
class ONNXRuntime(InferenceOnlyRuntime):
    def __init__(
        self,
        input_shapes: List[Union[Tuple, Config]],
        checkpoint: str,
        device: str = "auto",
        half_precision: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(
            checkpoint=checkpoint,
            device=device,
            half_precision=half_precision,
            verbose=verbose,
            **kwargs,
        )

        self.torch_dtype = torch.float16 if self.half_precision else torch.float32
        self.numpy_dtype = np.float16 if self.half_precision else np.float32

        self.running_batch = -1
        self.input_shapes = [(self.running_batch,) + TASK_UTILS.build(s).shape if isinstance(s, dict) else torch.Size(s) for s in input_shapes]

        self.checkpoint = checkpoint
        self._live_inputs: List[ort.OrtValue] = []

        self._assert_runtime()

    def _assert_runtime(self) -> None:
        if self.half_precision and self.device != "cpu":
            onnx_to_fp16(self.checkpoint, logger=None)

        self.device_id = parse_device_id(self.device)
        ep = list()
        if self.device == "cpu":
            ep.append("CPUExecutionProvider")
        elif "cuda" in self.device:
            ep.append(
                ("CUDAExecutionProvider", {"device_id": self.device_id}),
            )
        else:
            raise ValueError(f"The {self.device} device is not yet supported.")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(self.checkpoint, providers=ep, sess_options=so)

        self.io_binding = self.session.io_binding()
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_names = [i.name for i in self.session.get_inputs()]

        runtime = "onnxruntime"
        if "cuda" in self.device:
            runtime += "-gpu"
        try:
            ort_version = pkg_resources.get_distribution(runtime).version
        except pkg_resources.DistributionNotFound:
            raise Exception(
                f"""The {runtime} package dependency is not installed. Note that the onnxruntime and the onnxruntime-gpu are mutually exclusive,
                meaning you can not infer with both installed on the same virtual env."""
            )
        self.log_runtime(f"Inference backend set to: {runtime} {ort_version}, with the following checkpoint {os.path.abspath(self.checkpoint)}")

    def predict(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor]],
        data_samples: List[PoseDataSample],
    ) -> Tuple[torch.Tensor]:
        """Feed the tensors, run the model, return CUDA tensors."""
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)

        self._bind_inputs(inputs)
        self._bind_outputs()

        if "cuda" in self.device:
            torch.cuda.synchronize()

        self.session.run_with_iobinding(self.io_binding)

        outputs = tuple(torch.utils.dlpack.from_dlpack(o._ortvalue.to_dlpack()) for o in self.io_binding.get_outputs())

        self._live_inputs.clear()  # Served its purpose...
        return outputs

    def _bind_inputs(self, tensors: Tuple[torch.Tensor]):
        """Create OrtValues and bind them; resizes batch if needed."""
        for idx, (t, name, template_shape) in enumerate(zip(tensors, self.input_names, self.input_shapes)):
            B, *features_block = t.shape
            assert features_block == list(template_shape[1:]), f"Expected {template_shape[1:]}, got {features_block}"

            if B != self.running_batch:
                self.input_shapes[idx] = torch.Size([B, *features_block])
            self.running_batch = B

            ort_value = ort.OrtValue.ortvalue_from_numpy(
                t.detach().to(self.torch_dtype).cpu().contiguous().numpy(),
                self.device,
                self.device_id,
            )
            self.io_binding.bind_ortvalue_input(name, ort_value)
            self._live_inputs.append(ort_value)  # keep the pointers alive troughout gc...

    def _bind_outputs(self):
        for name in self.output_names:
            self.io_binding.bind_output(name, self.device, self.device_id)
