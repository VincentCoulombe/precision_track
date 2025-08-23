from collections.abc import Iterable
from typing import Optional, Union

import numpy as np
import torch
from mmengine.model import ImgDataPreprocessor

from precision_track.datasets.transforms.utils import Compose
from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample


@MODELS.register_module()
class InferencePreprocessor(ImgDataPreprocessor):
    SUPPORTED_INPUT_FORMAT = [np.ndarray, torch.Tensor, str]

    def __init__(
        self,
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        input_size: Optional[tuple] = (640, 640),
        pad_val: Optional[tuple] = (114, 114, 114),
        **kwargs,
    ):
        assert isinstance(input_size, tuple)
        assert isinstance(pad_val, tuple)
        self.input_size = input_size
        self.pipeline = None

        super().__init__(mean=mean, std=std, bgr_to_rgb=False, pad_value=pad_val[0])

    def forward(
        self,
        data: dict,
        *args,
        **kwargs,
    ) -> dict:
        batched = isinstance(data["data_samples"], Iterable)
        inputs = data["inputs"]
        if not batched:
            data["data_samples"] = [data["data_samples"]]
            assert data["inputs"].ndim == 3, f"The preprocessing entered non-batched inference mode, but the input tensor has {inputs.ndim} dimension."
            C, H, W = data["inputs"].shape
            data["inputs"] = data["inputs"].view(1, C, H, W)
        ds = data["data_samples"]
        assert len(inputs) == len(ds), f"The number of frames ({len(inputs)}) != the number of frame ids ({len(ds)}) "
        for i, (input_, data_sample) in enumerate(zip(inputs, ds)):
            if isinstance(data_sample, int):
                data_sample = self._id_to_data_sample(data_sample)
            if self.pipeline is None:
                self._build_pipeline()
            if not isinstance(input_, torch.Tensor):
                formatted_input = self._format_input(input_, data_sample)
                self.pipeline(formatted_input)
                input_ = torch.tensor(formatted_input.pop("img")).permute(2, 0, 1)
                data_sample.update(PoseDataSample(metainfo=formatted_input))
            data["inputs"][i] = input_
            data["data_samples"][i] = data_sample
        return super().forward(data, training=False)

    def _format_input(self, input_: Union[np.ndarray, str], data_sample: PoseDataSample) -> dict:
        if isinstance(input_, str):
            return dict(img_path=input_, img_id=data_sample.img_id)
        elif isinstance(input_, np.ndarray):
            if np.argmin(input_.shape) == 0:
                input_ = input_.transpose(1, 2, 0)
            return dict(img=input_, img_id=data_sample.img_id)
        else:
            raise ValueError(f"The provided input must by one of: {self.SUPPORTED_INPUT_FORMAT}, not {type(input_)}.")

    @staticmethod
    def _id_to_data_sample(id_: int) -> PoseDataSample:
        return PoseDataSample(metainfo=dict(img_id=id_))

    def _build_pipeline(self) -> None:
        self.pipeline = Compose(
            [
                dict(type="LoadImage"),
                dict(type="BottomupResize", input_size=self.input_size, pad_val=self.pad_value),
            ]
        )
