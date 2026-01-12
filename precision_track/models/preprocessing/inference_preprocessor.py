from collections.abc import Iterable
from typing import Optional, Union, Tuple
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
import torch
from mmengine.model import ImgDataPreprocessor, BaseDataPreprocessor

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


@MODELS.register_module()
class WildLifeReIDPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        batch_size: int,
        input_shape: Optional[Tuple] = (224, 224),
    ):
        super().__init__()

        assert len(input_shape) == 2
        self.input_shape = []
        for shape in input_shape:
            assert shape > 0
            self.input_shape.append(int(shape))
        self.input_shape = tuple(input_shape)

        self.transforms = T.Compose(
            [
                T.Resize(size=self.input_shape),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.transformed_inputs = torch.empty(
            batch_size,
            3,
            input_shape[0],
            input_shape[1],
            dtype=torch.float32,
        )
        assert batch_size > 0
        self.batch_size = int(batch_size)

    def forward(self, data: dict, *args, **kwargs) -> dict:
        inputs = data["inputs"]
        running_batch_size = len(inputs)
        assert running_batch_size <= self.batch_size

        for i, input_ in enumerate(inputs):
            self.transformed_inputs[i] = self.transforms(input_)

        data["inputs"] = self.transformed_inputs[:running_batch_size]
        return data

    def to(self, *args, **kwargs) -> nn.Module:
        out = super().to(*args, **kwargs)
        self.transformed_inputs = self.transformed_inputs.to(self._device)
        return out

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.cuda.current_device())
        self.transformed_inputs = self.transformed_inputs.to(self._device)
        return super().cuda()

    def musa(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.musa.current_device())
        self.transformed_inputs = self.transformed_inputs.to(self._device)
        return super().musa()

    def npu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.npu.current_device())
        self.transformed_inputs = self.transformed_inputs.to(self._device)
        return super().npu()

    def mlu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.mlu.current_device())
        self.transformed_inputs = self.transformed_inputs.to(self._device)
        return super().mlu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device("cpu")
        self.transformed_inputs = self.transformed_inputs.to(self._device)
        return super().cpu()
