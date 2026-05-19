import os
from typing import List, Union

import numpy as np
import torch
import yaml

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample

from .base import BaseBackend


@MODELS.register_module()
class ReIDBackend(BaseBackend):
    SUPPORTED_RUNTIMES = [".onnx", ".engine"]

    def __init__(self, checkpoint: str, metainfo: str) -> None:
        assert os.path.isfile(checkpoint), f"The provided re-identification checkpoint '{os.path.abspath(checkpoint)}' does not exists."
        assert (
            os.path.splitext(checkpoint)[1] in self.SUPPORTED_RUNTIMES
        ), f"The provided re-identification checkpoint '{os.path.abspath(checkpoint)}' must be one of : {self.SUPPORTED_RUNTIMES}."

        assert os.path.isfile(metainfo), f"The provided re-identification metadata file '{os.path.abspath(metainfo)}' does not exists."
        with open(metainfo, "r") as f:
            metadata = yaml.safe_load(f)

        self.identities = metadata.get("identities")
        assert isinstance(self.identities, list), f"The metadata file '{metainfo}' must contain a list of identities"

        self.disabled_identities = metadata.get("disabled_identities", [])
        assert isinstance(
            self.disabled_identities, list
        ), f"The metadata file '{metainfo}' contains invalid values for key 'disabled_identities': {self.disabled_identities}"

        unknown_disabled = set(self.disabled_identities) - set(self.identities)
        assert not unknown_disabled, (
            f"The metadata file '{metainfo}' lists 'disabled_identities' that are not part of 'identities': "
            f"{sorted(unknown_disabled)}. Disabled identities must be a subset of the model's identities."
        )

        self.nb_features = int(metadata.get("nb_features", 0))
        assert self.nb_features > 0

        input_shape = metadata.get("input_shape")
        assert input_shape is not None, f"'{metainfo}' must contain an input_shape."

        assert len(input_shape) == 2
        self.input_shape = [3]
        for shape in input_shape:
            assert shape > 0
            self.input_shape.append(int(shape))
        self.input_shape = tuple(self.input_shape)
        input_shape = [(-1,) + self.input_shape]
        self.input_shape = input_shape[-2:]

        super(ReIDBackend, self).__init__(
            dict(
                checkpoint=checkpoint,
                output_names=["output"],
                input_shapes=input_shape,
            )
        )

    def preprocess(self, inputs: np.ndarray, data_samples: List[Union[int, PoseDataSample]]):
        return dict(inputs=inputs, data_samples=data_samples)

    def postprocess(self, features: torch.Tensor, logits: torch.Tensor, data_samples: List[PoseDataSample]) -> List[dict]:
        return features, logits

    def loss(
        self,
        inputs: List[Union[np.ndarray, str]],
        data_samples: List[Union[int, PoseDataSample]],
        *args,
        **kwargs,
    ) -> dict:
        data = self.preprocess(inputs, data_samples)
        losses = self._runtime.loss(**data)
        return {"losses": losses, "data_samples": data_samples}

    def val_step(self, inputs: List[torch.Tensor], data_samples: List[dict], *args, **kwargs) -> list:
        return self._runtime.val_step(inputs, data_samples, *args, **kwargs)

    def test_step(self, inputs: List[torch.Tensor], data_samples: List[dict], *args, **kwargs) -> list:
        return self._runtime.test_step(inputs, data_samples, *args, **kwargs)
