from typing import List, Tuple, Union

import os
from torch import Tensor
import torch
import numpy as np
from precision_track.registry import OUTPUTS, MODELS
from precision_track.utils import PoseDataSample, parse_pose_metainfo


@MODELS.register_module()
class ActionRecognitionOracle:
    def __init__(
        self,
        directory: str,
        metainfo: str,
        format: str = "CsvActions",
        *args,
        **kwargs,
    ):
        assert os.path.exists(directory), f"{directory} does not exists."
        assert os.path.isdir(directory), f"{directory} is not a directory."
        self.sequences = dict()
        for filename in os.listdir(directory):
            gt = OUTPUTS.build({"type": format, "path": os.path.join(directory, filename)})
            gt.read()
            self.sequences[os.path.splitext(filename)[0]] = gt

        self.actions = parse_pose_metainfo(dict(from_file=metainfo)).get("actions")
        assert self.actions is not None
        assert len(self.actions) > 0
        self.actions = np.array(self.actions)

    def predict(self, inputs: Tuple[Tensor], data_samples: List[PoseDataSample] = None) -> Tuple[Tensor]:
        oracle_preds = []
        for data_sample in data_samples:
            sequence_name = data_sample.get("seq_name")
            assert sequence_name in self.sequences, f"{sequence_name} not in {self.sequences}."
            frame_id = data_sample.get("img_id")
            inst_id = data_sample.instance_id

            frame_data = np.array(self.sequences[sequence_name][frame_id])
            inst_id_idx = frame_data[:, 2] == str(inst_id)
            inst_frame_data = frame_data[inst_id_idx]
            if inst_frame_data.size == 0:
                raise ValueError(f"No Oracle value for sequence= {sequence_name}, frame_id={frame_id} and instance_id={inst_id}.")
            oracle_pred = (inst_frame_data[0, 3] == self.actions).astype(int)
            if oracle_pred.sum() != 1:
                raise ValueError(f"Expected a One-hot, but got: {oracle_pred} for sequence= {sequence_name}, frame_id={frame_id} and instance_id={inst_id}.")
            oracle_preds.append(oracle_pred)
        return torch.from_numpy(np.stack(oracle_preds))

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        return self.predict(**data)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        return self.val_step(data)

    def eval(self):
        return
