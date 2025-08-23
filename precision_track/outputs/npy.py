import os
from typing import Optional

import numpy as np
from mmengine.logging import print_log

from precision_track.registry import OUTPUTS

from .base import BaseOutput


@OUTPUTS.register_module()
class NpyEmbeddingOutput(BaseOutput):

    def __init__(
        self,
        path: str,
        buffer_size: Optional[int] = 100,
        instance_data: str = "pred_track_instances",
        ids_field: str = "ids",
        embs_field: str = "features",
    ):
        self.path = path
        self.buffer_size = buffer_size
        self.instance_data = instance_data
        self.ids_field = ids_field
        self.embs_field = embs_field
        self.reset()

    def reset(self):
        self.data = np.empty((0,), dtype=np.float32)
        self._length = 0
        self.emb_dim = None
        self.entities = list()

    def __call__(self, data: dict) -> None:
        """Load data into the output.

        Args:
            data (dict): The data to load into the output
        """
        track_data = data.get(self.instance_data, None)
        assert track_data is not None, f"data, does not contain {self.instance_data}. Heres data's keys: {data.keys()}"
        frame_ids = track_data[self.ids_field]
        if frame_ids.size == 0:
            return
        confirmed_ids = frame_ids >= 0
        frame_ids = frame_ids[confirmed_ids]
        frame_embs = track_data[self.embs_field][confirmed_ids].detach().cpu().numpy()

        new_instances = set(frame_ids.astype(int).tolist()) - set(self.entities)
        self.entities.extend(new_instances)

        running_emb_dim = frame_embs.shape[1]
        if self.emb_dim is None:
            self.emb_dim = running_emb_dim
            self.data = np.array([[id_] * self.emb_dim for id_ in self.entities], dtype=np.float32).reshape(1, len(self.entities), -1)
        else:
            nb_timesteps = self.data.shape[0]
            if nb_timesteps > 0:
                formatted_new_instances = np.zeros((nb_timesteps, len(new_instances), self.emb_dim), dtype=np.float32)
                if new_instances:
                    formatted_new_instances[0, ...] = np.array([[id_] * self.emb_dim for id_ in list(new_instances)], dtype=np.float32)
                self.data = np.concatenate((self.data, formatted_new_instances), axis=1)
        assert running_emb_dim == self.emb_dim

        new_embs = np.zeros((len(self.entities), self.emb_dim), dtype=np.float32)
        for i, id_ in enumerate(self.entities):
            idx = np.where(frame_ids == id_)[0]
            if idx.size == 1:
                new_embs[i] = frame_embs[idx]

        # Concatenante maintains the crucial floating precision, but copy the whole data structure in memory, O(N) each time... ONLY process small clips!
        if self.data.shape[0] <= self._length + 1:
            self._buff_data_structure()

        self.data[self._length + 1] = new_embs
        self._length += 1

    def _buff_data_structure(self):
        self.data = np.concatenate((self.data, np.zeros((self.buffer_size, self.data.shape[1], self.emb_dim), dtype=np.float32)))

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        return self.data[idx + 1]

    def save(self) -> None:
        """Save an output to disk at sink_dir."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        name, _ = os.path.splitext(self.path)
        self.path = f"{name}.npy"
        print_log(f"Saved output: {self.path}")
        np.save(self.path, self.data[: self._length + 1, ...])

    def read(self) -> None:
        """Read an output from source_path and load it into memory."""
        self.data = np.load(self.path)
        T, _, E = self.data.shape
        self.emb_dim = E
        self.entities = list(self.data[0, :, 0].astype(int))
        self._length = T - 1
