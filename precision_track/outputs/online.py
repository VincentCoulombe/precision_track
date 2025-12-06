from precision_track.registry import OUTPUTS

from .base import BaseOutput


@OUTPUTS.register_module()
class OnlinePthEmbeddingOutput(BaseOutput):

    def __init__(
        self,
        instance_data: str = "pred_track_instances",
        embs_field: str = "features",
    ):
        self.instance_data = instance_data
        self.embs_field = embs_field
        self.reset()

    def __call__(self, data: dict) -> None:
        """Load data into the output.

        Args:
            data (dict): The data to load into the output
        """
        track_data = data.get(self.instance_data, None)
        assert track_data is not None, f"data, does not contain {self.instance_data}. Heres data's keys: {data.keys()}"
        self.data.append(track_data["features"].detach().cpu())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def reset(self) -> None:
        self.data = []

    def save(self) -> None:
        raise NotImplementedError("OnlineNpyEmbeddingOutput does not support saving and readin files.")

    def read(self) -> None:
        raise NotImplementedError("OnlineNpyEmbeddingOutput does not support saving and readin files.")

    def valid(self) -> bool:
        raise NotImplementedError("OnlineNpyEmbeddingOutput does not support saving and readin files.")
