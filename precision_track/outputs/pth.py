import os

import torch
from mmengine.logging import print_log

from precision_track.registry import OUTPUTS

from .base import BaseOutput


@OUTPUTS.register_module()
class PthAppearanceDatabaseOutput(BaseOutput):
    def __init__(
        self,
        path: str,
        instance_data: str = "appearance_database",
        ids_field: str = "identities",
        embs_field: str = "features",
    ):
        name, _ = os.path.splitext(path)
        self.path = f"{name}.pth"

        self.instance_data = instance_data
        self.ids_field = ids_field
        self.embs_field = embs_field
        self.reset()

    def _compute_feature_hash(self, feature: torch.Tensor) -> int:
        """Compute a hash for a feature tensor to detect duplicates."""
        return hash(feature.numpy().tobytes())

    def __call__(self, data: dict) -> None:
        track_data = data.get(self.instance_data, None)
        assert track_data is not None, f"data, does not contain {self.instance_data}. Heres data's keys: {data.keys()}"
        if not track_data:
            return

        frame_id = data["img_id"]
        features = track_data[self.embs_field]
        identities = track_data[self.ids_field]

        valid_idx = identities > 0
        valid_features = features[valid_idx].detach().cpu()
        valid_identities = identities[valid_idx].detach().cpu()

        frame_feature_ids = []
        for i in range(len(valid_identities)):
            identity = int(valid_identities[i].item())
            feature = valid_features[i]

            feature_hash = self._compute_feature_hash(feature)
            registry_key = (identity, feature_hash)

            if registry_key not in self.feature_registry:
                feature_id = self.next_feature_id
                self.feature_registry[registry_key] = feature_id
                self.unique_features[feature_id] = feature
                self.unique_identities[feature_id] = identity
                self.next_feature_id += 1
            else:
                feature_id = self.feature_registry[registry_key]

            frame_feature_ids.append(feature_id)

        self.fact_frame_ids[frame_id] = frame_feature_ids

    def __getitem__(self, idx: int):
        feature_ids = self.fact_frame_ids.get(idx, [])
        if not feature_ids:
            return ([], [])

        features = torch.stack([self.unique_features[fid] for fid in feature_ids])
        identities = torch.tensor([self.unique_identities[fid] for fid in feature_ids])
        return (features, identities)

    def reset(self):
        self.fact_frame_ids = dict()  # frame_id -> [feature_id]
        self.unique_features = dict()  # feature_id -> Features
        self.unique_identities = dict()  # feature_id -> Identity
        self.feature_registry = dict()  # (Identity, features_hash) -> feature_id
        self.next_feature_id = 0

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        unique_features_cpu = {fid: feat for fid, feat in self.unique_features.items()}

        torch.save(
            {
                "fact_frame_ids": self.fact_frame_ids,
                "unique_features": unique_features_cpu,
                "unique_identities": self.unique_identities,
                "next_feature_id": self.next_feature_id,
            },
            self.path,
        )
        print_log(f"Saved output: {self.path}")

    def read(self):
        data = torch.load(self.path, weights_only=False)
        self.fact_frame_ids = data["fact_frame_ids"]
        self.unique_features = data["unique_features"]
        self.unique_identities = data["unique_identities"]
        self.next_feature_id = data["next_feature_id"]

        # Rebuild the feature_registry
        self.feature_registry = {}
        for fid, feat in self.unique_features.items():
            identity = self.unique_identities[fid]
            feature_hash = self._compute_feature_hash(feat)
            self.feature_registry[(identity, feature_hash)] = fid

    def valid(self) -> bool:
        return os.path.isfile(self.path)
