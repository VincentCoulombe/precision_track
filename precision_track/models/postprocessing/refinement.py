import numpy as np
from mmengine.registry import MODELS

from precision_track.utils import match_nearest_keypoint, parse_pose_metainfo

from .base import BaseActionPostProcessor


@MODELS.register_module()
class KeypointBasedActionRefinement(BaseActionPostProcessor):
    """Refine an action prediction based on the distance between the source and the sink keypoints."""

    SUPPORTED_criterias = {"nearest": match_nearest_keypoint}

    def __init__(self, action_to_refine: str, source_keypoints: list, sink_keypoints: list, criterias: list, refined_actions: list, metainfo: str):
        assert isinstance(action_to_refine, str)
        assert len(source_keypoints) == len(sink_keypoints) == len(criterias) == len(refined_actions)
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.actions = np.array(metainfo.get("actions", []))
        assert action_to_refine in self.actions, f"{action_to_refine} not in {self.actions.tolist()}."
        self.action_to_refine = action_to_refine
        nb_keypoints = metainfo.get("num_keypoints", 0)
        for src, sinks, metric, refined_action in zip(source_keypoints, sink_keypoints, criterias, refined_actions):
            assert 0 <= src < nb_keypoints
            if isinstance(sinks, int):
                assert 0 <= sinks < nb_keypoints
            elif isinstance(sinks, list):
                for sink in sinks:
                    assert 0 <= sink < nb_keypoints
            else:
                raise ValueError("sink_keypoints must be a list composed of either integers of lists of integers.")
            assert metric in self.SUPPORTED_criterias
            assert isinstance(refined_action, str)
        self.source_keypoints = source_keypoints
        self.sink_keypoints = sink_keypoints
        self.criterias = [self.SUPPORTED_criterias[metric] for metric in criterias]
        self.refined_actions = refined_actions

        super().__init__()

    def forward(self, data_sample: dict):
        actions = data_sample["pred_track_instances"]["actions"]
        action_mask = np.where(actions == self.action_to_refine)[0]
        if action_mask.size == 0:
            return data_sample
        to_refine = data_sample["pred_track_instances"]["keypoints"][action_mask]
        N, K, _ = data_sample["pred_track_instances"]["keypoints"].shape
        gallery = data_sample["pred_track_instances"]["keypoints"].reshape(N * K, 2)
        gallery_idx_to_sink = np.tile(np.arange(K), N)
        source_masks = []
        for i in range(action_mask.shape[0]):
            sm_i = np.ones_like(data_sample["pred_track_instances"]["keypoint_scores"], dtype=bool)
            sm_i[action_mask[i]] = 0
            source_masks.append(sm_i.reshape(-1))
        source_masks = np.stack(source_masks)
        for source, sinks, criteria, refined_action in zip(
            self.source_keypoints,
            self.sink_keypoints,
            self.criterias,
            self.refined_actions,
        ):
            sources_to_refine = to_refine[:, source, :]
            selected_sinks = criteria(sources_to_refine, gallery, mask=source_masks, idx_map=gallery_idx_to_sink)
            to_refine_idx = np.ma.getdata(action_mask[np.isin(selected_sinks, sinks)])
            actions[to_refine_idx] = refined_action

        return data_sample
