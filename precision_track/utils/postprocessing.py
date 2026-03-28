from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict

from precision_track.utils import PoseDataSample, xyxy_cxcywh


def postprocess_one_stage_detections(
    post_processor,
    scores: torch.Tensor,
    objectness: torch.Tensor,
    bboxes: torch.Tensor,
    kpts: torch.Tensor,
    kpt_vis: torch.Tensor,
    features: torch.Tensor,
    priors: torch.Tensor,
    data_samples: List[PoseDataSample],
    kpt_score_thr: Optional[int] = 0,
):
    assert bboxes.shape[0] == len(data_samples)

    scores = scores.sigmoid()
    objectness = objectness.sigmoid()
    scores *= objectness
    scores, labels = scores.max(2, keepdim=True)

    formatted_outputs = []
    for i, data_sample in enumerate(data_samples):
        i_bboxes = bboxes[i]
        i_scores = scores[i]
        i_kpts = kpts[i]
        i_kpt_vis = kpt_vis[i]
        i_labels = labels[i]
        i_features = features[i]

        i_bboxes, i_scores, i_kpts, i_kpt_vis, i_labels, i_features, i_priors, i_kept_idxs = post_processor(
            i_bboxes, i_scores, i_kpts, i_kpt_vis, i_labels, i_features, priors[0], torch.tensor(0)
        )

        i_scores = i_scores.flatten()
        i_kpt_vis = i_kpt_vis.sigmoid()
        i_labels = i_labels.flatten()

        input_size = data_sample.metainfo["input_size"]
        input_center = data_sample.metainfo["input_center"]
        input_scale = data_sample.metainfo["input_scale"]

        scale = torch.tensor(input_scale, dtype=torch.float32, device=i_bboxes.device)
        rescale = scale / torch.tensor(input_size, dtype=torch.float32, device=i_bboxes.device)
        translation = torch.tensor(input_center, dtype=torch.float32, device=i_bboxes.device) - 0.5 * scale

        i_kpts = i_kpts * rescale.view(1, 1, 2) + translation.view(1, 1, 2)
        i_kpts[i_kpt_vis < kpt_score_thr] = 0.0

        i_bboxes = i_bboxes * torch.tile(rescale, (i_bboxes.shape[0], 2)) + torch.tile(translation, (i_bboxes.shape[0], 2))
        i_bboxes = xyxy_cxcywh(i_bboxes)

        pred_instances = Dict()
        pred_instances.bboxes = i_bboxes
        pred_instances.scores = i_scores
        pred_instances.keypoints = i_kpts
        pred_instances.keypoint_scores = i_kpt_vis
        pred_instances.labels = i_labels
        pred_instances.features = F.normalize(i_features, p=2, dim=-1, eps=1e-12)
        pred_instances.kept_idxs = i_kept_idxs
        pred_instances.feature_maps = features[i]
        pred_instances.priors = i_priors

        formatted_pred_instances = {
            "ori_shape": getattr(data_sample, "ori_shape", None),
            "img_id": getattr(data_sample, "img_id", None),
            "seq_id": getattr(data_sample, "seq_id", None),
            "img_path": getattr(data_sample, "img_path", None),
            "id": getattr(data_sample, "id", None),
            "category_id": getattr(data_sample, "category_id", 1),
            "gt_instances": getattr(data_sample, "gt_instance_labels", None),
        }

        formatted_pred_instances["pred_instances"] = pred_instances
        formatted_outputs.append(formatted_pred_instances)
    return formatted_outputs


def postprocess_fpv_action_recognition(
    preds: Union[torch.Tensor, tuple],
    data_samples: Union[List[PoseDataSample], PoseDataSample],
    actions_map: np.ndarray,
    post_processor: Optional[Any] = None,
    action_threshold: Optional[float] = 0.75,
    social_action_threshold: Optional[float] = 0.5,
    group_actions_map: Optional[np.ndarray] = None,
    null_action: Optional[str] = None,
):
    if isinstance(data_samples, List):
        assert len(data_samples) == 1, "Action Recognition does not support batches."
        data_sample = data_samples[0]
    else:
        data_sample = data_samples
    if isinstance(preds, list):
        preds = preds[0]

    edge_probs = None
    social_logits = None
    if isinstance(preds, tuple):
        data_sample["pred_track_instances"]["action_embeddings"] = preds[1]
        if len(preds) >= 3:
            edge_probs = preds[2].squeeze(0)
        if len(preds) >= 4:
            social_logits = preds[3].squeeze(0)
        preds = preds[0].squeeze(0)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy().astype(np.float32)

    action_scores = preds.max(axis=-1)
    actions = actions_map[preds.argmax(axis=-1).reshape(-1)]
    if isinstance(null_action, str):
        actions[action_scores < action_threshold] = null_action
    action_scores = action_scores.reshape(-1)

    if edge_probs is not None and social_logits is not None and isinstance(group_actions_map, np.ndarray):
        edge_probs = edge_probs.detach().cpu().numpy()
        max_social_scores, social_actions = social_logits.max(dim=-1)

        is_social = torch.where((max_social_scores >= social_action_threshold) & (social_actions > 0))[0].cpu().numpy()

        social_actions = social_actions.detach().cpu().numpy()[is_social]

        valid_ids = data_sample["pred_track_instances"]["instances_id"]
        target_ids = np.full(len(valid_ids), "-1", dtype=object)

        if np.any(social_actions):
            actions[is_social] = group_actions_map[social_actions - 1]
            target_ids[is_social] = [",".join(map(str, valid_ids[edge_probs[i] >= social_action_threshold])) or "-1" for i in is_social]

        data_sample["pred_track_instances"]["target_ids"] = target_ids.astype(str)

    valid_context = data_sample["pred_track_instances"]["valid_action_recognition_context"]
    if isinstance(valid_context, torch.Tensor):
        valid_context = valid_context.detach().cpu().numpy()
    actions[~valid_context] = "Analyzing..."
    action_scores[~valid_context] = 1

    data_sample["pred_track_instances"]["actions"] = actions
    data_sample["pred_track_instances"]["action_scores"] = action_scores

    if post_processor is not None:
        data_sample = post_processor(data_sample)

    return data_sample
