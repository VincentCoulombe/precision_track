from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict

from precision_track.utils import PoseDataSample, xyxy_cxcywh


def format_detection_output(
    data_sample: PoseDataSample,
    bboxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    keypoints: torch.Tensor,
    keypoint_scores: torch.Tensor,
    features: torch.Tensor,
    scale: torch.Tensor,
    translation: torch.Tensor,
    kept_idxs: Optional[torch.Tensor] = None,
    feature_maps: Optional[torch.Tensor] = None,
    priors: Optional[torch.Tensor] = None,
    kpt_score_thr: float = 0.0,
    normalize_features: bool = True,
) -> dict:
    """Shared detection post-processing tail used by every detection backend.

    Given a single image's *already-decoded, already-NMS'd* predictions expressed in the model's
    (letterboxed) input space, this:
      1. rescales boxes/keypoints to the original image with the affine ``pt * scale + translation``
         (per-axis ``scale``/``translation`` (2,) — the inverse of whatever letterbox the backend used),
      2. converts boxes ``xyxy -> cxcywh``,
      3. zeroes keypoints whose (already-activated) score is below ``kpt_score_thr``,
      4. assembles the ``pred_instances`` structure + per-frame metadata the tracker expects.

    Backends differ only in how they produce ``bboxes``/``scores``/... and ``scale``/``translation``;
    the boilerplate here is identical, so both ``DetectionBackend`` and ``UltralyticsDetectionBackend``
    wrap this function.

    Args:
        bboxes: ``(N, 4)`` xyxy in input space.
        keypoints: ``(N, K, 2)`` in input space.
        keypoint_scores: ``(N, K)`` already-activated visibility/confidence.
        scale / translation: ``(2,)`` per-axis input->original mapping (``x`` then ``y``).
    """
    scale = scale.to(device=bboxes.device, dtype=torch.float32)
    translation = translation.to(device=bboxes.device, dtype=torch.float32)
    s4 = torch.cat([scale, scale])  # (4,) for xyxy
    t4 = torch.cat([translation, translation])

    keypoints = keypoints * scale.view(1, 1, 2) + translation.view(1, 1, 2)
    keypoints[keypoint_scores < kpt_score_thr] = 0.0

    bboxes = bboxes * s4 + t4
    bboxes = xyxy_cxcywh(bboxes)

    pred_instances = Dict()
    pred_instances.bboxes = bboxes
    pred_instances.scores = scores
    pred_instances.keypoints = keypoints
    pred_instances.keypoint_scores = keypoint_scores
    pred_instances.labels = labels
    pred_instances.features = F.normalize(features, p=2, dim=-1, eps=1e-12) if normalize_features else features
    pred_instances.kept_idxs = kept_idxs
    pred_instances.feature_maps = feature_maps
    pred_instances.priors = priors

    return {
        "ori_shape": getattr(data_sample, "ori_shape", None),
        "img_id": getattr(data_sample, "img_id", None),
        "seq_id": getattr(data_sample, "seq_id", None),
        "img_path": getattr(data_sample, "img_path", None),
        "id": getattr(data_sample, "id", None),
        "category_id": getattr(data_sample, "category_id", 1),
        "gt_instances": getattr(data_sample, "gt_instance_labels", None),
        "pred_instances": pred_instances,
    }


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
        i_bboxes, i_scores, i_kpts, i_kpt_vis, i_labels, i_features, i_priors, i_kept_idxs = post_processor(
            bboxes[i], scores[i], kpts[i], kpt_vis[i], labels[i], features[i], priors[0], torch.tensor(0)
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

        formatted_outputs.append(
            format_detection_output(
                data_sample,
                bboxes=i_bboxes,
                scores=i_scores,
                labels=i_labels,
                keypoints=i_kpts,
                keypoint_scores=i_kpt_vis,
                features=i_features,
                scale=rescale,
                translation=translation,
                kept_idxs=i_kept_idxs,
                feature_maps=features[i],
                priors=i_priors,
                kpt_score_thr=kpt_score_thr,
            )
        )
    return formatted_outputs


def _drop_batch_dim(tensor: torch.Tensor, expected_ndim: int) -> torch.Tensor:
    """Strip a leading batch axis only if the tensor still carries one.

    ``squeeze(0)`` cannot be used here: with a single tracked instance the instance
    axis is also of size 1 and would be collapsed along with (or instead of) the batch.
    """
    return tensor[0] if tensor.ndim == expected_ndim + 1 else tensor


def _single_data_sample(data_samples: Union[List[PoseDataSample], PoseDataSample]) -> PoseDataSample:
    if isinstance(data_samples, List):
        assert len(data_samples) == 1, "Action Recognition does not support batches."
        return data_samples[0]
    return data_samples


def empty_fpv_action_recognition(
    data_samples: Union[List[PoseDataSample], PoseDataSample],
    actions_dtype: Optional[Any] = "<U32",
    with_target_ids: Optional[bool] = False,
):
    """The fields an action recognition frame carries when no subject is tracked.

    Frames without instances are fed to no model at all: the pairwise graph of an empty
    set of subjects is undefined, and every runtime chokes on 0-sized inputs.
    """
    data_sample = _single_data_sample(data_samples)
    pred_track_instances = data_sample["pred_track_instances"]
    pred_track_instances["actions"] = np.empty(0, dtype=actions_dtype)
    pred_track_instances["action_scores"] = np.empty(0, dtype=np.float32)
    if with_target_ids:
        pred_track_instances["target_ids"] = np.empty(0, dtype=str)
    return data_sample


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
    data_sample = _single_data_sample(data_samples)
    if isinstance(preds, list):
        preds = preds[0]

    edge_probs = None
    social_logits = None
    if isinstance(preds, tuple):
        data_sample["pred_track_instances"]["action_embeddings"] = _drop_batch_dim(preds[1], 2)
        if len(preds) >= 3:
            edge_probs = _drop_batch_dim(preds[2], 2)
        if len(preds) >= 4:
            social_logits = _drop_batch_dim(preds[3], 2)
        preds = _drop_batch_dim(preds[0], 2)

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

        if social_actions.size > 0:
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
