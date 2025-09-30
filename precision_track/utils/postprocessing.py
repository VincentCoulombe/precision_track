from typing import Dict, Optional, List
import torch

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
        input_size = data_sample.metainfo["input_size"]
        input_center = data_sample.metainfo["input_center"]
        input_scale = data_sample.metainfo["input_scale"]

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
        pred_instances.features = i_features
        pred_instances.kept_idxs = i_kept_idxs
        pred_instances.feature_maps = features[i]
        pred_instances.priors = i_priors

        formatted_pred_instances = {
            "ori_shape": data_sample.ori_shape,
            "img_id": data_sample.img_id,
            "img_path": getattr(data_sample, "img_path", None),
            "id": getattr(data_sample, "id", None),
            "category_id": getattr(data_sample, "category_id", 1),
            "gt_instances": getattr(data_sample, "gt_instance_labels", None),
        }

        formatted_pred_instances["pred_instances"] = pred_instances
        formatted_outputs.append(formatted_pred_instances)
    return formatted_outputs
