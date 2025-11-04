import os
from logging import WARNING
from typing import List, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from addict import Dict
from mmengine import Config
from mmengine.logging import MMLogger, print_log
import logging

from precision_track.registry import MODELS, TRACKING
from precision_track.utils import iou_batch, bbox_overlaps, linear_assignment, reformat, PoseDataSample, calculate_pose_velocities, calculate_bbox_velocities

from .base import BaseAssignationAlgorithm


@TRACKING.register_module()
class GroundTruth(BaseAssignationAlgorithm):

    def __init__(
        self,
        gt_bbox_path: str,
        gt_kpts_path: Optional[str] = None,
        weight_iou_with_det_scores: bool = True,
        match_iou_thr: float = 0.9,
        appearance_extractor: Optional[str] = None,
        **kwargs,
    ):
        """GroundTruth: Tracking by doing detections-tracks associations
        over time using IoUs with manually defined trajectories provided by gt_bbox_path.

        Args:
            gt_bbox_path (str): The manually defined trajectories. Expected to follow the MOT format: https://arxiv.org/pdf/1906.04567.
            weight_iou_with_det_scores (bool, optional): If the detection/track IoUs are weighted by the detection's confidence. Defaults to True.
            match_iou_thr (float, optional): The minimum score for a detection/track association. Default to 0.9.
        """
        super().__init__(**kwargs)
        assert isinstance(weight_iou_with_det_scores, bool)
        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        assert 0.0 <= match_iou_thr <= 1.0
        self.match_iou_thrs = match_iou_thr
        assert os.path.exists(gt_bbox_path)
        assert os.path.splitext(gt_bbox_path)[1] == ".csv", "Tracking grund truths must respect the MOT format."
        self.gt_bbox = pd.read_csv(gt_bbox_path)
        assert list(self.gt_bbox.columns[:7]) == [
            "frame_id",
            "class_id",
            "instance_id",
            "x",
            "y",
            "w",
            "h",
        ], "Tracking grund truths must respect the MOT format."
        self.logger = MMLogger.get_current_instance()
        self.fields_to_remove.append("bboxes")

        self.gt_kpts = None
        if gt_kpts_path is not None:
            assert os.path.exists(gt_kpts_path)
            assert os.path.splitext(gt_kpts_path)[1] == ".csv", "Tracking grund truths must respect the MOT format."
            self.gt_kpts = pd.read_csv(gt_kpts_path)
            assert list(self.gt_bbox.columns[:3]) == [
                "frame_id",
                "class_id",
                "instance_id",
            ], "Tracking grund truths must respect the MOT format."
            self.fields_to_remove.append("keypoints")

        if isinstance(appearance_extractor, str):
            assert os.path.exists(appearance_extractor)
            assert self.gt_kpts is not None, "The appearance extraction process requires keypoints."
            self.a_e = MODELS.build(Config.fromfile(appearance_extractor).model)
        else:
            self.a_e = None

    def update_thresholds(self, tracking_thresholds: dict) -> None:
        pass

    def get_tracks_preds_ious(
        self,
        track_bboxes: np.ndarray,
        pred_instances: dict,
        pred_idx: np.ndarray,
    ) -> np.ndarray:

        det_bboxes = pred_instances["bboxes"][pred_idx].astype(np.float32)
        if track_bboxes.shape[0] == 0 or det_bboxes.shape[0] == 0:
            return np.zeros((track_bboxes.shape[0], det_bboxes.shape[0]), dtype=np.float32)
        return iou_batch(track_bboxes, det_bboxes)

    def assign_ids(
        self,
        dists: np.ndarray,
        tracks: Optional[dict] = None,
        track_ids: Optional[List[int]] = None,
        pred_instances: Optional[dict] = None,
        pred_idx: Optional[np.ndarray] = None,
        weight_iou_with_det_scores: Optional[bool] = False,
        match_iou_thr: Optional[float] = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if weight_iou_with_det_scores and pred_idx is not None:
            det_scores = pred_instances.scores[pred_idx]
            dists *= det_scores

        # add huge cost to dets/tracks class mismatch
        if tracks is not None and track_ids is not None and pred_idx is not None and pred_instances is not None:
            track_labels = np.array([tracks[id]["labels"] for id in track_ids], dtype=int)
            det_labels = pred_instances["labels"][pred_idx]
            same_labels = det_labels[None, :] == track_labels[:, None]
            dists = (dists * same_labels).astype(np.float32)

        matched_tracks, matched_dets = linear_assignment(1 - dists, match_iou_thr)
        return matched_tracks.astype(int), matched_dets.astype(int)

    def _tracks_to_pred_bboxes(self, tracks: dict, track_ids: List[int]) -> np.ndarray:
        num_tracks = len(track_ids)
        track_bboxes = np.zeros((num_tracks, 4), dtype=np.float32)
        track_velocities = np.empty((num_tracks, 2), dtype=np.float32)
        for i, track_id in enumerate(track_ids):
            track_bboxes[i] = tracks[track_id].pred_bboxe.astype(np.float32)
            track_velocities[i] = tracks[track_id].velocity.astype(np.float32)
        return track_bboxes, track_velocities

    def __call__(
        self,
        data_sample: dict,
        tracks: dict,
        confirmed_ids: List[int],
        unconfirmed_ids: List[int],
        *args,
        **kwargs,
    ) -> None:
        """Associate new detections to either already defined tracks or new
        ones.

        Args:
            data_sample (dict): The data sample containing the new detections will be modified in place.
            tracks (dict): The defined tracks.
            confirmed_ids (List[int]): The ids of the tracks old enough to be confirmed (not false positives).
            unconfirmed_ids (List[int]): The ids of the tracks too young to be asserted (might be false positives).
        """
        (
            pred_instances,
            frame_id,
            scores,
            num_tracks,
            num_dets,
            matched_trk_ids,
            matched_trk_bboxes,
            matched_features,
            matched_pred_idx,
            idx_counter,
        ) = self.init_call(data_sample, tracks)

        det_features = pred_instances["features"]
        frame_gt_bboxes = self.gt_bbox[self.gt_bbox.frame_id == frame_id].values
        if not frame_gt_bboxes.size:
            pred_instances = Dict({k: np.empty((0)) for k in pred_instances})
            pred_instances["bboxes"] = np.empty((0, 4))
            pred_instances["keypoints"] = np.empty((0, 0, 3))
            pred_instances["features"] = np.empty((0, 1))
            self.save_to_ds(
                data_samples=data_sample,
                detections=pred_instances,
                track_ids=np.empty((0)),
                pred_idx=np.empty((0), dtype=bool),
                predicted_bboxes=pred_instances["bboxes"],
            )
            return
        frame_gt_classes = frame_gt_bboxes[:, 1]
        frame_gt_instances = frame_gt_bboxes[:, 2]
        track_bboxes = reformat(frame_gt_bboxes[:, 3:7].reshape(-1, 4), "xywh", "cxcywh")
        if self.gt_kpts is not None:
            track_kpts = self.gt_kpts[self.gt_kpts.frame_id == frame_id].values[:, 3:]

        # Remove duplicate labels (in case of labelling errors)
        unique_gt, counts = np.unique(frame_gt_instances, return_counts=True)
        duplicates = unique_gt[counts > 1]
        duplicate_indices = [np.where(frame_gt_instances == dup)[0] for dup in duplicates]
        for duplicate_index in duplicate_indices:
            to_remove = duplicate_index[1:]
            frame_gt_instances = np.delete(frame_gt_instances, to_remove)
            frame_gt_classes = np.delete(frame_gt_classes, to_remove)
            track_bboxes = np.delete(track_bboxes, to_remove, axis=0)
            if self.gt_kpts is not None:
                track_kpts = np.delete(track_kpts, to_remove, axis=0)

        all_preds = np.ones_like(pred_instances["labels"], dtype=bool)
        frame_gt = Dict()
        for cls_id, inst_id, bbox in zip(frame_gt_classes, frame_gt_instances, track_bboxes):
            frame_gt[inst_id].labels = cls_id
            frame_gt[inst_id].instances_id = inst_id
            frame_gt[inst_id].bboxes = [bbox]

        ious = self.get_tracks_preds_ious(track_bboxes, pred_instances, all_preds)
        matched_tracks, matched_dets = self.assign_ids(
            dists=ious,
            tracks=frame_gt,
            track_ids=frame_gt_instances,
            pred_instances=pred_instances,
            pred_idx=all_preds,
            weight_iou_with_det_scores=self.weight_iou_with_det_scores,
            match_iou_thr=self.match_iou_thrs,
        )
        if matched_tracks.shape[0] == 0 and track_bboxes.shape[0] > 0:
            self.logger.warning(
                (
                    f"None of the labelled tracks where matched at the frame #{frame_id}."
                    "Either the labels are incorrect, the model is badly optimized or this is an edge case."
                ),
            )

        matched_pred_idx_tmp = np.where(all_preds)[0][matched_dets]
        num_matches = len(matched_dets)

        matched_pred_idx[:num_matches] = matched_dets
        matched_trk_bboxes[:num_matches] = track_bboxes[matched_tracks]
        matched_trk_ids[:num_matches] = frame_gt_instances[matched_tracks]
        matched_features[:num_matches] = det_features[matched_pred_idx_tmp]
        pred_instances["bboxes"] = track_bboxes[matched_tracks]
        if self.gt_kpts is not None:
            pred_instances["keypoints"] = track_kpts[matched_tracks]

        idx_counter = num_matches

        self.save_to_ds(
            data_samples=data_sample,
            detections=pred_instances,
            track_ids=matched_trk_ids[:idx_counter],
            pred_idx=matched_pred_idx[:idx_counter],
            predicted_bboxes=matched_trk_bboxes[:idx_counter],
            features=matched_features[:idx_counter],
        )

    def save_to_ds(
        self,
        data_samples: dict,
        detections: dict,
        track_ids: np.ndarray,
        pred_idx: np.ndarray,
        predicted_bboxes: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        appearance_features: Optional[np.ndarray] = None,
        cosine_similarities: Optional[np.ndarray] = None,
    ):
        data_samples["pred_track_instances"] = {"ids": track_ids, "bboxes": detections["bboxes"]}
        if self.gt_kpts is not None:
            N, K = detections["keypoints"].shape
            kpts = detections["keypoints"].reshape(N, K // 3, 3)[..., :2]
            data_samples["pred_track_instances"]["keypoints"] = kpts

        data_samples["pred_track_instances"].update({k: v[pred_idx] for k, v in detections.items() if k not in self.fields_to_remove})

        if predicted_bboxes is not None:
            data_samples["pred_track_instances"].update(
                {
                    "next_frame_bboxes": predicted_bboxes,
                }
            )
        if velocities is not None:
            data_samples["pred_track_instances"].update({"velocities": velocities})
        if features is not None:
            data_samples["pred_track_instances"].update({"features": features})

        if self.a_e is not None:
            data_samples["pred_track_instances"]["bboxes"] = torch.from_numpy(data_samples["pred_track_instances"]["bboxes"])
            kpts = torch.from_numpy(data_samples["pred_track_instances"]["keypoints"])
            kpt_scores = torch.from_numpy(data_samples["pred_track_instances"]["keypoint_scores"])
            features = torch.from_numpy(data_samples["pred_track_instances"]["features"])
            N, K, _ = kpts.shape
            block_kpts = torch.zeros_like(features)
            block_kpts[..., : 3 * K] = torch.cat((kpts.view(N, K, 2), kpt_scores.view(N, K, 1)), dim=2).view(N, 3 * K)

            inputs = torch.cat((block_kpts, features)).to(torch.float32)

            data_samples.update({"block_data_shape": inputs.shape, "n_kpts": K * 3})
            if inputs.numel():
                appearance_features, representations = self.a_e._forward(inputs, [data_samples])
                representations = representations.detach().cpu().numpy()
                appearance_features = appearance_features.detach().cpu().numpy()

            else:
                appearance_features = np.zeros((0, 0))
                representations = np.zeros((0, 0))

            data_samples["pred_track_instances"]["bboxes"] = data_samples["pred_track_instances"]["bboxes"].detach().cpu().numpy()

            data_samples["pred_track_instances"].update(
                {
                    "appearance_features": appearance_features,
                    "representations": representations,
                }
            )


@TRACKING.register_module()
class OnlineGroundTruth(BaseAssignationAlgorithm):
    def __init__(
        self,
        match_iou_thr: float = 0.99,
        weight_iou_with_det_scores: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert 0.0 <= match_iou_thr <= 1.0
        self.match_iou_thrs = match_iou_thr
        assert isinstance(weight_iou_with_det_scores, bool)
        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.logger = MMLogger.get_current_instance()
        self.fields_to_remove.extend(["bboxes", "keypoints", "actions"])
        self.gt_kpts = False
        self.gt_actions = False

    def init_call(self, data_sample: dict, tracks: dict):
        for k, v in data_sample.pred_instances.items():
            if k not in ["features", "keypoints", "actions"]:
                data_sample.pred_instances[k] = v
        self.kf.frame_id = data_sample.img_id
        num_tracks = len(tracks)
        num_dets = data_sample.pred_instances["scores"].shape[0]
        max_size = num_tracks + num_dets
        prior_features = data_sample.pred_instances.get("features", torch.empty(0))
        if not self.prior_feature_size:
            self.prior_feature_size = prior_features.size(1)
        self.feature_device = prior_features.device
        return (
            data_sample.pred_instances,
            torch.zeros(max_size, dtype=torch.int, device=self.feature_device),
            torch.zeros((max_size, 4), dtype=torch.float32, device=self.feature_device),
            torch.zeros((max_size, self.prior_feature_size), dtype=torch.float32, device=self.feature_device),
            torch.zeros(max_size, dtype=torch.int, device=self.feature_device),
            0,
        )

    def __call__(self, *args, **kwargs) -> None:
        self.loss(*args, **kwargs)

    def loss(
        self,
        data_sample: PoseDataSample,
        tracks: dict,
        confirmed_ids: List[int],
        unconfirmed_ids: List[int],
        *args,
        **kwargs,
    ) -> None:
        (
            pred_instances,
            matched_trk_ids,
            matched_trk_bboxes,
            matched_features,
            matched_pred_idx,
            idx_counter,
        ) = self.init_call(data_sample, tracks)

        det_features = pred_instances["features"]
        gt_instances = data_sample.gt_instances.to(self.feature_device).to(torch.float32)
        frame_gt_classes = gt_instances.labels
        frame_gt_instances = gt_instances.instances_id
        frame_gt_bboxes = gt_instances.bboxes
        if not frame_gt_bboxes.size or not gt_instances:
            pred_instances = Dict({k: torch.empty((0), device=self.feature_device) for k in pred_instances})
            pred_instances["bboxes"] = torch.empty((0, 4), device=self.feature_device)
            pred_instances["keypoints"] = torch.empty((0, 0, 3), device=self.feature_device)
            pred_instances["features"] = torch.empty((0, 1), device=self.feature_device)
            pred_instances["actions"] = torch.empty((0), device=self.feature_device)
            self.save_to_ds(
                data_sample=data_sample,
                detections=pred_instances,
                track_ids=torch.empty((0), device=self.feature_device),
                pred_idx=torch.empty((0), dtype=torch.bool, device=self.feature_device),
                predicted_bboxes=pred_instances["bboxes"],
            )
            return
        self.kf.multi_predict(tracks, confirmed_ids)

        if hasattr(gt_instances, "keypoints"):
            self.gt_kpts = True
            track_kpts = gt_instances.keypoints

        if hasattr(gt_instances, "actions"):
            self.gt_actions = True
            track_actions = gt_instances.actions

        # Remove duplicate labels (in case of labelling errors)
        unique_gt, counts = torch.unique(frame_gt_instances, return_counts=True)
        duplicates = unique_gt[counts > 1]
        for dup in duplicates:
            dup_idx = torch.where(frame_gt_instances == dup)[0]
            if dup_idx.numel() > 1:
                mask = torch.ones_like(frame_gt_instances, dtype=torch.bool)
                mask[dup_idx[1:]] = False
                frame_gt_instances = frame_gt_instances[mask]
                frame_gt_classes = frame_gt_classes[mask]
                frame_gt_bboxes = frame_gt_bboxes[mask]
                if self.gt_kpts:
                    track_kpts = track_kpts[mask]
                if self.gt_actions:
                    track_actions = track_actions[mask]

        all_preds = torch.ones_like(pred_instances["labels"], dtype=torch.bool)
        frame_gt = Dict()
        for cls_id, inst_id, bbox in zip(frame_gt_classes, frame_gt_instances, frame_gt_bboxes):
            inst_id = inst_id.item()
            cls_id = cls_id.item()
            frame_gt[inst_id].labels = cls_id
            frame_gt[inst_id].instances_id = inst_id
            frame_gt[inst_id].bboxes = [bbox]

        ious = self.get_tracks_preds_ious(frame_gt_bboxes, pred_instances, all_preds)
        matched_tracks, matched_dets = self.assign_ids(
            dists=ious,
            tracks=frame_gt,
            track_ids=frame_gt_instances.tolist(),
            pred_instances=pred_instances,
            pred_idx=all_preds,
            weight_iou_with_det_scores=self.weight_iou_with_det_scores,
            match_iou_thr=self.match_iou_thrs,
        )
        matched_pred_idx_tmp = torch.where(all_preds)[0][matched_dets]
        num_matches = len(matched_dets)

        matched_pred_idx[:num_matches] = matched_dets
        matched_trk_bboxes[:num_matches] = frame_gt_bboxes[matched_tracks]
        matched_trk_ids[:num_matches] = frame_gt_instances[matched_tracks]
        matched_features[:num_matches] = det_features[matched_pred_idx_tmp]
        pred_instances["bboxes"] = frame_gt_bboxes[matched_tracks]
        if self.gt_kpts:
            pred_instances["keypoints"] = track_kpts[matched_tracks]
        if self.gt_actions:
            pred_instances["actions"] = track_actions[matched_tracks]

        idx_counter = num_matches

        self.save_to_ds(
            data_sample=data_sample,
            detections=pred_instances,
            track_ids=matched_trk_ids[:idx_counter].detach().cpu().numpy(),
            pred_idx=matched_pred_idx[:idx_counter],
            predicted_bboxes=matched_trk_bboxes[:idx_counter],
            features=matched_features[:idx_counter],
        )

    def save_to_ds(
        self,
        data_sample: PoseDataSample,
        detections: dict,
        track_ids: np.ndarray,
        pred_idx: np.ndarray,
        predicted_bboxes: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ):
        data_sample.pred_track_instances = {"ids": track_ids, "bboxes": detections["bboxes"]}
        if self.gt_kpts:
            data_sample.pred_track_instances["keypoints"] = detections["keypoints"]
        if self.gt_actions:
            data_sample.pred_track_instances["actions"] = detections["actions"]

        for k, v in detections.items():
            if k not in self.fields_to_remove:
                data_sample.pred_track_instances[k] = v[pred_idx]

        if predicted_bboxes is not None:
            data_sample.pred_track_instances.update(
                {
                    "next_frame_bboxes": predicted_bboxes,
                }
            )
        if velocities is not None:
            data_sample.pred_track_instances.update({"velocities": velocities})
        if features is not None:
            data_sample.pred_track_instances.update({"features": features})

    def update_thresholds(self, tracking_thresholds: dict) -> None:
        pass

    def get_tracks_preds_ious(
        self,
        track_bboxes: np.ndarray,
        pred_instances: dict,
        pred_idx: np.ndarray,
    ) -> np.ndarray:

        det_bboxes = pred_instances["bboxes"][pred_idx].to(torch.float32)
        if track_bboxes.shape[0] == 0 or det_bboxes.shape[0] == 0:
            return torch.zeros((track_bboxes.shape[0], det_bboxes.shape[0]), dtype=torch.float32, device=self.feature_device)
        return bbox_overlaps(track_bboxes.view(1, -1, 4), reformat(det_bboxes, "cxcywh", "xyxy").view(1, -1, 4)).squeeze(0)

    def assign_ids(
        self,
        dists: np.ndarray,
        tracks: Optional[dict] = None,
        track_ids: Optional[List[int]] = None,
        pred_instances: Optional[dict] = None,
        pred_idx: Optional[np.ndarray] = None,
        weight_iou_with_det_scores: Optional[bool] = False,
        match_iou_thr: Optional[float] = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if weight_iou_with_det_scores and pred_idx is not None:
            det_scores = pred_instances.scores[pred_idx]
            dists *= det_scores

        # add huge cost to dets/tracks class mismatch
        if tracks is not None and track_ids is not None and pred_idx is not None and pred_instances is not None:
            track_labels = torch.tensor([tracks[id]["labels"] for id in track_ids], dtype=torch.int, device=self.feature_device)
            det_labels = pred_instances["labels"][pred_idx]
            same_labels = det_labels[None, :] == track_labels[:, None]
            dists = (dists * same_labels).detach().cpu().numpy()

        matched_tracks, matched_dets = linear_assignment(1 - dists, match_iou_thr)
        return torch.from_numpy(matched_tracks.astype(int)), torch.from_numpy(matched_dets.astype(int))
