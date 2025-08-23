from typing import List, Optional, Tuple

import numpy as np

from precision_track.registry import TRACKING
from precision_track.utils import iou_batch, linear_assignment

from .base import BaseAssignationAlgorithm


@TRACKING.register_module()
class SORT(BaseAssignationAlgorithm):

    def __init__(
        self,
        obj_score_thr: float = 0.6,
        weight_iou_with_det_scores: bool = True,
        match_iou_thr: float = 0.9,
        init_track_thr: float = 0.8,
        **kwargs,
    ):
        """SORT: Building trajectories by doing detections-tracks associations
        over time using IoUs.

        Args:
            obj_score_thr (float, optional): The confidence level threshold from which detections are considered for the association step.
            Defaults to 0.6.
            weight_iou_with_det_scores (bool, optional): If the detection/track IoUs are weighted by the detection's confidence. Defaults to True.
            match_iou_thr (float, optional): The minimum score for a detection/track association. Default to 0.9.
            init_track_thr (float, optional): The minimum level of detection confidence to init a new track. Defaults to 0.8.
        """
        super().__init__(**kwargs)
        assert 0.0 <= obj_score_thr <= 1.0
        self.obj_score_thr = obj_score_thr
        assert 0.0 <= init_track_thr <= 1.0
        self.init_track_thr = init_track_thr
        assert isinstance(weight_iou_with_det_scores, bool)
        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        assert 0.0 <= match_iou_thr <= 1.0
        self.match_iou_thrs = match_iou_thr

    def update_thresholds(self, tracking_thresholds: dict) -> None:
        if "init_thr" in tracking_thresholds:
            self.init_track_thr = tracking_thresholds["init_thr"]
        if "conf_thr" in tracking_thresholds:
            self.obj_score_thr = tracking_thresholds["conf_thr"]

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
        for i, track_id in enumerate(track_ids):
            track_bboxes[i] = tracks[track_id].pred_bboxe.astype(np.float32)
        return track_bboxes

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

        if num_tracks == 0 or num_dets == 0:
            matched_pred_idx_tmp = np.where(scores > self.init_track_thr)[0]
            num_new_tracks = len(matched_pred_idx_tmp)

            matched_trk_ids[:num_new_tracks] = np.arange(num_tracks, num_tracks + num_new_tracks)
            matched_trk_bboxes[:num_new_tracks] = pred_instances["bboxes"][matched_pred_idx_tmp]
            matched_features[idx_counter : idx_counter + num_new_tracks] = det_features[matched_pred_idx_tmp]
            matched_pred_idx[:num_new_tracks] = matched_pred_idx_tmp
            idx_counter = num_new_tracks
        else:
            self.kf.multi_predict(tracks, confirmed_ids)
            remaining_conf_idx = scores > self.obj_score_thr

            # 1. First match: high confidence detections -> confirmed tracks
            track_bboxes = self._tracks_to_pred_bboxes(tracks, confirmed_ids)
            ious = self.get_tracks_preds_ious(track_bboxes, pred_instances, remaining_conf_idx)
            matched_tracks, matched_dets = self.assign_ids(
                dists=ious,
                tracks=tracks,
                track_ids=confirmed_ids,
                pred_instances=pred_instances,
                pred_idx=remaining_conf_idx,
                weight_iou_with_det_scores=self.weight_iou_with_det_scores,
                match_iou_thr=self.match_iou_thrs,
            )

            matched_pred_idx_tmp = np.where(remaining_conf_idx)[0][matched_dets]
            num_matches = len(matched_pred_idx_tmp)

            matched_pred_idx[:num_matches] = matched_pred_idx_tmp
            matched_features[:num_matches] = det_features[matched_pred_idx_tmp]
            matched_trk_bboxes[:num_matches] = track_bboxes[matched_tracks]
            matched_trk_ids[:num_matches] = np.array(confirmed_ids)[matched_tracks]

            idx_counter = num_matches
            remaining_conf_idx[matched_pred_idx_tmp] = False

            # 2. Second match: remaining high confidence detections -> unconfirmed tracks
            if unconfirmed_ids:
                track_bboxes = self._tracks_to_pred_bboxes(tracks, unconfirmed_ids)
                ious = self.get_tracks_preds_ious(track_bboxes, pred_instances, remaining_conf_idx)
                matched_tracks, matched_dets = self.assign_ids(
                    dists=ious,
                    tracks=tracks,
                    track_ids=unconfirmed_ids,
                    pred_instances=pred_instances,
                    pred_idx=remaining_conf_idx,
                    weight_iou_with_det_scores=self.weight_iou_with_det_scores,
                    match_iou_thr=self.match_iou_thrs,
                )

                matched_pred_idx_tmp = np.where(remaining_conf_idx)[0][matched_dets]
                num_matches = len(matched_pred_idx_tmp)

                matched_trk_ids[idx_counter : idx_counter + num_matches] = np.array(unconfirmed_ids)[matched_tracks]
                matched_pred_idx[idx_counter : idx_counter + num_matches] = matched_pred_idx_tmp
                matched_trk_bboxes[idx_counter : idx_counter + num_matches] = track_bboxes[matched_tracks]
                matched_features[idx_counter : idx_counter + num_matches] = det_features[matched_pred_idx_tmp]
                remaining_conf_idx[matched_pred_idx_tmp] = False
                idx_counter += num_matches

            # 3.
            new_tracks_idx = (scores > self.init_track_thr) & remaining_conf_idx
            new_tracks_idx = np.where(new_tracks_idx)[0]
            num_new_tracks = len(new_tracks_idx)

            matched_pred_idx[idx_counter : idx_counter + num_new_tracks] = new_tracks_idx
            matched_trk_bboxes[idx_counter : idx_counter + num_new_tracks] = pred_instances["bboxes"][new_tracks_idx]
            max_id_confirmed = np.max(np.append(confirmed_ids, -1))
            max_id_unconfirmed = np.max(np.append(unconfirmed_ids, -1))
            max_id = np.max([max_id_confirmed, max_id_unconfirmed])
            new_ids = np.arange(max_id + 1, max_id + 1 + num_new_tracks)

            matched_trk_ids[idx_counter : idx_counter + num_new_tracks] = new_ids
            matched_features[idx_counter : idx_counter + num_new_tracks] = det_features[new_tracks_idx]

            idx_counter += num_new_tracks

        self.save_to_ds(
            data_samples=data_sample,
            detections=pred_instances,
            track_ids=matched_trk_ids[:idx_counter],
            pred_idx=matched_pred_idx[:idx_counter],
            predicted_bboxes=matched_trk_bboxes[:idx_counter],
            features=matched_features[:idx_counter],
        )
