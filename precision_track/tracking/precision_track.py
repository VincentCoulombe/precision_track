from typing import List, Optional

import numpy as np

from precision_track.registry import TRACKING
from precision_track.utils import batch_bbox_areas, oks_batch, parse_pose_metainfo

from .byte_track import ByteTrack


@TRACKING.register_module()
class PrecisionTrack(ByteTrack):

    def __init__(
        self,
        metafile: str,
        obj_score_thrs: Optional[dict] = dict(high=0.6, low=0.1),
        weight_iou_with_det_scores: Optional[bool] = False,
        match_iou_thrs: Optional[dict] = dict(high=0.9, low=0.5, tentative=0.9),
        init_track_thr: Optional[float] = 0.8,
        keep_searching_for: Optional[int] = 1,
        dynamic_temporal_scaling: Optional[bool] = False,
        alpha: Optional[float] = 0.5,
        with_kpt_weights: Optional[bool] = True,
        with_kpt_sigmas: Optional[bool] = False,
        nb_frames_retain: Optional[int] = 10,
        **kwargs,
    ):
        """PrecisionTrack: Building trajectories by doing detections-tracks associations
        over time using poses and IoUs.

        Args:
            metafile (str): The metainfo for the skeletons of the tracks.
            Need to minimally contain the sigmas of the keypoints (see Coco dataset for documentation).
            And the score thresholds obtained during the calibration process.
            obj_score_thrs (dict, optional): The confidence level thresholds from which detections are considered for the association steps.
            Defaults to dict(high=0.6, low=0.1).
            weight_iou_with_det_scores (bool, optional): If the detection/track IoUs are weighted by the detection's confidence. Defaults to True.
            match_iou_thrs (dict, optional): The minimum score for a detection/track association. Defaults to dict(high=0.9, low=0.5, tentative=0.9).
            init_track_thr (float, optional): The minimum level of detection confidence to init a new track. Defaults to 0.8.
            keep_searching_for (int, optional): How many frames after its last detection do we still consider a track alive. Defaults to 1.
            alpha (float, optional): The weight of the pose in the association score. Defaults to 0.2.
            with_kpt_weights (bool, optional): Weight the pose association score by each keypoint confidence. Defaults to True.
            with_kpt_sigmas (bool, optional): Weight the pose association score by each keypoint sigma. Defaults to False.
        """
        assert isinstance(keep_searching_for, int) and 0 < keep_searching_for
        self.keep_searching_for = keep_searching_for
        assert isinstance(with_kpt_weights, bool)
        self.with_kpt_weights = with_kpt_weights
        assert isinstance(with_kpt_sigmas, bool)
        self.with_kpt_sigmas = with_kpt_sigmas
        metadata = parse_pose_metainfo({"from_file": metafile})
        self.num_keypoints = metadata["num_keypoints"]
        assert isinstance(self.num_keypoints, int) and 0 <= self.num_keypoints
        self.sigmas = metadata["sigmas"].astype(np.float32) if with_kpt_sigmas else None
        assert isinstance(nb_frames_retain, int) and 0 < nb_frames_retain
        self.nb_frames_retain = nb_frames_retain
        assert isinstance(dynamic_temporal_scaling, bool)
        self.use_deltas_t = dynamic_temporal_scaling

        super().__init__(
            obj_score_thrs=obj_score_thrs,
            weight_iou_with_det_scores=weight_iou_with_det_scores,
            match_iou_thrs=match_iou_thrs,
            init_track_thr=init_track_thr,
            **kwargs,
        )
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha

        self.frame_id = 0

    def get_tracks_preds_oks(self, track_bboxes, track_poses, track_pose_confs, pred_instances, pred_idx, eps=1e-3):
        det_poses = pred_instances["keypoints"][pred_idx].astype(np.float32)
        det_poses_score = pred_instances["keypoint_scores"][pred_idx]
        det_weights = None
        if self.with_kpt_weights:
            det_weights = det_poses_score.astype(np.float32)
        det_areas = batch_bbox_areas(track_bboxes).astype(np.float32)
        if det_areas is None:
            img_id = pred_instances["img_id"]
            raise ValueError(f"Some bounding boxes at frame {img_id} have negative width or height.")
        if track_poses.shape[0] == 0 or det_poses.shape[0] == 0:
            return np.zeros((track_poses.shape[0], det_poses.shape[0]), dtype=np.float32)
        track_poses[track_pose_confs < 0.5] = 0.0
        det_poses[det_poses_score < 0.5] = 0.0
        return oks_batch(
            track_poses,
            det_poses,
            keypoint_weights=det_weights,
            keypoint_std=self.sigmas,
            areas=det_areas,
            eps=eps,
        )

    def get_cost_matrix(
        self,
        track_bboxes,
        track_poses,
        track_pose_confs,
        pred_instances,
        pred_idx,
        deltas_t,
        use_oks=True,
    ):
        ious = self.get_tracks_preds_ious(track_bboxes, pred_instances, pred_idx)
        if not use_oks:
            return ious

        oks = self.get_tracks_preds_oks(
            track_bboxes,
            track_poses,
            track_pose_confs,
            pred_instances,
            pred_idx,
        )

        dists = (self.alpha - self.alpha * deltas_t) * oks + ((1 - self.alpha) + (1 - self.alpha) * deltas_t) * ious

        return dists

    def _tracks_to_pred_bboxes_kpts(self, tracks: dict, track_ids: List[int]) -> np.ndarray:
        num_tracks = len(track_ids)
        track_poses = np.empty((num_tracks, self.num_keypoints, 2), dtype=np.float32)
        track_poses_conf = np.empty((num_tracks, self.num_keypoints), dtype=np.float32)
        track_bboxes = np.empty((num_tracks, 4), dtype=np.float32)
        track_detection_deltas = np.zeros((num_tracks, 1), dtype=np.float32)
        keep_trying = np.ones(num_tracks, dtype=bool)

        for i, track_id in enumerate(track_ids):
            track_poses[i] = tracks[track_id]["pred_keypoints"].astype(np.float32)
            track_bboxes[i] = tracks[track_id]["pred_bboxe"].astype(np.float32)
            track_poses_conf[i] = tracks[track_id]["keypoint_scores"][-1].astype(np.float32)
            trk_frame_id = tracks[track_id]["frame_ids"][-1]
            keep_trying[i] = trk_frame_id >= self.frame_id - self.keep_searching_for
            if trk_frame_id > 0 and self.use_deltas_t:
                track_detection_deltas[i] = (self.frame_id - trk_frame_id) / self.nb_frames_retain

        return track_bboxes, track_poses, track_poses_conf, track_detection_deltas, keep_trying

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
        self.frame_id = frame_id

        if num_tracks == 0 or num_dets == 0:
            matched_pred_idx_tmp = np.where(scores > self.init_track_thr)[0]
            num_new_tracks = len(matched_pred_idx_tmp)

            matched_trk_ids[:num_new_tracks] = np.arange(num_tracks, num_tracks + num_new_tracks)
            matched_trk_bboxes[:num_new_tracks] = pred_instances["bboxes"][matched_pred_idx_tmp]
            matched_features[idx_counter : idx_counter + num_new_tracks] = det_features[matched_pred_idx_tmp]
            matched_pred_idx[:num_new_tracks] = matched_pred_idx_tmp
            idx_counter = num_new_tracks
        else:
            # Predict current location
            self.kf.multi_predict(tracks, confirmed_ids)

            # High confidence bboxes
            remaining_conf_idx = scores > self.obj_score_thrs["high"]

            # Low confidence bboxes
            remaining_not_conf_idx = (~remaining_conf_idx) & (scores > self.obj_score_thrs["low"])

            # 1. First match: high confidence detections -> confirmed tracks
            confirmed_ids = np.array(confirmed_ids)
            (
                track_bboxes,
                track_poses,
                track_pose_confs,
                deltas_t,
                keep_trying,
            ) = self._tracks_to_pred_bboxes_kpts(tracks, confirmed_ids)

            dists = self.get_cost_matrix(
                track_bboxes,
                track_poses,
                track_pose_confs,
                pred_instances,
                remaining_conf_idx,
                deltas_t,
                use_oks=True,
            )

            matched_tracks, matched_dets = self.assign_ids(
                dists,
                tracks,
                confirmed_ids,
                pred_instances,
                remaining_conf_idx,
                self.weight_iou_with_det_scores,
                self.match_iou_thrs["high"],
            )

            matched_pred_idx_tmp = np.where(remaining_conf_idx)[0][matched_dets]
            num_matches = len(matched_pred_idx_tmp)

            matched_pred_idx[:num_matches] = matched_pred_idx_tmp
            matched_trk_bboxes[:num_matches] = track_bboxes[matched_tracks]
            matched_trk_ids[:num_matches] = confirmed_ids[matched_tracks]
            matched_features[:num_matches] = det_features[matched_pred_idx_tmp]

            idx_counter = num_matches
            remaining_conf_idx[matched_pred_idx_tmp] = False

            remaining_trk_ids = np.setdiff1d(
                confirmed_ids,
                matched_trk_ids[:idx_counter],
                assume_unique=True,
            )
            mask = np.isin(confirmed_ids, remaining_trk_ids) & keep_trying
            remaining_alive_confirmed_trk_ids = confirmed_ids[mask]

            # 2. Second match: remaining high confidence detections -> unconfirmed tracks
            if unconfirmed_ids:
                (
                    track_bboxes,
                    track_poses,
                    track_pose_confs,
                    deltas_t,
                    _,
                ) = self._tracks_to_pred_bboxes_kpts(tracks, unconfirmed_ids)

                dists = self.get_cost_matrix(
                    track_bboxes,
                    track_poses,
                    track_pose_confs,
                    pred_instances,
                    remaining_conf_idx,
                    deltas_t,
                    use_oks=True,
                )

                matched_tracks, matched_dets = self.assign_ids(
                    dists,
                    tracks,
                    unconfirmed_ids,
                    pred_instances,
                    remaining_conf_idx,
                    self.weight_iou_with_det_scores,
                    self.match_iou_thrs["tentative"],
                )

                matched_pred_idx_tmp = np.where(remaining_conf_idx)[0][matched_dets]
                num_matches = len(matched_pred_idx_tmp)

                matched_trk_ids[idx_counter : idx_counter + num_matches] = np.array(unconfirmed_ids)[matched_tracks]
                matched_pred_idx[idx_counter : idx_counter + num_matches] = matched_pred_idx_tmp
                matched_trk_bboxes[idx_counter : idx_counter + num_matches] = track_bboxes[matched_tracks]
                matched_features[idx_counter : idx_counter + num_matches] = det_features[matched_pred_idx_tmp]

                remaining_conf_idx[matched_pred_idx_tmp] = False
                idx_counter += num_matches

            # 3. Third match: low confidence detections -> remaining confirmed tracks
            if remaining_alive_confirmed_trk_ids.size > 0:
                (
                    track_bboxes,
                    track_poses,
                    track_pose_confs,
                    deltas_t,
                    _,
                ) = self._tracks_to_pred_bboxes_kpts(tracks, remaining_alive_confirmed_trk_ids)
                dists = self.get_cost_matrix(
                    track_bboxes,
                    track_poses,
                    track_pose_confs,
                    pred_instances,
                    remaining_not_conf_idx,
                    deltas_t,
                    use_oks=False,
                )
                matched_tracks, matched_dets = self.assign_ids(
                    dists,
                    tracks,
                    remaining_alive_confirmed_trk_ids,
                    pred_instances,
                    remaining_not_conf_idx,
                    False,
                    self.match_iou_thrs["low"],
                )
                matched_pred_idx_tmp = np.where(remaining_not_conf_idx)[0][matched_dets]
                num_matches = len(matched_pred_idx_tmp)

                matched_trk_ids[idx_counter : idx_counter + num_matches] = np.array(remaining_alive_confirmed_trk_ids)[matched_tracks]
                matched_pred_idx[idx_counter : idx_counter + num_matches] = matched_pred_idx_tmp
                matched_trk_bboxes[idx_counter : idx_counter + num_matches] = track_bboxes[matched_tracks]
                matched_features[idx_counter : idx_counter + num_matches] = det_features[matched_pred_idx_tmp]

                idx_counter += num_matches

            # 4. Fourth match: remaining high confidence detections -> new tracks
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
