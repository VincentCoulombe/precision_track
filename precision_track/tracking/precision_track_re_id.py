from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine import Config

from precision_track.models.backends import ReIDBackend
from precision_track.registry import TRACKING
from precision_track.utils import PoseDataSample

from .precision_track import PrecisionTrack


@TRACKING.register_module()
class PrecisionTrackReID(nn.Module, PrecisionTrack):
    KEEP_AS_TENSORS = ["appearances", "representations"]

    def __init__(
        self,
        metafile: str,
        re_id_cfg: Union[Config, str],
        obj_score_thrs: Optional[dict] = dict(high=0.6, low=0.1),
        weight_iou_with_det_scores: Optional[bool] = False,
        match_iou_thrs: Optional[dict] = dict(high=0.9, low=0.5, tentative=0.9),
        init_track_thr: Optional[float] = 0.8,
        keep_searching_for: Optional[int] = 1,
        dynamic_temporal_scaling: Optional[bool] = False,
        alpha: Optional[float] = 1 / 3,
        beta: Optional[float] = 1 / 3,
        with_kpt_weights: Optional[bool] = True,
        with_kpt_sigmas: Optional[bool] = False,
        nb_frames_retain: Optional[int] = 10,
        **kwargs,
    ):
        nn.Module.__init__(self)
        PrecisionTrack.__init__(
            self,
            metafile,
            obj_score_thrs,
            weight_iou_with_det_scores,
            match_iou_thrs,
            init_track_thr,
            keep_searching_for,
            dynamic_temporal_scaling,
            alpha,
            with_kpt_weights,
            with_kpt_sigmas,
            nb_frames_retain,
        )

        if isinstance(re_id_cfg, (Config, dict)):
            self.re_id_cfg = re_id_cfg
        elif isinstance(re_id_cfg, str):
            self.re_id_cfg = Config.fromfile(re_id_cfg).re_id_cfg
        else:
            raise ValueError(f"The ReId configuration file: {re_id_cfg} is invalid. It should either be a Config, a dict or a path to the config file.")
        self.re_id = ReIDBackend(self.re_id_cfg)
        half_precision = self.re_id_cfg.runtime.get("half_precision", False)
        self.dtype = torch.float16 if half_precision else torch.float32
        self.prior_feature_size = self.re_id_cfg.n_embd

        assert 0 <= beta <= 1
        assert 0 <= beta + alpha <= 1
        self.beta = beta

    def _tracks_to_pred_bboxes_kpts(self, tracks: dict, track_ids: List[int]) -> np.ndarray:
        num_tracks = len(track_ids)
        track_poses = np.empty((num_tracks, self.num_keypoints, 2), dtype=np.float32)
        track_poses_conf = np.empty((num_tracks, self.num_keypoints), dtype=np.float32)
        track_bboxes = np.empty((num_tracks, 4), dtype=np.float32)
        track_velocities = np.empty((num_tracks, 2), dtype=np.float32)
        track_detection_deltas = np.zeros((num_tracks, 1), dtype=np.float32)
        keep_trying = np.ones(num_tracks, dtype=bool)
        features = torch.empty((num_tracks, self.prior_feature_size), dtype=self.dtype, device=self.feature_device)

        for i, track_id in enumerate(track_ids):
            track_poses[i] = tracks[track_id]["pred_keypoints"].astype(np.float32)
            track_bboxes[i] = tracks[track_id]["pred_bboxe"].astype(np.float32)
            track_velocities[i] = tracks[track_id]["velocity"].astype(np.float32)
            track_poses_conf[i] = tracks[track_id]["keypoint_scores"][-1].astype(np.float32)
            trk_frame_id = tracks[track_id]["frame_ids"][-1]
            keep_trying[i] = trk_frame_id >= self.frame_id - self.keep_searching_for
            if trk_frame_id > 0 and self.use_deltas_t:
                track_detection_deltas[i] = (self.frame_id - trk_frame_id) / self.nb_frames_retain
            features[i] = tracks[track_id]["features"][-1]

        return track_bboxes, track_poses, track_velocities, track_poses_conf, track_detection_deltas, keep_trying, features

    def get_cost_matrix(
        self,
        track_bboxes,
        track_poses,
        track_pose_confs,
        pred_instances,
        pred_idx,
        deltas_t,
        track_features,
        use_oks_n_similarities=True,
    ):
        detection_features = pred_instances["features"][pred_idx]
        N_DET, E_DET = detection_features.shape
        N_TRK, E_TRK = track_features.shape
        if N_DET == 0 or N_TRK == 0:
            return np.zeros((track_bboxes.shape[0], detection_features.shape[0]), dtype=np.float32)
        assert E_DET == E_TRK == self.prior_feature_size
        inputs = torch.concat((track_features, detection_features))
        data_samples = [dict(nb_entities=N_TRK), dict(nb_entities=N_DET)]
        similarities = self.re_id.predict(inputs, data_samples).detach().cpu().numpy()

        ious = self.get_tracks_preds_ious(track_bboxes, pred_instances, pred_idx)
        if not use_oks_n_similarities:
            return ious

        oks = self.get_tracks_preds_oks(
            track_bboxes,
            track_poses,
            track_pose_confs,
            pred_instances,
            pred_idx,
        )
        ious_ponderation = 1 - (self.alpha + self.beta)
        overlaps = ious > 0
        dists = (
            (self.alpha - self.alpha * deltas_t) * oks
            + (self.beta - self.beta * deltas_t) * similarities * overlaps
            + (ious_ponderation - ious_ponderation * deltas_t) * ious
        )

        return dists

    def forward(self, *args, **kwargs) -> Any:
        mode = kwargs.pop("mode", "predict")
        if mode == "predict":
            return self.predict(*args, **kwargs)
        elif mode == "loss":
            return self.loss(*args, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "Only supports loss and predict mode.")

    def loss(self, data_samples: List[List[PoseDataSample]]) -> dict:
        return self.re_id.loss(inputs=torch.tensor([]), data_samples=data_samples)

    def val_step(self, data_samples: list, *args, **kwargs) -> list:
        return self.test_step(data_samples, *args, **kwargs)

    def test_step(self, data_samples: list, *args, **kwargs) -> list:
        return self.re_id.test_step(inputs=torch.tensor([]), data_samples=data_samples, *args, **kwargs)

    def predict(
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
            matched_vec_mag,
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
            matched_vec_mag[:num_new_tracks] = np.zeros((num_new_tracks, 2), dtype=np.float32)
            matched_features[idx_counter : idx_counter + num_new_tracks] = det_features[torch.from_numpy(matched_pred_idx_tmp)]
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
                track_vec_mag,
                track_pose_confs,
                deltas_t,
                keep_trying,
                features,
            ) = self._tracks_to_pred_bboxes_kpts(tracks, confirmed_ids)

            dists = self.get_cost_matrix(
                track_bboxes,
                track_poses,
                track_pose_confs,
                pred_instances,
                remaining_conf_idx,
                deltas_t,
                features,
                use_oks_n_similarities=True,
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
            matched_vec_mag[:num_matches] = track_vec_mag[matched_tracks]
            matched_trk_bboxes[:num_matches] = track_bboxes[matched_tracks]
            matched_trk_ids[:num_matches] = confirmed_ids[matched_tracks]
            matched_features[:num_matches] = det_features[torch.from_numpy(matched_pred_idx_tmp)]

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
                    track_vec_mag,
                    track_pose_confs,
                    deltas_t,
                    _,
                    features,
                ) = self._tracks_to_pred_bboxes_kpts(tracks, unconfirmed_ids)

                dists = self.get_cost_matrix(
                    track_bboxes,
                    track_poses,
                    track_pose_confs,
                    pred_instances,
                    remaining_conf_idx,
                    deltas_t,
                    features,
                    use_oks_n_similarities=True,
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
                matched_vec_mag[idx_counter : idx_counter + num_matches] = track_vec_mag[matched_tracks]
                matched_features[idx_counter : idx_counter + num_matches] = det_features[torch.from_numpy(matched_pred_idx_tmp)]

                remaining_conf_idx[matched_pred_idx_tmp] = False
                idx_counter += num_matches

            # 3. Third match: low confidence detections -> remaining confirmed tracks
            if remaining_alive_confirmed_trk_ids.size > 0:
                (
                    track_bboxes,
                    track_poses,
                    track_vec_mag,
                    track_pose_confs,
                    deltas_t,
                    _,
                    features,
                ) = self._tracks_to_pred_bboxes_kpts(tracks, remaining_alive_confirmed_trk_ids)
                dists = self.get_cost_matrix(
                    track_bboxes,
                    track_poses,
                    track_pose_confs,
                    pred_instances,
                    remaining_not_conf_idx,
                    deltas_t,
                    features,
                    use_oks_n_similarities=False,
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
                matched_vec_mag[idx_counter : idx_counter + num_matches] = track_vec_mag[matched_tracks]
                matched_features[idx_counter : idx_counter + num_matches] = det_features[torch.from_numpy(matched_pred_idx_tmp)]

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
            matched_vec_mag[idx_counter : idx_counter + num_new_tracks] = np.zeros((num_new_tracks, 2), dtype=np.float32)
            matched_features[idx_counter : idx_counter + num_new_tracks] = det_features[torch.from_numpy(new_tracks_idx)]

            idx_counter += num_new_tracks

        self.save_to_ds(
            data_samples=data_sample,
            detections=pred_instances,
            track_ids=matched_trk_ids[:idx_counter],
            pred_idx=matched_pred_idx[:idx_counter],
            predicted_bboxes=matched_trk_bboxes[:idx_counter],
            velocities=matched_vec_mag[:idx_counter],
            features=matched_features[:idx_counter],
        )
