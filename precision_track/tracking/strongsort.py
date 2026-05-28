from typing import List, Optional

import cv2
import numpy as np
import torch

from precision_track.models.backends import ReIDBackend
from precision_track.registry import MODELS, TRACKING
from precision_track.utils import crop_bbox

from .byte_track import ByteTrack


@TRACKING.register_module()
class StrongSORT(ByteTrack):

    def __init__(
        self,
        re_identificator: dict,
        data_preprocessor: dict,
        obj_score_thrs: dict = dict(high=0.6, low=0.1),
        weight_iou_with_det_scores: bool = True,
        match_iou_thrs: dict = dict(high=0.9, low=0.5, tentative=0.9),
        init_track_thr: float = 0.8,
        appearance_weight: float = 0.25,
        appearance_ema: float = 0.1,
        **kwargs,
    ):
        """StrongSORT: an adapted Deep OC-SORT. A ByteTrack-style IoU cascade
        whose high-confidence association cost is fused with a re-identification
        appearance cue.

        Each detection is cropped from the raw frame and pushed through a
        ``ReIDBackend``, which returns per-crop scores over a fixed set of known
        identities. Each track keeps an exponential moving average (EMA) of
        those identity-probability vectors; the appearance affinity between a
        track and a detection is the dot product of their probability vectors.

        Args:
            re_identificator (dict): The ``ReIDBackend`` kwargs (``checkpoint``,
                ``metainfo``) -- the ``re_identificator`` block of appearance.yaml.
            data_preprocessor (dict): Config of the crop preprocessor (built via
                the ``MODELS`` registry, e.g. ``WildLifeReIDPreprocessor``).
            obj_score_thrs (dict, optional): See ByteTrack. Defaults to dict(high=0.6, low=0.1).
            weight_iou_with_det_scores (bool, optional): See ByteTrack. Defaults to True.
            match_iou_thrs (dict, optional): See ByteTrack. Defaults to dict(high=0.9, low=0.5, tentative=0.9).
            init_track_thr (float, optional): See ByteTrack. Defaults to 0.8.
            appearance_weight (float, optional): Weight of the appearance cue in
                the fused high-confidence cost; the IoU keeps ``1 - weight``. Defaults to 0.25.
            appearance_ema (float, optional): EMA strength for new observations
                when updating a track's identity-probability vector. Defaults to 0.1.
        """
        super().__init__(
            obj_score_thrs=obj_score_thrs,
            weight_iou_with_det_scores=weight_iou_with_det_scores,
            match_iou_thrs=match_iou_thrs,
            init_track_thr=init_track_thr,
            **kwargs,
        )

        assert 0.0 <= appearance_weight <= 1.0
        self.appearance_weight = float(appearance_weight)
        assert 0.0 < appearance_ema < 1.0
        self.appearance_ema = float(appearance_ema)

        self.re_identificator = ReIDBackend(**re_identificator)
        self.crop_enlargement_factor = self.re_identificator.crop_enlargement_factor
        self.data_preprocessor = MODELS.build(data_preprocessor)
        self.img_size = self.re_identificator.input_shape[0][-2:]
        self.device = self.re_identificator.device
        self.identities = self.re_identificator.identities
        self.nb_identities = len(self.identities)

        # StrongSORT-owned appearance state: track id -> EMA identity-prob vector.
        self.track_identity_ema = {}

    def _crop(self, img: np.ndarray, cxcywh: np.ndarray, max_w: int, max_h: int) -> Optional[torch.Tensor]:
        """Crop one detection and return a preprocessed (3, H, W) tensor.

        Returns None when the (enlarged, clipped) box is degenerate.
        """
        crop = crop_bbox(img, cxcywh, max_w, max_h, self.crop_enlargement_factor)
        if crop is None:
            return crop
        crop = cv2.resize(crop, self.img_size)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.data_preprocessor(crop)

    def _detection_identity_probs(self, data_sample: dict, pred_instances: dict, num_dets: int) -> np.ndarray:
        """Return a (num_dets, I) array of identity probabilities per detection.

        Detections without a usable crop -- and every detection when no raw
        frame is available -- get a uniform 1/I row, leaving the appearance cue
        neutral so the tracker degrades gracefully to plain ByteTrack.
        """
        uniform = 1.0 / self.nb_identities
        probs = np.full((num_dets, self.nb_identities), uniform, dtype=np.float32)

        img = data_sample.get("img", None)
        if img is None or num_dets == 0:
            return probs

        max_h, max_w = img.shape[:2]
        bboxes = pred_instances["bboxes"]
        crops = []
        valid = np.zeros(num_dets, dtype=bool)
        for i in range(num_dets):
            crop = self._crop(img, bboxes[i], max_w, max_h)
            if crop is not None:
                crops.append(crop)
                valid[i] = True

        if crops:
            batch = torch.stack(crops).to(self.device)
            _, logits = self.re_identificator(batch, [])
            probs[valid] = logits.softmax(1).detach().cpu().float().numpy()
        return probs

    def _appearance_affinity(self, track_ids: List[int], det_identity_probs: np.ndarray, pred_idx: np.ndarray) -> np.ndarray:
        """Track-vs-detection appearance affinity = dot product of identity-prob
        vectors. Tracks without an EMA yet contribute a uniform vector."""
        det_probs = det_identity_probs[pred_idx]
        num_tracks = len(track_ids)
        num_dets = det_probs.shape[0]
        if num_tracks == 0 or num_dets == 0:
            return np.zeros((num_tracks, num_dets), dtype=np.float32)
        uniform = np.full(self.nb_identities, 1.0 / self.nb_identities, dtype=np.float32)
        trk_probs = np.stack([self.track_identity_ema.get(int(tid), uniform) for tid in track_ids])
        return (trk_probs @ det_probs.T).astype(np.float32)

    def _fuse(self, ious: np.ndarray, track_ids: List[int], det_identity_probs: np.ndarray, pred_idx: np.ndarray) -> np.ndarray:
        """Fuse IoU with the appearance affinity; appearance is gated by IoU > 0
        so it never creates a match without spatial overlap."""
        if ious.size == 0:
            return ious
        appearance = self._appearance_affinity(track_ids, det_identity_probs, pred_idx)
        w = self.appearance_weight
        return (1.0 - w) * ious + w * appearance * (ious > 0)

    def _prune_emas(self, tracks: dict) -> None:
        """Drop EMA entries for tracks that no longer exist."""
        self.track_identity_ema = {tid: ema for tid, ema in self.track_identity_ema.items() if tid in tracks}

    def _update_emas(self, track_ids: np.ndarray, pred_idx: np.ndarray, det_identity_probs: np.ndarray) -> None:
        """Update each matched/new track's identity-probability EMA."""
        a = self.appearance_ema
        for tid, det_i in zip(track_ids, pred_idx):
            tid = int(tid)
            p = det_identity_probs[int(det_i)]
            if tid in self.track_identity_ema:
                self.track_identity_ema[tid] = (1.0 - a) * self.track_identity_ema[tid] + a * p
            else:
                self.track_identity_ema[tid] = p.copy()

    def __call__(
        self,
        data_sample: dict,
        tracks: dict,
        confirmed_ids: List[int],
        unconfirmed_ids: List[int],
        *args,
        **kwargs,
    ) -> None:
        """Associate new detections to existing or new tracks, fusing IoU with a
        re-identification appearance cue in the high-confidence match stages.

        Args:
            data_sample (dict): The new detections; modified in place. Carries
                the raw frame under the ``img`` key when available.
            tracks (dict): The defined tracks.
            confirmed_ids (List[int]): Ids of tracks old enough to be confirmed.
            unconfirmed_ids (List[int]): Ids of tracks too young to be asserted.
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

        self._prune_emas(tracks)
        det_identity_probs = self._detection_identity_probs(data_sample, pred_instances, num_dets)

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
            remaining_conf_idx = scores > self.obj_score_thrs["high"]
            remaining_not_conf_idx = (~remaining_conf_idx) & (scores > self.obj_score_thrs["low"])

            # 1. First match: high confidence detections -> confirmed tracks.
            track_bboxes = self._tracks_to_pred_bboxes(tracks, confirmed_ids)
            ious = self.get_tracks_preds_ious(track_bboxes, pred_instances, remaining_conf_idx)
            dists = self._fuse(ious, confirmed_ids, det_identity_probs, remaining_conf_idx)
            matched_tracks, matched_dets = self.assign_ids(
                dists=dists,
                tracks=tracks,
                track_ids=confirmed_ids,
                pred_instances=pred_instances,
                pred_idx=remaining_conf_idx,
                weight_iou_with_det_scores=self.weight_iou_with_det_scores,
                match_iou_thr=self.match_iou_thrs["high"],
            )

            matched_pred_idx_tmp = np.where(remaining_conf_idx)[0][matched_dets]
            num_matches = len(matched_pred_idx_tmp)

            matched_pred_idx[:num_matches] = matched_pred_idx_tmp
            matched_features[:num_matches] = det_features[matched_pred_idx_tmp]
            matched_trk_bboxes[:num_matches] = track_bboxes[matched_tracks]
            matched_trk_ids[:num_matches] = np.array(confirmed_ids)[matched_tracks]

            idx_counter = num_matches
            remaining_conf_idx[matched_pred_idx_tmp] = False

            remaining_trk_ids = np.setdiff1d(confirmed_ids, matched_trk_ids[:idx_counter]).tolist()
            remaining_alive_confirmed_trk_ids = [id for id in remaining_trk_ids if tracks[id].frame_ids[-1] == frame_id - 1]

            # 2. Second match: remaining high confidence detections -> unconfirmed tracks.
            if unconfirmed_ids:
                track_bboxes = self._tracks_to_pred_bboxes(tracks, unconfirmed_ids)
                ious = self.get_tracks_preds_ious(track_bboxes, pred_instances, remaining_conf_idx)
                dists = self._fuse(ious, unconfirmed_ids, det_identity_probs, remaining_conf_idx)
                matched_tracks, matched_dets = self.assign_ids(
                    dists=dists,
                    tracks=tracks,
                    track_ids=unconfirmed_ids,
                    pred_instances=pred_instances,
                    pred_idx=remaining_conf_idx,
                    weight_iou_with_det_scores=self.weight_iou_with_det_scores,
                    match_iou_thr=self.match_iou_thrs["tentative"],
                )

                matched_pred_idx_tmp = np.where(remaining_conf_idx)[0][matched_dets]
                num_matches = len(matched_pred_idx_tmp)

                matched_trk_ids[idx_counter : idx_counter + num_matches] = np.array(unconfirmed_ids)[matched_tracks]
                matched_pred_idx[idx_counter : idx_counter + num_matches] = matched_pred_idx_tmp
                matched_trk_bboxes[idx_counter : idx_counter + num_matches] = track_bboxes[matched_tracks]
                matched_features[idx_counter : idx_counter + num_matches] = det_features[matched_pred_idx_tmp]

                remaining_conf_idx[matched_pred_idx_tmp] = False
                idx_counter += num_matches

            # 3. Third match: low confidence detections -> remaining confirmed tracks (IoU only).
            if remaining_alive_confirmed_trk_ids:
                track_bboxes = self._tracks_to_pred_bboxes(tracks, remaining_alive_confirmed_trk_ids)
                ious = self.get_tracks_preds_ious(track_bboxes, pred_instances, remaining_not_conf_idx)
                matched_tracks, matched_dets = self.assign_ids(
                    dists=ious,
                    tracks=tracks,
                    track_ids=remaining_alive_confirmed_trk_ids,
                    pred_instances=pred_instances,
                    pred_idx=remaining_not_conf_idx,
                    weight_iou_with_det_scores=False,
                    match_iou_thr=self.match_iou_thrs["low"],
                )

                matched_pred_idx_tmp = np.where(remaining_not_conf_idx)[0][matched_dets]
                num_matches = len(matched_pred_idx_tmp)

                matched_trk_ids[idx_counter : idx_counter + num_matches] = np.array(remaining_alive_confirmed_trk_ids)[matched_tracks]
                matched_pred_idx[idx_counter : idx_counter + num_matches] = matched_pred_idx_tmp
                matched_trk_bboxes[idx_counter : idx_counter + num_matches] = track_bboxes[matched_tracks]
                matched_features[idx_counter : idx_counter + num_matches] = det_features[matched_pred_idx_tmp]

                idx_counter += num_matches

            # 4. Remaining high confidence detections -> new tracks.
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

        self._update_emas(matched_trk_ids[:idx_counter], matched_pred_idx[:idx_counter], det_identity_probs)

        self.save_to_ds(
            data_samples=data_sample,
            detections=pred_instances,
            track_ids=matched_trk_ids[:idx_counter],
            pred_idx=matched_pred_idx[:idx_counter],
            predicted_bboxes=matched_trk_bboxes[:idx_counter],
            features=matched_features[:idx_counter],
        )
