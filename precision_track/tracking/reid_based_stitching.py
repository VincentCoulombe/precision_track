from typing import Dict, List, Optional

import numpy as np
import torch
from mmengine.logging import print_log

from precision_track.registry import TRACKING
from precision_track.models.runtimes.base import InferenceOnlyRuntime
from precision_track.utils import linear_assignment, parse_pose_metainfo

from .base import BaseStitchingAlgorithm


@TRACKING.register_module()
class ReIDBasedStitching(BaseStitchingAlgorithm):

    def __init__(
        self,
        metafile: str,
        capped_classes: Dict[str, int],
        reid_model: InferenceOnlyRuntime,
        verbose: Optional[bool] = True,
        **kwargs,
    ):
        """Stitch fragmented trajectories using a ReID model to match unconfirmed
        tracks to known identities from lost confirmed tracks.

        For each unconfirmed track, the ReID model predicts identity scores.
        Identities belonging to currently visible confirmed tracks are filtered out,
        then the remaining identities are assigned to unconfirmed tracks via
        linear assignment.

        Args:
            metafile (str): The metainfo for the skeletons of the tracks.
            capped_classes (Dict[str, int]): The classes whose tracks will be stitched.
                Instance IDs for these tracks will never exceed the integer cap.
            reid_model (InferenceOnlyRuntime): The ReID model, called as
                ``reid_model(features, [])`` and returning ``(..., logits)`` where
                logits has shape ``(batch, n_identities)``.
            verbose (bool): Whether to log initialization messages.
        """
        super().__init__(**kwargs)
        self.verbose = verbose
        metadata = parse_pose_metainfo({"from_file": metafile})
        assert "classes" in metadata, "The metadata must contain a list of the tracked classes."
        classes = metadata["classes"]

        assert isinstance(capped_classes, dict)
        self.capped_classes = {}
        for capped_cls, cap in capped_classes.items():
            if capped_cls in classes:
                if 0 <= cap:
                    self.capped_classes[capped_cls] = np.zeros(cap, dtype=bool)
                    if self.verbose:
                        print_log(f"The system will be tracking a maximum of {cap} {capped_cls}.", logger="current")
                elif self.verbose:
                    print_log(
                        f"Subjects classified as: '{capped_cls}' will be tracked without a cap, since you registered {cap} subjects in your config.",
                        logger="current",
                    )
            elif self.verbose:
                print_log(
                    f"Subjects classified as: '{capped_cls}' will be tracked without a cap, since it is not in the metainfo's list of classes: {classes}.",
                    logger="current",
                )

        self.reid_model = reid_model

        # cls -> {inst_id -> identity_idx}.
        self.inst_id_to_identity_idx: Dict[str, Dict[int, int]] = {cls: {} for cls in self.capped_classes}

    def __call__(
        self,
        tracks: dict,
        data_samples: dict,
        confirmed_ids: List[int],
        unconfirmed_ids: List[int],
    ):
        super().__call__(tracks, data_samples["pred_track_instances"])

        # Mask all unconfirmed tracks; they will be unmasked only if successfully stitched.
        for id_ in unconfirmed_ids:
            self.__masktrack__(id_)

        if not unconfirmed_ids:
            return

        # Reset occupancy and categorise confirmed tracks per class.
        for cls in self.capped_classes:
            self.capped_classes[cls][:] = False

        # cls -> set of inst_ids held by visible confirmed tracks
        alive_inst_ids: Dict[str, set] = {cls: set() for cls in self.capped_classes}
        # cls -> {inst_id: track_id} for lost confirmed tracks (candidates for reassignment)
        lost_tracks: Dict[str, Dict[int, int]] = {cls: {} for cls in self.capped_classes}

        for id_ in confirmed_ids:
            track = tracks[id_]
            cls = track.classes
            if cls not in self.capped_classes:
                continue
            inst_id = int(track.instances_id)
            cap = len(self.capped_classes[cls])
            if 0 < inst_id <= cap:
                self.capped_classes[cls][inst_id - 1] = True
                if not track.lost:
                    alive_inst_ids[cls].add(inst_id)
                else:
                    lost_tracks[cls][inst_id] = id_

        # For each capped class, run ReID on unconfirmed tracks and stitch them.
        for cls in self.capped_classes:
            # Collect unconfirmed tracks of this class that have feature vectors.
            unconf_ids_cls = []
            unconf_feats = []
            for id_ in unconfirmed_ids:
                if tracks[id_].classes != cls:
                    continue
                idx_mask = self.track_instances["ids"] == id_
                if not np.any(idx_mask):
                    continue
                feats = self.track_instances.get("features")
                if feats is None or (isinstance(feats, torch.Tensor) and feats.numel() == 0):
                    continue
                unconf_ids_cls.append(id_)
                unconf_feats.append(feats[idx_mask])  # shape (1, feat_dim)

            if not unconf_ids_cls:
                continue

            # Determine which inst_ids are available: either from lost confirmed tracks
            # or from slots not occupied by any confirmed track at all.
            available_inst_ids = list(lost_tracks[cls].keys())
            for i, occupied in enumerate(self.capped_classes[cls]):
                if not occupied:
                    available_inst_ids.append(i + 1)

            if not available_inst_ids:
                continue  # Every slot is held by an alive confirmed track.

            # Run ReID on the batch of unconfirmed tracks.
            batch = torch.cat(unconf_feats, dim=0)  # (n_unconf, feat_dim)
            outputs = self.reid_model(batch, [])
            logits = outputs[-1]  # (n_unconf, n_identities)
            scores = logits.softmax(dim=-1).cpu().float().numpy()  # (n_unconf, n_identities)
            n_identities = scores.shape[1]

            # Filter out identities already claimed by confirmed+alive tracks.
            taken_identity_idxs = set()
            for inst_id in alive_inst_ids[cls]:
                if inst_id in self.inst_id_to_identity_idx[cls]:
                    taken_identity_idxs.add(self.inst_id_to_identity_idx[cls][inst_id])

            available_identity_idxs = [i for i in range(n_identities) if i not in taken_identity_idxs]
            if not available_identity_idxs:
                continue

            # Build cost matrix: rows = unconfirmed tracks, cols = available identities.
            avail_arr = np.array(available_identity_idxs, dtype=int)
            cost_matrix = (1.0 - scores[:, avail_arr]).astype(np.float64)

            # Assign without threshold — false positives are corrected by the validator.
            matched_unconf_idxs, matched_avail_idxs = linear_assignment(cost_matrix, thresh=None)

            # Build reverse map: identity_idx -> inst_id from known prior assignments.
            identity_idx_to_inst_id = {v: k for k, v in self.inst_id_to_identity_idx[cls].items()}

            remaining_available = list(available_inst_ids)  # mutable copy for cold-start fallback

            for unconf_i, avail_j in zip(matched_unconf_idxs, matched_avail_idxs):
                identity_idx = available_identity_idxs[int(avail_j)]
                track_id = unconf_ids_cls[int(unconf_i)]

                # Resolve which inst_id this identity corresponds to.
                if identity_idx in identity_idx_to_inst_id:
                    inst_id = identity_idx_to_inst_id[identity_idx]
                    if inst_id not in remaining_available:
                        # The known slot is taken by an alive track — skip.
                        continue
                else:
                    # No prior mapping (cold start): claim the first available slot.
                    if not remaining_available:
                        continue
                    inst_id = remaining_available[0]

                remaining_available.remove(inst_id)
                self.__setinstid__(track_id, inst_id)
                self.capped_classes[cls][inst_id - 1] = True
                self.inst_id_to_identity_idx[cls][inst_id] = identity_idx

                # Remove the displaced lost confirmed track so it does not interfere.
                if inst_id in lost_tracks[cls]:
                    lost_track_id = lost_tracks[cls][inst_id]
                    if lost_track_id in tracks:
                        del tracks[lost_track_id]
