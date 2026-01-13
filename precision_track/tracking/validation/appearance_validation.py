from typing import List, Tuple, Optional, Dict
import torch
import numpy as np
import heapq
import cv2
import re

from .base_validation import BaseValidation

from precision_track.utils import cosine_similarity


class AppearanceValidation(BaseValidation):
    _UNIQUE_ID_PATTERN = re.compile(r"^(.+)_(\d+)$")

    def __init__(
        self,
        re_identificator: dict,
        batch_size: int,
        input_shape: Optional[Tuple] = (224, 224),
        validated_classes: Optional[List[str]] = None,
        memory_length: Optional[int] = 20,
        min_required_proof_to_confirm: Optional[int] = 10,
        confidence_level: Optional[float] = 0.95,
        features_ema: Optional[float] = 0.1,
        features_bank_size: Optional[int] = 1000,
        nb_features: Optional[int] = 128,
        *args,
        **kwargs,
    ) -> None:
        self._frame_size = None
        assert batch_size > 0
        self.batch_size = int(batch_size)

        assert len(input_shape) == 2
        self.input_shape = []
        for shape in input_shape:
            assert shape > 0
            self.input_shape.append(int(shape))
        self.input_shape = tuple(input_shape)

        if validated_classes is not None:
            assert isinstance(validated_classes, list)
            for cls in validated_classes:
                assert isinstance(cls, str)
        self.validated_classes = validated_classes

        self.re_identificator = ReIDBackend(**re_identificator)

        assert memory_length > 0
        self.memory_length = int(memory_length)

        assert min_required_proof_to_confirm > 0
        self.min_required_proof_to_confirm = int(min_required_proof_to_confirm)

        assert 0 < confidence_level < 1
        self.confidence_level = float(confidence_level)

        assert 0 < features_ema < 1
        self.strength_ema_new = float(features_ema)
        self.stregth_ema_baseline = 1 - self.strength_ema_new

        assert 0 < nb_features
        self.nb_features = int(nb_features)
        assert 0 < features_bank_size
        self.features_bank_size = features_bank_size

        self.alpha_counts = torch.zeros((self.features_bank_size, self.features_bank_size), dtype=torch.float32)
        self.beta_counts = torch.zeros((self.features_bank_size, self.features_bank_size), dtype=torch.float32)
        self.observation_count = torch.zeros(self.features_bank_size, dtype=torch.int32)
        self.did_not_check_since = torch.zeros(self.features_bank_size, dtype=torch.int32)
        self.features_bank = torch.zeros((self.features_bank_size, self.nb_features), dtype=torch.float32)
        self.unique_ids = np.zeros(self.features_bank_size, dtype=str)
        self.occupied_idxs = torch.zeros(self.features_bank_size, dtype=bool)
        self.epsilon_base = 0.25
        self.max_check_delay = 1000

    def _update_features_bank(self, priorities: heapq, frame: np.ndarray, to_switch: dict):
        inputs = []
        tracked_unique_ids = []
        for _, _ in zip(priorities, range(self.batch_size)):
            cls, instance_id, cxcywh = heapq.heappop(priorities)[1]
            if cls not in to_switch:
                to_switch[cls] = []
            unique_key = f"{cls}_{instance_id}"
            if 0 < instance_id:
                half_w = cxcywh[2] / 2
                half_h = cxcywh[3] / 2
                x1 = int(cxcywh[0] - half_w)
                y1 = int(cxcywh[1] - half_h)
                x2 = int(cxcywh[0] + half_w)
                y2 = int(cxcywh[1] + half_h)
                inputs.append(cv2.resize(frame[max(y1, 0) : min(y2, self.frame_size[1]), max(x1, 0) : min(x2, self.frame_size[0])], self.input_shape))
            tracked_unique_ids.append(unique_key)

        if len(inputs) == 0:
            return [], tracked_unique_ids

        extracted_features = self.re_identificator(inputs)
        updated_idxs = torch.zeros_like(self.occupied_idxs)
        for i, tracked_unique_id in enumerate(tracked_unique_ids):
            tracked_idx = np.where(tracked_unique_id == self.unique_ids)[0]
            if not tracked_idx.any():
                tracked_idx = self._find_first_available_idx()
                self.features_bank[tracked_idx] = extracted_features[i]
            else:
                self.features_bank[tracked_idx] = self.features_bank[tracked_idx] * self.stregth_ema_baseline + extracted_features[i] * self.strength_ema_new
            updated_idxs[tracked_idx] = True

        similarities = cosine_similarity(self.features_bank[updated_idxs], self.features_bank[self.occupied_idxs])
        return similarities, updated_idxs

    def _find_first_available_idx(self):
        available_idxs = torch.where(~self.occupied_idxs)[0]
        if len(available_idxs) > 0:
            return available_idxs[0].item()
        raise ValueError(f"The Appearance Extractor's feature bank, which can contains up to {self.features_bank_size} concurrent entities, is full.")

    def _build_priority_queue(self, track_instances: dict) -> int:
        isolated = track_instances["isolated"]

        priority_queue = []
        for cls, inst_id, cxcywh, score in zip(
            track_instances["classes"][isolated],
            track_instances["instances_id"][isolated],
            track_instances["bboxes"][isolated],
            track_instances["scores"][isolated],
        ):
            if self.validated_classes is None or cls in self.validated_classes:
                unique_key = f"{cls}_{inst_id}"
                if unique_key not in self.did_not_check_since:
                    self.did_not_check_since[unique_key] = 0
                did_not_check_since = self.did_not_check_since[unique_key] / self.max_check_delay
                heapq.heappush(priority_queue, (-float(did_not_check_since + score), (cls, int(inst_id), cxcywh.tolist())))
        return priority_queue

    def _register_no_checks(self, updated_idxs):
        self.did_not_check_since[~updated_idxs] += 1

    def _reset(self, keys):
        for key in keys:
            self.alpha_counts[key, :] = 0.0
            self.beta_counts[key, :] = 0.0
            self.observation_count[key] = 0
            self.did_not_check_since[key] = 0
            self.features_bank[key] = 0.0
            self.occupied_idxs[key] = False

    def _update_memory_collect_proof(self, similarities, updated_idxs):
        """
        Update Beta distribution for each candidate ID.
        similarity_scores: normalized cosine similarities [N_ids]
        """

        # Compute observation weight based on prediction entropy (confidence)
        entropy = -np.sum(similarities * np.log(similarities + 1e-10))
        max_entropy = np.log(len(similarities))
        confidence = 1.0 - (entropy / max_entropy)  # [0,1], higher = more confident

        self.observation_count[updated_idxs] += 1

        # Add new observation weighted by confidence
        self.alpha_counts[self.occupied_idxs, updated_idxs] += similarities * confidence
        self.beta_counts[self.occupied_idxs, updated_idxs] += (1 - similarities) * confidence

    def _get_confirmations(self, updated_idxs):

        confirmable_idxs = self.observation_count[updated_idxs] > self.min_required_proof_to_confirm
        if not confirmable_idxs.any():
            return []

        alphas = self.alpha_counts[confirmable_idxs]
        betas = self.beta_counts[confirmable_idxs]
        posterior_means = alphas / (alphas + betas)

        top1_idxs = torch.argmax(posterior_means, dim=1)
        top1_means = posterior_means[:, top1_idxs].copy()

        uncertainty_penalties = (1.0 - top1_means) / np.sqrt(len(confirmable_idxs.shape[0]))
        epsilons = self.epsilon_base * (1.0 + uncertainty_penalties)

        posterior_means[:, top1_idxs] = -1
        top2_means = torch.max(posterior_means, dim=1)

        are_confirmed = top1_means > epsilons + top2_means

        return top1_idxs if are_confirmed.any() else []

    def _init_validation(self, tracking_results: dict):
        if "correction_instances" not in tracking_results:
            tracking_results["correction_instances"] = {"instances_id": [], "corrected_id": []}

    def __call__(
        self,
        frame: np.ndarray,
        tracking_results: Optional[dict] = None,
    ) -> Optional[Dict[str, List[Tuple]]]:
        self._init_validation(tracking_results)
        track_instances = tracking_results["pred_track_instances"]
        priorities = self._build_priority_queue(track_instances)
        to_switch = {}

        similarities, updated_idxs = self._update_features_bank(priorities, frame, to_switch)
        self._update_memory_collect_proof(similarities, updated_idxs)
        confirmed_idxs = self._get_confirmations(updated_idxs)
        for updated_idx, confirmed_idx in zip(updated_idxs, confirmed_idxs):
            updated_idx = updated_idx.item()
            confirmed_idx = confirmed_idx.item()

            updated_unique_id = self.unique_ids[updated_idx]
            confirmed_unique_id = self.unique_ids[confirmed_idx]

            updated_cls, updated_inst_id = self._decode_unique_id(updated_unique_id)
            confirmed_cls, confirmed_inst_id = self._decode_unique_id(confirmed_unique_id)

            if updated_cls == confirmed_cls:

                are_oscillating = self._switching_back(to_switch.get(updated_cls, []), updated_inst_id, confirmed_inst_id)
                if updated_inst_id != confirmed_inst_id and not are_oscillating:

                    to_switch[updated_cls].append((updated_inst_id, confirmed_inst_id))
                    mask_a = (track_instances["instances_id"] == updated_inst_id) & (track_instances["classes"] == updated_cls)
                    mask_b = (track_instances["instances_id"] == confirmed_inst_id) & (track_instances["classes"] == confirmed_cls)
                    if mask_a.any() and mask_b.any():
                        track_instances["instances_id"][mask_a] = confirmed_inst_id
                        track_instances["instances_id"][mask_b] = updated_inst_id
                        self._register_correction(tracking_results, updated_inst_id, confirmed_inst_id)

        self._reset([updated_idxs, confirmed_idxs] + [key for key, delay in self.did_not_check_since.items() if delay >= self.max_check_delay])

        self._register_no_checks(updated_idxs)
        tracking_results["corrected_instances_id"] = to_switch
        return tracking_results, to_switch

    @classmethod
    def _decode_unique_id(cls, unique_id):
        match = cls._UNIQUE_ID_PATTERN.match(unique_id)
        if match:
            cls_ = match.group(1)
            id_ = match.group(2)
            return str(cls_), int(id_)
        raise ValueError(f"The Appearance Extractor failed to decode the following unique id: {unique_id}.")

    @staticmethod
    def _switching_back(switches: List[tuple], a: int, b: int):
        for switch in switches:
            if switch[0] == b and switch[1] == a:
                return True
        return False

    @staticmethod
    def _register_correction(tracking_results: dict, ori_id: int, corrected_id: int):
        tracking_results["correction_instances"]["instances_id"].append(ori_id)
        tracking_results["correction_instances"]["corrected_id"].append(corrected_id)
