from typing import List, Tuple, Optional, Dict, Tuple
import torch
import numpy as np
import heapq
from collections import defaultdict, deque
import cv2
from scipy import stats

from .base_validation import BaseValidation


class AppearanceValidation(BaseValidation):

    def __init__(
        self,
        re_identificator: dict,
        batch_size: int,
        input_shape: Optional[Tuple] = (224, 224),
        validated_classes: Optional[List[str]] = None,
        memory_length: Optional[int] = 20,
        min_required_proof_to_confirm: Optional[int] = 10,
        confidence_level: Optional[float] = 0.95,
        *args,
        **kwargs,
    ) -> None:
        self._frame_size = None
        assert batch_size > 0
        self.batch_size = int(batch_size)
        self.did_not_check_since = defaultdict(dict)
        self.max_check_delay = 1000
        self.proofs = dict()
        self.memory = dict()

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

    def _get_validations(self, priorities, frame, to_switch):
        inputs = []
        tracked_inst_ids = []
        for _, _ in zip(priorities, range(self.batch_size)):
            cls, instance_id, cxcywh = heapq.heappop(priorities)[1]
            if cls not in to_switch:
                to_switch[cls] = []
            if 0 < instance_id:
                half_w = cxcywh[2] / 2
                half_h = cxcywh[3] / 2
                x1 = int(cxcywh[0] - half_w)
                y1 = int(cxcywh[1] - half_h)
                x2 = int(cxcywh[0] + half_w)
                y2 = int(cxcywh[1] + half_h)
                inputs.append(cv2.resize(frame[y1:y2, x1:x2], self.input_shape))
            tracked_inst_ids.append((str(cls), int(instance_id)))
        inputs = torch.stack(inputs)
        return self.re_identificator(inputs), tracked_inst_ids

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
                if inst_id not in self.did_not_check_since[cls]:
                    self.did_not_check_since[cls][inst_id] = 0
                did_not_check_since = self.did_not_check_since[cls][inst_id] / self.max_check_delay
                heapq.heappush(priority_queue, (-float(did_not_check_since + score), (cls, int(inst_id), cxcywh.tolist())))
        return priority_queue

    def _register_no_checks(self, checked):
        for cls in self.did_not_check_since:
            for inst_id in self.did_not_check_since[cls]:
                was_checked = isinstance(checked.get(cls), None)
                if was_checked:
                    was_checked = isinstance(checked[cls].get(inst_id), None)
                if was_checked:
                    self.did_not_check_since[cls][inst_id] = 0
                elif inst_id < self.max_check_delay:
                    self.did_not_check_since[cls][inst_id] += 1

    def _update_memory_collect_proof(self, cls, tracked_inst_id, validated_inst_id):
        unique_key = f"{cls}_{tracked_inst_id}"
        if unique_key not in self.memory:
            self.memory[unique_key] = deque([validated_inst_id], maxlen=self.memory)
            self.proofs[unique_key] = np.zeros(self.nb_classes)
        else:
            unique_id_memory = self.memory[unique_key]
            if len(unique_id_memory) == self.memory_length:
                forgetting = unique_id_memory.popleft()
                self.proofs[unique_key][forgetting] -= 1

            unique_id_memory.append(validated_inst_id)
            self.proofs[unique_key][validated_inst_id] += 1

    def _get_confirmation(self, cls, tracked_inst_id):
        """
        Determines if there's statistically significant evidence to confirm an identity.

        Uses a binomial proportion test to check if the proportion of evidence for the
        leading candidate is significantly higher than would be expected by chance,
        at the specified confidence level.

        Returns True only if:
        1. We have enough total evidence (min_required_proof_to_confirm)
        2. The leading candidate has significantly more evidence than alternatives
        """
        unique_key = f"{cls}_{tracked_inst_id}"
        proofs = self.proofs[unique_key]
        total_proofs = proofs.sum()

        # Need minimum amount of evidence before we can be statistically confident
        if total_proofs < self.min_required_proof_to_confirm:
            return False

        # Get the candidate with most evidence
        max_proof = proofs.max()

        # At least 2 observations needed for the leading candidate
        if max_proof < 2:
            return False

        # Number of competing hypotheses (non-zero proof counts)
        num_candidates = (proofs > 0).sum()

        # If only one candidate, confirm if we have enough evidence
        if num_candidates == 1:
            return True

        # Expected proportion under null hypothesis (uniform distribution)
        null_proportion = 1.0 / num_candidates
        observed_proportion = max_proof / total_proofs

        # Two-tailed binomial test: is the observed proportion significantly
        # different from what we'd expect by chance?
        # Using normal approximation to binomial (valid when n*p > 5 and n*(1-p) > 5)
        n = int(total_proofs)
        p_null = null_proportion

        # Calculate z-score for the proportion test
        expected = n * p_null
        std_error = np.sqrt(n * p_null * (1 - p_null))

        if std_error == 0:
            # If no variance (shouldn't happen), fall back to simple majority
            return observed_proportion > 0.5

        z_score = (max_proof - expected) / std_error

        # One-tailed test: we only care if it's significantly GREATER
        # Convert confidence level to z-critical value
        z_critical = stats.norm.ppf(self.confidence_level)

        return z_score > z_critical

    def __call__(
        self,
        frame: np.ndarray,
        tracking_results: Optional[dict] = None,
    ) -> Optional[Dict[str, List[Tuple]]]:
        self._init_validation(tracking_results)
        track_instances = tracking_results["pred_track_instances"]
        priorities = self._build_priority_queue(track_instances)
        to_switch = {}
        insts_id = track_instances["instances_id"]

        validated_inst_ids, tracked_inst_ids = self._get_validations(priorities, frame, to_switch)

        for validated_inst_id, tracked_inst_id in zip(validated_inst_ids, tracked_inst_ids):
            validated_cls, validated_inst_id = validated_inst_id
            validated_inst_id = int(validated_inst_id)
            tracked_cls, tracked_inst_id = tracked_inst_id

            if tracked_cls == validated_cls:
                self._update_memory_collect_proof(tracked_cls, tracked_inst_id, validated_inst_id)
                is_confirmed = self._get_confirmation(tracked_cls, tracked_inst_id)

                if (
                    is_confirmed
                    and validated_inst_id != tracked_inst_id
                    and not self._switching_back(to_switch[tracked_cls], tracked_inst_id, validated_inst_id)
                ):
                    to_switch[tracked_cls].append((tracked_inst_id, validated_inst_ids))
                    mask_a = insts_id == tracked_inst_id
                    mask_b = insts_id == validated_inst_ids
                    insts_id[mask_a] = validated_inst_ids
                    insts_id[mask_b] = tracked_inst_id
                    self._register_correction(tracking_results, tracked_inst_id, validated_inst_ids)
        tracking_results["corrected_instances_id"] = to_switch
        return tracking_results, to_switch

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
