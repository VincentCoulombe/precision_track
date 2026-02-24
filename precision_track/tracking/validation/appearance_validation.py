from typing import List, Tuple, Optional, Dict
import torch
import numpy as np
import heapq
import re
import cv2
from mmengine.logging import print_log

from .base_validation import BaseValidation

from precision_track.registry import TRACKING, MODELS
from precision_track.utils import parse_pose_metainfo, clip, AppearanceClassifier
from precision_track.models.backends import ReIDBackend


@TRACKING.register_module()
class AppearanceValidation(BaseValidation):
    _UNIQUE_ID_PATTERN = re.compile(r"^(.+)_(\d+)$")
    _MAX_CONF = 0.9

    def __init__(
        self,
        metainfo: str,
        re_identificator: dict,
        data_preprocessor: dict,
        batch_size: int,
        unique_ids: list,
        validated_classes: List[str],
        input_shape: Optional[Tuple] = (224, 224),
        memory_length: Optional[int] = 20,
        min_consecutive_hits: Optional[int] = 5,
        confidence_level: Optional[float] = 0.95,
        features_ema: Optional[float] = 0.1,
        crop_enlargement_factor: Optional[float] = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(validated_classes)

        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.cls_to_labels = {cls_: i for i, cls_ in enumerate(metainfo.get("classes", []))}

        assert batch_size > 0
        self.batch_size = int(batch_size)

        assert len(input_shape) == 2
        self.input_shape = [3]
        for shape in input_shape:
            assert shape > 0
            self.input_shape.append(int(shape))
        self.img_size = tuple(self.input_shape[-2:])
        self.input_shape = tuple(self.input_shape)

        re_identificator["input_shape"] = [(-1,) + self.input_shape]

        self.re_identificator = ReIDBackend(**re_identificator)
        self.device = self.re_identificator.device
        self.identities = self.re_identificator.identities

        self.precision = torch.float16 if self.re_identificator.half_precision else torch.float32

        self.data_preprocessor = MODELS.build(data_preprocessor)

        assert memory_length > 0
        self.memory_length = int(memory_length)

        assert min_consecutive_hits > 0
        self.min_consecutive_hits = int(min_consecutive_hits)

        assert 0 < confidence_level < 1
        self.confidence_level = float(confidence_level)

        assert 0 < features_ema < 1
        self.strength_ema_new = float(features_ema)
        self.strength_ema_baseline = 1 - self.strength_ema_new

        crop_enlargement_factor = float(crop_enlargement_factor)
        assert 0.0 <= crop_enlargement_factor < 1.0
        self.crop_enlargement_factor = crop_enlargement_factor

        self.unique_ids_list = unique_ids
        print_log(
            f"Set to re-identify the following unique ids: {unique_ids}, based on their appearances.",
            logger="current",
        )

        assert len(self.identities) == len(
            unique_ids
        ), f"The AppearanceValidation module is set to re-identify {len(unique_ids)} distinct subjects ({unique_ids}) with an incoherant number of identities {len(self.identities)} ({self.identities})."
        self.nb_identities = len(unique_ids)

        self.has_been_observed = torch.zeros(self.nb_identities, dtype=torch.bool, device=self.device)
        self.consecutive_hits = torch.zeros(self.nb_identities, self.nb_identities, dtype=torch.int64, device=self.device)
        self.did_not_check_since = torch.zeros(self.nb_identities, dtype=torch.int64, device=self.device)
        self.identity_probabilities = torch.zeros((self.nb_identities, self.nb_identities), dtype=self.precision, device=self.device)

        self.unique_ids = {u: i for i, u in enumerate(self.unique_ids_list)}
        self.reverse_unique_ids = {i: u for i, u in enumerate(self.unique_ids_list)}
        self.identity2uid = dict()
        self.max_check_delay = 1000

    def _update(self, priorities: heapq, frame: np.ndarray, to_switch: dict):
        inputs = []
        updated_idxs = []
        tracked_conf = []
        qty_to_process = min(len(priorities), self.batch_size)
        for _ in range(qty_to_process):
            cls, instance_id, cxcywh, score = heapq.heappop(priorities)[1]
            if cls not in to_switch:
                to_switch[cls] = set()
            unique_key = f"{cls}_{instance_id}"
            w = cxcywh[2]
            h = cxcywh[3]
            w_enlargement = cxcywh[2] * self.crop_enlargement_factor
            h_enlargement = cxcywh[3] * self.crop_enlargement_factor
            cxcywh[2] = w + w_enlargement
            cxcywh[3] = h + h_enlargement

            clipped_coords = clip(np.array(cxcywh), "cxcywh", self.frame_size[1], self.frame_size[0])

            enlarged_half_w = clipped_coords[2] / 2
            enlarged_half_h = clipped_coords[3] / 2

            x1 = int(clipped_coords[0] - enlarged_half_w)
            y1 = int(clipped_coords[1] - enlarged_half_h)
            x2 = int(clipped_coords[0] + enlarged_half_w)
            y2 = int(clipped_coords[1] + enlarged_half_h)

            crop = cv2.resize(frame[y1:y2, x1:x2], self.img_size)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inputs.append(self.data_preprocessor(crop))
            updated_idxs.append(self.unique_ids[unique_key])
            tracked_conf.append(score)

        if len(inputs) == 0:
            return []

        features, logits = self.re_identificator(torch.stack(inputs).to(self.device), [])

        tensor_updated_idxs = torch.tensor(updated_idxs, dtype=torch.int64, device=self.device)
        tensor_tracked_conf = torch.tensor(tracked_conf, dtype=torch.float64, device=self.device)

        return self._ingest_updates(features=features, logits=logits, idxs=tensor_updated_idxs, scores=tensor_tracked_conf)

    def _ingest_updates(self, logits, idxs, *args, **kwargs):
        identity_probabilities = logits.softmax(1)

        first_observation = ~self.has_been_observed[idxs]

        observed = ~first_observation
        observed_idxs = idxs[observed]

        first_observation_idxs = idxs[first_observation]
        self.has_been_observed[first_observation_idxs] = True

        self.identity_probabilities[first_observation_idxs] = identity_probabilities[first_observation]

        self.identity_probabilities[observed_idxs] = (
            self.identity_probabilities[observed_idxs] * self.strength_ema_baseline + identity_probabilities[observed] * self.strength_ema_new
        )
        return observed_idxs

    def _build_priority_queue(self, track_instances: dict) -> list:
        isolated = track_instances["isolated"]

        priority_queue = []
        for cls, inst_id, cxcywh, score in zip(
            track_instances["classes"][isolated],
            track_instances["instances_id"][isolated],
            track_instances["bboxes"][isolated],
            track_instances["scores"][isolated],
        ):
            if inst_id >= 0 and (self.validated_classes is None or cls in self.validated_classes):
                unique_key = f"{cls}_{inst_id}"
                idx = self.unique_ids.get(unique_key)
                assert (
                    idx is not None
                ), f"The appearance validator encountered the following not registered unique id at runtime: {unique_key}. The registered unique ids are: {self.unique_ids_list}."
                did_not_check_since = self.did_not_check_since[idx] / self.max_check_delay
                heapq.heappush(priority_queue, (-float(did_not_check_since + score), (cls, int(inst_id), cxcywh.tolist(), score)))
        return priority_queue

    def _register_no_checks(self, updated_idxs):
        self.did_not_check_since += 1
        self.did_not_check_since[updated_idxs] = 0

    def _reset(self, keys):
        self.has_been_observed[keys] = 0
        self.consecutive_hits[keys] = 0.0
        self.did_not_check_since[keys] = 0

    def _get_confirmations(self, updated_idxs):
        updated_id_probs = self.identity_probabilities[updated_idxs, :]

        max_return = updated_id_probs.max(1)
        identity_idxs = max_return.indices
        updated_id_prob = max_return.values
        conf_mask = updated_id_prob > min(self.strength_ema_baseline, self._MAX_CONF)

        conf_identity_idxs = identity_idxs[conf_mask]
        confirmed_idxs = updated_idxs[conf_mask]

        keep = self.consecutive_hits[confirmed_idxs, conf_identity_idxs].clone()
        self.consecutive_hits[confirmed_idxs] = 0
        self.consecutive_hits[confirmed_idxs, conf_identity_idxs] = keep + 1

        hits_mask = self.consecutive_hits[confirmed_idxs, conf_identity_idxs] >= self.min_consecutive_hits
        self.consecutive_hits[confirmed_idxs[hits_mask], conf_identity_idxs[hits_mask]] = 0

        return confirmed_idxs, conf_identity_idxs, hits_mask

    def _update_and_get_confirmations(self, priorities, frame, to_switch):
        updated_idxs = self._update(priorities, frame, to_switch)
        if len(updated_idxs) == 0:
            return None
        return updated_idxs, *self._get_confirmations(updated_idxs)

    def _init_validation(self, tracking_results: dict):
        if "correction_instances" not in tracking_results:
            tracking_results["correction_instances"] = {"instances_id": [], "class_id": [], "corrected_id": []}
        if "appearance_validation_instances" not in tracking_results:
            tracking_results["appearance_validation_instances"] = {
                "labels": [],
                "instances_id": [],
                "identity": [],
            } | {f"{i}_score": [] for i in self.identities}

    def __call__(
        self,
        frame: np.ndarray,
        tracking_results: Optional[dict] = None,
    ) -> Optional[Dict[str, List[Tuple]]]:

        frame_id = tracking_results["img_id"]
        if frame_id in [585, 730]:
            stop = True

        if self._frame_size is None:
            self.frame_size = tracking_results["ori_shape"][:2]
        self._init_validation(tracking_results)
        track_instances = tracking_results["pred_track_instances"]
        priorities = self._build_priority_queue(track_instances)
        to_switch = {}

        confirmations = self._update_and_get_confirmations(priorities, frame, to_switch)

        if confirmations is None:
            self._register_no_checks(torch.tensor([], dtype=torch.int64, device=self.device))
            tracking_results["corrected_instances_id"] = to_switch
            return tracking_results, to_switch

        updated_idxs, confirmed_idxs, conf_identity_idxs, hits_mask = confirmations

        idx_to_reset = []

        for updated_idx, confirmed_identity_idx, is_a_hit in zip(confirmed_idxs, conf_identity_idxs, hits_mask):
            updated_idx = updated_idx.item()
            confirmed_identity_idx = confirmed_identity_idx.item()

            updated_unique_id = self.reverse_unique_ids[updated_idx]
            updated_cls, updated_inst_id = self._decode_unique_id(updated_unique_id)

            confirmed_identity = self.identities[confirmed_identity_idx]
            u_id_linked_to_conf_identity = self.identity2uid.get(confirmed_identity)

            self._update_appearance_validation_instances(
                updated_cls=updated_cls,
                updated_inst_id=updated_inst_id,
                confirmed_identity=confirmed_identity,
                updated_idx=updated_idx,
                tracking_results=tracking_results,
            )

            if not is_a_hit:  # The identity has not been confirmed enough to be a hit
                continue

            if u_id_linked_to_conf_identity is None:  # First confirmation
                self.identity2uid[confirmed_identity] = updated_unique_id
                continue

            confirmed_cls, confirmed_inst_id = self._decode_unique_id(u_id_linked_to_conf_identity)

            if updated_cls != confirmed_cls:  # Not same class
                continue

            are_oscillating = (confirmed_inst_id, updated_inst_id) in to_switch.get(updated_cls, set())

            if updated_inst_id == confirmed_inst_id or are_oscillating:  # No need to switch
                continue

            to_switch[updated_cls].add((updated_inst_id, confirmed_inst_id))
            mask_a = (track_instances["instances_id"] == updated_inst_id) & (track_instances["classes"] == updated_cls)
            mask_b = (track_instances["instances_id"] == confirmed_inst_id) & (track_instances["classes"] == confirmed_cls)
            if mask_a.any() and mask_b.any():
                track_instances["instances_id"][mask_a] = confirmed_inst_id
                track_instances["instances_id"][mask_b] = updated_inst_id
                self._register_correction(tracking_results, track_instances["labels"][mask_b], updated_inst_id, confirmed_inst_id)
                idx_to_reset.extend([updated_idx, self.unique_ids[u_id_linked_to_conf_identity]])

        idx_to_reset = torch.tensor(idx_to_reset, dtype=torch.int64, device=self.device)
        stale_idxs = torch.where(self.did_not_check_since >= self.max_check_delay)[0]
        self._reset(torch.cat([idx_to_reset, stale_idxs]))

        self._register_no_checks(updated_idxs)
        tracking_results["corrected_instances_id"] = to_switch
        return tracking_results, to_switch

    def _update_appearance_validation_instances(self, updated_cls, updated_inst_id, confirmed_identity, updated_idx, tracking_results):
        label = int(self.cls_to_labels.get(updated_cls, -1))
        tracking_results["appearance_validation_instances"]["labels"].append(label)
        tracking_results["appearance_validation_instances"]["instances_id"].append(updated_inst_id)
        tracking_results["appearance_validation_instances"]["identity"].append(confirmed_identity)
        for i, identity in enumerate(self.identities):
            tracking_results["appearance_validation_instances"][f"{identity}_score"].append(self.identity_probabilities[updated_idx, i].item())

    @classmethod
    def _decode_unique_id(cls, unique_id):
        match = cls._UNIQUE_ID_PATTERN.match(unique_id)
        if match:
            cls_ = match.group(1)
            id_ = match.group(2)
            return str(cls_), int(id_)
        raise ValueError(f"The Appearance Extractor failed to decode the following unique id: {unique_id}.")

    @staticmethod
    def _register_correction(tracking_results: dict, class_id: int, ori_id: int, corrected_id: int):
        tracking_results["correction_instances"]["instances_id"].append(ori_id)
        tracking_results["correction_instances"]["class_id"].append(class_id)
        tracking_results["correction_instances"]["corrected_id"].append(corrected_id)


@TRACKING.register_module()
class MetricBasedAppearanceValidation(AppearanceValidation):
    def __init__(
        self,
        metainfo: str,
        re_identificator: dict,
        data_preprocessor: dict,
        batch_size: int,
        unique_ids: list,
        validated_classes: List[str],
        input_shape: Optional[Tuple] = (224, 224),
        memory_length: Optional[int] = 20,
        min_consecutive_hits: Optional[int] = 5,
        confidence_level: Optional[float] = 0.95,
        features_ema: Optional[float] = 0.01,
        crop_enlargement_factor: Optional[float] = 0.0,
        max_appearance_classifier_size: Optional[int] = 1000,
        max_k: Optional[int] = 15,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            metainfo=metainfo,
            re_identificator=re_identificator,
            data_preprocessor=data_preprocessor,
            batch_size=batch_size,
            unique_ids=unique_ids,
            validated_classes=validated_classes,
            input_shape=input_shape,
            memory_length=memory_length,
            min_consecutive_hits=min_consecutive_hits,
            confidence_level=confidence_level,
            features_ema=features_ema,
            crop_enlargement_factor=crop_enlargement_factor,
            *args,
            **kwargs,
        )
        self.appearance_classifier = None
        self.identities = self.unique_ids_list
        self.max_appearance_classifier_size = max_appearance_classifier_size

        self.max_k = max_k

    def _ingest_updates(self, features, idxs, scores, *args, **kwargs):
        if self.appearance_classifier is None:
            feature_dim = features.shape[1]
            self.appearance_classifier = AppearanceClassifier(
                identities=range(self.nb_identities),
                features_size=feature_dim,
                device=self.device,
                precision=self.precision,
                max_size_per_id=self.max_appearance_classifier_size,
                k=self.max_k,
            )

        first_observation = ~self.has_been_observed[idxs]

        observed = ~first_observation
        observed_idxs = idxs[observed]

        first_observation_idxs = idxs[first_observation]
        self.has_been_observed[first_observation_idxs] = True

        if first_observation.any():
            _ = self.appearance_classifier(features[first_observation], first_observation_idxs, scores[first_observation])

        if len(observed_idxs) == 0:
            return idxs, observed_idxs, observed_idxs

        predicted_idx = self.appearance_classifier(features[observed], observed_idxs, scores[observed])

        return idxs, observed_idxs, predicted_idx

    def _update_and_get_confirmations(self, priorities, frame, to_switch):

        updated_stats = self._update(priorities, frame, to_switch)
        if len(updated_stats) == 0:
            return None

        updated_idxs, confirmed_idxs, conf_identity_idxs = updated_stats

        keep = self.consecutive_hits[confirmed_idxs, conf_identity_idxs].clone()
        self.consecutive_hits[confirmed_idxs] = 0
        self.consecutive_hits[confirmed_idxs, conf_identity_idxs] = keep + 1

        hits_mask = self.consecutive_hits[confirmed_idxs, conf_identity_idxs] >= self.min_consecutive_hits
        self.consecutive_hits[confirmed_idxs[hits_mask], conf_identity_idxs[hits_mask]] = 0

        return updated_idxs, confirmed_idxs, conf_identity_idxs, hits_mask

    def _reset(self, keys):
        self.consecutive_hits[keys] = 0.0
        self.did_not_check_since[keys] = 0

    def _update_appearance_validation_instances(self, tracking_results, *args, **kwargs):
        # TODO pour visualization... avoir un nouveau csv output qui save les PCA 2D coords the chaque features dans la database
        # TODO visualization: 2D scatter plot avec 1 couleur par identifiant + display en noir les hits
        tracking_results["appearance_database"]["features"] = self.appearance_classifier._database
        tracking_results["appearance_database"]["identities"] = self.appearance_classifier._identities

    def _init_validation(self, tracking_results: dict):
        if "correction_instances" not in tracking_results:
            tracking_results["correction_instances"] = {"instances_id": [], "class_id": [], "corrected_id": []}
        if "appearance_database" not in tracking_results:
            tracking_results["appearance_database"] = {}
