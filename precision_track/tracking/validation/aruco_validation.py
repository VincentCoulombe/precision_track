import heapq
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmengine.logging import MMLogger

from precision_track.registry import TRACKING
from precision_track.utils import clip, euc_dist_batch, reformat

from .base_validation import BaseValidation


@TRACKING.register_module()
class ArucoValidation(BaseValidation):
    REFINEMENTS = {
        "none": cv2.aruco.CORNER_REFINE_NONE,
        "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
        "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
        "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
    }

    def __init__(
        self,
        num_tags: int,
        tags_size: int,
        valid_tags: List[int],
        parameters: Dict[str, Union[int, float]],
        predefined_dict: Optional[str] = None,
        refinement: Optional[str] = "none",
        validated_classes: Optional[List[str]] = None,
        tag_kpt: Optional[str] = 7,
        kpt_conf_thr: Optional[float] = 0.5,
        estimation_range: Optional[int] = 50,
        timeout_after: Optional[float] = 0.02,
        min_sample_size: Optional[int] = 50,
        min_precision: Optional[float] = 0.9,
        memory_length: Optional[int] = 100,
    ) -> None:
        """A validation algorithm consisting of reading the Aruco markers on the tags attached
        to subjects. Please refer to : https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

        Args:
            num_tags (int): The number of tags in the Aruco's dictionary.
            tags_size (int): The bit size of each Tag
            valid_tags (List[int]): The markers ids we care about reading.
            parameters (Dict[str, Union[int, float]]): The reading parameters.
            predefined_dict (Optional[str], optional): Named of the predefined dictionary. Defaults to None.
            refinement (Optional[str], optional): The name of the refinement step. Defaults to "none".
            validated_classes (Optional[List[str]], optional): List of the classes which support validation. If None, all the classes are considered.
            tag_kpt (Optional[str], optional): The keypoint on which the tag is attached. Defaults to 7.
            kpt_conf_thr (Optional[float], optional): the confidence threshold to consider reading a marker on a keypoint. Defaults to 0.5.
            estimation_range (Optional[int], optional): In how much pixels do we search for a marker around the keypoint. Defaults to 50.
            timeout_after (Optional[float], optional): The maximum latency of the step. Defaults to 0.02.
            min_sample_size (Optional[int], optional): The minimal sample size to calculate the precision on. Defaults to 50.
            min_precision (Optional[float], optional): What precision does the tag need to have for a reading to count as a detection.
            Defaults to 0.9.
        """
        super(ArucoValidation, self).__init__()
        self.logger = MMLogger.get_current_instance()
        assert isinstance(num_tags, int) and isinstance(tags_size, int)
        assert num_tags >= 0 and tags_size >= 0
        self.dictionnary = cv2.aruco.Dictionary_create(num_tags, tags_size)
        if predefined_dict is not None:
            self.dictionnary = cv2.aruco.Dictionary_get(getattr(cv2.aruco, predefined_dict))
            self.logger.info(msg=f"ARUCO VALIDATION: Overwriting {num_tags} {tags_size}x{tags_size} with predefined dictionary {predefined_dict}.")
        self.parameters = cv2.aruco.DetectorParameters_create()
        assert isinstance(parameters, dict)
        for key, value in parameters.items():
            setattr(self.parameters, key, value)
        self.parameters.cornerRefinementMethod = self.REFINEMENTS[refinement]
        self.valid_tags = np.array(valid_tags)
        self.warned_full_frame = False
        if validated_classes is not None:
            assert isinstance(validated_classes, list)
            for cls in validated_classes:
                assert isinstance(cls, str)
        self.validated_classes = validated_classes
        assert isinstance(tag_kpt, int)
        self.tag_kpt = tag_kpt
        assert 0.0 <= kpt_conf_thr <= 1.0
        self.kpt_conf_thr = kpt_conf_thr
        self.estimation_range = estimation_range
        assert 0 < timeout_after
        self.timeout_after = timeout_after
        self.detection_counts = np.zeros((len(valid_tags), len(valid_tags)), dtype=np.uint32)
        self.running_det_counts = np.zeros_like(self.detection_counts)
        self.tag2instance_id = {k: -1 for k in valid_tags}

        assert 0 < min_sample_size
        self.min_sample_size = min_sample_size
        assert 0.0 <= min_precision <= 1.0
        self.min_precision = min_precision

        assert isinstance(memory_length, int) and 0 <= memory_length
        self.memory_length = memory_length
        self.memory = {}

    @staticmethod
    def _init_validation(tracking_results: dict):
        tracking_results["validation_instances"] = {
            "bboxes": [],
            "instances_id": [],
            "tags_id": [],
            "tags_precision": [],
        }
        tracking_results["correction_instances"] = {
            "instances_id": [],
            "tags_id": [],
            "corrected_id": [],
        }

    @staticmethod
    def _register_correction(tracking_results: dict, ori_id: int, corrected_id: int, switched_tag_id: int):
        tracking_results["correction_instances"]["instances_id"].append(ori_id)
        tracking_results["correction_instances"]["corrected_id"].append(corrected_id)
        tracking_results["correction_instances"]["tags_id"].append(switched_tag_id)

    def _build_priority_queue(self, track_instances: dict) -> int:
        conf_kpts = track_instances["keypoint_scores"][:, self.tag_kpt] > self.kpt_conf_thr
        kpts = track_instances["keypoints"][:, self.tag_kpt][conf_kpts]
        dists = euc_dist_batch(kpts)
        isolated = np.sum(dists > self.estimation_range, axis=1) == kpts.shape[0] - 1

        priority_queue = []
        for cls, inst_id, xy, confirmed, score in zip(
            track_instances["classes"][conf_kpts][isolated],
            track_instances["instances_id"][conf_kpts][isolated],
            kpts[isolated],
            track_instances["confirmed"][conf_kpts][isolated],
            track_instances["scores"][conf_kpts][isolated],
        ):
            if self.validated_classes is None or cls in self.validated_classes:
                heapq.heappush(priority_queue, (float(confirmed + score), (cls, int(inst_id), xy.tolist())))
        return priority_queue

    def _estimate_range(self, xy: np.ndarray):
        cxcywh = clip(np.array([xy[0], xy[1], self.estimation_range, self.estimation_range]), "cxcywh", self.frame_size[0], self.frame_size[1])
        return (
            int(cxcywh[0] - cxcywh[2] / 2),
            int(cxcywh[1] - cxcywh[3] / 2),
            int(cxcywh[2]),
            int(cxcywh[3]),
        )

    def _detect_markers(
        self,
        frame: np.ndarray,
        tracking_results: dict,
        ajustments: Optional[np.ndarray] = None,
        full_frame: Optional[bool] = True,
    ):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionnary, parameters=self.parameters)
        if ids is None:
            corners = np.array(corners)
            valid_ids = np.array([])
            valid_idx = np.array([])
        else:
            valid_ids, ids_idx, valid_idx = np.intersect1d(ids, self.valid_tags, assume_unique=False, return_indices=True)
            valid_ids = valid_ids.reshape(-1)
            corners = np.concatenate(corners).reshape(-1, 8)[ids_idx].astype(np.int32)
            corners = reformat(corners, "polygon", "cxcywh").reshape(-1, 4)
            if ajustments is not None:
                corners += ajustments
            tracking_results["validation_instances"]["bboxes"].extend(corners.tolist())
        if full_frame:
            tracking_results["validation_instances"]["instances_id"].extend((np.zeros_like(valid_ids) - 1).tolist())
        tracking_results["validation_instances"]["tags_id"].extend(valid_ids.tolist())
        return valid_ids, valid_idx, corners

    def _get_tags_precision(self, tags_idx: np.ndarray):
        max_hits = np.max(self.running_det_counts[:, tags_idx], axis=0).astype(float)
        return np.divide(max_hits, np.sum(self.running_det_counts[:, tags_idx], axis=0), out=np.zeros_like(max_hits), where=max_hits != 0)

    def _get_inst_precision(self, instance_id: int, tags_idx: int):
        tags_det = self.running_det_counts[instance_id - 1, tags_idx].astype(float)
        return np.divide(tags_det, np.sum(self.running_det_counts[instance_id - 1, :], axis=0), out=np.zeros_like(tags_det), where=tags_det != 0)

    @staticmethod
    def _switching_back(switches: List[tuple], a: int, b: int):
        for switch in switches:
            if switch[0] == b and switch[1] == a:
                return True
        return False

    def _refresh_memory(self, frame_id: int):
        self.memory[frame_id] = np.zeros_like(self.detection_counts)
        if frame_id >= self.memory_length:
            self.running_det_counts = np.maximum(self.running_det_counts - self.memory[frame_id - self.memory_length], 0)
            del self.memory[frame_id - self.memory_length]

    def _register_validate_detections(self, tracking_results: dict, instance_id: int, valid_idx: np.ndarray, frame_id: int):
        # 1) Register
        self.detection_counts[instance_id - 1, valid_idx] += 1
        self.memory[frame_id][instance_id - 1, valid_idx] += 1
        self.running_det_counts[instance_id - 1, valid_idx] += 1
        tags_precision = self._get_tags_precision(valid_idx)
        tracking_results["validation_instances"]["tags_precision"].extend(tags_precision)

        # 2) Validate
        insts_precision = self._get_inst_precision(instance_id, valid_idx)
        tag_ids = self.valid_tags[valid_idx]
        nb_dets = self.running_det_counts[instance_id - 1, valid_idx]

        for id_, inst_p, tag_p, n_d in zip(tag_ids, insts_precision, tags_precision, nb_dets):
            if inst_p >= self.min_precision and tag_p >= self.min_precision and n_d >= self.min_sample_size:
                detected_inst_id = self.tag2instance_id[id_]
                free = detected_inst_id < 0 or detected_inst_id not in self.tag2instance_id.values()
                if free:
                    # The tag is permanently linked to that instance id
                    self.tag2instance_id[id_] = instance_id
                elif detected_inst_id != instance_id:
                    # There as been an id switch
                    return (instance_id, detected_inst_id), id_
        return None, None

    def __call__(
        self,
        frame: np.ndarray,
        tracking_results: Optional[dict] = None,
    ) -> Optional[Dict[str, List[Tuple]]]:
        """Validate the positions of the subject by detecting the markers on
        the frame. If no tracking is taking place, detect on the whole frame.

        Args:
            frame (np.ndarray): The frame to validate on.
            tracking_results (Optional[dict], optional): The tracking results. Modified in place. Defaults to None.

        Returns:
            List[Tuple]: the trackign results and a list of potential switches.
        """
        if tracking_results is None:
            tracking_results = dict()
            self._init_validation(tracking_results)
            if not self.warned_full_frame:
                self.logger.warning(msg="ARUCO VALIDATION: The algorithm is not tracking anything, trying to locate tags on the full frame.")
                self.warned_full_frame = True
            _, _, _ = self._detect_markers(frame, tracking_results)
            return tracking_results, None

        started_at = perf_counter()
        self._init_validation(tracking_results)
        track_instances = tracking_results["pred_track_instances"]
        priorities = self._build_priority_queue(track_instances)
        to_switch = {}
        frame_id = tracking_results["img_id"]
        self._refresh_memory(frame_id)

        while perf_counter() - started_at < self.timeout_after and priorities:
            cls, instance_id, xy = heapq.heappop(priorities)[1]
            if cls not in to_switch:
                to_switch[cls] = []
            if 0 < instance_id:
                x, y, w, h = self._estimate_range(np.array(xy, dtype=np.float64))
                cropped_frame = frame[y : y + h, x : x + w]
                # save frame
                # cv2.imwrite(f"./{instance_id}_{perf_counter()}.jpg", cropped_frame)
                valid_ids, valid_idx, _ = self._detect_markers(
                    cropped_frame,
                    tracking_results,
                    ajustments=np.array([x, y, 0, 0]),
                    full_frame=False,
                )
                if valid_ids.size > 0:
                    tracking_results["validation_instances"]["instances_id"].extend([self.tag2instance_id[t] for t in valid_ids])
                    switches, switched_tag_id = self._register_validate_detections(tracking_results, instance_id, valid_idx, frame_id)
                    if switches is not None:
                        insts_id = track_instances["instances_id"]
                        _, detected_instance_id = switches
                        if not self._switching_back(to_switch[cls], instance_id, detected_instance_id):
                            to_switch[cls].append(switches)
                            mask_a = insts_id == instance_id
                            mask_b = insts_id == detected_instance_id
                            insts_id[mask_a] = detected_instance_id
                            insts_id[mask_b] = instance_id
                            self._register_correction(tracking_results, instance_id, detected_instance_id, switched_tag_id)
        tracking_results["corrected_instances_id"] = to_switch
        return tracking_results, to_switch
