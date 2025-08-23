from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from precision_track.registry import TRACKING
from precision_track.utils import iou_batch, linear_assignment, reformat

from .base import BaseStitchingAlgorithm


@TRACKING.register_module()
class SearchBasedStitching(BaseStitchingAlgorithm):

    def __init__(
        self,
        capped_classes: Dict[str, int],
        beta: Optional[float] = 1.0,
        match_thr: Optional[float] = 0.9,
        eps: Optional[float] = 1e-2,
        **kwargs,
    ):
        """This search-based stitching algorithm aims to piece together
        fragmented trajectories of tracks belonging to classes that cannot
        leave the frame and have a fixed number over time, referred to as
        'capped classes.' By applying this method, the instance IDs of these
        tracks will always remain within a predefined limit.

        Args:
            capped_classes (Dict[str, int]): The classes whose tracks will be stitched. the
            instance IDs of these tracks will never exceed the integers given as values.
            beta (float, optional): The expension's decay rate for the search zones. Defaults to 1.0.
            match_thr (float, optional): The matching threshold between a track and a search zone. Defaults to 0.9.
            eps (float, optional): The relevancy of the lost track last seen direction in the search zone calculation. Defaults to 1e-2.
        """
        super().__init__(**kwargs)
        assert isinstance(capped_classes, dict)
        for capped_cls, cap in capped_classes.items():
            assert isinstance(capped_cls, str) and 0 <= cap
        self.capped_classes = {cls: np.zeros(capped_classes[cls], dtype=bool) for cls in capped_classes}
        self.search_zones = {cls: {} for cls in self.capped_classes}
        assert eps <= 1e-1, "A high epsilon could cause search zone update instability."
        self.eps = eps
        assert 0.0 < beta
        self.beta = beta
        assert 0.0 <= match_thr <= 1.0
        self.match_thr = match_thr
        self._frame_size = None

        self.init_vels = deque([], maxlen=100)
        self.avg_init_vel = 0.1
        self._counter = 1

    @property
    def frame_size(self):
        if self._frame_size is None:
            raise ValueError("Frame size not set for the search based stitching.")
        return self._frame_size

    @frame_size.setter
    def frame_size(self, frame_size: Tuple[int, int]):
        self._frame_size = frame_size

    def assign_ids(
        self,
        search_pos: np.ndarray,
        search_labels: np.ndarray,
        new_xyxy: np.ndarray,
        new_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not search_pos.size or not new_xyxy.size:
            return np.array([], dtype=int), np.array([], dtype=int)

        mean_pos = search_pos[:, :4]
        gious = iou_batch(
            mean_pos.astype(np.float32),
            new_xyxy.astype(np.float32),
            general=True,
        )

        # add huge cost to dets/search zones class mismatch
        gate_match = new_labels[None, :] == search_labels[:, None]
        gate_cost = (1 - gate_match.astype(int)) * 1e9
        dists = 1 - gious + gate_cost
        matched_tracks, matched_dets = linear_assignment(dists, self.match_thr)
        return matched_tracks.astype(int), matched_dets.astype(int)

    def _id_in_range(self, track: dict):
        cls_instances = self.capped_classes[track.classes]
        if 0 < track.instances_id <= len(cls_instances):
            cls_instances[track.instances_id - 1] = True
            return True
        return False

    def _late_init(self, track: dict):
        cls_instances = self.capped_classes[track.classes]
        if not np.all(cls_instances):
            idx = np.argmax(~cls_instances)
            return idx + 1
        return -1

    @staticmethod
    def _set_sides(search_zone: dict):
        search_zone["sides"] = np.array(
            [
                [-max(search_zone["cxcywh"][2], 0.1) / 2, 0],
                [max(search_zone["cxcywh"][2], 0.1) / 2, 0],
                [0, -max(search_zone["cxcywh"][3], 0.1) / 2],
                [0, max(search_zone["cxcywh"][3], 0.1) / 2],
            ]  # Left  # Right  # Top  # Bottom
        )

    def init_search_zone(self, track: dict, frame_id: int) -> dict:
        """Init a search zone based on the last prediction of a track and the
        cosine, of the angle between the last prediction and the sides of the
        search zone.

        Args:
            track (dict): The track
            frame_id (int): The frame id

        Returns:
            dict: The search zone
        """
        cxcywh = track["pred_bboxe"]
        vec_norm = np.linalg.norm(track["velocity"])
        if vec_norm == 0.0:
            track["velocity"] += 0.1
            vec_norm = np.linalg.norm(track["velocity"])
        search_zone = dict(
            id=self._counter,
            labels=track.labels,
            classes=track.classes,
            instances_id=track.instances_id,
            frame_id=frame_id,
            x=1,
            cosines=np.zeros(2),
            vec_norm=vec_norm,
            cxcywh=cxcywh,
        )
        self._counter += 1
        self._set_sides(search_zone)
        sides_norms = np.linalg.norm(search_zone["sides"], axis=1)
        dot_products = search_zone["sides"] @ track["velocity"]
        cosines = dot_products / (sides_norms * search_zone["vec_norm"])
        search_zone["cosines"] = (cosines + 1) / 2
        search_zone["cosines"] = np.stack([search_zone["cosines"], search_zone["cosines"]], axis=-1)
        self.init_vels.append(search_zone["vec_norm"])
        self.avg_init_vel = max(self.avg_init_vel, np.mean(self.init_vels))
        return search_zone

    def update_search_zone(self, search_zone: dict) -> None:
        """Update the search zone based on the last prediction of a track and
        the cosine, of the angle between the last prediction and the sides of
        the search zone.

        Args:
            search_zone (dict): The search zone
        """
        search_zone["x"] += 1
        enlarge_value = self._get_enlarge_value(search_zone["vec_norm"], search_zone["x"])
        delta_sides = ((search_zone["cosines"] + self.eps) * enlarge_value) * search_zone["sides"]
        search_zone["sides"] += delta_sides
        search_zone["cxcywh"] = reformat(
            np.array(
                [
                    max(search_zone["cxcywh"][0] + search_zone["sides"][0, 0], 0),
                    max(search_zone["cxcywh"][1] + search_zone["sides"][2, 1], 0),
                    min(
                        search_zone["cxcywh"][0] + search_zone["sides"][1, 0],
                        self.frame_size[0],
                    ),
                    min(
                        search_zone["cxcywh"][1] + search_zone["sides"][3, 1],
                        self.frame_size[1],
                    ),
                ]
            ),
            "xyxy",
            "cxcywh",
        )
        self._set_sides(search_zone)

    def _get_enlarge_value(self, initial_velocity: float, x: int) -> float:
        return np.sqrt(initial_velocity + self.avg_init_vel / max(initial_velocity, 0.1)) * np.exp(-self.beta * x)

    def __call__(
        self,
        tracks: dict,
        data_samples: dict,
        confirmed_ids: List[int],
        unconfirmed_ids: List[int],
    ):
        """Stitch the new tracks to lost trajectories by building and updating
        search zones based on the last prediction of the lost trajectories and
        the cosine of the angle between the velocity vector of the lost
        trajectory last's measurement and the sides of the search zone.

        Args:
            tracks (dict): The tracks, containing the lost and confirmed tracks as well as
                their trajectories
            data_samples (dict): This current frame output
            confirmed_ids (List[int]): List of confirmed tracks ids
            unconfirmed_ids (List[int]): List of unconfirmed tracks ids
        """
        super().__call__(tracks, data_samples["pred_track_instances"])
        for id_ in unconfirmed_ids:
            self.__masktrack__(id_)
        confirmed_surplus_ids = []
        confirmed_surplus_labels = []
        running_ids = Counter()
        lost_but_alive_ids = []
        lost_but_alive_inst_ids = []
        late_inits = {}

        # 1.1) Determine if new search zones need to be initiated (how many trajectories are lost)
        for confirmed_id in confirmed_ids:
            track = tracks[confirmed_id]
            cls = track.classes
            lbl = track.labels
            if cls in self.capped_classes:
                instances_id = int(track.instances_id)
                id_in_range = self._id_in_range(track)
                if cls not in late_inits:
                    late_inits[cls] = {}
                if not track.lost:
                    if not id_in_range:
                        free_inst_id = self._late_init(track)
                        if free_inst_id > 0:
                            late_inits[cls][confirmed_id] = (lbl, free_inst_id)
                        else:
                            confirmed_surplus_ids.append(confirmed_id)
                            confirmed_surplus_labels.append(lbl)
                    else:
                        running_ids[lbl] += 1
                        if instances_id in self.search_zones[cls]:
                            del self.search_zones[cls][instances_id]
                else:
                    lost_but_alive_ids.append(confirmed_id)
                    lost_but_alive_inst_ids.append(instances_id)
                    if id_in_range and instances_id not in self.search_zones[cls]:
                        self.search_zones[cls][instances_id] = self.init_search_zone(track, data_samples["img_id"])

        # 1.2) Filter out the late inits
        for cls in late_inits:
            if not self.search_zones[cls]:
                assigned_ids = []
                for confirmed_id, (label, new_inst_id) in late_inits[cls].items():
                    inst_id_index = new_inst_id - 1
                    if not self.capped_classes[cls][inst_id_index]:
                        self.capped_classes[cls][inst_id_index] = True
                        self.__setinstid__(confirmed_id, new_inst_id)
                        assigned_ids.append(confirmed_id)
                        running_ids[label] += 1
                for assigned_id in assigned_ids:
                    del late_inits[cls][assigned_id]
            confirmed_surplus_labels.extend(lbl for lbl, _ in late_inits[cls].values())
            confirmed_surplus_ids.extend(late_inits[cls].keys())

        # 1.3) Updating currently running search zones
        search_areas = {}
        search_pos = []
        search_labels = []
        search_classes = []
        search_instance_ids = []
        for cls, search_zone in self.search_zones.items():
            for search_zone in search_zone.values():
                self.update_search_zone(search_zone)
                search_classes.append(search_zone["classes"])
                search_pos.append(search_zone["cxcywh"])
                search_labels.append(search_zone["labels"])
                search_instance_ids.append(search_zone["instances_id"])

        search_areas["bboxes"] = search_pos
        search_areas["labels"] = search_labels
        search_areas["instances_id"] = search_instance_ids
        data_samples["search_areas"] = search_areas

        if len(confirmed_surplus_ids) == 0:
            return

        # 2) Matching new tracks to search zones (trajectories)
        confirmed_surplus_ids = np.array(confirmed_surplus_ids)
        idx = np.isin(data_samples["pred_track_instances"]["ids"], confirmed_surplus_ids)
        confirmed_surplus_ids = data_samples["pred_track_instances"]["ids"][idx]
        xyxy = np.empty((len(confirmed_surplus_ids), 4), dtype=np.float32)
        for i, id_ in enumerate(confirmed_surplus_ids):
            xyxy[i] = tracks[id_]["bboxes"][0]  # 1st detected bbox of the new track
        labels = data_samples["pred_track_instances"]["labels"][idx]
        matched_zones, matched_new_trk = self.assign_ids(
            np.array(search_pos),
            np.array(search_labels),
            xyxy,
            labels,
        )

        # 3) Changing the instance id of the matched tracks
        matched_ids = confirmed_surplus_ids[matched_new_trk]
        matched_labels = np.array(confirmed_surplus_labels)[matched_new_trk]
        matched_searched_instance_ids = np.array(search_instance_ids)[matched_zones]
        matched_searched_labels = np.array(search_labels)[matched_zones]
        for label, cls_ in enumerate(self.capped_classes):
            missing_nb = abs(self.capped_classes[cls_].sum() - running_ids[label])
            label_idx = matched_labels == label
            matched_label_idx = matched_ids[label_idx][:missing_nb]
            matched_searched_label_idx = matched_searched_labels == label
            matched_searched_label_idx = matched_searched_instance_ids[matched_searched_label_idx][:missing_nb]
            for matched_id, matched_searched_instance_id in zip(matched_label_idx, matched_searched_label_idx):
                self.__setinstid__(matched_id, matched_searched_instance_id)
                del self.search_zones[tracks[matched_id].classes][matched_searched_instance_id]
                tracks_to_kill = np.where(np.isin(lost_but_alive_inst_ids, matched_searched_instance_id))[0]
                for track_to_kill in tracks_to_kill:
                    del tracks[lost_but_alive_ids[track_to_kill]]

        # 4) Masking the unmatched tracks with instance ids > the max number of instances
        for id_ in confirmed_surplus_ids:
            self.__masktrack__(id_)
