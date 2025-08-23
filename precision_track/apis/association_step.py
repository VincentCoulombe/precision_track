import json
import logging
import os.path as osp
from collections import deque
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from addict import Dict
from mmengine.config import Config
from mmengine.logging import print_log

from precision_track.registry import TRACKING
from precision_track.utils import parse_pose_metainfo


class AssociationStep(nn.Module):

    def __init__(
        self,
        tracking_algorithm: Config,
        motion_algorithm: Config,
        metafile: str,
        thresholds_file: Optional[str] = "",
        stitching_algorithm: Optional[Config] = None,
        nb_frames_retain: Optional[int] = 10,
        num_tentatives: Optional[int] = 3,
        memory_length: Optional[int] = 30,
        verbose: Optional[bool] = True,
        **kwargs,
    ) -> None:
        """Leverages a tracking algorithm (like ByteTrack), a motion (Like a
        kalman filter) and a stitching algorithm (like SearchBased stitching)
        to link detections over-time.

        Args:
            tracking_algorithm (Config): The provided tracking algorithm's config
            motion_algorithm (Config): The provided motion algorithm's config
            metafile (str): The metainfo of the tracks.
            stitching_algorithm (Config, optional): The provided stitching algorithm's config. Defaults to None.
            nb_frames_retain (int, optional): The number of frames a track is kept in the data structures after its last detection. Defaults to 10.
            num_tentatives (int, optional): The number of consecutive detections before an unconfirmed track become confirmed. Defaults to 3.
            memory_length (int, optional): The number of past frames data kept in RAM. Defaults to 30.
        """
        super().__init__()
        self.verbose = verbose
        self.num_frames_retain = nb_frames_retain
        assert 0 < nb_frames_retain
        metadata = parse_pose_metainfo({"from_file": metafile})
        assert "classes" in metadata, "The metadata must contain a list of the tracked classes."
        self.classes = metadata["classes"]
        tracking_algorithm["metafile"] = metafile
        tracking_algorithm["nb_frames_retain"] = nb_frames_retain
        self.tracking_algorithm = TRACKING.build(tracking_algorithm)

        tracking_thr = None
        stitching_hyperparams = dict()

        if isinstance(thresholds_file, str) and osp.exists(thresholds_file):
            with open(thresholds_file, "r") as f:
                try:
                    hyperparams = json.load(f)
                except json.JSONDecodeError:
                    hyperparams = {}
            tracking_thr = hyperparams.get("tracking_thresholds")
            stitching_hyperparams = hyperparams.get("stitching_hyperparams", dict())
        if tracking_thr is not None:
            self.tracking_algorithm.update_thresholds(tracking_thr)
            if self.verbose:
                print_log(f"Dynamically updating tracking thresholds to: {tracking_thr}.", logger="current")

        self.motion = TRACKING.build(motion_algorithm)
        self.tracking_algorithm.kf = self.motion
        if stitching_algorithm is not None:
            if stitching_hyperparams is not None:
                stitching_algorithm.update(stitching_hyperparams)
                if self.verbose and len(stitching_hyperparams) > 0:
                    print_log(f"Dynamically updating stitching hyperparameters to: {stitching_hyperparams}.", logger="current")
            stitching_algorithm["classes"] = self.classes
            self.stitching_algorithm = TRACKING.build(stitching_algorithm)
        else:
            self.stitching_algorithm = None

        assert 0 <= num_tentatives
        self.num_tentatives = num_tentatives
        assert 0 < memory_length
        self.memory_length = memory_length
        self.reset()

    @property
    def confirmed_ids(self) -> List:
        """Confirmed ids in the tracker."""
        return [id for id, track in self.tracks.items() if not track.tentative]

    @property
    def unconfirmed_ids(self) -> List:
        """Unconfirmed ids in the tracker."""
        return [id for id, track in self.tracks.items() if track.tentative]

    def reset(self) -> None:
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()
        self.registry = dict()

    def init_frame(self, data_sample: dict, corrections: Optional[Dict[str, List[Tuple]]] = None) -> None:
        frame_id = data_sample.get("img_id", None)
        assert frame_id is not None, "frame_id must be provided"
        if isinstance(frame_id, (torch.Tensor, np.ndarray)):
            assert frame_id.ndim == 0, "frame_id must be a scalar"
            frame_id = frame_id.item()
        assert isinstance(frame_id, int), "frame_id must be an integer"
        if frame_id == 0:
            self.reset()
        frame_size = data_sample.get("ori_shape", None)
        assert frame_size is not None, "ori_shape must be provided"
        self.motion.frame_size = (frame_size[1], frame_size[0])
        self.frame_size = frame_size
        if self.stitching_algorithm is not None:
            self.stitching_algorithm.frame_size = frame_size
        if corrections is not None:
            for cls in corrections:
                for correction in corrections[cls]:
                    inst_id_a, inst_id_b = correction
                    id_a, id_b = -1, -1
                    for id_, trk in self.tracks.items():
                        if trk["instances_id"] == inst_id_a and trk["classes"] == cls:
                            id_a = id_
                        elif trk["instances_id"] == inst_id_b and trk["classes"] == cls:
                            id_b = id_
                    assert (
                        id_a >= 0
                    ), f"The {cls} {inst_id_a} was alive during the validation of the frame {frame_id-1}, but is not at the beginning of frame {frame_id}."
                    self.tracks[id_a]["instances_id"] = inst_id_b
                    if id_b >= 0:
                        self.tracks[id_b]["instances_id"] = inst_id_a
                    elif self.stitching_algorithm is not None and inst_id_b in self.stitching_algorithm.search_zones[cls]:
                        # The track to switch is currently hidden
                        self.stitching_algorithm.search_zones[cls][inst_id_a] = self.stitching_algorithm.search_zones[cls][inst_id_b]
                        self.stitching_algorithm.search_zones[cls][inst_id_a]["instances_id"] = inst_id_a
                        del self.stitching_algorithm.search_zones[cls][inst_id_b]

        return frame_id

    def update(self, **kwargs) -> None:
        """Update the tracker.

        Args:
            kwargs (dict[str: Tensor | int]): The `str` indicates the
                name of the input variable. `ids` and `frame_ids` are
                obligatory in the keys.
        """
        memo_items = [k for k, v in kwargs.items() if v is not None]
        rm_items = [k for k in kwargs if k not in memo_items]
        self.memo_ids = []
        self.memo_confirmed = []
        for item in rm_items:
            kwargs.pop(item)
        if not hasattr(self, "memo_items"):
            self.memo_items = memo_items
        else:
            assert memo_items == self.memo_items
        assert "ids" in memo_items
        num_objs = len(kwargs["ids"])
        id_indice = memo_items.index("ids")
        assert "frame_ids" in memo_items
        frame_id = int(kwargs["frame_ids"])
        if isinstance(kwargs["frame_ids"], int):
            kwargs["frame_ids"] = torch.tensor([kwargs["frame_ids"]] * num_objs)
        for v in kwargs.values():
            if len(v) != num_objs:
                raise ValueError("kwargs value must both equal")

        for obj in zip(*kwargs.values()):
            id = int(obj[id_indice])
            if id in self.tracks:
                self.update_track(id, obj)
            else:
                self.init_track(id, obj)
            self.confirm_track(id)
            self.memo_ids.append(self.tracks[id]["instances_id"])
            self.memo_confirmed.append(0 if self.tracks[id].tentative else 1)
        self.pop_invalid_tracks(frame_id)

    def init_track(
        self,
        id: int,
        obj: Tuple[Union[torch.Tensor, np.ndarray]],
    ) -> None:
        """Initialize a track."""
        self.tracks[id] = Dict()
        for k, v in zip(self.memo_items, obj):
            if k == "labels":
                v = v.item()
                if v not in self.registry:
                    self.registry[v] = 0
                self.registry[v] += 1
                self.tracks[id]["instances_id"] = self.registry[v]
                self.tracks[id][k] = v
                self.tracks[id]["classes"] = self.classes[v]
            elif k == "frame_ids":
                self.tracks[id][k] = deque([v.item()], maxlen=self.memory_length)
                self.tracks[id].init_frame = v.item()
            elif k != "ids":
                self.tracks[id][k] = deque([v], maxlen=self.memory_length)
        self.tracks[id].tentative = self.tracks[id].frame_ids[-1] != 0
        self.motion.initiate(self.tracks[id])

    def update_track(self, id: int, obj: Tuple[Union[torch.Tensor, np.ndarray]]) -> None:
        """Update a track."""
        for k, v in zip(self.memo_items, obj):
            if k == "labels":
                actual_label = self.tracks[id][k]
                actual_cls = self.classes[actual_label]
                v = v.item()
                if actual_label != v:
                    new_cls = self.classes[v]
                    self.tracks[id]["classes"] = new_cls
                    self.tracks[id]["labels"] = v
                    if v not in self.registry:
                        self.registry[v] = 0
                    self.registry[v] += 1
                    self.tracks[id]["instances_id"] = self.registry[v]
                    print_log(
                        msg=f"The track {id} went from a {actual_cls} to a {new_cls} at frame {self.tracks[id].frame_ids[-1]}",
                        logger="current",
                        level=logging.WARNING,
                    )
            elif k == "frame_ids":
                self.tracks[id][k].append(v.item())
            elif k != "ids":
                self.tracks[id][k].append(v)
        self.motion.update(self.tracks[id])

    def confirm_track(self, id: int) -> None:
        if self.tracks[id].tentative and len(self.tracks[id]["bboxes"]) >= self.num_tentatives:
            self.tracks[id].tentative = False

    def pop_invalid_tracks(self, frame_id: int) -> None:
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            last_seen_frame = v["frame_ids"][-1]

            # The track is alive and well
            case1 = last_seen_frame == frame_id
            v.lost = not case1
            # The track have been lost for too long
            case2 = frame_id - last_seen_frame >= self.num_frames_retain

            # The track have been lost, but was unconfirmed
            case3 = not case1 and v.tentative

            if case3 or case2:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def forward(
        self,
        data_sample: dict,
        corrections: Optional[List[Tuple]] = None,
        *args,
        **kwargs,
    ) -> dict:
        assert isinstance(data_sample, dict)
        return self.associate(
            data_sample,
            corrections,
            *args,
            **kwargs,
        )

    def associate(
        self,
        data_sample: dict,
        corrections: Optional[List[Tuple]] = None,
        *args,
        **kwargs,
    ) -> dict:
        """A full detection/track association step consisting of a tracking
        algorithm, a motion algorithm and maybe a stitching algorithm. Perform
        switches if necessary.

        Args:
            data_sample (dict): The data_sample coming from a detector.
            corrections (Optional[List[Tuple]], optional): The potential switches to perform. Defaults to None.

        Returns:
            dict: The data_sample modified in place.
        """
        frame_id = self.init_frame(data_sample, corrections)
        self.tracking_algorithm(
            data_sample,
            self.tracks,
            self.confirmed_ids,
            self.unconfirmed_ids,
            self.motion,
            *args,
            **kwargs,
        )
        self.num_tracks += data_sample["pred_track_instances"]["ids"].shape[0] - len(self.tracks)
        self.update(frame_ids=frame_id, **data_sample["pred_track_instances"])
        data_sample["pred_track_instances"]["classes"] = np.array([self.classes[lbl] for lbl in data_sample["pred_track_instances"]["labels"]])
        data_sample["pred_track_instances"]["instances_id"] = np.array(self.memo_ids)
        data_sample["pred_track_instances"]["confirmed"] = np.array(self.memo_confirmed)
        data_sample["pred_track_instances"]["velocities"] = np.array([self.tracks[id_].mean[2:4] for id_ in data_sample["pred_track_instances"]["ids"]])

        if self.stitching_algorithm is not None:
            self.stitching_algorithm(
                self.tracks,
                data_sample,
                self.confirmed_ids,
                self.unconfirmed_ids,
                *args,
                **kwargs,
            )

        return data_sample
