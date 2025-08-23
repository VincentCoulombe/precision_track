import abc
import logging
import os
from collections import OrderedDict
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from mmengine.logging import print_log

from precision_track.registry import OUTPUTS
from precision_track.utils import parse_pose_metainfo, reformat, to_numpy

from .base import BaseOutput


class BaseCsvOutput(BaseOutput):
    SUPPORTED_PRECISION = {32: "float32", 64: "float64"}
    EXTENSION = ".csv"
    MAPPING_EXTENSION = ".npy"

    def __init__(
        self,
        path: str,
        instance_data: str,
        columns: List[str],
        confidence_threshold: float = 0.5,
        precision: int = 32,
        ids_field: str = "instances_id",
        save_mapping: bool = False,
    ) -> None:
        """A BaseCsvOutput iteratively store the relevant dict's instances data
        and can save it to csv afterward.

        Args:
            path (str): The path for either save the content of a BaseCsvOutput to a csv or to read a csv into a BaseCsvOutput
            instance_data (str): The name of the relevant instances data inside the dict
            columns (List[str]): The name of the csv's columns
            confidence_threshold (float, optional): The threshold from which data is retained. Defaults to 0.5.
            precision (int, optional): The saving precision for the data. Defaults to 32.
        """
        self.reset()
        if precision not in self.SUPPORTED_PRECISION:
            raise ValueError(f"Precision {precision} not supported. Supported precisions are {list(self.SUPPORTED_PRECISION.keys())}")
        self.precision = precision
        self.columns = columns
        self.confidence_threshold = confidence_threshold
        self.supported_instance_data = ["pred_track_instances", "pred_instances"]
        self.instance_data = instance_data
        self._setup_path(path)
        self.ids_field = ids_field
        self.save_mapping = save_mapping

    def _setup_path(self, path: str):
        raw_path, _ = os.path.splitext(path)
        path = f"{raw_path}{self.EXTENSION }"
        self.path = os.path.abspath(path)
        self.mapping_path = os.path.abspath(f"{raw_path}_mapping{self.MAPPING_EXTENSION }")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    @abc.abstractmethod
    def __call__(self, data: dict) -> None:
        pass

    def __len__(self) -> int:
        if self.frame_id_mapping:
            return max(self.frame_id_mapping) + 1
        return 0

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self) -> List[Any]:
        if self._current >= self.__len__():
            raise StopIteration
        out = self.__getitem__(self._current)
        self._current += 1
        return out

    def __getitem__(self, idx: Union[int, slice]) -> List[Any]:
        if isinstance(idx, slice):
            result = []
            for i in range(*idx.indices(self.__len__())):
                result.extend(self.__getitem__(i))
            return result
        else:
            idx_range = self.frame_id_mapping.get(idx)
            if idx_range is None:
                return []
            return self.get_result_slice(idx_range[0], idx_range[1])

    def get_result_slice(self, start: int, end: int) -> List[Any]:
        return self.results[start:end]

    def reset(self) -> None:
        self.frame_id_mapping = OrderedDict()
        self.results = []
        self.curr_frame_idx = 0

    def save(self) -> None:
        """Save the data to csv and also a mapping of the data (for faster
        __getitem__).

        Saves only one frame_id_mapping
        """
        formatted_results = self.to_dataframe()
        formatted_results.to_csv(self.path, index=False)
        print_log(f"Saved output: {self.path}")
        if self.save_mapping:
            np.save(self.mapping_path, self.frame_id_mapping, allow_pickle=True)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results, columns=["frame_id", "class_id", "instance_id"] + self.columns)
        df["frame_id"] = df["frame_id"].astype("uint32")
        df["class_id"] = df["class_id"].astype("uint16")
        df["instance_id"] = df["instance_id"].astype("int16")
        for col in self.columns:
            df[col] = df[col].astype(self.SUPPORTED_PRECISION[self.precision])
        return df

    def read(self) -> None:
        """Load a csv and a mapping (for faster __getitem__)"""
        assert os.path.exists(self.path), f"{self.path} does not exist."
        self.results = pd.read_csv(self.path)
        self.columns = self.results.columns[3:].tolist()
        self.results = self.results.values.tolist()
        if os.path.exists(self.mapping_path):
            self.frame_id_mapping = np.load(self.mapping_path, allow_pickle=True).item()
        else:
            self._infer_mapping()
        if not self.results or not self.frame_id_mapping:
            print_log(f"{self.path} is empty.", level=logging.WARNING)
        else:
            self.curr_frame_idx = self.frame_id_mapping[self.__len__() - 1][1] + 1

    def _infer_mapping(self):
        current_frame = None
        start_index = 0

        for i, row in enumerate(self.results):
            frame_id = int(row[0])

            if current_frame is None:
                current_frame = frame_id

            if frame_id != current_frame:
                self.frame_id_mapping[current_frame] = (start_index, i)
                current_frame = frame_id
                start_index = i

        if current_frame is not None:
            self.frame_id_mapping[current_frame] = (start_index, len(self.results))

        return self.frame_id_mapping

    def _add_row(self, *args) -> None:
        self.results.append(list(args))

    def _update_frame_id_mapping(self, frame_id: int, increment: int):
        if frame_id not in self.frame_id_mapping:
            curr_frame_idx = self.curr_frame_idx + increment
            self.frame_id_mapping[frame_id] = (
                self.curr_frame_idx,
                curr_frame_idx,
            )
            self.curr_frame_idx = curr_frame_idx

    def _set_ids(self, instance_data: dict):
        return (
            np.zeros_like(to_numpy(instance_data["labels"])) - 1
            if self.instance_data
            not in [
                "pred_track_instances",
                "validation_instances",
                "correction_instances",
                "search_areas",
            ]
            else instance_data[self.ids_field]
        )

    def _get_ds_info(self, data_sample: dict):
        instance_data = data_sample.get(self.instance_data, None)
        if instance_data is None:
            raise ValueError(f"The provided data sample do not contain the expected instance data ({self.instance_data}).")
        return instance_data, data_sample["img_id"]


@OUTPUTS.register_module()
class CsvBoundingBoxes(BaseCsvOutput):
    SUPPORTED_FORMATS = ["cxcywh", "xyxy", "xywh"]

    def __init__(
        self,
        path: str,
        bbox_format: str = "cxcywh",
        instance_data: str = "pred_instances",
        confidence_threshold: float = 0.5,
        precision: int = 32,
        ids_field: str = "instances_id",
        save_bbox_format: list = None,
        *args,
        **kwargs,
    ) -> None:
        if save_bbox_format is None:
            self.save_bbox_format = ["cx", "cy", "w", "h"]
        else:
            assert isinstance(save_bbox_format, list)
            assert len(save_bbox_format) == 4
            self.save_bbox_format = save_bbox_format
        self.save_bbox_format_str = "".join(self.save_bbox_format)
        assert self.save_bbox_format_str in self.SUPPORTED_FORMATS
        super().__init__(
            path=path,
            precision=precision,
            confidence_threshold=confidence_threshold,
            columns=self.save_bbox_format + ["score"],
            instance_data=instance_data,
            ids_field=ids_field,
        )
        assert bbox_format in self.SUPPORTED_FORMATS, f"The currently supported bboxe formats are: {self.SUPPORTED_FORMATS}"
        self.bbox_format = bbox_format
        self.supported_instance_data.append("gt_instances")
        assert self.instance_data in self.supported_instance_data, f"The provided instance_data must be one one {self.supported_instance_data}"

    def __call__(self, det_data_sample: dict):
        instance_data, frame_id = self._get_ds_info(det_data_sample)
        ids = self._set_ids(instance_data)
        i = 0
        for id_, label, bbox, score in zip(
            ids,
            instance_data["labels"],
            instance_data["bboxes"],
            instance_data["scores"],
        ):
            label = to_numpy(label)
            bbox = to_numpy(bbox)
            score = to_numpy(score)
            if (score >= self.confidence_threshold and self.instance_data in ["pred_instances", "gt_instances"]) or (
                id_ >= 0 and self.instance_data == "pred_track_instances"
            ):
                if self.bbox_format != self.save_bbox_format_str:
                    bbox = reformat(bbox, self.bbox_format, self.save_bbox_format_str)
                self._add_row(frame_id, label, id_, *bbox, score)
                i += 1
        self._update_frame_id_mapping(frame_id, i)


@OUTPUTS.register_module()
class CsvValidations(BaseCsvOutput):
    SUPPORTED_FORMATS = ["cxcywh"]

    def __init__(
        self,
        path: str,
        bbox_format: str = "cxcywh",
        instance_data: str = "validation_instances",
        precision: int = 32,
        ids_field: str = "instances_id",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            path=path,
            precision=precision,
            columns=["cx", "cy", "w", "h", "tags_precision"],
            instance_data=instance_data,
            ids_field=ids_field,
        )
        assert bbox_format in self.SUPPORTED_FORMATS, f"The currently supported bboxe formats are: {self.SUPPORTED_FORMATS}"
        self.bbox_format = bbox_format
        self.supported_instance_data = ["validation_instances"]
        assert self.instance_data in self.supported_instance_data, f"The provided instance_data must be one one {self.supported_instance_data}"

    def __call__(self, det_data_sample: dict):
        instance_data, frame_id = self._get_ds_info(det_data_sample)
        ids = self._set_ids(instance_data)
        i = 0
        for id_, label, bbox, precision in zip(
            ids,
            instance_data["tags_id"],
            instance_data["bboxes"],
            instance_data["tags_precision"],
        ):
            label = to_numpy(label)
            bbox = to_numpy(bbox)
            precision = to_numpy(precision)
            if self.bbox_format != "cxcywh":
                bbox = reformat(bbox, self.bbox_format, "cxcywh")
            self._add_row(frame_id, label, id_, *bbox, precision)
            i += 1
        self._update_frame_id_mapping(frame_id, i)


@OUTPUTS.register_module()
class CsvCorrections(BaseCsvOutput):

    def __init__(
        self,
        path: str,
        instance_data: str = "correction_instances",
        precision: int = 32,
        ids_field: str = "instances_id",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            path=path,
            precision=precision,
            columns=["corrected_id"],
            instance_data=instance_data,
            ids_field=ids_field,
        )
        self.supported_instance_data = ["correction_instances"]
        assert self.instance_data in self.supported_instance_data, f"The provided instance_data must be one one {self.supported_instance_data}"

    def __call__(self, det_data_sample: dict):
        instance_data, frame_id = self._get_ds_info(det_data_sample)
        ids = self._set_ids(instance_data)
        i = 0
        for id_, label, corrected_id in zip(
            ids,
            instance_data["tags_id"],
            instance_data["corrected_id"],
        ):
            label = to_numpy(label)
            corrected_id = to_numpy(corrected_id)
            self._add_row(frame_id, label, id_, corrected_id)
            i += 1
        self._update_frame_id_mapping(frame_id, i)


@OUTPUTS.register_module()
class CsvSearchAreas(BaseCsvOutput):
    SUPPORTED_FORMATS = ["cxcywh"]

    def __init__(
        self,
        path: str,
        bbox_format: str = "cxcywh",
        instance_data: str = "search_areas",
        precision: int = 32,
        ids_field: str = "instances_id",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            path=path,
            precision=precision,
            columns=["cx", "cy", "w", "h"],
            instance_data=instance_data,
            ids_field=ids_field,
        )
        assert bbox_format in self.SUPPORTED_FORMATS, f"The currently supported bboxe formats are: {self.SUPPORTED_FORMATS}"
        self.bbox_format = bbox_format
        self.supported_instance_data = ["search_areas"]
        assert self.instance_data in self.supported_instance_data, f"The provided instance_data must be one one {self.supported_instance_data}"

    def __call__(self, det_data_sample: dict):
        instance_data, frame_id = self._get_ds_info(det_data_sample)
        ids = self._set_ids(instance_data)
        i = 0
        for id_, label, bbox in zip(
            ids,
            instance_data["labels"],
            instance_data["bboxes"],
        ):
            label = to_numpy(label)
            bbox = to_numpy(bbox)
            if self.bbox_format != "cxcywh":
                bbox = reformat(bbox, self.bbox_format, "cxcywh")
            self._add_row(frame_id, label, id_, *bbox)
            i += 1
        self._update_frame_id_mapping(frame_id, i)


@OUTPUTS.register_module()
class CsvKeypoints(BaseCsvOutput):

    def __init__(
        self,
        path: str,
        instance_data: str = "pred_instances",
        confidence_threshold: float = 0.5,
        precision: int = 32,
        ids_field: str = "instances_id",
        **kwargs,
    ) -> None:
        super().__init__(
            path,
            precision=precision,
            confidence_threshold=confidence_threshold,
            columns=[],
            instance_data=instance_data,
            ids_field=ids_field,
        )
        self.supported_instance_data.append("gt_instances")
        assert self.instance_data in self.supported_instance_data, f"The provided instance_data must be one one {self.supported_instance_data}"

    def __call__(self, det_data_sample):
        instance_data, frame_id = self._get_ds_info(det_data_sample)
        ids = self._set_ids(instance_data)
        i = 0
        for id_, label, keypoints, scores, score in zip(
            ids,
            instance_data["labels"],
            instance_data["keypoints"],
            instance_data["keypoint_scores"],
            instance_data["scores"],
        ):
            label = to_numpy(label)
            keypoints = to_numpy(keypoints)
            keypoint_scores = to_numpy(scores)
            score = to_numpy(score)
            if (score >= self.confidence_threshold and self.instance_data in ["pred_instances", "gt_instances"]) or (
                id_ >= 0 and self.instance_data == "pred_track_instances"
            ):
                poses = np.concatenate((keypoints, keypoint_scores.reshape(-1, 1)), axis=1)
                poses = np.nan_to_num(poses, nan=0.0).flatten().tolist()
                self._add_row(frame_id, label, id_, poses)
                i += 1
        self._update_frame_id_mapping(frame_id, i)

    def _add_row(self, frame_id, class_id, object_id, keypoints):
        self._set_columns(frame_id, keypoints)
        super()._add_row(frame_id, class_id, object_id, *keypoints)

    def _set_columns(self, frame_id: int, keypoints: list):
        if not self.columns:
            self.columns = [f"{coord}{i}" for i in range(len(keypoints) // 3) for coord in ("x", "y", "score")]
        else:
            assert len(keypoints) == len(self.columns), f"Inconsistent number of keypoints: {len(keypoints)}, expected: {len(self.columns)} as frame{frame_id}"

    def reset(self):
        self.columns = []
        super().reset()


@OUTPUTS.register_module()
class CsvVelocities(BaseCsvOutput):

    def __init__(
        self,
        path: str,
        confidence_threshold: float = 0.5,
        precision: int = 32,
        ids_field: str = "instances_id",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            path=path,
            precision=precision,
            confidence_threshold=confidence_threshold,
            columns=["vx", "vy"],
            instance_data="pred_track_instances",
            ids_field=ids_field,
        )

    def __call__(self, det_data_sample: dict):
        instance_data, frame_id = self._get_ds_info(det_data_sample)
        ids = self._set_ids(instance_data)
        i = 0
        for id_, label, vel in zip(
            ids,
            instance_data["labels"],
            instance_data["velocities"],
        ):
            label = to_numpy(label)
            vel = to_numpy(vel)
            if id_ >= 0:
                self._add_row(frame_id, label, id_, *vel)
                i += 1
        self._update_frame_id_mapping(frame_id, i)


@OUTPUTS.register_module()
class CsvActions(BaseCsvOutput):
    SUPPORTED_PRECISION = {-1: str}

    def __init__(
        self,
        path: str,
        instance_data: str = "pred_track_instances",
        ids_field: str = "instances_id",
        metainfo: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            path=path,
            precision=-1,
            columns=["action", "action_scores"],
            instance_data=instance_data,
            ids_field=ids_field,
        )
        assert self.instance_data in self.supported_instance_data, f"The provided instance_data must be one one {self.supported_instance_data}"
        self.metainfo = metainfo
        if self.metainfo is not None:
            self.metainfo = parse_pose_metainfo(dict(from_file=self.metainfo))

    def __call__(self, det_data_sample: dict):
        instance_data, frame_id = self._get_ds_info(det_data_sample)
        ids = self._set_ids(instance_data)
        i = 0
        for id_, label, action, action_scores in zip(
            ids,
            instance_data["labels"],
            instance_data["actions"],
            instance_data["action_scores"],
        ):
            if not isinstance(action, str) and self.metainfo is not None:
                action = self.metainfo["actions"][action]
            self._add_row(frame_id, label, id_, action, action_scores)
            i += 1
        self._update_frame_id_mapping(frame_id, i)

    def read(self) -> None:
        """Load a csv and a mapping (for faster __getitem__)"""
        assert os.path.exists(self.path), f"{self.path} does not exist."
        self.results = pd.read_csv(self.path, keep_default_na=True)
        self.results.fillna("", inplace=True)
        self.columns = self.results.columns.to_list()
        self.results = self.results.values.tolist()
        if os.path.exists(self.mapping_path):
            self.frame_id_mapping = np.load(self.mapping_path, allow_pickle=True).item()
        else:
            self._infer_mapping()
        if not self.results or not self.frame_id_mapping:
            print_log(f"{self.path} is empty.", level=logging.WARNING)
        else:
            self.curr_frame_idx = self.frame_id_mapping[self.__len__() - 1][1] + 1
