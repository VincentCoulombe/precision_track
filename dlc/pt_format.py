"""Standalone re-implementation of the PrecisionTrack CSV writers.

DeepLabCut lives in its own Python 3.10 environment (see ``dlc/README.md``), which
cannot import ``precision_track`` (Python 3.11, torch 2.7.1, numpy 1.26, mmengine).
This module therefore mirrors ``precision_track/outputs/csv.py`` using nothing but
numpy and pandas, so that ``dlc/track.py`` emits files that are byte-compatible with
``CsvBoundingBoxes`` and ``CsvKeypoints``.

The ``__call__(data_sample)`` interface is kept identical to the real writers on
purpose: ``tests/test_dlc_format_parity.py`` feeds the very same ``dict`` to both
implementations and asserts the resulting dataframes match. Any drift in the real
writers breaks that test.
"""

import abc
import os
from collections import OrderedDict
from typing import Any, List, Tuple

import numpy as np
import pandas as pd


def to_numpy(x):
    if isinstance(x, list):
        return np.array(x)
    elif isinstance(x, (np.ndarray, int, float, str, np.generic)):
        return x
    raise TypeError(f"{type(x)} not yet supported.")


def keypoints_cxcywh(keypoints: np.ndarray) -> np.ndarray:
    """Tightest bounding box enclosing the non-NaN keypoints of an instance.

    Mirrors ``precision_track.utils.formatting.keypoints_cxcywh``.
    """
    mask = ~np.isnan(keypoints).any(1)
    if not mask.any():
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    keypoints = keypoints[mask]
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    w, h = xmax - xmin, ymax - ymin
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    return np.array([cx, cy, w, h], dtype=np.float32)


def cxcywh_xywh_1d(cxcywh: np.ndarray) -> np.ndarray:
    if np.isnan(cxcywh).any():
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cx, cy, w, h = cxcywh
    return np.array([cx - w / 2, cy - h / 2, w, h], dtype=np.float32)


def xywh_cxcywh_1d(xywh: np.ndarray) -> np.ndarray:
    if np.isnan(xywh).any():
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    x, y, w, h = xywh
    return np.array([x + w / 2, y + h / 2, w, h], dtype=np.float32)


def cxcywh_xyxy_1d(cxcywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = cxcywh
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def xyxy_cxcywh_1d(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)


_TRANSFORMATIONS = {
    "cxcywh_xywh": cxcywh_xywh_1d,
    "xywh_cxcywh": xywh_cxcywh_1d,
    "cxcywh_xyxy": cxcywh_xyxy_1d,
    "xyxy_cxcywh": xyxy_cxcywh_1d,
}


def reformat(instance: np.ndarray, old: str, new: str) -> np.ndarray:
    """Reformat a single bounding box from the ``old`` to the ``new`` format.

    Only the 1D subset of ``precision_track.utils.formatting.reformat`` that ``dlc/``
    actually needs is supported.
    """
    if old == new:
        return instance
    key = f"{old}_{new}"
    if key not in _TRANSFORMATIONS:
        raise NotImplementedError(f"Reformatting from '{old}' to '{new}' is not supported.")
    return _TRANSFORMATIONS[key](instance)


class BaseCsvOutput(metaclass=abc.ABCMeta):
    SUPPORTED_PRECISION = {32: "float32", 64: "float64"}
    EXTENSION = ".csv"

    def __init__(
        self,
        path: str,
        instance_data: str,
        columns: List[str],
        confidence_threshold: float = 0.5,
        precision: int = 32,
        ids_field: str = "instances_id",
    ) -> None:
        self.reset()
        if precision not in self.SUPPORTED_PRECISION:
            raise ValueError(f"Precision {precision} not supported. Supported precisions are {list(self.SUPPORTED_PRECISION.keys())}")
        self.precision = precision
        self.columns = columns
        self.confidence_threshold = confidence_threshold
        self.supported_instance_data = ["pred_track_instances", "pred_instances", "next_frame_pred_track_instances"]
        self.instance_data = instance_data
        self._setup_path(path)
        self.ids_field = ids_field

    def _setup_path(self, path: str) -> None:
        raw_path, _ = os.path.splitext(path)
        self.path = os.path.abspath(f"{raw_path}{self.EXTENSION}")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    @abc.abstractmethod
    def __call__(self, data: dict) -> None:
        pass

    @property
    def results(self):
        return self._results

    def __len__(self) -> int:
        if self.frame_id_mapping:
            return max(self.frame_id_mapping) + 1
        return 0

    def __getitem__(self, idx: int) -> List[Any]:
        idx_range = self.frame_id_mapping.get(idx)
        if idx_range is None:
            return [[]]
        return self._results[idx_range[0] : idx_range[1]]

    def reset(self) -> None:
        self.frame_id_mapping = OrderedDict()
        self._results = []
        self.curr_frame_idx = 0

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._results, columns=["frame_id", "class_id", "instance_id"] + self.columns)
        df["frame_id"] = df["frame_id"].astype("uint32")
        df["class_id"] = df["class_id"].astype("uint16")
        df["instance_id"] = df["instance_id"].astype("int16")
        for col in self.columns:
            df[col] = df[col].astype(self.SUPPORTED_PRECISION[self.precision])
        return df

    def save(self) -> None:
        self.to_dataframe().to_csv(self.path, index=False)
        print(f"Saved output: {self.path}")

    def _add_row(self, *args) -> None:
        self._results.append(list(args))

    def _update_frame_id_mapping(self, frame_id: int, increment: int) -> None:
        if frame_id not in self.frame_id_mapping:
            curr_frame_idx = self.curr_frame_idx + increment
            self.frame_id_mapping[frame_id] = (self.curr_frame_idx, curr_frame_idx)
            self.curr_frame_idx = curr_frame_idx

    def _set_ids(self, instance_data: dict):
        return (
            np.zeros_like(to_numpy(instance_data["labels"])) - 1
            if self.instance_data
            not in [
                "pred_track_instances",
                "next_frame_pred_track_instances",
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

    @abc.abstractmethod
    def scale(self, ori_scale: Tuple[int, int], new_scale: Tuple[int, int]) -> None:
        pass


class CsvBoundingBoxes(BaseCsvOutput):
    """Writes ``frame_id,class_id,instance_id,cx,cy,w,h,score`` (default format)."""

    SUPPORTED_FORMATS = ["cxcywh", "xyxy", "xywh"]

    def __init__(
        self,
        path: str,
        subtype: str = "tracked_bboxes",
        bbox_format: str = "cxcywh",
        instance_data: str = "pred_instances",
        confidence_threshold: float = 0.1,
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
        self.subtype = str(subtype)

    def __call__(self, det_data_sample: dict) -> None:
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
                id_ >= 0 and self.instance_data in ["pred_track_instances", "next_frame_pred_track_instances"]
            ):
                if self.bbox_format != self.save_bbox_format_str:
                    bbox = reformat(bbox, self.bbox_format, self.save_bbox_format_str)
                self._add_row(frame_id, label, id_, *bbox, score)
                i += 1
        self._update_frame_id_mapping(frame_id, i)

    def scale(self, ori_scale: Tuple[int, int], new_scale: Tuple[int, int]) -> None:
        if not self._results:
            return
        np_results = np.array(self._results, dtype=np.float32)
        ratio = np.array(new_scale, dtype=np.float32) / np.array(ori_scale, dtype=np.float32)
        np_results[:, 3:7:2] *= ratio[0]
        np_results[:, 4:7:2] *= ratio[1]
        self._results = np_results.tolist()


class CsvKeypoints(BaseCsvOutput):
    """Writes ``frame_id,class_id,instance_id,x0,y0,score0,x1,y1,score1,...``."""

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

    def __call__(self, det_data_sample: dict) -> None:
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

    def _add_row(self, frame_id, class_id, object_id, keypoints) -> None:
        self._set_columns(frame_id, keypoints)
        super()._add_row(frame_id, class_id, object_id, *keypoints)

    def _set_columns(self, frame_id: int, keypoints: list) -> None:
        if not self.columns:
            self.columns = [f"{coord}{i}" for i in range(len(keypoints) // 3) for coord in ("x", "y", "score")]
        else:
            assert len(keypoints) == len(self.columns), f"Inconsistent number of keypoints: {len(keypoints)}, expected: {len(self.columns)} as frame{frame_id}"

    def reset(self) -> None:
        self.columns = []
        super().reset()

    def scale(self, ori_scale: Tuple[int, int], new_scale: Tuple[int, int]) -> None:
        if not self._results:
            return
        np_results = np.array(self._results, dtype=np.float32)
        ratio = np.array(new_scale, dtype=np.float64) / np.array(ori_scale, dtype=np.float64)
        np_results[:, 3:-1:3] *= ratio[0]
        np_results[:, 4:-1:3] *= ratio[1]
        self._results = np_results.tolist()
