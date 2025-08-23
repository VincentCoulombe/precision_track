from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv
from supervision.annotators.base import ImageType
from supervision.annotators.utils import ColorLookup, resolve_color
from supervision.detection.utils import get_data_item, is_data_equal
from supervision.validators import validate_class_id, validate_data, validate_keypoint_confidence, validate_tracker_id, validate_xy

from .palette import ColorPalette


def validate_keypoints_fields(
    xy: Any,
    class_id: Any,
    tracker_id: Any,
    confidence: Any,
    data: Dict[str, Any],
) -> None:
    n = len(xy)
    m = len(xy[0]) if len(xy) > 0 else 0
    validate_xy(xy, n, m)
    validate_class_id(class_id, n)
    validate_tracker_id(tracker_id, n)
    validate_keypoint_confidence(confidence, n, m)
    validate_data(data, n)


@dataclass
class Keypoints(sv.KeyPoints):
    xy: npt.NDArray[np.float32]
    class_id: Optional[npt.NDArray[np.int_]] = None
    tracker_id: Optional[np.ndarray] = None
    confidence: Optional[npt.NDArray[np.float32]] = None
    data: Dict[str, Union[npt.NDArray[Any], List]] = field(default_factory=dict)

    def __post_init__(self):
        validate_keypoints_fields(
            xy=self.xy,
            confidence=self.confidence,
            tracker_id=self.tracker_id,
            class_id=self.class_id,
            data=self.data,
        )

    def __len__(self) -> int:
        return len(self.xy)

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[float],
            Optional[int],
            Optional[int],
            Dict[str, Union[np.ndarray, List]],
        ]
    ]:
        for i in range(len(self.xy)):
            yield (
                self.xy[i],
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
                get_data_item(self.data, i),
            )

    def __eq__(self, other: Keypoints) -> bool:
        return all(
            [
                np.array_equal(self.xy, other.xy),
                np.array_equal(self.class_id, other.class_id),
                np.array_equal(self.confidence, other.confidence),
                np.array_equal(self.tracker_id, other.tracker_id),
                is_data_equal(self.data, other.data),
            ]
        )

    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray, str]) -> Union[Keypoints, List, np.ndarray, None]:
        if isinstance(index, str):
            return self.data.get(index)
        if isinstance(index, int):
            index = [index]
        return Keypoints(
            xy=self.xy[index],
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            tracker_id=self.tracker_id[index] if self.tracker_id is not None else None,
            data=get_data_item(self.data, index),
        )


class VertexAnnotator(sv.VertexAnnotator):

    def __init__(
        self,
        color: ColorPalette,
        color_lookup: ColorLookup,
        radius: int = 4,
    ) -> None:
        self.color = color
        self.color_lookup = color_lookup
        self.radius = radius

    def annotate(self, scene: ImageType, key_points: Keypoints) -> ImageType:

        if len(key_points) == 0:
            return scene

        for i, xy in enumerate(key_points.xy):
            color = resolve_color(
                color=self.color,
                detections=key_points,
                detection_idx=i,
                color_lookup=self.color_lookup,
            )
            for x, y in xy:
                missing_x = np.allclose(x, 0)
                missing_y = np.allclose(y, 0)
                if missing_x or missing_y:
                    continue
                cv2.circle(
                    img=scene,
                    center=(int(x), int(y)),
                    radius=self.radius,
                    color=color.as_bgr(),
                    thickness=-1,
                )

        return scene


class EdgeAnnotator(sv.EdgeAnnotator):

    def __init__(
        self,
        color: ColorPalette,
        color_lookup: ColorLookup,
        edges: List[Tuple[int, int]],
        thickness: int = 3,
    ) -> None:
        self.color_lookup = color_lookup
        self.thickness = thickness
        self.edges = edges
        self.color = color

    def annotate(self, scene: ImageType, key_points: Keypoints) -> ImageType:
        if len(key_points) == 0:
            return scene

        for i, xy in enumerate(key_points.xy):
            color = resolve_color(
                color=self.color,
                detections=key_points,
                detection_idx=i,
                color_lookup=self.color_lookup,
            )

            for class_a, class_b in self.edges:
                xy_a = xy[class_a]
                xy_b = xy[class_b]
                missing_a = np.allclose(xy_a, 0)
                missing_b = np.allclose(xy_b, 0)
                if missing_a or missing_b:
                    continue

                cv2.line(
                    img=scene,
                    pt1=(int(xy_a[0]), int(xy_a[1])),
                    pt2=(int(xy_b[0]), int(xy_b[1])),
                    color=color.as_bgr(),
                    thickness=self.thickness,
                )

        return scene
