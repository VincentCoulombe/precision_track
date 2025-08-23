import abc
from typing import List, Optional, Tuple

import numpy as np
import supervision as sv

from precision_track.registry import VISUALIZERS
from precision_track.utils import reformat

from .keypoints import EdgeAnnotator, Keypoints, VertexAnnotator
from .palette import ColorPalette


class Annotation(metaclass=abc.ABCMeta):

    def __init__(self, palette: Optional[dict]):
        if palette is None:
            self.palette = ColorPalette()
        else:
            assert isinstance(palette, dict)
            self.palette = ColorPalette(**palette)
        self.color_lookup = sv.ColorLookup.TRACK

    @abc.abstractmethod
    def __call__(
        self,
        img: np.ndarray,
        output_values: np.ndarray,
    ) -> np.ndarray:
        assert isinstance(img, np.ndarray)
        assert img.size >= 2
        assert isinstance(output_values, np.ndarray)


class DetAnnotation(Annotation):
    SUPPORTED_FORMAT = ["cxcywh", "xywh"]

    def __init__(self, palette: Optional[dict] = None, format: Optional[str] = "cxcywh"):
        super().__init__(palette)
        assert format in self.SUPPORTED_FORMAT, f"The requested format ({format}) is not in the supported formats: {self.SUPPORTED_FORMAT}."
        self.format = format

    def to_supervision(self, output_values: np.ndarray) -> sv.Detections:
        assert isinstance(output_values, np.ndarray)
        if output_values.size < 2:
            return
        return sv.Detections(
            confidence=output_values[:, -1],
            xyxy=reformat(output_values[:, 3:7], self.format, "xyxy"),
            class_id=output_values[:, 1],
            mask=None,
            tracker_id=output_values[:, 2],
        )


class KptAnnotation(Annotation):

    def __init__(
        self,
        palette: Optional[dict] = None,
        confidence_threshold: Optional[float] = 0.5,
        with_scores: Optional[bool] = True,
    ):
        super().__init__(palette)
        assert 0.0 <= confidence_threshold <= 1.0
        self.confidence_threshold = confidence_threshold
        self.with_scores = with_scores

    def to_supervision(
        self,
        output_values: np.ndarray,
    ) -> Keypoints:
        assert isinstance(output_values, np.ndarray)
        if output_values.size < 2:
            return
        if self.with_scores:
            keypoint_scores = output_values[:, 5::3]
            xs = output_values[:, 3::3]
            ys = output_values[:, 4::3]
            keypoints = np.dstack((xs, ys)).ravel().reshape(output_values.shape[0], -1, 2)
            conf_mask = keypoint_scores < self.confidence_threshold
            conf_mask = conf_mask.reshape(output_values.shape[0], -1, 1)
            conf_mask = np.concatenate((conf_mask, conf_mask), axis=2)
            keypoints[conf_mask] = 0.0
        else:
            keypoints = output_values[:, 3:].reshape(output_values.shape[0], -1, 2)
        return Keypoints(
            xy=keypoints,
            class_id=output_values[:, 1],
            tracker_id=output_values[:, 2],
        )


class Label(DetAnnotation):
    SUPPORTED_INFO = ["class", "score", "id"]

    def __init__(
        self,
        palette: Optional[dict] = None,
        format: Optional[str] = "cxcywh",
        info: Optional[List[str]] = None,
        class_id_to_class: Optional[dict] = None,
        label_position: Optional[str] = "TOP_CENTER",
        text_color: Optional[List[int]] = None,
        text_scale: Optional[float] = 0.5,
        text_thickness: Optional[int] = 1,
        text_padding: Optional[int] = 10,
        border_radius: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(palette, format)
        assert label_position in sv.Position.list(), f"Label position must be one of {sv.Position.list()}, but got {label_position} instead."
        self.info = ["class"] if info is None else info
        assert all(
            element in self.SUPPORTED_INFO for element in info
        ), f"One of the requested info is not supported. The supported infos are: {self.SUPPORTED_INFO}."
        self.class_id_to_class = {} if class_id_to_class is None else class_id_to_class
        assert isinstance(self.class_id_to_class, dict)
        if text_color is None:
            text_color = [0, 0, 0]
        assert isinstance(text_color, list)
        for t_c in text_color:
            assert isinstance(t_c, int)
            assert t_c >= 0 and t_c < 255
        self.label_annotator = sv.LabelAnnotator(
            color=self.palette,
            text_position=sv.Position[label_position],
            color_lookup=self.color_lookup,
            text_color=sv.Color(*text_color),
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            border_radius=border_radius,
        )

    def __call__(
        self,
        img: np.ndarray,
        output_values: np.ndarray,
        additionnal_labels: Optional[np.ndarray],
        *args,
        **kwargs,
    ) -> np.ndarray:
        super().__call__(img, output_values)
        detections = self.to_supervision(output_values)
        if detections is None:
            return img
        labels = self.format_labels(detections, additionnal_labels)
        return self.label_annotator.annotate(img, detections, labels)

    def format_labels(self, detections: sv.Detections, additionnal_labels: Optional[np.ndarray] = None):
        labels = []
        additionnal_label = ""
        for cls_, conf, lbl in zip(detections.class_id, detections.confidence, detections.tracker_id):
            if isinstance(additionnal_labels, np.ndarray) and additionnal_labels.ndim == 2:
                additional_track_ids = additionnal_labels[:, 2].astype(int)
                addtionnal_mask = additional_track_ids == lbl
                if (additional_track_ids == lbl).any():
                    additionnal_label = f"| {additionnal_labels[addtionnal_mask][:, 3][0]}"
            cls_ = self.class_id_to_class.get(cls_, int(cls_)) if "class" in self.info else ""
            conf = f": {conf:.2f}" if "score" in self.info else ""
            lbl = f" {int(lbl)}" if "id" in self.info else ""
            labels.append(f"{cls_}{lbl}{conf}{additionnal_label}")
        return labels


@VISUALIZERS.register_module()
class Dot(DetAnnotation):

    def __init__(self, palette: Optional[dict] = None, radius: int = 5, format: Optional[str] = "cxcywh", *args, **kwargs) -> None:
        """Can be painted by a BoundingBoxePainter.

        Args:
            palette (Optional[dict], optional): The color palette. Defaults to None.
            radius (int, optional): The radius of the Dot. Defaults to 5.
        """
        super().__init__(palette, format)
        self.dot = sv.DotAnnotator(
            color=self.palette,
            radius=radius,
            color_lookup=self.color_lookup,
            *args,
            **kwargs,
        )

    def __call__(self, img: np.ndarray, output_values: np.ndarray, *args, **kwargs) -> None:
        super().__call__(img, output_values)
        detections = self.to_supervision(output_values)
        if detections is None:
            return img
        return self.dot.annotate(img, detections)


@VISUALIZERS.register_module()
class Corner(DetAnnotation):

    def __init__(
        self,
        palette: Optional[dict] = None,
        format: Optional[str] = "cxcywh",
        thickness=2,
        corner_length=10,
        *args,
        **kwargs,
    ) -> None:
        """Can be painted by a BoundingBoxePainter.

        Args:
            palette (Optional[dict], optional): The color palette. Defaults to None.
            thickness (int, optional): The thickness of the lines. Defaults to 2.
            corner_length (int, optional): The size of the corners. Defaults to 10.
        """
        super().__init__(palette, format)
        self.box = sv.BoxCornerAnnotator(
            color=self.palette,
            thickness=thickness,
            corner_length=corner_length,
            color_lookup=self.color_lookup,
            *args,
            **kwargs,
        )

    def __call__(self, img: np.ndarray, output_values: np.ndarray, *args, **kwargs) -> None:
        super().__call__(img, output_values)
        detections = self.to_supervision(output_values)
        if detections is None:
            return img
        return self.box.annotate(img, detections)


@VISUALIZERS.register_module()
class Box(DetAnnotation):

    def __init__(self, palette: Optional[dict] = None, format: Optional[str] = "cxcywh", thickness=2, *args, **kwargs) -> None:
        """Can be painted by a BoundingBoxePainter.

        Args:
            palette (Optional[dict], optional): The color palette. Defaults to None.
            thickness (int, optional): The thickness of the lines. Defaults to 2.
        """
        super().__init__(palette, format)
        self.box = sv.BoxAnnotator(
            color=self.palette,
            thickness=thickness,
            color_lookup=self.color_lookup,
            *args,
            **kwargs,
        )

    def __call__(self, img: np.ndarray, output_values: np.ndarray, *args, **kwargs) -> None:
        super().__call__(img, output_values)
        detections = self.to_supervision(output_values)
        if detections is None:
            return img
        return self.box.annotate(img, detections)


@VISUALIZERS.register_module()
class Ellipse(DetAnnotation):

    def __init__(self, palette: Optional[dict] = None, format: Optional[str] = "cxcywh", thickness=2, *args, **kwargs) -> None:
        """Can be painted by a BoundingBoxePainter.

        Args:
            palette (Optional[dict], optional): The color palette. Defaults to None.
            thickness (int, optional): The thickness of the lines. Defaults to 2.
        """
        super().__init__(palette, format)
        self.elipse = sv.EllipseAnnotator(
            color=self.palette,
            thickness=thickness,
            color_lookup=self.color_lookup,
            *args,
            **kwargs,
        )

    def __call__(self, img: np.ndarray, output_values: np.ndarray, *args, **kwargs) -> None:
        super().__call__(img, output_values)
        detections = self.to_supervision(output_values)
        if detections is None:
            return img
        return self.elipse.annotate(img, detections)


@VISUALIZERS.register_module()
class Mask(DetAnnotation):

    def __init__(self, palette: Optional[dict] = None, opacity=1, *args, **kwargs) -> None:
        super().__init__(palette)
        self.mask = sv.HaloAnnotator(
            color=self.palette,
            opacity=opacity,
            color_lookup=self.color_lookup,
            *args,
            **kwargs,
        )

    def __call__(self, img: np.ndarray, output_values: np.ndarray, *args, **kwargs) -> None:
        raise NotImplementedError


class Joint(KptAnnotation):

    def __init__(
        self,
        palette: Optional[dict] = None,
        radius: int = 5,
        with_scores: Optional[bool] = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(palette=palette, with_scores=with_scores)
        self.joint = VertexAnnotator(
            color=self.palette,
            radius=radius,
            color_lookup=self.color_lookup,
            *args,
            **kwargs,
        )

    def __call__(self, img: np.ndarray, output_values: np.ndarray, *args, **kwargs) -> None:
        super().__call__(img, output_values)
        keypoints = self.to_supervision(output_values)
        if keypoints is None:
            return img
        return self.joint.annotate(img, keypoints)


class Link(KptAnnotation):

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        palette: Optional[dict] = None,
        thickness: Optional[int] = 1,
        with_scores: Optional[bool] = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(palette=palette, with_scores=with_scores)
        self.links = EdgeAnnotator(
            color=self.palette,
            edges=edges,
            thickness=thickness,
            color_lookup=self.color_lookup,
            *args,
            **kwargs,
        )

    def __call__(
        self,
        img: np.ndarray,
        output_values: np.ndarray,
        verify: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> None:
        super().__call__(img, output_values)
        if verify:
            return
        keypoints = self.to_supervision(output_values)
        if keypoints is None:
            return img
        return self.links.annotate(img, keypoints)


class Arrow(Link):

    def __init__(
        self,
        amplitude: Optional[int] = 5,
        color: Optional[List[int]] = None,
        thickness: int = 1,
        *args,
        **kwargs,
    ) -> None:
        edges = [(0, 1), (1, 2), (1, 3)]
        if color is None:
            color = [0, 0, 0]
        palette = dict(size=1, names=[[tuple([c / 255 for c in color])]])
        super().__init__(palette=palette, edges=edges, thickness=thickness, *args, **kwargs)
        self.tip_size = amplitude * 5
        self.shaft_size = amplitude

    def __call__(
        self,
        img: np.ndarray,
        output_values: np.ndarray,
        anchor_coords: np.ndarray,
        *args,
        **kwargs,
    ) -> None:
        _ = super().__call__(img, output_values, verify=True)
        if output_values.size < 2 or output_values[:, -2:].size != anchor_coords.size:
            return img
        vector = np.stack([anchor_coords, output_values[:, -2:] + anchor_coords], axis=1)
        if vector.shape[0] == 0:
            return None
        keypoints = Keypoints(
            xy=vector.reshape(-1, 2, 2),
            class_id=output_values[:, 1],
            tracker_id=output_values[:, 2],
        )
        x1, y1, x2, y2 = (
            keypoints.xy[..., 0, 0],
            keypoints.xy[..., 0, 1],
            keypoints.xy[..., 1, 0],
            keypoints.xy[..., 1, 1],
        )
        magnetudes = np.abs(output_values[:, -2:])
        direction_of_vector = np.arctan2(y2 - y1, x2 - x1)
        x2 = x2 + magnetudes[:, 0] * self.shaft_size * np.cos(direction_of_vector)
        y2 = y2 + magnetudes[:, 1] * self.shaft_size * np.sin(direction_of_vector)
        x3 = x2 - self.tip_size * np.cos(direction_of_vector + np.pi / 6)
        y3 = y2 - self.tip_size * np.sin(direction_of_vector + np.pi / 6)
        x4 = x2 - self.tip_size * np.cos(direction_of_vector - np.pi / 6)
        y4 = y2 - self.tip_size * np.sin(direction_of_vector - np.pi / 6)
        keypoints.xy = np.stack(
            [
                np.stack([x1, y1], axis=-1),
                np.stack([x2, y2], axis=-1),
                np.stack([x3, y3], axis=-1),
                np.stack([x4, y4], axis=-1),
            ],
            axis=1,
        )
        return self.links.annotate(img, keypoints)
