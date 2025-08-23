import abc
from typing import Dict, List, Optional

import numpy as np
from mmengine.config import Config

from precision_track.registry import VISUALIZERS
from precision_track.utils import parse_pose_metainfo

from .annotations import Arrow, Dot, Joint, Label, Link


class BasePainter(metaclass=abc.ABCMeta):

    def __init__(self, supported_outputs: List[str]) -> None:
        self.annotations = []
        self.supported_outputs = supported_outputs

    def __call__(self, frame: np.ndarray, outputs: List[Dict[str, np.ndarray]], idx: int) -> None:
        """Paint an output on a frame.

        Args:
            frame (np.ndarray): The frame to be painted on.
            outputs (List[Dict[str, np.ndarray]]): The outputs, the painters will paint what they support.
            idx (int): The frame id
        """
        for output in outputs:
            output_name, output_values = next(iter(output.items()))
            if self.output_is_supported(output_name):
                for annotation in self.annotations:
                    frame = annotation(img=frame, output_values=output_values, idx=idx)
        return frame

    def output_is_supported(self, output_class: str):
        return output_class in self.supported_outputs


@VISUALIZERS.register_module()
class BoundingBoxPainter(BasePainter):

    def __init__(self, annotations: List[Config], palette: Optional[dict]) -> None:
        """Paints any combination of supported annotations. Check the
        annotations module to see which ones are supported.

        Args:
            annotations (List[Config]): A list of supported annotations
            palette (Optional[dict]): The color palette
        """
        super().__init__(["CsvBoundingBoxes"])
        self.annotations = []
        for ann in annotations:
            ann.update({"palette": palette})
            self.annotations.append(VISUALIZERS.build(ann))


@VISUALIZERS.register_module()
class SearchAreaPainter(BasePainter):

    def __init__(self, annotations: List[Config], color: Optional[List[int]] = None, *args, **kwargs) -> None:
        """Paints any combination of supported annotations. Check the
        annotations module to see which ones are supported.

        Args:
            annotations (List[Config]): A list of supported annotations
            palette (Optional[dict]): The color palette
        """
        if color is None:
            color = [255, 255, 255]
        assert isinstance(color, list)
        for c in color:
            assert 0 <= c <= 255
        super().__init__(["CsvSearchAreas"])
        self.annotations = []
        for ann in annotations:
            ann.update({"palette": dict(nan_color=color)})
            self.annotations.append(VISUALIZERS.build(ann))

    def __call__(self, frame: np.ndarray, outputs: List[Dict[str, np.ndarray]], idx: int) -> None:
        for output in outputs:
            output_name, output_values = next(iter(output.items()))
            if self.output_is_supported(output_name):
                for annotation in self.annotations:
                    frame = annotation(
                        img=frame,
                        output_values=(
                            np.concatenate(
                                (output_values, np.ones((output_values.shape[0], 1))),
                                axis=1,
                            )
                            if output_values.size > 1
                            else output_values
                        ),
                        idx=idx,
                    )
        return frame


class MaskPainter(BasePainter):

    def __init__(self, annotations: List[Config], palette: Optional[dict]) -> None:
        raise NotImplementedError


@VISUALIZERS.register_module()
class ValidationPainter(BasePainter):

    def __init__(self, radius: Optional[int] = 20, palette: Optional[dict] = None, *args, **kwargs) -> None:
        """Draw a Dot on the validation.

        Args:
            radius (Optional[int], optional): The radius of the dot. Defaults to 20.
            palette (Optional[dict]): The color palette
        """
        super().__init__(["CsvValidations"])
        self.annotations = [Dot(radius=radius, palette=palette)]


@VISUALIZERS.register_module()
class KeypointsPainter(BasePainter):
    """Paints keypoints and the skeleton if a metafile_path is provided.

    Args:
        BasePainter (_type_): _description_
    """

    def __init__(
        self,
        metafile_path: Optional[str] = None,
        palette: Optional[dict] = None,
        joint_radius: Optional[int] = 5,
        link_thickness: Optional[int] = 1,
        with_scores: Optional[bool] = True,
    ) -> None:
        """Paint keypoints. Also paint skeletons if a metafile_path is
        provided.

        Args:
            metafile_path (Optional[str], optional): the metafile needed to draw the skeletons. Defaults to None.
            palette (Optional[dict], optional): The color palette. Defaults to None.
            joint_radius (int, optional): The size of the keypoints. Defaults to 5.
            link_thickness (int, optional): The size of the edges on the skeletons. Defaults to 1.
            with_scores (bool, optional): If the input data also contain the keypoint scores.
        """
        super().__init__(["CsvKeypoints"])
        self.annotations = [
            Joint(
                radius=joint_radius,
                palette=palette,
                with_scores=with_scores,
            )
        ]
        if metafile_path is not None:
            self.metadata = parse_pose_metainfo({"from_file": metafile_path})
            self.annotations.append(
                Link(
                    edges=self.metadata["skeleton_links"],
                    thickness=link_thickness,
                    palette=palette,
                    with_scores=with_scores,
                )
            )


@VISUALIZERS.register_module()
class VelocityPainter(BasePainter):

    def __init__(
        self, amplitude: Optional[int] = 20, anchor: Optional[int] = 0, color: Optional[List[int]] = None, thickness: int = 1, *args, **kwargs
    ) -> None:
        """Draw the velocity of the track via an arrow.

        Args:
            amplitude (Optional[int], optional): The size of the arrow. Defaults to 50.
            anchor (Optional[int], optional): To which keypoint is the arrow attached. Defaults to 0.
            color (Optional[List[int]], optional): The color of the arrow. Defaults to None.
        """
        super().__init__(["CsvVelocities"])
        self.annotations = [
            Arrow(
                color=color,
                amplitude=amplitude,
                thickness=thickness,
            )
        ]
        self.anchor = anchor

    def __call__(self, frame: np.ndarray, outputs: List[Dict[str, np.ndarray]], idx: int) -> None:
        output_values = None
        anchors = None
        for output in outputs:
            output_name, temp = next(iter(output.items()))
            if self.output_is_supported(output_name):
                output_values = temp.copy()
            if output_name in ["CsvKeypoints"] and temp.size > 1:
                anchors = temp[:, 3:].reshape(temp.shape[0], -1, 3)[:, self.anchor, :2]
        if output_values is not None and anchors is not None:
            frame = self.annotations[0](img=frame, output_values=output_values, anchor_coords=anchors, idx=idx)
        return frame


@VISUALIZERS.register_module()
class LabelPainter(BasePainter):

    def __init__(
        self,
        palette: Optional[dict] = None,
        format: Optional[str] = "cxcywh",
        info: Optional[List[str]] = None,
        metafile_path: Optional[dict] = None,
        label_position: Optional[str] = "TOP_CENTER",
        text_color: Optional[List[int]] = None,
        text_scale: Optional[float] = 0.5,
        text_thickness: Optional[int] = 1,
        text_padding: Optional[int] = 10,
        border_radius: Optional[int] = 1,
        *args,
        **kwargs,
    ) -> None:
        """Paint a label.

        Args:
            palette (Optional[dict], optional): The color palette. Defaults to None.
            info (Optional[List[str]], optional): A list of what info to display. Check Label to see supported infos. Defaults to None.
            metafile_path (Optional[str], optional): the metafile needed to draw the skeletons. Defaults to None.
            label_position (Optional[str], optional): Where are the labels relative to the detections. Defaults to "TOP_CENTER".
            text_color (Optional[List[int]], optional): The color of the text. Defaults to None.
            text_scale (Optional[float], optional): The scale of the text. Defaults to 0.5.
            text_thickness (Optional[int], optional): The thickness of the text. Defaults to 1.
            text_padding (Optional[int], optional): The padding of the text. Defaults to 10.
            border_radius (Optional[int], optional): The border radius of the label's boxes. Defaults to 1.
        """
        super().__init__(["CsvBoundingBoxes", "CsvActions"])
        class_id_to_class = None
        if metafile_path is not None:
            metadata = parse_pose_metainfo({"from_file": metafile_path})
            class_id_to_class = {i: cls_ for i, cls_ in enumerate(metadata["classes"])}
        self.annotations = [
            Label(
                palette,
                format,
                info,
                class_id_to_class,
                label_position,
                text_color,
                text_scale,
                text_thickness,
                text_padding,
                border_radius,
            )
        ]

    def __call__(self, frame: np.ndarray, outputs: List[Dict[str, np.ndarray]], idx: int) -> None:
        """Paint an output on a frame.

        Args:
            frame (np.ndarray): The frame to be painted on.
            outputs (List[Dict[str, np.ndarray]]): The outputs, the painters will paint what they support.
            idx (int): The frame id
        """
        bboxes_values = None
        action_values = None
        for output in outputs:
            output_name, output_values = next(iter(output.items()))
            if output_name == "CsvBoundingBoxes":
                bboxes_values = output_values
            elif output_name == "CsvActions":
                action_values = output_values

        for annotation in self.annotations:
            frame = annotation(img=frame, output_values=bboxes_values, additionnal_labels=action_values, idx=idx)
        return frame
