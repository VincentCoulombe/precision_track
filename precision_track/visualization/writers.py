import abc
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import yaml

from precision_track.registry import VISUALIZERS
from precision_track.utils import parse_pose_metainfo

from .palette import ColorPalette


class BaseWriter(metaclass=abc.ABCMeta):

    def __init__(
        self,
        text_anchor: List[int],
        text_color: Optional[List[int]] = None,
        text_scale: Optional[int] = 1,
        text_thickness: Optional[int] = 1,
        text_padding: Optional[int] = 3,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            text_anchor (List[int]): Where to position the text. Takes the form [x, y]
            text_color (Optional[List[int]], optional): RGB of the color. Defaults to None.
            text_scale (Optional[int], optional): Scale of the text. Defaults to 1.
            text_thickness (Optional[int], optional): Thickness of the text. Defaults to 1.
            text_padding (Optional[int], optional): Padding of the text. Defaults to 3.
        """
        self.text_anchor = sv.Point(*text_anchor)
        if text_color is None:
            text_color = [0, 0, 0]
        self.text_color = sv.Color(*text_color)
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.text_padding = text_padding
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX

    @abc.abstractmethod
    def __call__(self, frame: np.ndarray, outputs: Tuple[Dict[str, np.ndarray]], idx: int) -> None:
        """Write on a frame.

        Args:
            frame (np.ndarray): The frame to be written on.
            outputs (Tuple[Dict[str, np.ndarray]]): The outputs.
            idx (int): The frame id
        """

    def _get_text_width_height(self, text: str) -> Tuple:
        return cv2.getTextSize(
            text=text,
            fontFace=self.text_font,
            fontScale=self.text_scale,
            thickness=self.text_thickness,
        )[0]

    def _get_text_rectangle(self, text_x: int, text_y: int, text_width: int, text_height: int) -> sv.Rect:
        return sv.Rect(
            x=text_x,
            y=text_y,
            width=text_width,
            height=text_height,
        ).pad(self.text_padding)

    @staticmethod
    def _pad_frame(frame: np.ndarray, text_rect: sv.Rect, frame_shape: tuple) -> np.ndarray:
        top_pad = max(-text_rect.top_left.y, 0)
        left_pad = max(-text_rect.top_left.x, 0)
        bottom_pad = max(text_rect.bottom_right.y, frame_shape[1]) - frame_shape[1]
        right_pad = max(text_rect.bottom_right.x, frame_shape[0]) - frame_shape[0]
        text_rect.x = max(0, text_rect.x)
        text_rect.y = max(0, text_rect.y)
        frame = np.concatenate(
            [
                np.zeros((top_pad, frame_shape[1], 3), dtype=np.uint8),
                frame,
                np.zeros((bottom_pad, frame_shape[1], 3), dtype=np.uint8),
            ],
        )
        frame = np.concatenate(
            [
                np.zeros((frame.shape[0], left_pad, 3), dtype=np.uint8),
                frame,
                np.zeros((frame.shape[0], right_pad, 3), dtype=np.uint8),
            ],
            axis=1,
        )
        return frame

    def _get_anchor(self) -> Tuple[int, int]:
        return self.text_anchor.as_xy_int_tuple()

    def write(self, frame: np.ndarray, text: str):
        text_width, text_height = self._get_text_width_height(text)

        x, y = self._get_anchor()

        cv2.putText(
            img=frame,
            text=text,
            org=(x - text_width // 2, y + text_height // 2),
            fontFace=self.text_font,
            fontScale=self.text_scale,
            color=self.text_color.as_bgr(),
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )
        return frame


@VISUALIZERS.register_module()
class FrameIdWriter(BaseWriter):

    def __call__(self, frame: np.ndarray, _: Tuple[Dict[str, np.ndarray]], idx: int) -> None:
        return self.write(frame, f"Frame {idx}")


@VISUALIZERS.register_module()
class CorrectionWriter(BaseWriter):

    def __init__(self, metainfo: str, color: Optional[List[int]] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bg_color = sv.Color(*(color if color is not None else [255, 0, 0]))
        self.metainfo_classes = {i: cls_ for i, cls_ in enumerate(parse_pose_metainfo({"from_file": metainfo}).get("classes", []))}

    def _get_corrections(self, outputs: Tuple[Dict[str, np.ndarray]], idx: int) -> np.ndarray:
        for output in outputs:
            if "CsvCorrections" in output:
                data = output["CsvCorrections"]
                if data is not None and data.size > 0:
                    return data[data[:, 0].astype(int) == idx]
        return np.array([])

    def __call__(self, frame: np.ndarray, outputs: Tuple[Dict[str, np.ndarray]], idx: int) -> np.ndarray:
        corrections = self._get_corrections(outputs, idx)
        if corrections.size == 0:
            return frame
        _, line_height = self._get_text_width_height("A")
        line_step = line_height + self.text_padding * 2
        for i, row in enumerate(corrections):
            class_id = self.metainfo_classes.get(int(float(row[1])), "?")
            instance_id = int(float(row[2]))
            corrected_id = int(float(row[3]))
            text = f"Correction for class '{class_id}': {instance_id} -> {corrected_id}"
            text_width, _ = self._get_text_width_height(text)
            x = max(self.text_anchor.x, text_width // 2 + self.text_padding)
            y = self.text_anchor.y + i * line_step
            rect = self._get_text_rectangle(x - text_width // 2, y - line_height // 2, text_width, line_height)
            sv.draw_filled_rectangle(frame, rect, self.bg_color)
            original_y = self.text_anchor.y
            self.text_anchor = sv.Point(x, y)
            frame = self.write(frame, text)
            self.text_anchor = sv.Point(x, original_y)
        return frame


@VISUALIZERS.register_module()
class AppearanceDetectionWriter(BaseWriter):
    _UNIQUE_ID_PATTERN = re.compile(r"^(.+)_(\d+)$")

    def __init__(
        self,
        unique_ids: List[str],
        re_id_metainfo: str,
        metainfo: str,
        palette: Optional[dict] = None,
        text_color: Optional[List[int]] = None,
        bar_width: int = 40,
        bar_height: int = 80,
        bar_spacing: int = 10,
        cache_scores: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            text_anchor=[0, 0],
            text_color=text_color,
            text_scale=0.5,
            text_thickness=1,
            text_padding=30,
        )
        classes = []
        inst_ids = []
        for unique_id in unique_ids:
            match = self._UNIQUE_ID_PATTERN.match(unique_id)
            if match:
                cls_ = match.group(1)
                id_ = match.group(2)
                classes.append(str(cls_))
                inst_ids.append(int(id_))

        self.unique_ids = unique_ids
        self.classes = classes
        self.inst_ids = inst_ids
        self.metainfo_classes = parse_pose_metainfo({"from_file": metainfo}).get("classes", [])

        assert os.path.isfile(re_id_metainfo), f"The provided re-identification metadata file '{os.path.abspath(metainfo)}' does not exists."
        with open(re_id_metainfo, "r") as f:
            re_id_metainfo = yaml.safe_load(f)

        self.identities = re_id_metainfo.get("identities")
        assert isinstance(self.identities, list), f"The metadata file '{metainfo}' must contain a list of identities"

        self.uid2ids = dict()

        for cls_ in self.classes:
            assert cls_ in self.metainfo_classes, f"The following unique id class: {cls_} is not in the '{metainfo}' classes: {self.metainfo_classes}."

        self.class_map = {cls_: i for i, cls_ in enumerate(self.metainfo_classes)}

        self.palette = ColorPalette(**palette) if palette is not None else ColorPalette()
        self.title = "Appearance Validation Scores"
        self.title_scale = self.text_scale + 0.3
        self.title_thickness = self.text_thickness + 1
        (self.w_t, self.h_t), _ = cv2.getTextSize(
            text=self.title,
            fontFace=self.text_font,
            fontScale=self.title_scale,
            thickness=self.title_thickness,
        )
        self.bar_width = bar_width
        self.bar_height = bar_height
        self.bar_spacing = bar_spacing
        self.row_height = bar_height + 60
        self.initialized = False
        self.cache_scores = cache_scores
        self.cached_top3 = {}
        self.max_uid_width = max(self._get_text_width_height(uid)[0] for uid in unique_ids)
        self.max_identity_width = max(self._get_text_width_height(id_)[0] for id_ in self.identities)
        label_overflow = max(0, self.max_uid_width - self.bar_width)
        self.effective_bar_spacing = max(self.bar_spacing, label_overflow + 10)

    def _get_validations(self, outputs: Tuple[Dict[str, np.ndarray]]) -> np.ndarray:
        for output in outputs:
            if "CsvAppearanceValidations" in output.keys():
                return output.get("CsvAppearanceValidations", np.array([]))
        return np.array([])

    def _draw_y_axis(self, frame: np.ndarray, x: int, y: int) -> None:
        axis_x = x - 5
        cv2.line(frame, (axis_x, y), (axis_x, y + self.bar_height), (0, 0, 0), 1)
        for val, label in [(0.0, "0"), (0.5, "0.5"), (1.0, "1")]:
            tick_y = y + self.bar_height - int(val * self.bar_height)
            cv2.line(frame, (axis_x - 3, tick_y), (axis_x, tick_y), (0, 0, 0), 1)
            lw, lh = self._get_text_width_height(label)
            cv2.putText(
                img=frame,
                text=label,
                org=(axis_x - lw - 5, tick_y + lh // 2),
                fontFace=self.text_font,
                fontScale=self.text_scale - 0.1,
                color=self.text_color.as_bgr(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )

    def _draw_bar(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        score: float,
        label: str,
        color: sv.Color,
        show_label: bool = True,
    ) -> None:
        bar_h = int(score * self.bar_height)
        bar_top = y + self.bar_height - bar_h

        cv2.rectangle(
            frame,
            (x, bar_top),
            (x + self.bar_width, y + self.bar_height),
            color.as_bgr(),
            -1,
        )
        cv2.rectangle(
            frame,
            (x, bar_top),
            (x + self.bar_width, y + self.bar_height),
            (0, 0, 0),
            1,
        )

        if show_label:
            label_w, _ = self._get_text_width_height(label)
            label_x = x + (self.bar_width - label_w) // 2
            label_y = bar_top - 5
            cv2.putText(
                img=frame,
                text=label,
                org=(label_x, label_y),
                fontFace=self.text_font,
                fontScale=self.text_scale,
                color=self.text_color.as_bgr(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )

    def __call__(self, frame: np.ndarray, outputs: Tuple[Dict[str, np.ndarray]], idx: int) -> None:
        if not self.initialized:
            self.x = frame.shape[1] + self.text_padding
            self.y = self.text_padding
            self.initialized = True

        y_axis_padding = 30
        uid_label_padding = self.max_uid_width + 15
        bars_width = 3 * (self.bar_width + self.effective_bar_spacing)
        panel_width = max(uid_label_padding + self.max_identity_width + y_axis_padding + bars_width, self.w_t + 20)
        panel_height = len(self.unique_ids) * self.row_height + self.h_t * 3 + self.text_padding * 2

        text_rect = self._get_text_rectangle(
            self.x,
            self.y,
            panel_width,
            panel_height,
        )

        frame = self._pad_frame(frame, text_rect, frame.shape)
        sv.draw_filled_rectangle(frame, text_rect, sv.Color(255, 255, 255))

        cv2.putText(
            img=frame,
            text=self.title,
            org=(
                self.x + (panel_width - self.w_t) // 2,
                self.y + self.h_t + 10,
            ),
            fontFace=self.text_font,
            fontScale=self.title_scale,
            color=self.text_color.as_bgr(),
            thickness=self.title_thickness,
            lineType=cv2.LINE_AA,
        )

        validations = self._get_validations(outputs)

        if not self.cache_scores:
            self.cached_top3 = {}

        if validations is not None and validations.size > 0:
            for unique_id, cls_, inst_id in zip(self.unique_ids, self.classes, self.inst_ids):
                for row in validations:
                    if int(float(row[1])) == self.class_map[cls_] and int(float(row[2])) == inst_id:
                        identity = str(row[3])
                        if identity in self.identities:
                            self.uid2ids[unique_id] = identity

                        scores = row[4:].astype(float)

                        top3_indices = np.argsort(scores)[-3:][::-1]
                        top3_scores = scores[top3_indices]
                        top3_identities = [self.identities[i] for i in top3_indices]

                        self.cached_top3[unique_id] = (top3_identities, top3_scores)

        bar_start_x = self.x + uid_label_padding + y_axis_padding + self.max_identity_width + 15
        for row_idx, unique_id in enumerate(self.unique_ids):
            row_y = self.y + self.h_t * 3 + row_idx * self.row_height + 20

            id_color = self.palette.by_idx(row_idx + 1)

            text = unique_id
            identity = self.uid2ids.get(unique_id)
            if identity is not None:
                text = f"{unique_id} ({identity})"

            cv2.putText(
                img=frame,
                text=text,
                org=(self.x, row_y + self.bar_height // 2),
                fontFace=self.text_font,
                fontScale=self.text_scale,
                color=id_color.as_bgr(),
                thickness=self.text_thickness + 1,
                lineType=cv2.LINE_AA,
            )

            self._draw_y_axis(frame, bar_start_x, row_y)

            top_3 = self.cached_top3.get(unique_id)
            if top_3 is None:
                if not self.cache_scores:
                    lines = ["Not confident enough to", "assert an identity"]
                    line_sizes = [self._get_text_width_height(line) for line in lines]
                    line_h = line_sizes[0][1]
                    line_step = line_h + self.text_padding
                    total_h = line_h + line_step * (len(lines) - 1)
                    bars_total_width = 3 * (self.bar_width + self.effective_bar_spacing)
                    msg_y = row_y + self.bar_height // 2 - total_h // 2 + line_h
                    for i, (line, (lw, _)) in enumerate(zip(lines, line_sizes)):
                        msg_x = bar_start_x + (bars_total_width - lw) // 2
                        cv2.putText(
                            img=frame,
                            text=line,
                            org=(msg_x, msg_y + i * line_step),
                            fontFace=self.text_font,
                            fontScale=self.text_scale,
                            color=self.text_color.as_bgr(),
                            thickness=self.text_thickness,
                            lineType=cv2.LINE_AA,
                        )
                continue

            for bar_idx in range(len(top_3[0])):
                score = top_3[1][bar_idx]
                identity = top_3[0][bar_idx]
                bar_x = bar_start_x + bar_idx * (self.bar_width + self.effective_bar_spacing)
                bar_color = self.palette.by_idx(row_idx + 1)

                show_label = bar_idx < 2
                self._draw_bar(frame, bar_x, row_y, score, identity, bar_color, show_label)

        return frame


@VISUALIZERS.register_module()
class TagsDetectionWriter(BaseWriter):

    def __init__(
        self,
        tag_ids: List[int],
        palette: Optional[dict] = None,
        text_color: Optional[List[int]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            text_anchor=[0, 0],
            text_color=text_color,
            text_scale=1.12,
            text_thickness=2,
            text_padding=50,
        )
        self.tag_ids = np.array(tag_ids)
        self.table_data = np.zeros((len(tag_ids), 4), dtype="U6")
        self.table_data[:, 1] = tag_ids
        self.table_data[:, 0] = ["-" for _ in tag_ids]
        for i in [2, 3]:
            self.table_data[:, i] = ["0" for _ in tag_ids]
        self.palette = ColorPalette(**palette) if palette is not None else ColorPalette()
        self.table_colors = [self.text_color for _ in tag_ids]
        self.title = "Aruco Tags Association and Detection"
        self.w_t, self.h_t = self._get_text_width_height(self.title)
        self.header = "Instances ID    Tags ID    Tags Detection    Tags Precision"
        self.w_h, self.h_h = self._get_text_width_height(self.header)

    def _get_validations(self, outputs: Tuple[Dict[str, np.ndarray]]) -> np.ndarray:
        for output in outputs:
            if "CsvTailtagValidations" in output.keys():
                return output.get("CsvTailtagValidations", np.array([]))
        return np.array([])

    def __call__(self, frame: np.ndarray, outputs: Tuple[Dict[str, np.ndarray]], idx: int) -> None:
        if self.text_anchor.x == 0 and self.text_anchor.y == 0:
            self.x = frame.shape[0] + self.text_padding
            self.y = 0 + self.text_padding
            self.cols_x = [self.x, self.x + 290, self.x + 490, self.x + 825]
        text_rect = self._get_text_rectangle(
            self.x,
            self.y,
            self.w_h,
            frame.shape[1] - self.text_padding,
        )

        frame = self._pad_frame(frame, text_rect, frame.shape)
        sv.draw_filled_rectangle(frame, text_rect, sv.Color(255, 255, 255))
        cv2.putText(
            img=frame,
            text=self.title,
            org=(
                self.x + abs(self.w_t // 2 - self.w_h // 2),
                self.y + self.h_t,
            ),
            fontFace=self.text_font,
            fontScale=self.text_scale,
            color=self.text_color.as_bgr(),
            thickness=self.text_thickness + 1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img=frame,
            text=self.header,
            org=(
                self.x,
                self.y + self.h_h + self.h_t * 4,
            ),
            fontFace=self.text_font,
            fontScale=self.text_scale,
            color=self.text_color.as_bgr(),
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )
        validations = self._get_validations(outputs)
        if validations.size > 1:
            to_update_idx = np.where(np.isin(self.tag_ids, validations[:, 1]))[0]
        else:
            to_update_idx = []

        for row, valid_id in enumerate(self.tag_ids):
            _, r_h = self._get_text_width_height(self.table_data[row, 1])
            thickness = 0
            if np.isin(row, to_update_idx):
                thickness = 1
                inst_ids = validations[validations[:, 1] == valid_id][:, 2]
                self.table_data[row, 2] = int(self.table_data[row, 2]) + len(inst_ids)
                precision = validations[validations[:, 1] == valid_id][:, 7][0]
                self.table_data[row, 3] = f"{precision*100:.2f}"
                inst_id = inst_ids[0]
                if self.table_data[row, 0] == "-" and inst_id >= 0:
                    self.table_data[row, 0] = str(int(inst_id))
                    self.table_colors[row] = self.palette.by_idx(inst_id)
            for col, col_x in enumerate(self.cols_x):
                cv2.putText(
                    img=frame,
                    text=self.table_data[row, col],
                    org=(
                        col_x,
                        self.y + self.h_h * 4 + self.h_t * 4 + row * (r_h + 25),
                    ),
                    fontFace=self.text_font,
                    fontScale=self.text_scale - 0.12,
                    color=self.table_colors[row].as_bgr(),
                    thickness=self.text_thickness + thickness,
                    lineType=cv2.LINE_AA,
                )
        return frame
