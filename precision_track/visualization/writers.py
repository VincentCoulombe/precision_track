import abc
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from precision_track.registry import VISUALIZERS

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
            if "CsvValidations" in output.keys():
                return output.get("CsvValidations", np.array([]))
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
