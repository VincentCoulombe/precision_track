import os
import re
from logging import WARNING
from typing import Dict, Iterator, List, Optional, Union

import cv2
import numpy as np
import supervision as sv
from mmengine.logging import print_log
from mmengine.utils import track_iter_progress

from precision_track.registry import VISUALIZERS
from precision_track.utils import VideoReader

from .result import Result


class Visualizer:
    SUPPORTED_VIDEO_EXTENSIONS = [".avi", ".mp4"]
    SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

    def __init__(
        self,
        painters: Optional[List[dict]] = None,
        writers: Optional[List[dict]] = None,
        palette: Optional[dict] = None,
        size: Optional[tuple] = None,
        *args,
        **kwargs,
    ) -> None:
        """Leverage the many supported Painters and Writers to draw the
        tracking outputs on the footage for visualization.

        Args:
            painters (Optional[List[dict]], optional): the Painters used for drawing on the image. Defaults to None.
            writers (Optional[List[dict]], optional): The Writers used to write on the image. Defaults to None.
            palette (Optional[dict], optional): The args that will be passed to the color palette. Defaults to None.
        """
        assert palette is None or isinstance(palette, dict)
        self.painters = []
        if isinstance(painters, list):
            for painter in painters:
                painter.update({"palette": palette})
                self.painters.append(VISUALIZERS.build(painter))

        self.writers = []
        if isinstance(writers, list):
            for writer in writers:
                writer.update({"palette": palette})
                self.writers.append(VISUALIZERS.build(writer))

        if size is not None:
            assert isinstance(size, tuple) and len(size) == 2 and 0 < size[0] and 0 < size[1]
            self.h = int(size[0])
            self.w = int(size[1])
        else:
            self.h = None
            self.w = None

    @staticmethod
    def save_frames(
        source_path: str,
        sink_dir: str,
        start_idx: Optional[int] = 0,
        stride: Optional[int] = 1,
        end_idx: Optional[int] = None,
    ) -> None:
        """Save frames from a video into a folder.

        Args:
            source_path (str): The path to the video.
            sink_dir (str): The directory where the frames will be stored.
            start_idx (Optional[int], optional): The frame id at which the saving start. If None, the saving will start from the beginning.
            stride (Optional[int], optional): The stride for which the frames are saved. If None, will be 1.
            end_idx (Optional[int], optional): The frame id at which the saving end. If None, the saving will stop at the end.
        """

        frames_generator = sv.get_video_frames_generator(source_path, start=start_idx, end=end_idx, stride=stride)
        os.makedirs(sink_dir, exist_ok=True)
        images_sink = sv.ImageSink(
            target_dir_path=sink_dir,
            overwrite=True,
            image_name_pattern="{:05d}.jpeg",
        )
        with images_sink:
            for frame in frames_generator:
                images_sink.save_image(frame)

    def _process_frame(
        self,
        frame: Union[np.ndarray, str],
        outputs: List[Dict[str, np.ndarray]],
        idx: int,
    ) -> np.ndarray:
        if isinstance(frame, str):
            self._assert_img_ext(os.path.splitext(frame)[1])
            frame = cv2.imread(frame)
        if self.h is None:
            self.h, self.w = frame.shape[:2]
        for painter in self.painters:
            frame = painter(frame, outputs, idx)
        frame = cv2.resize(frame, (self.w, self.h))
        for writer in self.writers:
            frame = writer(frame, outputs, idx)
        return frame

    def __call__(
        self,
        source_path: str,
        result: Result,
        sink_path: str,
    ) -> None:
        """Paint and write the result on a video or a folder of images. Stops painting and writing
        when the result runs out or the source of images end.

        Args:
            source_path (str): The path to either the source video or the source folder.
            result (Result): The result.
            sink_path (str): The path where the painted and written video/images will be saved.
        """
        source_name, source_ext = os.path.splitext(source_path)
        _, sink_ext = os.path.splitext(sink_path)
        if len(source_ext) == 0:
            pattern = rf"\d+(?=({'|'.join(self.SUPPORTED_IMAGE_EXTENSIONS)}))"
            source_names = []
            source_ids = []
            for filename in os.listdir(source_name):
                file_id = re.search(pattern, filename)
                if file_id is None:
                    raise ValueError(
                        f"The folder {os.path.abspath(source_name)} contains the file {filename}"
                        f"for which the extension is not within the supported image extensions: {self.SUPPORTED_IMAGE_EXTENSIONS}."
                    )
                source_ids.append(int(re.search(pattern, filename).group()))
                source_names.append(os.path.join(source_name, filename))

            iterator = zip(track_iter_progress(sorted(zip(source_ids, source_names), key=lambda x: x[0]), len(source_ids)), result)
            fps = 30
        else:
            video_reader = VideoReader(source_path)
            iterator = enumerate(zip(track_iter_progress((video_reader, len(video_reader))), result))
            fps = video_reader.fps
        if len(sink_ext) == 0:
            os.makedirs(sink_path, exist_ok=True)
            self._write_to_folder(sink_path, iterator)
        else:
            os.makedirs(os.path.dirname(sink_path), exist_ok=True)
            self._write_to_video(sink_path, iterator, fps)

    def _assert_vid_ext(self, ext: str) -> None:
        assert ext in self.SUPPORTED_VIDEO_EXTENSIONS, f"Video extension must be one of: {self.SUPPORTED_VIDEO_EXTENSIONS}, not {ext}."

    def _assert_img_ext(self, ext: str) -> None:
        assert ext in self.SUPPORTED_IMAGE_EXTENSIONS, f"Image extension must be one of: {self.SUPPORTED_IMAGE_EXTENSIONS}, not {ext}."

    def _get_fourcc(self, ext: str) -> cv2.VideoWriter_fourcc:
        self._assert_vid_ext(ext)
        if ext == ".mp4":
            return cv2.VideoWriter_fourcc(*"mp4v")
        elif ext == ".avi":
            return cv2.VideoWriter_fourcc(*"XVID")
        else:
            raise ValueError(f"Unsupported video format ({ext}).")

    def _write_to_video(self, sink_path: str, iterator: Iterator, fps: int):
        _, sink_ext = os.path.splitext(sink_path)
        fourcc = self._get_fourcc(sink_ext)

        video_writer = None
        for idx, (frame, outputs) in iterator:
            frame = self._process_frame(frame, outputs, idx)
            if video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    sink_path,
                    fourcc,
                    fps,
                    (w, h),
                )
            video_writer.write(frame)
        if isinstance(video_writer, cv2.VideoWriter):
            video_writer.release()
        else:
            print_log("Attempted to render empty file(s).", logger="current", level=WARNING)

    def _write_to_folder(self, sink_folder: str, iterator: Iterator):
        for (idx, frame_name), outputs in iterator:
            frame = self._process_frame(frame_name, outputs, idx)
            cv2.imwrite(os.path.join(sink_folder, os.path.basename(frame_name)), frame)
