"""Video helpers for the SLEAP pipeline (OpenCV only, no precision_track import).

Verbatim copy of ``dlc/video_utils.py`` -- see the note in ``pt_format.py`` on why each
benchmark directory keeps its own copy.
"""

import os
import os.path as osp
from typing import Optional, Tuple

import cv2


def video_resolution(video_path: str) -> Tuple[int, int]:
    """Return the ``(height, width)`` of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return height, width


def frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def downsize_video(video_path: str, output_path: str, width: int, height: int) -> str:
    """Rescale every frame of ``video_path`` to ``width x height`` and return ``output_path``.

    Unlike the previous helper in ``precision_track/evaluation/utils``, this returns the
    path it wrote so callers can use the result directly.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.resize(frame, (width, height)))

    cap.release()
    out.release()
    return output_path


def ensure_resolution(video_path: str, img_size: Optional[Tuple[int, int]]) -> Tuple[str, Tuple[int, int], Tuple[int, int]]:
    """Rescale ``video_path`` to ``img_size`` (height, width) if it does not already match.

    Returns ``(path_to_use, (ori_height, ori_width), (new_height, new_width))``. When no
    rescaling is needed (or ``img_size`` is None), the original path and identical
    resolutions are returned, so callers can unconditionally rescale their outputs back.
    """
    ori_height, ori_width = video_resolution(video_path)
    if img_size is None:
        return video_path, (ori_height, ori_width), (ori_height, ori_width)

    new_height, new_width = img_size
    if (ori_height, ori_width) == (new_height, new_width):
        return video_path, (ori_height, ori_width), (ori_height, ori_width)

    stem, ext = osp.splitext(video_path)
    rescaled_path = f"{stem}_{new_height}x{new_width}{ext}"
    if not osp.isfile(rescaled_path):
        print(f"Rescaling video from {ori_height}x{ori_width} to {new_height}x{new_width}")
        downsize_video(video_path, rescaled_path, width=new_width, height=new_height)
    else:
        print(f"Reusing already rescaled video: {rescaled_path}")
    return rescaled_path, (ori_height, ori_width), (new_height, new_width)
