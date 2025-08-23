# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import io
import warnings
from typing import Optional

import cv2
import mmengine.fileio as fileio
import numpy as np
import torch
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION, IMREAD_UNCHANGED
from mmengine.utils import is_str

from precision_track.registry import TRANSFORMS

from .base import BaseTransform

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None

try:
    import tifffile
except ImportError:
    tifffile = None

jpeg = None
supported_backends = ["cv2", "turbojpeg", "pillow", "tifffile"]

imread_flags = {
    "color": IMREAD_COLOR,
    "grayscale": IMREAD_GRAYSCALE,
    "unchanged": IMREAD_UNCHANGED,
    "color_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    "grayscale_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE,
}

imread_backend = "cv2"


def _pillow2array(img, flag: str = "color", channel_order: str = "bgr") -> np.ndarray:
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "unchanged":
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ["color", "grayscale"]:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != "RGB":
            if img.mode != "LA":
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert("RGB")
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert("RGBA")
                img = Image.new("RGB", img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ["color", "color_ignore_orientation"]:
            array = np.array(img)
            if channel_order != "rgb":
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ["grayscale", "grayscale_ignore_orientation"]:
            img = img.convert("L")
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", ' f'"color_ignore_orientation" or "grayscale_ignore_orientation"' f" but got {flag}"
            )
    return array


def _jpegflag(flag: str = "color", channel_order: str = "bgr"):
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "color":
        if channel_order == "bgr":
            return TJPF_BGR
        elif channel_order == "rgb":
            return TJCS_RGB
    elif flag == "grayscale":
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')


def imfrombytes(content: bytes, flag: str = "color", channel_order: str = "bgr", backend: Optional[str] = None) -> np.ndarray:
    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(f"backend: {backend} is not supported. Supported " "backends are 'cv2', 'turbojpeg', 'pillow', 'tifffile'")
    if backend == "turbojpeg":
        img = jpeg.decode(content, _jpegflag(flag, channel_order))  # type: ignore
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return img
    elif backend == "pillow":
        with io.BytesIO(content) as buff:
            img = Image.open(buff)
            img = _pillow2array(img, flag, channel_order)
        return img
    elif backend == "tifffile":
        with io.BytesIO(content) as buff:
            img = tifffile.imread(buff)
        return img
    else:
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if is_str(flag) else flag
        img = cv2.imdecode(img_np, flag)
        if flag == IMREAD_COLOR and channel_order == "rgb":
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img


class LoadImageFromFile(BaseTransform):
    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "color",
        imdecode_backend: str = "cv2",
        file_client_args: Optional[dict] = None,
        ignore_empty: bool = False,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn('"file_client_args" will be deprecated in future. ' 'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError('"file_client_args" and "backend_args" cannot be set ' "at the same time.")

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results["img_path"]
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(filename, backend_args=self.backend_args)
            img = imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f"failed to load image: {filename}"
        if self.to_float32:
            img = img.astype(np.float32)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"ignore_empty={self.ignore_empty}, "
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"imdecode_backend='{self.imdecode_backend}', "
        )

        if self.file_client_args is not None:
            repr_str += f"file_client_args={self.file_client_args})"
        else:
            repr_str += f"backend_args={self.backend_args})"

        return repr_str


@TRANSFORMS.register_module()
class LoadImage(LoadImageFromFile):
    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        try:
            if "img" not in results:
                # Load image from file by :meth:`LoadImageFromFile.transform`
                results = super().transform(results)
            else:
                img = results["img"]
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if "img_path" not in results:
                    results["img_path"] = None
                results["img_shape"] = img.shape[:2]
                results["ori_shape"] = img.shape[:2]
        except Exception as e:
            e = type(e)(f'`{str(e)}` occurs when loading `{results["img_path"]}`.' "Please check whether the file exists.")
            raise e

        return results


@TRANSFORMS.register_module()
class LoadImageSequence(LoadImageFromFile):
    def transform(self, results: dict) -> Optional[dict]:
        try:
            assert isinstance(results["img_shapes"], list) & isinstance(results["ori_shapes"], list)
            if "imgs" not in results:
                imgs = []
                assert isinstance(results["img_paths"], list)
                for img_path in results["img_paths"]:
                    results["img_path"] = img_path
                    results = super().transform(results)
                    imgs.append(results["img"])
                    results["img_shapes"].append(results["img_shape"])
                    results["ori_shapes"].append(results["ori_shape"])
                del results["img_path"]
                del results["img"]
                del results["img_shape"]
                del results["ori_shape"]
            else:
                imgs = results["imgs"]
                assert isinstance(imgs, list)
                for img in imgs:
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()
                    assert isinstance(img, np.ndarray)
                    if self.to_float32:
                        img = img.astype(np.float32)
                    imgs.append(img)
                    if "img_paths" not in results:
                        results["img_paths"] = None
                    results["img_shapes"].append(img.shape[:2])
                    results["ori_shapes"].append(img.shape[:2])
            results["imgs"] = np.stack(imgs)
        except Exception as e:
            e = type(e)(f'`{str(e)}` occurs when loading `{results["img_path"]}`.' "Please check whether the file exists.")
            raise e

        return results
