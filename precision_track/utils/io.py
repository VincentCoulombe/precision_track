# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import os
import os.path as osp
import pkgutil
import re
from collections import OrderedDict, namedtuple
from importlib import import_module
from typing import Callable, Dict, List, Union

import cv2
import mmengine
import torch
from cv2 import CAP_PROP_FOURCC, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_POS_FRAMES
from mmengine.dist import get_dist_info
from mmengine.fileio import get_file_backend
from mmengine.fileio import load as load_file
from mmengine.logging import print_log
from mmengine.model import BaseTTAModel, is_model_wrapper
from mmengine.utils import check_file_exist, digit_version, mkdir_or_exist, track_progress
from mmengine.utils.dl_utils import load_url

SUPPORTED_IMG_BACKEND = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".pgm", ".pbm", ".ppm", ".ras"]
ENV_MMENGINE_HOME = "MMENGINE_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


class Cache:

    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache[key] if key in self._cache else default
        return val


class VideoReader:
    def __init__(self, filename, cache_capacity=10):
        # Check whether the video path is a url
        if not filename.startswith(("https://", "http://")):
            check_file_exist(filename, "Video file not found: " + filename)
        self.filename = filename
        self._vcap = cv2.VideoCapture(filename)
        assert cache_capacity > 0
        self._cache = Cache(cache_capacity)
        self._position = 0
        # get basic info
        self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
        self._fps = self._vcap.get(CAP_PROP_FPS)
        self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(CAP_PROP_FOURCC)

    @property
    def vcap(self):
        """:obj:`cv2.VideoCapture`: The raw VideoCapture object."""
        return self._vcap

    @property
    def opened(self):
        """bool: Indicate whether the video is opened."""
        return self._vcap.isOpened()

    @property
    def width(self):
        """int: Width of video frames."""
        return self._width

    @property
    def height(self):
        """int: Height of video frames."""
        return self._height

    @property
    def resolution(self):
        """tuple: Video resolution (width, height)."""
        return (self._width, self._height)

    @property
    def fps(self):
        """float: FPS of the video."""
        return self._fps

    @property
    def frame_cnt(self):
        """int: Total frames of the video."""
        return self._frame_cnt

    @property
    def fourcc(self):
        """str: "Four character code" of the video."""
        return self._fourcc

    @property
    def position(self):
        """int: Current cursor position, indicating frame decoded."""
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        """Read the next frame.

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode, cache and return it.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        # pos = self._position
        if self._cache:
            img = self._cache.get(self._position)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.put(self._position, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            self._position += 1
        return img

    def get_frame(self, frame_id):
        """Get frame by index.

        Args:
            frame_id (int): Index of the expected frame, 0-based.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(f'"frame_id" must be between 0 and {self._frame_cnt - 1}')
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        ret, img = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, img)
            self._position += 1
        return img

    def current_frame(self):
        """Get the current frame (frame that is just visited).

        Returns:
            ndarray or None: If the video is fresh, return None, otherwise
            return the frame.
        """
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def cvt2frames(self, frame_dir, file_start=0, filename_tmpl="{:06d}.jpg", start=0, max_num=0, show_progress=True):
        """Convert a video to frame images.

        Args:
            frame_dir (str): Output directory to store all the frame images.
            file_start (int): Filenames will start from the specified number.
            filename_tmpl (str): Filename template with the index as the
                placeholder.
            start (int): The starting frame index.
            max_num (int): Maximum number of frames to be written.
            show_progress (bool): Whether to show a progress bar.
        """
        mkdir_or_exist(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError("start must be less than total frame number")
        if start > 0:
            self._set_real_position(start)

        def write_frame(file_idx):
            img = self.read()
            if img is None:
                return
            filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
            cv2.imwrite(filename, img)

        if show_progress:
            track_progress(write_frame, range(file_start, file_start + task_num))
        else:
            for i in range(task_num):
                write_frame(file_start + i)

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.get_frame(i) for i in range(*index.indices(self.frame_cnt))]
        # support negative indexing
        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError("index out of range")
        return self.get_frame(index)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()


def get_videos_in_dir(dir_path: str):
    assert os.path.exists(dir_path)
    videos = []
    for file in os.listdir(dir_path):
        cap = cv2.VideoCapture(os.path.join(dir_path, file))
        if cap.isOpened():
            videos.append(file)
        cap.release()
    return videos


def get_imgs_in_dir(dir_path: str):
    assert os.path.exists(dir_path)
    images = []
    for file in os.listdir(dir_path):
        _, ext = os.path.splitext(file)
        if ext in SUPPORTED_IMG_BACKEND:
            images.append(file)
    return images


def infer_paths(path_or_dir: Union[List[str], str]) -> Union[List[str], str]:
    if isinstance(path_or_dir, list):
        return path_or_dir
    assert osp.exists(path_or_dir), f"{path_or_dir} does not exists."
    if osp.isdir(path_or_dir):
        root = osp.abspath(osp.dirname(path_or_dir))
        paths = []
        for file in os.listdir(path_or_dir):
            paths.append(osp.join(root, file))
        if paths:
            return paths
        return root
    elif osp.isfile(path_or_dir):
        return [path_or_dir]
    else:
        raise ValueError(f"Expected either an existing path, a directory or a list of paths. Not, {type(path_or_dir)}.")


def kwargs_to_args(kwargs, arg_keys):
    return tuple(kwargs[k] for k in arg_keys)


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes: Dict[str, Callable] = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(f"{prefix} is already registered as a loader backend, " 'add "force=True" if you want to override it')
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_ceph`
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger="current"):
        """Load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Defaults to None
            logger (str): The logger for message. Defaults to 'current'.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        print_log(f"Loads checkpoint by {class_name[10:]} backend from path: " f"{filename}", logger=logger)
        return checkpoint_loader(filename, map_location)


@CheckpointLoader.register_scheme(prefixes="")
def load_from_local(filename, map_location):
    """Load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f"{filename} can not be found.")
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=("http://", "https://"))
def load_from_http(filename, map_location=None, model_dir=None, progress=os.isatty(0)):
    """Load checkpoint through HTTP or HTTPS scheme path. In distributed
    setting, this function only download checkpoint at local rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (string, optional): directory in which to save the object,
            Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    rank, world_size = get_dist_info()
    if rank == 0:
        checkpoint = load_url(filename, model_dir=model_dir, map_location=map_location, progress=progress)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = load_url(filename, model_dir=model_dir, map_location=map_location, progress=progress)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=[r"(\S+\:)?s3://", r"(\S+\:)?petrel://"])
def load_from_ceph(filename, map_location=None, backend="petrel"):
    """Load checkpoint through the file path prefixed with s3.  In distributed
    setting, this function download ckpt at all ranks to different temporary
    directories.

    Args:
        filename (str): checkpoint file path with s3 prefix
        map_location (str, optional): Same as :func:`torch.load`.
        backend (str, optional): The storage backend type.
            Defaults to 'petrel'.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    file_backend = get_file_backend(filename, backend_args={"backend": backend})
    with io.BytesIO(file_backend.get(filename)) as buffer:
        checkpoint = torch.load(buffer, map_location=map_location, weights_only=False)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=("modelzoo://", "torchvision://"))
def load_from_torchvision(filename, map_location=None):
    """Load checkpoint through the file path prefixed with modelzoo or
    torchvision.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    model_urls = get_torchvision_models()
    if filename.startswith("modelzoo://"):
        print_log('The URL scheme of "modelzoo://" is deprecated, please ' 'use "torchvision://" instead', logger="current", level=logging.WARNING)
        model_name = filename[11:]
    else:
        model_name = filename[14:]
    return load_from_http(model_urls[model_name], map_location=map_location)


def _get_mmengine_home():
    mmengine_home = os.path.expanduser(os.getenv(ENV_MMENGINE_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "mmengine")))

    mkdir_or_exist(mmengine_home)
    return mmengine_home


@CheckpointLoader.register_scheme(prefixes=("open-mmlab://", "openmmlab://"))
def load_from_openmmlab(filename, map_location=None):
    """Load checkpoint through the file path prefixed with open-mmlab or
    openmmlab.

    Args:
        filename (str): checkpoint file path with open-mmlab or
        openmmlab prefix
        map_location (str, optional): Same as :func:`torch.load`.
          Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    model_urls = get_external_models()
    prefix_str = "open-mmlab://"
    if filename.startswith(prefix_str):
        model_name = filename[13:]
    else:
        model_name = filename[12:]
        prefix_str = "openmmlab://"

    deprecated_urls = get_deprecated_model_names()
    if model_name in deprecated_urls:
        print_log(f"{prefix_str}{model_name} is deprecated in favor " f"of {prefix_str}{deprecated_urls[model_name]}", logger="current", level=logging.WARNING)
        model_name = deprecated_urls[model_name]
    model_url = model_urls[model_name]
    # check if is url
    if model_url.startswith(("http://", "https://")):
        checkpoint = load_from_http(model_url, map_location=map_location)
    else:
        filename = osp.join(_get_mmengine_home(), model_url)
        if not osp.isfile(filename):
            raise FileNotFoundError(f"{filename} can not be found.")
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes="mmcls://")
def load_from_mmcls(filename, map_location=None):
    """Load checkpoint through the file path prefixed with mmcls.

    Args:
        filename (str): checkpoint file path with mmcls prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    model_urls = get_mmcls_models()
    model_name = filename[8:]
    checkpoint = load_from_http(model_urls[model_name], map_location=map_location)
    checkpoint = _process_mmcls_checkpoint(checkpoint)
    return checkpoint


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None, revise_keys=[(r"^module\.", "")]):
    checkpoint = CheckpointLoader.load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")

    return load_checkpoint_to_model(model, checkpoint, strict, logger, revise_keys)


def load_checkpoint_to_model(model, checkpoint, strict=False, logger=None, revise_keys=[(r"^module\.", "")]):

    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict({re.sub(p, r, k): v for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class _IncompatibleKeys(namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])):

    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Defaults to False.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    missing_keys = []
    err_msg = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, local_state_dict, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_model_wrapper(module) or isinstance(module, BaseTTAModel):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        if hasattr(module, "_load_state_dict_post_hooks"):
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with "
                    "``register_load_state_dict_post_hook`` are not expected "
                    "to return new values, if incompatible_keys need to be "
                    "modified, it should be done inplace."
                )

    load(module, state_dict)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append("unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print_log(err_msg, logger=logger, level=logging.WARNING)


def get_torchvision_models():
    import torchvision

    if digit_version(torchvision.__version__) < digit_version("0.13.0a0"):
        model_urls = dict()
        # When the version of torchvision is lower than 0.13, the model url is
        # not declared in `torchvision.model.__init__.py`, so we need to
        # iterate through `torchvision.models.__path__` to get the url for each
        # model.
        for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
            if ispkg:
                continue
            _zoo = import_module(f"torchvision.models.{name}")
            if hasattr(_zoo, "model_urls"):
                _urls = getattr(_zoo, "model_urls")
                model_urls.update(_urls)
    else:
        # Since torchvision bumps to v0.13, the weight loading logic,
        # model keys and model urls have been changed. Here the URLs of old
        # version is loaded to avoid breaking back compatibility. If the
        # torchvision version>=0.13.0, new URLs will be added. Users can get
        # the resnet50 checkpoint by setting 'resnet50.imagent1k_v1',
        # 'resnet50' or 'ResNet50_Weights.IMAGENET1K_V1' in the config.
        json_path = osp.join(mmengine.__path__[0], "hub/torchvision_0.12.json")
        model_urls = mmengine.load(json_path)
        if digit_version(torchvision.__version__) < digit_version("0.14.0a0"):
            weights_list = [cls for cls_name, cls in torchvision.models.__dict__.items() if cls_name.endswith("_Weights")]
        else:
            weights_list = [torchvision.models.get_model_weights(model) for model in torchvision.models.list_models(torchvision.models)]

        for cls in weights_list:
            # The name of torchvision model weights classes ends with
            # `_Weights` such as `ResNet18_Weights`. However, some model weight
            # classes, such as `MNASNet0_75_Weights` does not have any urls in
            # torchvision 0.13.0 and cannot be iterated. Here we simply check
            # `DEFAULT` attribute to ensure the class is not empty.
            if not hasattr(cls, "DEFAULT"):
                continue
            # Since `cls.DEFAULT` can not be accessed by iterating cls, we set
            # default urls explicitly.
            cls_name = cls.__name__
            cls_key = cls_name.replace("_Weights", "").lower()
            model_urls[f"{cls_key}.default"] = cls.DEFAULT.url
            for weight_enum in cls:
                cls_key = cls_name.replace("_Weights", "").lower()
                cls_key = f"{cls_key}.{weight_enum.name.lower()}"
                model_urls[cls_key] = weight_enum.url

    return model_urls


def get_external_models():
    mmengine_home = _get_mmengine_home()
    default_json_path = osp.join(mmengine.__path__[0], "hub/openmmlab.json")
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmengine_home, "open_mmlab.json")
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)

    return default_urls


def get_mmcls_models():
    mmcls_json_path = osp.join(mmengine.__path__[0], "hub/mmcls.json")
    mmcls_urls = load_file(mmcls_json_path)

    return mmcls_urls


def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmengine.__path__[0], "hub/deprecated.json")
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)

    return deprecate_urls


def _process_mmcls_checkpoint(checkpoint):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Some checkpoints converted from 3rd-party repo don't
        # have the "state_dict" key.
        state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)

    return new_checkpoint
