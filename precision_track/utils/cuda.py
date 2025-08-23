import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # for eventual linting
    import pycuda.driver as cuda_mod  # noqa
    import tensorrt as trt_mod  # noqa

_DISABLE_MSG = os.getenv("PRECISION_TRACK_DISABLE_CUDA", "0") in {"1", "true", "TRUE"}


def cuda_available():
    return torch.cuda.is_available() and torch.version.cuda is not None


def get_device() -> str:
    return "cuda" if cuda_available() else "cpu"


@lru_cache(None)
def _import_trt():
    if _DISABLE_MSG:
        raise ImportError("CUDA/TensorRT disabled via PRECISION_TRACK_DISABLE_CUDA=1")
    try:
        import importlib

        trt = importlib.import_module("tensorrt")
        return trt
    except Exception as e:
        raise ImportError(
            'TensorRT not available. Install extras: pip install "precision_track[cuda]" ' "and ensure a compatible NVIDIA driver/CUDA is installed."
        ) from e


@lru_cache(None)
def _import_pycuda():
    if _DISABLE_MSG:
        return None
    try:
        import importlib

        cuda = importlib.import_module("pycuda.driver")
        cuda.init()
        return cuda
    except Exception:
        return None


def trt_available() -> bool:
    try:
        _import_trt()
        return True
    except Exception:
        return False


def pycuda_available() -> bool:
    return _import_pycuda() is not None


def get_trt():
    return _import_trt()


def get_pycuda():
    return _import_pycuda()
