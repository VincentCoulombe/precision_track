from mmengine.logging import print_log

from .action_recognition import ActionRecognitionBackend
from .detection import DetectionBackend
from .online import OnlineBackend
from .re_identification import ReIDBackend
from .ultralytics_detection import UltralyticsDetectionBackend

__all__ = [
    "DetectionBackend",
    "UltralyticsDetectionBackend",
    "ReIDBackend",
    "ActionRecognitionBackend",
    "OnlineBackend",
    "build_detection_backend",
]


def build_detection_backend(detector_cfg):
    """Instantiate the detection backend, auto-selecting the Ultralytics one when appropriate.

    Inspects the resolved checkpoint (``.onnx``/``.engine``) metadata: if it was exported by
    Ultralytics, returns :class:`UltralyticsDetectionBackend`; otherwise the default
    :class:`DetectionBackend`. Keeps the existing config shape unchanged (no ``type`` needed).
    """
    from precision_track.utils.deployment import is_ultralytics_checkpoint, set_runtime_attributes

    runtime = detector_cfg.get("runtime", {})
    checkpoint = runtime.get("checkpoint", "") or runtime.get("deploying_directory")
    resolved = None
    try:
        _, resolved = set_runtime_attributes(checkpoint)
    except Exception:
        resolved = None

    if resolved is not None and is_ultralytics_checkpoint(resolved):
        print_log("Auto-detected an Ultralytics checkpoint; using UltralyticsDetectionBackend.", logger="current")
        return UltralyticsDetectionBackend(**detector_cfg)
    return DetectionBackend(**detector_cfg)
