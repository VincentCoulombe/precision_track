import logging
from copy import deepcopy
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine import Config
from mmengine.logging import print_log
from torchvision.ops import batched_nms

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, format_detection_output

from .detection import DetectionBackend


def letterbox(img: np.ndarray, new_shape: int, color=(114, 114, 114), scaleup: bool = True) -> np.ndarray:
    """Aspect-preserving resize + centered padding, identical to Ultralytics' ``LetterBox``.

    Reimplements ``ultralytics.data.augment.LetterBox`` (``auto=False, center=True, stride=32``) with
    only OpenCV/NumPy so no ultralytics dependency is needed. Verified pixel-identical to Ultralytics.
    """
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (round(w0 * r), round(h0 * r))  # (w, h)
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    xyxy = torch.empty_like(boxes)
    half = boxes[:, 2:] / 2
    xyxy[:, :2] = boxes[:, :2] - half
    xyxy[:, 2:] = boxes[:, :2] + half
    return xyxy


@MODELS.register_module()
class UltralyticsDetectionBackend(DetectionBackend):
    """Detection backend for Ultralytics YOLO exports (``.onnx`` / ``.engine``).

    The export is expected **without in-graph NMS** (``nms=False``): ``output0`` is the raw head
    ``[B, 4+nc(+K*3), num_anchors]``. Decoding and NMS are done **here on the GPU**
    (``torchvision.ops.batched_nms``) — unlike an in-graph-NMS export whose ``NonMaxSuppression`` /
    ``NonZero`` ops have no ONNX-Runtime CUDA kernel and force a slow CPU tail.

    Preprocessing (letterbox + RGB + ``/255``, via the standalone :func:`letterbox`) and output rescaling
    reuse the shared :func:`precision_track.utils.format_detection_output`, so the post-processing tail is
    identical to :class:`DetectionBackend`. Coordinates are produced in the model's letterboxed input
    space, then mapped back to the original image by that shared util.

    Produces **no appearance `features`** (placeholder zeros) — feature-dependent stages (action
    recognition, clustered-feature output) are disabled upstream by the tracker.
    """

    FEATURE_DIM = 1  # placeholder; appearance-dependent stages are disabled for this backend

    def __init__(
        self,
        runtime: Config,
        conf_thr: float = 0.1,
        iou_thr: float = 0.65,
        max_det: int = 300,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> None:
        from precision_track.utils.deployment import read_ultralytics_metadata, set_runtime_attributes

        runtime = deepcopy(runtime)
        checkpoint = runtime.get("checkpoint", "") or runtime.get("deploying_directory")
        runtime_type, resolved = set_runtime_attributes(checkpoint)
        assert runtime_type in ("ONNXRuntime", "TensorRTRuntime"), (
            f"UltralyticsDetectionBackend only supports .onnx and .engine checkpoints, got a " f"{runtime_type} for '{checkpoint}'."
        )

        meta = metadata or read_ultralytics_metadata(resolved) or {}
        self._task = str(meta.get("task", "detect"))
        self.is_pose = self._task == "pose"
        imgsz = meta.get("imgsz", 640)
        imgsz = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
        self.imgsz = int(imgsz)
        kpt_shape = meta.get("kpt_shape", (1, 3)) if self.is_pose else (1, 3)
        self.num_keypoints = int(kpt_shape[0])
        self.class_names = meta.get("names", None)
        self.conf_thr = float(conf_thr)
        self.iou_thr = float(iou_thr)
        self.max_det = int(max_det)

        # Preprocessing + (for raw exports) NMS are done here, so no data pre/post-processor is needed.
        runtime["input_shapes"] = [dict(type="ImageShape", n_channels=3, width=self.imgsz, height=self.imgsz)]
        # The Ultralytics graph carries no PyTorch head; give the runtime a head-less stand-in so the
        # parent's YOLOX temperature hook (`runtime.model.head`) is a safe no-op instead of crashing.
        if runtime.get("model") is None:
            runtime["model"] = SimpleNamespace(head=None)
        kwargs.pop("data_preprocessor", None)
        kwargs.pop("data_postprocessor", None)

        super().__init__(runtime=runtime, data_preprocessor=None, data_postprocessor=None, **kwargs)

        print_log(
            f"UltralyticsDetectionBackend ready (task={self._task}, imgsz={self.imgsz}" f"{f', keypoints={self.num_keypoints}' if self.is_pose else ''}).",
            logger="current",
            level=logging.INFO,
        )

    # ------------------------------------------------------------------ preprocess
    def preprocess(self, images, ids):
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(ids, (list, tuple)):
            ids = [ids]

        tensors, data_samples = [], []
        for img_in, id_ in zip(images, ids):
            img = self._to_bgr_hwc(img_in)  # HWC, BGR, uint8
            ori_shape = (int(img.shape[0]), int(img.shape[1]))
            lb_img = letterbox(img, self.imgsz)  # HWC, BGR, uint8, (imgsz, imgsz)
            rgb = np.ascontiguousarray(lb_img[..., ::-1])  # BGR -> RGB
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
            tensors.append(t)

            ds = id_ if isinstance(id_, PoseDataSample) else PoseDataSample(metainfo=dict(img_id=int(id_)))
            ds.set_metainfo(dict(ori_shape=ori_shape, img_shape=(self.imgsz, self.imgsz)))
            data_samples.append(ds)

        inputs = torch.stack(tensors, dim=0).to(self.device)
        return dict(inputs=inputs, data_samples=data_samples)

    @staticmethod
    def _to_bgr_hwc(img_in: Union[np.ndarray, torch.Tensor, str]) -> np.ndarray:
        if isinstance(img_in, str):
            img = cv2.imread(img_in)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_in}")
            return img
        if isinstance(img_in, torch.Tensor):
            img_in = img_in.detach().cpu().numpy()
        img = np.asarray(img_in)
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        return np.ascontiguousarray(img.astype(np.uint8))

    # ----------------------------------------------------------------- postprocess
    def postprocess(self, pred, data_samples: List[PoseDataSample]):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred = pred.float()  # raw head: [B, 4+nc(+K*3), num_anchors]

        outputs = []
        for i, ds in enumerate(data_samples):
            bboxes, scores, labels, kpts_xy, kpt_conf = self._decode_and_nms(pred[i])

            n = int(bboxes.shape[0])
            scale, translation = self._letterbox_affine(ds.metainfo["ori_shape"], pred.device)
            zeros = torch.zeros((n, self.FEATURE_DIM), device=pred.device, dtype=bboxes.dtype)
            outputs.append(
                format_detection_output(
                    ds,
                    bboxes=bboxes,
                    scores=scores,
                    labels=labels,
                    keypoints=kpts_xy,
                    keypoint_scores=kpt_conf,
                    features=zeros,
                    scale=scale,
                    translation=translation,
                    kept_idxs=torch.arange(n, device=pred.device),
                    feature_maps=zeros,
                    priors=torch.zeros((n, 2), device=pred.device, dtype=bboxes.dtype),
                    kpt_score_thr=self.kpt_score_thr,
                )
            )
        return outputs

    def _decode_and_nms(self, row: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Decode + GPU NMS one image's raw head output ``[C, num_anchors]`` (input-space xyxy)."""
        x = row.transpose(0, 1)  # [num_anchors, C]
        K = self.num_keypoints
        if self.is_pose:
            nc = x.shape[1] - 4 - K * 3
            kpts = x[:, 4 + nc :].reshape(-1, K, 3)
        else:
            nc = x.shape[1] - 4
            kpts = None
        conf, cls = x[:, 4 : 4 + nc].max(1)
        keep = conf > self.conf_thr
        boxes = xywh2xyxy(x[keep, :4])
        conf, cls = conf[keep], cls[keep]
        if kpts is not None:
            kpts = kpts[keep]

        idx = batched_nms(boxes, conf, cls, self.iou_thr)[: self.max_det]
        boxes, conf, cls = boxes[idx], conf[idx], cls[idx]
        n = boxes.shape[0]
        if kpts is not None:
            kpts = kpts[idx]
            kpts_xy, kpt_conf = kpts[..., :2], kpts[..., 2]
        else:
            kpts_xy = torch.zeros((n, K, 2), device=row.device, dtype=boxes.dtype)
            kpt_conf = torch.zeros((n, K), device=row.device, dtype=boxes.dtype)
        return boxes, conf, cls.long(), kpts_xy, kpt_conf

    def _letterbox_affine(self, ori_shape, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-axis (scale, translation) mapping letterboxed-input coords back to the original image.

        Inverse of :func:`letterbox` (``scaleup=True``), matching Ultralytics' ``scale_coords`` (float
        padding): ``orig = input / gain - pad / gain``.
        """
        h0, w0 = ori_shape
        gain = min(self.imgsz / h0, self.imgsz / w0)
        pad_x = (self.imgsz - round(w0 * gain)) / 2
        pad_y = (self.imgsz - round(h0 * gain)) / 2
        scale = torch.tensor([1.0 / gain, 1.0 / gain], device=device)
        translation = torch.tensor([-pad_x / gain, -pad_y / gain], device=device)
        return scale, translation
