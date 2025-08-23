# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple, Union

import torch
from mmengine.config import Config
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from torch import Tensor

from precision_track.utils import PoseDataSample, parse_pose_metainfo


@MODELS.register_module()
class DetectionModel(BaseModel):

    def __init__(
        self,
        backbone: Config,
        metainfo: Union[str, dict],
        neck: Optional[Config] = None,
        head: Optional[Config] = None,
        feature_extraction_head: Optional[Config] = None,
        train_cfg: Optional[Config] = None,
        test_cfg: Optional[Config] = None,
        data_preprocessor: Optional[Config] = None,
        use_syncbn: bool = False,
        init_cfg: Optional[Config] = None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.metainfo = self._load_metainfo(metainfo)
        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            assert "classes" in self.metainfo and isinstance(
                self.metainfo["classes"], list
            ), f"The metadatafile: {metainfo} does not contain the mendatory key: classes."

            head["num_classes"] = len(self.metainfo["classes"])
            head["num_keypoints"] = self.metainfo.get("num_keypoints", 0)
            self.head = MODELS.build(head)
            self.head.test_cfg = self.test_cfg.copy()

        if feature_extraction_head is not None:
            self.feature_extraction_head = MODELS.build(feature_extraction_head)
            self.feature_extraction_head.test_cfg = self.test_cfg.copy()

        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log("Using SyncBatchNorm()", "current")

    def switch_to_deploy(self):
        """Switch the sub-modules to deploy mode."""
        for name, layer in self.named_modules():
            if layer == self:
                continue
            if callable(getattr(layer, "switch_to_deploy", None)):
                print_log(f"module {name} has been switched to deploy mode", "current")
                layer.switch_to_deploy(self.test_cfg)

    @property
    def with_neck(self) -> bool:
        """bool: whether the pose estimator has a neck."""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_head(self) -> bool:
        """bool: whether the pose estimator has a head."""
        return hasattr(self, "head") and self.head is not None

    @property
    def with_feature_extraction(self) -> bool:
        """bool: whether the pose estimator has a feature extraction head."""
        return hasattr(self, "feature_extraction_head") and self.feature_extraction_head is not None

    @staticmethod
    def _load_metainfo(metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            return None

        if isinstance(metainfo, dict):
            return parse_pose_metainfo(metainfo)
        elif isinstance(metainfo, str):
            return parse_pose_metainfo(dict(from_file=metainfo))
        else:
            raise ValueError(f"metainfo should either be the metadata dictionnary or its path. Not, {metainfo}.")

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[PoseDataSample]],
        mode: str = "tensor",
        *args,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor], dict]:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: 'tensor', 'predict' and 'loss':

        - 'tensor': Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - 'predict': Forward and return the predictions, which are fully
        processed to a list of :obj:`PoseDataSample`.
        - 'loss': Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general
            data_samples (list[:obj:`PoseDataSample`], optional): The
                annotation of every sample. Defaults to ``None``
            mode (str): Set the forward mode and return value type. Defaults
                to ``'tensor'``

        Returns:
            The return type depends on ``mode``.

            - If ``mode='tensor'``, return a tensor or a tuple of tensors
            - If ``mode='predict'``, return a list of :obj:``PoseDataSample``
                that contains the pose predictions
            - If ``mode='loss'``, return a dict of tensor(s) which is the loss
                function value
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if mode == "loss":
            return self.loss(
                inputs,
                data_samples,
                *args,
                **kwargs,
            )
        elif mode == "predict":
            # use customed metainfo to override the default metainfo
            if self.metainfo is not None:
                for data_sample in data_samples:
                    data_sample.set_metainfo(self.metainfo)
            return self.predict(
                inputs,
                data_samples,
                *args,
                **kwargs,
            )
        elif mode == "tensor":
            return self._forward(
                inputs,
                *args,
                **kwargs,
            )
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "Only supports loss, predict and tensor mode.")

    def loss(
        self,
        inputs: Tensor,
        data_samples: List[PoseDataSample],
        val_step: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> dict:
        feats = self.extract_feat(inputs)

        losses = dict()
        if self.with_head:
            losses, head_feats = self.head.loss(
                feats,
                data_samples,
                train_cfg=self.train_cfg,
                return_features=True,
                *args,
                **kwargs,
            )
            if val_step:
                losses["detections"] = head_feats
            if self.with_feature_extraction:
                loss_feat, feats = self.feature_extraction_head.loss(
                    detection_head_features=head_feats,
                    priors_features=feats,
                    batch_data_samples=data_samples,
                    train_cfg=self.train_cfg,
                    return_features=True,
                    *args,
                    **kwargs,
                )
                if val_step:
                    losses["extracted_features"] = feats
                losses.update(loss_feat)

        return losses

    def predict(self, inputs: Tensor, data_samples: List[PoseDataSample] = None) -> Tuple[Tensor]:
        return self._forward(inputs, data_samples)

    def _forward(self, inputs: Tensor, data_samples: Optional[List[PoseDataSample]] = None) -> Union[Tensor, Tuple[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            Union[Tensor | Tuple[Tensor]]: forward output of the network.
        """

        x = self.extract_feat(inputs)
        if self.with_head:
            y = self.head.forward(x)
            if self.with_feature_extraction:
                feats = self.feature_extraction_head.forward(x)
                y = y[:5] + (feats,) + y[6:]
            return y
        return x

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        data = self.data_preprocessor(data, False)
        return [self.loss(**data, val_step=True)]

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)

        return x

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args, **kwargs):
        """A hook function to.

        1) convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        2) remove the weights in data_preprocessor to avoid warning
        `unexpected key in source state_dict: ...`. These weights are
        initialized with given arguments and remain same during training
        and inference.

        The hook will be automatically registered during initialization.
        """

        keys = list(state_dict.keys())

        # remove the keys in data_preprocessor to avoid warning
        for k in keys:
            if k in ("data_preprocessor.mean", "data_preprocessor.std"):
                del state_dict[k]

        version = local_meta.get("version", None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        for k in keys:
            if "keypoint_head" in k:
                v = state_dict.pop(k)
                k = k.replace("keypoint_head", "head")
                state_dict[k] = v
