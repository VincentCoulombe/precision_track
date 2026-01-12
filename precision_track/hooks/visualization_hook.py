# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from typing import Optional, Sequence

import mmengine
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmengine.structures import InstanceData
from mmengine.dataset import Compose
from copy import deepcopy
import cv2
import numpy as np

from precision_track.datasets.transforms.loading import imfrombytes
from precision_track.registry import HOOKS
from precision_track.utils import PoseDataSample, merge_data_samples, reformat
from precision_track.visualization import ColorPalette


@HOOKS.register_module()
class ActivatedPriorsVisualizationHook(Hook):
    ALLOWED_AUG_SEQ = ["LoadImage", "BottomupRandomAffine", "FilterAnnotations", "GenerateTarget", "PackPoseInputs"]

    def __init__(
        self,
        augmentations: dict,
        radius: Optional[int] = 4,
        font_size: Optional[float] = 1.0,
        palette_size: Optional[int] = 100,
        interval: Optional[int] = 50,
        out_dir: Optional[str] = None,
    ):
        # 1) vérifier que les augmentations sont OK
        aug_names = [x["type"] for x in augmentations]
        assert (
            aug_names == self.ALLOWED_AUG_SEQ
        ), f"The Activated Prior visualization is intended to be ran with only the augmentations: {self.ALLOWED_AUG_SEQ}. The provided augmentations are: {aug_names}."
        self.pipeline = Compose(augmentations)

        self.palette = ColorPalette(size=palette_size)
        if isinstance(out_dir, str):
            os.makedirs(out_dir, exist_ok=True)
            self.out_dir = out_dir

        self._iter = 0
        self.interval = interval

        self.radius = int(radius)
        self.font_size = float(font_size)

    def after_assignment_iter(self, data_sample: PoseDataSample, priors: list, assigned_gt_idx: list, strides: list) -> None:
        if self._iter % self.interval == 0:
            img_path = data_sample.img_path
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            dummy_ds = deepcopy(data_sample).to_dict()
            dummy_ds = dummy_ds | dummy_ds["gt_instances"] | dict(num_keypoints=np.array([dummy_ds["gt_instances"]["keypoints"].shape[0]]))
            transformed_image = self.pipeline(dummy_ds)["inputs"].numpy().transpose(1, 2, 0)
            mlvl_imgs = {int(strides[0]): np.ascontiguousarray(deepcopy(transformed_image), dtype=np.uint8) for strides in strides}
            lbls_center_coords = reformat(dummy_ds["gt_instances"]["bboxes"], "xyxy", "cxcywh")[:, :2]

            for (x, y, stride_x, stride_y), idx in zip(priors, assigned_gt_idx):
                lbl_center_coords = lbls_center_coords[idx]
                lbl_center_coords = (int(lbl_center_coords[0].item()), int(lbl_center_coords[1].item()))
                if int(stride_x) in mlvl_imgs:
                    img = mlvl_imgs[int(stride_x)]
                    for (dot_x, dot_y), color_idx in zip(
                        [(int(x), int(y)), lbl_center_coords], [self.palette.by_idx(idx).as_rgb(), self.palette.by_idx(-1).as_rgb()]
                    ):
                        cv2.circle(img, (dot_x, dot_y), radius=self.radius, color=color_idx, thickness=-1)

            for k, v in mlvl_imgs.items():
                cv2.imwrite(os.path.join(self.out_dir, f"{img_name}_stride={k}.jpg"), v)

        self._iter += 1


@HOOKS.register_module()
class PoseVisualizationHook(Hook):
    """Pose Estimation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``out_dir`` is specified, it means that the prediction results
        need to be saved to ``out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        enable (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        enable: bool = False,
        interval: int = 50,
        kpt_thr: float = 0.3,
        show: bool = False,
        wait_time: float = 0.0,
        out_dir: Optional[str] = None,
        backend_args: Optional[dict] = None,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.kpt_thr = kpt_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn(
                "The show is True, it means that only "
                "the prediction results are visualized "
                "without storing data, so vis_backends "
                "needs to be excluded."
            )

        self.wait_time = wait_time
        self.enable = enable
        self.out_dir = out_dir
        self._test_index = 0
        self.backend_args = backend_args

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict, outputs: Sequence[PoseDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        self._visualizer.set_dataset_meta(runner.val_evaluator.dataset_meta)

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = data_batch["data_samples"][0].get("img_path")
        img_bytes = fileio.get(img_path, backend_args=self.backend_args)
        img = imfrombytes(img_bytes, channel_order="rgb")
        data_sample = outputs[0]

        # revert the heatmap on the original image
        data_sample = merge_data_samples([data_sample])

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                os.path.basename(img_path) if self.show else "val_img",
                img,
                data_sample=data_sample,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=True,
                show=self.show,
                wait_time=self.wait_time,
                kpt_thr=self.kpt_thr,
                step=total_curr_iter,
            )

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict, outputs: Sequence[PoseDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        if self.out_dir is not None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp, self.out_dir)
            mmengine.mkdir_or_exist(self.out_dir)

        self._visualizer.set_dataset_meta(runner.test_evaluator.dataset_meta)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.get("img_path")
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = imfrombytes(img_bytes, channel_order="rgb")
            data_sample = merge_data_samples([data_sample])

            out_file = None
            if self.out_dir is not None:
                out_file_name, postfix = os.path.basename(img_path).rsplit(".", 1)
                index = len([fname for fname in os.listdir(self.out_dir) if fname.startswith(out_file_name)])
                out_file = f"{out_file_name}_{index}.{postfix}"
                out_file = os.path.join(self.out_dir, out_file)

            self._visualizer.add_datasample(
                os.path.basename(img_path) if self.show else "test_img",
                img,
                data_sample=data_sample,
                show=self.show,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=True,
                wait_time=self.wait_time,
                kpt_thr=self.kpt_thr,
                out_file=out_file,
                step=self._test_index,
            )
