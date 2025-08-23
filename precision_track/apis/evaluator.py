from typing import Optional

import motmetrics as mm
import numpy as np

from precision_track.evaluation.methods.mot import MOTEvaluation
from precision_track.utils import parse_pose_metainfo, reformat


class Evaluator(MOTEvaluation):
    def __init__(self, metafile: str, save_path: Optional[str] = None) -> None:
        metadata = parse_pose_metainfo({"from_file": metafile})
        super().__init__(classes=metadata.get("classes", ["Unknown"]), save_path=save_path)

    def update(self, frame_preds: np.ndarray, frame_gt: np.ndarray, max_iou: int = 0.5):
        """Update the motmetrics accumulator with the current frame predictions
        and ground truth.

        Args:
            frame_preds (np.ndarray): The predictions for all the instances in the current frame.
                Of shape [[class_id, object_id, bb_left, bb_top, bb_width, bb_height]]
            frame_gt (np.ndarray): The ground truth for all the instances in the current frame.
                Of shape [[class_id, object_id, bb_left, bb_top, bb_width, bb_height]]
            idx (int): The frame idx from within the frame sequence.
        """
        for a, n in zip([frame_preds, frame_gt], ["predictions", "ground truth"]):
            assert a.shape[1] == 7, f"The {n} must have the following format: [[class_id, object_id, bb_left, bb_top, bb_width, bb_height, score]]."
        for cls, acc in self.accs.items():
            label = self.classes.index(cls)
            cls_gt = frame_gt[frame_gt[:, 0] == label][:, 1:-1]
            cls_preds = frame_preds[frame_preds[:, 0] == label][:, 1:-1]
            cls_preds[:, 1:] = reformat(cls_preds[:, 1:], "cxcywh", "xywh")

            ious = mm.distances.iou_matrix(
                cls_gt[:, 1:],
                cls_preds[:, 1:],
                max_iou=0.5,
            )
            acc.update(
                cls_gt[:, 0].astype(int).tolist(),
                cls_preds[:, 0].astype(int).tolist(),
                ious,
            )
