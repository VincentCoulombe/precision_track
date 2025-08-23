import os
from typing import List, Optional

import motmetrics as mm
import numpy as np
import pandas as pd
from mmengine.logging import MMLogger


class MOTEvaluation(object):
    supported_metrics = [
        "mota",
        "idf1",
        "idp",
        "idr",
        "precision",
        "recall",
        "idfp",
        "idfn",
        "idtp",
        "num_switches",
        "num_detections",
    ]

    def __init__(self, classes: List[str], save_path: Optional[str] = None) -> None:
        mm.lap.default_solver = "scipy"
        assert isinstance(classes, list)
        self.classes = classes
        self.accs = {cls: mm.MOTAccumulator(auto_id=True) for cls in self.classes}
        self.logger = MMLogger.get_current_instance()
        self.save_path = None
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            name, _ = os.path.splitext(save_path)
            self.save_path = f"{name}.csv"

    def update(self, frame_preds: dict, frame_gt: dict):
        """Update the motmetrics accumulator with the current frame predictions and ground truth
        Args:
            frame_preds (dict): The predictions for all the instances of a class in the current frame.
            Of shape {class: [[object_id, bb_left, bb_top, bb_width, bb_height]]}
            frame_gt (dict): The ground truth for all the instances of a class in the current frame.
            Of shape {class: [[object_id, bb_left, bb_top, bb_width, bb_height]]}
        """
        for cls, acc in self.accs.items():
            cls_gt = np.array(frame_gt[cls]) if len(frame_gt[cls]) > 0 else np.zeros((0, 5))
            cls_preds = np.array(frame_preds[cls]) if len(frame_preds[cls]) > 0 else np.zeros((0, 5))
            cost = mm.distances.iou_matrix(
                cls_gt[:, 1:],
                cls_preds[:, 1:],
                max_iou=0.5,
            )

            acc.update(
                cls_gt[:, 0].astype(int).tolist(),
                cls_preds[:, 0].astype(int).tolist(),
                cost,
            )

    def evaluate(self):
        evaluation = {cls: {sm: -1 for sm in self.supported_metrics} for cls in self.classes}
        mh = mm.metrics.create()

        summary = mh.compute_many(
            list(self.accs.values()),
            metrics=self.supported_metrics,
            names=list(self.accs.keys()),
        )
        for cls, metric in evaluation.items():
            for sm in self.supported_metrics:
                metric[sm] = summary.loc[cls][sm]
        evaluation = {k: v for k, v in evaluation.items() if v["num_detections"] > 0.0}

        if self.save_path is not None:
            self.logger.info(f"Saving the MOT evaluation to {self.save_path}")
            eval_df = pd.DataFrame(evaluation).T.reset_index()
            eval_df.rename(columns={"index": "class"}, inplace=True)
            eval_df.to_csv(self.save_path, index=False)
        return evaluation

    def reset(self):
        self.accs = {cls: mm.MOTAccumulator(auto_id=True) for cls in self.classes}
