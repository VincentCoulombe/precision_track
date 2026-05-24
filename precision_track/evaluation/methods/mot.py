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

    def identity_purity(self):
        """Frame-weighted GT-side identity purity computed from acc.mot_events.

        For each GT identity g, let f_gp be the number of frames where g and
        predicted id p are Hungarian-matched (IoU >= threshold), and f_g the
        number of frames where g exists at all. Purity is
            sum_g max_p f_gp  /  sum_g f_g
        i.e. the fraction of GT-present frames carrying the dominant predicted
        id for that GT. Unlike IDF1 it does not enforce a global one-to-one
        mapping, so a tracker that swaps and then recovers only loses the
        frames inside the swap window.
        """
        matched_types = {"MATCH", "SWITCH"}
        gt_types = matched_types | {"MISS"}
        purity = {cls: {"purity_gt": -1.0, "num_gt_frames": 0} for cls in self.classes}
        for cls, acc in self.accs.items():
            ev = acc.mot_events
            matched = ev[ev["Type"].isin(matched_types)]
            gt_present = ev[ev["Type"].isin(gt_types)].groupby("OId").size()
            if matched.empty or gt_present.sum() == 0:
                continue
            co = matched.groupby(["OId", "HId"]).size().unstack(fill_value=0)
            dominant = co.max(axis=1).reindex(gt_present.index, fill_value=0)
            purity[cls] = {
                "purity_gt": float(dominant.sum()) / float(gt_present.sum()),
                "num_gt_frames": int(gt_present.sum()),
            }
        return {k: v for k, v in purity.items() if v["num_gt_frames"] > 0}

    def reset(self):
        self.accs = {cls: mm.MOTAccumulator(auto_id=True) for cls in self.classes}
