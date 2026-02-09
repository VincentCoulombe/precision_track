import os
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from precision_track.evaluation.utils.mot import evaluate_mot
from precision_track.registry import METRICS


@METRICS.register_module()
class CLEARMetrics(BaseMetric):
    """As defined in: https://www.researchgate.net/publication/26523191_Evaluating_multiple_object_tracking_performance_The_CLEAR_MOT_metrics"""

    default_prefix = "CLEAR"
    metrics_agg = dict(
        mota="mean",
        idf1="mean",
        idp="mean",
        idr="mean",
        precision="mean",
        recall="mean",
        idfp="mean",
        idfn="mean",
        idtp="mean",
        num_switches="sum",
        num_detections="sum",
    )

    def __init__(
        self,
        metainfo: str,
        collect_device: str = "cpu",
        output_file: Optional[str] = None,
        prefix: Optional[str] = None,
        report_every_prcnt: Optional[float] = 1.0,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metainfo = metainfo
        self.logger = MMLogger.get_current_instance()
        self.output_file = None
        self.output_results = None
        self.report_every_prcnt = report_every_prcnt
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            output_file, _ = os.path.splitext(output_file)
            self.output_file = f"{output_file}.csv"
            self.output_results = defaultdict(dict)
            self.logger.info(f"The test results will be saved at {os.path.abspath(self.output_file)}.")

    def process(self, data_batch: Any, data_samples: Any) -> None:
        for pred, gt in zip(data_batch, data_samples):
            evaluation_results = evaluate_mot(
                pred,
                gt,
                self.metainfo,
                save_path=None,
                verbose=True,
                report_every_prcnt=self.report_every_prcnt,
            )
            self.results.append(evaluation_results)

    def compute_metrics(self, results: list) -> dict:
        if self.output_results is not None:
            self.all_evaluations = []
            for result in results:
                self.all_evaluations.extend(result)

        metrics = defaultdict(list)

        for result in results:
            last_eval = result[-1]
            for cls, scores in last_eval.items():
                if cls == "completion_prcnt":
                    continue
                metrics[cls].append([max(float(scores[s]), 0.0) for s in self.metrics_agg.keys()])

        out_metrics = defaultdict(float)
        overall = defaultdict(list)
        for cls in metrics:
            cls_metrics = np.array(metrics[cls])
            for i, metric in enumerate(self.metrics_agg):
                i_metrics = cls_metrics[:, i]
                if self.metrics_agg[metric] == "mean":
                    score = np.mean(i_metrics)
                else:
                    score = np.sum(i_metrics)
                out_metrics[f"{cls}/{metric}"] = score
                overall[f"Overall/{metric}"].append(score)

        for k in overall:
            metric = k.split("/")[1]
            if self.metrics_agg[metric] == "mean":
                score = np.mean(overall[k])
            else:
                score = np.sum(overall[k])
            out_metrics[f"Overall/{metric}"] = score

        if self.output_results is not None:
            self.save_results()

        return out_metrics

    def save_results(self):
        prcnt_checkpoints = sorted(set(eval["completion_prcnt"] for eval in self.all_evaluations))
        classes = []
        for eval in self.all_evaluations:
            for k in eval.keys():
                if k != "completion_prcnt" and k not in classes:
                    classes.append(k)

        rows = []
        for metric in self.metrics_agg.keys():
            for cls in classes:
                row = {"Metric": metric, "Class": cls}
                for prcnt in prcnt_checkpoints:
                    values = [max(float(eval[cls][metric]), 0.0) for eval in self.all_evaluations if eval["completion_prcnt"] == prcnt and cls in eval]
                    if values:
                        if self.metrics_agg[metric] == "mean":
                            row[f"{int(prcnt * 100)}%"] = np.mean(values)
                        else:
                            row[f"{int(prcnt * 100)}%"] = np.sum(values)
                rows.append(row)

        df = pd.DataFrame(rows)
        df = df.round(4)
        df.to_csv(self.output_file, index=False)
