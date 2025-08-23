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
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metainfo = metainfo
        self.logger = MMLogger.get_current_instance()
        self.output_file = None
        self.output_results = None
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
            )
            self.results.append(evaluation_results)

    def compute_metrics(self, results: list) -> dict:
        metrics = defaultdict(list)
        for result in results:
            for cls, scores in result.items():
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
                if self.output_results is not None:
                    self.output_results[cls][metric] = score
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
                self.output_results["Overall"][metric] = score

        if self.output_results is not None:
            self.save_results()

        return out_metrics

    def save_results(self):
        df = pd.DataFrame.from_dict(self.output_results, orient="index")
        if "Overall" in df.index:
            overall_row = df.loc[["Overall"]]
            df = pd.concat([overall_row, df.drop(index="Overall")])

        df.index.name = "Class"
        df = df.round(4)
        df.to_csv(self.output_file)
