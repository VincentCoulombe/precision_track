import os
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from precision_track.registry import METRICS
from precision_track.utils import iou_batch, reformat


@METRICS.register_module()
class PoseEstimationECEMetric(BaseMetric):
    default_prefix = ""
    SUPPORTED_BIN_DISTS = ["uniform", "proportional"]

    def __init__(
        self,
        output_dir: Optional[str] = None,
        iou_thr: Optional[float] = 0.5,
        bin_distribution: Optional[str] = "uniform",
        n_bins: Optional[int] = 10,
        collect_device: Optional[str] = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        """Pose estimation Expected Calibration Error (PoseECE) metric. Divides the confidence
        outputs into equally-sized interval bins. In each bin, compute the confidence gap:

        bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

        Where the accuracy_in_bin is calculated based on the IoUs and OKSs of the predicted
        bounding boxes and their labels respectively.

        The final metric is a weighted average of the gaps, based on the number of samples in
        each bin.

        Args:
            output_dir (str, optional): Directory of the .csv files were the ECEs can be saved.
            collect_device (str): Device name used for collecting results from
                different ranks during distributed training. Must be ``'cpu'`` or
                ``'gpu'``. Defaults to ``'cpu'``
            prefix (str, optional): The prefix that will be added in the metric
                names to disambiguate homonymous metrics of different evaluators.
                If prefix is not provided in the argument, ``self.default_prefix``
                will be used instead. Defaults to ``None``
        """
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.logger = MMLogger.get_current_instance()
        self.output_files = None
        self.output_results = None
        if output_dir is not None:
            output_dir = os.path.dirname(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_dir, _ = os.path.splitext(output_dir)
            self.output_files = [
                os.path.join(output_dir, "uncalibrated_detections.csv"),
                os.path.join(output_dir, "calibrated_detections.csv"),
            ]
            self.output_results = []
            for _ in self.output_files:
                self.output_results.append(
                    dict(
                        bin=[],
                        total=[],
                        proportion=[],
                        count_true_detection=[],
                        count_false_detection=[],
                        mean_accuracy=[],
                        mean_confidence=[],
                    )
                )
            self.logger.info(f"The detection calibration results will be saved at {os.path.abspath(self.output_files[1])}")
        assert 0 <= iou_thr <= 1, "The IoU threshold must be a float between 0 and 1."
        self.iou_thr = iou_thr

        assert bin_distribution in self.SUPPORTED_BIN_DISTS, f"{bin_distribution} is not one of the supported bin distributions: {self.SUPPORTED_BIN_DISTS}."
        self.bin_distribution = bin_distribution

        assert 0 < n_bins
        self.n_bins = n_bins

        self.bin_lowers = None
        self.bin_uppers = None

        self.learning_rate = 1e-2

        self.temperature = 0.75
        self.bboxes_temp_optim_dir = -1 if self.temperature < 1 else 1

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            score_logits = data_sample["logits"][0].cpu()
            objectness_logits = data_sample["logits"][1].cpu()
            scores, labels = self._logits_to_scores_and_labels(score_logits, objectness_logits)
            assert np.allclose(scores.cpu().numpy(), data_sample["pred_instances"]["scores"].cpu().numpy())

            gt_bboxes = data_sample["gt_instances"]["bboxes"].cpu().numpy()
            gt_labels = data_sample["gt_instances"].get("labels")
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.cpu()
                valid_mask = torch.zeros_like(scores)
                if gt_bboxes.size:
                    gt_bboxes = reformat(gt_bboxes, "xyxy", "cxcywh")
                    ious = torch.tensor(iou_batch(gt_bboxes, data_sample["pred_instances"]["bboxes"].cpu().numpy()))
                    cls_ = labels.view(1, -1) == gt_labels.view(-1, 1)
                    ious *= cls_
                    val, gt_idx = torch.max(ious, dim=0)
                    iou_mask = val > self.iou_thr
                    gt_idx = gt_idx[iou_mask]
                    valid_mask[iou_mask] = 1

                self.results.append(
                    dict(
                        score_logits=score_logits,
                        objectness_logits=objectness_logits,
                        scores=scores,
                        score_labels=valid_mask,
                    )
                )

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        scores = []
        score_logits = []
        objectness_logits = []
        score_labels = []
        for result in results:
            scores.append(result["scores"])
            score_logits.append(result["score_logits"])
            objectness_logits.append(result["objectness_logits"])
            score_labels.append(result["score_labels"])

        scores = torch.cat(scores)
        score_logits = torch.cat(score_logits)
        score_labels = torch.cat(score_labels)
        objectness_logits = torch.cat(objectness_logits)

        sorted_scores, _ = torch.sort(scores)
        if self.bin_distribution == "uniform":
            min_ = round(sorted_scores[0].item(), 2)
            bin_boundaries = torch.linspace(min_, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        if self.bin_distribution == "proportional":
            bin_size = (len(sorted_scores) + self.n_bins - 1) // self.n_bins
            bins = torch.split(sorted_scores, bin_size)
            self.bin_lowers = []
            self.bin_uppers = []
            for bin in bins:
                self.bin_lowers.append(round(bin[0].item(), 2))
                self.bin_uppers.append(round(bin[-1].item(), 2))
            self.bin_lowers = torch.tensor(self.bin_lowers)
            self.bin_uppers = torch.tensor(self.bin_uppers)

        self.bboxes_ece, bboxes_loss = self._calculate_ece(scores, score_labels, 0, by_prop=False)
        self.bboxes_loss = bboxes_loss.item()
        self.logger.info(f"Pre calibration ECE: {self.bboxes_ece: .4f}. Pre calibration loss: {self.bboxes_loss: .4f}")

        metrics = dict(bboxes_ece=self.bboxes_ece)

        self._find_optimal_temperature(score_logits, objectness_logits, score_labels)

        metrics["calibrated_temperature"] = self.temperature

        calibrated_scores, _ = self._logits_to_scores_and_labels(score_logits * self.temperature, objectness_logits)
        calibrated_bboxes_ece, loss = self._calculate_ece(calibrated_scores, score_labels, 1, by_prop=False)
        self.logger.info(f"Optimal temperature found: {self.temperature: .4f}")
        self.logger.info(f"Post calibration ECE: {calibrated_bboxes_ece: .4f}. Post calibration loss: {loss.item(): .4f}")
        metrics["calibrated_bboxes_ece"] = calibrated_bboxes_ece

        if self.output_files is not None:
            self.logger.info("Saving the calibration results...")
            for path, result in zip(self.output_files, self.output_results):
                df = pd.DataFrame(result)
                df.to_csv(path)

        return metrics

    def _calculate_score_confidences(self, score_logits, objectness_logits):
        scores = score_logits.sigmoid()
        objectness = objectness_logits.sigmoid()
        return scores * objectness

    def _calculate_ece(self, confidences, accuracies, output_idx, by_prop=True):
        ece = torch.zeros(1)
        loss = torch.zeros(1)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                bin_ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                bin_loss = torch.abs(((bin_upper + bin_lower) / 2) - accuracy_in_bin)

                if by_prop:
                    bin_ece *= prop_in_bin
                    bin_loss *= prop_in_bin
                ece += bin_ece
                loss += bin_loss

                if self.output_results is not None and output_idx >= 0:
                    self.output_results[output_idx]["bin"].append(f"{bin_lower.item():.2f}-{bin_upper.item():.2f}")
                    self.output_results[output_idx]["total"].append(in_bin.sum().item())
                    self.output_results[output_idx]["proportion"].append(prop_in_bin.item())
                    ctd = accuracies[in_bin].sum().item()
                    self.output_results[output_idx]["count_true_detection"].append(ctd)
                    self.output_results[output_idx]["count_false_detection"].append(accuracies[in_bin].shape[0] - ctd)
                    self.output_results[output_idx]["mean_accuracy"].append(accuracy_in_bin.item())
                    self.output_results[output_idx]["mean_confidence"].append(avg_confidence_in_bin.item())
            else:
                bin_ece = 1
                bin_loss = 1
            ece += bin_ece
            loss += bin_loss

        return ece.item(), loss

    def _find_optimal_temperature(self, logits, objectness_logits, labels, nb_epochs=100):
        best_loss = self.bboxes_loss
        best_temp = 1
        for _ in range(nb_epochs):
            temp_logits = logits * self.temperature
            calibrated_scores, _ = self._logits_to_scores_and_labels(temp_logits, objectness_logits)
            _, loss = self._calculate_ece(calibrated_scores, labels, -1, by_prop=False)
            grad = self.bboxes_loss - loss
            optimizing = grad.sign().item()
            loss = loss.item()
            if optimizing == 0:
                break
            self.bboxes_temp_optim_dir *= optimizing
            self.temperature += self.learning_rate * self.bboxes_temp_optim_dir
            if loss < best_loss:
                best_loss = loss
                best_temp = self.temperature
            self.bboxes_loss = 0.9 * loss + 0.1 * self.bboxes_loss
        self.temperature = best_temp

    @staticmethod
    def _logits_to_scores_and_labels(score_logits, objectness_logits):
        scores = score_logits.sigmoid()
        objectness = objectness_logits.sigmoid()
        scores = scores * objectness
        scores, labels = scores.max(1, keepdim=True)
        return scores.flatten(), labels
