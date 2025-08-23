# Copyright (c) WildlifeDatasets . All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import operator
import os
from collections import defaultdict
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from sklearn.metrics import classification_report, confusion_matrix

from precision_track.outputs import CsvBoundingBoxes, CsvSearchAreas
from precision_track.registry import METRICS
from precision_track.utils import PoseDataSample, batch_bbox_areas, iou_batch, linear_assignment, oks_batch, parse_pose_metainfo, reformat, wilson_bounds


@METRICS.register_module()
class SequentialSimilarityMetric(BaseMetric):
    default_prefix = "SequentialSimilarityMetric"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        """The SequentialSimilarityMetric evaluate accuracy of a cosine similarity matrix between previous and current appearance embeddings.

        Args:
            collect_device (str): Device name used for collecting results from
                different ranks during distributed training. Must be ``'cpu'`` or
                ``'gpu'``. Defaults to ``'cpu'``
            prefix (str, optional): The prefix that will be added in the metric
                names to disambiguate homonymous metrics of different evaluators.
                If prefix is not provided in the argument, ``self.default_prefix``
                will be used instead. Defaults to ``None``
        """
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Any, data_samples: Any) -> None:
        """Process one batch of data samples and predictions.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Any): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            cosine_similarities = data_sample["similarities"]
            targets = data_sample["targets"]
            assert cosine_similarities.shape == targets.shape, f"{cosine_similarities.shape} != {targets.shape}"
            targets = torch.argmax(data_sample["targets"], dim=1)
            preds = torch.argmax(cosine_similarities, dim=1)
            n_true = (targets == preds).sum().item()
            ce = F.cross_entropy(input=cosine_similarities, target=targets).item()

            targets = targets.cpu().numpy()
            preds = preds.cpu().numpy()
            sequence_data_samples = data_sample["data_samples"]
            assert len(sequence_data_samples) == 2
            t0_bboxes = sequence_data_samples[0]["pred_instances"]["bboxes"].cpu().numpy()
            t1_bboxes = sequence_data_samples[1]["pred_instances"]["bboxes"].cpu().numpy()
            ious = iou_batch(t0_bboxes, t1_bboxes)
            assert cosine_similarities.shape == ious.shape
            iou_preds = np.argmax(ious, axis=1)
            n_true_ious = (targets == iou_preds).sum().item()
            n_agreed_ious = (preds == iou_preds).sum().item()
            gt_areas = batch_bbox_areas(sequence_data_samples[0]["gt_instances"]["bboxes"])
            oks = oks_batch(
                sequence_data_samples[0]["pred_instances"]["keypoints"].cpu().numpy(),
                sequence_data_samples[1]["pred_instances"]["keypoints"].cpu().numpy(),
                gt_areas,
            )
            assert oks.shape == ious.shape
            oks_preds = np.argmax(oks, axis=1)
            n_true_oks = (targets == oks_preds).sum().item()
            n_agreed_oks = (preds == oks_preds).sum().item()
            self.results.append(
                dict(
                    n_true=n_true,
                    n_pred=targets.size,
                    ce=ce,
                    n_true_ious=n_true_ious,
                    n_agreed_ious=n_agreed_ious,
                    n_true_oks=n_true_oks,
                    n_agreed_oks=n_agreed_oks,
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

        metrics = defaultdict(list)
        for result in results:
            n_pred = result["n_pred"]
            if n_pred > 0:
                metrics["accuracy"].append(result["n_true"] / n_pred)
                metrics["cross_entropy"].append(result["ce"])
                metrics["ious_accuracy"].append(result["n_true_ious"] / n_pred)
                metrics["oks_accuracy"].append(result["n_true_oks"] / n_pred)
                metrics["agrees_w_ious"].append(result["n_agreed_ious"] / n_pred)
                metrics["agrees_w_oks"].append(result["n_agreed_oks"] / n_pred)
        for k in metrics:
            metrics[k] = sum(metrics[k]) / len(metrics[k])
        return metrics


@METRICS.register_module()
class SequentialAverageAccuracy(BaseMetric):
    default_prefix = "SequentialAverageAccuracy"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Any, data_samples: Any) -> None:
        """Process one batch of data samples and predictions.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Any): A batch of outputs from
                the model.
        """

        for i, probs in enumerate(data_samples):
            gt = data_batch["data_samples"][i].gt_instance_labels.action_labels.to(probs.device).long()
            ce = F.cross_entropy(input=probs.view(1, -1), target=gt).item()
            pred = torch.argmax(probs)
            acc = gt == pred
            gt_label = data_batch["data_samples"][i].gt_instances.actions
            self.results.append((gt_label.item(), acc.int().item(), ce))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        metrics = defaultdict(list)
        cross_entropy = []
        for result in results:
            metrics[str(result[0])].append(result[1])
            cross_entropy.append(result[2])
        lengths = []
        total = 0.0
        for k in metrics:
            k = str(k)
            lenk = len(metrics[k])
            sumk = sum(metrics[k])
            lengths.append(lenk)
            acck = sumk / (lenk + 1e-6)
            total += acck
            metrics[k] = acck
        metrics["Mean"] = total / (len(metrics) + 1e-6)
        metrics["Cross Entropy"] = sum(cross_entropy) / len(cross_entropy)
        return metrics


@METRICS.register_module()
class MultiClassActionRecognitionMetrics(BaseMetric):
    default_prefix = "ActionRecognition"
    label_index_modes = ["last", "spacial"]

    def __init__(
        self,
        metainfo: str,
        confusion_matrix_save_dir: str = None,
        label_index_mode: str = "last",
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        self.metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.confusion_matrix_save_dir = confusion_matrix_save_dir
        if self.confusion_matrix_save_dir is not None:
            os.makedirs(self.confusion_matrix_save_dir, exist_ok=True)
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.label_to_action = defaultdict(str)
        for i, acc in enumerate(self.metainfo.get("actions", [])):
            self.label_to_action[i] = acc
        self.best_f1 = 0
        assert label_index_mode in self.label_index_modes, f"{label_index_mode} must be one of: {self.label_index_modes}."
        self.label_index_mode = label_index_mode

    def _fetch_gt_action(self, idx: int, probs: torch.Tensor, gt_data_samples: List[PoseDataSample]):
        if self.label_index_mode == "last":
            return (
                gt_data_samples[idx].gt_instance_labels.action_labels[-1].to(probs.device).long(),
                gt_data_samples[idx].gt_instances.actions[-1].item(),
            )
        elif self.label_index_mode == "spacial":
            assert len(gt_data_samples) == 1, "spacial index mode requires batch size = 1."
            gt_data_sample = gt_data_samples[0]
            return (
                gt_data_sample.gt_instance_labels.action_labels[idx].to(probs.device).long(),
                gt_data_sample.gt_instances.actions[idx].item(),
            )

    def process(self, data_batch: Any, data_samples: Any) -> None:
        """Process one batch of data samples and predictions."""
        if isinstance(data_samples, list):
            data_samples = data_samples[0]
        for i, probs in enumerate(data_samples):
            gt, action = self._fetch_gt_action(i, probs, data_batch["data_samples"])
            pred = torch.argmax(probs).item()
            ce = F.cross_entropy(input=probs.view(1, -1), target=gt.view(-1)).item()
            gt_label = gt.item()
            self.label_to_action[str(gt_label)] = action
            self.results.append((action, gt_label, pred, ce))

    def compute_metrics(self, results: list) -> dict:
        """Compute macro F1, balanced accuracy, per-class accuracy, CE, and confusion matrix."""
        y_true = [r[1] for r in results]
        y_pred = [r[2] for r in results]
        ce_vals = [r[3] for r in results]

        metrics = dict()

        report = classification_report(y_true, y_pred, digits=3, output_dict=True)
        f1 = report["macro avg"]["f1-score"]

        metrics["Macro F1"] = f1
        metrics["Cross Entropy"] = sum(ce_vals) / len(ce_vals)
        for k in report:
            if k in self.label_to_action:
                action = self.label_to_action[k]
                metrics[f"{action} Precision"] = report[k]["precision"]
                metrics[f"{action} Recall"] = report[k]["recall"]

        if f1 > self.best_f1:
            self.best_f1 = f1
        if os.path.isdir(self.confusion_matrix_save_dir):
            labels = sorted(list(set(y_true + y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            label_names = [str(self.label_to_action.get(i, f"Class_{i}")) for i in labels]
            df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
            save_path = os.path.join(self.confusion_matrix_save_dir, f"confusion_matrix_f1_{f1:.3f}.csv")
            df_cm.to_csv(save_path)

            metrics["Confusion Matrix"] = f"Saved to {save_path}"

        return metrics


@METRICS.register_module()
class ReIDKnnClassifier(BaseMetric):
    default_prefix = "ReIDTop1Accuracy"

    def __init__(self, k: int = 1, collect_device: str = "cpu", prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.k = k
        self.results_set = False

    def process(self, data_batch: Any, data_samples: Any) -> None:
        for i, data_sample in enumerate(data_samples):
            query_labels = data_sample["query_labels"].cpu().numpy()
            gallery_labels = data_sample["gallery_labels"].cpu().numpy()

            similarities = data_sample["similarities"]
            scores, idx = similarities.topk(k=self.k, dim=1)
            preds = gallery_labels[idx.cpu().numpy()]
            scores = scores.cpu().numpy()

            data = []
            for pred, score in zip(preds, scores):
                vals, counts = np.unique(pred, return_counts=True)
                winners = vals[counts.max() == counts]

                # Check for ties
                if len(winners) == 1:
                    best_pred = winners[0]
                    best_score = score[best_pred == pred].mean()
                else:
                    is_winner = np.isin(pred, winners)
                    ties = pd.Series(score[is_winner]).groupby(pred[is_winner]).mean()
                    best_pred = ties.idxmax()
                    best_score = ties.max()
                data.append([best_pred, best_score])

            preds, scores = list(zip(*data))

            self.results.append(sum(preds == query_labels) / len(preds))

    def compute_metrics(self, results: list) -> dict:
        accuracies = np.array(results)
        return dict(avg=accuracies.mean())


@METRICS.register_module()
class SearchZoneStitchingMetric(BaseMetric):
    default_prefix = "Stitching"
    CONF_LEVEL = 0.95
    MIN_SAMPLES = 30
    IOU_THR = 0.65

    def __init__(
        self,
        metainfo: str,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metainfo = metainfo
        self._set_map()

    def _set_map(self):
        self.pred_to_gt = defaultdict()

    def process(self, data_batch: Any, data_samples: Any, reset_map: bool = False) -> None:
        if reset_map:
            self._set_map()

        bboxes = CsvBoundingBoxes(
            path=data_batch[0],
            precision=64,
        )
        bboxes.read()
        zones = CsvSearchAreas(
            path=data_batch[1],
            precision=64,
        )
        zones.read()
        zones = zones.to_dataframe()

        gts = CsvBoundingBoxes(
            path=data_samples[0],
            precision=64,
        )
        gts.read()

        zones = zones.sort_values(["instance_id", "frame_id"]).reset_index(drop=True)
        zones["n_active_zone"] = zones.groupby("frame_id")["instance_id"].transform("nunique")
        new_zone = zones["instance_id"].ne(zones["instance_id"].shift()) | (zones["frame_id"] != zones["frame_id"].shift() + 1)
        zones["chunk_id"] = new_zone.cumsum()
        for (_, _), g in zones.groupby(["instance_id", "chunk_id"]):
            instance_id = g["instance_id"].tolist()[0]

            if instance_id not in self.pred_to_gt:
                self.map_gt_to_pred_id(bboxes, gts, instance_id, len(bboxes))

            n_active_zone = g["n_active_zone"].tolist()[-1]
            first_frame = g["frame_id"].tolist()[0]
            last_frame = g["frame_id"].tolist()[-1]
            frame_before = first_frame - 1
            frame_after = last_frame + 1

            frame_before_entities = np.array(bboxes[frame_before])
            frame_before_ids = frame_before_entities[:, 2]

            first_frame_entities = np.array(bboxes[first_frame])
            first_frame_ids = first_frame_entities[:, 2]

            mask = ~np.isin(frame_before_ids, first_frame_ids)
            idxs = np.where(mask)[0]
            lost_ids = frame_before_ids[idxs]

            if len(lost_ids) == 0:
                self.results.append((n_active_zone, 0))  # Triggered a search without loosing entities...
            elif instance_id not in lost_ids:
                self.results.append((n_active_zone, 0))  # Triggered a search for the wrong instance id...
            elif bboxes[frame_after]:
                inst_id_bbox_mask = np.array(bboxes[frame_after])[:, 2] == instance_id
                if np.any(inst_id_bbox_mask):
                    inst_id_bbox = np.array(bboxes[frame_after])[inst_id_bbox_mask][:, 3:7]

                    delay, coresponding_bbox_mask = self.get_coresponding_bbox_mask(gts, self.pred_to_gt[instance_id], frame_after)
                    if np.any(coresponding_bbox_mask):
                        corresponding_bbox = reformat(np.array(gts[frame_after + delay])[coresponding_bbox_mask][:, 3:7], "xywh", "cxcywh")
                        corresponding = iou_batch(inst_id_bbox, corresponding_bbox).item() > self.IOU_THR / (1 + 0.1 * delay)
                        if corresponding:
                            self.results.append((n_active_zone, 1))  # The lost entity was correctly stitched...
                        else:
                            self.results.append((n_active_zone, 0))  # The lost entity was incorrectly stitched...
                    else:
                        self.results.append((n_active_zone, 0))  # The lost entity was incorrectly stitched...
                else:
                    self.results.append((n_active_zone, 0))  # The lost entity was not stitched...

    def compute_metrics(self, results: list) -> dict:
        total_scores = defaultdict(list)
        for n_active_zone, score in self.results:
            total_scores[n_active_zone].append(score)
        out = dict()
        score = 0
        for n_active_zone, total_score in total_scores.items():
            n_score = sum(total_score) / len(total_score)
            out[f"{n_active_zone} active search zones"] = n_score
            score += n_score * len(total_score) / len(self.results)
        out["weighted avg"] = score
        return out

    def map_gt_to_pred_id(self, preds: CsvBoundingBoxes, gts: CsvBoundingBoxes, pred_id: int, stop_at: int):
        n = 0
        counts = defaultdict(int)
        i = 0

        while True:
            i += 1
            if i == stop_at:
                self.pred_to_gt[pred_id] = -1
                return

            frame_preds = np.array(preds[i])
            frame_gts = np.array(gts[i])
            inst_pred = frame_preds[:, 2] == pred_id
            if not np.any(inst_pred):
                continue

            matched_gt_idx = self.get_best_gt_id(frame_preds[inst_pred][:, 3:7], frame_gts[:, 3:7])
            if matched_gt_idx == -1:
                continue
            matched_gt_id = int(frame_gts[matched_gt_idx, 2].item())
            counts[matched_gt_id] += 1

            n += 1
            if n < self.MIN_SAMPLES:
                continue

            most = (-1, -1)
            second_most = (-1, -1)
            ranked = dict(sorted(counts.items(), key=operator.itemgetter(1), reverse=True))
            for i, (k, v) in zip(range(2), ranked.items()):
                if i == 0:
                    most = (k, v)
                elif i == 1:
                    second_most = (k, v)

            low_winner, _ = wilson_bounds(most[1], n, self.CONF_LEVEL)
            _, up_runner_up = wilson_bounds(second_most[1], n, self.CONF_LEVEL)

            if low_winner > up_runner_up:
                break

        self.pred_to_gt[pred_id] = most[0]

    def get_best_gt_id(self, pred, gts):
        ious = iou_batch(pred, reformat(gts, "xywh", "cxcywh"))
        x, y = linear_assignment(-ious)
        if ious[x, y] < self.IOU_THR:
            return -1
        return y

    @staticmethod
    def get_coresponding_bbox_mask(gts, gt_id, frame_after, max_retry=5):
        coresponding_bbox_mask = np.array(False)
        n = -1
        max_retry = max(0, max_retry)
        while n < max_retry and not np.any(coresponding_bbox_mask):
            n += 1
            coresponding_bbox_mask = np.array(gts[frame_after + n])[:, 2] == gt_id
        return n, coresponding_bbox_mask
