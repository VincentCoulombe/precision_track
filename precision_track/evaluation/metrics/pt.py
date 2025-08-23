import os
from collections import Counter
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from addict import Dict as Addict
from mmengine import Config
from mmengine.evaluator import BaseMetric
from mmengine.fileio import load
from mmengine.logging import MMLogger

from precision_track.registry import METRICS
from precision_track.tracking.precision_track import PrecisionTrack
from precision_track.utils import calc_distances, keypoint_auc, keypoint_pck_accuracy, linear_assignment, reformat, rmse_batch


@METRICS.register_module()
class PoseTrackingMetric(BaseMetric):
    SUPPORTED_INPUT_FORMATS = ["xyxy", "cxcywh"]
    default_prefix = "PT"

    def __init__(
        self,
        ann_file: str,
        association_cfg: dict,
        metafile: str,
        input_format: str = "cxcywh",
        kpt_score_thr: float = 0.5,
        beta: Optional[int] = 0.5,
        output_file: Optional[str] = None,
        out_normalized_distances: Optional[bool] = False,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        """The pose tracking (PT) metric evaluate the ability of the
        pose-estimator to detect a maximum of subjects present in the frame and to infer the
        correct pose of those detected subjects. As such, the metric take the following,
        beta configurable form: (1 - beta) x recall + beta x oks.

        Args:
            ann_file (str): The coco style annotation path.
            association_cfg (dict): The association config. PrecisionTrack's
                associations are used in the evaluation, therefore this config
                correspond to the desired PrecisionTrack parameters.
            metafile (str): Path to the meta information for dataset, such as class
                information.
            kpt_score_thr (float): The threshold at which the predicted keypoints
                are considered for the metric calculations.
            input_format (str, optional): The expected input bounding boxes input format.
                Defaults to "cxcywh".
            beta (Optional[int], optional): The level of importance given to the OKS in
                the PT metric evaluation. Defaults to 0.5.
            output_file (str, optional): Path of the .csv file were the metrics can be saved.
            out_normalized_distances (bool, optional): Output the normalized distances for each keypoint.
            collect_device (str): Device name used for collecting results from
                different ranks during distributed training. Must be ``'cpu'`` or
                ``'gpu'``. Defaults to ``'cpu'``
            prefix (str, optional): The prefix that will be added in the metric
                names to disambiguate homonymous metrics of different evaluators.
                If prefix is not provided in the argument, ``self.default_prefix``
                will be used instead. Defaults to ``None``
        """
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        assert "annotations" in load(ann_file), "Ground truth annotations are required for evaluation."
        self.pt = PrecisionTrack(**association_cfg)
        self.beta = beta
        assert (
            input_format in self.SUPPORTED_INPUT_FORMATS
        ), f"The input format: {input_format} is not one of the supported inpur format: {self.SUPPORTED_INPUT_FORMATS}."
        self.input_format = input_format
        assert 0.0 < kpt_score_thr < 1.0, f"kpt_scores_thr must be between 0 and 1, not {kpt_score_thr}."
        self.kpt_score_thr = kpt_score_thr

        metafile = Config.fromfile(metafile)
        assert isinstance(metafile.get("dataset_info").get("classes"), list), f"The metadata file: {metafile} must contain a list of the detected classes."
        self.classes = metafile["dataset_info"]["classes"]
        self.keypoints = [k["name"] for k in metafile["dataset_info"]["keypoint_info"].values()]

        self.logger = MMLogger.get_current_instance()
        self.output_file = None
        self.output_results = None
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            output_file, _ = os.path.splitext(output_file)
            self.output_file = f"{output_file}.csv"
            self.output_results = dict()
            self.logger.info(f"The test results will be saved at {os.path.abspath(self.output_file)}.")
        self.dist_file = None
        self.dist_results = None
        if out_normalized_distances:
            if output_file is not None:
                self.dist_file = os.path.join(os.path.dirname(self.output_file), "kpt_normalized_distances.csv")
                self.dist_results = dict()
                for kpt in self.keypoints:
                    self.dist_results[kpt] = []
            else:
                self.logger.warning("Need an output file to also output the PCK distributions.")

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            preds = data_sample["pred_instances"]
            for k, v in preds.items():
                preds[k] = v.cpu().numpy()
            bboxes = np.array(preds.get("bboxes", []), dtype=np.float32)
            has_pred = bboxes.size
            if self.input_format == "xyxy":
                bboxes = reformat(bboxes, "xyxy", "cxcywh")
            pred_bboxes = Addict(
                dict(
                    bboxes=bboxes,
                    labels=np.array(preds.get("labels", []), dtype=np.float32),
                    scores=np.array(preds.get("scores", []), dtype=np.float32),
                )
            )
            pred_keypoints = np.array(preds.get("keypoints", []), dtype=np.float32)
            pred_keypoint_scores = np.array(preds.get("keypoint_scores", []), dtype=np.float32)

            gt = data_sample["gt_instances"].to("cpu")
            gt_labels = np.array(gt.get("labels", []), dtype=np.float32)
            gt_bboxes = np.array(gt.get("bboxes", []), dtype=np.float32)
            gt_keypoints = np.array(gt.get("keypoints", []), dtype=np.float32)
            gt_keypoint_scores = np.array(gt.get("keypoints_visible", []), dtype=np.float32)
            gt_idx = np.arange(gt_bboxes.shape[0])
            gt_bbox_size = np.array([], dtype=np.float32)
            gt_masks = np.zeros_like(gt_keypoint_scores, dtype=bool)

            has_gt = gt_bboxes.size
            if has_gt:
                gt_bboxes = reformat(gt_bboxes, "xyxy", "cxcywh")
                assert (
                    0 <= gt_labels.all() < len(self.classes)
                ), f"""There have been at least {np.max(gt_labels)+1} detected classes,
                but the provided classes list only contains {len(self.classes)} classes."""
            nb_bboxes, nb_kpts = gt_bboxes.shape[0], gt_keypoints.shape[0]
            assert nb_bboxes == nb_kpts, f"There is expected to be the same amount of poses ({nb_kpts}) than bboxes ({nb_bboxes}) for each labeled image."

            keep_idx = pred_bboxes["scores"] >= self.pt.obj_score_thrs["low"]
            result = dict(
                gt_count=Counter(gt_labels),
                pred_count=Counter(pred_bboxes["labels"][keep_idx]),
                matches=Counter(),
            )

            ious = self.pt.get_tracks_preds_ious(gt_bboxes, pred_bboxes, keep_idx)
            matched_gts, matched_preds = self.pt.assign_ids(
                dists=np.nan_to_num(ious, nan=1.0),
                tracks={i: dict(labels=l) for i, l in zip(gt_idx, gt_labels)} if gt_labels.size else None,
                track_ids=gt_idx,
                pred_instances=pred_bboxes,
                pred_idx=keep_idx,
                weight_iou_with_det_scores=self.pt.weight_iou_with_det_scores,
                match_iou_thr=self.pt.match_iou_thrs["low"],
            )

            assert matched_gts.shape[0] == matched_preds.shape[0]

            if has_gt:
                gt_bbox_size = np.max(np.concatenate((gt_bboxes[:, 2], gt_bboxes[:, 3]), axis=0))
                gt_bbox_size = np.array([gt_bbox_size, gt_bbox_size]).reshape(-1, 2)
                gt_bboxes = gt_bboxes[matched_gts, :]
                gt_keypoints = gt_keypoints[matched_gts, :, :]
                gt_keypoint_scores = gt_keypoint_scores[matched_gts, :]
                gt_labels = gt_labels[matched_gts]
                gt_masks = gt_keypoint_scores.astype(bool)
                result["matches"].update(gt_labels)
            if has_pred:
                pred_keypoints = pred_keypoints[matched_preds, :, :]
                pred_keypoint_scores = pred_keypoint_scores[matched_preds, :]
                confident_kpts = pred_keypoint_scores >= self.kpt_score_thr
                pred_keypoints[~confident_kpts] = 0.0
            if has_gt and has_pred:
                gt_masks &= confident_kpts

            result.update(
                dict(
                    pred_keypoints=pred_keypoints,
                    pred_keypoint_scores=pred_keypoint_scores,
                    gt_bboxes=gt_bboxes,
                    gt_keypoints=gt_keypoints,
                    gt_keypoint_scores=gt_keypoint_scores,
                    gt_bbox_size=gt_bbox_size,
                    gt_masks=gt_masks,
                    gt_labels=gt_labels,
                )
            )
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        tps = Counter()
        fps = Counter()
        fns = Counter()
        total = Counter()
        proportions = np.zeros(len(self.classes))
        oks = dict()
        rmse = dict()
        pck_20 = dict()
        pck_5 = dict()
        kpts_auc = dict()
        eps = 1e-3
        for result in results:
            tps.update(result["matches"])
            total.update(result["gt_count"])
            for m, r in zip([fps, fns], [result["pred_count"], result["gt_count"]]):
                r_counter = Counter(r)
                r_counter.subtract(result["matches"])
                m.update(r_counter)
            if result["gt_keypoints"].shape[0] > 0:
                for lbl in np.unique(result["gt_labels"]):
                    mask = result["gt_labels"] == lbl
                    gt_bboxes = result["gt_bboxes"][mask, :]
                    gt_keypoints = result["gt_keypoints"][mask, :, :]
                    gt_keypoint_scores = result["gt_keypoint_scores"][mask, :]
                    gt_bbox_size = result["gt_bbox_size"]
                    gt_masks = result["gt_masks"][mask, :]

                    pred_keypoints = result["pred_keypoints"][mask, :, :]
                    pred_keypoint_scores = result["pred_keypoint_scores"][mask, :]

                    for m in [oks, rmse, pck_20, pck_5, kpts_auc]:
                        if lbl not in m:
                            m[lbl] = []
                    result_oks = self.pt.get_tracks_preds_oks(
                        gt_bboxes,
                        gt_keypoints,
                        gt_keypoint_scores,
                        dict(
                            keypoints=pred_keypoints,
                            keypoint_scores=pred_keypoint_scores,
                        ),
                        np.ones(pred_keypoints.shape[0], dtype=bool),
                        eps=0,
                    )
                    matched_gts, matched_preds = linear_assignment(1 - result_oks, 1.0)

                    oks[lbl].append(
                        np.nan_to_num(
                            np.mean(
                                result_oks[matched_gts.astype(int), matched_preds.astype(int)].flatten(),
                                axis=0,
                            )
                        )
                    )
                    if pred_keypoints.shape[0] > 0:
                        rmse[lbl].append(rmse_batch(gt_keypoints, pred_keypoints))
                        for thr, pck in zip([0.2, 0.05], [pck_20, pck_5]):
                            pck[lbl].append(
                                keypoint_pck_accuracy(
                                    pred_keypoints,
                                    gt_keypoints,
                                    gt_masks,
                                    thr,
                                    gt_bbox_size,
                                )[1]
                            )
                        if self.dist_file is not None:
                            assert gt_masks.shape[1] == len(self.keypoints)
                            distances = calc_distances(pred_keypoints, gt_keypoints, gt_masks, gt_bbox_size)
                            for i, d in enumerate(distances):
                                d = d[d > 0]
                                if d.size >= 1:
                                    d = np.mean(d)
                                else:
                                    d = np.NaN
                                self.dist_results[self.keypoints[i]].append(d)

                        kpts_auc[lbl].append(
                            keypoint_auc(
                                pred_keypoints,
                                gt_keypoints,
                                gt_masks,
                                norm_factor=gt_bbox_size[0][0].item(),
                            )
                        )

        assert len(tps) == len(oks) == len(rmse) == len(pck_20) == len(pck_5) == len(kpts_auc)
        metrics = dict()
        avgs = [{} for _ in range(8)]
        for lbl, cls_ in enumerate(self.classes):
            if lbl in tps:
                lbl_oks = np.mean(oks[lbl], axis=0)
                recall = tps[lbl] / (tps[lbl] + fns[lbl])
                precision = tps[lbl] / (tps[lbl] + fps[lbl]) if lbl in fps else 0
                f1_score = 2 / (1 / max(recall, eps) + 1 / max(precision, eps))
                lbl_rmse = np.array(rmse[lbl])
                lbl_rmse = np.mean(lbl_rmse[lbl_rmse > 0], axis=0)
                lbl_pck_20 = np.mean(pck_20[lbl], axis=0)
                lbl_pck_5 = np.mean(pck_5[lbl], axis=0)
                lbl_kpts_auc = np.mean(kpts_auc[lbl], axis=0)
                proportions[int(lbl)] = total[lbl]
                formatted_metrics = self.format_metrics(
                    recall,
                    precision,
                    f1_score,
                    lbl_oks,
                    lbl_rmse,
                    lbl_pck_20,
                    lbl_pck_5,
                    lbl_kpts_auc,
                    proportions[int(lbl)],
                )
                metrics.update({f"{cls_}/{k}": v for k, v in formatted_metrics.items()})
                if self.output_results is not None:
                    self.output_results[cls_] = formatted_metrics
                for i, r in enumerate(
                    [
                        recall,
                        precision,
                        f1_score,
                        lbl_oks,
                        lbl_rmse,
                        lbl_pck_20,
                        lbl_pck_5,
                        lbl_kpts_auc,
                    ]
                ):
                    avgs[i][lbl] = r

        avgs_mat = np.zeros((len(avgs), len(self.classes)))
        for i, r in enumerate(avgs):
            i_list, r_list = list(avgs[i].keys()), list(avgs[i].values())
            if i_list:
                avgs_mat[i][np.array(i_list)] = np.array(r_list)
        evaluation_length = sum(total.values())
        proportions /= evaluation_length

        formatted_metrics = self.format_metrics(*((avgs_mat @ proportions).tolist() + [evaluation_length]))
        metrics.update({f"Overall/{k}": v for k, v in formatted_metrics.items()})
        if self.output_results is not None:
            self.output_results["Overall"] = formatted_metrics
            self.save_results()
        return metrics

    def format_metrics(
        self,
        recall: float,
        precision: float,
        f1: float,
        oks: float,
        rmse: float,
        pck_20: float,
        pck_5: float,
        auc: float,
        sample_size: int,
    ):
        return {
            "Metric": (1 - self.beta) * f1 + self.beta * oks,
            "Recall": recall,
            "Precision": precision,
            "F1 Score": f1,
            "Assignation's OKS": oks,
            "Assignation's RMSE": rmse,
            "Assignation's PCK:0.2": pck_20,
            "Assignation's PCK:0.05": pck_5,
            "Assignation's Keypoints AUC": auc,
            "Sample Size": sample_size,
        }

    def save_results(self):
        df = pd.DataFrame.from_dict(self.output_results, orient="index")
        overall_row = df.loc[["Overall"]]
        df = pd.concat([overall_row, df.drop(index="Overall")])

        df.index.name = "Class"
        df = df.round(4)
        df.to_csv(self.output_file)

        if self.dist_file is not None:
            df = pd.DataFrame.from_dict(self.dist_results)
            df.to_csv(self.dist_file, index=False)
