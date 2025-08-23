import os
from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from mmengine import Config

from precision_track import Tracker
from precision_track.datasets import VideoDataset
from precision_track.evaluation.metrics import CLEARMetrics, SearchZoneStitchingMetric
from precision_track.outputs.display import display_progress_bar
from precision_track.utils import VideoReader, load_hyperparameter_dict


class GridSearchBase(ABC):
    """Template method for tracker-centric grid searches."""

    def __init__(
        self,
        tracking_config: Config,
        video_paths: Union[List[str], str],
        gt_paths: Union[List[str], str],
        metadata_path: str,
        save_path: Optional[str] = None,
    ) -> None:
        self.tracking_config = tracking_config
        self.dataset = VideoDataset(video_paths, gt_paths)
        self.metadata_path = metadata_path
        self.save_path = save_path

    @abstractmethod
    def _build_evaluator(self): ...

    @abstractmethod
    def _param_grid(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def _patch_hyperparams(self, params: Dict[str, Any]) -> None: ...

    @abstractmethod
    def _tracker_outputs(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def _process_evaluator(self, evaluator, gt_path: str) -> None: ...

    @abstractmethod
    def _get_metrics(self, evaluator, params: Dict[str, Any]) -> dict: ...

    def score_fn(self, mota: float, idf1: float) -> float:
        return 0.5 * mota + 0.5 * idf1

    def __call__(self) -> pd.DataFrame:
        grid = self._param_grid()
        total = len(grid)
        results: List[Dict[str, Any]] = []

        for idx, params in enumerate(grid, start=1):
            evaluator = self._build_evaluator()

            for sample in self.dataset:
                video = VideoReader(sample["inputs"])
                gt_path = sample["data_samples"]

                self._patch_hyperparams(params)

                tracker = Tracker(
                    detector=self.tracking_config.get("detector"),
                    assigner=self.tracking_config.get("assigner"),
                    validator=self.tracking_config.get("validator"),
                    analyzer=None,
                    outputs=self._tracker_outputs(),
                    batch_size=self.tracking_config.get("batch_size"),
                    verbose=False,
                )
                tracker(video=video)

                self._process_evaluator(evaluator, gt_path)

            results.append(self._get_metrics(evaluator, params))

            display_progress_bar(idx, total)

        df = pd.DataFrame(results).sort_values(by=["score"], ascending=False).reset_index(drop=True).fillna(0)
        if self.save_path:
            df.to_csv(self.save_path, index=False)
        return df


class ThresholdsGridSearch(GridSearchBase):
    """Search (low, high, init) detection thresholds."""

    def __init__(
        self,
        tracking_config: Config,
        video_paths: Union[List[str], str],
        gt_paths: Union[List[str], str],
        metadata_path: str,
        output_path: str,
        low_thr_range: List[float] = (0.01, 0.05, 0.1),
        high_thr_range: List[float] = (0.4, 0.45, 0.5, 0.55, 0.6),
        init_thr_range: List[float] = (0.65, 0.7, 0.75, 0.8),
        save_path: Optional[str] = None,
    ):
        super().__init__(tracking_config, video_paths, gt_paths, metadata_path, save_path)
        self.output_path = output_path
        self.low_thr, self.high_thr, self.init_thr = low_thr_range, high_thr_range, init_thr_range

    def _build_evaluator(self):
        return CLEARMetrics(metainfo=self.metadata_path)

    def _param_grid(self):
        return [dict(low_thr=low, high_thr=h, init_thr=i) for low, h, i in product(self.low_thr, self.high_thr, self.init_thr)]

    def _patch_hyperparams(self, p):
        updated = load_hyperparameter_dict(
            self.tracking_config.load_from,
            "tracking_thresholds",
            dict(low_thr=p["low_thr"], conf_thr=p["high_thr"], init_thr=p["init_thr"]),
        )
        assert os.path.samefile(updated, self.tracking_config.hyperparams), (
            f"Please ensure that the actual hyperparameters file ({self.tracking_config.hyperparams}), defined by 'hypeparams' in the tracking config, "
            f"used for tracking is the same than the one automatically updated by the search algorithm: {updated}."
        )

    def _tracker_outputs(self):
        return [
            dict(
                type="CsvBoundingBoxes",
                path=self.output_path,
                instance_data="pred_track_instances",
                precision=64,
            )
        ]

    def _process_evaluator(self, evaluator, gt_path: str):
        evaluator.process(data_batch=[self.output_path], data_samples=[gt_path])

    def _get_metrics(self, evaluator, params: Dict[str, Any]):
        metrics = evaluator.evaluate(len(self.dataset))
        mota = metrics["CLEAR/Overall/mota"]
        idf1 = metrics["CLEAR/Overall/idf1"]

        return {**params, "mota": mota, "idf1": idf1, "score": self.score_fn(mota, idf1)}


class StitchingHyperparamsGridSearch(GridSearchBase):
    """Search stitching (beta, match_thr, eps) hyper-params."""

    def __init__(
        self,
        tracking_config: Config,
        video_paths: Union[List[str], str],
        gt_paths: Union[List[str], str],
        metadata_path: str,
        bboxes_path: str,
        search_zones_path: str,
        beta_range: List[float] = (0.25, 0.5, 1.0, 1.5),
        match_thr_range: List[float] = (0.8, 0.9),
        eps_range: List[float] = (1e-2, 1e-1),
        save_path: Optional[str] = None,
    ):
        super().__init__(tracking_config, video_paths, gt_paths, metadata_path, save_path)
        self.bboxes_path = bboxes_path
        self.zones_path = search_zones_path
        self.beta, self.match_thr, self.eps = beta_range, match_thr_range, eps_range

    def _build_evaluator(self):
        return [SearchZoneStitchingMetric(metainfo=self.metadata_path), CLEARMetrics(metainfo=self.metadata_path)]

    def _param_grid(self):
        return [dict(beta=b, match_thr=m, eps=e) for b, m, e in product(self.beta, self.match_thr, self.eps)]

    def _patch_hyperparams(self, p):
        updated = load_hyperparameter_dict(
            self.tracking_config.load_from,
            "stitching_hyperparams",
            dict(beta=p["beta"], match_thr=p["match_thr"], eps=p["eps"]),
        )
        assert os.path.samefile(updated, self.tracking_config.hyperparams), (
            f"Please ensure that the actual hyperparameters file ({self.tracking_config.hyperparams}), defined by 'hypeparams' in the tracking config,"
            f"used for tracking is the same than the one automatically updated by the search algorithm: {updated}."
        )

    def _tracker_outputs(self):
        return [
            dict(
                type="CsvBoundingBoxes",
                path=self.bboxes_path,
                instance_data="pred_track_instances",
                precision=64,
            ),
            dict(
                type="CsvSearchAreas",
                path=self.zones_path,
                instance_data="search_areas",
                precision=64,
            ),
        ]

    def _process_evaluator(self, evaluator, gt_path: str):
        evaluator[0].process(data_batch=[self.bboxes_path, self.zones_path], data_samples=[gt_path], reset_map=True)
        evaluator[1].process(data_batch=[self.bboxes_path], data_samples=[gt_path])

    def _get_metrics(self, evaluator, params: Dict[str, Any]):
        stitching_metrics = evaluator[0].compute_metrics(len(self.dataset))
        mot_metrics = evaluator[1].evaluate(len(self.dataset))
        mota = mot_metrics["CLEAR/Overall/mota"]
        idf1 = mot_metrics["CLEAR/Overall/idf1"]
        return {**params, **stitching_metrics, "mota": mota, "idf1": idf1, "score": self.score_fn(mota, idf1)}
