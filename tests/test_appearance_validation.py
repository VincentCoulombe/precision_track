import json
import os

import numpy as np
import pandas as pd
import pytest
from mmengine import Config

from precision_track import Tracker
from precision_track.utils import VideoReader, load_validation_config

ROOT = os.path.join(os.getcwd(), "tests")


@pytest.fixture
def config():
    return os.path.join(ROOT, "configs/tracking_w_reid.py")


@pytest.fixture
def expected_results_path():
    return os.path.join(ROOT, "work_dir/gt_corrections.csv")


@pytest.fixture
def video():
    return VideoReader(os.path.join(ROOT, "../assets/striped_mice_sanity_check.mp4"))


def test_appearance_validation(expected_results_path, config, video):
    config = Config.fromfile(config)
    load_validation_config(config)

    r_path = "./tests/work_dir/tracked_corrections.csv"
    config["outputs"] = [
        dict(
            type="CsvCorrections",
            path=r_path,
            instance_data="correction_instances",
            precision=32,
        ),
    ]
    try:
        tracker = Tracker(
            detector=config.get("detector"),
            assigner=config.get("assigner"),
            validator=config.get("validator"),
            analyzer=config.get("analyzer"),
            outputs=config.get("outputs"),
            batch_size=config.get("batch_size"),
            verbose=True,
        )
        tracker(video=video)

        expected_results = pd.read_csv(expected_results_path)
        print(expected_results)
        results = pd.read_csv(r_path)
        print(results)

        assert len(results) == len(expected_results), f"{len(results)} != {len(expected_results)}"
        assert np.all(results.values[:, 0] <= expected_results.values[:, 0])

    finally:
        if os.path.exists(r_path):
            os.remove(r_path)
