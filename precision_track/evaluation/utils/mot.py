from typing import Optional

import os
import numpy as np
import pandas as pd

from precision_track.apis.evaluator import Evaluator
from precision_track.outputs.display import display_mot_results, display_progress_bar
from precision_track.utils import assert_mot_file_is_ok


def evaluate_mot(
    result_path: str,
    ground_truth_path: str,
    metadata_path: str,
    save_path: Optional[str] = None,
    verbose: Optional[bool] = True,
    report_every_prcnt: Optional[float] = 1.0,
) -> dict:
    results = pd.read_csv(result_path)
    results = results.values

    gt = pd.read_csv(ground_truth_path)
    is_ok = assert_mot_file_is_ok(gt, ground_truth_path)
    assert is_ok, f"The {os.path.abspath(ground_truth_path)} file is not correctly formatted. Expected the frame_id, class_id, instances_id, x, y, w, h"
    gt = gt.values
    evaluator = Evaluator(metafile=metadata_path, save_path=save_path)
    unique_frames = np.unique(gt[:, 0])
    max_frame = np.max(unique_frames)
    evaluations = []
    report_every_prcnt = float(report_every_prcnt)
    if report_every_prcnt <= 0:
        report_every_prcnt = 0.1
    if report_every_prcnt > 1.0:
        report_every_prcnt = 1.0
    num_checkpoints = int(1.0 / report_every_prcnt)
    if report_every_prcnt < 1.0:
        reporting_idx = set(np.linspace(0, len(unique_frames) - 1, num=num_checkpoints, dtype=int))
    else:
        reporting_idx = {len(unique_frames) - 1}
    reporting_prcnt = np.linspace(report_every_prcnt, 1.0, num=num_checkpoints)
    prcnt_iter = iter(reporting_prcnt)
    for i, frame in enumerate(unique_frames):
        if verbose:
            display_progress_bar(frame, max_frame)
        frame_gt = gt[gt[:, 0] == frame][:, 1:].astype(int)
        frame_results = results[results[:, 0] == frame][:, 1:].astype(int)
        evaluator.update(frame_results, frame_gt)
        if i in reporting_idx:
            prcnt = next(prcnt_iter)
            evaluations.append(evaluator.evaluate() | dict(completion_prcnt=round(float(prcnt), 4)))
    if verbose:
        display_mot_results(evaluations)
    return evaluations


def evaluate_identity_purity(
    result_path: str,
    ground_truth_path: str,
    metadata_path: str,
    save_path: Optional[str] = None,
    verbose: Optional[bool] = True,
    report_every_prcnt: Optional[float] = 1.0,
) -> dict:
    results = pd.read_csv(result_path)
    results = results.values

    gt = pd.read_csv(ground_truth_path)
    is_ok = assert_mot_file_is_ok(gt, ground_truth_path)
    assert is_ok, f"The {os.path.abspath(ground_truth_path)} file is not correctly formatted. Expected the frame_id, class_id, instances_id, x, y, w, h"
    gt = gt.values
    evaluator = Evaluator(metafile=metadata_path, save_path=save_path)
    unique_frames = np.unique(gt[:, 0])
    max_frame = np.max(unique_frames)
    evaluations = []
    report_every_prcnt = float(report_every_prcnt)
    if report_every_prcnt <= 0:
        report_every_prcnt = 0.1
    if report_every_prcnt > 1.0:
        report_every_prcnt = 1.0
    num_checkpoints = int(1.0 / report_every_prcnt)
    if report_every_prcnt < 1.0:
        reporting_idx = set(np.linspace(0, len(unique_frames) - 1, num=num_checkpoints, dtype=int))
    else:
        reporting_idx = {len(unique_frames) - 1}
    reporting_prcnt = np.linspace(report_every_prcnt, 1.0, num=num_checkpoints)
    prcnt_iter = iter(reporting_prcnt)
    for i, frame in enumerate(unique_frames):
        if verbose:
            display_progress_bar(frame, max_frame)
        frame_gt = gt[gt[:, 0] == frame][:, 1:].astype(int)
        frame_results = results[results[:, 0] == frame][:, 1:].astype(int)
        evaluator.update(frame_results, frame_gt)
        if i in reporting_idx:
            prcnt = next(prcnt_iter)
            evaluations.append(evaluator.identity_purity() | dict(completion_prcnt=round(float(prcnt), 4)))
    return evaluations
