from typing import Optional

import numpy as np
import pandas as pd
from mmengine.config import Config

from precision_track.apis.evaluator import Evaluator
from precision_track.outputs.display import display_mot_results, display_progress_bar


def evaluate_mot(
    result_path: str,
    ground_truth_path: str,
    metadata_path: str,
    save_path: Optional[str] = None,
    verbose: Optional[bool] = True,
) -> dict:
    results = pd.read_csv(result_path)
    results = results.values
    gt = pd.read_csv(ground_truth_path).values
    evaluator = Evaluator(metafile=metadata_path, save_path=save_path)
    unique_frames = np.unique(gt[:, 0])
    max_frame = np.max(unique_frames)
    for frame in unique_frames:
        if verbose:
            display_progress_bar(frame, max_frame)
        frame_gt = gt[gt[:, 0] == frame][:, 1:].astype(int)
        frame_results = results[results[:, 0] == frame][:, 1:].astype(int)
        evaluator.update(frame_results, frame_gt)
    ev = evaluator.evaluate()
    if verbose:
        display_mot_results(ev)
    return ev
