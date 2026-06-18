import json
import os
from collections import deque
from time import perf_counter

from precision_track.outputs.display import display_progress_bar


def batch_tracking(video, detector, batch_size, result, association_step, validator=None, analyzer=None, verbose=True, profile=""):
    b_frames = []
    b_idx = []
    outputs = deque()
    frames = deque()
    frame_id = 0
    empty = False
    switches = None
    total_frames = len(video)
    fps = video.fps
    profile_dict = dict()
    profile_dir = os.path.dirname(profile)
    is_profiling = os.path.isdir(profile_dir)
    if is_profiling:
        os.makedirs(os.path.dirname(profile), exist_ok=True)
        profile_dict = dict(
            detection=[],
            saving_results=[],
        )
    while True:
        frame = video.read()
        if len(b_frames) == batch_size or (empty and b_frames):
            for output in detector(inputs=b_frames, data_samples=b_idx, profile=profile_dict.get("detection")):
                outputs.appendleft(output)
            b_frames, b_idx = [], []
        if frame is not None:
            b_frames.append(frame)
            b_idx.append(frame_id)
            frames.appendleft(frame)
            frame_id += 1
            if verbose:
                display_progress_bar(frame_id, total_frames)
        else:
            empty = True
        if outputs:
            output = outputs.pop()
            frame = frames.pop()
            output["img"] = frame
            output = association_step(output, switches, profile_dict if is_profiling else None)
            if validator is not None:
                if is_profiling:
                    validation_start = perf_counter()
                output, switches = validator(frame, output)
                if is_profiling:
                    profile_dict.setdefault("validation", []).append(perf_counter() - validation_start)
            if analyzer is not None:
                if is_profiling:
                    analysis_start = perf_counter()
                output = analyzer.predict(output)
                if is_profiling:
                    profile_dict.setdefault("analysis", []).append(perf_counter() - analysis_start)
            output["fps"] = fps
            result(output, profile_dict.get("saving_results"))
        elif empty and not b_frames:
            break
    if is_profiling:
        with open(profile, "w") as f:
            json.dump(profile_dict, f)
    return result
