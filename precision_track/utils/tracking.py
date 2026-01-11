from collections import deque
from precision_track.outputs.display import display_progress_bar


def batch_tracking(video, detector, batch_size, result, association_step, validator=None, analyzer=None, verbose=True):
    b_frames = []
    b_idx = []
    outputs = deque()
    frames = deque()
    frame_id = 0
    empty = False
    switches = None
    total_frames = len(video)
    fps = video.fps
    while True:
        frame = video.read()
        if len(b_frames) == batch_size or (empty and b_frames):
            for output in detector(inputs=b_frames, data_samples=b_idx):
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
            output = association_step(output, switches)
            frame = frames.pop()
            if validator is not None:
                if validator._frame_size is None:
                    validator.frame_size = frame.shape[:2]
                output, switches = validator(frame, output)
            if analyzer is not None:
                output = analyzer.predict(output)
            output["fps"] = fps
            result(output)
        elif empty and not b_frames:
            break
    return result
