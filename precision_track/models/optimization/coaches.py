import abc
import random
import numpy as np
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from precision_track.registry import COACHES


class BaseCoach(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def load_info(self, info: dict) -> None:
        """_summary_

        Args:
            info (dict): _description_
        """

    @abc.abstractmethod
    def select_idx_labels(self, labels: dict) -> dict:
        """_summary_

        Args:
            labels (dict): _description_

        Returns:
            dict: _description_
        """

    @abc.abstractmethod
    def get_idx(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """


@COACHES.register_module()
class ActionRecognitionCoach(BaseCoach):
    def __init__(self, block_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = int(block_size)
        self.action_to_blocks = defaultdict(list)  # [[seq, start, end, (class_id, inst_id)]]
        self.actions = []
        self.sequences_map = dict()
        self._rng = random.Random()

    def set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng.seed(seed)

    def load_info(self, info):
        assert "actions_output" in info, "ActionRecognitionCoach needs 'actions_output'."
        seq = os.path.basename(info.get("sequence_dir", "")) or info.get("sequence_dir", "")

        rows = info["actions_output"].results
        rows = sorted(rows, key=lambda r: (r[1], r[2], r[0]))

        seq_idx = info["sequence_idx"]
        self.sequences_map[seq] = seq_idx

        open_blocks = {}  # (class_id, inst_id) -> [action, start, end]
        last_frame_by_key = {}

        for row in rows:
            frame_id, class_id, inst_id, action = row[0], row[1], row[2], row[3]
            key = (class_id, inst_id)

            if key in last_frame_by_key and key in open_blocks and frame_id != last_frame_by_key[key] + 1:
                prev_action, start, end = open_blocks[key]
                self.action_to_blocks[prev_action].append((seq, start, end, key))
                del open_blocks[key]

            if key not in open_blocks:
                open_blocks[key] = [action, frame_id, frame_id]
            else:
                prev_action, start, end = open_blocks[key]
                if action != prev_action:
                    self.action_to_blocks[prev_action].append((seq, start, end, key))
                    open_blocks[key] = [action, frame_id, frame_id]
                else:
                    open_blocks[key][2] = frame_id

            last_frame_by_key[key] = frame_id

        for key, (prev_action, start, end) in open_blocks.items():
            self.action_to_blocks[prev_action].append((seq, start, end, key))

        self.actions = [a for a, blocks in self.action_to_blocks.items() if blocks]

    def get_idx(self) -> int:
        """Return a valid start index for a window of length block_size that ENDS with the chosen action."""
        if not self.actions:
            raise RuntimeError("No actions available to sample from.")

        action = self._rng.choice(self.actions)
        blocks = self.action_to_blocks[action]
        if not blocks:
            raise RuntimeError(f" {action} has no blocks available to sample from.")

        start = 0
        while start <= self.block_size:
            seq, start, end, subject_id = self._rng.choice(blocks)
        seq_end_idx = self._rng.randint(start, end)
        seq_start_idx = seq_end_idx - self.block_size + 1
        assert seq_start_idx >= 0

        return self.sequences_map[seq], seq_start_idx, dict(subject_id=subject_id)

    def select_idx_labels(self, labels):
        cat, id_ = labels["subject_id"]
        id_mask = (labels["category_id"] == cat).astype(bool) & (labels["instance_id"] == id_).astype(bool)

        labels["category_id"] = labels["category_id"][id_mask]
        labels["instance_id"] = labels["instance_id"][id_mask]
        labels["bbox"] = labels["bbox"][id_mask]
        labels["bbox_score"] = labels["bbox_score"][id_mask]

        labels["action_label"] = labels["action_label"][id_mask]
        labels["action"] = labels["action"][id_mask]

        if "keypoints" in labels:
            labels["keypoints"] = labels["keypoints"][id_mask]
        if "keypoints_visible" in labels:
            labels["keypoints_visible"] = labels["keypoints_visible"][id_mask]

        return labels
