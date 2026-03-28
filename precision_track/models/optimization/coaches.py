import abc
import os
import random
from collections import defaultdict
from logging import WARNING
from typing import Optional

import numpy as np
from mmengine.logging import print_log

from precision_track.registry import COACHES
from precision_track.utils import parse_pose_metainfo


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
    def __init__(self, metainfo: str, block_size: int, ignore_idx: Optional[int] = -100, verbose: Optional[bool] = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = int(block_size)
        self.action_to_blocks = defaultdict(list)  # [[seq, start, end, (class_id, inst_id)]]
        metafile = parse_pose_metainfo(dict(from_file=metainfo))
        assert "actions" in metafile, f"Your {metainfo} meta-info file must contains a list of actions if you want to use the ActionRecognitionCoach."
        self.actions = metafile["actions"]
        self.np_actions = np.array(self.actions)
        self.sequences_map = dict()
        self._rng = random.Random()
        self._ignore_idx = ignore_idx
        self.verbose = verbose

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
                self._end_action_block(prev_action, seq, start, end, key)
                del open_blocks[key]

            if key not in open_blocks:
                open_blocks[key] = [action, frame_id, frame_id]
            else:
                prev_action, start, end = open_blocks[key]
                if action != prev_action:
                    self._end_action_block(prev_action, seq, start, end, key)
                    open_blocks[key] = [action, frame_id, frame_id]
                else:
                    open_blocks[key][2] = frame_id

            last_frame_by_key[key] = frame_id

        for key, (prev_action, start, end) in open_blocks.items():
            self._end_action_block(prev_action, seq, start, end, key)

        actions = [a for a, blocks in self.action_to_blocks.items() if blocks]
        for action in actions:
            assert action in self.actions, f"The {action} action (from the labels) is not registered in the meta-info file's action list: {self.actions}."

    def _end_action_block(self, action, seq, start, end, key):
        # First, handle the first self.block_size frame's edge case
        safe_start = max(start, self.block_size)
        if end > safe_start:
            self.action_to_blocks[action].append((seq, safe_start, end, key))

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

        return self.sequences_map[seq], seq_start_idx, dict(subject_id=subject_id, selected_action=action)

    def select_idx_labels(self, labels):
        cat, id_ = labels["subject_id"]
        id_mask = (labels["category_id"] == cat).astype(bool) & (labels["instance_id"] == id_).astype(bool)

        labels["category_id"] = labels["category_id"][id_mask]
        labels["instance_id"] = labels["instance_id"][id_mask]
        labels["bbox"] = labels["bbox"][id_mask]
        labels["bbox_score"] = labels["bbox_score"][id_mask]

        nb_instances = labels["bbox"].shape[0]
        if nb_instances > 1 and self.verbose:
            seq_dir = labels["sequence_dir"]
            abs_frame_id = labels["absolute_frame_id"]
            print_log(
                f"Images's '{seq_dir}' labels contains de following duplication: frame id: {abs_frame_id}, class id: {cat}, instance id: {id_}.",
                logger="current",
                level=WARNING,
            )

        if id_mask.any():
            action_label = labels["selected_action"]
            labels["action_label"] = np.repeat(np.array([action_label]), nb_instances, axis=0)
            labels["action"] = np.repeat(np.where(self.np_actions == action_label)[0], nb_instances, axis=0)
        else:
            labels["action_label"] = np.array([])
            labels["action"] = np.array([])

        if "keypoints" in labels:
            labels["keypoints"] = labels["keypoints"][id_mask]
        if "keypoints_visible" in labels:
            labels["keypoints_visible"] = labels["keypoints_visible"][id_mask]

        return labels
