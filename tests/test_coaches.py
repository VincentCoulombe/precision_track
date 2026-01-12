import pytest
from collections import defaultdict
from typing import Dict, Tuple, Iterable
import os

from precision_track.models.optimization import ActionRecognitionCoach
from precision_track.outputs.csv import CsvActions


def _frame_map(info: dict) -> Dict[Tuple[int, int], Dict[int, Tuple[str, str]]]:
    """
    Build: key -> {frame_id: (seq_name, action)}
    key = (class_id, instance_id)
    """
    seq = info.get("sequence_dir", "")
    fmap = defaultdict(dict)
    for r in info["actions_output"].results:
        frame_id, class_id, inst_id, action = r[0], r[1], r[2], r[3]
        key = (class_id, inst_id)
        fmap[key][int(frame_id)] = (seq, str(action))
    return fmap


def _observed_actions(info: dict) -> set:
    return {str(r[3]) for r in info["actions_output"].results}


def _all_blocks(coach: ActionRecognitionCoach) -> Iterable[Tuple[str, int, int, Tuple[int, int], str]]:
    """
    Yield tuples (seq, start, end, key, action) for every recorded block.
    """
    for action, blocks in coach.action_to_blocks.items():
        for seq, start, end, key in blocks:
            yield (seq, int(start), int(end), key, action)


@pytest.fixture
def info():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "work_dir", "actions.csv")
    actions = CsvActions(path)
    actions.read()
    return dict(sequence_dir=os.path.basename(path), actions_output=actions)


# def test_load_info(info):
#     """Ensures that each block in the action recognition coach's action_to_blocks contains, for each sequence + subject, the correct action."""
#     coach = ActionRecognitionCoach(block_size=3)
#     coach.set_seed(123)
#     coach.load_info(info)
#     fmap = _frame_map(info)

#     blocks_by_frame = {}
#     for blk_seq, start, end, key, action in _all_blocks(coach):
#         assert key in fmap, f"Key {key} not present in source data."
#         frames_for_key = fmap[key]
#         for f in range(start, end + 1):
#             assert f in frames_for_key, f"Missing frame {f} for key {key} in [{start},{end}]."
#             seq_f, action_f = frames_for_key[f]
#             assert seq_f == blk_seq, f"Seq mismatch at frame {f}: {seq_f} != {blk_seq}"
#             assert str(action_f) == str(action), f"Action mismatch at frame {f}: {action_f} != {action}"

#             k = (blk_seq, key, int(f))
#             prev = blocks_by_frame.get(k)
#             assert prev in (None, str(action)), f"Conflicting actions for {k}: {prev} vs {action}"
#             blocks_by_frame[k] = str(action)

#     fmap_by_frame = {}
#     seq_name = info.get("sequence_dir", "")
#     for key, frames in fmap.items():
#         for f, (seq_f, action_f) in frames.items():
#             k = (seq_f or seq_name, key, int(f))
#             fmap_by_frame[k] = str(action_f)

#     extra_in_blocks = set(blocks_by_frame.keys()) - set(fmap_by_frame.keys())
#     assert not extra_in_blocks, f"Blocks contain frames not in fmap: {sorted(list(extra_in_blocks))[:10]}"

#     missing_in_blocks = set(fmap_by_frame.keys()) - set(blocks_by_frame.keys())
#     assert not missing_in_blocks, f"Frames in fmap not covered by blocks: {sorted(list(missing_in_blocks))[:10]}"

#     for k in blocks_by_frame.keys():
#         assert blocks_by_frame[k] == fmap_by_frame[k], f"Action mismatch at {k}: blocks={blocks_by_frame[k]} fmap={fmap_by_frame[k]}"


# def test_actions_subset_of_observed(load_actions):
#     """
#     Sanity: the coach.actions set must be a subset of actions actually observed in the rows.
#     """
#     for info in load_actions():
#         coach = ActionRecognitionCoach(block_size=2)
#         coach.load_info(info)
#         observed = _observed_actions(info)
#         assert set(coach.actions).issubset(observed), "coach.actions includes unseen actions."


# def test_get_idx_reproducibility(load_actions):
#     """
#     With a fixed seed, repeated sampling should be reproducible across instances.
#     Uses the first scenario yielded by the fixture.
#     """
#     # Grab first provided info only
#     gen = load_actions()
#     try:
#         info = next(iter(gen))
#     except TypeError:
#         # If the fixture returns a list instead of a generator
#         info = load_actions()[0]

#     c1 = ActionRecognitionCoach(block_size=3)
#     c2 = ActionRecognitionCoach(block_size=3)
#     c1.set_seed(999)
#     c2.set_seed(999)
#     c1.load_info(info)
#     c2.load_info(info)

#     seq1 = [c1.get_idx() for _ in range(20)]
#     seq2 = [c2.get_idx() for _ in range(20)]
#     assert seq1 == seq2, "Sampling with the same seed should be identical."


# def test_get_idx_within_global_bounds(load_actions):
#     """
#     get_idx should always return a start index that is within the global numeric bounds
#     covered by any block recorded (not asserting per-block window fit, which is the caller's concern).
#     """
#     for info in load_actions():
#         coach = ActionRecognitionCoach(block_size=5)
#         coach.set_seed(0)
#         coach.load_info(info)

#         # Determine global min/max over all blocks that exist
#         starts = []
#         ends = []
#         for _, s, e, _, _ in _all_blocks(coach):
#             starts.append(s)
#             ends.append(e)

#         # If no blocks exist at all, the class would raise in get_idx; skip such dataset
#         if not starts:
#             pytest.skip("No blocks present in this dataset.")

#         gmin, gmax = min(starts), max(ends)

#         for _ in range(50):
#             idx = coach.get_idx()
#             assert gmin - coach.block_size <= idx <= gmax, f"Sampled idx {idx} out of plausible global bounds [{gmin - coach.block_size}, {gmax}]."


if __name__ == "__main__":
    pytest.main(["-x", os.path.realpath(__file__)])
