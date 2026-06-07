import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from mmengine.logging import print_log

__all__ = ["refine_corrections_offline"]


# Output ``type`` keys (see configs/tasks/tracking.py) used to locate the relevant files.
# Filenames are user-configurable, so everything is resolved from the outputs config.
_CORRECTIONS_TYPE = "CsvCorrections"
_BBOXES_TYPE = "CsvBoundingBoxes"
_APPEARANCE_TYPE = "CsvAppearanceValidations"

# The ``instance_data`` that distinguishes the several CsvBoundingBoxes outputs.
_TRACKED_BBOXES_DATA = "pred_track_instances"
_DETECTED_BBOXES_DATA = "pred_instances"


def _find_output(outputs: list, type_: str, **match) -> Optional[dict]:
    for out in outputs:
        if out.get("type") != type_:
            continue
        if all(out.get(k) == v for k, v in match.items()):
            return out
    return None


def _build_presence(bboxes: pd.DataFrame) -> Dict[Tuple[int, int], np.ndarray]:
    """Map every (class_id, instance_id) to its sorted array of present frames."""
    presence = {}
    for (cls, iid), grp in bboxes.groupby(["class_id", "instance_id"], sort=False):
        presence[(int(cls), int(iid))] = np.sort(grp["frame_id"].to_numpy())
    return presence


def _build_confirmations(
    validations: pd.DataFrame, identities: set
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """Map every (class_id, instance_id) to its sorted confirmation frames and identities.

    A confirmation is a row whose ``identity`` is a real identity name (i.e. in the
    validator's ``identities`` list); this excludes the empty string, the ``"?"`` written
    at correction frames, and NaN.
    """
    validations = validations.copy()
    validations["identity"] = validations["identity"].fillna("").astype(str)
    validations = validations[validations["identity"].isin(identities)]

    confirmations = {}
    for (cls, iid), grp in validations.groupby(["class_id", "instance_id"], sort=False):
        grp = grp.sort_values("frame_id")
        confirmations[(int(cls), int(iid))] = (
            grp["frame_id"].to_numpy(),
            grp["identity"].to_numpy(),
        )
    return confirmations


def _last_confirmation(
    confirmations: dict, cls: int, iid: int, frame: int
) -> Tuple[Optional[int], Optional[str]]:
    """Most recent confirmation strictly before ``frame`` for (cls, iid)."""
    entry = confirmations.get((cls, iid))
    if entry is None:
        return None, None
    frames, idents = entry
    pos = int(np.searchsorted(frames, frame, side="left"))
    if pos == 0:
        return None, None
    return int(frames[pos - 1]), str(idents[pos - 1])


def _gap_before(
    presence: dict, cls: int, iid: int, frame: int
) -> Optional[Tuple[int, int]]:
    """Most recent disappearance of (cls, iid) relative to the correction ``frame``.

    Isolates the contiguous present-run ending at ``frame - 1`` and returns
    ``(missing_frame, reappearance_frame)`` where ``missing_frame = reappearance - 1``.
    Returns ``None`` when the id is not present at ``frame - 1`` or never disappeared
    (the run extends back to the id's first appearance, i.e. a birth, not a switch).
    """
    p = presence.get((cls, iid))
    if p is None:
        return None
    anchor = frame - 1
    le = p[p <= anchor]
    if le.size == 0 or le[-1] != anchor:
        return None
    gaps = np.where(np.diff(le) != 1)[0]
    if gaps.size == 0:
        return None
    reappearance = int(le[gaps[-1] + 1])
    return reappearance - 1, reappearance


def _compute_swaps(
    corrections: pd.DataFrame,
    presence: dict,
    confirmations: dict,
    disabled_identities: set,
) -> List[Tuple[int, int, int, int, int]]:
    """Resolve each correction into a swap window ``(cls, A, B, switch, end)``.

    Implements the backward walk: from the correction frame, the first event reached
    decides -- a disappearance triggers propagation back to the reappearance frame, a
    confirmation cancels it.
    """
    swaps = []
    last_corr_frame_for_pair = {}

    corrections = corrections.sort_values("frame_id")
    for frame, cls, a, b in zip(
        corrections["frame_id"].astype(int),
        corrections["class_id"].astype(int),
        corrections["instance_id"].astype(int),
        corrections["corrected_id"].astype(int),
    ):
        frame, cls, a, b = int(frame), int(cls), int(a), int(b)

        c_a, ident_a = _last_confirmation(confirmations, cls, a, frame)
        c_b, ident_b = _last_confirmation(confirmations, cls, b, frame)
        enabled_a = c_a is not None and ident_a not in disabled_identities
        enabled_b = c_b is not None and ident_b not in disabled_identities
        if not (enabled_a or enabled_b):
            continue

        candidates = []  # (missing_frame, reappearance_frame)
        conf_frames = []
        for enabled, iid, c_x in ((enabled_a, a, c_a), (enabled_b, b, c_b)):
            if not enabled:
                continue
            conf_frames.append(c_x)
            gap = _gap_before(presence, cls, iid, frame)
            if gap is not None and gap[1] > c_x:
                candidates.append(gap)
        if not candidates:
            continue

        missing = max(m for m, _ in candidates)
        # A confirmation more recent than the chosen switch means we would have hit it
        # first on the way back: do nothing.
        if missing <= max(conf_frames):
            continue

        switch = missing + 1
        end = frame - 1

        # Safeguard against overlapping windows for a repeated (cls, {A, B}) pair.
        key = (cls, frozenset((a, b)))
        prev = last_corr_frame_for_pair.get(key)
        last_corr_frame_for_pair[key] = frame
        if prev is not None and switch <= prev:
            switch = prev + 1

        if switch > end:
            continue
        swaps.append((cls, a, b, switch, end))

    return swaps


def _apply_swaps(path: str, swaps: List[Tuple[int, int, int, int, int]]) -> bool:
    """Apply every swap window to a single CSV file in place. Returns True if it changed."""
    df = pd.read_csv(path)
    if "instance_id" not in df.columns:
        return False

    frame_ids = df["frame_id"].to_numpy()
    class_ids = df["class_id"].to_numpy()

    changed = False
    for cls, a, b, switch, end in swaps:
        window = (frame_ids >= switch) & (frame_ids <= end) & (class_ids == cls)
        instance_ids = df["instance_id"].to_numpy()
        is_a = window & (instance_ids == a)
        is_b = window & (instance_ids == b)
        if is_a.any() or is_b.any():
            df.loc[is_a, "instance_id"] = b
            df.loc[is_b, "instance_id"] = a
            changed = True

    if changed:
        df.to_csv(path, index=False)
    return changed


def refine_corrections_offline(outputs: list, validator) -> None:
    """Retro-actively close the gap between an ID switch and its online correction.

    For every correction recorded by the ``CsvCorrections`` output this walks back through
    the tracked bounding boxes (presence) and the appearance validations (confirmations)
    to find the most probable switch frame, then propagates the ID swap across that gap in
    every per-frame output file that carries an ``instance_id`` (overwriting them in place).

    File paths are resolved from the tracking ``outputs`` config via their ``type`` key so
    that user-renamed outputs are still handled correctly.

    Args:
        outputs: The tracking ``outputs`` config (list of output dicts, each with ``type``
            and ``path``).
        validator: The initialised appearance validator instance, used for its
            ``identities`` and ``disabled_identities`` attributes.
    """
    corrections_out = _find_output(outputs, _CORRECTIONS_TYPE)
    tracked_out = _find_output(outputs, _BBOXES_TYPE, instance_data=_TRACKED_BBOXES_DATA)
    appearance_out = _find_output(outputs, _APPEARANCE_TYPE)
    detected_out = _find_output(outputs, _BBOXES_TYPE, instance_data=_DETECTED_BBOXES_DATA)

    if corrections_out is None or tracked_out is None or appearance_out is None:
        print_log(
            "Offline correction refinement skipped: the corrections, tracked bounding boxes "
            "or appearance validations output is not configured.",
            logger="current",
        )
        return

    corrections_path = corrections_out["path"]
    if not os.path.exists(corrections_path):
        print_log(
            f"Offline correction refinement skipped: '{corrections_path}' not found.",
            logger="current",
        )
        return

    corrections = pd.read_csv(corrections_path)
    if corrections.empty:
        print_log("Offline correction refinement skipped: no corrections to refine.", logger="current")
        return

    bboxes_path = tracked_out["path"]
    validations_path = appearance_out["path"]
    assert os.path.exists(bboxes_path), f"Offline correction refinement requires '{bboxes_path}'."
    assert os.path.exists(validations_path), f"Offline correction refinement requires '{validations_path}'."

    identities = set(validator.identities)
    disabled_identities = set(validator.disabled_identities)

    presence = _build_presence(pd.read_csv(bboxes_path))
    confirmations = _build_confirmations(pd.read_csv(validations_path), identities)

    swaps = _compute_swaps(corrections, presence, confirmations, disabled_identities)
    if not swaps:
        print_log(
            "Offline correction refinement: no correction could be propagated back in time.",
            logger="current",
        )
        return

    # Refine every CSV output carrying an instance_id, except the corrections record itself
    # and the detected (untracked) bounding boxes. Non-CSV outputs and id-less files (e.g.
    # timestamps) are filtered out by the extension check and the instance_id guard.
    excluded_paths = {corrections_path}
    if detected_out is not None:
        excluded_paths.add(detected_out["path"])

    refined = []
    for out in outputs:
        path = str(out.get("path", ""))
        if not path.endswith(".csv") or path in excluded_paths:
            continue
        if os.path.exists(path) and _apply_swaps(path, swaps):
            refined.append(os.path.basename(path))

    print_log(
        f"Offline correction refinement: propagated {len(swaps)} correction(s) and refined "
        f"{len(refined)} file(s): {refined}.",
        logger="current",
    )
