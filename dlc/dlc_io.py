"""Reading and writing DeepLabCut project artifacts.

Two directions are covered:

* **Writing** the annotation half of a multi-animal DLC project
  (``labeled-data/<folder>/CollectedData_<scorer>.h5``/``.csv`` and the multi-animal
  fields of ``config.yaml``). ``sleap-io`` can *read* DLC projects but has no DLC
  writer, so this is implemented here.
* **Reading** the tracked ``.h5`` that ``deeplabcut.analyze_videos`` produces, which
  ``dlc/track.py`` converts into PrecisionTrack CSVs.
"""

import glob
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TRACKED_H5_SUFFIXES = ("_el.h5", "_bx.h5", "_sk.h5")


def individual_names(n_individuals: int) -> List[str]:
    """DLC's own naming scheme for animals: ``individual1 ... individualN``."""
    return [f"individual{i + 1}" for i in range(n_individuals)]


def build_collected_data(
    frames: Sequence[Tuple[str, np.ndarray]],
    individuals: List[str],
    bodyparts: List[str],
    scorer: str,
    folder_name: str,
) -> pd.DataFrame:
    """Assemble a multi-animal ``CollectedData`` dataframe.

    Args:
        frames: ``(image_filename, keypoints)`` pairs, where ``keypoints`` is a
            ``(n_instances, n_bodyparts, 2)`` array using NaN for unlabelled points.
            Instances are assigned to ``individuals`` in the order they appear.
        individuals: The project's animal identities.
        bodyparts: The project's ``multianimalbodyparts``, in order.
        scorer: The annotator name (top column level).
        folder_name: The ``labeled-data/<folder_name>`` directory holding the images.

    Returns:
        A dataframe indexed by ``("labeled-data", folder_name, image_filename)`` with
        ``(scorer, individual, bodypart, coord)`` columns.
    """
    columns = pd.MultiIndex.from_product(
        [[scorer], individuals, bodyparts, ["x", "y"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    n_cols = len(individuals) * len(bodyparts) * 2

    index, rows = [], []
    for filename, keypoints in frames:
        row = np.full(n_cols, np.nan, dtype=np.float64)
        n_instances = min(len(keypoints), len(individuals))
        if n_instances:
            stride = len(bodyparts) * 2
            flat = np.asarray(keypoints[:n_instances], dtype=np.float64).reshape(n_instances, stride)
            row[: n_instances * stride] = flat.reshape(-1)
        index.append(("labeled-data", folder_name, filename))
        rows.append(row)

    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index), columns=columns)


def write_collected_data(df: pd.DataFrame, dest_dir: str, scorer: str) -> Tuple[str, str]:
    """Write ``CollectedData_<scorer>.h5`` and ``.csv`` into ``dest_dir``."""
    os.makedirs(dest_dir, exist_ok=True)
    h5_path = osp.join(dest_dir, f"CollectedData_{scorer}.h5")
    csv_path = osp.join(dest_dir, f"CollectedData_{scorer}.csv")
    df.to_hdf(h5_path, key="df_with_missing", mode="w")
    df.to_csv(csv_path)
    return h5_path, csv_path


def multianimal_config_updates(
    individuals: List[str],
    bodyparts: List[str],
    skeleton: List[List[str]],
    identity: bool = False,
) -> Dict:
    """The ``config.yaml`` fields that make a DLC project multi-animal.

    ``bodyparts: MULTI!`` is DLC's sentinel telling it to read ``multianimalbodyparts``
    instead of the single-animal ``bodyparts`` list.
    """
    return {
        "individuals": list(individuals),
        "multianimalbodyparts": list(bodyparts),
        "uniquebodyparts": [],
        "bodyparts": "MULTI!",
        "skeleton": [list(edge) for edge in skeleton],
        "identity": bool(identity),
    }


def find_tracked_h5(video_path: str, dest_folder: str) -> str:
    """Locate the tracked ``.h5`` DLC wrote for ``video_path``.

    ``analyze_videos(..., auto_track=True)`` names its output
    ``<video stem>DLC_<net>_<task><date>shuffle<k>_<snapshot>_<tracker>.h5``, where the
    tracker suffix is ``el`` (ellipse, the default), ``bx`` or ``sk``. The unstitched
    detections file (no suffix) is ignored.
    """
    stem = osp.splitext(osp.basename(video_path))[0]
    candidates = []
    for suffix in TRACKED_H5_SUFFIXES:
        candidates.extend(glob.glob(osp.join(dest_folder, f"{stem}DLC*{suffix}")))
    if not candidates:
        available = sorted(osp.basename(p) for p in glob.glob(osp.join(dest_folder, f"{stem}DLC*.h5")))
        raise FileNotFoundError(
            f"No tracked DLC output ({'/'.join(TRACKED_H5_SUFFIXES)}) found for '{stem}' in '{dest_folder}'. "
            f"Files present: {available or 'none'}. Did analyze_videos run with auto_track=True?"
        )
    return max(candidates, key=osp.getmtime)


def read_tracked_h5(h5_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Read a tracked DLC ``.h5`` into ``(dataframe, individuals, bodyparts)``.

    The returned dataframe is indexed by frame number with
    ``(individuals, bodyparts, coords)`` columns, ``coords`` being ``x``/``y``/
    ``likelihood``. Single-animal outputs are given a synthetic ``individual1`` level so
    downstream code has a single shape to handle.
    """
    df = pd.read_hdf(h5_path)
    if "scorer" in (df.columns.names or []):
        df = df.droplevel("scorer", axis=1)
    if "individuals" not in (df.columns.names or []):
        df = pd.concat({"individual1": df}, axis=1, names=["individuals"])

    individuals = list(df.columns.get_level_values("individuals").unique())
    bodyparts = list(df.columns.get_level_values("bodyparts").unique())
    return df, individuals, bodyparts


def tracked_frames(
    df: pd.DataFrame,
    individuals: List[str],
    bodyparts: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorise a tracked dataframe into ``(frame_ids, poses)``.

    ``poses`` has shape ``(n_frames, n_individuals, n_bodyparts, 3)`` holding
    ``x, y, likelihood``, with NaN wherever an individual is absent from a frame.
    """
    ordered = df.reindex(
        columns=pd.MultiIndex.from_product(
            [individuals, bodyparts, ["x", "y", "likelihood"]],
            names=["individuals", "bodyparts", "coords"],
        )
    )
    poses = ordered.to_numpy(dtype=np.float64).reshape(len(ordered), len(individuals), len(bodyparts), 3)
    return np.asarray(ordered.index, dtype=np.int64), poses


def read_project_config(config_path: str) -> Dict:
    """Read a DLC ``config.yaml`` without importing deeplabcut."""
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def project_dest_folder(video_path: str, destfolder: Optional[str]) -> str:
    """Where DLC writes a video's analysis results (its directory unless overridden)."""
    return destfolder if destfolder else osp.dirname(osp.abspath(video_path))
