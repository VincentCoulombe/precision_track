"""Score every configuration a ``sweep_tracking.py`` run produced, ranked by IDF1.

**Run this in the ``precision_track`` environment, not the ``sleap`` one** -- it imports
``precision_track``. The two-environment split is the same one ``sleap/README.md`` §8
describes: SLEAP produces the CSVs, PrecisionTrack scores them.

Nothing here reimplements a metric. Each configuration goes through
``precision_track.evaluation.utils.mot.evaluate_mot`` unchanged, which also leaves a
per-config ``mot_evaluation.csv`` behind, so the winner's file is already in the exact
shape the report ships.

Every configuration attempted is listed in the summary, including the ones that made
things worse -- a sweep that reports only its winner hides the evidence that the winner
is real.

Example:
    python sweep_evaluate.py --sweep-dir ../work_dir/sleap_sweep/bottomup \\
        --gt ~/Documents/datasets/stripedmice/mot_v2/bboxes/val/<video>.csv \\
        --metainfo ../configs/metadata/stripedmice.py
"""

import argparse
import json
import os.path as osp
from typing import Dict, List, Optional

import pandas as pd

from precision_track.evaluation.utils.mot import evaluate_mot

RANK_COLUMNS = ["config", "peak_threshold", "idf1", "mota", "precision", "recall", "num_switches", "num_detections", "idp", "idr"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", required=True, help="Directory produced by sweep_tracking.py")
    parser.add_argument("--gt", required=True, help="Ground-truth MOT CSV (xywh) for the swept video")
    parser.add_argument("--metainfo", required=True, help="Path to the metadata .py, e.g. ../configs/metadata/stripedmice.py")
    parser.add_argument("--output", default=None, help="Where to write sweep_summary.csv. Defaults to <sweep-dir>/sweep_summary.csv.")
    parser.add_argument("--only", nargs="*", default=None, metavar="NAME", help="Score only these configuration directories.")
    return parser.parse_args()


def load_manifest(sweep_dir: str) -> Dict[str, Dict]:
    """Map config name -> its recorded parameters, if sweep_tracking.py left a manifest."""
    path = osp.join(sweep_dir, "sweep_manifest.json")
    if not osp.isfile(path):
        return {}
    with open(path, "r") as f:
        return {entry["name"]: entry for entry in json.load(f)}


def threshold_from_name(name: str) -> Optional[float]:
    """Recover the peak threshold from a ``pt<digits>_...`` config name.

    ``sweep_tracking.py`` encodes it in the directory name, so the ranking still carries a
    threshold column when a config predates the manifest or its entry was lost.
    """
    head = name.split("_", 1)[0]
    if not head.startswith("pt") or not head[2:].isdigit():
        return None
    digits = head[2:]
    return float(f"0.{digits[1:]}") if digits.startswith("0") else float(f"0.{digits}")


def config_dirs(sweep_dir: str, only: Optional[List[str]]) -> List[str]:
    import glob

    found = sorted(osp.dirname(p) for p in glob.glob(osp.join(sweep_dir, "*", "tracked_bboxes.csv")))
    if only:
        wanted = set(only)
        found = [d for d in found if osp.basename(d) in wanted]
    if not found:
        raise SystemExit(f"No */tracked_bboxes.csv under {sweep_dir}. Run sweep_tracking.py first.")
    return found


def score_one(config_dir: str, gt: str, metainfo: str) -> Optional[Dict]:
    """Run evaluate_mot on one configuration and return its final-frame metrics."""
    pred = osp.join(config_dir, "tracked_bboxes.csv")
    save_path = osp.join(config_dir, "mot_evaluation.csv")

    if pd.read_csv(pred).empty:
        print(f"  {osp.basename(config_dir):28s} no predictions, skipped")
        return None

    # report_every_prcnt=1.0 -> a single checkpoint at the last frame, i.e. the headline.
    evaluate_mot(result_path=pred, ground_truth_path=gt, metadata_path=metainfo, save_path=save_path, verbose=False, report_every_prcnt=1.0)

    if not osp.isfile(save_path):
        print(f"  {osp.basename(config_dir):28s} evaluate_mot wrote no CSV, skipped")
        return None
    df = pd.read_csv(save_path)
    if df.empty:
        print(f"  {osp.basename(config_dir):28s} empty evaluation, skipped")
        return None

    # One row per class; these datasets carry a single class.
    row = df.iloc[0].to_dict()
    row["config"] = osp.basename(config_dir)
    return row


def main(args):
    sweep_dir = osp.abspath(osp.expanduser(args.sweep_dir))
    gt = osp.abspath(osp.expanduser(args.gt))
    metainfo = osp.abspath(osp.expanduser(args.metainfo))
    manifest = load_manifest(sweep_dir)
    dirs = config_dirs(sweep_dir, args.only)

    print(f"Scoring {len(dirs)} configuration(s) in {sweep_dir}")
    print(f"  ground truth: {gt}\n")

    rows = []
    for config_dir in dirs:
        row = score_one(config_dir, gt, metainfo)
        if row is None:
            continue
        entry = manifest.get(row["config"], {})
        row["peak_threshold"] = entry.get("peak_threshold", threshold_from_name(row["config"]))
        row["inst_per_frame"] = entry.get("inst_per_frame")
        row["empty_frames_prcnt"] = entry.get("empty_frames_prcnt")
        row["tracking_seconds"] = entry.get("tracking_seconds")
        rows.append(row)
        print(f"  {row['config']:28s} IDF1={row.get('idf1', float('nan')):.4f}  MOTA={row.get('mota', float('nan')):.4f}")

    if not rows:
        raise SystemExit("No configuration produced a scoreable result.")

    df = pd.DataFrame(rows).sort_values("idf1", ascending=False)
    ordered = [c for c in RANK_COLUMNS if c in df.columns] + [c for c in df.columns if c not in RANK_COLUMNS]
    df = df[ordered]

    output = osp.abspath(osp.expanduser(args.output)) if args.output else osp.join(sweep_dir, "sweep_summary.csv")
    df.to_csv(output, index=False)

    best_idf1 = df.iloc[0]
    best_mota = df.sort_values("mota", ascending=False).iloc[0]
    print(f"\nWrote {output} ({len(df)} configs, ranked by IDF1)")
    print(f"  best IDF1: {best_idf1['config']}  IDF1={best_idf1['idf1']:.4f}  MOTA={best_idf1['mota']:.4f}")
    print(f"  best MOTA: {best_mota['config']}  IDF1={best_mota['idf1']:.4f}  MOTA={best_mota['mota']:.4f}")
    print("\nPer-config evaluations are in <config>/mot_evaluation.csv; copy the winners' files into the report directory.")
    return df


if __name__ == "__main__":
    main(parse_args())
