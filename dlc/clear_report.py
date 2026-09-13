"""Produce PrecisionTrack's ``mean_CLEAR_metrics_over_all_videos.csv`` for DLC results.

**Run this in the ``precision_track`` environment, not the ``dlc`` one** -- it imports
``precision_track``. See section 8 of ``dlc/README.md``.

``CLEARMetrics`` is a plain mmengine ``BaseMetric``, so it can be driven directly without
a Runner or a testing loop. Its ``5% ... 100%`` columns are the frame-evolution axis: each
column holds the metric computed cumulatively from frame 0 up to that fraction of the
video, because the underlying ``MOTAccumulator`` is never reset between checkpoints.

Nothing here reimplements the metric or the CSV format; both come from
``precision_track/evaluation/metrics/clear.py`` unchanged, so the output is
column-compatible with the per-tracker CSVs that ``run_tracking_study.py`` collects.

Example:
    python clear_report.py \\
        --pred ../work_dir/dlc_grid/M_affinity01_animals5/tracked_bboxes.csv \\
        --gt ~/Documents/datasets/stripedmice/mot_v2/bboxes/val/<video>.csv \\
        --metainfo ../configs/metadata/stripedmice.py \\
        --output ~/Documents/.../mean_CLEAR_metrics_over_all_videos.csv
"""

import argparse
import os.path as osp

from precision_track.evaluation.metrics.clear import CLEARMetrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", nargs="+", required=True, help="Prediction CSV(s), cxcywh, one per video")
    parser.add_argument("--gt", nargs="+", required=True, help="Ground-truth CSV(s), xywh, one per video, same order as --pred")
    parser.add_argument("--metainfo", required=True, help="Path to the metadata .py file, e.g. ../configs/metadata/stripedmice.py")
    parser.add_argument("--output", required=True, help="Where to write mean_CLEAR_metrics_over_all_videos.csv")
    parser.add_argument(
        "--report-every-prcnt",
        type=float,
        default=0.05,
        help="Frame-evolution granularity. 0.05 (default) gives the 20 columns 5%%..100%% used by configs/tasks/testing_tracking.py.",
    )
    return parser.parse_args()


def main(args):
    if len(args.pred) != len(args.gt):
        raise SystemExit(f"--pred has {len(args.pred)} entries but --gt has {len(args.gt)}; they must pair up one video at a time.")

    # CLEARMetrics.__init__ calls os.path.dirname(output_file) unguarded, so a bare
    # filename would try to makedirs("").
    output = osp.abspath(osp.expanduser(args.output))
    preds = [osp.abspath(osp.expanduser(p)) for p in args.pred]
    gts = [osp.abspath(osp.expanduser(g)) for g in args.gt]

    metric = CLEARMetrics(
        metainfo=osp.abspath(osp.expanduser(args.metainfo)),
        output_file=output,
        report_every_prcnt=args.report_every_prcnt,
    )

    # Note the argument order: data_batch holds PREDICTIONS, data_samples holds GROUND TRUTH.
    metric.process(preds, gts)

    # save_results() fires inside compute_metrics() and writes the CSV.
    summary = metric.compute_metrics(metric.results)

    print(f"\nWrote {output}")
    for key in ("mouse/mota", "mouse/idf1", "Overall/mota", "Overall/idf1"):
        if key in summary:
            print(f"  {key:16s} {summary[key]:.4f}")
    return summary


if __name__ == "__main__":
    main(parse_args())
