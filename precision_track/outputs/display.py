import sys

import numpy as np
from tabulate import tabulate


def display_progress_bar(iteration, total, length=50):
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\rProgress: |{bar}| {iteration}/{total}")
    sys.stdout.flush()
    if iteration == total:
        print()


def display_latency(times: np.ndarray, title, buffer_size=5, precision=3):
    assert len(times) >= buffer_size
    times = times[buffer_size:]
    mean = np.mean(times)
    table = [
        ["Mean", mean],
        ["Std", np.std(times)],
        ["Min", np.min(times)],
        ["Median", np.median(times)],
        ["Max", np.max(times)],
    ]
    print(
        "\n"
        + tabulate(
            table,
            headers=[title, "Inference Time (s)"],
            tablefmt="github",
            floatfmt=f".{precision}f",
            stralign="left",
        )
    )
    return mean


def display_mot_results(evaluation: dict, precision=3):
    table = []
    for cls, metrics in evaluation.items():
        table.append([f"MOTA on {cls}", metrics["mota"]])
        table.append([f"IDF1 on {cls}", metrics["idf1"]])
        table.append([f"IDP on {cls}", metrics["idp"]])
        table.append([f"IDR on {cls}", metrics["idr"]])
        table.append([f"Precision on {cls}", metrics["precision"]])
        table.append([f"Recall on {cls}", metrics["recall"]])
        table.append([f"IDFP on {cls}", int(metrics["idfp"])])
        table.append([f"IDFN on {cls}", int(metrics["idfn"])])
        table.append([f"IDTP on {cls}", int(metrics["idtp"])])
        table.append([f"Num Switches on {cls}", int(metrics["num_switches"])])
        table.append([f"Num Detections on {cls}", int(metrics["num_detections"])])
    print(
        "\n"
        + tabulate(
            table,
            headers=["Metric", "Score"],
            tablefmt="github",
            floatfmt=f".{precision}f",
            stralign="left",
        )
    )
