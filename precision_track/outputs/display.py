import sys
from typing import List, Union

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


def display_mot_results(evaluations: Union[List[dict], dict], precision=3):
    if isinstance(evaluations, dict):
        evaluations = [evaluations]

    headers = ["Metric"] + [f"Up to {eval['completion_prcnt']*100}%" for eval in evaluations]

    classes = []
    for eval in evaluations:
        for k in eval.keys():
            if k != "completion_prcnt" and k not in classes:
                classes.append(k)

    metric_defs = [
        ("MOTA", "mota", False),
        ("IDF1", "idf1", False),
        ("IDP", "idp", False),
        ("IDR", "idr", False),
        ("Precision", "precision", False),
        ("Recall", "recall", False),
        ("IDFP", "idfp", True),
        ("IDFN", "idfn", True),
        ("IDTP", "idtp", True),
        ("Num Switches", "num_switches", True),
        ("Num Detections", "num_detections", True),
    ]

    table = []
    for cls in classes:
        for metric_name, metric_key, is_int in metric_defs:
            row = [f"{metric_name} on {cls}"]
            for eval in evaluations:
                value_dict = eval.get(cls, {})
                if value_dict:
                    value = eval[cls][metric_key]
                    row.append(int(value) if is_int else value)
            table.append(row)

    print(
        "\n"
        + tabulate(
            table,
            headers=headers,
            tablefmt="github",
            floatfmt=f".{precision}f",
            stralign="left",
        )
    )


def display_class_balance(data: dict):
    total = sum(data.values())
    table = []
    for k, v in sorted(data.items(), key=lambda x: -x[1]):
        bar = "█" * (v * 50 // total)  # scale to max 50 chars
        table.append([k, v, f"{v/total:.2%}", bar])

    print("\n" + tabulate(table, headers=["Class", "Count", "Percent", "Bar"]) + "\n")
