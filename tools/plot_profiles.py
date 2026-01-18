import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
import numpy as np
from mmengine import Config

from precision_track.utils import load_user_configs


def identify_peaks(values, std_threshold=2.0):
    values_array = np.array(values)
    valid_values = values_array[~np.isnan(values_array)]
    if len(valid_values) == 0:
        return []

    mean_val = np.nanmean(values_array)
    std_val = np.nanstd(values_array)
    threshold = mean_val + std_threshold * std_val

    peaks = []
    for i in range(1, len(values_array) - 1):
        if np.isnan(values_array[i]) or np.isnan(values_array[i - 1]) or np.isnan(values_array[i + 1]):
            continue
        if values_array[i] > values_array[i - 1] and values_array[i] > values_array[i + 1] and values_array[i] > threshold:
            peaks.append(i)

    return peaks


def plot_single_metric(key, values, output_path, std_threshold, palette, color_idx):
    """Plot a single metric and save to file."""
    fig, ax = plt.subplots(figsize=(12, 6))

    color = palette[color_idx % len(palette)]
    values_array = np.array(values, dtype=float)

    if key == "detection":
        x_label = "Batch #"
        y_label = "Delay per Image (s)"
    else:
        x_label = "Frame #"
        y_label = "Delay (s)"

    jit_frames = np.where((values_array >= 0.03) & (~np.isnan(values_array)))[0]

    plot_values = values_array.copy()
    plot_values[jit_frames] = np.nan

    x_indices = np.arange(len(values_array))

    ax.plot(x_indices, plot_values, color=color, linewidth=1.5)

    for jit_idx in jit_frames:
        y_neighbors = []
        if jit_idx > 0 and not np.isnan(values_array[jit_idx - 1]):
            y_neighbors.append(values_array[jit_idx - 1])
        if jit_idx < len(values_array) - 1 and not np.isnan(values_array[jit_idx + 1]):
            y_neighbors.append(values_array[jit_idx + 1])

        if y_neighbors:
            y_position = np.mean(y_neighbors)
        else:
            valid_vals = values_array[~np.isnan(values_array)]
            y_position = np.mean(valid_vals) if len(valid_vals) > 0 else 0

        ax.annotate(
            f"JiT {jit_idx}",
            xy=(jit_idx, y_position),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="orange",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="orange", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color="orange", lw=1.5),
        )

    peak_indices = identify_peaks(values_array, std_threshold=std_threshold)

    for peak_idx in peak_indices:
        if peak_idx in jit_frames:
            continue
        y_value = values_array[peak_idx]
        if np.isnan(y_value):
            continue
        ax.annotate(
            f"{peak_idx}",
            xy=(peak_idx, y_value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="red",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{key.replace('_', ' ').title()} - Time Profile")
    ax.grid(True, alpha=0.3)

    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_multiple_metrics(data_dict, output_path, std_threshold, palette):
    """Plot multiple metrics in a grid layout."""
    n_plots = len(data_dict)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows), sharey=True)

    if n_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (key, values) in enumerate(data_dict.items()):
        color = palette[i % len(palette)]
        ax = axes_flat[i]
        values_array = np.array(values, dtype=float)

        if key == "detection":
            x_label = "Batch #"
            y_label = "Delay per Image (s)"
        else:
            x_label = "Frame #"
            y_label = "Delay (s)"

        jit_frames = np.where((values_array >= 0.03) & (~np.isnan(values_array)))[0]

        plot_values = values_array.copy()
        plot_values[jit_frames] = np.nan

        x_indices = np.arange(len(values_array))

        ax.plot(x_indices, plot_values, color=color, linewidth=1.5)

        for jit_idx in jit_frames:
            y_neighbors = []
            if jit_idx > 0 and not np.isnan(values_array[jit_idx - 1]):
                y_neighbors.append(values_array[jit_idx - 1])
            if jit_idx < len(values_array) - 1 and not np.isnan(values_array[jit_idx + 1]):
                y_neighbors.append(values_array[jit_idx + 1])

            if y_neighbors:
                y_position = np.mean(y_neighbors)
            else:
                valid_vals = values_array[~np.isnan(values_array)]
                y_position = np.mean(valid_vals) if len(valid_vals) > 0 else 0

            ax.annotate(
                f"JiT {jit_idx}",
                xy=(jit_idx, y_position),
                xytext=(0, -15),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="orange",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="orange", alpha=0.8),
                arrowprops=dict(arrowstyle="->", color="orange", lw=1.5),
            )

        peak_indices = identify_peaks(values_array, std_threshold=std_threshold)

        for peak_idx in peak_indices:
            if peak_idx in jit_frames:
                continue
            y_value = values_array[peak_idx]
            if np.isnan(y_value):
                continue
            ax.annotate(
                f"{peak_idx}",
                xy=(peak_idx, y_value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="red",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8),
            )

        ax.set_xlabel(x_label)
        if i % n_cols == 0:
            ax.set_ylabel(y_label)
        ax.set_title(f"{key.replace('_', ' ').title()} - Time Profile")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_profiling_data(json_path: str, std_threshold: float = 2.0) -> None:

    system_configs_path = "../configs/tasks/tracking.py"
    with open("../configs/user_configs.yaml", "r") as f:
        user_configs = yaml.safe_load(f)
    load_user_configs(user_configs, system_configs_path)

    config = Config.fromfile(system_configs_path)

    output_dir = os.path.join(config.work_dir, "profiles", "graphs")
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    palette = sns.color_palette("deep")

    detection_data = {}
    tracking_data = {}

    for key, values in data.items():
        if key == "detection":
            detection_data[key] = values
        else:
            tracking_data[key] = values

    if detection_data:
        detection_path = os.path.join(output_dir, "detection_profile.png")
        plot_single_metric("detection", detection_data["detection"], detection_path, std_threshold, palette, 0)
        print(f"Saved: {detection_path}")

    if tracking_data:
        tracking_path = os.path.join(output_dir, "tracking_profile.png")
        if len(tracking_data) == 1:
            key, values = list(tracking_data.items())[0]
            plot_single_metric(key, values, tracking_path, std_threshold, palette, 1)
        else:
            plot_multiple_metrics(tracking_data, tracking_path, std_threshold, palette)
        print(f"Saved: {tracking_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot profiling data.")
    parser.add_argument("json_file", help="Path to the profiling JSON file")
    parser.add_argument("--std-threshold", type=float, default=2.0, help="Number of standard deviations above mean to consider a peak (default: 2.0)")
    args = parser.parse_args()

    plot_profiling_data(args.json_file, std_threshold=args.std_threshold)
