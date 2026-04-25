import argparse
import os

import cv2
import matplotlib
import numpy as np
from umap import UMAP

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from precision_track.outputs.csv import CsvBoundingBoxes
from precision_track.outputs.pth import PthAppearanceDatabaseOutput
from precision_track.utils import VideoReader
from precision_track.visualization.painters import BoundingBoxPainter, LabelPainter
from precision_track.visualization.palette import ColorPalette
from precision_track.visualization.writers import FrameIdWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize subject appearances over time")
    parser.add_argument("data_dir", help="Directory containing tracked_bboxes.csv and appearance_database.pth")
    parser.add_argument("video", help="Path to the raw video")
    args = parser.parse_args()
    return args


def main(args):
    assert os.path.isdir(args.data_dir)
    appearances_path = os.path.join(args.data_dir, "appearance_database.pth")
    assert os.path.isfile(appearances_path)
    appearances = PthAppearanceDatabaseOutput(appearances_path)
    appearances.read()

    feature_id_to_frame = {}
    for frame_id, fids in appearances.fact_frame_ids.items():
        for fid in fids:
            feature_id_to_frame.setdefault(fid, frame_id)

    feature_ids = list(appearances.unique_features.keys())
    embeddings = np.stack([appearances.unique_features[fid].numpy() for fid in feature_ids])

    identities = np.array([appearances.unique_identities[fid] for fid in feature_ids])

    identity_to_instance_id = {}
    for fid in feature_ids:
        identity = appearances.unique_identities[fid]
        identity_to_instance_id[identity] = int(identity.split("_")[-1])

    n_neighbors = max(2, min(50, int(0.1 * len(feature_ids))))
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.3, n_components=2, metric="cosine", random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    print(f"UMAP done: {embeddings_2d.shape}, {len(np.unique(identities))} unique subjects")

    fid_to_idx = {fid: i for i, fid in enumerate(feature_ids)}

    x_range = embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min() or 1
    y_range = embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min() or 1
    x_lim = (embeddings_2d[:, 0].min() - 0.05 * x_range, embeddings_2d[:, 0].max() + 0.05 * x_range)
    y_lim = (embeddings_2d[:, 1].min() - 0.05 * y_range, embeddings_2d[:, 1].max() + 0.05 * y_range)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    video_source = VideoReader(args.video)
    fps = video_source.fps
    w, h = video_source.resolution
    canvas_w = w // 2

    FIG_SIZE_PX = 800
    DPI = 100
    output_path = os.path.join(args.data_dir, f"{video_name}_w_appearances.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w + canvas_w, h))

    tracked_bboxes = CsvBoundingBoxes(
        os.path.join(args.data_dir, "tracked_bboxes.csv"),
        subtype="tracked_bboxes",
    )
    tracked_bboxes.read()

    palette_cfg = dict(names=["Spectral", "deep"], size=20, nan_color=[255, 255, 255])
    palette = ColorPalette(**palette_cfg)

    bbox_painter = BoundingBoxPainter(
        annotations=[dict(type="Box", thickness=6, format="cxcywh")],
        subtype="tracked_bboxes",
        palette=palette_cfg,
    )
    label_painter = LabelPainter(
        palette=palette_cfg,
        info=["class", "id", "score"],
        format="cxcywh",
        label_position="TOP_CENTER",
        text_color=[0, 0, 0],
        text_scale=2,
        text_thickness=2,
        text_padding=10,
        border_radius=1,
    )
    frame_id_writer = FrameIdWriter(
        text_anchor=[100, 10],
        text_color=[255, 255, 255],
        text_scale=1,
        text_thickness=2,
        text_padding=10,
    )

    for frame_id, frame_bboxes in enumerate(tracked_bboxes):
        fig, ax = plt.subplots(figsize=(FIG_SIZE_PX / DPI, FIG_SIZE_PX / DPI), dpi=DPI)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.axis("off")

        fids = appearances.fact_frame_ids.get(frame_id, [])
        if fids:
            indices = [fid_to_idx[fid] for fid in fids if fid in fid_to_idx]
            pts = embeddings_2d[indices]
            colors = [tuple(c / 255.0 for c in palette.by_idx(identity_to_instance_id[identities[i]]).as_rgb()) for i in indices]
            ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=100, edgecolors="black", linewidths=0.5)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        canvas_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        plt.close(fig)

        video_frame = video_source.get_frame(frame_id)
        if video_frame is None:
            video_frame = np.full((h, w, 3), 255, dtype=np.uint8)

        outputs = [{"CsvBoundingBoxes-tracked_bboxes": np.array(frame_bboxes)}]
        video_frame = bbox_painter(video_frame, outputs, frame_id)
        video_frame = label_painter(video_frame, outputs, frame_id)
        video_frame = frame_id_writer(video_frame, outputs, frame_id)

        sq = min(canvas_w, h)
        canvas_sq = cv2.resize(canvas_bgr, (sq, sq))
        pad_top = (h - sq) // 2
        pad_bottom = h - sq - pad_top
        pad_left = (canvas_w - sq) // 2
        pad_right = canvas_w - sq - pad_left
        canvas_padded = cv2.copyMakeBorder(canvas_sq, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        combined = np.concatenate([video_frame, canvas_padded], axis=1)
        out.write(combined)

    out.release()
    print(f"Saved: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main(parse_args())
