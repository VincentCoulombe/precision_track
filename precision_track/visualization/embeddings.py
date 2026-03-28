import os
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from umap import UMAP

from precision_track.registry import OUTPUTS
from precision_track.utils import VideoReader
from precision_track.utils.formatting import reformat

from .palette import ColorPalette


def visualize_embeddings(
    embeddings_path: str,
    bboxes_config: Dict,
    timesteps: Union[int, list[int]],
    labels_config: Optional[Dict] = None,
    video_path: Optional[str] = None,
    output_path: Optional[str] = None,
    min_dist: float = 0.3,
    random_state: int = 42,
    figsize: Tuple[int, int] = (16, 12),
    show_scatter: bool = True,
    show_images: bool = True,
    labels_to_display: list[Union[str, int]] = None,
    target_bbox_size: Tuple[int, int] = (64, 64),
    max_n_neighbors: int = 100,
) -> plt.Figure:
    if not os.path.isfile(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    video = None
    if show_images:
        if video_path is None or bboxes_config is None:
            raise ValueError("video_path and bboxes_config are required when show_images=True")

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video = VideoReader(video_path)

    labels = None
    if labels_config is not None:
        labels = OUTPUTS.build(labels_config)
        labels.read()

    bboxes_output = OUTPUTS.build(bboxes_config)
    bboxes_output.read()
    bbox_format = bboxes_output.bbox_format

    data = np.load(embeddings_path)

    T, N, E = data.shape
    entity_ids = data[0, :, 0].astype(int)
    embeddings_data = data[1:, :, :]

    assert (
        len(bboxes_output) == T - 1
    ), f"The provided embeddings and the provided bounding boxes do not contain the same amount of frames : {len(bboxes_output)} != {T - 1}."

    if labels is not None:
        assert len(labels) == T - 1, f"The provided embeddings and the provided labels do not contain the same amount of frames : {len(labels)} != {T - 1}."

    timestep_indices = []
    selected_timesteps = []

    if isinstance(timesteps, int):
        if timesteps < 0 or timesteps >= T - 1:
            raise ValueError(f"Timesteps {timesteps} out of range [0, {T-2}[")
        rng = np.random.RandomState(random_state)
        selected_timesteps = sorted(rng.choice(T - 1, size=timesteps, replace=False).tolist())
        title_suffix = f"{len(selected_timesteps)} randomly selected Timesteps."
    elif isinstance(timesteps, (list, tuple)):
        selected_timesteps = list(timesteps)
        for t in selected_timesteps:
            if t < 0 or t >= T - 1:
                raise ValueError(f"Timestep {t} out of range [0, {T-2}[")
        title_suffix = f"{len(selected_timesteps)} selected Timesteps."
    else:
        raise ValueError(f"Invalid timestep format: {timesteps}. Expected int or list of ints")

    embeddings_list = []
    bboxes_list = []
    unique_ids = []
    ids_list = []
    labels_list = []
    for t in selected_timesteps:
        frame_output = np.array(bboxes_output[t]).astype(int)
        visible_ids = frame_output[:, 2]
        visible_mask = np.isin(visible_ids, entity_ids)
        visible_idx = []
        for visible_id in visible_ids[visible_mask]:
            visible_idx.append(np.where(visible_id == entity_ids)[0][0])
        visible_idx = np.array(visible_idx, dtype=int)

        if labels is not None:
            frame_labels = np.array(labels[t])
            label_ids = frame_labels[:, 2].astype(int)
            visible_label_mask = np.isin(label_ids, entity_ids)
            visible_label_idx = []
            for visible_id in label_ids[visible_label_mask]:
                visible_label_idx.append(np.where(visible_id == entity_ids)[0][0])

            _, intersect_mask_lbl, intersect_mask = np.intersect1d(visible_label_idx, visible_idx, return_indices=True)
            visible_idx = visible_idx[intersect_mask]
            label_ids = label_ids[intersect_mask_lbl]
            embedding_ids = entity_ids[visible_idx]

            assert np.all(label_ids == embedding_ids)
            labels_list.extend(frame_labels[intersect_mask_lbl, 3])
        else:
            # TODO not sure....
            intersect_mask = visible_idx.sort()
            embedding_ids = entity_ids[visible_idx]

        visible_ids = visible_ids[intersect_mask]
        assert np.all(visible_ids == embedding_ids)

        ids_list.extend(visible_ids)
        unique_ids.extend(list(np.setdiff1d(visible_ids, unique_ids)))
        bboxes_list.append(frame_output[intersect_mask, 3:7])
        embeddings_list.append(embeddings_data[t, visible_idx, :])
        timestep_indices.extend([t] * len(visible_ids))

    embeddings_to_plot = np.vstack(embeddings_list)
    ids_to_plot = np.array(ids_list).reshape(-1)
    labels_to_plot = np.array(labels_list).reshape(-1)

    assert len(embeddings_to_plot) == len(ids_to_plot)
    if labels is not None:
        assert len(embeddings_to_plot) == len(labels_to_plot)

    max_n_neighbors = int(max_n_neighbors)
    assert max_n_neighbors >= 25, "max_n_neighbors must be bigger or equal to 25."
    n_neighbors = max(25, min(max_n_neighbors, int(0.1 * len(ids_to_plot))))

    # Apply UMAP dimensionality reduction
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=random_state,
    )

    embeddings_2d = reducer.fit_transform(embeddings_to_plot)

    fig, ax = plt.subplots(figsize=figsize)
    n_entities = len(unique_ids)

    unique_labels = np.unique(labels_to_plot)
    if labels is not None:
        if labels_to_display is None:
            labels_to_display_mask = np.ones_like(unique_labels, dtype=bool)
        else:
            labels_to_display_mask = np.isin(unique_labels, labels_to_display)
        unique_labels_to_display = unique_labels[labels_to_display_mask]
        size = len(unique_labels)
        label_to_color_idx = {label: idx for idx, label in enumerate(unique_labels)}
    else:
        size = n_entities

    palette = ColorPalette(size=size, normalized_colors=True)

    def get_colors(labels_to_plot, mask, unique_labels):
        if labels_to_plot.size > 0:
            labels_to_plot_i = labels_to_plot[mask]
            to_display_mask = np.isin(labels_to_plot_i, unique_labels)
            color_idx = [label_to_color_idx[entity_label] for entity_label in labels_to_plot_i[to_display_mask]]
            colors = [palette.by_idx(idx).as_rgb() for idx in color_idx]
        else:
            colors = [palette.by_idx(i).as_rgb()]
            to_display_mask = None
        return colors, to_display_mask

    # Plot scatter points if requested
    if show_scatter:
        for i, entity_id in enumerate(unique_ids):
            mask = ids_to_plot == entity_id

            if labels is not None:
                colors, to_display_mask = get_colors(labels_to_plot, mask, unique_labels_to_display)
                points_x = embeddings_2d[mask, 0][to_display_mask]
                points_y = embeddings_2d[mask, 1][to_display_mask]
            else:
                colors = [palette.by_idx(i).as_rgb()]
                points_x = embeddings_2d[mask, 0]
                points_y = embeddings_2d[mask, 1]

            ax.scatter(
                points_x,
                points_y,
                c=colors,
                alpha=0.75,
                s=100,
                edgecolors="black",
                linewidths=0.5,
            )

    # Overlay bounding box images if requested
    if show_images and video is not None:
        bboxes_to_plot = np.vstack(bboxes_list)

        for i, entity_id in enumerate(unique_ids):
            mask = ids_to_plot == entity_id

            if labels is not None:
                colors, to_display_mask = get_colors(labels_to_plot, mask, unique_labels_to_display)
                indices = np.where(mask)[0][to_display_mask]
            else:
                indices = np.where(mask)[0]

            for idx in indices:
                t = timestep_indices[idx]

                entity_bbox = bboxes_to_plot[idx]
                bbox_xyxy = reformat(entity_bbox, bbox_format, "xyxy")
                x1, y1, x2, y2 = bbox_xyxy.astype(int)

                frame = video.get_frame(t)
                if frame is None:
                    continue

                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                bbox_crop = frame[y1:y2, x1:x2]
                bbox_crop_resized = cv2.resize(bbox_crop, target_bbox_size)
                bbox_crop_rgb = cv2.cvtColor(bbox_crop_resized, cv2.COLOR_BGR2RGB)

                imagebox = OffsetImage(bbox_crop_rgb, zoom=0.3)
                if labels is not None:
                    entity_label = labels_to_plot[idx]
                    color_idx = label_to_color_idx[entity_label]
                    edge_color = palette.by_idx(color_idx).as_rgb()
                    linewidth = 2
                else:
                    edge_color = palette.by_idx(i).as_rgb()
                    linewidth = 1

                ab = AnnotationBbox(
                    imagebox,
                    (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                    frameon=True,
                    pad=0.1,
                    bboxprops=dict(edgecolor=edge_color, linewidth=linewidth, facecolor="white", alpha=1.0),
                )

                ax.add_artist(ab)

    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.set_title(f"UMAP of: {title_suffix}", fontsize=14)
    ax.grid(True, alpha=0.3)

    if show_scatter or show_images:
        legend_elements = []

        if labels is not None:
            for label in unique_labels_to_display:
                color_idx = label_to_color_idx[label]
                color = palette.by_idx(color_idx).as_rgb()
                legend_elements.append(Patch(facecolor=color, edgecolor="black", label=f"Label: {label}"))
        else:
            for i, entity_id in enumerate(unique_ids):
                color = palette.by_idx(i).as_rgb()
                legend_elements.append(Patch(facecolor=color, edgecolor="black", label=f"ID: {entity_id}"))

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()

    # Save or show
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {os.path.abspath(output_path)}")
    else:
        plt.show()

    return fig
