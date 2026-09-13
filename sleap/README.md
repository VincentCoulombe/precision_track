<div align="center">

# SLEAP ↔ PrecisionTrack

**Train a [SLEAP](https://github.com/talmolab/sleap) pose model on a PrecisionTrack dataset, track with it, and get PrecisionTrack-formatted results back.**

</div>

This directory holds everything needed to benchmark SLEAP against PrecisionTrack on **the exact same data**. It does three things:

1. **Converts** a PrecisionTrack COCO pose dataset into SLEAP `.slp` label files ([`coco2sleap.py`](coco2sleap.py)), using [SLEAP IO](https://io.sleap.ai/latest/).
2. **Trains** a SLEAP model on it ([`train.py`](train.py)), reusing your original train/val split, with every SLEAP architecture selectable through one flag.
3. **Tracks** a video and writes `tracked_kpts.csv` and `tracked_bboxes.csv` ([`track.py`](track.py)) in the very same format PrecisionTrack writes, so the two systems can be visualized and evaluated side by side.

It is the twin of [`dlc/`](../dlc/README.md), deliberately: the same commands in the same order, the same output files, the same evaluation path.

- **⚠️IMPORTANT⚠️** SLEAP **cannot** share PrecisionTrack's environment. Everything in this directory runs in a separate conda environment, described in section 2.

| | PrecisionTrack | SLEAP |
| --- | --- | --- |
| Python | 3.11 | 3.11 – 3.13 |
| Deep learning | torch 2.7.1 + ONNX Runtime / TensorRT | torch + Lightning (pinned by `sleap-nn`) |
| numpy | 1.26.0 | resolved by SLEAP |
| Where it runs | Docker image (`docker/`) or Colab | the `sleap` conda environment below |

- **Note:** SLEAP is **no longer TensorFlow-based**. Since 1.6, all neural network work is delegated to [`sleap-nn`](https://github.com/talmolab/sleap-nn), a PyTorch + Lightning backend. Older instructions pinning `tensorflow` and Python 3.7 do not apply.

Because of the environment split, [`pt_format.py`](pt_format.py) is a standalone copy of PrecisionTrack's CSV writers that depends on nothing but numpy and pandas. `tests/test_sleap_format_parity.py` (run from the PrecisionTrack environment) asserts it stays byte-identical to `precision_track/outputs/csv.py`, and identical to `dlc/pt_format.py`.

## Contents

```text
sleap/
├── README.md          # this file
├── requirements.txt   # tabulate, PyYAML, pandas (SLEAP itself is installed separately)
├── coco2sleap.py      # COCO dataset            -> train.slp / val.slp
├── train.py           # .slp files              -> trained model(s)
├── track.py           # trained model + video   -> PrecisionTrack CSVs
├── profile_stages.py  # per-stage timing        -> throughput CSV
├── clear_report.py    # PrecisionTrack CLEAR metrics (runs in the other environment)
├── slp_io.py          # .slp reader and model-directory resolver
├── pt_format.py       # standalone CsvKeypoints / CsvBoundingBoxes
├── video_utils.py     # video resizing helpers
├── datasets/          # generated .slp files (git-ignored)
└── models/            # trained checkpoints (git-ignored)
```

## 1) Install mamba (or conda)

If you already have conda, mamba or miniforge, skip to section 2. Otherwise install [Miniforge](https://github.com/conda-forge/miniforge), which ships `mamba`:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Close and reopen your terminal so `mamba` lands on your `PATH`.

- **Note:** Windows users should run all of this inside WSL, exactly as described in section 5.1 of the [main README](../README.md).

## 2) Create the SLEAP environment

```bash
mamba create -y -n sleap python=3.11
mamba activate sleap
```

## 3) Install SLEAP and this pipeline's dependencies

Pick the extra matching your CUDA version — `nn-cuda130`, `nn-cuda128`, `nn-cuda118`, or `nn-cpu`:

```bash
pip install "sleap[nn-cuda130]"
pip install -r sleap/requirements.txt
```

- **Note:** check your CUDA version with `nvidia-smi` before choosing. On a CUDA machine, verify the GPU is visible before training:

  ```bash
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

- **Note:** installing the full `sleap` package (rather than `sleap-nn` alone) also gives you `sleap-label`, the annotation GUI. It is worth having: `sleap-label datasets/<name>/train.slp` renders the converted dataset, which is the fastest way to confirm a conversion produced sane skeletons before committing to a long training run.

- **Note:** OpenCV needs a few system libraries that slim Ubuntu images lack. If you hit `ImportError: libGL.so.1`:

  ```bash
  sudo apt-get install -y libgl1 libglib2.0-0
  ```

All the commands below are run from inside this directory:

```bash
cd sleap
```

## 4) Convert your PrecisionTrack dataset into SLEAP labels

PrecisionTrack datasets are COCO-formatted (see section 3 of the [main README](../README.md)): an `annotations/train.json` + `annotations/val.json` pair and a flat `images/` folder. `coco2sleap.py` reads each split with `sleap_io.load_coco` and writes it back out with `sleap_io.save_slp`.

```bash
python coco2sleap.py ~/Documents/datasets/stripedmice/july_2026_640x640 \
    --name july_2026_640x640 --require-identity
```

This creates `datasets/july_2026_640x640/`, containing:

- `train.slp` and `val.slp`, one per COCO split. **Your original train/val boundary is preserved by construction** — the split is simply which file a frame is in.
- `train_id.slp` and `val_id.slp` when `--require-identity` is passed: the subset of frames where *every* instance carries an identity, which is what the `multi_class_*` architectures need (section 5).
- `dataset_info.json`, recording the skeleton, per-split counts and identity coverage.

Useful options:

- **<u>--require-identity</u>** — also write the identity-only `.slp` files. Cheap; do it if you might ever train an identity head.
- **<u>--embed</u>** — bake the images into the `.slp` rather than referencing them on disk. Makes the file self-contained (and much larger); useful for moving a dataset to another machine.
- **<u>--splits</u>** — which annotation files to convert. Defaults to `train val`.
- **<u>--keep-empty</u>** — keep frames left with no usable instance. They are dropped by default, along with annotations whose every keypoint is missing.

- **Note:** this converter is much shorter than [`coco2dlc.py`](../dlc/coco2dlc.py) for one reason: `sleap-io` can *write* the SLEAP format. There is no DLC writer, so the DLC converter has to build DeepLabCut's `CollectedData` HDF5 by hand and record the split in a side-car JSON for `train.py` to re-impose. Here the split is the two files.

- **Note:** keypoints marked as not visible in COCO (visibility flag `0`) become `NaN`, which is how SLEAP represents an unlabelled node.

## 5) Train

```bash
python train.py --dataset-dir datasets/july_2026_640x640 \
    --head-config bottomup --max-epochs 200 --batch-size 4
```

`--head-config` selects the architecture. **Two of them are two models**, a centroid detector followed by a pose model that runs on its crops; `train.py` trains those in sequence and writes a `model_group.json` recording the order inference needs, so `track.py` always takes a single `--model`.

| `--head-config` | Models trained | What it is |
| --- | --- | --- |
| `bottomup` | 1 | Confidence maps + part-affinity fields, grouped after inference. The direct analogue of the multi-animal DLC baseline. |
| `top-down` | 2 | `centroid` → `centered_instance`. Usually the most accurate on small-animal data. |
| `single_instance` | 1 | One animal per frame, no grouping. |
| `multi_class_bottomup` | 1 | `bottomup` plus a head that classifies identity from appearance. |
| `multi_class_topdown` | 2 | `centroid` → `multi_class_topdown`. |

Useful options: `--backbone` (`unet` by default; also `unet_medium_rf`, `unet_large_rf`, `convnext`, `swint`), `--batch-size` (default 4), `--lr` (default 1e-3), `--crop-size` (the cropping heads only), `--scale`, `--device`, `--run-name`, and `--skip-evaluation`.

- **⚠️IMPORTANT⚠️** The two `multi_class_*` heads learn identity from appearance, so they need **every** instance in a frame labelled with a consistent identity — a frame containing one unlabelled animal actively teaches the head that the animal belongs to no class. `train.py` counts fully-identified frames before training and **refuses** to start when there are too few, rather than handing you a model that looks trained and tracks worse than no identity head at all. PrecisionTrack's COCO exports carry identities in `attributes.object_id`, but usually only on a small annotated subset. Two things unblock a real identity run: annotating identities across the dataset, or converting a single clip on its own — COCO identities are numbered **per source clip**, so `object_id 1` in two different clips is two different animals, and a single global identity head over several clips is ill-posed. `--allow-thin-identity` overrides the refusal if you want the run anyway.

- **Note:** `sleap_nn.train` defaults `save_ckpt` to `False`, which trains a model and then discards it. `train.py` always passes `save_ckpt=True`; if you call `sleap_nn` yourself, do the same or there will be nothing to track with.

Each run directory ends up at `models/<run-name>/`, holding `best.ckpt` and `training_config.yaml` — the two files inference needs.

## 6) Track a video

```bash
python track.py <video>.mp4 \
    --model models/bottomup \
    --out-dir ../work_dir/sleap_bottomup --max-instances 5 --img-size 640 640
```

This runs pose estimation and tracking in one `predict()` call, reports its speed, then converts the predicted `.slp` into PrecisionTrack CSVs. The speed report looks like this (numbers are illustrative):

```text
| SLEAP: predict + track | Value    |
|------------------------|----------|
| Frames                 | 9000     |
| Total time (s)         | 152.310  |
| Latency per frame (ms) | 16.923   |
| Throughput (FPS)       | 59.089   |
```

Useful options:

- **<u>--max-instances</u>** — the real animal count. Caps both detections per frame and identities created.
- **<u>--img-size</u>** — `HEIGHT WIDTH` to rescale the video to before tracking, so a speed comparison against PrecisionTrack is measured at the same resolution. Output coordinates are rescaled back to the source resolution automatically.
- **<u>--peak-threshold</u>** — minimum keypoint confidence. This is the main knob when animals go undetected: lowering it trades precision for recall.
- **<u>--scoring-method</u>** — how an instance is matched to a track: `oks` (default), `iou`, `cosine_sim`, `euclidean_dist`.
- **<u>--candidates-method</u>** — `fixed_window` (default) or `local_queues`.
- **<u>--window-size</u>** — how many frames of history association considers.
- **<u>--use-kalman</u>**, **<u>--use-flow</u>**, **<u>--post-connect-single-breaks</u>** — motion filtering, optical-flow candidate shifting, and single-frame gap bridging. The first and last require `--max-instances`.
- **<u>--no-tracking</u>** — pose estimation only, every detection its own id. Useful for measuring the detector separately from the tracker.
- **<u>--skip-inference</u>** — reuse the `.slp` from a previous run and only redo the conversion.

- **Note:** `--max-instances` is passed to the tracker as `max_tracks`, which **`fixed_window` ignores**. SLEAP silently switches to `local_queues` to honour it, so a sweep over `--candidates-method` will not do what you expect unless you account for that; `track.py` prints a note when it applies.

Use [`profile_stages.py`](profile_stages.py) with the same arguments to time pose estimation and tracking separately and write `mean_throughput_per_substep.csv`.

## 7) Output formats

Both files follow PrecisionTrack's MOT-style CSV layout: three identifier columns (`frame_id`, `class_id`, `instance_id`) followed by the payload. They are named exactly as PrecisionTrack's own tracking outputs, and are identical in format to `dlc/track.py`'s.

| File | Header | PrecisionTrack equivalent |
| --- | --- | --- |
| `tracked_kpts.csv` | `frame_id,class_id,instance_id,x0,y0,score0,x1,y1,score1,...` | `CsvKeypoints` |
| `tracked_bboxes.csv` | `frame_id,class_id,instance_id,cx,cy,w,h,score` | `CsvBoundingBoxes` |

`instance_id` is the index of the SLEAP track the instance was assigned, and is stable across frames. Detections the tracker left unassigned get trailing ids of their own rather than being dropped. `class_id` is always `0`: SLEAP skeletons hold a single class.

- **⚠️IMPORTANT⚠️** SLEAP has **no notion of a bounding box**. Each box is derived as the tightest one enclosing that instance's detected keypoints — the same derivation PrecisionTrack applies internally through `keypoints_cxcywh` — and its `score` is the mean confidence of those keypoints. Boxes are therefore systematically tighter than annotated ground-truth boxes, which is worth keeping in mind when reading IoU-based metrics. If your skeleton reaches an extremity your ground-truth boxes do not cover, `--bbox-exclude-keypoints` leaves those nodes out of the box while still writing them to `tracked_kpts.csv`.

## 8) Visualize and evaluate with PrecisionTrack

**Switch back to the PrecisionTrack environment** for this section.

Because `track.py` writes the same file names `configs/tasks/tracking.py` uses, PrecisionTrack's own visualizer reads SLEAP's results directly. Point `saving_directory` in `configs/user_configs.yaml` at the parent of your `--out-dir`, then:

```bash
cd tools
python visualize.py <video>.mp4 ../work_dir/sleap_bottomup/sleap_vis.mp4
```

For MOT evaluation, feed `tracked_bboxes.csv` straight to `evaluate_mot` — **keep the default `cxcywh`**. `Evaluator.update` reformats the predictions from `cxcywh` to `xywh` itself; it is the *ground-truth* file that must already be `xywh` (`frame_id,class_id,instance_id,x,y,w,h,score`, as validated by `assert_mot_file_is_ok`).

```python
from precision_track.evaluation.utils.mot import evaluate_mot

evaluate_mot(
    result_path="../work_dir/sleap_bottomup/tracked_bboxes.csv",      # cxcywh, as written by track.py
    ground_truth_path="<your MOT dataset>/bboxes/val/<video>.csv",    # xywh
    metadata_path="../configs/metadata/stripedmice.py",
    save_path="../work_dir/sleap_bottomup/mot_evaluation.csv",
)
```

For the frame-evolution report the rest of the tracking study uses, [`clear_report.py`](clear_report.py) drives `CLEARMetrics` directly and writes `mean_CLEAR_metrics_over_all_videos.csv`:

```bash
python clear_report.py \
    --pred ../work_dir/sleap_bottomup/tracked_bboxes.csv \
    --gt <your MOT dataset>/bboxes/val/<video>.csv \
    --metainfo ../configs/metadata/stripedmice.py \
    --output ../work_dir/sleap_bottomup/mean_CLEAR_metrics_over_all_videos.csv
```

`--bbox-format xywh` exists for the other direction — producing a ground-truth-shaped file, or feeding external MOT tooling that expects the `x,y,w,h` layout. Do not pass it for `result_path`, or every box will be shifted by half its width and height.

## Troubleshooting

- **`--model: <path> is neither a model directory nor a model_group.json`** — point `--model` at a run directory under `models/`, or at the `model_group.json` a two-model `top-down` run wrote.
- **Training finishes but `models/<run>/best.ckpt` does not exist** — `sleap_nn.train` was called with `save_ckpt=False`. `train.py` always sets it; a hand-rolled call must too.
- **`Only N frame(s) have a complete set of identities`** — a `multi_class_*` head was requested without the identity labels to support it. See the ⚠️IMPORTANT⚠️ note in section 5.
- **`ImportError: libGL.so.1`** — `sudo apt-get install -y libgl1 libglib2.0-0`.
- **CUDA out of memory during training** — lower `--batch-size`, or `--scale` below 1.0.
- **CUDA out of memory during tracking** — lower `--batch-size` (inference defaults to 4).

## Caveats when comparing against PrecisionTrack

- Unless a `multi_class_*` head was trained (see section 5), SLEAP has no appearance-based identity, so identity-aware metrics measure its frame-to-frame association alone.
- Bounding boxes are keypoint-derived, not predicted (see section 7). Compare predicted and ground-truth box areas before trusting any IoU-based metric.
- Always pass `--img-size` matching your PrecisionTrack input resolution before comparing throughput.
- `profile_stages.py`'s substep names are SLEAP's own and do not align row-for-row with PrecisionTrack's pipeline stages. Only the `end_to_end` row is comparable.
