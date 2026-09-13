<div align="center">

# DeepLabCut ↔ PrecisionTrack

**Train a multi-animal [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) model on a PrecisionTrack dataset, track with it, and get PrecisionTrack-formatted results back.**

</div>

This directory holds everything needed to benchmark DeepLabCut (DLC) against PrecisionTrack on **the exact same data**. It does three things:

1. **Converts** a PrecisionTrack COCO pose dataset into a multi-animal DLC project ([`coco2dlc.py`](coco2dlc.py)), using [SLEAP IO](https://io.sleap.ai/latest/) to parse the COCO side.
2. **Trains** a DLC model on it ([`train.py`](train.py)), reusing your original train/val split.
3. **Tracks** a video and writes `tracked_kpts.csv` and `tracked_bboxes.csv` ([`track.py`](track.py)) in the very same format PrecisionTrack writes, so the two systems can be visualized and evaluated side by side.

- **⚠️IMPORTANT⚠️** DeepLabCut **cannot** share PrecisionTrack's environment. Everything in this directory runs in a separate conda environment, described in section 2.

| | PrecisionTrack | DeepLabCut |
| --- | --- | --- |
| Python | 3.11 | 3.10 |
| Deep learning | torch 2.7.1 + ONNX Runtime / TensorRT | torch (pinned by DLC) |
| numpy | 1.26.0 | resolved by DLC |
| Where it runs | Docker image (`docker/`) or Colab | the `dlc` conda environment below |

Because of that split, [`pt_format.py`](pt_format.py) is a standalone copy of PrecisionTrack's CSV writers that depends on nothing but numpy and pandas. `tests/test_dlc_format_parity.py` (run from the PrecisionTrack environment) asserts it stays byte-identical to `precision_track/outputs/csv.py`.

## Contents

```text
dlc/
├── README.md          # this file
├── requirements.txt   # sleap-io, tables, tabulate, PyYAML
├── coco2dlc.py        # COCO dataset            -> DLC project
├── train.py           # DLC project             -> trained model
├── track.py           # trained model + video   -> PrecisionTrack CSVs
├── dlc_io.py          # DLC project writer and tracked-.h5 reader
├── pt_format.py       # standalone CsvKeypoints / CsvBoundingBoxes
├── video_utils.py     # video resizing helpers
└── projects/          # generated DLC projects (git-ignored)
```

## 1) Install mamba (or conda)

If you already have conda, mamba or miniforge, skip to section 2. Otherwise install [Miniforge](https://github.com/conda-forge/miniforge), which ships `mamba`:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Close and reopen your terminal so `mamba` lands on your `PATH`.

- **Note:** Windows users should run all of this inside WSL, exactly as described in section 5.1 of the [main README](../README.md).

## 2) Create the DeepLabCut environment

```bash
mamba create -y -n dlc python=3.10
mamba activate dlc
```

## 3) Install DeepLabCut and this pipeline's dependencies

```bash
pip install "deeplabcut[pytorch]"
pip install -r dlc/requirements.txt
```

- **Note:** `deeplabcut[pytorch]` installs the PyTorch engine, which is the one this pipeline targets. On a CUDA machine, verify the GPU is visible before training:

  ```bash
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

- **Note:** OpenCV needs a few system libraries that slim Ubuntu images lack. If you hit `ImportError: libGL.so.1`:

  ```bash
  sudo apt-get install -y libgl1 libglib2.0-0
  ```

All the commands below are run from inside this directory:

```bash
cd dlc
```

## 4) Convert your PrecisionTrack dataset into a DLC project

PrecisionTrack datasets are COCO-formatted (see section 3 of the [main README](../README.md)): a `annotations/train.json` + `annotations/val.json` pair and a flat `images/` folder. `coco2dlc.py` reads both splits with `sleap_io.load_coco`, then writes the multi-animal DLC project that SLEAP IO itself cannot write.

```bash
python coco2dlc.py ~/Documents/datasets/MICE/pose-estimation_640x640 --name mice --scorer coco
```

This creates `projects/mice-coco-<date>/`, containing:

- `config.yaml` with your dataset's keypoints as `multianimalbodyparts`, your COCO `skeleton` as DLC edges, and `individual1..individualN`.
- `labeled-data/<name>/` holding every image of both splits plus `CollectedData_<scorer>.h5` and `.csv`.
- `training-datasets/coco_split.json`, which records your original train/val split so `train.py` can reproduce it.

Useful options:

- **<u>--individuals</u>** — number of animals. Defaults to `auto`, i.e. the largest number of annotations found in a single image (20 for the MICE dataset, 6 for the striped-mice one). Pass an explicit integer to cap it.
- **<u>--symlink</u>** — symlink the images into `labeled-data/` instead of copying them. Saves disk space; keep the source dataset in place.
- **<u>--videos</u>** — videos to register in the project's `video_sets`. Optional: DLC only needs them for its own frame-extraction workflow, which this pipeline replaces.
- **<u>--splits</u>** — which annotation files to convert. Defaults to `train val`.

- **⚠️IMPORTANT⚠️** PrecisionTrack's COCO datasets carry **no track IDs**, so instances are assigned to `individual1..individualN` in annotation order and `identity` is left `false` in `config.yaml`. This is correct for multi-animal DLC — it assembles animals from part-affinity fields, which does not need consistent identities across frames — but it does mean DLC's optional identity head is not trained. Only pass `--identity` if your annotations really do carry persistent identities.

- **Note:** Keypoints marked as not visible in COCO (visibility flag `0`) become `NaN`, which is how DLC represents an unlabelled bodypart. Both the `0/1` and `0/2` visibility conventions are handled.

## 5) Train

```bash
python train.py --project projects/mice-coco-<date>/config.yaml --net-type resnet_50 --epochs 200
```

`train.py` runs DLC's three standard steps in order:

1. `create_training_dataset` with the PyTorch engine, passing the train/test indices recorded in `coco_split.json` so DLC evaluates on **your** val split rather than a fresh random one. Pass `--random-split` to let DLC sample its own instead.
2. `train_network`.
3. `evaluate_network`, which prints RMSE and mAP.

Useful options: `--batch-size` (default 8), `--save-epochs` (snapshot interval, default 25), `--device` (e.g. `cuda:0`), `--shuffle` (default 1), `--skip-training-dataset` to reuse an existing shuffle, and `--skip-evaluation`.

- **Note:** `resnet_50` is DLC's default multi-animal backbone on the PyTorch engine. Other options (`dekr_w32`, `top_down_resnet_50`, …) depend on your installed version; list them with `python -c "from deeplabcut.pose_estimation_pytorch import available_models; print(available_models())"`.

## 6) Track a video

```bash
python track.py ../assets/20mice.avi \
    --project projects/mice-coco-<date>/config.yaml \
    --n-tracks 20 --out-dir ../work_dir/20mice --img-size 640 640
```

This runs `analyze_videos(..., auto_track=True)` (pose estimation, tracklet conversion and stitching in one call), reports its speed, then converts DLC's tracked `.h5` into PrecisionTrack CSVs. The speed report looks like this (numbers are illustrative):

```text
| DeepLabCut: analyze_videos | Value    |
|----------------------------|----------|
| Frames                     | 1471     |
| Total time (s)             | 34.951   |
| Latency per frame (ms)     | 23.760   |
| Throughput (FPS)           | 42.089   |
```

Useful options:

- **<u>--img-size</u>** — `HEIGHT WIDTH` to rescale the video to before tracking. DLC does not resize internally, so matching the resolution your PrecisionTrack model was trained on is what makes a speed comparison fair. Output coordinates are rescaled back to the source resolution automatically.
- **<u>--n-tracks</u>** — number of animals to stitch. Defaults to the project's `individuals` count.
- **<u>--bbox-format</u>** — `cxcywh` (default) or `xywh`. See section 8.
- **<u>--pcutoff</u>** — drop keypoints below this likelihood. Defaults to `0`, keeping everything.
- **<u>--skip-analysis</u>** — reuse the `.h5` from a previous run and only redo the conversion.

## 7) Output formats

Both files follow PrecisionTrack's MOT-style CSV layout: three identifier columns (`frame_id`, `class_id`, `instance_id`) followed by the payload. They are named exactly as PrecisionTrack's own tracking outputs.

| File | Header | PrecisionTrack equivalent |
| --- | --- | --- |
| `tracked_kpts.csv` | `frame_id,class_id,instance_id,x0,y0,score0,x1,y1,score1,...` | `CsvKeypoints` |
| `tracked_bboxes.csv` | `frame_id,class_id,instance_id,cx,cy,w,h,score` | `CsvBoundingBoxes` |

`instance_id` is the index of the DLC individual the tracklet stitcher assigned, and is stable across frames. `class_id` is always `0`: DLC projects hold a single class.

- **⚠️IMPORTANT⚠️** DeepLabCut has **no notion of a bounding box**. Each box is derived as the tightest one enclosing that individual's detected keypoints — the same derivation PrecisionTrack applies internally through `keypoints_cxcywh` — and its `score` is the mean likelihood of those keypoints. Boxes are therefore systematically tighter than annotated ground-truth boxes, which is worth keeping in mind when reading IoU-based metrics.

## 8) Visualize and evaluate with PrecisionTrack

**Switch back to the PrecisionTrack environment** for this section.

Because `track.py` writes the same file names `configs/tasks/tracking.py` uses, PrecisionTrack's own visualizer reads DLC's results directly. Point `saving_directory` in `configs/user_configs.yaml` at the parent of your `--out-dir`, then:

```bash
cd tools
python visualize.py ../assets/20mice.avi ../work_dir/20mice/dlc_vis.mp4
```

For MOT evaluation, feed `tracked_bboxes.csv` straight to `evaluate_mot` — **keep the default `cxcywh`**. `Evaluator.update` reformats the predictions from `cxcywh` to `xywh` itself; it is the *ground-truth* file that must already be `xywh` (`frame_id,class_id,instance_id,x,y,w,h,score`, as validated by `assert_mot_file_is_ok`).

```python
from precision_track.evaluation.utils.mot import evaluate_mot

evaluate_mot(
    result_path="../work_dir/20mice/tracked_bboxes.csv",     # cxcywh, as written by track.py
    ground_truth_path="<your MOT dataset>/bboxes/val/<video>.csv",   # xywh
    metadata_path="../configs/metadata/mice.py",
    save_path="../work_dir/20mice/mot_evaluation.csv",
)
```

`--bbox-format xywh` exists for the other direction — producing a ground-truth-shaped file, or feeding external MOT tooling that expects the `x,y,w,h` layout. Do not pass it for `result_path`, or every box will be shifted by half its width and height.

## Troubleshooting

- **`No tracked DLC output (_el.h5/_bx.h5/_sk.h5) found`** — `analyze_videos` produced detections but no stitched tracks. Confirm it ran with `auto_track=True` (the default here) and that `--n-tracks` is at least 1.
- **`Cannot open the video file!` during conversion** — DLC could not read a video passed to `--videos`. The project is still created, without a `video_sets` entry; this does not affect training or tracking.
- **`HDF5 ... tables` errors** — `pip install tables` (already in `requirements.txt`).
- **`Image ... appears in more than one split`** — your `train.json` and `val.json` overlap. DLC cannot hold the same image twice; fix the dataset split.
- **CUDA out of memory during training** — lower `--batch-size`.

## Caveats when comparing against PrecisionTrack

- DLC is trained without identities (see section 4), so identity-aware metrics measure the tracklet stitcher alone.
- Bounding boxes are keypoint-derived, not predicted (see section 7).
- DLC does not resize frames during inference. Always pass `--img-size` matching your PrecisionTrack input resolution before comparing throughput.
