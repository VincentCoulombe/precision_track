# PrecisionTrack toolkit guide

PrecisionTrack's toolkit is composed of a series of configurable tools that you can run directly from your terminal. Here is a guide on how to use every one of them.

---

## Overview

**⚠️IMPORTANT⚠️** Make sure to run each tool from within precision_track's `tools` directory.

```bash
cd ./tools
```

**⚠️IMPORTANT⚠️** Users wanting to use the package on a CPU only machine (which is very not recommended) might have to first send the following command:

```bash
export ATEN_CPU_CAPABILITY=default
```

Before calling each tool. Else, PyTorch might create a segfault error (internal bug not fixed yet).

**Note**: For Windows and WSL users. You'll first need to change your directory to PrecisionTrack's. To do so, follow these steps:

1. Launch WSL
2. Launch precision_track's Docker container from inside WSL
3. Enter the following command:

   ```bash
   ls
   ```

   You should then see the content of your precision_track's directory. Something like:

   ```bash
   Colab      assets       docker           pytest.ini    setup.py  tracking_predictions.csv
   LICENSE    checkpoints  precision_track  requirements  tests     work_dir
   README.md  configs      pyproject.toml   setup.cfg     tools
   ```

4. Enter the following command to enter the `tools` directory:

   ```bash
   cd ./tools
   ```

- **There are ten tools in the `tools` directory**
  - `train_detection.py` — Orchestrate the training and deployment of **Detection models**.
  - `train_action_recognition.py` — Orchestrate the training and deployment of **MART** (and optionally **GMART**) models.

  - `test_detection.py` — Evaluate the trained **Detection models** on your **COCO-style Dataset**. Report [Pose-Tracking Metrics](https://www.biorxiv.org/content/10.1101/2024.12.26.630112v3). The reports will be logged and also saved in the `precision_track/work_dir/testing_runs/<dataset_name>` directory.

  - `test_tracking.py` — Evaluate the trained **Detection models** on your **MOT-style Benchmark**. Report [CLEAR Metrics](https://link.springer.com/article/10.1155/2008/246309). The reports will be logged and also saved in the `precision_track/work_dir/testing_runs/<dataset_name>` directory.

  - `test_action_recognition.py` — Evaluate the trained **MART** or **GMART models** on your **action recognition dataset**. Report the [standards classification metrics](https://cohere.com/blog/classification-eval-metrics). The reports will be logged and also saved in the `precision_track/work_dir/testing_runs/<dataset_name>` directory.

  - `track.py` — Run tracking on pre‑recorded videos.

  - `batch_track.py` — Batch-track every video of a **MOT-style dataset** to automatically generate **MOT-formatted bounding boxes annotations**.

  - `batch_track_directory.py` — Run the **full configured tracking pipeline** (and every enabled downstream task) on **every video inside a directory**.

  - `visualize.py` — Render tracking + action recognition from MOT outputs.

  - `plot_profiles.py` — Plot per-frame timing profiles from a profiling JSON produced by `track.py --profile=true`.

- **Configuration**
  - Via your [user configuration file](https://github.com/VincentCoulombe/precision_track/tree/main/configs)
  - Or, with a **local web UI** that validates your settings and launches these tools for you:

    ```bash
    pip install -r ../requirements/web.txt
    python -m web_ui   # from the precision_track root; opens http://127.0.0.1:8000
    ```

    See [`web_ui/README.md`](../web_ui/README.md) for details.

---

## 1) train_detection.py

- **Purpose:** Train Detection models.

- **Inputs:**:
  - `--format_dataset`. True to resize and format your COCO-style dataset (accelerate the model's training significantly), False otherwise. Default to True.
  - `--test`. True to automatically launch the `test_detection.py` tool after the training run, False otherwise. Default to True.
  - `--deploy`. True to deploy the trained model and generate optimized runtime checkpoints in your `<deploying_directory>` directory, False otherwise. Default to True.
  - `--calibrate`. True to calibrate and generate or update the `hyperparams.json` file in your `<deploying_directory>` directory, False otherwise. Default to True.
  - `--optimize_hyperparams` True to optimize your tracking hyperparameters and generate a `hyperparams.json` file in your `<deploying_directory>` directory, False otherwise. Default to True.

- **Outputs:** The training log as well as the most performant and the last checkpoints will be saved in the `precision_track/work_dir/training_runs/<dataset_name>` directory. A `hyperparams.json` file and `DEPLOYED` checkpoints will be saved in your `<deploying_directory>`. Testing metrics will be saved in the `precision_track/work_dir/testing_runs/<dataset_name>` directory.
  - Please refer to our [checkpoints and hyperparameters](https://github.com/VincentCoulombe/precision_track/tree/main/checkpoints) guide for more details.

- **Examples**

  ```bash
  python train_detection.py --test=true --deploy=true --calibrate=true --optimize_hyperparams=false --format_dataset=true
  ```

---

## 2) train_action_recognition.py

- **Purpose:** Train Action Recognition models. Trains **MART** first, then automatically trains **GMART** if `with_group_action_recognition` is enabled in your settings.

- **Inputs:**:
  - `--test`. True to automatically launch the `test_action_recognition.py` tool after the training run, False otherwise. Default to True.
  - `--deploy`. True to deploy the trained model and generate optimized runtime checkpoints in your `<deploying_directory>` directory, False otherwise. Default to True.
  - `--config`. Path to the training config file. Default to `../configs/tasks/training_action_recognition.py`.

- **Outputs:** The training log as well as the most performant and the last checkpoints will be saved in the `precision_track/work_dir/training_runs/<dataset_name>` directory. `DEPLOYED` checkpoints will be saved in your `<deploying_directory>`. Testing metrics will be saved in the `precision_track/work_dir/testing_runs/<dataset_name>` directory.
  - When `with_group_action_recognition` is enabled, a second GMART training run follows automatically, producing an additional `gmart_DEPLOYED` checkpoint.

- **Examples**

  ```bash
  python train_action_recognition.py --deploy=true
  ```

---

## 3) test_detection.py

- **Purpose:** Evaluate Detection models (trained checkpoints) on val/test splits.

- **Outputs:** The metrics will be saved in the `precision_track/work_dir/testing_runs/<dataset_name>` directory.

- **Examples**

  ```bash
  python test_detection.py
  ```

---

## 4) test_tracking.py

- **Purpose:** Evaluate your PrecisionTracker on labelled MOT benchmarks.

- **Outputs:** The metrics will be saved in the `precision_track/work_dir/testing_runs/<dataset_name>` directory.

- **Examples**

  ```bash
  python test_tracking.py
  ```

---

## 5) test_action_recognition.py

- **Purpose:** Evaluate your MART model on val/test splits.

- **Outputs:** The metrics will be saved in the `precision_track/work_dir/testing_runs/<dataset_name>` directory.

- **Examples**

  ```bash
  python test_action_recognition.py
  ```

---

## 6) track.py — run tracking on videos

- **Purpose:** run detector/pose/action heads + association on pre‑recorded media.
- **Inputs:**
  - `video` (positional) — path to the recording file.
  - `--profile` — set to `true` to record per-frame timing data. Saves a `profile_<timestamp>.json` to your `saving_directory`. Not supported in pipelined mode. Default to `false`.
- **Outputs:** All the available outputs will be saved at the defined `work_dir` from the settings. Heres a list of all the possible outputs:
  - `tracked_bboxes.csv`: Contains the MOT formatted bounding boxes of all the tracked subjects over the whole recording.
  - `tracked_identities.csv`: Same as `tracked_bboxes.csv`, but the third column holds the validated `instances_identity` (e.g. an appearance identity name or an ArUco tag number) instead of the tracker-assigned `instance_id`. The identity is carried forward from the most recent confirmation, so a subject's rows stay empty until its identity is first confirmed. Only available when a validation/ReID algorithm is used.
  - `detected_bboxes.csv`: Contains the MOT formatted bounding boxes of all the detected subjects for every frame of the recording.
  - `tracked_kpts.csv`: Contains the MOT formatted keypoints of all the tracked subjects over the whole recording.
  - `tracked_velocities.csv`: Contains the MOT formatted velocities of all the tracked subjects over the whole recording.
  - `stitching_search_areas.csv`: Contains the MOT formatted search areas over the whole recording. Only available when a stitching algorithm is used.
  - `tracked_validations.csv`: Contains the MOT formatted validations over the whole recording. Only available when a validation/ReID algorithm is used.
  - `tracked_corrections.csv`: Contains the MOT formatted corrections over the whole recording. Only available when a validation/ReID algorithm is used.
  - `tracked_actions.csv`: Contains the MOT formatted actions over the whole recording. Only available when an action recognition algorithm is used.
  - `timestamps.csv`: Contains a mapping between the frame ids and the actual time the video has been running (in seconds).
- **Examples**
  ```bash
  python track.py video data/sample.mp4
  python track.py video data/sample.avi
  ```

---

## 7) batch_track.py — auto-generate MOT annotations for a dataset

- **Purpose:** Batch-track an entire **MOT-style dataset** to automatically produce **MOT-formatted bounding boxes annotations**. Where `track.py` runs on a single video and emits the full set of outputs into your `work_dir`, `batch_track.py` walks every video of a dataset and writes a single, MOT-compatible `tracked_bboxes` file per video, named after the video itself. This is the fastest way to bootstrap (or pre-annotate) a dataset that so far only contains videos: the resulting files are ready to be reviewed/corrected and reused as ground truth by `test_tracking.py`.

- **Inputs:**
  - `--force` — set to `true` to re-process and **overwrite** videos that already have a bounding boxes file. Default to `false`.

  The dataset root is read from the `mot_data_root` value of your [user configuration file](https://github.com/VincentCoulombe/precision_track/tree/main/configs). The expected layout is the standard MOT structure:

  ```text
  mot_data_root/
  ├── videos/{train,val}/<stem>.{mp4,avi,mov,mkv,mpg,mpeg}
  └── bboxes/{train,val}/<stem>.csv      # generated here (next to any ground truth)
  ```

- **Outputs:** For every video found under `videos/train` and `videos/val`, a `bboxes/<split>/<stem>.csv` file containing the **MOT-formatted bounding boxes** of all the tracked subjects over the whole recording. The columns are `frame_id, class_id, instance_id, x, y, w, h, score` (top-left `xywh`), matching the ground-truth schema expected by `test_tracking.py`.
  - **Resumable / non-destructive:** if a `bboxes/<split>/<stem>.csv` already exists (a previous run, or hand-labelled ground truth), the video is **skipped with a warning** so existing annotations are never overwritten. Use `--force=true` to regenerate them.
  - Each video is tracked independently, so `instance_id`s restart from the first subject in every file.

- **Examples**

  ```bash
  <!-- Annotate the dataset defined in your user configs. -->
  python batch_track.py

  <!-- Re-generate every annotation, overwriting existing files. -->
  python batch_track.py --force=true
  ```

---

## 8) batch_track_directory.py — track every video in a directory

- **Purpose:** Run the **full configured tracking pipeline** on **every video inside a directory**.

- **Inputs:**
  - `directory` (positional) — path to the directory containing the videos to track. Recognized extensions: `.mp4, .avi, .mov, .mkv, .mpg, .mpeg`.
  - `--recursive` — set to `true` to also track videos in sub-directories. Default to `false`.
  - `--restart-tracker-instance` — set to `false` to have **one single tracker instance** track the whole directory so that **track IDs stay consistent across the videos**. Set to `true` (the default) to build a **fresh tracker for every video** (IDs restart at `1` per video, current behaviour).

- **Outputs:** For each video, the same set of files `track.py` produces, written to a **per-video sub-directory** of your `saving_directory`: `<saving_directory>/<video_name>/`. Keeping each video in its own sub-directory avoids the mixed-output pitfall described in the [Tracking parameters](https://github.com/VincentCoulombe/precision_track/tree/main/configs) guide and keeps every run cleanly separated for the **visualization tool**.

- **⚠️IMPORTANT⚠️ — about `--restart-tracker-instance=false`:** This mode is meant for **one long recording that was split into consecutive sub-videos**. It takes for granted that every video in the directory is a **frame-by-frame follow-up of the previous one** (the tracker carries its IDs, motion and appearance state across the video boundaries, treating the whole directory as a single continuous recording). **If the videos are not consecutive fragments of the same recording, the results can be really bad** — tracks will be mismatched across videos. Leave it at the default (`true`) unless your directory really is one recording chopped into parts.

- **Known limitations of `--restart-tracker-instance=false`:**
  - **All videos must share the same resolution.** A differing resolution raises a hard error when tracking runs pipelined (the default on machines with 3+ physical CPU cores) and a warning otherwise.
  - **Track IDs keep counting up across videos.** Since the ID counters are never reset, the first ID of the second video continues from where the first video left off (e.g. it may start at `7`, not `1`). This is intended so a subject keeps the same ID across the whole recording.
  - **Offline correction refinement runs per video without cross-video identity state** when tracking is pipelined (same behaviour as a single pipelined `track.py` run).

- **Examples**

  ```bash
  <!-- Track every video directly inside the directory. -->
  python batch_track_directory.py ../recordings/session_1/

  <!-- Track every video in the directory and all its sub-directories. -->
  python batch_track_directory.py ../recordings/ --recursive=true

  <!-- One long recording split into parts: keep IDs consistent across the sub-videos. -->
  python batch_track_directory.py ../recordings/session_1/ --restart-tracker-instance=false
  ```

---

## 9) visualize.py — render tracking & actions

- **Purpose:** Turn the available MOT outputs, in the defined `work_dir` from the settings, into annotated videos. The visuals are completely configurable in the "Visualization" section of the `tasks/tracking.py` setting file.
- **Inputs:** `source` (path to the recording file) `sink` (path to the annotated video file)
- **Outputs:** An annotated video will be saved at the provided sink path.
- **Examples**
  ```bash
  python visualize.py source data/sample.mp4 sink data/annotated_data_sample.mp4
  ```

## 10) plot_profiles.py — visualize timing profiles

- **Purpose:** Turn a profiling JSON produced by `track.py --profile=true` into annotated timing charts. Highlights JiT (just-in-time compilation) spikes and statistical outlier peaks.
- **Inputs:**
  - `json_file` (positional) — path to the `profile_<timestamp>.json` file in your `saving_directory`.
  - `--std-threshold` — number of standard deviations above the mean to flag a frame as a peak. Default to `2.0`.
- **Outputs:** PNG plots saved alongside the input JSON:
  - `detection_profile.png` — batch-level detection timing.
  - `tracking_profile.png` — per-frame tracking timing (one subplot per tracked component).
- **Examples**

  ```bash
  python plot_profiles.py ../work_dir/profile_20250101_120000.json
  python plot_profiles.py ../work_dir/profile_20250101_120000.json --std-threshold=3.0
  ```

---

## Example workflows

- **Train → Test → Deploy**

  ```bash
  <!-- Train, Test and Deploy Detection model. -->
  python train_detection.py --test=true --calibrate=true --deploy=true --optimize_hyperparams=false
  ```

- **Track → Visualize**

  ```bash
  <!-- Track on a specified video and the render the results. -->
  python track.py video <your video name>.mp4
  python visualize.py source <your video name>.mp4 sink annotated_<your video name>.mp4
  ```

- **Auto-annotate a dataset → Benchmark**

  ```bash
  <!-- Generate MOT-formatted annotations for every video of the dataset, then benchmark on it. -->
  python batch_track.py
  python test_tracking.py
  ```

- **Track → Profile → Plot**

  ```bash
  <!-- Track with profiling enabled, then plot the timing results. -->
  python track.py video <your video name>.mp4 --profile=true
  python plot_profiles.py <saving_directory>/profile_<timestamp>.json
  ```
