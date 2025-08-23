# PrecisionTrack toolkit guide

PrecisionTrack's toolkit is composed of a series of configurable applications that you can run directly from your terminal. Here is a guide on how to use every one of them.

---

## Overview

- **Five applications**
  - `train.py` — Orchestrate the training of Detection, Feature Extraction and Action recognition models.
  - `test.py` — Evaluate the trained models, reports the metrics described in our publication.
  - `deploy.py` — Package/Export models to ONNX and TensorRT engines. Also automatically optimizes various tracking hyperparameters. The optimal hyperparameters are then saved in .json format.
  - `track.py` — Run tracking on pre‑recorded videos
  - `visualize.py` — Render tracking + action recognition from MOT outputs
- **Configuration**
  - Single source of truth via config file. Once the settings files are configured, the integration is seamless.
  - see our [Settings & Configuration Guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs)

---

## 1) train.py

- **Purpose:** Detection, Feature Extraction and Action recognition models.

- **Inputs:** The desired task's configuration file:

  - "../configs/tasks/training_detection.py"
  - "../configs/tasks/training_feature_extraction.py"
  - "../configs/tasks/training_action_recognition.py"

- **Outputs:** The training log as well as the most performant checkpoint will be saved at the defined `training_work_dir` from the settings.

- **Examples**

  ```bash
  python train.py --config ../configs/tasks/training_detection.py
  ```

---

## 2) test.py

- **Purpose:** Evaluate Detection, Feature Extraction and Action recognition models trained checkpoints on val/test splits.

- **Inputs:** The desired task's configuration file:

  - "../configs/tasks/testing_detection.py"
  - "../configs/tasks/testing_feature_extraction.py"
  - "../configs/tasks/testing_action_recognition.py"

- **Outputs:** The metrics will be saved at the defined `testing_work_dir` from the settings.

- **Examples**

  ```bash
  python test.py --config ../configs/tasks/testing_detection.py
  ```

---

## 3) deploy.py — deploy/export the models

- **Purpose:** Optimizes and deploys all the configured models as well as the tracking hyperparameters.
- **Outputs:** Deployed Python, ONNX and TensorRT as well as the hyperparameters.json file will be saved at defined `deployed_directory` from the settings.
- **Examples**
  ```bash
  python deploy.py
  ```

---

## 4) track.py — run tracking on videos

- **Purpose:** run detector/pose/action heads + association on pre‑recorded media.
- **Inputs:** `--video PATH` (path to the recording file)
- **Outputs:** All the available outputs will be saved at the defined `work_dir` from the settings. Heres a list of all the possible outputs:
  - `bboxes.csv`: Contains the MOT formatted bounding boxes of all the tracked subjects over the whole recording.
  - `kpts.csv`: Contains the MOT formatted keypoints of all the tracked subjects over the whole recording.
  - `velocities.csv`: Contains the MOT formatted velocities of all the tracked subjects over the whole recording.
  - `search_areas.csv`: Contains the MOT formatted search areas over the whole recording. Only available when a stitching algorithm is used when tracking.
  - `validations.csv`: Contains the MOT formatted validations over the whole recording. Only available when a validation/ReID algorithm is used when tracking.
  - `corrections.csv`: Contains the MOT formatted corrections over the whole recording. Only available when a validation/ReID algorithm is used when tracking.
  - `actions.csv`: Contains the MOT formatted actions over the whole recording. Only available when an action recognition algorithm is used when tracking.
- **Examples**
  ```bash
  python track.py --video data/sample.mp4
  python track.py --video data/sample.avi
  ```

---

## 5) visualize.py — render tracking & actions

- **Purpose:** Turn the available MOT outputs, in the defined `work_dir` from the settings, into annotated videos. The visuals are completely configurable in the "Visualization" section of the `tasks/tracking.py` setting file.
- **Inputs:** `--source PATH` (path to the recording file) `--sink PATH` (path to the annotated video file)
- **Outputs:** An annotated video will be saved at the provided sink path.
- **Examples**
  ```bash
  python visualize.py --source data/sample.mp4 --sink data/annotated_data_sample.mp4
  ```

## Example workflows

- **Train → Test → Deploy**

  ```bash
  <!-- First, train and test a detection model -->
  python train.py --config ../configs/tasks/training_detection.py
  python test.py --config ../configs/tasks/testing_detection.py

  <!-- Second, train and test a feature extraction model -->
  python train.py --config ../configs/tasks/training_feature_extraction.py
  python test.py --config ../configs/tasks/testing_feature_extraction.py

  <!-- Third, deploy your models -->
  python deploy.py
  ```

- **Track → Visualize**

  ```bash
  python track.py --video last_experiment.mp4
  python visualize.py --source last_experiment.mp4 --sink annotated_last_experiment.mp4
  ```
