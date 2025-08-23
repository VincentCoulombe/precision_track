# PrecisionTrack Configuration – Parameter Guide ✨

This guide helps you **parametrize PrecisionTrack for any custom animal‑tracking project**.
Follow the step‑by‑step checklist, then consult each section for detailed explanations of every flag.

---

## 📦 Step‑by‑step guide 🚶‍♂️🛠️📁

1. **Create a folder** under `./settings/` to hold your dataset‑specific config and metadata.
1. **Copy an existing settings file** (e.g. `mice.py` or `ap.py`) into your the same folder and rename it `path_to_your_custom_dataset_settings.py`.
   - Copy‑pasting saves time: you only tweak what differs from the template.
   - Parameters marked **Keep, but do not change unless you know what you are doing** should stay at their default values – they’re universal.

---

---

## 🗂 Base configuration

| Parameter | Meaning                                                                                                        |
| --------- | -------------------------------------------------------------------------------------------------------------- |
| `_base_`  | Path to the **parent** config inherited by this file. Keeps shared defaults in one place – just keep it as is. |

---

## ⚙️ 1. Common

### Metadata & experiment tracking

- **`metainfo`**: Python file holding your dataset meta‑information (keypoint names, skeleton, class labels). Please refer to our [metadata guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/metadata) for setupping instructions
- **`wandb_logging`**: `True/False` – Toggles Weights & Biases experiment tracking. OPTIONAL, please refer to our [Weight & Biases guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/wandb) for setupping instructions

---

## 🔍 2. Detection Configuration

- **`with_pose_estimation`**: If `True`, automatically configure the model to train, test, deploy, track and visualize with keypoints. Without considering keypoints if `False`.

### 2.0 Architecture

- **`widen_factor`** / **`deepen_factor`**: Multipliers that scale the width (channels) and depth (layers) of the detection model's architecture. **Keep, but do not change unless you know what you are doing.**

### 2.1 Training

- **`data_root`**: The root folder of your COCO styled dataset. This directory should include training and validation annotations (.json files) as well as images.
- **`training_work_dir`**: Where training logs, checkpoints, and visualisations will be written.
- **`resume`**: Resume training from the last checkpoint if it exists. If you want to resume training, flag it as True and change the default **training_checkpoint** value for the one you wish to resume at.
- **`training_checkpoint`**: Path to a starting weight file (pre‑training or previous run).

#### Data & augmentation

- **`input_size`**: `(H, W)` pixels after resize. All images are letter‑boxed to this shape. **Keep, but do not change unless you know what you are doing.**
- **`pad_value`**: Pixel value used when padding (114 ≈ ImageNet mean × 255). **Keep, but do not change unless you know what you are doing.**

#### Optimiser & schedule

| Parameter              | Purpose                    | Keep default?                                                    |
| ---------------------- | -------------------------- | ---------------------------------------------------------------- |
| `base_lr`              | Initial learning rate      | ✅                                                               |
| `batch_size`           | Global batch size          | Adjust as high as you can without being Out Of Memory (OOM).     |
| `weight_decay`         | L2 regularisation          | ✅                                                               |
| `ema_momentum`         | EMA decay                  | ✅                                                               |
| `num_epochs`           | Total epochs               | ✅                                                               |
| `num_epochs_pipeline1` | Epochs of stage‑1 training | ✅                                                               |
| `warmup_epochs`        | LR warm‑up length          | ✅                                                               |
| `val_interval`         | Eval every _N_ epochs      | Adjust regarding your preference. The default value is adequate. |

#### Split paths

- **`training_anns_path`** / **`training_imgs_path`**: The path to your COCO training .json file and folder containing training images.
- **`validation_anns_path`** / **`validation_imgs_path`**: The path to your COCO validation .json file and folder containing training images.

### 2.2 Testing

- **`testing_work_dir`**: Where testing logs and metrics will be written.
- **`testing_checkpoint`**: The model weights you wish to use for testing.
- **`half_precision`**: Activates FP16/AMP for faster inference. **Keep, but do not change unless you know what you are doing.**
- **`testing_anns_path`** / **`testing_imgs_path`**: The path to your COCO testing .json file and folder containing training images.
- **`testing_output_file`**: A summary of the detection testing metrics will be written there.

### 2.3 Calibration

- **`calibration_output_dir`**: Where calibration logs and metrics will be written.

### 2.4 Feature Extraction

| Parameter                | Purpose                                  | Keep default?                                                |
| ------------------------ | ---------------------------------------- | ------------------------------------------------------------ |
| `fe_base_lr`             | Feature‑extractor LR                     | ✅                                                           |
| `fe_batch_size`          | Batch size                               | Adjust as high as you can without being Out Of Memory (OOM). |
| `fe_weight_decay`        | Regularisation                           | ✅                                                           |
| `fe_num_epochs`          | Training epochs                          | ✅                                                           |
| `fe_val_interval`        | Eval every _N_ epochs                    | ✅                                                           |
| `fe_training_checkpoint` | Auto‑fills with last detector checkpoint | n/a (ajusts dynamically)                                     |

### 2.5 Deployment

- **`sanity_check_img`**: Quick smoke‑test image. Pick any image from your dataset.
- **`deployment_device`**: Target hardware (`"cuda"`, `"cpu"`). **Keep, but do not change unless you know what you are doing.**
- **`deployed_directory`** / **`deployed_name`**: Folder & filename of the exported detection model (ONNX, TensorRT, etc.).

---

## 🎯 3. Tracking

### 3.0 Runtime setup

- **`tracking_checkpoint`**: Detection's model (either deployed or not) that will actually be used _inside_ the tracker.
- **`pipelined`**: If `True`, detection & association run asynchronously. Speedup the inference.
- **`tracking_batch_size`**: Frames processed per batch. Speedup the inference.
- **`num_tentatives`**: Consecutive hits before a new track is confirmed.
- **`nb_frames_retain`**: Frames to keep a _lost_ track before deletion.
- **`with_validation`**: Enables Tailtag system re-identification. **Requires Tailtags on the frame.**
- **`with_action_recognition`**: Adds an action‑recognition pass to every track.

### 3.1 Stitching & identity management

- **`stitching_algorithm`**: Dict that selects and configures the multi‑view stitching backend.
  - `capped_classes` A dictionnary containing the classes you want to cap and their corresponding values.
  - `beta`, `match_thr` etc. fine‑tune matching cost and thresholds. Can be adjusted for better performances. You can actually visualize the search-zones with the visualization tool.

### 3.2 Validation (optional)

When `with_validation = True`:

- **`validator`**: Dictionary describing ArUco detection hyper‑parameters. They can be tuned for better performances. Please refer to the [Tailtag publication](https://www.biorxiv.org/content/10.1101/2024.11.07.622536v1) for details.
  - `valid_tags`: the list of the ArUco tag ids visible in the recordings.
  - `timeout_after`: Max seconds per frame before giving up tag detection. **Keep, but do not change unless you know what you are doing.**

### 3.3 Threshold tuning utility

- **`low_thr_range`**, **`high_thr_range`**, **`init_thr_range`**: Candidate grids scanned during tracking hyper‑parameter search. **Keep, but do not change unless you know what you are doing.**

### 3.4 Benchmark testing

- **`hyperparams`**: JSON file storing the tuned thresholds. The file is generated during the ./tools.deploy.py script execution.
- **`low_thr`**, **`high_thr`**, **`init_thr`**: Thresholds used for testing the tracking system. **Keep, but do not change unless you know what you are doing.**
- **`testing_video_paths`** / **`testing_gt_paths`**: Paths to the tracking testing videos and MOT formatted ground truth files.

---

## 🎬 4. Action Recognition

### 4.0 Runtime (inference)

- **`mart_checkpoint`**: Path to your MART checkpoint.
- **`inference_resolution`**: `(H, W)` The expected resolution of the inference's recordings.
- **`block_size`**: Number of frames per temporal block. **Keep, but do not change unless you know what you are doing.**
- **`n_embd_*`**: Embedding dimensions for dynamics, pose, and fused features. **Keep, but do not change unless you know what you are doing.**
- **`assigner`**: Controls track memory length for feature blocks. **Keep, but do not change unless you know what you are doing.**

If `with_action_recognition = True` additional keys are required:

- **`action_recognition_input_names`** / **`action_recognition_output_names`**: ONNX IO tensor names. **Keep, but do not change unless you know what you are doing.**
- **`analyzer`**: Full backend description (pre/post‑processors, runtime, labels). **Keep, but do not change unless you know what you are doing.**

### 4.1 Training flags

- **Learning schedule**: `action_recognition_base_lr`, `weight_decay`, `dropout`, `num_iter`, `warmup_iter`, `val_interval`. **Keep, but do not change unless you know what you are doing.**
- **Dataset root**: `action_recognition_data_root`. The root folder of your MOT styled dataset. This directory should include MOT training and validation annotations (.csv files) as well as videos.
- **Train split lists**: `*_train_sequences`, `*_train_bboxes_gt_paths`, `*_train_keypoints_gt_paths`, `*_train_actions_gt_paths`. Path to your MOT training videos, bounding boxes, keypoints and actions.
- **Val split lists**: `*_val_sequences`, etc. . Path to your MOT validation videos, bounding boxes, keypoints and actions.

### 4.2 Testing

- **`mart_testing_checkpoint`**: Path to the MART's checkpoint you wish to test.
- **Test split lists**: `*_test_sequences`, etc. . Path to your MOT testing videos, bounding boxes, keypoints and actions.

### 4.3 Deployment

- **`mart_deployed_directory`** / **`mart_deployed_name`**: MART's final exported model location.

---
