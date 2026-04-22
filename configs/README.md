# PrecisionTrack – User Configuration Guide

Welcome! 👋  
This guide explains **how to configure PrecisionTrack** by editing a single file:
👉`./user_configs.yaml/`

---

# Overview

### The **`./user_configs.yaml/` file** have three section:

1. **booleans** → Enable or disable functionalities
2. **training** → Training parameters, directories and paths
3. **tracking** → Tracking parameters
4. **action_recognition** → Action Recognition parameters
5. **group_action_recognition** → Group Action Recognition parameters
6. **validation** → Validation parameters
7. **visualization** → Visualization parameters

**⚠️IMPORTANT⚠️** Unless explicitely specified, make sure the your **paths** are **relative** to the `tools` directory from your **precision_track** GitHub directory.

**⚠️IMPORTANT⚠️** For Windows and WSL users. Paths **outside** of your precision_track's directory will not exists within your docker container. Ensure that all the provided paths are **inside** your precision_track's directory.

---

# 1. Booleans – Enable or disable functionalities

- **<u>pipelined</u>**  
  Runs processes in parallel to make tracking _faster_.
  - Recommended for real-time use.
  - Significantly accelerate the processing speed if your are performing multiple downstream tasks (action recognition and/or re-identification).

- **<u>with_validation</u>**  
  Enables **Re-identification**.
  - Set as `true` only if you have a valid `validation_configuration_file` and want to perform animal re-identification.

- **<u>with_action_recognition</u>**  
  Enables the MART model to recognize animal actions.
  - Set as `true` only if you have trained a MART model (Guides and tutorials on how to do it coming out soon).

- **<u>with_group_action_recognition</u>**  
  Enables the GMART model to recognize animal social actions. **with_action_recognition** also need to be enable to perform group action recognition.

- **<u>with_pose_estimation</u>**  
  Enables full pose (keypoints + skeleton).
  - Set as `false` if you want box-only tracking.
  - Set as `true` only if your COCO formatted dataset (`data_root`) contain keypoints.
  - Mandatory to perform **action recognition** and **group action recognition**

---

# 2. General directories and paths

- **<u>metainfo</u>**  
  A small python file that describes your species: names of keypoints, skeleton shape, etc. Please refer to our [metadata guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/metadata) for more details.

---

# 3. Training parameters, directories and paths

### These parameters tell PrecisionTrack where your dataset is located and how training should run.

- **<u>dataset_name</u>**  
  The label that will appear in logs and created sub-directories under the `precision_track/work_dir/training_runs` and `precision_track/work_dir/testing_runs` directories. Namely, the training logs and testing metrics will be saved there.

- **<u>data_root</u>**  
  Root directory of your **COCO-style dataset**. Expected structure:

```text
  <data_root>/
  ├── annotations/
  │ ├── train.json/
  │ ├── val.json/
  ├── images/
  │ ├── image_1.jpg   NOTE: Images may have any filename.
  │ ├── image_2.jpg
  │ ├── ...
```

**⚠️IMPORTANT⚠️** For Windows and WSL users. Paths **outside** of your precision_track's directory will not exists within your docker container. Ensure that all the provided paths are **inside** your precision_track's directory.

- **<u>resume</u>**  
  Turn ON if you want to continue a stopped training. Turning this on will:
  - Load the checkpoint saved at the `training_checkpoint` path. Therefore, `training_checkpoint` should be the path to the last saved checkpoint from the run you want to resume.
  - Resume the training from this checkpoint.

  **NOTE** Your latest training checkpoint is indicated in the `../<saving_directory>/training_runs/<dataset_name>/last_checkpoint` file.

- **<u>training_checkpoint</u>**  
  Path to a `.pth` file used to initialize training for transfer learning. Therefore, starting your training from a checkpoint strongly improves performance.
  - Can be either a model you already pretrained or our available **AP Checkpoint** (recommended):  
    [Download here](https://drive.google.com/drive/folders/1_U9fDDAW7UYm_xelod9ehrSNFdpYUz0o).
  - For better organization, create a dedicated **checkpoints directory** (e.g., `../checkpoints/`). You may store the AP Checkpoint locally as:  
    `../checkpoints/ap/model_ap.pth`.

- **<u>deploying_directory</u>**  
  Directory where all deployment artifacts are saved after a successful training run. At minimum, a `_DEPLOYED.pth` (which is a lighter copy of the last checkpoint of your training run (`../work_dir/training_runs/<dataset_name>/epoch_300.pth`)) will be saved here. Also, a `_DEPLOYED.onnx` file will be saved if your machine supports FP16 conversion (it should). For more informations about **ONNX** please visit [The following](https://onnx.ai/). Finally, a `.engine` checkpoint will be saved if your machine is **CUDA accelerated**. This last checkpoint is the absolute fastest version of your model. As such, it will be the system's preferred **runtime** whenever it is available. For more informations about **TensorRT Engines** please visit [The following](https://docs.nvidia.com/tensorrt/index.html).
  - We also highly recommend creating a **checkpoints directory** (`../checkpoints/`) in order to organize all your model's weights. We recommend your `deploying_directory` to be inside this **checkpoints directory**. For example, you could set your `deploying_directory` as `../checkpoints/v1/` then train your network. This will tell your PrecisionTrack trainer to first create the `../checkpoints/v1/` directory then automatically save your last training checkpoint to this directory.

  - Please refer to our [checkpoints and hyperparameters](https://github.com/VincentCoulombe/precision_track/tree/main/checkpoints) guide for more details.

- **<u>deploying_sanity_check_img_path</u>**
  Path (this path is relative to your `data_root` directory, not your `tools` directory) to **any** image from your **COCO-style dataset**. This image will be used to ensure that the `.onnx` and the `.engine` checkpoints are accurate. This is only relevant if the [training tool's](https://github.com/VincentCoulombe/precision_track/tree/main/tools) `deploy` option is set to `true`.

- **<u>batch_size</u>**  
  How many images are processed at once.
  - Bigger = faster but requires more VRAM. For example, 24GB GPUs can load batches of 38 images each.

- **<u>wandb_logging</u>**  
  Enables training visualization through Weight & Biases. Please refer to our [Weight & Biases guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/wandb) for setupping instructions.

---

# 4. Tracking parameters

- **<u>saving_directory</u>**
  Specifies where PrecisionTrack will **write all tracking outputs** (e.g., bounding boxes, poses, velocities, actions, etc...). The **visualization tool** also **reads from this directory** when rendering videos.

  **⚠️IMPORTANT⚠️**: The **visualization tool** assumes every file in the `saving_directory` belongs to the **same tracking run**. If the `saving_directory` contains **leftovers from older runs**, you may get **incorrect or mixed visualizations**.

  **Best practice**: use a **fresh** or automatically cleaned **directory** for **each run**.

- **<u>num_subjects</u>**
  Tell PrecisionTrack **how many subjects of each classes are in the scene**.
  - If animals **can enter or leave the scene**, set it to **-1**.
  - Ensure that the classes set here are coherent with those in your **metadata file**.

- **<u>tracking_checkpoint_name</u>**  
  This config allow you to select, by name, a specific checkpoint, from inside your `deploying_directory`, to track with. This checkpoint could be:
  - Your `_DEPLOYED.pth`
  - Your `_DEPLOYED.onnx`
  - Your `.engine`

  If left empty (""), PrecisionTrack will automatically select a tracking checkpoint from your **deploying_directory**. The selection is performed **following this priority** : `.engine` -> `_DEPLOYED.onnx` -> `_DEPLOYED.pth`.

- **<u>output_clustered_features</u>**
  If enabled, will save your clustered features at the following path: `<saving_directory>/features.npy`, which you will then be able to use for further analysis. **Warning** saving the clustered features will SIGNIFICANTLY slow down the tracking process.

- **<u>mot_data_root</u>**
  Path to your MOT dataset. Expected structure:

  ```bash
  <mot_data_root>/
    ├── bboxes/
    │ ├── video1.csv # NOTE: This is your MOT annotations (your bounding bboxes). They can have any name
    │ ├── video2.csv
    │ ├── etc...
    ├── videos/
    │ ├── video1.mp4 # NOTE: Your videos must match their correspondig MOT bboxes files.
    │ ├── video2.avi
    │ ├── etc...
  ```

  This dataset can be used to quantitatively test your tracking performances (using the `test_tracking.py` tool).

  **NOTE**: The [MOT](https://motchallenge.net/) (or Multi-Object tracking) dataset format is very popular in the MOT communauty. As such, there is a lot of reference online on how to build your own. More specifically, our [MICE sequential dataset](https://drive.google.com/drive/folders/1WcDkX-92X6SCgZPAZXFyDc6EGUzU0Onq?usp=drive_link) (which is MOT formatted) could be used a a valid reference point. To create our MICE sequential dataset, we used CVAT and [Julien Audet-Welke's guide](https://github.com/juauw/CVAT_pipeline).

---

# 5. Action Recognition parameters

- **<u>mart_checkpoint_name</u>** Name of the MART checkpoint you want to use to infer animal actions during tracking. Like it is the case for **tracking_checkpoint_name**, the **mart_checkpoint_name** is assumed to be saved inside your `deploying_directory`. This checkpoint could be:
  - Your `_DEPLOYED.pth`
  - Your `_DEPLOYED.onnx`
  - Your `.engine`

- **<u>action_recognition_data_root</u>** Path to your action recognition dataset. Expected structure:

  ```text
  <action_recognition_data_root>/
    ├── bboxes/
    │ ├── train/
    | | ├── video1.csv
    │ ├── val/
    | | ├── video2.csv
    ├── keypoints/
    │ ├── train/
    | | ├── video1.csv
    │ ├── val/
    | | ├── video2.csv
    ├── actions/
    │ ├── train/
    | | ├── video1.csv
    │ ├── val/
    | | ├── video2.csv
    ├── videos/
    | | ├── video1.mp4
    │ ├── val/
    | | ├── video2.avi
  ```

  This structure is very close to the expected **mot_data_root**'s, but it contains training (train) and validaiton (val) splits. It also requires a `./keypoints` and a `./actions` subdirectory. Again, please refer to our [MICE sequential dataset](https://drive.google.com/drive/folders/1WcDkX-92X6SCgZPAZXFyDc6EGUzU0Onq?usp=drive_link) as a valid Action Recognition dataset.

  **NOTE**: We achieved our MOT-styled bounding boxes and keypoints annotations by following [Julien Audet-Welke's guide](https://github.com/juauw/CVAT_pipeline) and our action labels by reformatting annotations obtained through manual labelling on the [BORIS](https://www.boris.unito.it/) software.

- **<u>output_action_recognition_embeddings</u>**
  If enabled, will save the action embeddings produced by your MART model at the following path: `<saving_directory>/action_embeddings.npy`, which you will then be able to use for further analysis. **Warning** saving the clustered features will SIGNIFICANTLY slow down the tracking process.

---

# 6. Group Action Recognition parameters

- **<u>gmart_checkpoint_name</u>** Name of the GMART checkpoint you want to use to infer animal actions during tracking. Like it is the case for **tracking_checkpoint_name**, the **gmart_checkpoint_name** is assumed to be saved inside your `deploying_directory`. This checkpoint could be:
  - Your `_DEPLOYED.pth`
  - Your `_DEPLOYED.onnx`
  - Your `.engine`

**NOTE**: Your **gmart_checkpoint_name** is wrapping your **mart_checkpoint_name**. Therefore, it contains but checkpoints.

**NOTE**: Even if a **gmart_checkpoint_name** is set, no group action reocgnition will takep lace if **with_group_action_recognition** is disabled.

**NOTE**: You will need to add the three following keys to your **metainfo** file in order to properly enable Group Action Recognition:

- `social_actions`
- `null_action`
- `distance_keypoint_pairs`
  Please refer to our [metadata guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/metadata) for more details.

---

# 7. Validation parameters

- **<u>validation_configuration_file</u>**
  Path to your validation config file. In order to keep this config file as lean as possible for the non validation users, I decided to keep the validation-related configs separate. the path by defaut refers to the validation config use to obtain the results in the manuscript.

- **<u>output_appearance_database</u>**
  If enabled, will save the action embeddings produced by your MART model at the following path: `<saving_directory>/appearance_database.npy`, which you will then be able to use for further analysis. **Warning** saving the clustered features will SIGNIFICANTLY slow down the tracking process.

  **NOTE**: You can use the `./tools/visualize_appearances.py` tool to visualize your appearance features offline, meaning once the tracking is completed and your have access to at least the following three files:
  - `<saving_directory>/appearance_database.npy`
  - `<saving_directory>/tracked_bboxes.csv`
  - `<path to your video>`

---

# 8. Visualization parameters

- **<u>display_bounding_boxes</u>**: Render the tracked subject's bounding boxes, given that a `tracked_bboxes.csv` file exists in the `saving_directory`.

- **<u>display_poses</u>**: Render the tracked subject's poses, given that a `tracked_kpts.csv` file exists in the `saving_directory`.

- **<u>display_velocities</u>**: Render the tracked subject's velocities, (in the form of an arrow) given that a `tracked_velocities.csv` file exists in the `saving_directory`.

- **<u>display_species</u>**: Add the tracked subject's predicted species to the subject's label bars, given that a `tracked_bboxes.csv` file exists in the `saving_directory`.

- **<u>display_confidence_scores</u>**: Add the tracked subject's confidence score (which means how confident the subject's detections are) to the subject's label bars, given that a `tracked_bboxes.csv` file exists in the `saving_directory`.

- **<u>display_actions</u>**: Add the tracked subject's predicted actions to the subject's label bars, given that a `tracked_actions.csv` file exists in the `saving_directory`.

- **<u>display_search_zones</u>**: Render the tracked subject's search zones (as described in the manuscript), given that a `stitching_search_areas.csv` file exists in the `saving_directory`.

- **<u>display_validations</u>**: Render the tracked subject's validations (Tailtag detections), given that a `tracked_validations.csv` file exists in the `saving_directory`.

- **<u>display_untracked_detections</u>**: Render detected bounding boxes (will be bright white), given that a `detected_bboxes.csv` file exists in the `saving_directory`.

- **<u>display_predicted_bounding_boxes</u>**: Render where the model think the subject will be the next time it will see it (will be bright white corners), given that a `detected_bboxes.csv` file exists in the `saving_directory`.

---

# 🧪 Tips

- If something breaks:
  1. Read the log, it is pretty verbose, it should tell you exactly what went wrong.
  2. Check the configs (mainly the paths) linked to the process that failed.
  3. Ensure you have understood and formatted your dataset correctly.
  4. If nothing is working, you can contact us directly for help, or open an issue in the repository.

- YAML files are sensitive to indentation — avoid using tabs.
