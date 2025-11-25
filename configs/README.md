# PrecisionTrack – User Configuration Guide

Welcome! 👋  
This guide explains **how to configure PrecisionTrack** by editing a single file:
👉`./user_configs.yaml/`

---

# Overview

### The **`./user_configs.yaml/` file** have three section:

1. **booleans** → Turn features ON or OFF
2. **training** → Training parameters, directories and paths
3. **tracking** → Tracking parameters

---

# 1. Booleans – Turn Features On/Off

### Use this section to enable or disable functionalities.

- **pipelined**  
  Runs processes in parallel to make tracking _faster_.

  - Recommended for real-time use.

- **with_validation**  
  Enables **Tailtag/ArUco tag re-identification**.

  - Turn ON only if your animals wear Tailtags.

- **with_action_recognition**  
  Enables the MART model to recognize animal actions.

  - Turn ON only if you have trained a MART model (Guides and tutorials on how to do it coming out soon).

- **with_pose_estimation**  
  Enables full pose (keypoints + skeleton).
  - Turn OFF if you want box-only tracking.
  - Turn ON only if you have trained your PrecisionTrack on a COCO formatted dataset containing keypoints.

---

# 2. Training parameters, directories and paths

### These parameters tell PrecisionTrack where your dataset is located and how training should run.

- **metainfo**  
  A small python file that describes your species: names of keypoints, skeleton shape, etc. Please refer to our [metadata guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/metadata) for more details.

- **dataset_name**  
  The label that will appear in logs and created sub-directories under the `precision_track/work_dir/` directory. Namely, the training logs and testing metrics will be saved there.

- **data_root**  
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

- **resume**  
  Turn ON if you want to continue a stopped training.

- **training_checkpoint**  
  Path to a `.pth` file used to initialize training.

  - Can be a pretrained model or a checkpoint from a previous run.
  - If you do not have a checkpoint from a previous run, we highly recommend downloading our [AP Checkpoint](https://drive.google.com/drive/folders/1_U9fDDAW7UYm_xelod9ehrSNFdpYUz0o) as your starting checkpoint.

- **testing_checkpoint**  
  Path to a `.pth` file used to testing both the detections and (optionally) the tracking performances of your model. By default, it takes the last saved checkpoint of the training run (epoch_300.pth)

  - The checkpoint **does not** have to be in your `precision_track/work_dir/training_runs/` directory. In fact, we suggest not to leave it there as it might get override by futur training runs.
  - As a good practice, I suggest you keep a collection of well named checkpoints. This way, you will be able to perform regression tests and compare the performances of your models.

- **deploying_directory**  
  Directory where all deployment artifacts are saved after a successful training run. At minimum, a `_DEPLOYED.pth` (which is a lighter copy of your **testing_checkpoint**) will be provided. Also, a `_DEPLOYED.onnx` file will be provided if your machine supports FP16 conversion (it should). For more informations about **ONNX** please visit [The following](https://onnx.ai/). Finally, a `.engine` checkpoint will be provided if your machine is **CUDA accelerated**. This last checkpoint is the absolute fastest version of your model. As such, it will be the system's preferred **runtime** whenever it is available. For more informations about **TensorRT Engines** please visit [The following](https://docs.nvidia.com/tensorrt/index.html).

- **deploying_sanity_check_img_path**
  path (relative to **data_root**) to **any** image from your **COCO-style dataset**. This image will be used to ensure that the `.onnx` and the `.engine` checkpoints are accurate.

- **batch_size**  
  How many images are processed at once.

  - Bigger = faster but requires more VRAM. For example, 24GB GPUs can load batches of 38 images each.

- **wandb_logging**  
  Enables training visualization through Weight & Biases. Please refer to our [Weight & Biases guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/wandb) for setupping instructions.

---

# 3. Tracking parameters

- **num_subjects**
  Tell PrecisionTrack **how many subjects of each classes are in the scene**

  - If animals **can enter or leave the scene**, set it to **-1**.
  - Ensure that the classes set here are coherent with those in your **metadata file**.

- **tracking_checkpoint**  
  This config allow you to select a specific checkpoint to track with. This checkpoint could be:

  - Your **testing_checkpoint**
  - Your `_DEPLOYED.pth`
  - Your `_DEPLOYED.onnx`
  - Your `.engine`

  If left empty (""), PrecisionTrack will automatically select a tracking checkpoint from your **deploying_directory**. The selection is performed **following this priority** : `.engine` -> `_DEPLOYED.onnx` -> `_DEPLOYED.pth`.

# 🧪 Tips

- If something breaks:

  1. Read the log, it is pretty verbose, it should tell you exactly what went wrong.
  2. Check the configs (mainly the paths) linked to the process that failed.
  3. Ensure you have understood and formatted your dataset correctly.
  4. If nothing is working, you can contact us directly for help, or open an issue in the repository.

- YAML files are sensitive to indentation — avoid using tabs.
