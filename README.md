<div align="center">
  <img width="50%" src="https://raw.githubusercontent.com/VincentCoulombe/precision_track/main/assets/logo.png"/>
  <div>&nbsp;</div>

<!--- TODO ajouter badge vers publication-->

[![Tests](https://github.com/VincentCoulombe/precision_track/actions/workflows/tests.yaml/badge.svg)](https://github.com/VincentCoulombe/precision_track/actions/workflows/tests.yaml)
[![Formating](https://github.com/VincentCoulombe/precision_track/actions/workflows/formatting.yaml/badge.svg)](https://github.com/VincentCoulombe/precision_track/actions/workflows/formatting.yaml)
[![flake8](https://github.com/VincentCoulombe/precision_track/actions/workflows/flake8.yaml/badge.svg)](https://github.com/VincentCoulombe/precision_track/actions/workflows/flake8.yaml)
[![license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/VincentCoulombe/precision_track/tree/main/blob/pose/LICENSE)<!--- TODO modifier link license-->

<!---
TODO Ajouter mes tags de isitmaintained lorsque le repo sera publique

[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)
-->

<div align="center">
  <img src="https://raw.githubusercontent.com/VincentCoulombe/precision_track/main/assets/ap_visuals.png" width="75%"/>
</div>

</div>

PrecisionTrack is a real-time, online, multi-animal tracking system. It can be extended such as with our provided Tailtags validation plugin to track animals over extended periods.
Furthermore, we provide built-in individual action recognition and group-level social behaviour analysis, enabling behavioral and social dynamics analysis at scale.

## Companion repositories — train & export custom models

PrecisionTrack currently includes two companion reporitories:

### [precision_track-detection](https://github.com/VincentCoulombe/precision_track-detection) — track using Ultralytics YOLO detectors

- **What it does:** trains an [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) detection and (optionally **pose-estimation**) model on your PrecisionTrack-style COCO dataset + `metadata.py` file, then exports to model into a format PrecisionTrack supports **(ONNX (`.onnx` files) & TensorRT (`.engine` files))** This will allow your to use an Ultralytics detector instead of our own.

- **How it fits in:** drop the exported `.onnx`/`.engine` into your `deploying_directory` (or set `tracking_checkpoint_name`) — it is picked up automatically.

- **NOTE**: Ultralytics detectors emits no appearance features, so **action recognition is unavailable** while using them (PrecisionTrack logs a warning).

- **How to use & configure:**

  ```bash
  git clone https://github.com/VincentCoulombe/precision_track-detection
  cd precision_track-detection
  pip install -e .
  # edit configs/user_configs.yaml (model, coco_dataset_root, metadata), then:
  cd tools && python train_test_deploy.py     # -> checkpoints/<name>.onnx (+ .engine)
  ```

### [precision_track-ReID](https://github.com/VincentCoulombe/precision_track-ReID) — track using appearance-based subject re-identification

- **What it does:** trains an **appearance re-identification model** (built on [wildlife-tools](https://github.com/WildlifeDatasets/wildlife-tools); MegaDescriptor/CLIP backbones) from your MOT-dataset crops, then exports an **ONNX (`.onnx` files)**.

- **How it fits in:** point `re_identificator.checkpoint` in [`configs/settings/validation/appearance.yaml`](https://github.com/VincentCoulombe/precision_track/tree/main/configs/settings/validation) at the exported ONNX and set `with_validation: true`. See the [validation configuration guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/settings/validation) for details.

- **How to use & configure:**

  ```bash
  git clone https://github.com/VincentCoulombe/precision_track-ReID
  cd precision_track-ReID
  pip install -e .
  python tools/create_crops_dataset.py        # build the crops dataset from your MOT data
  # edit configs/user_configs.yaml, then:
  python tools/train_test_deploy.py           # -> *_DEPLOYED.onnx
  ```

> **🆕 New: a point-and-click Web UI** — configure PrecisionTrack and run every tool from your browser, with no YAML editing and no CLI flags to remember.

<div align="center">
  <img width="80%" src="https://raw.githubusercontent.com/VincentCoulombe/precision_track/main/assets/web_ui.png"/>
</div>

See the [Web UI](#web-ui) section to get started.

## Demos

<div align="center">

  <div style="margin-bottom:3em;">
    <p style="margin:0; font-size:1.2em; font-weight:bold;">
      A clip from the <a href="https://github.com/VincentCoulombe/precision_track/main/assets/full_clip_slow_logo.mp4">multi-species demo</a>.
    </p>
    <img width="60%" src="https://raw.githubusercontent.com/VincentCoulombe/precision_track/main/assets/AP.gif"/>
  </div>

  <div style="margin-bottom:3em;">
    <p style="margin:0; font-size:1.2em; font-weight:bold;">
      A clip from the <a href="https://github.com/VincentCoulombe/precision_track/main/assets/MICE.mp4">PrecisionTrack demo</a>.
    </p>
    <img width="60%" src="https://raw.githubusercontent.com/VincentCoulombe/precision_track/main/assets/MICE.gif"/>
  </div>

  <div style="margin-bottom:3em;">
    <p style="margin:0; font-size:1.2em; font-weight:bold;">
      A clip from the <a href="https://github.com/VincentCoulombe/precision_track/main/assets/PrecisionTrack+MART+Tailtags.mp4">PrecisionTrack with MART and Tailtags ReID demo</a>.
    </p>
    <img width="80%" src="https://raw.githubusercontent.com/VincentCoulombe/precision_track/main/assets/PrecisionTrack+MART+Tailtags.gif"/>
  </div>

</div>

## Web UI

The **easiest way** to use PrecisionTrack is its built-in Web UI: a local, single-user browser app that runs in the **same Python environment as PrecisionTrack** (it imports `precision_track` to validate your setup exactly the way the tools will). It is the recommended path for getting started — the CLI and `tools/` remain available for advanced or scripted workflows.

```bash
pip install -r requirements/web.txt
python -m web_ui            # serves http://127.0.0.1:8000 and opens your browser
```

With it you can:

- **Configure** `user_configs.yaml` through a form with live validation and file/folder pickers — no hand-editing YAML. Please refer to our [configuration documentation](https://github.com/VincentCoulombe/precision_track/tree/main/configs) for more details.
- **Edit** the separate validation / ReID configuration through a dedicated editor.
- **Run** any tool: pick it, set its options, and watch its output stream live in an embedded terminal — no CLI flags to remember.

For full details, see the [Web UI guide](https://github.com/VincentCoulombe/precision_track/tree/main/web_ui).

## Quick Navigation

- [Companion repositories](#companion-repositories--train--export-custom-models)
- [Demos](#demos)
- [Web UI](#web-ui)
- [Resources](#resources)
- [Where to start?](#where-to-start)
- [Tutorials](#tutorials)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Resources

- The [MICE dataset](https://drive.google.com/drive/folders/18Ikogjyeo_CRe9Z_iQfqYrfOfGt_dMma?usp=drive_link).
- The [Tailtag system plans](https://drive.google.com/drive/folders/1xXyVqE7a5kezlJp9c5zaJLlI_olqdnOl?usp=drive_link) for the 3x3 and 4x4 tags (the reported results were obtained using the 4x4 tags).
- [Training checkpoints](https://drive.google.com/drive/folders/1fpKgfnE3xD9xicfxzWdXDmA1p5lE8qmm?usp=drive_link) from the AP and MICE datasets.

## Where to start?

You will notice that there are alot of README.md files troughout the repository. And that most of them are very exhaustives. This amount to a ton of documentation. In fact, all the information you need is written somewhere in the repository. In order to not feel overwhelmed, I encourage you to use an AI assistant to help you summarize the content of the README.md (and answer your plain english questions). Either way, here is an inventory of all the usefull README.md files.

### General Documentation

This file

### Installation-related Documentation

- `docker/README.md`

### Configuration-related Documentation

- `configs/README.md`
- `configs/metadata/README.md`
- `configs/settings/validation/README.md`
- `configs/wandb/README.md`

### Checkpoints-related Documentation

- `checkpoints/README.md`

### Tools-related Documentation

- `tools/README.md`

### GUI

- `web_ui/README.md`

### 1) Install python on you machine

You are going to need the Python interpreter to label your experiments and to train, test, deploy, track and visualize your experiments using PrecisionTrack.

- [How to install Python on MAC](https://www.youtube.com/watch%3Fv%3Dnhv82tvFfkM&ved=2ahUKEwjIlP7Ul4iPAxUCC3kGHWV_H6gQ3aoNegQIGBAN&usg=AOvVaw3-TNQae7NFVvkURS-L2hwk)
- [How to install Python on Windows](https://m.youtube.com/watch%3Fv%3DNES0LRUFMBE%26pp%3D0gcJCfwAo7VqN5tD&ved=2ahUKEwjl572AmIiPAxUmkYkEHUqPBdQQ3aoNegQIERAO&usg=AOvVaw3RFqjmp-6ySX5s75reHs9b)

### 2) Define your metadata.py file

The metadata.py file is the first of the three key inputs you'll need in order to use PrecisionTrack effectively:

- **Metadata file**
- **Annotation files**
- **Configuration file**

The metadata file contains essential information about your subjects. Namely, their classes (We typically classify them by species), skeletons (for pose-estimation), and actions (for action recognition). For a detailed explanation of the metadata's file expected structure (including examples), please refer to our [metadata guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/metadata). In this section, we will focus on how to create your own `metadata.py` file.

#### 2.1) Start from an existing metadata file

In the `./configs/metadata/` subfolder, you will find pre-made metadata files for the MICE, Animal Pose (AP), and Microsoft COCO datasets.
We recommend starting by copying one of these files and modifying it to match the requirements of your experiment.

#### 2.2) Modify the existing metadata file

If your goal is only to track subjects, without estimating their poses or inferring their actions, your `metadata.py` file will be minimal, since there will be no skeletons or actions to define.

A minimal example looks like this:

```python
dataset_info = dict(
    dataset_name="ENTER YOUR DATASET NAME HERE",
    paper_info=dict(),
    keypoint_info={},
    skeleton_info={},
    joint_weights=[],
    sigmas=[],
    classes=[],
    actions=[],
)
```

However, if you plan to track poses or infer actions, you will need to define your skeletons, keypoints, and action labels. If you also want **group action recognition** (GMART), three additional fields are required in your metadata file:

- `null_action` — the background/fallback label (e.g. `"Other"`)
- `social_actions` — subset of `actions` that represent inter-subject interactions
- `distance_keypoint_pairs` — cross-subject keypoint pairs used as spatial priors by GMART

In that case, follow the instructions in our [metadata guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/metadata) to properly adapt your copied metadata file to your specific needs.

### 3) Creating a subject detection & pose-estimation (COCO-formatted) Dataset

COCO is the most popular object-detection and pose-estimation dataset format in the world. There is a lot of sources online on how you can format your dataset into a COCO dataset. As such, I highly encourage you to check some out (or ask an AI to list you some if you already have a detection and pose-estimation dataset. Most notably, you can use the [SLEAP IO](https://io.sleap.ai/latest/) repository to format your current DLC or SLEAP dataset into COCO.

Unfortunately, if you do not currently have an object-detection and pose-estimation dataset, you will need one. This means that you will need to create one. While we have seen trivial projects achieve good tracking results after been trained on COCO-formatted datasets containing as few as 50 labelled images, we recommend labelling at least ~250 images (just to be safe). If you notice subpar detection quality during tracking, it likely means you either need to label more frames or verify the quality of your existing labels. We will explain how to address both of these issues in the following subsections.

#### 3.1) Record experiments

This step is relatively straightforward but critically important for achieving optimal tracking performance. We strongly recommend recording your training data from the same camera viewpoint and setup that you will use for actual PrecisionTrack deployments. This ensures that the algorithm learns from data that closely matches the real-world conditions under which it will operate, such as lighting, background, perspective, and subject scale.

By matching these conditions, you reduce the risk of degraded performance caused by domain shift, where the model encounters visual patterns it was not exposed to during training. In short, the closer your training recordings are to your intended application setup, the better PrecisionTrack will generalize to your actual experiments.

#### 3.2) Extract frames uniformly from recordings

It is neither necessary nor efficient to label every single frame from your recordings. Consecutive frames are often too similar, resulting in redundant data that provides little additional benefit for training. Instead, we recommend uniformly sampling frames from your recordings to create a more diverse and representative dataset.

Choose a sampling interval that captures sufficient variation in your subjects’ positions, postures, and interactions. For example, in scenarios where subjects move slowly, you can use a larger interval between frames, while faster or more dynamic activities may require shorter intervals to capture meaningful changes.

- **Note:** You can extract the frames of a video using [ffmpeg](https://ffmpeg.org/).
  - [How to do it on windows](https://www.youtube.com/watch%3Fv%3DxH_KEOxeHac&ved=2ahUKEwiP87X9j6mPAxW3lokEHXKgOiIQ3aoNegQIGBAN&usg=AOvVaw3wQwbPsQvgiDkdqgBrC5t6)
  - [How to do it on Linux](https://www.youtube.com/watch%3Fv%3DYpH6lc8X8BY&ved=2ahUKEwjGgemFkamPAxUvrokEHS_rIaEQ3aoNegQIEBAg&usg=AOvVaw0Rz_30ZvGlB3Ti8V7IrrBx)

#### 3.3) Randomly select the _n_ frames you would like to label

By this stage, you may have accumulated hundreds—or even thousands—of extracted frames. While labelling all of them would not be a waste (as it would inevitably produce a stronger tracker), it is rarely the most efficient use of your time. In our experience, beyond roughly 1,000 labelled frames, the improvement in tracking accuracy per additional labelled frame begins to plateau, leading to steep diminishing returns relative to the time invested labelling frames.

For this reason, we recommend selecting an initial set of _n_ frames to label, keeping the total below this threshold for your first training cycle. As noted in the previous section, aim for frames that capture a broad range of scenarios, including different poses, interactions, backgrounds, and lighting conditions. The more diverse and representative your labelled set, the better your tracker will generalize to the wide variety of situations it may encounter during real-world use.

#### 3.4) Label your _n_ frames

To label your selected _n_ frames, we strongly recommend using the popular CVAT labelling platform for this task. If you follow this approach, you can benefit from the excellent work of [Julien Audet-Welke](https://github.com/juauw/CVAT_pipeline), who has thoroughly documented the entire process and even developed custom Python scripts to automate much of it.
We suggest reviewing his guide to streamline your labelling workflow.

Would you choose to follow Julien's guide or not, you will need COCO formatted labels in order to train your own custom PrecisionTracker.

- **Important:** Your subject’s keypoints labelling order must exactly match the order of the ids in the `keypoint_info` field of your `metadata.py` file.

- **Important** The quality of your annotations is the most important thing here. You want pixel perfect bounding boxes and keypoints as well as no false positive an/or false netagive annotations in your dataset. It is much, much better to have a small dataset of high quality than a huge dataset of bad quality.

### 4) Creating an Action Recognition dataset (optional)

**This guide contains a step-by-step guide to creating an action recignition dataset + a fully working example**

If you want to train a MART action recognition algorithm, you will need an action recognition dataset (note that you only need an action recognition dataset if you want to leveraged a trained MART algorithm to augment PrecisionTrack so it can produce behavioral data on top of tracking data).

- **NOTE** Please refer to our [MICE sequential dataset](https://drive.google.com/drive/folders/1WcDkX-92X6SCgZPAZXFyDc6EGUzU0Onq?usp=drive_link) as a valid Action Recognition dataset.

Your Action Recognition dataset will be composed of 3 [MOT-styled](https://motchallenge.net/) annotation files for each of your video files:

- A MOT file containing the subject's bounding boxes (bboxes)
- A MOT file containing the subject's keypoints
- A MOT file containing the subject's actions

All these four files will need to share the same name. Obviously, this mean that they will be saved in different directories. More speciffically, PrecisionTrack expect your action recognition dataset to have the followign structure:

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

- **How we created our action recognition dataset**: We created our action recognition dataset by doing the following:
  1. Train a good (80%+ detection F1 & 80%+ OKS) subject detection and pose-estimation model
  2. Use the `batch_track_directory.py` tool with `with_pose_estimation: true` (from inside the `user_configs.yaml` file),
     Please refer to the [tooling documentation](https://github.com/VincentCoulombe/precision_track/tree/main/tools) and the [configuration documentation](https://github.com/VincentCoulombe/precision_track/tree/main/configs). This tool will generate tracking results for every videos in the provided directory. The relevant tracking results (for creating action recognition dataset) will be saved in the `<saving_directory>/*/{tracked_bboxes.csv, kpts.csv}` files.
  3. Use the `visualize.py` tool (with only `display_bounding_boxes` set to `true` as Visualization parameter) to create a visual for every tracking results obtained in 2).
  4. Manually review the tracking results and correct the tracking errors (ID switches).
  - You can manually edit the tracking switches by editing the `instance_id` column (meaning switching back the swapped IDs) of the `tracked_bboxes.csv` file saved inside the `saving_directory` from which the visual was created.
  5. replace the `instance_id` column in your `kpts.csv` file with the one inside your `tracked_bboxes.csv` file (the one you just edited) by copy/pasting it. This will synchronize your pose-estimation results with your edited bounding boxes results.
  6. Re-running the step 3) to generate up-to-date visuals. These new visuals will take into account your manual corrections.
  7. Creating your `<action_recognition_data_root>` folder and formatting it according to the guide above.
  - This mean that you will need to rename your `tracked_bboxes.csv` and `kpts.csv` files so their names matches their corresponding video names.
  8. Uploading your visualization videos to [BORIS](https://www.boris.unito.it/) and manually annotating actions for every tracked subjects.
  9. Converting your [BORIS](https://www.boris.unito.it/) annotatinos into our action recognition format
  10. Saving the action recognition `.csv` file into the `<action_recognition_data_root>/actions/` directory (following the format above) and renaming them to match their corresponding video names.
- **Alternatively**: You can use the `create_mot_dataset.py` tool with `with_pose_estimation: true` (from inside the `user_configs.yaml` file),
  Please refer to the [tooling documentation](https://github.com/VincentCoulombe/precision_track/tree/main/tools) and the [configuration documentation](https://github.com/VincentCoulombe/precision_track/tree/main/configs) for more details, to automatically create an action recognition dataset. This second method (while fully automatic) is harder to manually review and correct.

- **NOTE** We used the the [BORIS](https://www.boris.unito.it/) software to label our actions then reformatted the labels to fit our needs.

- **NOTE** Actions with non null values > 0 under the `target_ids` columns will be considered as social actions and will be used to train your GMART algorithm.

### 5) Installing mandatory third-party software (local execution only)

Using our [COLAB Notebooks](https://github.com/VincentCoulombe/precision_track/tree/main/Colab)? You can **skip this entire section**.

#### 5.1) Install Windows Subsystem for Linux (WSL) - **Windows users only**

Open [Administrator PowerShell](https://www.youtube.com/watch%3Fv%3DUegCqUZcnq8&ved=2ahUKEwi9q-Dw46aPAxUTwvACHSz9GHsQ3aoNegQIFxAO&usg=AOvVaw3UJ0yzE6YAWzjWEyuyx5py) → run:

```powershell
wsl --install
```

Reboot your computer.

If WSL is not launching, check the following:

1. Open a PowerShell terminal
2. Type `wsl lst`
3. If no distribution is installed, install Ubuntu: `wsl.exe --install Ubuntu-22.04`

**All subsequent commands** are run **inside your WSL (Ubuntu) terminal**.
If a command is denied, prefix with **sudo** (e.g., `sudo apt-get update`). Doing so, the system will ask you for your password.

#### 5.2.1) Install Docker (inside WSL/Ubuntu)

To install [Docker](https://www.youtube.com/watch%3Fv%3Datb4nL-wI_M&ved=2ahUKEwj5men36qaPAxW7l4kEHZyuLZU4ChDdqg16BAgVEA4&usg=AOvVaw3w6GndpM3xsu3cwUr7s2rk), simply run the following:

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

Verify your installation by running:

```bash
sudo docker run hello-world
```

#### 5.2.2) Install Docker (inside Windows)

1. [Install Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
2. Open Docker Desktop
3. Go to Settings-Ressources-WSL integration and activate Enable integration with my default WSL distro and make sure that the Distro version is on (see screenshot below)
<div align="center">
  <img src="https://raw.githubusercontent.com/VincentCoulombe/precision_track/main/assets/docker_desktop_installation.png" width="75%"/>
</div>
4. Apply/restart

### 5.3) Ensure your machine is CUDA-accelerated (for GPU use)

##### 5.3.1) Check for an NVIDIA GPU and driver

First, ensure that your machine contains an NVIDIA Graphic Processor Unit (GPU) with at least 8BG of VRAM.

Inside WSL:

```bash
  sudo nvidia-smi
```

- If you see your GPU and a CUDA version as well as its specifications such the number of tensor cores and the amount of VRAM, you’re good.
- If not: install the latest Windows [NVIDIA driver](<(https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html)>) (do not install a Linux driver inside WSL), then wsl --update and try again.

##### 5.3.2) Allow Docker containers to access your NVIDIA GPU

Run the following bash command in your terminal:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

sudo systemctl restart docker
```

Test GPU inside a container:

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

#### 5.4) Clone the PrecisionTrack repository

You are going to need Git to clone this repository locally.

- [How to install Git on Windows](https://www.youtube.com/watch%3Fv%3Dt2-l3WvWvqg&ved=2ahUKEwiHu-OdtYiPAxV0rYkEHSFYMdAQ3aoNegQIExAO&usg=AOvVaw2BD43-Xq8afuWQ8HnbJxjv)

Run the following git command in your terminal:

- **Important** We recommend a choosing a stable, memorable path since this folder will be mounted in your Docker container (will be introduced in the next sub-section). thus, will become your main workspace. A sensible choice would be to clone PrecisionTrack in your `C:\Users\<YourUser>\Documents` directory.

- **Important 2** (Windows users only) Your `C:\Users\<YourUser>\Documents` Windows directory is reachable from inside your WSL instance under the `/mnt/c/Users/<YourUser>/Documents` directory.

```bash
  sudo apt-get install -y git
  cd /mnt/c/Users/<YourUser>/Documents   # choose a meaningful workspace directory
  git clone https://github.com/VincentCoulombe/precision_track.git
  cd precision_track
```

#### 5.5) Setup PrecisionTrack's execution environment

**Step 1 — Build the Docker image (once)**

Make the script executable, then build. The script auto-detects your hardware (CUDA vs CPU), or you can force a target with a flag:

```bash
chmod +x ./docker/building_image.sh

bash ./docker/building_image.sh            # auto-detect (recommended)
bash ./docker/building_image.sh --cuda     # force CUDA build
bash ./docker/building_image.sh --cpu      # force CPU build
bash ./docker/building_image.sh --both     # build both images
bash ./docker/building_image.sh --skip-tests   # skip post-build sanity checks
```

**Step 2 — Launch the container (each session)**

```bash
chmod +x ./docker/launching_container.sh

bash ./docker/launching_container.sh            # auto-detect (recommended)
bash ./docker/launching_container.sh --cuda     # force CUDA container
bash ./docker/launching_container.sh --cpu      # force CPU container
bash ./docker/launching_container.sh --update   # pull latest code before starting
```

The container stays alive as long as the terminal remains open. Closing it stops the container — no data is lost since all outputs are written directly to your host machine. Re-run the launch command to start a new session. If the requested image does not yet exist, the launch script will build it automatically.

For the full reference (environment variable overrides, troubleshooting, file-layout guide), see the [Docker guide](https://github.com/VincentCoulombe/precision_track/tree/main/docker).

**Edit locally, run in Docker**

- Edit your settings, metadatas and datasets on your host machine (as you normally would).
- Run the PrecisionTrack's tool **inside the launched environment (Docker container)**.

**Host-side outputs**: All logs, metrics, checkpoints and results are automatically written to the set working directories on your **host machine**. For example, if you keep the [default working directory](https://github.com/VincentCoulombe/precision_track/tree/main/configs), the outputs will get written inside the `precision_track/work_dir` directory.

**Read/write scope (IMPORTANT)**:

- The container has read/write access **only** to your `precision_track` directory.
- Keep everything under `precision_track/`:
  - Datasets → `precision_track/datasets/<your datasets>/`
  - User Configs → `precision_track/configs/user_configs.yaml`
  - Metadata → `precision_track/configs/metadata/<your metadata file>.py`

### 6) Define your `user_configs.yaml` (configuration file)

- **Tip:** the easiest way to create and edit this file is the [Web UI](#web-ui), which renders `user_configs.yaml` as a form and validates every field for you — no manual YAML editing required.

The `user_configs.yaml` allow you to customize your own PrecisionTracker. In it, you will define:

- what features you want to enable and/or disable.

- Paths to your annotations and metadata files.

- The number of tracked subjects.

- And much more...

For a detailed breakdown of the file’s structure and all available options, consult our [user configuration guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs).

If you plan to use **re-identification** (the appearance-based PrecisionTrack-ReID or the Tailtag/ArUco system), its settings live in a **separate validation configuration file**. Everything about configuring it (choosing a strategy, pointing to your model, enabling or disabling identities, and an important note on the appearance pipeline's warm-up behaviour) is documented in our dedicated [validation configuration guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/settings/validation).

### 7) Get started with PrecisionTrack’s Toolkit

You’ve now configured all the essential inputs:

- **Metadata file**
- **Annotation files**
- **User configs file**

…and set up a compatible execution environment:

- **Local Docker Container**
- **Google COLAB Notebooks**

With these in place, you’re ready to make the most of PrecisionTrack’s features. The recommended way to run the tools is the [Web UI](#web-ui) — pick a tool, set its options, and watch it run live in your browser. For advanced or scripted use, you can instead follow our [tooling guide](https://github.com/VincentCoulombe/precision_track/tree/main/tools) **AND** our [checkpoints and hyperparameters](https://github.com/VincentCoulombe/precision_track/tree/main/checkpoints) to understand where to go from here. You can also use our pre-configured [COLAB Notebooks](https://github.com/VincentCoulombe/precision_track/tree/main/Colab) to train, test, deploy, track and visualize your experiments.

## Tutorials

PrecisionTrack offers multiple tools to train, test, deploy, track and visualize. It is completely configurable for your needs. To do so, Please refer to our [tooling documentation](https://github.com/VincentCoulombe/precision_track/tree/main/tools) and our [workflow tutorial](#where-to-start) for more details.

PrecisionTrack extends MMEngine's configuration style. If you are not familiar with it, please refer to [MMPose Overview](https://mmpose.readthedocs.io/en/latest/) and [MMengine Config Files](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html).

For a detailed explaination on how to parametrize PrecisionTrack for your needs, please refer to our [configuration documentation](https://github.com/VincentCoulombe/precision_track/tree/main/configs)

<details>
<summary>MMPose Tutorials</summary>

- [A 20-minute Tour to MMPose](https://mmpose.readthedocs.io/en/latest/guide_to_framework.html)
- [Demos](https://mmpose.readthedocs.io/en/latest/demos.html)
- [Inference](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html)
- [Configs](https://mmpose.readthedocs.io/en/latest/user_guides/configs.html)
- [Prepare Datasets](https://mmpose.readthedocs.io/en/latest/user_guides/prepare_datasets.html)
- [Train and Test](https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html)
- [Deployment](https://mmpose.readthedocs.io/en/latest/user_guides/how_to_deploy.html)
- [Model Analysis](https://mmpose.readthedocs.io/en/latest/user_guides/model_analysis.html)
- [Dataset Annotation and Preprocessing](https://mmpose.readthedocs.io/en/latest/user_guides/dataset_tools.html)

</details>

<details>
<summary>Useful Tools</summary>

- [Browse coco json](https://github.com/open-mmlab/mmyolo/blob/main/docs/en/useful_tools/browse_coco_json.md)
- [Print config](https://github.com/open-mmlab/mmyolo/blob/main/docs/en/useful_tools/print_config.md)
- [Visualization scheduler](https://github.com/open-mmlab/mmyolo/blob/main/docs/en/useful_tools/vis_scheduler.md)
- [Log analysis](https://github.com/open-mmlab/mmyolo/blob/main/docs/en/useful_tools/log_analysis.md)

</details>

## Contributing

We appreciate all contributions to improving PrecisionTrack. Please refer to our [code of conduct](.github/CODE_OF_CONDUCT.md) and our pull request [template](.github/pull_request_template.md) for the contributing guideline.

## Acknowledgements

Many of our implementations take root from publicly available work. We thank authors of:

- [MMCV](https://github.com/open-mmlab/mmcv)
- [MMdetection](https://github.com/open-mmlab/mmdetection)
- [MMPose](https://github.com/open-mmlab/mmpose)
- [MMDeploy](https://github.com/open-mmlab/mmdeploy)
- [SAM2](https://github.com/facebookresearch/sam2)
- [ByteTrack](https://github.com/FoundationVision/ByteTrack)
- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [Supervision](https://github.com/roboflow/supervision)

## Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{precision_track2025,
    title={PrecisionTrack: A Platform for Automated Long-Term Social Behavior Analysis in Naturalized Environments},
    author={Coulombe & al},
    year={2025}
}
```

## License

This project is released under the [GPL 3.0 license](LICENSE).
