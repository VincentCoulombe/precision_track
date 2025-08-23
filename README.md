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
Furthermore, we provide postprocessing algorithms such as the Multi-animal Action Recognition Transformer (MART) which enables action recognition, behavioural and social dynamics analysis at scale.

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

## Quick Navigation

- [Demos](#demos)
- [Resources](#resources)
- [Where to start?](#where-to-start)
- [Installation](#installation)
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

### 1) Install python on you machine

You are going to need the Python interpreter to label your experiments and to train, test, deploy, track and visualize your experiments using PrecisionTrack.

- [How to install Python on MAC](https://www.youtube.com/watch%3Fv%3Dnhv82tvFfkM&ved=2ahUKEwjIlP7Ul4iPAxUCC3kGHWV_H6gQ3aoNegQIGBAN&usg=AOvVaw3-TNQae7NFVvkURS-L2hwk)
- [How to install Python on Windows](https://m.youtube.com/watch%3Fv%3DNES0LRUFMBE%26pp%3D0gcJCfwAo7VqN5tD&ved=2ahUKEwjl572AmIiPAxUmkYkEHUqPBdQQ3aoNegQIERAO&usg=AOvVaw3RFqjmp-6ySX5s75reHs9b)

### 2) Define your metadata.py file

The metadata.py file is the first of the three key inputs you'll need in order to use PrecisionTrack effectively:

- **Metadata file**
- **Annotation files**
- **Settings file**

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

However, if you plan to track poses or infer actions, you will need to define your skeletons, keypoints, and action labels. In that case, follow the instructions in our [metadata guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs/metadata) to properly adapt your copied metadata file to your specific needs.

### 3) Labelling data (getting annotations)

Next, you will need labelled data to train your PrecisionTrack algorithm. While we have seen trivial projects achieve good tracking results after been trained with as few as 50 labelled images, we recommend labelling at least 100 images (just to be safe). If you notice subpar detection quality during tracking, it likely means you either need to label more frames or verify the quality of your existing labels. We will explain how to address both of these issues in the following subsections.

#### 3.1) Record experiments

This step is relatively straightforward but critically important for achieving optimal tracking performance. We strongly recommend recording your training data from the same camera viewpoint and setup that you will use for actual PrecisionTrack deployments. This ensures that the algorithm learns from data that closely matches the real-world conditions under which it will operate, such as lighting, background, perspective, and subject scale.

By matching these conditions, you reduce the risk of degraded performance caused by domain shift, where the model encounters visual patterns it was not exposed to during training. In short, the closer your training recordings are to your intended application setup, the better PrecisionTrack will generalize to your actual experiments.

#### 3.2) Extract frames uniformly from recordings

It is neither necessary nor efficient to label every single frame from your recordings. Consecutive frames are often too similar, resulting in redundant data that provides little additional benefit for training. Instead, we recommend uniformly sampling frames from your recordings to create a more diverse and representative dataset.

Choose a sampling interval that captures sufficient variation in your subjects’ positions, postures, and interactions. For example, in scenarios where subjects move slowly, you can use a larger interval between frames, while faster or more dynamic activities may require shorter intervals to capture meaningful changes.

#### 3.3) Randomly select the _n_ frames you would like to label

By this stage, you may have accumulated hundreds—or even thousands—of extracted frames. While labelling all of them would not be a waste (as it would inevitably produce a stronger tracker), it is rarely the most efficient use of your time. In our experience, beyond roughly 1,000 labelled frames, the improvement in tracking accuracy per additional labelled frame begins to plateau, leading to steep diminishing returns relative to the time invested labelling frames.

For this reason, we recommend selecting an initial set of _n_ frames to label, keeping the total below this threshold for your first training cycle. As noted in the previous section, aim for frames that capture a broad range of scenarios, including different poses, interactions, backgrounds, and lighting conditions. The more diverse and representative your labelled set, the better your tracker will generalize to the wide variety of situations it may encounter during real-world use.

#### 3.4) Label your _n_ frames

To label your selected _n_ frames, we strongly recommend using the popular CVAT labelling platform for this task. If you follow this approach, you can benefit from the excellent work of [Julien Audet-Welke](https://github.com/juauw/CVAT_pipeline), who has thoroughly documented the entire process and even developed custom Python scripts to automate much of it.
We suggest reviewing his guide to streamline your labelling workflow.

Would you choose to follow Julien's guide or not, you will need COCO formatted labels in order to train your own custom PrecisionTracker.

### 4) Installing mandatory third party software (for local execution only)

If you are planning on using PrecisionTrack throught our provided [COLAB Notebooks](https://github.com/VincentCoulombe/precision_track/tree/main/COLAB), then you can simply **ignore this section**, as it is only relevant for users wanting to use PrecisionTrack locally.

#### WORK-IN-PROGRESS

#### 4.1) Ensure that your machine is CUDA accelerated

Run the following bash command in your terminal:

```bash
  nvidia-smi
```

- If you get a table with your GPU name and a CUDA Version field. Your system is CUDA accelerated and the proper drivers are already installed.

  - Then, you need to ensure that your NVIDIA GPU have at least 8Gb of VRAM (8000MiB).

- If you get “Command not found”. Your system has either no NVIDIA GPU or your NVIDIA drivers are not properly installed.

  - Please ensure your system is CUDA accelerated before moving forward. Refer to [NVIDIA's driver installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html)

#### 4.2) Clone the PrecisionTrack's repository locally

You are going to need Git to clone this repository locally.

- [How to install Git on MAC](https://www.youtube.com/watch%3Fv%3DB4qsvQ5IqWk&ved=2ahUKEwj_pZe2tYiPAxVak4kEHRv2ClUQ3aoNegQIGBAJ&usg=AOvVaw0n2JpqE2yxaD-KHmzrSIb0)
- [How to install Git on Windows](https://www.youtube.com/watch%3Fv%3Dt2-l3WvWvqg&ved=2ahUKEwiHu-OdtYiPAxV0rYkEHSFYMdAQ3aoNegQIExAO&usg=AOvVaw2BD43-Xq8afuWQ8HnbJxjv)

Run the following git command in your terminal:

```bash
  git clone https://github.com/VincentCoulombe/precision_track.git
```

#### 4.3) Setup PrecisionTrack's execution environment

TODO (Docker)

### 5) Define your settings.py (settings file)

The settings file contains all the essential configuration parameters for running PrecisionTrack. In this file, you will specify:

- The paths to your annotations and metadata files

- Where to save training and testing logs and metrics

- Where to store model checkpoints

- The number of subjects you are tracking

- And much more...

For a detailed explanation of the file’s structure and all available options, refer to our [settings guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs). In this section, we will focus on how to create your own `settings.py` file.

#### 5.1) Start from an existing settings file

In the `./configs/settings/` subfolder, you will find pre-made metadata files for the MICE, Animal Pose (AP), and Microsoft COCO datasets.
We recommend starting by copying one of these files and modifying it to match the requirements of your experiment.

#### 5.2) Modify the existing settings file

Follow our [settings guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs) to properly adapt your copied settings file. In most cases, you will only need to adjust around a dozen fields, such as dataset paths, output directories, and subject counts. We discourage modifying other parameters unless you are experienced with PrecisionTrack’s configuration system, as incorrect changes could lead to unexpected results.

### 5) Enjoy PrecisionTrack's Toolkit

By now, you should have configured all the essential inputs for PrecisionTrack:

- **Metadata file**
- **Annotation files**
- **Settings file**

…and set up a suitable execution environment:

- **Local Docker container**
- **Google COLAB Notebooks**

With everything in place, you’re ready to make the most of PrecisionTrack’s features. As such, we encourage you to either follow our [tooling guide](https://github.com/VincentCoulombe/precision_track/tree/main/tools) or our pre-configured [COLAB Notebooks](https://github.com/VincentCoulombe/precision_track/tree/main/COLAB) to train, test, deploy track and visualize your PrecisionTracker.

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
    title={Scaling Up Social Behavior Studies: Real-Time, Large-Scale and Prolonged Social Behavior Analysis with PrecisionTrack},
    author={Coulombe & al},
    year={2025}
}
```

## License

This project is released under the [GPL 3.0 license](LICENSE).
