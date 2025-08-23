# 📚 Metadata Configuration Guide (`dataset_info`)

PrecisionTrack's metadata files follows the MMPose format. It consist of a python file containing the `dataset_info` dictionary. This dictionnary declares **what each keypoint means, how they connect, and how to evaluate them**.\
Below is a step‑by‑step recipe on how to create your own metadata file. To proceed, we will use MICE dataset's `mice.py` metadata file as an example.

______________________________________________________________________

## 1. Template (minimal)

```python
dataset_info = dict(
    dataset_name="my_dataset",
    paper_info=dict(),              # bibliographic record (optional)
    keypoint_info={                 # list your keypoints here
        0: dict(name="", id=0, type="", swap=""),
        # …
    },
    skeleton_info={                 # links between keypoints, forms the poses
        0: dict(link=("", ""), id=0),
        # …
    },
    joint_weights=[],               # 1 float per keypoint
    sigmas=[],                      # 1 σ per keypoint (for OKS)
    classes=[],                     # object categories
    actions=[],                     # optional behaviour labels
)
```

> **Why so many fields?**\
> Model architectures, training loss, data‑augmentation, visualisation and evaluation all rely on this metadata.

______________________________________________________________________

## 2. Field‑by‑field cookbook

| Field | Purpose | How to fill it | Quick example |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **`dataset_name`** | Unique short slug used in configs and logs. | • Keep it lowercase, no spaces.<br>• Change whenever the annotation schema changes. | `"dogs_running_v1"` |
| **`paper_info`** | Auto‑generates citations in reports. | Keys accepted by MMEngine: `author`, `title`, `container`, `year`, `homepage`. | `dict(author="Doe et al.", title="CaninePose 2025", year=2025)` |
| **`keypoint_info`** | Full specification for _each_ keypoint. Drives:<br>• **The name of the keypoint** (`name`)<br>• **The unique identifier of the keypoint. Must start at 0 and increment continuously without gaps.** (`id`)<br>• **Keypoint symmerical reflection (for the flipping augmentation)** (`swap`)<br>• **Half‑body aug.** (`type`)<br>• Visual colours (optional `color`) | | `dict(name="Right Ear", id=1, type="upper", swap="Left Ear")` |
| **`skeleton_info`** | Edge list connecting keypoints for drawing limbs & PAF‑based losses. | Order is arbitrary; use tuples of keypoint `name`s. | `0: dict(link=("Nose", "Left Eye"), id=0)` |
| **`joint_weights`** | Per‑keypoint loss weight. Tip: ↑ weight for small, hard‑to‑see landmarks. | List length == `len(keypoint_info)`; defaults to 1.0. | `[1, 1, 2, 2, …]` |
| **`sigmas`** | Normalised labelling error (σ) used in **OKS** metric. Smaller σ ⇒ stricter. | Compute `σ = (expected error) / object_size`. Use COCO as rough guide. | Humans (COCO) ≈ 0.026 – 0.107 |
| **`classes`** | Object categories present in the annotations. | String list; first item becomes default label `id=0`. | `["mouse"]` |
| **`actions`** | Optional behaviour/action labels for the action-recognition downstream task. | Provide `"Other"`/`"None"` as the filler class. | `["Other", "Running", "Sleeping"]` |

______________________________________________________________________

## 3. Step‑by‑step: Creating your own metadata file.

1. **List keypoints**\
   Write down every landmark you annotated, decide whether it belongs to the “upper” (head/torso) or “lower” (limbs/tail) group, and set left/right symmerical reflection (swaps).

1. **Design the skeleton**\
   Connect adjacent landmarks to make a sensible stick‑figure which we refer to as a pose. Order and colours do **not** affect training; they only influence visualisation and downstream tasks such as action-recognition.

1. **Assign weights**\
   If all keypoints are equally important, keep them at `1.0`. Raise weights for tiny parts (e.g., bird beaks) or for medically critical landmarks.

1. **Estimate sigmas**\
   Rule of thumb:

   \[
   \\sigma = \\frac{\\text{avg. pixel error}}{\\text{object diagonal}}
   \]

   For small animals (~150 px wide), a 3 px annotation error gives σ≈0.02.

1. **Populate classes & actions**\
   Add lists containing your dataset classes and actions to those keys respectively. Even if you track a single species without specifying its actions now, future‑proof by keeping the list format.

______________________________________________________________________

## 4. Worked examples

### 4.1 Bird pose (5 keypoints)

```python
dataset_info = dict(
    dataset_name="birds5k",
    keypoint_info={
        0: dict(name="Beak", id=0, type="upper", swap=""),
        1: dict(name="Left Wing", id=1, type="upper", swap="Right Wing"),
        2: dict(name="Right Wing", id=2, type="upper", swap="Left Wing"),
        3: dict(name="Tail Base", id=3, type="lower", swap=""),
        4: dict(name="Tail Tip", id=4, type="lower", swap=""),
    },
    skeleton_info={
        0: dict(link=("Beak", "Left Wing"), id=0),
        1: dict(link=("Beak", "Right Wing"), id=1),
        2: dict(link=("Tail Base", "Tail Tip"), id=2),
    },
    joint_weights=[1, 1, 1, 0.5, 0.5],
    sigmas=[0.02] * 5,
    classes=["bird"],
    actions=["Other", "Flying", "Perching"]
)
```

### 4.2 Multi‑species demo (dogs & cats)

```python
dataset_info = dict(
    dataset_name="pets10k",
    keypoint_info={…},                # shared schema
    skeleton_info={…},
    joint_weights=[1] * 17,
    sigmas=[0.03] * 17,
    classes=["dog", "cat"],
    actions=["Other", "Walking", "Running", "Jumping"]
)
```

### 4.3 Human upper‑body only

```python
keypoint_info = {
    0: dict(name="Nose", id=0, type="upper", swap=""),
    1: dict(name="Left Eye", id=1, type="upper", swap="Right Eye"),
    2: dict(name="Right Eye", id=2, type="upper", swap="Left Eye"),
    3: dict(name="Left Shoulder", id=3, type="upper", swap="Right Shoulder"),
    4: dict(name="Right Shoulder", id=4, type="upper", swap="Left Shoulder"),
}
sigmas = [0.026, 0.025, 0.025, 0.035, 0.035]  # taken from COCO
```

______________________________________________________________________

## 5. Validation snippet (Python)

```python
from precision_track.utils import parse_pose_metainfo

info = parse_pose_metainfo(dict(from_file="path_to_your_metadata_file.py"))   # raises KeyError if anything is missing
print(f"{info.num_keypoints} keypoints loaded ✔︎")
```

______________________________________________________________________

## 6. Troubleshooting checklist

- **Index error during flip augmentation?**\
  Swap pairs missing or unequal counts.
- **OKS AP stuck at 0?**\
  Check that `sigmas` length == `joint_weights` length == `num_keypoints`.
- **Loss dominated by one landmark?**\
  Scale down its `joint_weights`.
- **Visual skeleton looks wrong?**\
  Edge names in `skeleton_info` must match `keypoint_info["name"]`.

______________________________________________________________________

## 7. Further reading

- [MMPose docs — Preparing custom datasets](https://mmpose.readthedocs.io/en)
- [COCO Keypoint Evaluation (OKS)](https://cocodataset.org/#keypoints-eval)
