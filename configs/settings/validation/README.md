# PrecisionTrack – Validation Configuration Guide

Welcome! 👋  
This guide explains **how to configure PrecisionTrack's validation (re-identification) pipeline** by editing one of the config files in this directory.  
👉 The path to your chosen config file is set via `validation_configuration_file` in `./user_configs.yaml`.

---

# Overview

Two validation strategies are available, each with its own config file:

1. **appearance.yaml** → Appearance-based re-identification (PrecisionTrack-ReID)
2. **aruco.yaml** → The Tailtag system

---

# 1. appearance.yaml – PrecisionTrack-ReID

Uses a deep learning model to re-identify animals by their visual appearance (coat pattern, texture, etc.), without requiring any physical markers on the animals.

> ## ⚠️ Warm-up period: expect unstable results until every subject has been seen ⚠️
>
> The appearance re-identification pipeline needs a **warm-up period** at the start of a video. During this period its corrections are **provisional and may change**. The pipeline only becomes stable once it has observed **all** the subjects whose identities are _enabled_ (i.e. every identity **not** listed in `disabled_identities`). Until then, do not trust the early identity assignments — they will settle on their own once everyone has appeared.
>
> **Why this happens:**
> The validator does not know in advance which on-screen track belongs to which identity. It learns this **one subject at a time** as the animals appear:
>
> 1. **The first time** it confidently recognizes a given identity, it simply **writes it down**. It does **not** correct anything yet, because it has nothing to compare against.
> 2. A real **ID-switch correction** can only happen when the recognized identity is **already registered but pointing at a _different_ track**. That is the only situation where the validator knows for sure that two tracks got swapped and can fix it.
>
> The consequence: as long as some enabled identities have **never been seen**, the registration is incomplete. The validator may temporarily hand an identity to the wrong animal simply because the rightful owner hasn't shown up yet. The moment that missing subject finally appears and gets recognized, the conflict is detected and the identities are swapped back into place. This is why the **first few seconds can show a burst of ID switches** that then disappear.
>
> **Practical guidance:**
>
> - Make sure every enabled subject becomes clearly visible **early** in the recording (good lighting, separated animals, no occlusion). The sooner each one is seen, the sooner the pipeline stabilizes.
> - Treat identities assigned during the warm-up period as tentative; rely on them only after every enabled subject has appeared at least once.
> - Identities you have placed in `disabled_identities` do **not** count — the pipeline never waits for them and never assigns them (see the [Disabling identities](#disabling-identities-letting-multiple-subjects-share-the-same-identity) section below).

- **<u>type</u>**  
  Must be set to `AppearanceValidation`. Tells PrecisionTrack which validation backend to instantiate.

- **<u>data_preprocessor.type</u>**  
  Preprocessor used to normalize and resize animal crops before they are passed to the re-identification model.  
  Currently, the only supported value is `WildLifeReIDPreprocessor`.

- **<u>re_identificator.metainfo</u>**  
  Path to the YAML metadata file describing the re-identification model (class names, input shape, etc.).

- **<u>re_identificator.checkpoint</u>**  
  Path to the ONNX checkpoint of the re-identification model. This is the model that will produce appearance embeddings used to distinguish individuals.

- **<u>validated_classes</u>**  
  List of animal classes (as defined in your **metainfo** file) on which re-identification will be applied. Animals belonging to classes not listed here will not be re-identified.

- **<u>min_consecutive_hits</u>** _(optional, default `5`)_  
  this defines the sensitivity of the validation process.
  - **Lower** → the validator reacts faster and corrects more aggressively, but is more prone to **false ID switches**.
  - **Higher** → the validator is more conservative: fewer false switches, but it takes longer to fix real ID swaps.

  This is the main knob to reach for when tuning the trade-off described in the [warm-up note](#️-warm-up-period-expect-unstable-results-until-every-subject-has-been-seen-️) above and in [Tips](#tips) (tip #3): if you see frequent ID switches, **increase** it; if corrections feel too slow, **decrease** it.

### Disabling identities (letting multiple subjects share the same identity)

By default, PrecisionTrack expects a **one-to-one mapping** between tracked subjects and re-identification identities: each subject is matched to its own distinct identity.

In some experiments this is not desirable. For example, two animals may be **visually indistinguishable** (same coat colour and pattern), so the re-identification model cannot reliably tell them apart. Forcing the validator to re-identify them only produces even more ID switches.

For these cases you can **disable** specific identities. A disabled identity is still produced by your trained re-identification model, but the validation process **ignores it**: it is never selected as a confirmed prediction and never triggers an ID correction. Subjects whose appearance matches a disabled identity are simply left to the motion tracker, uncorrected.

Disabled identities are declared with the **`disabled_identities`** key in the re-identification **metainfo file** (the YAML pointed to by `re_identificator.metainfo`).

- **<u>disabled_identities</u>** _(optional)_  
  List of identity names to ignore during validation. Every entry must already appear in the `identities` list of the same metainfo file. Disabled identities will never trigger the evidence-based re-identification pipeline. Here are a few possible use cases when you might considr disabling identities:
  1. There are subjects in your experiment that you do not want to track
  2. A few identities are often misclassified by your re-identification model (theyre hard to distinguish under certain condition) and you prefer to remove them from your study for accuracy purposes
  3. Your experiment purposely contains a group of control subjects (which are unmarked) and you prefer if the tracker does not try to re-identify them.

**Example** — a metainfo file with no disabled identities (default behaviour, every subject is re-identified):

```yaml
identities:
  - White_1
  - White_2
  - Black
  - Brown
input_shape:
  - 224
  - 224
nb_features: 128
confidence_threshold: 0.75
bbox_enlargement: 0.5
```

**Example** — the two white mice are indistinguishable, so identity `White` is disabled. `Black` and `Brown` are still re-identified normally, while the white mice are tracked by motion only and may share the `White` identity:

```yaml
identities:
  - White
  - Black
  - Brown
disabled_identities:
  - White
input_shape:
  - 224
  - 224
nb_features: 128
confidence_threshold: 0.75
bbox_enlargement: 0.5
```

You can disable more than one identity by listing each on its own line:

```yaml
disabled_identities:
  - White
  - Grey
```

**⚠️IMPORTANT⚠️** Every name in `disabled_identities` must match an entry in `identities` exactly (case-sensitive). An unknown name will raise an error at startup.

---

# 2. aruco.yaml – The Tailtag system

- **<u>type</u>**  
  Must be set to `ArucoValidation`. Tells PrecisionTrack which validation backend to instantiate.

- **<u>validated_classes</u>**  
  List of animal classes on which ArUco-based re-identification will be applied.

- **<u>num_tags</u>**  
  Total number of unique ArUco tags in your dictionary (i.e., the total number of different markers that could theoretically be detected).

- **<u>tags_size</u>**  
  Size of each marker expressed as the number of internal bit cells per side (e.g., `3` means a 3×3 bit grid). Must match the physical tags used in your experiment.

- **<u>predefined_dict</u>**  
  Name of a predefined OpenCV ArUco dictionary (e.g., `"DICT_4X4_50"`). Set to `null` to generate a custom dictionary based on `num_tags` and `tags_size`.

- **<u>parameters</u>**  
  Fine-grained OpenCV `ArUcoDetector` parameters that control how markers are detected. These are the most impactful ones:
  - **<u>minMarkerPerimeterRate</u>** / **<u>maxMarkerPerimeterRate</u>**  
    Minimum and maximum accepted marker perimeter as a fraction of the image's maximum dimension. Markers that appear too small or too large will be rejected. Adjust based on how big tags appear in your videos.

  - **<u>adaptiveThreshWinSizeMin</u>** / **<u>adaptiveThreshWinSizeMax</u>** / **<u>adaptiveThreshWinSizeStep</u>**  
    Range and step size of the adaptive thresholding window. The detector sweeps through this range to improve robustness under variable lighting.

  - **<u>polygonalApproxAccuracyRate</u>**  
    Accuracy of the polygonal approximation used to identify marker corners. Lower values enforce a stricter square shape; higher values are more permissive.

  - **<u>minOtsuStdDev</u>**  
    Minimum standard deviation of pixel intensities required before attempting Otsu thresholding on a candidate region. Prevents false positives in flat, featureless image regions.

  - **<u>perspectiveRemovePixelPerCell</u>**  
    Number of pixels allocated per bit cell when correcting perspective distortion. Higher values increase accuracy at the cost of computation.

  - **<u>perspectiveRemoveIgnoredMarginPerCell</u>**  
    Fraction of each cell's border (per side) ignored when reading bit values after perspective correction. Reduces sensitivity to blurry or imprecise cell edges.

- **<u>refinement</u>**  
  Post-detection corner refinement strategy. Options: `none`, `contour`, `apriltag`. Use `none` if speed is a priority or markers are printed and attached cleanly.

- **<u>tag_kpt</u>**  
  Index of the keypoint on the animal (as defined in your **metainfo** file) that is expected to be co-located with the ArUco tag. PrecisionTrack will center its search window around this keypoint.

- **<u>kpt_conf_thr</u>**  
  Minimum confidence score for `tag_kpt` before PrecisionTrack attempts to detect a tag. Detections with low-confidence keypoints are skipped to avoid false reads.

- **<u>estimation_range</u>**  
  Radius (in pixels) around `tag_kpt` within which the detector will search for an ArUco marker.

- **<u>timeout_after</u>**  
  Maximum time (in seconds) spent attempting to detect a tag on a single animal in a single frame. Acts as a safety valve to prevent slow detections from stalling the pipeline.

- **<u>min_sample_size</u>**  
  Minimum number of successful tag detections required before an identity is assigned to a track. Accumulating multiple reads reduces the chance of a mis-identification caused by a single bad frame.

- **<u>valid_tags</u>**  
  Exhaustive list of tag IDs that are physically present in your experiment. Any detected ID not in this list will be discarded as a false positive.

  **⚠️IMPORTANT⚠️** Ensure this list reflects exactly the tags attached to your animals. A missing ID means that animal will never be re-identified; an extra ID may cause false matches.

---

# Tips

- If re-identification results seem wrong:
  1. For **appearance.yaml**: verify that the `checkpoint` and `metainfo` paths are correct and that your model was trained on a similar species/dataset.
  2. For **aruco.yaml**: check that `valid_tags` matches the tags physically in your experiment and that `estimation_range` is large enough to cover the tag's position relative to `tag_kpt`.
  3. Increase `min_sample_size` to make identity assignments more conservative if you observe frequent ID switches.
  4. If nothing is working, you can contact us directly for help, or open an issue in the repository.

- YAML files are sensitive to indentation — avoid using tabs.
