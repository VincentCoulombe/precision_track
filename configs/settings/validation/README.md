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
