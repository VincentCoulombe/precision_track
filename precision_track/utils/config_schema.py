# Copyright (c) Vincent Coulombe
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .datasets import assert_coco_dataset_directory, check_if_mot_dataset_is_ok, parse_pose_metainfo
from .deployment import set_runtime_attributes
from .io import SUPPORTED_VIDEO_BACKEND

CONFIGS_README = "configs/README.md"
ROOT_README = "README.md"
METADATA_README = "configs/metadata/README.md"
VALIDATION_README = "configs/settings/validation/README.md"

DOC_LINKS = {
    "booleans": (CONFIGS_README, "#1-booleans--enable-or-disable-functionalities"),
    "general": (CONFIGS_README, "#2-general-directories-and-paths"),
    "training": (CONFIGS_README, "#3-training-parameters-directories-and-paths"),
    "tracking": (CONFIGS_README, "#4-tracking-parameters"),
    "action_recognition": (CONFIGS_README, "#5-action-recognition-parameters"),
    "group_action_recognition": (CONFIGS_README, "#6-group-action-recognition-parameters"),
    "validation": (CONFIGS_README, "#7-validation-parameters"),
    "visualization": (CONFIGS_README, "#8-visualization-parameters"),
    "metainfo": (METADATA_README, ""),
    "data_root": (ROOT_README, "#3-creating-a-subject-detection--pose-estimation-coco-formatted-dataset"),
    "action_recognition_data_root": (ROOT_README, "#4-creating-an-action-recognition-dataset-optional"),
    "validation_configuration_file": (VALIDATION_README, ""),
}


def doc_pointer(key: str) -> str:
    file, anchor = DOC_LINKS.get(key, (CONFIGS_README, ""))
    return f"See {file}{anchor} for details on how to configure this."


def format_config_errors(exc) -> str:
    lines = ["Invalid 'user_configs.yaml':"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        if loc:
            lines.append(f"  - [{loc}] {msg}")
        else:
            lines.append(msg)
    return "\n".join(lines)


def _deploying_directory_has_content(deploying_directory: str) -> bool:
    return os.path.isdir(deploying_directory) and len(os.listdir(deploying_directory)) > 0


def _check_action_recognition_dataset(dataset_root_dir: str) -> List[str]:
    root = Path(dataset_root_dir)
    errors = []
    for split in ("train", "val"):
        videos_dir = root / "videos" / split
        if not videos_dir.is_dir():
            errors.append(f"videos/{split} directory not found: {videos_dir}")
            continue
        video_files = sorted(f for f in videos_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_BACKEND)
        if not video_files:
            errors.append(f"videos/{split} contains no video files: {videos_dir}")
            continue
        for video_file in video_files:
            stem = video_file.stem
            for sub in ("bboxes", "keypoints", "actions"):
                expected = root / sub / split / f"{stem}.csv"
                if not expected.is_file():
                    errors.append(f"missing {sub}/{split}/{stem}.csv for video '{video_file.name}'")
    return errors


class BooleansConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipelined: bool
    with_validation: bool
    with_offline_correction_refinement: bool
    with_action_recognition: bool
    with_group_action_recognition: bool
    with_pose_estimation: bool


class GeneralConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metainfo: str

    @field_validator("metainfo")
    @classmethod
    def metainfo_is_valid(cls, v: str) -> str:
        try:
            parse_pose_metainfo(dict(from_file=v))
        except Exception as e:
            raise ValueError(f"Invalid metainfo file '{v}': {e} {doc_pointer('metainfo')}")
        return v


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str
    data_root: str
    resume: bool
    training_checkpoint: str
    deploying_directory: str
    deploying_sanity_check_img_path: str
    batch_size: int = Field(ge=16)
    wandb_logging: bool

    @field_validator("dataset_name")
    @classmethod
    def dataset_name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(f"dataset_name must not be blank. {doc_pointer('training')}")
        return v

    @field_validator("data_root")
    @classmethod
    def data_root_is_valid_coco(cls, v: str) -> str:
        try:
            assert_coco_dataset_directory(v)
        except Exception as e:
            raise ValueError(f"Invalid COCO-style dataset at data_root='{v}': {e} {doc_pointer('data_root')}")
        return v

    @field_validator("training_checkpoint")
    @classmethod
    def training_checkpoint_exists(cls, v: str) -> str:
        if not os.path.isfile(v):
            raise ValueError(
                f"training_checkpoint does not exist: '{v}'. This should be a .pth checkpoint used to initialize "
                f"training (e.g. transfer learning from the AP checkpoint). {doc_pointer('training')}"
            )
        return v

    @field_validator("deploying_directory")
    @classmethod
    def deploying_directory_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                f"training.deploying_directory is blank. Every checkpoint field (tracking_checkpoint_name, "
                f"mart_checkpoint_name, gmart_checkpoint_name, hyperparameters_file_name) is resolved as "
                f"'deploying_directory/<name>', so an empty value would break all of them. Set it to the directory "
                f"where your deployed checkpoints are (or should be) saved, e.g. '../checkpoints/my_model/'. "
                f"{doc_pointer('training')}"
            )
        return v

    @model_validator(mode="after")
    def sanity_check_img_exists(self) -> "TrainingConfig":
        full_path = os.path.join(self.data_root, self.deploying_sanity_check_img_path)
        if not os.path.isfile(full_path):
            raise ValueError(
                f"deploying_sanity_check_img_path does not resolve to a real file: '{full_path}' "
                f"(this path is relative to data_root, not the tools/ directory). {doc_pointer('training')}"
            )
        return self


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    saving_directory: str
    num_subjects: Dict[str, int]
    tracking_checkpoint_name: str
    hyperparameters_file_name: str
    output_clustered_features: bool
    mot_data_root: str

    @field_validator("saving_directory")
    @classmethod
    def saving_directory_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                f"tracking.saving_directory is blank. This is where PrecisionTrack writes all tracking outputs "
                f"(tracked_bboxes.csv, tracked_kpts.csv, etc.), and where the visualization tool reads from — it "
                f"can't be left empty. {doc_pointer('tracking')}"
            )
        return v

    @field_validator("num_subjects")
    @classmethod
    def num_subjects_values_valid(cls, v: Dict[str, int]) -> Dict[str, int]:
        for cls_name, count in v.items():
            if count == 0 or count < -1:
                raise ValueError(
                    f"num_subjects['{cls_name}'] = {count} is invalid — must be a positive integer, or -1 if "
                    f"subjects can freely enter/leave the scene. {doc_pointer('tracking')}"
                )
        return v

    @field_validator("mot_data_root")
    @classmethod
    def mot_data_root_is_valid(cls, v: str) -> str:
        ok, msg = check_if_mot_dataset_is_ok(v)
        if not ok:
            raise ValueError(f"Invalid MOT dataset at mot_data_root='{v}': {msg} {doc_pointer('tracking')}")
        return v


class ActionRecognitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mart_checkpoint_name: str
    action_recognition_data_root: str
    output_action_recognition_embeddings: bool


class GroupActionRecognitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gmart_checkpoint_name: str


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    validation_configuration_file: str
    output_appearance_database: bool


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_bounding_boxes: bool
    display_poses: bool
    display_velocities: bool
    display_species: bool
    display_confidence_scores: bool
    display_actions: bool
    display_search_zones: bool
    display_validations: bool
    display_untracked_detections: bool
    display_predicted_bounding_boxes: bool


class UserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    booleans: BooleansConfig
    general: GeneralConfig
    training: TrainingConfig
    tracking: TrackingConfig
    action_recognition: ActionRecognitionConfig
    group_action_recognition: GroupActionRecognitionConfig
    validation: ValidationConfig
    visualization: VisualizationConfig

    @model_validator(mode="after")
    def cross_section_checks(self) -> "UserConfig":
        errors: List[str] = []

        if self.booleans.with_group_action_recognition and not self.booleans.with_action_recognition:
            errors.append(
                "booleans.with_group_action_recognition is true but booleans.with_action_recognition is false — "
                "group action recognition requires action recognition to also be enabled. " + doc_pointer("group_action_recognition")
            )
        if (self.booleans.with_action_recognition or self.booleans.with_group_action_recognition) and not self.booleans.with_pose_estimation:
            errors.append(
                "booleans.with_pose_estimation is false, but action/group-action recognition is enabled — pose "
                "estimation is mandatory for action recognition. " + doc_pointer("booleans")
            )

        try:
            metainfo_classes = set(parse_pose_metainfo(dict(from_file=self.general.metainfo))["classes"])
        except Exception:
            metainfo_classes = None  # already reported by GeneralConfig's own field validator
        if metainfo_classes is not None:
            for cls_name in self.tracking.num_subjects:
                if cls_name not in metainfo_classes:
                    errors.append(
                        f"tracking.num_subjects references class '{cls_name}', which is not defined in your "
                        f"metainfo file's 'classes' list ({sorted(metainfo_classes)}). {doc_pointer('tracking')}"
                    )

        deploying_directory = self.training.deploying_directory
        if _deploying_directory_has_content(deploying_directory):
            checkpoint_fields = [("tracking", "tracking_checkpoint_name")]
            if self.booleans.with_action_recognition:
                checkpoint_fields.append(("action_recognition", "mart_checkpoint_name"))
            if self.booleans.with_group_action_recognition:
                checkpoint_fields.append(("group_action_recognition", "gmart_checkpoint_name"))

            for section, field_name in checkpoint_fields:
                name = getattr(getattr(self, section), field_name)
                checkpoint_path = os.path.join(deploying_directory, name) if name else deploying_directory
                try:
                    set_runtime_attributes(checkpoint_path)
                except ValueError as e:
                    errors.append(f"{section}.{field_name}: {e} {doc_pointer(section)}")

            hyperparams_path = os.path.join(deploying_directory, self.tracking.hyperparameters_file_name)
            if not os.path.isfile(hyperparams_path):
                errors.append(f"tracking.hyperparameters_file_name does not resolve to a real file: '{hyperparams_path}'. {doc_pointer('tracking')}")

        if self.booleans.with_action_recognition:
            for e in _check_action_recognition_dataset(self.action_recognition.action_recognition_data_root):
                errors.append(f"action_recognition.action_recognition_data_root: {e} {doc_pointer('action_recognition_data_root')}")

        if self.booleans.with_validation:
            errors.extend(self._check_validation_config())

        if errors:
            raise ValueError("\n".join(f"  - {e}" for e in errors))

        return self

    def _check_validation_config(self) -> List[str]:
        import yaml

        errors: List[str] = []
        path = self.validation.validation_configuration_file
        if not os.path.isfile(path):
            errors.append(
                f"validation.validation_configuration_file does not exist: '{path}' (required because "
                f"booleans.with_validation is true). {doc_pointer('validation_configuration_file')}"
            )
            return errors

        try:
            with open(path, "r") as f:
                validation_config = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            errors.append(f"Failed to read validation_configuration_file '{path}': {e}. {doc_pointer('validation_configuration_file')}")
            return errors

        validated_classes = validation_config.get("validated_classes")
        if validated_classes is None:
            errors.append(
                f"'{path}' is missing 'validated_classes', which tells the system which classes to validate. "
                f"{doc_pointer('validation_configuration_file')}"
            )
            return errors
        if not isinstance(validated_classes, list):
            errors.append(
                f"'{path}''s 'validated_classes' must be a list, not {type(validated_classes).__name__}. {doc_pointer('validation_configuration_file')}"
            )
            return errors

        for validated_class in validated_classes:
            nb_class_subjects = self.tracking.num_subjects.get(validated_class)
            if nb_class_subjects is None:
                errors.append(
                    f"'{path}' wants to validate class '{validated_class}', but it has no fixed number of "
                    f"individuals defined in tracking.num_subjects. {doc_pointer('validation_configuration_file')}"
                )
            elif nb_class_subjects <= 0:
                errors.append(
                    f"'{path}' wants to validate class '{validated_class}', but tracking.num_subjects['{validated_class}'] "
                    f"= {nb_class_subjects} — validation requires a fixed, positive subject count (not -1). "
                    f"{doc_pointer('validation_configuration_file')}"
                )

        return errors
