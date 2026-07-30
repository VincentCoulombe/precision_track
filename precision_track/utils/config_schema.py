# Copyright (c) Vincent Coulombe
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shape validation for ``user_configs.yaml``.

This module answers "is this file well-formed?" — are all eight sections present, are the
keys known, are the types right. It performs **no filesystem access** and knows nothing about
which tool is running.

"Is this configuration usable?" — does this path exist, is this a COCO dataset, is this class
declared in the metainfo — lives in :mod:`config_fields`, where every check is declared
together with the tools and booleans that make it relevant.

:func:`validate_user_config` runs both and returns a single report, so a user sees every
problem at once instead of one per run.
"""

from typing import Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from .config_fields import (  # noqa: F401  (re-exported for backwards compatibility)
    DOC_LINKS,
    ConfigIssue,
    ValidationReport,
    doc_pointer,
    run_checks,
)


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


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str
    data_root: str
    resume: bool
    training_checkpoint: str
    deploying_directory: str
    deploying_sanity_check_img_path: str
    batch_size: int
    wandb_logging: bool


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    saving_directory: str
    num_subjects: Dict[str, int]
    tracking_checkpoint_name: str
    hyperparameters_file_name: str
    output_clustered_features: bool
    mot_data_root: str


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
    """The *shape* of ``user_configs.yaml``. Semantics live in :mod:`config_fields`."""

    model_config = ConfigDict(extra="forbid")
    booleans: BooleansConfig
    general: GeneralConfig
    training: TrainingConfig
    tracking: TrackingConfig
    action_recognition: ActionRecognitionConfig
    group_action_recognition: GroupActionRecognitionConfig
    validation: ValidationConfig
    visualization: VisualizationConfig


def _shape_issues(exc: ValidationError):
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        yield ConfigIssue(field=loc, message=msg, severity="error")


def format_config_errors(exc: ValidationError) -> str:
    """Render a pydantic ``ValidationError`` the way the tools print it."""
    report = ValidationReport(issues=list(_shape_issues(exc)))
    return report.format()


def validate_user_config(
    cfg: Mapping,
    tool: Optional[str] = None,
    flags: Optional[Mapping[str, bool]] = None,
) -> ValidationReport:
    """Validate ``cfg`` for ``tool`` and return every finding in one report.

    ``tool=None`` validates against the union of all tools, which is the conservative
    reading used when the caller is unknown.

    Shape problems never mask semantic ones: both passes always run, so a user who has a
    typo *and* a bad path is told about both at once.
    """
    report = ValidationReport()
    try:
        UserConfig(**dict(cfg))
    except ValidationError as exc:
        report.issues.extend(_shape_issues(exc))
    except TypeError as exc:  # cfg is not a mapping of sections
        report.issues.append(ConfigIssue(field="", message=str(exc)))
        return report

    # Semantic checks read through ``.get`` with defaults, so they are safe to run even when
    # the shape pass failed — that is what lets us report everything in one go.
    report.issues.extend(run_checks(cfg, tool=tool, flags=flags).issues)
    return report
