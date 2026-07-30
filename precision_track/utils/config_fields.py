# Copyright (c) Vincent Coulombe
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .datasets import assert_coco_dataset_directory, check_if_mot_dataset_is_ok, parse_pose_metainfo
from .deployment import set_runtime_attributes
from .io import SUPPORTED_VIDEO_BACKEND
from .paths import resolve_from, resolve_from_tools

# --------------------------------------------------------------------------- #
# Documentation pointers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
TRAIN_DETECTION = "train_detection"
TEST_DETECTION = "test_detection"
TRAIN_ACTION_RECOGNITION = "train_action_recognition"
TEST_ACTION_RECOGNITION = "test_action_recognition"
TEST_TRACKING = "test_tracking"
TRACK = "track"
BATCH_TRACK_DIRECTORY = "batch_track_directory"
CREATE_MOT_DATASET = "create_mot_dataset"
VISUALIZE = "visualize"
PLOT_PROFILES = "plot_profiles"
VISUALIZE_APPEARANCES = "visualize_appearances"

#: Tools that read ``user_configs.yaml`` at all.
CONFIG_TOOLS = frozenset(
    {
        TRAIN_DETECTION,
        TEST_DETECTION,
        TRAIN_ACTION_RECOGNITION,
        TEST_ACTION_RECOGNITION,
        TEST_TRACKING,
        TRACK,
        BATCH_TRACK_DIRECTORY,
        CREATE_MOT_DATASET,
        VISUALIZE,
    }
)

ALL_TOOLS = CONFIG_TOOLS | {PLOT_PROFILES, VISUALIZE_APPEARANCES}

#: Tools that build a live tracking pipeline (detector + assigner) from the deployed checkpoints.
PIPELINE_TOOLS = frozenset({TRACK, BATCH_TRACK_DIRECTORY, CREATE_MOT_DATASET, TEST_TRACKING})
#: Tools that train or evaluate MART / GMART.
AR_TOOLS = frozenset({TRAIN_ACTION_RECOGNITION, TEST_ACTION_RECOGNITION})
#: Tools that need a real detector checkpoint on disk.
DETECTOR_TOOLS = PIPELINE_TOOLS | AR_TOOLS
#: Tools that resolve the validation configuration (``visualize`` does so via ``load_writers``).
VALIDATION_TOOLS = PIPELINE_TOOLS | {VISUALIZE}


# --------------------------------------------------------------------------- #
# Context + findings
# --------------------------------------------------------------------------- #
@dataclass
class ValidationContext:
    """What the config is being validated *for*.

    ``tool=None`` means "unknown caller": every tool-gated check runs, which reproduces the
    historical all-or-nothing behaviour and is the safe default for the web UI's
    whole-config view.
    """

    tool: Optional[str] = None
    flags: Mapping[str, bool] = field(default_factory=dict)
    booleans: Mapping[str, bool] = field(default_factory=dict)
    _metainfo_cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    def flag(self, name: str) -> bool:
        # Unknown flags default to True: a caller that does not describe its flags gets the
        # conservative reading (the stage runs, so the parameter matters).
        return bool(self.flags.get(name, True))

    def boolean(self, name: str) -> bool:
        return bool(self.booleans.get(name, False))

    def metainfo(self, path: str):
        """Parse a metainfo file at most once per validation run (it is not cheap)."""
        key = str(path)
        if key not in self._metainfo_cache:
            try:
                self._metainfo_cache[key] = parse_pose_metainfo(dict(from_file=resolve_from_tools(path)))
            except Exception:
                self._metainfo_cache[key] = None
        return self._metainfo_cache[key]


@dataclass(frozen=True)
class ConfigIssue:
    field: str
    message: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"[{self.field}] {self.message}" if self.field else self.message


@dataclass
class ValidationReport:
    issues: List[ConfigIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ConfigIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ConfigIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = ["Invalid 'user_configs.yaml':"]
        lines += [f"  - {issue}" for issue in self.errors]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  - {issue}" for issue in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Field declarations
# --------------------------------------------------------------------------- #
Check = Callable[[Mapping, ValidationContext], Optional[str]]


@dataclass(frozen=True)
class ConfigField:
    """One semantic check, plus the conditions under which it is relevant."""

    id: str
    check: Check
    tools: Optional[FrozenSet[str]] = None
    when: Optional[Callable[[ValidationContext], bool]] = None
    flags: Tuple[str, ...] = ()
    severity: str = "error"

    def applies(self, ctx: ValidationContext) -> bool:
        if ctx.tool is not None and self.tools is not None and ctx.tool not in self.tools:
            return False
        if self.when is not None and not self.when(ctx):
            return False
        return all(ctx.flag(f) for f in self.flags)

    def run(self, cfg: Mapping, ctx: ValidationContext) -> Optional[ConfigIssue]:
        message = self.check(cfg, ctx)
        if message is None:
            return None
        return ConfigIssue(field=self.id, message=message, severity=self.severity)


def get(cfg: Mapping, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, default)


# --- boolean predicates (Table 2) ------------------------------------------- #
def _ar_on(ctx: ValidationContext) -> bool:
    return ctx.boolean("with_action_recognition") or ctx.boolean("with_group_action_recognition")


def _gar_on(ctx: ValidationContext) -> bool:
    return ctx.boolean("with_group_action_recognition")


def _validation_on(ctx: ValidationContext) -> bool:
    return ctx.boolean("with_validation")


# --- individual checks ------------------------------------------------------ #
def _check_metainfo(cfg, ctx):
    value = get(cfg, "general", "metainfo", "")
    if not str(value).strip():
        return f"No metainfo file provided. {doc_pointer('metainfo')}"
    if ctx.metainfo(value) is None:
        abs_path = resolve_from_tools(value)
        try:
            parse_pose_metainfo(dict(from_file=abs_path))
        except Exception as exc:
            return f"Invalid metainfo file '{value}' (resolved to '{abs_path}'): {exc} {doc_pointer('metainfo')}"
    return None


def _check_dataset_name(cfg, ctx):
    if not str(get(cfg, "training", "dataset_name", "")).strip():
        return f"dataset_name must not be blank — it names the work sub-directories. {doc_pointer('training')}"
    return None


def _check_data_root(cfg, ctx):
    value = get(cfg, "training", "data_root", "")
    if not str(value).strip():
        return f"No data_root provided. {doc_pointer('data_root')}"
    try:
        assert_coco_dataset_directory(resolve_from_tools(value))
    except Exception as exc:
        return f"Invalid COCO-style dataset at data_root='{value}': {exc} {doc_pointer('data_root')}"
    return None


def _check_training_checkpoint(cfg, ctx):
    value = get(cfg, "training", "training_checkpoint", "")
    if not str(value).strip() or str(value).lower() == "none":
        return None  # training from scratch is legitimate
    if not os.path.isfile(resolve_from_tools(value)):
        return (
            f"training_checkpoint does not exist: '{value}' (resolved to "
            f"'{resolve_from_tools(value)}'). This should be a .pth checkpoint used to initialize "
            f"training (e.g. transfer learning from the AP checkpoint), or blank to train from "
            f"scratch. {doc_pointer('training')}"
        )
    return None


def _check_deploying_directory(cfg, ctx):
    value = get(cfg, "training", "deploying_directory", "")
    if not str(value).strip():
        return (
            f"training.deploying_directory is blank. Every checkpoint field "
            f"(tracking_checkpoint_name, mart_checkpoint_name, gmart_checkpoint_name, "
            f"hyperparameters_file_name) is resolved as 'deploying_directory/<name>', so an empty "
            f"value would break all of them. {doc_pointer('training')}"
        )
    return None


def _check_sanity_check_img(cfg, ctx):
    data_root = get(cfg, "training", "data_root", "")
    value = get(cfg, "training", "deploying_sanity_check_img_path", "")
    if not str(value).strip():
        return f"No deploying_sanity_check_img_path provided. {doc_pointer('training')}"
    abs_path = resolve_from(data_root, value)
    if not os.path.isfile(abs_path):
        return (
            f"deploying_sanity_check_img_path does not resolve to a real file: '{abs_path}' "
            f"(this path is relative to data_root, not to the tools/ directory). {doc_pointer('training')}"
        )
    return None


def _check_batch_size(cfg, ctx):
    value = get(cfg, "training", "batch_size")
    if not isinstance(value, int) or isinstance(value, bool):
        return f"batch_size must be an integer, got: {value!r}. {doc_pointer('training')}"
    if value < 16:
        return f"batch_size must be at least 16 — training is unstable below that. Got {value}. {doc_pointer('training')}"
    return None


def _check_saving_directory(cfg, ctx):
    if not str(get(cfg, "tracking", "saving_directory", "")).strip():
        return (
            f"tracking.saving_directory is blank. This is where PrecisionTrack writes all tracking "
            f"outputs (tracked_bboxes.csv, tracked_kpts.csv, ...), and where the visualization tool "
            f"reads from — it can't be left empty. {doc_pointer('tracking')}"
        )
    return None


def _check_num_subjects(cfg, ctx):
    value = get(cfg, "tracking", "num_subjects")
    if not isinstance(value, dict) or not value:
        return f"num_subjects must be a non-empty mapping of class name -> count, e.g. " f"{{\"mouse\": 20}}. {doc_pointer('tracking')}"
    for cls_name, count in value.items():
        if not isinstance(count, int) or isinstance(count, bool):
            return f"num_subjects['{cls_name}'] must be an integer, got: {count!r}. {doc_pointer('tracking')}"
        if count == 0 or count < -1:
            return (
                f"num_subjects['{cls_name}'] = {count} is invalid — must be a positive integer, or "
                f"-1 if subjects can freely enter/leave the scene. {doc_pointer('tracking')}"
            )
    meta = ctx.metainfo(get(cfg, "general", "metainfo", ""))
    if meta is not None:
        classes = set(meta.get("classes", []))
        unknown = [c for c in value if c not in classes]
        if unknown:
            return (
                f"num_subjects references class(es) {unknown}, which are not defined in your metainfo "
                f"file's 'classes' list ({sorted(classes)}). {doc_pointer('tracking')}"
            )
    return None


def _checkpoint_check(section: str, key: str, allow_empty: bool) -> Check:
    """Build a check for a checkpoint *name* resolved inside ``deploying_directory``."""

    def _check(cfg, ctx):
        name = str(get(cfg, section, key, "") or "").strip()
        deploy_dir = str(get(cfg, "training", "deploying_directory", "") or "").strip()
        if not deploy_dir:
            return None  # already reported by training.deploying_directory
        abs_dir = resolve_from_tools(deploy_dir)
        if not name:
            if not allow_empty:
                return f"{section}.{key} is blank. Set it to the name of the checkpoint inside " f"'{deploy_dir}'. {doc_pointer(section)}"
            if not os.path.isdir(abs_dir):
                return (
                    f"{key} is blank, so PrecisionTrack auto-selects a checkpoint from "
                    f"deploying_directory — but '{abs_dir}' does not exist. {doc_pointer(section)}"
                )
            target = abs_dir
        else:
            target = resolve_from(deploy_dir, name)
        try:
            set_runtime_attributes(target)
        except ValueError as exc:
            return f"{exc} {doc_pointer(section)}"
        return None

    return _check


def _check_hyperparameters_file(cfg, ctx):
    name = str(get(cfg, "tracking", "hyperparameters_file_name", "") or "").strip()
    deploy_dir = str(get(cfg, "training", "deploying_directory", "") or "").strip()
    if not name or not deploy_dir:
        return None
    abs_path = resolve_from(deploy_dir, name)
    if not os.path.isfile(abs_path):
        return (
            f"tracking.hyperparameters_file_name does not resolve to a real file: '{abs_path}'. "
            f"PrecisionTrack will fall back to default values for every missing hyperparameter. "
            f"{doc_pointer('tracking')}"
        )
    return None


def _mot_check(require_annotations: bool) -> Check:
    def _check(cfg, ctx):
        value = get(cfg, "tracking", "mot_data_root", "")
        if not str(value).strip():
            return f"No mot_data_root provided. {doc_pointer('tracking')}"
        ok, msg = check_if_mot_dataset_is_ok(resolve_from_tools(value), require_annotations=require_annotations)
        if not ok:
            return f"Invalid MOT dataset at mot_data_root='{value}': {msg} {doc_pointer('tracking')}"
        return None

    return _check


def _check_action_recognition_data_root(cfg, ctx):
    value = get(cfg, "action_recognition", "action_recognition_data_root", "")
    if not str(value).strip():
        return f"No action_recognition_data_root provided. {doc_pointer('action_recognition_data_root')}"
    root = Path(resolve_from_tools(value))
    errors: List[str] = []
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
            for sub in ("bboxes", "keypoints", "actions"):
                expected = root / sub / split / f"{video_file.stem}.csv"
                if not expected.is_file():
                    errors.append(f"missing {sub}/{split}/{video_file.stem}.csv for video '{video_file.name}'")
    if errors:
        return "; ".join(errors) + f" {doc_pointer('action_recognition_data_root')}"
    return None


def _check_validation_configuration_file(cfg, ctx):
    import yaml

    path = get(cfg, "validation", "validation_configuration_file", "")
    if not str(path).strip():
        return f"No validation configuration file provided (required because booleans.with_validation is true). {doc_pointer('validation_configuration_file')}"
    abs_path = resolve_from_tools(path)
    if not os.path.isfile(abs_path):
        return (
            f"validation.validation_configuration_file does not exist: '{abs_path}' (required "
            f"because booleans.with_validation is true). {doc_pointer('validation_configuration_file')}"
        )
    try:
        with open(abs_path, "r") as f:
            validation_config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        return f"Failed to read validation_configuration_file '{abs_path}': {exc}. {doc_pointer('validation_configuration_file')}"

    validated_classes = (validation_config or {}).get("validated_classes")
    if validated_classes is None:
        return (
            f"'{abs_path}' is missing 'validated_classes', which tells the system which classes to "
            f"validate. {doc_pointer('validation_configuration_file')}"
        )
    if not isinstance(validated_classes, list):
        return f"'{abs_path}''s 'validated_classes' must be a list, not " f"{type(validated_classes).__name__}. {doc_pointer('validation_configuration_file')}"

    num_subjects = get(cfg, "tracking", "num_subjects") or {}
    problems = []
    for validated_class in validated_classes:
        count = num_subjects.get(validated_class) if isinstance(num_subjects, dict) else None
        if count is None:
            problems.append(f"wants to validate class '{validated_class}', but it has no fixed number of individuals defined in tracking.num_subjects")
        elif not isinstance(count, int) or count <= 0:
            problems.append(
                f"wants to validate class '{validated_class}', but tracking.num_subjects['{validated_class}'] = {count} "
                f"— validation requires a fixed, positive subject count (not -1)"
            )
    if problems:
        return f"'{abs_path}' " + "; ".join(problems) + f". {doc_pointer('validation_configuration_file')}"
    return None


# --- cross-section checks --------------------------------------------------- #
def _check_gar_requires_ar(cfg, ctx):
    b = cfg.get("booleans", {})
    if b.get("with_group_action_recognition") and not b.get("with_action_recognition"):
        return (
            "booleans.with_group_action_recognition is true but booleans.with_action_recognition is "
            "false — group action recognition requires action recognition to also be enabled. " + doc_pointer("group_action_recognition")
        )
    return None


def _check_ar_requires_pose(cfg, ctx):
    b = cfg.get("booleans", {})
    if (b.get("with_action_recognition") or b.get("with_group_action_recognition")) and not b.get("with_pose_estimation"):
        return (
            "booleans.with_pose_estimation is false, but action/group-action recognition is enabled "
            "— pose estimation is mandatory for action recognition. " + doc_pointer("booleans")
        )
    return None


def _check_refine_requires_validation(cfg, ctx):
    b = cfg.get("booleans", {})
    if b.get("with_offline_correction_refinement") and not b.get("with_validation"):
        return (
            "booleans.with_offline_correction_refinement is true but booleans.with_validation is "
            "false — offline correction refinement refines the corrections made by re-identification, "
            "so it does nothing on its own. " + doc_pointer("booleans")
        )
    return None


def _display_requires(flag: str, boolean: str, why: str) -> Check:
    def _check(cfg, ctx):
        if cfg.get("visualization", {}).get(flag) and not cfg.get("booleans", {}).get(boolean):
            return f"visualization.{flag} is true but booleans.{boolean} is false — {why}. " + doc_pointer("visualization")
        return None

    return _check


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
FIELDS: Sequence[ConfigField] = (
    # --- general ---
    ConfigField("general.metainfo", _check_metainfo, tools=CONFIG_TOOLS),
    # --- training ---
    ConfigField(
        "training.dataset_name",
        _check_dataset_name,
        tools=frozenset({TRAIN_DETECTION, TEST_DETECTION, TEST_TRACKING}) | AR_TOOLS,
    ),
    ConfigField("training.data_root", _check_data_root, tools=frozenset({TRAIN_DETECTION, TEST_DETECTION})),
    ConfigField("training.data_root", _check_data_root, tools=frozenset({TRAIN_ACTION_RECOGNITION}), flags=("deploy",)),
    ConfigField("training.training_checkpoint", _check_training_checkpoint, tools=frozenset({TRAIN_DETECTION})),
    ConfigField("training.deploying_directory", _check_deploying_directory, tools=DETECTOR_TOOLS | {TEST_DETECTION}),
    ConfigField(
        "training.deploying_directory",
        _check_deploying_directory,
        tools=frozenset({TRAIN_DETECTION}),
        flags=("deploy",),
    ),
    ConfigField(
        "training.deploying_sanity_check_img_path",
        _check_sanity_check_img,
        tools=frozenset({TRAIN_DETECTION, TRAIN_ACTION_RECOGNITION}),
        flags=("deploy",),
    ),
    ConfigField(
        "training.batch_size",
        _check_batch_size,
        tools=frozenset({TRAIN_DETECTION, TEST_DETECTION, TRAIN_ACTION_RECOGNITION}),
    ),
    # --- tracking ---
    ConfigField("tracking.saving_directory", _check_saving_directory, tools=PIPELINE_TOOLS | {VISUALIZE}),
    ConfigField(
        "tracking.num_subjects",
        _check_num_subjects,
        tools=PIPELINE_TOOLS | {VISUALIZE, TRAIN_ACTION_RECOGNITION},
    ),
    ConfigField(
        "tracking.tracking_checkpoint_name",
        _checkpoint_check("tracking", "tracking_checkpoint_name", allow_empty=True),
        tools=DETECTOR_TOOLS,
    ),
    ConfigField(
        "tracking.hyperparameters_file_name",
        _check_hyperparameters_file,
        tools=DETECTOR_TOOLS,
        severity="warning",
    ),
    ConfigField("tracking.mot_data_root", _mot_check(require_annotations=True), tools=frozenset({TEST_TRACKING})),
    ConfigField(
        "tracking.mot_data_root",
        _mot_check(require_annotations=True),
        tools=frozenset({TRAIN_DETECTION}),
        flags=("optimize_hyperparams",),
    ),
    # create_mot_dataset PRODUCES the annotations, so it only needs the videos.
    ConfigField(
        "tracking.mot_data_root",
        _mot_check(require_annotations=False),
        tools=frozenset({CREATE_MOT_DATASET}),
    ),
    # --- action recognition ---
    ConfigField(
        "action_recognition.mart_checkpoint_name",
        _checkpoint_check("action_recognition", "mart_checkpoint_name", allow_empty=False),
        tools=DETECTOR_TOOLS,
        when=_ar_on,
    ),
    ConfigField(
        "action_recognition.action_recognition_data_root",
        _check_action_recognition_data_root,
        tools=AR_TOOLS,
        when=_ar_on,
    ),
    # --- group action recognition ---
    ConfigField(
        "group_action_recognition.gmart_checkpoint_name",
        _checkpoint_check("group_action_recognition", "gmart_checkpoint_name", allow_empty=False),
        tools=DETECTOR_TOOLS,
        when=_gar_on,
    ),
    # --- validation ---
    ConfigField(
        "validation.validation_configuration_file",
        _check_validation_configuration_file,
        tools=VALIDATION_TOOLS,
        when=_validation_on,
    ),
    # --- cross-section ---
    ConfigField("booleans.with_group_action_recognition", _check_gar_requires_ar),
    ConfigField("booleans.with_pose_estimation", _check_ar_requires_pose),
    ConfigField(
        "booleans.with_offline_correction_refinement",
        _check_refine_requires_validation,
        severity="warning",
    ),
    ConfigField(
        "visualization.display_actions",
        _display_requires("display_actions", "with_action_recognition", "no actions will be rendered"),
        tools=frozenset({VISUALIZE}),
        severity="warning",
    ),
    ConfigField(
        "visualization.display_validations",
        _display_requires("display_validations", "with_validation", "no validations will be rendered"),
        tools=frozenset({VISUALIZE}),
        severity="warning",
    ),
    ConfigField(
        "visualization.display_poses",
        _display_requires("display_poses", "with_pose_estimation", "no keypoints will be tracked"),
        tools=frozenset({VISUALIZE}),
        severity="warning",
    ),
)

#: Dotted field id -> the checks declared for it (a field may have several, e.g. one per tool).
FIELDS_BY_ID: Dict[str, List[ConfigField]] = {}
for _f in FIELDS:
    FIELDS_BY_ID.setdefault(_f.id, []).append(_f)


def make_context(cfg: Mapping, tool: Optional[str] = None, flags: Optional[Mapping[str, bool]] = None) -> ValidationContext:
    """Build the context for ``cfg``, honouring the tools that force a boolean on.

    ``train_action_recognition.py`` / ``test_action_recognition.py`` always run with action
    recognition enabled, whatever the yaml says. That is modelled here rather than by mutating
    the user's config.
    """
    booleans = dict(cfg.get("booleans", {}))
    if tool in AR_TOOLS:
        booleans["with_action_recognition"] = True
    return ValidationContext(tool=tool, flags=dict(flags or {}), booleans=booleans)


def run_checks(
    cfg: Mapping,
    tool: Optional[str] = None,
    flags: Optional[Mapping[str, bool]] = None,
    only: Optional[str] = None,
) -> ValidationReport:
    """Run every applicable semantic check and collect *all* findings.

    ``only`` restricts the run to a single dotted field id (the web UI's per-field mode).
    """
    ctx = make_context(cfg, tool=tool, flags=flags)
    fields = FIELDS_BY_ID.get(only, []) if only is not None else FIELDS
    report = ValidationReport()
    seen = set()
    for f in fields:
        if not f.applies(ctx):
            continue
        issue = f.run(cfg, ctx)
        if issue is not None and (issue.field, issue.message) not in seen:
            seen.add((issue.field, issue.message))
            report.issues.append(issue)
    return report
