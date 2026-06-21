"""Per-field validation that reuses PrecisionTrack's own utilities.

Every validator returns ``(ok: bool, message: str)``. Nothing here re-implements
dataset/metainfo checking — it delegates to ``precision_track.utils`` so the UI
agrees exactly with what the tools will accept at launch time.
"""

import os
from typing import Any, Callable, Dict, Tuple

from precision_track.utils import (
    assert_coco_dataset_directory,
    check_if_mot_dataset_is_ok,
    parse_pose_metainfo,
)

from .paths import resolve_from, resolve_from_tools

Result = Tuple[bool, str]


def _get(cfg: Dict, section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, default)


# --------------------------------------------------------------------------- #
# Individual field validators. Each takes the full (nested) config dict.
# --------------------------------------------------------------------------- #
def v_metainfo(cfg: Dict) -> Result:
    value = _get(cfg, "general", "metainfo", "")
    if not value:
        return False, "No metainfo file provided."
    abs_path = resolve_from_tools(value)
    if not os.path.isfile(abs_path):
        return False, f"Metainfo file not found: {abs_path}"
    try:
        parse_pose_metainfo({"from_file": abs_path})
    except Exception as exc:  # AssertionError, FileNotFoundError, parsing errors
        return False, f"Metainfo failed to load: {exc}"
    return True, "Metainfo loads correctly."


def v_data_root(cfg: Dict) -> Result:
    value = _get(cfg, "training", "data_root", "")
    if not value:
        return False, "No data_root provided."
    abs_path = resolve_from_tools(value)
    try:
        assert_coco_dataset_directory(abs_path)
    except Exception as exc:  # NotADirectoryError, FileNotFoundError, AssertionError
        return False, f"Invalid COCO dataset: {exc}"
    return True, "Valid COCO dataset directory."


def v_training_checkpoint(cfg: Dict) -> Result:
    value = _get(cfg, "training", "training_checkpoint", "")
    if not value or str(value).lower() == "none":
        return True, "No training checkpoint (training from scratch)."
    abs_path = resolve_from_tools(value)
    if not os.path.isfile(abs_path):
        return False, f"Training checkpoint not found: {abs_path}"
    return True, "Training checkpoint exists."


def v_deploying_sanity_check_img(cfg: Dict) -> Result:
    data_root = _get(cfg, "training", "data_root", "")
    value = _get(cfg, "training", "deploying_sanity_check_img_path", "")
    if not value:
        return False, "No sanity-check image provided."
    abs_path = resolve_from(data_root, value)
    if not os.path.isfile(abs_path):
        return False, f"Sanity-check image not found: {abs_path}"
    return True, "Sanity-check image exists."


def v_batch_size(cfg: Dict) -> Result:
    value = _get(cfg, "training", "batch_size")
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return False, f"batch_size must be an integer, got: {value!r}"
    if ivalue < 16:
        return False, f"batch_size must be at least 16 (training is unstable below 16). Got {ivalue}."
    return True, f"batch_size = {ivalue}."


def v_num_subjects(cfg: Dict) -> Result:
    value = _get(cfg, "tracking", "num_subjects")
    if not isinstance(value, dict) or not value:
        return False, "num_subjects must be a non-empty mapping of class name -> count."
    for k, v in value.items():
        if not isinstance(k, str):
            return False, f"num_subjects keys must be class names (strings), got: {k!r}"
        if not isinstance(v, int) or isinstance(v, bool):
            return False, f"num_subjects['{k}'] must be an integer, got: {v!r}"
    # Class-coherence with the metainfo is a soft warning, handled in validate_all.
    return True, "num_subjects is a valid dict[str, int]."


def _checkpoint_in_deploy(cfg: Dict, name: str, allow_empty: bool) -> Result:
    deploy_dir = _get(cfg, "training", "deploying_directory", "")
    if not name:
        if allow_empty:
            return True, "Empty — PrecisionTrack will auto-select a checkpoint."
        return False, "No checkpoint name provided."
    if not deploy_dir:
        return False, "deploying_directory is not set, cannot locate the checkpoint."
    abs_path = resolve_from(deploy_dir, name)
    if not os.path.isfile(abs_path):
        return False, f"Checkpoint not found in deploying_directory: {abs_path}"
    return True, "Checkpoint exists in deploying_directory."


def v_tracking_checkpoint_name(cfg: Dict) -> Result:
    return _checkpoint_in_deploy(cfg, _get(cfg, "tracking", "tracking_checkpoint_name", ""), allow_empty=True)


def v_hyperparameters_file_name(cfg: Dict) -> Result:
    return _checkpoint_in_deploy(cfg, _get(cfg, "tracking", "hyperparameters_file_name", ""), allow_empty=False)


def v_mot_data_root(cfg: Dict) -> Result:
    value = _get(cfg, "tracking", "mot_data_root", "")
    if not value:
        return False, "No mot_data_root provided."
    ok, msg = check_if_mot_dataset_is_ok(resolve_from_tools(value))
    return ok, msg if msg else "Valid MOT dataset directory."


def v_mart_checkpoint_name(cfg: Dict) -> Result:
    return _checkpoint_in_deploy(cfg, _get(cfg, "action_recognition", "mart_checkpoint_name", ""), allow_empty=False)


def v_action_recognition_data_root(cfg: Dict) -> Result:
    value = _get(cfg, "action_recognition", "action_recognition_data_root", "")
    if not value:
        return False, "No action_recognition_data_root provided."
    ok, msg = check_if_mot_dataset_is_ok(resolve_from_tools(value))
    return ok, msg if msg else "Valid MOT dataset directory."


def v_gmart_checkpoint_name(cfg: Dict) -> Result:
    return _checkpoint_in_deploy(cfg, _get(cfg, "group_action_recognition", "gmart_checkpoint_name", ""), allow_empty=False)


def v_validation_configuration_file(cfg: Dict) -> Result:
    value = _get(cfg, "validation", "validation_configuration_file", "")
    if not value:
        return False, "No validation configuration file provided."
    abs_path = resolve_from_tools(value)
    if not os.path.isfile(abs_path):
        return False, f"Validation configuration file not found: {abs_path}"
    return True, "Validation configuration file exists."


# Map dotted field id -> validator. These are the fields with real checks;
# booleans and visualization flags are not listed (no verification needed).
VALIDATORS: Dict[str, Callable[[Dict], Result]] = {
    "general.metainfo": v_metainfo,
    "training.data_root": v_data_root,
    "training.training_checkpoint": v_training_checkpoint,
    "training.deploying_sanity_check_img_path": v_deploying_sanity_check_img,
    "training.batch_size": v_batch_size,
    "tracking.num_subjects": v_num_subjects,
    "tracking.tracking_checkpoint_name": v_tracking_checkpoint_name,
    "tracking.hyperparameters_file_name": v_hyperparameters_file_name,
    "tracking.mot_data_root": v_mot_data_root,
    "action_recognition.mart_checkpoint_name": v_mart_checkpoint_name,
    "action_recognition.action_recognition_data_root": v_action_recognition_data_root,
    "group_action_recognition.gmart_checkpoint_name": v_gmart_checkpoint_name,
    "validation.validation_configuration_file": v_validation_configuration_file,
}

# Fields only relevant when a feature toggle is on. Used by validate_all so we
# don't flag (e.g.) action-recognition paths for a tracking-only workflow.
_AR_FIELDS = {
    "action_recognition.mart_checkpoint_name",
    "action_recognition.action_recognition_data_root",
    "group_action_recognition.gmart_checkpoint_name",
}
_VALIDATION_FIELDS = {"validation.validation_configuration_file"}

# Fields whose failure is a non-blocking warning rather than an error: the file
# may legitimately not exist yet (e.g. hyperparameters.json is generated by
# train_detection's --calibrate / --optimize_hyperparams step).
_SOFT_FIELDS = {"tracking.hyperparameters_file_name"}


def validate_field(field: str, cfg: Dict) -> Dict[str, Any]:
    fn = VALIDATORS.get(field)
    if fn is None:
        return {"field": field, "ok": True, "level": "ok", "message": ""}
    ok, message = fn(cfg)
    if not ok and field in _SOFT_FIELDS:
        # Non-blocking: surface as a warning, keep the field "valid" for saving.
        return {"field": field, "ok": True, "level": "warning", "message": message}
    return {"field": field, "ok": ok, "level": "ok" if ok else "error", "message": message}


def _metainfo_classes(cfg: Dict):
    try:
        abs_path = resolve_from_tools(_get(cfg, "general", "metainfo", ""))
        meta = parse_pose_metainfo({"from_file": abs_path})
        return meta.get("classes", [])
    except Exception:
        return None


def validate_all(cfg: Dict) -> Dict[str, list]:
    """Validate every relevant field. Returns {errors:[...], warnings:[...]}.

    Feature-gated fields are skipped when their toggle is off. ``num_subjects``
    class-coherence with the metainfo is reported as a (non-blocking) warning.
    """
    booleans = cfg.get("booleans", {})
    with_ar = bool(booleans.get("with_action_recognition"))
    with_val = bool(booleans.get("with_validation"))

    errors, warnings = [], []
    for field in VALIDATORS:
        if field in _AR_FIELDS and not with_ar:
            continue
        if field in _VALIDATION_FIELDS and not with_val:
            continue
        ok, message = VALIDATORS[field](cfg)
        if not ok:
            bucket = warnings if field in _SOFT_FIELDS else errors
            bucket.append({"field": field, "message": message})

    # Soft warning: num_subjects classes should exist in the metainfo.
    num_subjects = _get(cfg, "tracking", "num_subjects")
    if isinstance(num_subjects, dict):
        classes = _metainfo_classes(cfg)
        if classes is not None:
            for cls in num_subjects:
                if cls not in classes:
                    warnings.append(
                        {
                            "field": "tracking.num_subjects",
                            "message": f"Class '{cls}' is not defined in the metainfo classes {classes}.",
                        }
                    )
    return {"errors": errors, "warnings": warnings}
