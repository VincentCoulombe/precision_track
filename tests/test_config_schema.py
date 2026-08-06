"""Tests for the context-aware ``user_configs.yaml`` validation.

The point of this layer is that a parameter is only enforced for the tools that actually
read it, and only when the booleans that make it relevant are on. These tests encode that
matrix directly, so a regression in the registry shows up as a failing table entry rather
than as a mysterious ``SystemExit`` in somebody's tracking run.
"""

import copy
import os

import pytest
import yaml

from precision_track.utils.config_fields import (
    AR_TOOLS,
    CONFIG_TOOLS,
    DETECTOR_TOOLS,
    FIELDS_BY_ID,
    PIPELINE_TOOLS,
    run_checks,
)
from precision_track.utils.config_schema import UserConfig, validate_user_config
from precision_track.utils.paths import USER_CONFIG_PATH, resolve_from_tools

ROOT = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def shipped_config():
    with open(USER_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


#: The boolean state the gating tables below start from. ``user_configs.yaml`` is meant to be
#: edited, so the matrix is pinned here rather than inherited from whatever the developer running
#: the suite happens to have enabled locally.
BASELINE_BOOLEANS = dict(
    pipelined=False,
    with_validation=False,
    with_offline_correction_refinement=False,
    with_action_recognition=False,
    with_group_action_recognition=False,
    with_pose_estimation=True,
)


@pytest.fixture
def cfg(shipped_config):
    cfg = copy.deepcopy(shipped_config)
    cfg["booleans"].update(BASELINE_BOOLEANS)
    return cfg


def errors_for(cfg, tool=None, flags=None):
    return {i.field for i in validate_user_config(cfg, tool=tool, flags=flags).errors}


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def test_shipped_config_has_the_expected_shape(shipped_config):
    """The config that ships with the repo must satisfy the pydantic schema."""
    UserConfig(**shipped_config)


def test_path_resolution_is_independent_of_the_working_directory(cfg, tmp_path, monkeypatch):
    """Validation must not depend on where the process was launched from.

    Paths in ``user_configs.yaml`` are relative to ``tools/``; resolving them against the
    process CWD used to invent errors for anyone running from the repository root.
    """
    from_tools = errors_for(cfg, tool="track")
    monkeypatch.chdir(tmp_path)
    assert errors_for(cfg, tool="track") == from_tools


# --------------------------------------------------------------------------- #
# Table 1 — per-tool gating
# --------------------------------------------------------------------------- #
TRAINING_ONLY_BREAKAGE = {
    "training.data_root": lambda c: c["training"].update(data_root="../../does-not-exist/"),
    "training.training_checkpoint": lambda c: c["training"].update(training_checkpoint="../does-not-exist.pth"),
    "training.batch_size": lambda c: c["training"].update(batch_size=4),
}

#: Tools that never read the training block (Table 1).
NON_TRAINING_TOOLS = sorted(PIPELINE_TOOLS | {"visualize"})


@pytest.mark.parametrize("tool", NON_TRAINING_TOOLS)
@pytest.mark.parametrize("field", sorted(TRAINING_ONLY_BREAKAGE))
def test_training_parameters_are_not_enforced_for_non_training_tools(cfg, tool, field):
    TRAINING_ONLY_BREAKAGE[field](cfg)
    assert field not in errors_for(cfg, tool=tool)


@pytest.mark.parametrize("field", sorted(TRAINING_ONLY_BREAKAGE))
def test_training_parameters_are_enforced_for_train_detection(cfg, field):
    TRAINING_ONLY_BREAKAGE[field](cfg)
    assert field in errors_for(cfg, tool="train_detection")


def test_mot_benchmark_is_only_required_by_the_tools_that_read_it(cfg):
    cfg["tracking"]["mot_data_root"] = "../../does-not-exist/"
    assert "tracking.mot_data_root" in errors_for(cfg, tool="test_tracking")
    for tool in ("track", "batch_track_directory", "visualize", "test_detection"):
        assert "tracking.mot_data_root" not in errors_for(cfg, tool=tool), tool


def test_create_mot_dataset_can_bootstrap_from_videos_only(cfg, tmp_path):
    """``create_mot_dataset.py`` *writes* the annotations, so it must not demand them.

    ``test_tracking.py`` reads them, so for that tool the same root is invalid.
    """
    videos = tmp_path / "videos" / "val"
    videos.mkdir(parents=True)
    (videos / "clip.mp4").write_bytes(b"")
    cfg["tracking"]["mot_data_root"] = str(tmp_path)

    assert "tracking.mot_data_root" not in errors_for(cfg, tool="create_mot_dataset")
    assert "tracking.mot_data_root" in errors_for(cfg, tool="test_tracking")


def test_tools_that_read_no_configuration_are_never_blocked(cfg):
    """``plot_profiles`` / ``visualize_appearances`` consume zero parameters."""
    cfg["training"]["data_root"] = "../../does-not-exist/"
    cfg["tracking"]["mot_data_root"] = "../../does-not-exist/"
    cfg["general"]["metainfo"] = "../does-not-exist.py"
    for tool in ("plot_profiles", "visualize_appearances"):
        assert errors_for(cfg, tool=tool) == set(), tool


def test_unknown_tool_validates_conservatively(cfg):
    """``tool=None`` must enforce the union, so an unknown caller is never under-checked."""
    cfg["training"]["data_root"] = "../../does-not-exist/"
    assert "training.data_root" in errors_for(cfg, tool=None)


# --------------------------------------------------------------------------- #
# Table 1, note 2 — CLI flag gating
# --------------------------------------------------------------------------- #
def test_optimize_hyperparams_gates_the_mot_benchmark(cfg):
    cfg["tracking"]["mot_data_root"] = "../../does-not-exist/"
    assert "tracking.mot_data_root" not in errors_for(cfg, tool="train_detection", flags={"optimize_hyperparams": False})
    assert "tracking.mot_data_root" in errors_for(cfg, tool="train_detection", flags={"optimize_hyperparams": True})


def test_deploy_flag_gates_the_sanity_check_image(cfg):
    cfg["training"]["deploying_sanity_check_img_path"] = "images/does-not-exist.jpg"
    assert "training.deploying_sanity_check_img_path" not in errors_for(cfg, tool="train_detection", flags={"deploy": False})
    assert "training.deploying_sanity_check_img_path" in errors_for(cfg, tool="train_detection", flags={"deploy": True})


def test_unspecified_flags_are_read_conservatively(cfg):
    """A caller that does not describe its flags gets the stage-runs reading."""
    cfg["training"]["deploying_sanity_check_img_path"] = "images/does-not-exist.jpg"
    assert "training.deploying_sanity_check_img_path" in errors_for(cfg, tool="train_detection")


# --------------------------------------------------------------------------- #
# Table 2 / 3 — boolean gating
# --------------------------------------------------------------------------- #
def test_action_recognition_parameters_are_gated_on_the_boolean(cfg):
    cfg["action_recognition"]["mart_checkpoint_name"] = "does-not-exist.pth"
    assert "action_recognition.mart_checkpoint_name" not in errors_for(cfg, tool="track")

    cfg["booleans"]["with_action_recognition"] = True
    assert "action_recognition.mart_checkpoint_name" in errors_for(cfg, tool="track")


def test_group_action_recognition_parameters_are_gated_on_the_boolean(cfg):
    cfg["group_action_recognition"]["gmart_checkpoint_name"] = "does-not-exist.pth"
    assert "group_action_recognition.gmart_checkpoint_name" not in errors_for(cfg, tool="track")

    cfg["booleans"]["with_action_recognition"] = True
    cfg["booleans"]["with_group_action_recognition"] = True
    assert "group_action_recognition.gmart_checkpoint_name" in errors_for(cfg, tool="track")


def test_validation_config_is_gated_on_with_validation(cfg):
    cfg["validation"]["validation_configuration_file"] = "../does-not-exist.yaml"
    assert "validation.validation_configuration_file" not in errors_for(cfg, tool="track")

    cfg["booleans"]["with_validation"] = True
    assert "validation.validation_configuration_file" in errors_for(cfg, tool="track")


def test_action_recognition_tools_force_the_boolean_on(cfg):
    """The AR tools always run with AR enabled, whatever the yaml says."""
    cfg["booleans"]["with_action_recognition"] = False
    cfg["action_recognition"]["action_recognition_data_root"] = "../../does-not-exist/"
    for tool in sorted(AR_TOOLS):
        assert "action_recognition.action_recognition_data_root" in errors_for(cfg, tool=tool), tool


#: Table 3 — the legal (pose, AR, GAR, validation, refinement) states.
LEGAL_BOOLEAN_STATES = [
    (False, False, False, False, False),
    (False, False, False, True, False),
    (False, False, False, True, True),
    (True, False, False, False, False),
    (True, False, False, True, False),
    (True, False, False, True, True),
    (True, True, False, False, False),
    (True, True, False, True, False),
    (True, True, False, True, True),
    (True, True, True, False, False),
    (True, True, True, True, False),
    (True, True, True, True, True),
]

ILLEGAL_BOOLEAN_STATES = [
    (False, True, False, False, False),  # AR without pose estimation
    (False, False, True, False, False),  # GAR without AR
    (True, False, True, False, False),  # GAR without AR
]


def _apply_booleans(cfg, state):
    pose, ar, gar, validation, refine = state
    cfg["booleans"].update(
        with_pose_estimation=pose,
        with_action_recognition=ar,
        with_group_action_recognition=gar,
        with_validation=validation,
        with_offline_correction_refinement=refine,
    )


@pytest.mark.parametrize("state", LEGAL_BOOLEAN_STATES)
def test_legal_boolean_combinations_raise_no_cross_section_error(cfg, state):
    _apply_booleans(cfg, state)
    cross_section = {
        "booleans.with_group_action_recognition",
        "booleans.with_pose_estimation",
    }
    assert errors_for(cfg, tool="track") & cross_section == set()


@pytest.mark.parametrize("state", ILLEGAL_BOOLEAN_STATES)
def test_illegal_boolean_combinations_are_rejected(cfg, state):
    _apply_booleans(cfg, state)
    assert errors_for(cfg, tool="track")


def test_inert_offline_refinement_warns_but_does_not_block(cfg):
    cfg["booleans"]["with_validation"] = False
    cfg["booleans"]["with_offline_correction_refinement"] = True
    report = validate_user_config(cfg, tool="track")
    assert report.ok
    assert any(i.field == "booleans.with_offline_correction_refinement" for i in report.warnings)


# --------------------------------------------------------------------------- #
# Reporting behaviour
# --------------------------------------------------------------------------- #
def test_every_problem_is_reported_in_a_single_pass(cfg):
    """A bad field must not mask the cross-section checks (it used to)."""
    cfg["training"]["batch_size"] = 4
    cfg["training"]["data_root"] = "../../does-not-exist/"
    cfg["booleans"]["with_group_action_recognition"] = True

    fields = errors_for(cfg, tool="train_detection")
    assert {"training.batch_size", "training.data_root", "booleans.with_group_action_recognition"} <= fields


def test_shape_errors_do_not_mask_semantic_errors(cfg):
    cfg["visualization"]["display_bounding_box"] = True  # typo -> unknown key
    cfg["training"]["data_root"] = "../../does-not-exist/"

    fields = errors_for(cfg, tool="train_detection")
    assert "visualization.display_bounding_box" in fields
    assert "training.data_root" in fields


def test_missing_hyperparameters_file_is_a_warning_not_an_error(cfg):
    cfg["tracking"]["hyperparameters_file_name"] = "does-not-exist.json"
    report = validate_user_config(cfg, tool="track")
    assert report.ok
    assert any(i.field == "tracking.hyperparameters_file_name" for i in report.warnings)


# --------------------------------------------------------------------------- #
# num_subjects
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value",
    [
        {},  # no classes at all
        {"mouse": 0},  # zero subjects is meaningless
        {"mouse": True},  # bool must not be silently coerced to 1
        {"mouse": -2},  # only -1 means "unbounded"
        {"rat": 5},  # class absent from the metainfo
    ],
)
def test_invalid_num_subjects_is_rejected(cfg, value):
    cfg["tracking"]["num_subjects"] = value
    assert "tracking.num_subjects" in errors_for(cfg, tool="track")


@pytest.mark.parametrize("value", [{"mouse": 1}, {"mouse": 20}, {"mouse": -1}])
def test_valid_num_subjects_is_accepted(cfg, value):
    cfg["tracking"]["num_subjects"] = value
    assert "tracking.num_subjects" not in errors_for(cfg, tool="track")


# --------------------------------------------------------------------------- #
# Registry integrity
# --------------------------------------------------------------------------- #
def test_every_registry_field_targets_a_real_config_key(shipped_config):
    """Guards against a typo in a registry id silently disabling a check."""
    for field_id in FIELDS_BY_ID:
        section, _, key = field_id.partition(".")
        assert section in shipped_config, field_id
        assert key in shipped_config[section], field_id


def test_registry_tools_are_known_tools():
    for fields in FIELDS_BY_ID.values():
        for f in fields:
            if f.tools is None:
                continue
            assert f.tools <= (CONFIG_TOOLS | DETECTOR_TOOLS), f.id


def test_web_ui_and_cli_share_one_implementation(cfg):
    """The UI adapter must report exactly what the registry reports."""
    from web_ui import validators

    cfg["training"]["data_root"] = "../../does-not-exist/"

    ui = validators.validate_field("training.data_root", cfg)
    registry = run_checks(cfg, only="training.data_root")

    assert ui["level"] == "error"
    assert ui["message"] == registry.errors[0].message

    ui_all = {e["field"] for e in validators.validate_all(cfg, tool="train_detection")["errors"]}
    assert ui_all == errors_for(cfg, tool="train_detection")
