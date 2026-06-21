"""Load/save the validation configuration file (appearance.yaml | aruco.yaml)
and the ReID metainfo's identities / disabled_identities.

All writes are round-tripped with ruamel and backed up with a timestamped
``.bak`` like the main config.
"""

import datetime as _dt
import io
import os
import shutil
from typing import Dict

from ruamel.yaml import YAML

from .paths import resolve_from_tools

APPEARANCE = "AppearanceValidation"
ARUCO = "ArucoValidation"


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    return y


def _to_plain(node):
    if isinstance(node, dict):
        return {k: _to_plain(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_to_plain(v) for v in node]
    return node


def _backup(abs_path: str) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{abs_path}.{ts}.bak"
    shutil.copy2(abs_path, bak)
    return bak


def _dump(abs_path: str, data) -> str:
    backup = _backup(abs_path) if os.path.isfile(abs_path) else ""
    buf = io.StringIO()
    _yaml().dump(data, buf)
    with open(abs_path, "w") as f:
        f.write(buf.getvalue())
    return backup


# Defaults used when the user switches strategy via the `type` field.
APPEARANCE_TEMPLATE = {
    "type": APPEARANCE,
    "data_preprocessor": {"type": "WildLifeReIDPreprocessor"},
    "re_identificator": {"metainfo": "", "checkpoint": ""},
    "validated_classes": [],
}

ARUCO_TEMPLATE = {
    "type": ARUCO,
    "validated_classes": [],
    "num_tags": 32,
    "tags_size": 3,
    "predefined_dict": None,
    "parameters": {
        "minMarkerPerimeterRate": 0.1,
        "maxMarkerPerimeterRate": 0.9,
        "adaptiveThreshWinSizeMin": 7,
        "adaptiveThreshWinSizeMax": 23,
        "adaptiveThreshWinSizeStep": 10,
        "polygonalApproxAccuracyRate": 0.14,
        "minOtsuStdDev": 1,
        "perspectiveRemovePixelPerCell": 13,
        "perspectiveRemoveIgnoredMarginPerCell": 0.35,
    },
    "refinement": "none",
    "tag_kpt": 7,
    "kpt_conf_thr": 0.5,
    "estimation_range": 120,
    "timeout_after": 0.02,
    "min_sample_size": 25,
    "valid_tags": [],
}


def load_validation_config(value: str) -> Dict:
    abs_path = resolve_from_tools(value)
    if not os.path.isfile(abs_path):
        return {"ok": False, "message": f"File not found: {abs_path}"}
    with open(abs_path, "r") as f:
        data = _to_plain(_yaml().load(f))
    strategy = data.get("type", "")
    result = {"ok": True, "path": abs_path, "strategy": strategy, "config": data}
    # If appearance-based, surface the linked ReID metainfo path for the identity editor.
    if strategy == APPEARANCE:
        reid = (data.get("re_identificator") or {}).get("metainfo", "")
        result["reid_metainfo_path"] = reid
    return result


def save_validation_config(value: str, config: Dict) -> Dict:
    abs_path = resolve_from_tools(value)
    if not os.path.isfile(abs_path):
        # Create a fresh file from scratch (round-trip not possible, no comments to keep).
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        backup = _dump(abs_path, config)
        return {"saved": True, "backup": os.path.basename(backup) if backup else None}
    with open(abs_path, "r") as f:
        loaded = _yaml().load(f)
    # Replace top-level keys; nested dicts/lists are replaced wholesale (simpler and
    # the validation files are small / flat enough that comment loss is limited to
    # removed keys only).
    for k in list(loaded.keys()):
        if k not in config:
            del loaded[k]
    for k, v in config.items():
        loaded[k] = v
    backup = _dump(abs_path, loaded)
    return {"saved": True, "backup": os.path.basename(backup) if backup else None}


def template_for(strategy: str) -> Dict:
    return dict(ARUCO_TEMPLATE if strategy == ARUCO else APPEARANCE_TEMPLATE)


def load_reid_metainfo(value: str) -> Dict:
    abs_path = resolve_from_tools(value)
    if not os.path.isfile(abs_path):
        return {"ok": False, "message": f"ReID metainfo not found: {abs_path}"}
    with open(abs_path, "r") as f:
        data = _to_plain(_yaml().load(f)) or {}
    return {
        "ok": True,
        "path": abs_path,
        "identities": data.get("identities", []) or [],
        "disabled_identities": data.get("disabled_identities", []) or [],
    }


def save_reid_metainfo(value: str, identities: list, disabled_identities: list) -> Dict:
    abs_path = resolve_from_tools(value)
    if not os.path.isfile(abs_path):
        return {"saved": False, "message": f"ReID metainfo not found: {abs_path}"}
    # Enforce the README rule: every disabled identity must exist in identities (case-sensitive).
    unknown = [d for d in disabled_identities if d not in identities]
    if unknown:
        return {"saved": False, "message": f"disabled_identities not present in identities: {unknown}"}
    with open(abs_path, "r") as f:
        loaded = _yaml().load(f)
    loaded["identities"] = identities
    if disabled_identities:
        loaded["disabled_identities"] = disabled_identities
    elif "disabled_identities" in loaded:
        del loaded["disabled_identities"]
    backup = _dump(abs_path, loaded)
    return {"saved": True, "backup": os.path.basename(backup) if backup else None}
