"""Round-trip load/save of user_configs.yaml.

Uses ruamel.yaml in round-trip mode so the section banners and inline comments
(e.g. the ``batch_size`` warning) survive every save. The Configure form
auto-saves each field as it is committed, so writes are frequent and **no backup
files are created** — ``user_configs.yaml`` itself is the record of truth.
"""

import io
import os
import threading
from typing import Any, Dict

from ruamel.yaml import YAML

from .paths import USER_CONFIG_PATH, resolve_from_tools

# Serialize writes so concurrent autosaves can't interleave on the file.
_write_lock = threading.Lock()


def _new_yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096  # avoid line wrapping of long paths
    return y


def _to_plain(node: Any) -> Any:
    """Convert ruamel containers to plain python (JSON-serializable) structures."""
    if isinstance(node, dict):
        return {k: _to_plain(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_to_plain(v) for v in node]
    return node


def load_config_plain() -> Dict:
    """Return user_configs.yaml as plain nested dict (for the JSON API)."""
    with open(USER_CONFIG_PATH, "r") as f:
        data = _new_yaml().load(f)
    return _to_plain(data)


def _apply_values(loaded: Any, incoming: Dict) -> None:
    """Recursively write changed values into the ruamel object.

    Only assigns when a value actually differs, so untouched fields keep their
    original formatting (flow-style maps, scalar casing, inline comments).
    """
    for key, value in incoming.items():
        if isinstance(value, dict) and key in loaded and isinstance(loaded[key], dict):
            _apply_values(loaded[key], value)
        elif key not in loaded or _differs(loaded[key], value):
            loaded[key] = value


def _differs(existing: Any, incoming: Any) -> bool:
    try:
        return _to_plain(existing) != incoming
    except Exception:
        return True


def _run_side_effects(cfg: Dict) -> list:
    """Create directories that the config promises to exist. Returns notes."""
    notes = []
    for section, key in [("training", "deploying_directory"), ("tracking", "saving_directory")]:
        value = cfg.get(section, {}).get(key)
        if value:
            abs_path = resolve_from_tools(value)
            if not os.path.isdir(abs_path):
                os.makedirs(abs_path, exist_ok=True)
                notes.append(f"Created {key}: {abs_path}")
    return notes


def write_config(incoming: Dict) -> Dict:
    """Persist the config to user_configs.yaml (no backup) and run side effects.

    Round-trips through ruamel so comments/banners are preserved and only changed
    values are rewritten. Returns the list of directories created as a side effect.
    """
    with _write_lock:
        created = _run_side_effects(incoming)
        with open(USER_CONFIG_PATH, "r") as f:
            loaded = _new_yaml().load(f)
        _apply_values(loaded, incoming)
        buf = io.StringIO()
        _new_yaml().dump(loaded, buf)
        with open(USER_CONFIG_PATH, "w") as f:
            f.write(buf.getvalue())
    return {"created_dirs": created}
