# Copyright (c) Vincent Coulombe
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

# precision_track/precision_track/utils/paths.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
CONFIGS_DIR = REPO_ROOT / "configs"
USER_CONFIG_PATH = CONFIGS_DIR / "user_configs.yaml"
ASSETS_DIR = REPO_ROOT / "assets"


def resolve_from_tools(value: str) -> str:
    """Resolve a user-config path (relative to ``tools/``) to an absolute path."""
    value = str(value)
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(str(TOOLS_DIR / value))


def resolve_from(base: str, value: str) -> str:
    """Resolve ``value`` relative to ``base``, itself relative to ``tools/`` when not absolute."""
    value = str(value)
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(resolve_from_tools(base), value))


def is_within_repo(path: str) -> bool:
    """True if ``path`` resolves inside the repository root (filesystem-browse guard)."""
    try:
        Path(path).resolve().relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False
