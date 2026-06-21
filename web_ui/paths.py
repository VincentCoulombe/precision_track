"""Single source of truth for path resolution.

The PrecisionTrack README defines path conventions the validators depend on:
- Most paths in ``user_configs.yaml`` are relative to the ``tools/`` directory.
- ``deploying_sanity_check_img_path`` is relative to ``data_root``.
- Checkpoint *name* fields are relative to ``deploying_directory``.
"""

import os
from pathlib import Path

# web_ui/ lives at the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
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
    """Resolve ``value`` relative to ``base`` (both as user-config conventions).

    ``base`` itself is interpreted relative to ``tools/`` when not absolute.
    """
    abs_base = resolve_from_tools(base)
    value = str(value)
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(abs_base, value))


def is_within_repo(path: str) -> bool:
    """True if ``path`` resolves inside the repository root (filesystem-browse guard)."""
    try:
        Path(path).resolve().relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False
