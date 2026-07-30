"""Path resolution for the web UI.

The conventions (most paths relative to ``tools/``; ``deploying_sanity_check_img_path``
relative to ``data_root``; checkpoint names relative to ``deploying_directory``) are defined
once in :mod:`precision_track.utils.paths` and shared with the CLI validators, so the UI and
the tools always resolve a config path to the same place.
"""

from precision_track.utils.paths import (  # noqa: F401
    ASSETS_DIR,
    CONFIGS_DIR,
    REPO_ROOT,
    TOOLS_DIR,
    USER_CONFIG_PATH,
    is_within_repo,
    resolve_from,
    resolve_from_tools,
)
