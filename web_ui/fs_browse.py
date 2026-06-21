"""Filesystem listing for the modal file/folder picker.

This is a local single-user tool and datasets routinely live outside the repo
(e.g. ``../../datasets/...``), so browsing is not restricted to the repo root.
Hidden entries are omitted; an optional extension filter narrows file results.
"""

import os
from typing import Dict, List, Optional

from .paths import REPO_ROOT


def list_dir(path: Optional[str], dirs_only: bool = False, exts: Optional[List[str]] = None) -> Dict:
    base = os.path.abspath(path) if path else str(REPO_ROOT)
    if not os.path.isdir(base):
        base = os.path.dirname(base) or "/"

    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in (exts or [])]
    entries = []
    try:
        for name in sorted(os.listdir(base), key=str.lower):
            if name.startswith("."):
                continue
            full = os.path.join(base, name)
            is_dir = os.path.isdir(full)
            if not is_dir:
                if dirs_only:
                    continue
                if exts and os.path.splitext(name)[1].lower() not in exts:
                    continue
            entries.append({"name": name, "path": full, "is_dir": is_dir})
    except PermissionError:
        pass

    parent = os.path.dirname(base.rstrip("/")) or "/"
    return {
        "cwd": base,
        "parent": parent if parent != base else None,
        "entries": entries,
    }
