"""Adapter over PrecisionTrack's config validation registry.

This module holds **no validation policy of its own**. Every rule — which paths must exist,
which parameters a given tool actually reads, which booleans make a parameter relevant — is
declared once in :mod:`precision_track.utils.config_fields` and shared with the CLI tools.
That is what keeps the UI and the tools from disagreeing about what a valid config is.

Every function returns UI-shaped dicts; the registry returns
:class:`~precision_track.utils.config_fields.ConfigIssue` objects.
"""

from typing import Any, Dict, List, Optional

from precision_track.utils.config_fields import FIELDS_BY_ID, run_checks
from precision_track.utils.config_schema import validate_user_config


def _level(severity: str) -> str:
    return "warning" if severity == "warning" else "error"


def validate_field(field: str, cfg: Dict, tool: Optional[str] = None) -> Dict[str, Any]:
    """Validate one dotted field id (``"training.data_root"``) against the whole config.

    Warnings keep ``ok=True`` so the value still saves — a missing ``hyperparameters.json``
    is expected before the calibration step has ever run.
    """
    if field not in FIELDS_BY_ID:
        return {"field": field, "ok": True, "level": "ok", "message": ""}

    report = run_checks(cfg, tool=tool, only=field)
    if not report.issues:
        return {"field": field, "ok": True, "level": "ok", "message": ""}

    issue = report.errors[0] if report.errors else report.warnings[0]
    level = _level(issue.severity)
    return {"field": field, "ok": level != "error", "level": level, "message": issue.message}


def validate_all(cfg: Dict, tool: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
    """Validate the whole config, optionally for one specific tool.

    ``tool=None`` reports against the union of every tool — the right view for the Configure
    form. Passing a tool narrows it to what that tool actually reads, which is what the run
    endpoint uses so an invalid ``data_root`` cannot block a tracking run.
    """
    report = validate_user_config(cfg, tool=tool)
    return {
        "errors": [{"field": i.field, "message": i.message} for i in report.errors],
        "warnings": [{"field": i.field, "message": i.message} for i in report.warnings],
    }
