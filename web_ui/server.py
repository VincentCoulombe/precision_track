"""FastAPI app: JSON API + static frontend for the PrecisionTrack web UI."""

import os
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from precision_track.utils import parse_pose_metainfo

from . import config_io, fs_browse, job, validation_io, validators
from .paths import ASSETS_DIR, REPO_ROOT, TOOLS_DIR, resolve_from_tools
from .tools_spec import TOOLS, build_argv

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

app = FastAPI(title="PrecisionTrack UI")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Force the browser to revalidate static assets (HTML/JS/CSS).

    This is a local dev tool whose front-end files change often; without this,
    browsers aggressively cache the ES modules and keep serving stale UI after
    an edit. ``no-cache`` still uses ETag/Last-Modified, so unchanged files come
    back as a cheap 304 — it just guarantees the browser checks every load.
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@app.get("/api/config")
def get_config():
    cfg = config_io.load_config_plain()
    return {
        "config": cfg,
        "metainfo_classes": _metainfo_classes(cfg),
        "paths": {"tools_dir": str(TOOLS_DIR), "repo_root": str(REPO_ROOT)},
    }


@app.get("/api/resolve")
def get_resolve(path: str = Query(...), base: Optional[str] = None):
    """Resolve a config path to an absolute path (for picker start dirs)."""
    from .paths import resolve_from

    return {"abs": resolve_from(base, path) if base else resolve_from_tools(path)}


@app.post("/api/config/field")
def post_config_field(field: str = Body(...), config: dict = Body(...)):
    """Validate a single committed field; persist the whole config unless the
    field is an error (so an invalid value never lands in user_configs.yaml)."""
    res = validators.validate_field(field, config)
    if res.get("level") == "error":
        return {"saved": False, **res}
    created = config_io.write_config(config)
    return {"saved": True, "created_dirs": created.get("created_dirs", []), **res}


@app.post("/api/validate/{field}")
def post_validate(field: str, config: dict = Body(..., embed=True)):
    return validators.validate_field(field, config)


@app.get("/api/metainfo/classes")
def get_metainfo_classes(path: str = Query(...)):
    abs_path = resolve_from_tools(path)
    if not os.path.isfile(abs_path):
        raise HTTPException(404, f"Metainfo not found: {abs_path}")
    try:
        meta = parse_pose_metainfo({"from_file": abs_path})
    except Exception as exc:
        raise HTTPException(400, f"Metainfo failed to load: {exc}")
    return {"classes": meta.get("classes", []), "num_keypoints": meta.get("num_keypoints")}


def _metainfo_classes(cfg: dict):
    try:
        meta = parse_pose_metainfo({"from_file": resolve_from_tools(cfg.get("general", {}).get("metainfo", ""))})
        return meta.get("classes", [])
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Validation config + ReID identities
# --------------------------------------------------------------------------- #
@app.get("/api/validation-config")
def get_validation_config(path: str = Query(...)):
    return validation_io.load_validation_config(path)


@app.post("/api/validation-config")
def post_validation_config(path: str = Body(...), config: dict = Body(...)):
    return validation_io.save_validation_config(path, config)


@app.get("/api/validation-template")
def get_validation_template(strategy: str = Query(...)):
    return {"config": validation_io.template_for(strategy)}


@app.get("/api/reid-metainfo")
def get_reid_metainfo(path: str = Query(...)):
    return validation_io.load_reid_metainfo(path)


@app.post("/api/reid-metainfo")
def post_reid_metainfo(path: str = Body(...), identities: list = Body(...), disabled_identities: list = Body(default=[])):
    return validation_io.save_reid_metainfo(path, identities, disabled_identities)


# --------------------------------------------------------------------------- #
# Filesystem browse
# --------------------------------------------------------------------------- #
@app.get("/api/fs")
def get_fs(path: Optional[str] = None, dirs_only: bool = False, exts: Optional[str] = None):
    ext_list = [e for e in (exts or "").split(",") if e]
    return fs_browse.list_dir(path, dirs_only=dirs_only, exts=ext_list)


# --------------------------------------------------------------------------- #
# Tools + run
# --------------------------------------------------------------------------- #
@app.get("/api/tools")
def get_tools():
    return {"tools": TOOLS}


@app.post("/api/run")
async def post_run(tool: str = Body(...), values: dict = Body(default={})):
    if tool not in TOOLS:
        raise HTTPException(404, f"Unknown tool: {tool}")
    try:
        argv = build_argv(tool, values)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    try:
        j = await job.start(tool, argv)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return j.info()


@app.get("/api/run")
def get_run():
    j = job.current()
    if j is None:
        return {"running": False, "job": None, "tail": [], "live": ""}
    return {"running": j.status == "running", "job": j.info(), "tail": j.tail(), "live": j.live}


@app.get("/api/run/stream")
async def get_run_stream():
    j = job.current()
    if j is None:
        raise HTTPException(404, "No job to stream.")
    return StreamingResponse(job.stream(j), media_type="text/event-stream")


@app.post("/api/run/stop")
async def post_run_stop():
    return await job.stop()


# --------------------------------------------------------------------------- #
# Static frontend + assets
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(404)
def _404(request, exc):
    return JSONResponse({"detail": "Not found"}, status_code=404)
