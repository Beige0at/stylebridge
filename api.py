"""StyleBridge API.

A small FastAPI wrapper around the engine. Uploads land in a spool dir; every
run gets a job id and a single worker thread chews through them one at a time.
The browser (or a JSON client) polls /jobs/{id} until the result is ready.

Run it with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import copy
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import stylebridge_engine as engine

# Spool dirs, all created next to this file.
BASE = Path(__file__).resolve().parent
SPOOL = BASE / "api_spool"
SPOOL.mkdir(exist_ok=True)
_uploads = SPOOL / "uploads"
_results = SPOOL / "results"
_uploads.mkdir(exist_ok=True)
_results.mkdir(exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ---- job store ------------------------------------------------------------ #
# _jobs maps id -> job dict, _job_order keeps submission order. Both are only
# touched while holding _jobs_lock, so the request thread and the worker thread
# never observe a half-written job.
_jobs = {}
_job_order = []
_jobs_lock = threading.Lock()
_worker_ready = threading.Event()
_worker_lock = threading.Lock()
_worker_started = False

app = FastAPI(title="StyleBridge", version="0.2.0")

STATIC = BASE / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class TransferRequest(BaseModel):
    content: str                          # stored upload name
    style: str                            # stored upload name
    optimizer: str = "adam"
    style_mode: str = "swd"
    saliency_guard: str = "off"
    content_weight: float = 1
    style_weight: float = 1e4
    num_steps: int = 500
    lbfgs_steps: int = 20
    num_projections: int | None = None
    resolution: int = 400
    guard_content_boost: float = 2.0
    guard_blend: float = 1.0
    line_search: str | None = None        # "none" gets mapped to None
    seed: int = 0


def _now():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Thread-safe job store helpers
# --------------------------------------------------------------------------- #
def _job_snapshot(job_id):
    """Deep copy a job (or None) so callers can render it outside the lock."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        return copy.deepcopy(job) if job is not None else None


def _job_update(job_id, **fields):
    """Atomically set fields on a job, if it still exists."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _ensure_worker():
    """Start the worker thread the first time a job is queued.

    Doing this lazily rather than at import time means importing the module (or
    a TestClient that never runs startup events) doesn't spawn a thread as a
    side effect.
    """
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_worker_loop, daemon=True).start()
            _worker_started = True


def _safe_ext(name: str) -> str:
    # only accept the image types we can actually read
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext or '(none)'}")
    return ext


def _save_upload(upload: UploadFile) -> str:
    # store under a random name so uploads never collide
    ext = _safe_ext(upload.filename or "")
    dest = _uploads / f"{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return dest.name


@app.post("/uploads", status_code=201)
def upload_file(file: UploadFile):
    # returns the stored name, which the client then passes to /jobs
    return {"name": _save_upload(file)}


@app.post("/jobs")
def create_job(req: TransferRequest):
    # make sure both files actually exist in the spool before queuing anything
    uploads = _uploads_map()
    if req.content not in uploads:
        raise HTTPException(404, f"Unknown content file: {req.content}")
    if req.style not in uploads:
        raise HTTPException(404, f"Unknown style file: {req.style}")
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id, "status": "queued", "created": _now(),
            "request": req.model_dump(), "error": None, "result": None,
        }
        _job_order.append(job_id)
    _ensure_worker()
    _worker_ready.set()                 # poke the worker if it's sleeping
    return {"id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _job_snapshot(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _uploads_map():
    # set of filenames currently sitting in the uploads dir
    return {p.name for p in _uploads.iterdir()}


# --------------------------------------------------------------------------- #
# htmx UI endpoints
# --------------------------------------------------------------------------- #
def _fragment(job: dict) -> str:
    """Render the htmx result fragment for a job snapshot.

    While the job is queued/running the fragment carries an hx-trigger so the
    browser keeps polling; once it's done/failed we drop the trigger and stop.
    """
    job_id = job["id"]
    status = job["status"]
    poll = ('hx-get="/ui/job/%s" hx-trigger="load delay:500ms, every 2s" '
            'hx-swap="outerHTML"' % job_id)
    if status in ("queued", "running"):
        req = job.get("request") or {}
        return (f'<div id="poll" {poll}><p class="status running">'
                f'&#9203; {status.capitalize()}&hellip; '
                f'steps={req.get("num_steps", "?")} res={req.get("resolution", "?")} '
                f'opt={req.get("optimizer", "?")}</p></div>')
    if status == "failed":
        return (f'<div id="poll"><p class="status failed">&#10060; failed: '
                f'{job.get("error") or "unknown error"}</p></div>')
    # done, so show the image plus a one-line summary
    res = job.get("result") or {}
    final = res.get("final_loss")
    final_txt = f"{final:.4g}" if isinstance(final, (int, float)) else "n/a"
    return (f'<div id="poll">'
            f'<p class="status done">&#10003; done '
            f'in {res.get("wall_s", "?")} s &middot; {res.get("steps", "?")} steps '
            f'&middot; final loss {final_txt}</p>'
            f'<div class="result"><img src="/ui/job/{job_id}/image" '
            f'alt="stylised output"></div>'
            f'</div>')


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.post("/ui/transfers", response_class=HTMLResponse)
def ui_transfer(
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    optimizer: str = Form("adam"),
    style_mode: str = Form("swd"),
    saliency_guard: str = Form("off"),
    content_weight: float = Form(1),
    style_weight: float = Form(1e4),
    num_steps: int = Form(250),
    lbfgs_steps: int = Form(20),
    resolution: int = Form(400),
    guard_content_boost: float = Form(2.0),
    guard_blend: float = Form(1.0),
    line_search: str = Form("none"),
    seed: int = Form(0),
):
    # the form posts both files plus the settings. We save them and queue a job.
    cname = _save_upload(content)
    sname = _save_upload(style)
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        job = {
            "id": job_id, "status": "queued", "created": _now(),
            "request": {
                "content": cname, "style": sname,
                "optimizer": optimizer, "style_mode": style_mode,
                "saliency_guard": saliency_guard,
                "content_weight": content_weight, "style_weight": style_weight,
                "num_steps": num_steps, "lbfgs_steps": lbfgs_steps,
                "resolution": resolution,
                "guard_content_boost": guard_content_boost,
                "guard_blend": guard_blend,
                "line_search": line_search,
                "num_projections": None,
                "seed": seed,
            },
            "error": None, "result": None,
        }
        _jobs[job_id] = job
        _job_order.append(job_id)
        # snapshot under the lock so the worker can't mutate it mid-render
        snapshot = copy.deepcopy(job)
    _ensure_worker()
    _worker_ready.set()
    return _fragment(snapshot)


@app.get("/ui/job/{job_id}", response_class=HTMLResponse)
def ui_job(job_id: str):
    job = _job_snapshot(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return _fragment(job)


@app.get("/ui/job/{job_id}/image")
def ui_job_image(job_id: str):
    job = _job_snapshot(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(404, "Result not ready")
    img = _results / job["result"]["image"]
    if not img.exists():
        raise HTTPException(404, "Result image missing")
    return FileResponse(img, media_type="image/jpeg")


def _run_job(job_id: str) -> None:
    # runs on the worker thread, reads its request from a snapshot so it isn't
    # holding the lock while the (long) transfer runs
    snap = _job_snapshot(job_id)
    if snap is None:
        return
    req = snap["request"]
    content = _uploads / req["content"]
    style = _uploads / req["style"]
    ext = _safe_ext(req["content"])
    out_path = _results / f"{job_id}{ext}"
    mask_path = _results / f"{job_id}_mask.npy"
    loss_path = _results / f"{job_id}_loss.json"

    try:
        import json
        _job_update(job_id, status="running", started=_now())
        t0 = time.time()
        target, history = engine.transfer(
            str(content), str(style),
            optimizer=req["optimizer"],
            style_mode=req["style_mode"],
            content_weight=req["content_weight"],
            style_weight=req["style_weight"],
            num_steps=req["num_steps"],
            lbfgs_steps=req["lbfgs_steps"],
            num_projections=req["num_projections"],
            resolution=req["resolution"],
            saliency_guard=req["saliency_guard"],
            guard_content_boost=req["guard_content_boost"],
            guard_blend=req["guard_blend"],
            line_search=(None if req["line_search"] == "none" else req["line_search"]),
            # only ask for a mask file when the guard is actually on
            mask_file=(
                str(mask_path) if req["saliency_guard"] != "off" else None
            ),
            seed=req.get("seed", 0),
            verbose=False,          # keep the server log clean
        )
        engine.save_image(target, str(out_path))
        with loss_path.open("w") as fh:
            json.dump({"loss": history}, fh)
        _job_update(
            job_id, status="done", finished=_now(),
            result={
                "image": out_path.name,
                "losses": loss_path.name,
                "mask": mask_path.name if mask_path.exists() else None,
                "steps": len(history),
                "final_loss": float(history[-1]) if history else None,
                "wall_s": round(time.time() - t0, 2),
            })
    except Exception as exc:
        # surface the failure to the client instead of hanging the job
        _job_update(job_id, status="failed",
                    error=f"{type(exc).__name__}: {exc}", finished=_now())


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = _job_snapshot(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"Job not ready (status={job['status']})")
    img = _results / job["result"]["image"]
    if not img.exists():
        raise HTTPException(404, "Result image missing")
    return FileResponse(img, media_type="image/jpeg")


def _worker_loop():
    # sits on the event until something is queued, then drains all queued jobs
    while True:
        _worker_ready.wait()
        _worker_ready.clear()
        with _jobs_lock:
            order = list(_job_order)
        for jid in order:
            # claim the job under the lock so we don't double-process it
            with _jobs_lock:
                job = _jobs.get(jid)
                if job is None or job.get("status") != "queued":
                    continue
                job["status"] = "running"
            _run_job(jid)
