"""Benchmark harness.

Runs a matrix of transfer() configs as separate subprocesses (each one gets a
clean GPU), scores the output with SSIM / LPIPS / SIFID, and appends a row to a
CSV plus the loss curve to JSON.

    python run_benchmarks.py            # every batch defined below
    python run_benchmarks.py --batch B  # just one batch
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from skimage.metrics import structural_similarity as compare_ssim

# metric models are loaded lazily and cached here
_LPIPS = None
_LPIPS_TR = None

# --- paths ----------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "stylebridge_engine.py"
RESULTS = ROOT / "results"
OUTDIR = RESULTS / "outputs"
LOSSDIR = RESULTS / "losses"

# ---- Evaluation set ------------------------------------------------------ #
# the images referenced by the batch definitions and the harness CLI
CONTENT_SETS = {
    "landscape": "pexels-mitbg000-17827032.jpg",
    "portrait": "philip-martin-portrait.jpg",
    "portrait2": "jurica-koletic-portrait.jpg",
}
STYLES = {
    "kanagawa": "Under-the-Wave-off-Kanagawa-1024x691.jpg",
    "vggrassy": "art-institute-of-chicago-3m7YpBz3I7Y-unsplash.jpg",
    "cma": "the-cleveland-museum-of-art-879tYIRL5eE-unsplash.jpg",
}


def cfg(**kw):
    # tiny helper so the batch entries below read cleanly
    return kw


# ---- Batch definitions --------------------------------------------------- #
BATCHES = {
    # A: objective + projection count (all Adam, fixed weights)
    "A": [
        cfg(name="A_gram", style_mode="gram", num_projections=200, num_steps=200),
        cfg(name="A_swd_proj64", style_mode="swd", num_projections=64, num_steps=200),
        cfg(name="A_swd_proj128", style_mode="swd", num_projections=128, num_steps=200),
        cfg(name="A_swd_auto200", style_mode="swd", num_projections=200, num_steps=200),
    ],
    # B: optimiser modes, all on SWD
    "B": [
        cfg(name="B_adam", optimizer="adam", num_steps=250),
        cfg(name="B_adam_lbfgs", optimizer="adam_lbfgs", num_steps=250, lbfgs_steps=20),
        cfg(name="B_lbfgs", optimizer="lbfgs", num_steps=30),
    ],
    # C: saliency guard on a portrait (SWD + Adam)
    "C": [
        cfg(name="C_guard_off", saliency_guard="off", num_steps=250),
        cfg(name="C_guard_haar", saliency_guard="haar", num_steps=250),
        cfg(name="C_guard_mediapipe", saliency_guard="mediapipe", num_steps=250),
    ],
}


def content_of(content_set):
    return ROOT / CONTENT_SETS[content_set]


def style_of(key):
    return ROOT / STYLES[key]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def measure_ssim(content_path, generated_path):
    """Whole-image grayscale SSIM between content and output (higher = better)."""
    img_content = cv2.imread(str(content_path), cv2.IMREAD_GRAYSCALE)
    img_gen = cv2.imread(str(generated_path), cv2.IMREAD_GRAYSCALE)
    if img_content is None or img_gen is None:
        return float("nan")
    img_gen = cv2.resize(img_gen, (img_content.shape[1], img_content.shape[0]))
    score, _ = compare_ssim(img_content, img_gen, full=True)
    return float(score)


def _face_boxes(gray, name=""):
    """Haar face boxes at the original resolution (frontal + profile)."""
    eq = cv2.equalizeHist(gray)
    data = cv2.data.haarcascades
    dets = [cv2.CascadeClassifier(data + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(data + "haarcascade_profileface.xml")]
    boxes = []
    for det in dets:
        for sf in (1.05, 1.1):
            boxes.extend(tuple(b) for b in det.detectMultiScale(eq, scaleFactor=sf,
                                                                minNeighbors=4, minSize=(40, 40)))
    # drop boxes that mostly overlap one we already kept
    keep = []
    for b in boxes:
        overlap = False
        for k in keep:
            ix = max(0, min(b[0]+b[2], k[0]+k[2]) - max(b[0], k[0]))
            iy = max(0, min(b[1]+b[3], k[1]+k[3]) - max(b[1], k[1]))
            if ix*iy > 0.5 * min(b[2]*b[3], k[2]*k[3]):
                overlap = True
                break
        if not overlap:
            keep.append(b)
    return len(keep) != 0, keep


def measure_face_ssim(content_path, generated_path):
    """SSIM over the union of detected face boxes.

    Returns (score, face_fraction), or (nan, 0) if no face is found. It's an
    approximation: we compute the SSIM map on the bounding ROI and average only
    the pixels that fall inside the boxes.
    """
    img_content = cv2.imread(str(content_path), cv2.IMREAD_GRAYSCALE)
    img_gen = cv2.imread(str(generated_path), cv2.IMREAD_GRAYSCALE)
    if img_content is None or img_gen is None:
        return float("nan"), 0.0
    img_gen = cv2.resize(img_gen, (img_content.shape[1], img_content.shape[0]))
    found, boxes = _face_boxes(img_content)
    if not found:
        return float("nan"), 0.0
    # paint the face boxes into a mask
    mask = np.zeros_like(img_content, dtype=float)
    for (x, y, w, h) in boxes:
        mask[y:y+h, x:x+w] = 1.0
    frac = float((mask > 0).mean())
    if frac == 0:
        return float("nan"), 0.0
    # crop to the bounding ROI of all faces, then average the SSIM map only over
    # the masked pixels inside it
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    roi_c = img_content[y0:y1, x0:x1]
    roi_g = img_gen[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1] > 0
    if m.sum() == 0:
        return float("nan"), frac
    full_map = compare_ssim(roi_c, roi_g, full=True)[1]
    face_score = float(full_map[m].mean())
    return face_score, frac


def measure_masked_ssim(content_path, generated_path, mask_npy, win=7):
    """SSIM inside the strong core of the guard mask the engine actually used.

    We erode the >0.9 core first so only pixels whose whole SSIM window sits
    inside the mask get counted; otherwise stylised neighbours drag the score down.
    """
    if mask_npy is None or not mask_npy.exists():
        return float("nan")
    content = cv2.imread(str(content_path), cv2.IMREAD_GRAYSCALE)
    gen = cv2.imread(str(generated_path), cv2.IMREAD_GRAYSCALE)
    if content is None or gen is None:
        return float("nan")
    mask = np.load(mask_npy)
    mh, mw = mask.shape
    content_r = cv2.resize(content, (mw, mh))
    gen_r = cv2.resize(gen, (mw, mh))
    strong = mask > 0.9
    interior = cv2.erode(strong.astype(np.uint8), np.ones((win, win), np.uint8)) > 0
    if interior.sum() == 0:
        return float("nan")
    full_map = compare_ssim(content_r, gen_r, data_range=255, full=True)[1]
    return float(full_map[interior].mean())


def measure_lpips(content_path, generated_path, target_size=256):
    """LPIPS (AlexNet) between content and output. Lower = perceptually closer.

    The model is loaded once and reused, and runs on CPU so it doesn't compete
    with the GPU runner for memory.
    """
    global _LPIPS, _LPIPS_TR
    try:
        import lpips
    except ImportError:
        return float("nan")
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="alex")
        _LPIPS.eval()
        _LPIPS_TR = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((target_size, target_size)),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    try:
        a = _load_rgb(content_path)
        b = _load_rgb(generated_path)
    except Exception as exc:
        _warn_metric("lpips", exc)
        return float("nan")
    ta, tb = _LPIPS_TR(a).unsqueeze(0), _LPIPS_TR(b).unsqueeze(0)
    with torch.no_grad():
        dist = float(_LPIPS(ta, tb).item())
    return dist


def _load_rgb(path):
    """Load an image as RGB, raising a clear error for missing/empty files.

    The old version returned None on any failure, which turned a wrong path into
    an unexplained `nan` metric with no clue why.
    """
    from PIL import Image
    path = str(path)
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"image not found: {path}")
    if Path(path).stat().st_size == 0:
        raise ValueError(f"image file is empty: {path}")
    return Image.open(path).convert("RGB")


def _warn_metric(name, exc):
    # say which metric bailed and why, instead of leaving a bare nan
    print(f"[harness] {name} skipped: {exc}")


_SIFID_INC = None


def _sifid_inception():
    """Load InceptionV3 once (on CPU); we hook its Mixed_6e feature map."""
    global _SIFID_INC
    if _SIFID_INC is None:
        import torchvision.models as tvm
        inc = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1).eval()
        for p in inc.parameters():
            p.requires_grad_(False)
        _SIFID_INC = inc
    return _SIFID_INC


def _sifid_stats(pil_img, target=299):
    """Mean + covariance of the Mixed_6e feature vectors, one row per pixel."""
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((target, target)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    x = tf(pil_img).unsqueeze(0)
    inc = _sifid_inception()
    feats = {}
    # grab the Mixed_6e activation with a temporary hook
    hook = inc.Mixed_6e.register_forward_hook(
        lambda m, i, o: feats.setdefault("f", o))
    with torch.no_grad():
        inc(x)
    hook.remove()
    f = feats["f"].squeeze(0)                      # C,H,W
    f = f.reshape(f.size(0), -1).t().float()       # N,C (each pixel is a sample)
    return f.mean(0), torch.cov(f.t()).numpy()


def _sym_sqrt(m):
    """Symmetric matrix square root via eigendecomposition.

    Tiny negative eigenvalues from rounding get clipped to 0 before the sqrt.
    """
    m = (m + m.T) / 2
    w, v = np.linalg.eigh(m)
    return (v * np.sqrt(np.maximum(w, 0))) @ v.T


def measure_sifid(style_path, generated_path):
    """Single-Image FID between the style image and the output. Lower = closer.

    Fréchet distance over Inception Mixed_6e statistics.
    """
    try:
        a = _load_rgb(style_path)
        b = _load_rgb(generated_path)
    except Exception as exc:
        _warn_metric("sifid", exc)
        return float("nan")
    m1, c1 = _sifid_stats(a)
    m2, c2 = _sifid_stats(b)
    # Tr(sqrt(S1 S2)) via the symmetric form sqrt(S1^1/2 S2 S1^1/2). The raw
    # product S1@S2 isn't symmetric PSD, so we can't eigendecompose it directly.
    s1 = _sym_sqrt(c1)
    t = float(np.trace(_sym_sqrt(s1 @ c2 @ s1)))
    d = float(np.sum((m1.numpy() - m2.numpy()) ** 2)
              + np.trace(c1) + np.trace(c2) - 2 * t)
    return d


def measure_protected_mse(content_path, generated_path, mask_npy):
    """Mean squared pixel error between output and content over the mask core.

    ~0 means the guard really did keep those pixels equal to the content.
    """
    if mask_npy is None or not mask_npy.exists():
        return float("nan")
    content = cv2.imread(str(content_path), cv2.IMREAD_GRAYSCALE)
    gen = cv2.imread(str(generated_path), cv2.IMREAD_GRAYSCALE)
    if content is None or gen is None:
        return float("nan")
    mask = np.load(mask_npy)
    mh, mw = mask.shape
    content_r = cv2.resize(content, (mw, mh))
    gen_r = cv2.resize(gen, (mw, mh))
    core = mask > 0.9
    if core.sum() == 0:
        return float("nan")
    return float(np.mean((content_r[core].astype(float) - gen_r[core].astype(float)) ** 2))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_one(entry, content_set, style_key):
    # one config, run in its own subprocess so GPU state can't leak between runs
    name = f"{content_set}_{style_key}_{entry['name']}"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOSSDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"{name}.jpg"
    loss_path = LOSSDIR / f"{name}.json"
    mask_path = LOSSDIR / f"{name}_mask.npy"

    content_path = content_of(content_set)
    style_path = style_of(style_key)

    # python kwarg name -> hypenated CLI flag
    FLAG = {"style_mode": "style-mode", "saliency_guard": "saliency-guard",
            "content_weight": "content-weight", "style_weight": "style-weight",
            "num_steps": "num-steps", "lbfgs_steps": "lbfgs-steps",
            "num_projections": "num-projections"}

    # build the engine command (resolution + output files are always set)
    cmd = [sys.executable, str(ENGINE), str(content_path), str(style_path),
           "--out", str(out_path), "--loss-file", str(loss_path),
           "--mask-file", str(mask_path),
           "--resolution", str(entry.get("resolution", 400))]
    for flag in ("optimizer", "style_mode", "saliency_guard"):
        if flag in entry:
            cmd += [f"--{FLAG.get(flag, flag)}", str(entry[flag])]
    for flag in ("content_weight", "style_weight", "num_steps", "lbfgs_steps",
                 "num_projections"):
        if flag in entry:
            cmd += [f"--{FLAG.get(flag, flag)}", str(entry[flag])]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-1] if (proc.stdout or proc.stderr) else ""

    # read back the loss curve the engine wrote (if it got that far)
    final_loss = float("nan")
    steps = 0
    if loss_path.exists():
        try:
            data = json.loads(loss_path.read_text())
            hist = data["loss"]
            steps = len(hist)
            final_loss = float(hist[-1])
        except Exception:
            pass

    ok = proc.returncode == 0
    have_out = out_path.exists() and out_path.stat().st_size > 0
    if not ok:
        print(f"[harness] {name}: engine failed (rc={proc.returncode}); "
              f"last output: {tail[:200]}")
    # only score if there's an actual output, otherwise record nan
    ssim = measure_ssim(content_path, out_path) if have_out else float("nan")
    face_ssim, face_frac = (
        measure_face_ssim(content_path, out_path)
        if (have_out and content_set.startswith("portrait")) else (float("nan"), 0.0))
    if have_out and content_set.startswith("portrait"):
        # guard-specific metrics only make sense on portraits
        masked_ssim = measure_masked_ssim(content_path, out_path, mask_path)
        protect_mse = measure_protected_mse(content_path, out_path, mask_path)
    else:
        masked_ssim = protect_mse = float("nan")
    lpips = measure_lpips(content_path, out_path) if have_out else float("nan")
    sifid = measure_sifid(style_path, out_path) if have_out else float("nan")

    row = {
        "name": name, "content_set": content_set, "style": style_key,
        "optimizer": entry.get("optimizer", "adam"),
        "style_mode": entry.get("style_mode", "swd"),
        "saliency_guard": entry.get("saliency_guard", "off"),
        "resolution": entry.get("resolution", 400),
        "num_steps": entry.get("num_steps", ""),
        "lbfgs_steps": entry.get("lbfgs_steps", ""),
        "num_projections": entry.get("num_projections", "auto"),
        "ssim": ssim, "face_ssim": face_ssim, "face_frac": face_frac,
        "masked_ssim": masked_ssim, "protect_mse": protect_mse,
        "lpips": lpips, "sifid": sifid,
        "wall_s": round(wall, 2), "steps": steps,
        "final_loss": final_loss, "returncode": proc.returncode,
        "tail": tail[:120],
    }
    print(f"[harness] {name}: ok={ok} ssim={ssim:.4f} face_ssim={face_ssim:.4f} "
          f"masked_ssim={masked_ssim:.4f} protect_mse={protect_mse:.3f} lpips={lpips:.3f} "
          f"sifid={sifid:.3f} wall={wall:.1f}s loss={final_loss:.4g} steps={steps}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None, help="run only e.g. 'A'")
    ap.add_argument("--content", default="landscape",
                    help="content set: landscape | portrait")
    ap.add_argument("--style", default="kanagawa")
    args = ap.parse_args()

    batches = [args.batch] if args.batch else list(BATCHES)
    csv_path = RESULTS / "benchmark_results.csv"

    # keep whatever is already in the CSV so runs accumulate
    existing = []
    if csv_path.exists():
        with open(csv_path) as f:
            existing = list(csv.DictReader(f))

    rows = []
    for b in batches:
        for entry in BATCHES[b]:
            # batch C is a guard test, meaningless unless the content is a portrait
            if b == "C" and not args.content.startswith("portrait"):
                continue
            rows.append(run_one(entry, args.content, args.style))

    all_rows = existing + rows
    # rebuild the header from every key we've seen, so adding a column is safe
    fields = []
    for r in all_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[harness] appended to {csv_path}")


if __name__ == "__main__":
    main()
