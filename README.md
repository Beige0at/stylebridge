# StyleBridge

**Neural style transfer with a Sliced-Wasserstein objective, selectable optimisers, and a
saliency-guided face guard.**

StyleBridge is an optimisation-class neural style transfer engine built on VGG-19. It
combines two ideas that are unusual together:

- a **Sliced-Wasserstein Distance (SWD)** style objective instead of the classic Gram
  matrix, which preserves the content's geometry far better, and
- a **Saliency Guard** that spatially protects faces so portraits stay recognisable while
  the rest of the image is freely stylised.

It ships as a core PyTorch engine, a FastAPI service with an htmx web UI, a benchmark
harness, and a written evaluation against published methods on the Mould & Rosin
*NPRgeneral* benchmark.

---

## Highlights

- **Two objectives:** Gram (baseline) and SWD (advanced), selectable per run.
- **Three optimisers:** Adam, pure L-BFGS, and a hybrid Adam → L-BFGS refinement mode.
- **Saliency Guard:** OpenCV Haar cascades (default) or MediaPipe face-mesh convex hull.
- **Reproducible:** SWD projections are fixed from a seeded generator, so runs are
  deterministic (`--seed`).
- **Safe:** L-BFGS divergence is detected and rolled back; metric inputs are validated
  instead of silently returning `NaN`.
- **Web app:** asynchronous job queue + htmx dashboard (Gruvbox light theme).

---

## Results at a glance

### SWD vs. Gram (landscape pair, VGG-19, 400 px, 200 steps)

| Configuration | SSIM ↑ | LPIPS ↓ | SIFID ↓ |
|---|---|---|---|
| Gram | 0.593 | 0.480 | 18.16 |
| SWD (proj 64) | 0.723 | 0.334 | 20.27 |
| SWD (proj 128) | 0.718 | 0.361 | 20.34 |
| SWD (proj 200) | 0.724 | 0.358 | 20.73 |

SWD roughly doubles SSIM and cuts LPIPS by ~30% versus Gram on the same pair, and is
robust to the projection count.

### Optimisers (same pair, SWD)

| Mode | SSIM ↑ | LPIPS ↓ | Wall (s) |
|---|---|---|---|
| Adam (250 steps) | 0.725 | 0.379 | 25.6 |
| Adam + L-BFGS refine | 0.717 | 0.404 | 27.4 |
| L-BFGS (30 steps) | **0.732** | **0.348** | **5.7** |

Pure L-BFGS reaches comparable quality in a fraction of the time on this benign
landscape, but is brittle on adversarial SWD problems — hence the hybrid mode.

### Saliency Guard (face-ROI SSIM)

| Pair | Guard off | Haar | MediaPipe |
|---|---|---|---|
| Frontal portrait | 0.687 | **0.715** | 0.685 |
| Profile portrait | 0.419 | **0.449** | 0.427 |

The guard measurably preserves faces (up to ~+7% face-ROI SSIM) while the background is
stylised freely — a capability none of the compared published methods provide natively.

> The full 20×10 NPRgeneral benchmark (600 transfers) is under `results/`; see
> `results/apples_to_apples_NPRgeneral.md` for the complete head-to-head table against
> published methods.

---

## Repository layout

```
final_submission/
├── stylebridge_engine.py     # core engine: VGG19, Gram/SWD, Adam/L-BFGS, guard
├── api.py                    # FastAPI backend + htmx UI endpoints
├── run_benchmarks.py         # benchmark harness (batches A/B/C) + metrics
├── run_mould_rosin.py        # 20×10 NPRgeneral apples-to-apples benchmark
├── test_api.py               # end-to-end REST job-cycle test
├── test_ui.py                # end-to-end htmx polling test
├── static/index.html         # web UI (Gruvbox light theme)
├── models/                   # MediaPipe face-landmark model (optional guard)
├── styles/                   # 10 style images for the benchmark
├── nprgeneral/               # 20 NPRgeneral content images
├── <sample>.jpg              # sample images used by tests + harness
├── results/                  # benchmark CSVs, loss histories, summary docs
├── requirements.txt
├── setup.sh                  # create venv + install
├── run_server.sh             # launch the API/UI
└── .gitignore
```

---

## Requirements

- Python 3.10+ (developed on 3.13)
- PyTorch + TorchVision (CPU, CUDA or ROCm)
- See `requirements.txt` for the full list

### Installation

```bash
./setup.sh
source .stylebridge_env/bin/activate
```

Manual equivalent:

```bash
python3 -m venv .stylebridge_env
source .stylebridge_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**AMD ROCm** (the reference machine is an RX 6700 XT): install the ROCm wheels first.

```bash
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/rocm6.1
pip install -r requirements.txt
```

On first use, TorchVision downloads the VGG-19 weights (~550 MB) into the torch cache.

---

## Quick start (CLI)

```bash
python stylebridge_engine.py CONTENT STYLE --out result.jpg \
    --optimizer adam --style-mode swd --num-steps 250 --resolution 400 \
    --saliency-guard haar
```

Key flags: `--optimizer {adam,adam_lbfgs,lbfgs}`, `--style-mode {swd,gram}`,
`--saliency-guard {off,haar,mediapipe}`, `--resolution`, `--num-steps`, `--seed`,
`--loss-file`, `--mask-file`, `--quiet`.

Python API:

```python
from stylebridge_engine import transfer, save_image

target, history = transfer("content.jpg", "style.jpg",
                           optimizer="adam", style_mode="swd",
                           num_steps=250, resolution=400, saliency_guard="haar")
save_image(target, "out.jpg")
```

---

## Web application

```bash
./run_server.sh        # or: uvicorn api:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. The UI uploads content/style images, submits a job and
polls until the result is ready.

REST endpoints: `POST /uploads`, `POST /jobs`, `GET /jobs/{id}`,
`GET /jobs/{id}/result`, plus the htmx routes under `/ui/`.

---

## Tests

```bash
python test_api.py     # REST: upload -> job -> poll -> fetch result
python test_ui.py      # htmx polling flow
```

Both run the real engine on a small, fast configuration and print `OK` on success.

---

## Reproducing the benchmark

```bash
# Ablations: objective, optimiser, saliency guard
python run_benchmarks.py --batch A --content landscape --style kanagawa
python run_benchmarks.py --batch B --content landscape --style kanagawa
python run_benchmarks.py --batch C --content portrait2 --style kanagawa

# Full apples-to-apples run (20 content x 10 style x 3 configs = 600 transfers)
python run_mould_rosin.py
```

The NPRgeneral run is resume-aware: pairs already present in the CSV are skipped.

---

## Metrics

- **SSIM** — structural similarity of content vs output (higher is better).
- **LPIPS** — perceptual distance, AlexNet backbone (lower is better).
- **SIFID** — Fréchet Inception Distance between style and output statistics
  (lower is better), using a numerically stable symmetrised square root.

---

> The written project report is submitted separately through the module and is not
> included in this repository.

---

## Acknowledgements

Benchmark content set: **NPRgeneral** (Mould & Rosin, 2016, 2017). Metrics: SSIM
(Wang et al., 2004), LPIPS (Zhang et al., 2018), SIFID (Shaham et al., 2019). The SWD
style objective follows Heitz et al. (CVPR 2021); the comparison protocol follows
Ioannou & Maddock (2024).

## License

Coursework submitted for CM3070 (University of London, student 240282174). No license
is granted for reuse unless the author adds one.
