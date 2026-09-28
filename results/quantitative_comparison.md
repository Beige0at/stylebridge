# StyleBridge — Quantitative Comparison vs Published NST Algorithms

Consolidated, current benchmark numbers (re-run with deterministic
fixed-projection SWD). All runs on VGG19 (AMD RX 6700 XT / ROCm), res 400.
Full per-config CSV: `results/benchmark_results.csv`; 600-pair benchmark:
`results/mould_rosin/mould_rosin_benchmark.csv`.

Metrics: SSIM (whole-image, skimage, higher=better) · LPIPS (AlexNet,
lower=better) · SIFID (Inception FID vs style image, lower=better) · wall time.

## 1. StyleBridge ablations

OC = landscape + Kanagawa; P1 = frontal portrait (philip); P2 = profile
portrait (jurica) — face area ≈24% of frame.

| Config | SSIM↑ | LPIPS↓ | SIFID↓ | Face-SSIM↑ | Core-SSIM↑ | Wall(s)↓ |
|---|---|---|---|---|---|---|
| Gram · Adam (OC) | 0.593 | 0.480 | 18.16 | — | — | 13.2 |
| **SWD · Adam proj64 (OC)** | **0.723** | 0.334 | 20.27 | — | — | 17.6 |
| SWD · Adam proj128 (OC) | 0.718 | 0.361 | 20.34 | — | — | 25.4 |
| SWD · Adam proj200 (OC) | 0.724 | 0.358 | 20.73 | — | — | 31.7 |
| SWD · Adam 250 (OC) | 0.725 | 0.379 | 20.70 | — | — | 25.6 |
| SWD · Adam+L-BFGS (OC) | 0.717 | 0.404 | 20.08 | — | — | 27.4 |
| SWD · L-BFGS 30 (OC) | **0.732** | **0.348** | 20.92 | — | — | **5.7** |
| SWD · Adam · guard off (P1) | 0.721 | 0.272 | 36.47 | 0.687 | — | 32.0 |
| SWD · Adam · guard Haar (P1) | 0.722 | 0.271 | 37.60 | **0.715** | 0.917 | 35.2 |
| SWD · Adam · guard MP (P1) | 0.722 | 0.272 | 36.84 | 0.685 | 0.904 | 32.9 |
| SWD · Adam · guard off (P2) | 0.378 | 0.218 | 32.68 | 0.419 | — | 34.5 |
| SWD · Adam · guard Haar (P2) | 0.387 | **0.203** | 33.00 | **0.449** | 0.755 | 37.2 |
| SWD · Adam · guard MP (P2) | 0.380 | 0.208 | 33.43 | 0.427 | 0.786 | 35.2 |

Reading the numbers: SWD consistently beats Gram on content preservation
(SSIM 0.72 vs 0.59; LPIPS 0.33 vs 0.48). The saliency guard protects faces: on
the profile portrait it lifts face-ROI SSIM 0.419→0.449 and keeps the protected
core at 0.75–0.92 SSIM. Guard also improves portrait LPIPS (0.218→0.203).

## 2. Head-to-head vs published SOTA (Ioannou & Maddock 2024, CGF)

Means over 20 content × 10 style pairs (200 per config) on the Mould & Rosin
set, same three metrics.

| Method | SSIM↑ | LPIPS↓ | SIFID↓ |
|---|---|---|---|
| AdaIN | 0.2588 | 0.5174 | 0.6223 |
| AdaAttN | 0.4705 | 0.4766 | 18.3407 |
| ArtFlow | 0.4547 | 0.4824 | 0.9657 |
| CSBNet | 0.3601 | 0.4780 | 5.0953 |
| IEContraAST | 0.4065 | 0.4439 | 0.8600 |
| MCCNet | 0.4547 | 0.4714 | 2.3755 |
| RAST | 0.5383 | 0.3101 | 1.7580 |
| SANet | 0.3096 | 0.5373 | 1.1592 |
| StyTr2 | 0.4882 | 0.4693 | 1.1089 |
| **StyleBridge SWD · L-BFGS** | **0.6565 ± 0.145** | **0.1551 ± 0.094** | 26.10 ± 6.01 |
| **StyleBridge SWD · Adam** | **0.6499 ± 0.138** | **0.2020 ± 0.081** | 25.90 ± 5.82 |
| **StyleBridge Gram · Adam** | **0.5719 ± 0.144** | 0.3394 ± 0.091 | 24.07 ± 6.45 |

## 3. What the comparison does and does not show

**Content preservation (SSIM, LPIPS) — ours is strong.** Every StyleBridge
configuration exceeds the best published SSIM (we worst 0.572 vs RAST 0.538),
and SWD+L-BFGS LPIPS (0.155) is about twice as close to the content as the best
published (0.310). This is the expected profile of an optimisation method with a
strong content term plus SWD.

**Style fidelity (SIFID) — ours looks weak; read with caution.** Our SIFID
(~24–26) is far higher than published (~0.6–18). Two legitimate readings:
(i) a real trait — SWD leans toward content preservation at the expense of the
reference's global Inception statistics; (ii) protocol differences — published
SIFID uses different Inception checkpoints/feature layers and preprocessing. We
therefore present SIFID as indicative only.

**Speed.** Our L-BFGS path (6.7 s/pair) is far faster than published
optimisation NST, but not comparable to feed-forward nets because StyleBridge
is an optimisation engine, not a trained decoder.

## 4. Caveats

- SSIM/LPIPS/SIFID are evaluation-set-specific; cross-method comparisons are
  positioning, not strict equivalence.
- SIFID magnitude is very sensitive to feature layer/checkpoint.
- LPIPS uses AlexNet here; published tables often use VGG.
- Published values have no reported variance.
