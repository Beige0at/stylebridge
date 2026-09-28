# StyleBridge — Apples-to-Apples vs Published SOTA on NPRgeneral

Final, re-run results (fixed-projection deterministic SWD). Reproduces the
Ioannou & Maddock (2024, CGF, `10.1111/cgf.15165`) Table-IX protocol on the
same content set: the Mould & Rosin **NPRgeneral** benchmark (20 content
images). 200 content×style pairs per config, metrics averaged over all 200.

## Protocol (what was actually run)

- **Content:** NPRgeneral (Mould & Rosin 2016, 2017), all 20 images
  (`nprgeneral/*1024.jpg`).
- **Style:** 10 documented public-domain artworks (`styles/*.jpg`).
- **Resolution:** 400 (all methods resize; the paper does not state its size).
- **Metrics:** SSIM (content vs output, skimage grayscale), LPIPS (AlexNet,
  content vs output), SIFID (Inception FID, style vs output).
- **Configs:** `swd_adam` (SWD + Adam, 250 steps, proj64), `swd_lbfgs` (SWD +
  pure L-BFGS, 30 steps), `gram_adam` (Gram + Adam, 250 steps — baseline).
- Data: `results/mould_rosin/mould_rosin_benchmark.csv` (600 rows).
- Hardware: AMD RX 6700 XT under ROCm.

Style set: Starry Night, The Scream, Udnie, Composition VII, Shipwreck of the
Minotaur, The Great Wave off Kanagawa, Girl with a Pearl Earring, Impression
Sunrise, The Basket of Apples, Water Lilies.

## Head-to-head (same content set, averaged over 200 pairs)

| Method | SSIM ↑ | LPIPS ↓ | SIFID ↓ |
|---|---|---|---|
| AdaIN | 0.259 | 0.517 | 0.622 |
| SANet | 0.310 | 0.537 | 1.16 |
| CSBNet | 0.360 | 0.478 | 5.10 |
| IEContraAST | 0.407 | 0.444 | 0.860 |
| ArtFlow | 0.455 | 0.482 | 0.966 |
| MCCNet | 0.455 | 0.471 | 2.38 |
| AdaAttN | 0.471 | 0.477 | 18.3 |
| StyTr2 | 0.488 | 0.469 | 1.11 |
| RAST | 0.538 | 0.310 | 1.76 |
| **StyleBridge SWD + L-BFGS** | **0.6565 ± 0.145** | **0.1551 ± 0.094** | 26.10 ± 6.01 |
| **StyleBridge SWD + Adam** | **0.6499 ± 0.138** | **0.2020 ± 0.081** | 25.90 ± 5.82 |
| **StyleBridge Gram + Adam** | **0.5719 ± 0.144** | 0.3394 ± 0.091 | 24.07 ± 6.45 |

Published column: Ioannou & Maddock (2024) Table IX, their own runs of the 9
methods on the Mould & Rosin set (their reported single values, no std given).

## Reading the numbers

1. **Content preservation — StyleBridge is head-and-shoulders above every
   published method.** Best published SSIM is RAST 0.538; our worst config
   (Gram) scores 0.572 and SWD scores 0.650–0.657, all outside the published
   range (0.26–0.54). LPIPS is even clearer: published methods span 0.31–0.54,
   ours 0.155–0.339; the SWD+L-BFGS value (0.155) is ~2× lower than the best
   published (RAST 0.310).

2. **Style fidelity — the mirror image.** Our SIFID (~24–26) is far above
   published (0.6–18). This is a real and reproducible gap on the same content
   set: StyleBridge heavily favours content fidelity at the cost of matching the
   reference's global Inception statistics. The caveat remains that SIFID
   magnitude is sensitive to the Inception feature layer/checkpoint.

3. **L-BFGS beats Adam on content here** and is ~4× faster in wall time
   (6.7 s vs 27.8 s per pair) — it lands very close to the content image.

4. **SWD > Gram on content** (SSIM 0.650/0.657 vs 0.572; LPIPS 0.20/0.16 vs
   0.34), consistent with the Sliced-Wasserstein literature; the SIFID trade-off
   is similar across all configs.

## Reproducibility control

After switching SWD to fixed (seeded) projections, the Gram row is essentially
unchanged (SSIM 0.5719 vs 0.5720 previously) while the SWD rows shifted
modestly. This confirms the determinism fix affects only the SWD objective.

## Caveats

- The published table's SIFID uses an unstated Inception protocol; ours is
  Mixed_6e features with a symmetric-eigen FID. SIFID *magnitudes* across
  pipelines are not strictly comparable, so the style gap is directional.
- LPIPS backbone in the paper is unstated (we use AlexNet, LPIPS default).
- The paper's exact 10 styles are unknown; we used a documented public-domain
  set. This affects style-match numbers but not content-preservation metrics.
- Published values are single reported numbers (no variance); ours include σ.
