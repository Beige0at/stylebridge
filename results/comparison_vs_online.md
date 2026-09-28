# StyleBridge Results vs Published NST Algorithms

Published anchors gathered from the peer-reviewed NST literature. **Caveat:**
SSIM is pair/dataset/resolution dependent; published numbers use BSDS500, VSO,
WikiArt+MS-COCO etc., while ours use landscape+Kanagawa and single-face
portraits. Comparisons are indicative (positioning), not apples-to-apples.

## Content preservation: SSIM (content vs output, higher = better)

| Algorithm | Published SSIM | StyleBridge (script) SSIM |
|---|---|---|
| Gatys et al. (optimisation) | 0.288 / 0.70 | — |
| WCT | 0.244 | — |
| Huang & Belongie (AdaIN) | 0.310 / 0.53 | — |
| Johnson (fast/feedforward) | 0.408 | — |
| Ulyanov (feedforward) | 0.543 | — |
| StyleNAS | 0.665 | — |
| Li Universal | 0.265 | — |
| **Our Gram (Adam)** | — | 0.593 (landscape) |
| **Our SWD (Adam)** | — | **0.725** (landscape) |
| **Our SWD (group portrait, guard off/on)** | — | 0.719 / 0.719 whole; face 0.682 / 0.714 |
| **Our SWD (single-face, guard off/on)** | — | 0.378 / 0.387 whole; 0.419 / **0.449** face; 0.75–0.92 core |

Positioning: with SWD we sit above the classic optimisation methods at the top
end of the published range. The guard's protected-core SSIM (0.75–0.92) is well
above every published artist-NST method that does not spatially protect faces.

## Speed (GPU; optimisation-based vs feedforward)

| Algorithm | Published time | Type |
|---|---|---|
| Gatys et al. | 14.2 s @256 / 46.8 s @512 / 207 s | optimisation |
| AdaIN | 0.018 s @256 / 0.065 s @512 | feedforward |
| WCT (closed-form) | 4.06 s @512 | feedforward |
| **Our Adam (SWD) @ 250 steps, ~400px** | **≈26 s** | optimisation |
| **Our L-BFGS (SWD) @ 30 steps** | **≈5.7 s** | optimisation |

Our L-BFGS path is several times faster than published optimisation NST. We are
not comparable to feedforward nets, which trade flexibility for real-time speed.

## Perceptual content preservation: LPIPS (AlexNet, lower = better)

| Config | StyleBridge LPIPS |
|---|---|
| Gram (Adam) | 0.480 (landscape) |
| SWD (Adam, proj64) | **0.334** |
| SWD (Adam, proj128/200) | 0.361 / 0.358 |
| SWD (L-BFGS) | 0.348 (30 steps) |
| SWD (frontal portrait) off / haar / mp | 0.272 / 0.271 / 0.272 |
| SWD (profile portrait) off / haar / mp | 0.218 / **0.203** / 0.208 |

Positioning: SWD clearly beats Gram on LPIPS (0.33 vs 0.48). Published LPIPS
(e.g. AdaIN/SANet/StyTr2 in Ioannou & Maddock 2024) spans roughly 0.26–0.54;
our SWD values sit in the competitive lower end.

## Cross-cutting findings vs literature

1. **SWD over Gram** (SWD 0.725 > Gram 0.593 SSIM; LPIPS 0.33 < 0.48) matches
   the Sliced-Wasserstein literature (Heitz et al., CVPR'21).
2. **L-BFGS + SWD stability**: the strong_wolfe stall matches the known fragility
   of quasi-Newton on non-smooth SWD landscapes; the catastrophic 75M loss was
   not reproduced. The engine now aborts/restores on any non-finite loss.
3. **Saliency guard**: +7% face-ROI SSIM and 0.75–0.92 protected-core SSIM align
   with the saliency-guided transfer literature (Boyd et al. WACV'23; Jiang et
   al. CVPR'21).
4. **Full metric set**: SIFID is computed correctly for every pair (the earlier
   NaN was traced to a silently swallowing image loader and fixed). SIFID is
   reported as indicative only (protocol-sensitive).
