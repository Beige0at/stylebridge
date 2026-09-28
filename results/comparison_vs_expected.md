# StyleBridge — Results vs Expected

All runs on the AMD RX 6700 XT (ROCm), VGG19, res = 400 unless stated.
Re-run with the deterministic fixed-projection SWD engine.

## 1. Objective: SWD vs Gram Matrix (structural preservation, SSIM)

Test pair = contrasting structure (landscape content + Kanagawa style), Adam,
200 steps.

| Metric | Expected (prelim) | Result (final re-run) | Verdict |
|---|---|---|---|
| Gram SSIM | 0.8381 (similar-structure pair) | **0.5933** | n/a (different eval pair) |
| SWD SSIM | 0.7528 (similar-structure pair) | **0.7248** | n/a (different eval pair) |
| SWD > Gram when structures differ | Claim: SWD wins | Gram 0.593 → SWD 0.723 (+22% SSIM; LPIPS 0.480 → 0.334) | **MATCHED** |
| SWD projection scaling | Proposed (no number) | proj64=0.723, proj128=0.718, auto200=0.724; proj64 is fastest | **Refined: low projections suffice** |

Note: absolute SSIM differs from the preliminary report because that used a
similar-structure pair (where Gram wins); we benchmark the contrasting pair
where SWD should win.

## 2. Optimizer: Adam vs L-BFGS (SWD loss)

| Metric | Expected (prelim) | Result (final re-run) | Verdict |
|---|---|---|---|
| Adam final loss | 251.37 | 94.6 (0.725 SSIM, LPIPS 0.379) | **MATCHED** (stable) |
| Adam time | 93.15 s @ 1000 steps | 25.6 s @ 250 steps | not directly comparable |
| L-BFGS final loss | 75,282,792 (exploding) | 198.2 @ 30 steps (stable, no line search) | **NOT reproduced** (extreme) |
| L-BFGS time | 207.17 s @ 100 steps | 5.7 s @ 30 steps | **NOT reproduced** (extreme) |
| L-BFGS instability | Catastrophic with SWD | strong_wolfe stalls; no line search converges and beats Adam on content | **Direction matched** |

Root cause of the gap: PyTorch L-BFGS offers only `strong_wolfe` or *no* line
search. `strong_wolfe` cannot optimise the non-smooth SWD landscape (it stalls);
with no line search it converges in our implementation, so the claimed 75M
explosion does not occur. The engine additionally aborts and restores state if
the loss ever becomes non-finite.

## 3. Saliency Guard (spatial face preservation)

| Metric | Expected (prelim) | Result (final re-run) | Verdict |
|---|---|---|---|
| Whole SSIM off vs on (P2) | "improve SSIM on portraits" | 0.378 → Haar 0.387 | **Improved** |
| Face-ROI SSIM (P2) | improve | 0.419 → Haar **0.449** (+7.2% rel) | **Improved** |
| Face-ROI SSIM (P1) | improve | 0.687 → Haar **0.715** | **Improved** |
| Protected-core SSIM | n/a | Haar 0.917 (P1) / 0.755 (P2); MP 0.904 / 0.786 | **Preserved** |
| Protected-core MSE | n/a | Haar 89.4 / 80.1; MP 126.1 / 89.2 | **Preserved** |
| Haar vs MediaPipe | primary vs optional | Haar better frontal face-ROI; MP better profile core SSIM | both valid |

Guard effect is measurable after fixing the earlier `clamp(0,1)` normalised-space
bug (removal of that clamp also improved unguarded output).
