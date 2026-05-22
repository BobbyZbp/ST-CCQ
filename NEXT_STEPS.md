# FOCUS — Next Steps

State as of the ICML workshop submission (seed-0, medium online + medium/large diagnostic).
This file is the pick-up-where-I-left-off list.

## Where the paper currently stands

- **Diagnostic claim (strong):** transient outlier phenomenon replicates on medium + large (8+8 ckpts).
- **Online claim (weak):** only 850K cleanly confirms the mechanism; 550K reverses; 600K/900K null.
- **Single seed.** This is the #1 reviewer objection.

Honest self-grade: solid *workshop* paper (~60% accept, ~7% spotlight). Not yet main-track.

## Priority-ordered upgrades

### P0 — Multi-seed (biggest review-impact change)
- Run **3 seeds** (seed 0/1/2) for all 4 medium ckpts × 3 methods (WSRL / FOCUS-Low / FOCUS-High).
- Report mean ± std; check whether the 850K failure and the 550K reversal survive seeds.
- 36 runs × ~80 min ≈ ~48 GPU-h.
- **If 850K holds across seeds → the core claim becomes defensible.**

### P0 — Cross-env ONLINE (not just diagnostic)
- Large-diverse 1M pretrain ckpt already exists (`calql_redq_pretrain_large*`).
- Run large 900K (CV=0.824, the extreme outlier ckpt) × {WSRL, FOCUS-Low, FOCUS-High} × 3 seeds.
- Requires large offline success ≥ ~0.35 (currently 900K ≈ 0.35 — borderline; consider 850K large too).

### P1 — Baseline comparison (currently only vs WSRL)
- Add at least one alternative critic-aggregation baseline:
  - DEA (directional ensemble aggregation, Werge et al. 2025) — cited but not compared.
  - Uncertainty/disagreement-weighted target.
- Needed to answer "is FOCUS-Low better than just down-weighting high-variance heads?"

### P1 — Ablate the η decomposition
- Compare η-ranking (ρ·d̃) vs ρ-only ranking vs d-only ranking on 850K.
- The whole point of `d_h` is unproven without this.

### P1 — K-sweep / less destructive intervention
- K=5 (half-ensemble) is aggressive and reverses 550K. Try K=7/8/9 (Trim-1/2/3).
- Hypothesis: removing only the literal outlier (Trim-1, K=9) may fix the 550K reversal
  without hurting 850K. (We set up but did NOT run this — see git history for cfs_*_trim1.)

### P2 — Adaptive / smooth selection (turns "audit" into "method")
- Replace binary half-cut with η-weighted target sampling, or CV-gated selection
  ("apply FOCUS-Low only when CV(ρ) > threshold").
- This is the path from workshop → main-track contribution.

## What main-track would additionally need (beyond the above)

Pick ONE:
- **Multi-task diversity:** add a non-AntMaze family (D4RL Kitchen or Adroit) and show
  the diagnostic + online claim there.
- **Theory:** connect ρ_h to an overestimation/Bellman-residual bound (why footprint predicts drift).
- **Clear SOTA win:** adaptive FOCUS beats WSRL + DEA across all ckpts, no reversal.

## Open empirical questions

- Is the 550K reversal real or single-seed noise? (P0 multi-seed answers this.)
- Does the 850K failure hold under *standard* (not reduced) warmup?
- Does removing only the top-1 outlier head (K=9) avoid the 550K reversal?
- Why does Corr(ρ,d) flip sign across checkpoints? (Currently just observed, not explained.)

## Compute reference

| Tier | GPU-h | Wall on 2×4090 |
|------|------:|---------------:|
| 3 seeds medium | ~120 | ~3 d |
| + cross-env online (3 seeds) | +180 | ~4 d |
| + baselines on 850K | +60 | ~1.5 d |
| + new env (Kitchen, 3 seeds) | +200 | ~5 d |

## Infra notes

- Pretrain config (must match for new ckpts): `--agent calql --use_redq --batch_size 256
  --utd 1 --reward_scale 10 --reward_bias -5 --num_offline_steps 1_000_000 --save_interval 50000`.
- Online config: `--utd 4 --batch_size 1024 --warmup_steps 1250 --num_online_steps 200000`.
- wandb: everything lands in project `wsrl` (the `--project` flag is overridden); filter by
  run-name regex — see `results/cfs/NAMING.md`.
- Dedup wandb runs by `(ckpt, method)` keeping max `n_evals` (many crashed predecessors).
