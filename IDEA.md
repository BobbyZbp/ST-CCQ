# FOCUS — Core Idea & Research Notes

The "why", written for future-me. Code/results live elsewhere; this is the thinking.

## One-sentence idea

A conservatively pretrained REDQ critic ensemble is transferred wholesale into
online fine-tuning, but its heads are **not equally trustworthy** as online
Bellman-target generators — and *which* heads are risky is a **checkpoint-dependent,
transient** property that offline policy success does not reveal. FOCUS audits this
at transfer time.

## The mechanism we believe in

1. Offline conservative training (CalQL) pushes some critic heads to be more
   pessimistic than others — unevenly, and differently at each checkpoint.
2. Online REDQ targets = `min` over a random size-2 subset of the 10 heads. A
   single over-pessimistic head, if frequently selected by the `min`, repeatedly
   drags the target down → bad learning signal during the fragile transition.
3. So we score each head by **footprint** `ρ_h` (how pessimistically it deviates
   from a checkpoint-internal Bellman reference) and **exposure** `d_h` (how often
   it wins the `min`). High `η = ρ·d̃` = "pessimistic AND frequently in the target".

## What we actually found (and how strong it is)

| Finding | Strength | Evidence |
|---------|----------|----------|
| Footprint heterogeneity varies wildly across checkpoints (CV 0.04→0.82) | **strong** | medium 8 + large 8 ckpt sweep |
| Max-footprint head identity is transient (7/7 transitions, both envs) | **strong** | per-head CSVs |
| Offline success ⊥ critic heterogeneity (550K=850K success, 2× CV gap) | **strong** | Table 1 |
| Forcing high-footprint heads hurts online (850K: Frac 0.892→0.514) | **medium** | single seed, 1 ckpt |
| FOCUS-Low helps / is safe | **weak** | matches WSRL at 850K, reverses at 550K |

The diagnostic story is the real contribution. The intervention is a stress-test,
not a finished algorithm — and we say so.

## The 550K counterexample (the honest crack in the story)

At 550K (highest CV=0.635, max ρ=10.5 on head 6), the predicted ordering
**reverses**: FOCUS-High slightly *beats* WSRL, FOCUS-Low *underperforms*.

Candidate explanations (untested, ranked by plausibility):
1. **Early-training useful pessimism.** 550K offline success ≈ 0.65 but the policy
   is still immature; the high-footprint head may carry *useful* conservative
   anchoring that prevents early online over-optimism. Removing it (FOCUS-Low)
   lets the policy over-explore unreliable regions.
2. **K=5 is too aggressive.** Cutting half the ensemble removes useful diversity,
   not just the one outlier. Trim-1 (K=9) might not reverse. **(testable, high priority)**
3. **Single-seed noise.** 550K Final swings (WSRL 1.00 vs FOCUS-Low 0.85) are
   single-trajectory artifacts; last-5 means are closer. **(multi-seed answers this)**

→ This is why the paper's framing is "risk–diversity tradeoff" not "FOCUS wins".
If #2 is right, there's a real method here (adaptive/Trim-K). If #3, the whole
online story is underpowered. **Resolve via multi-seed + K-sweep first.**

## Why this is a different angle from prior work

Prior critic-ensemble work = pessimism / uncertainty / exploration / adaptive
aggregation. All ask "how to *combine* heads". FOCUS asks "what *structure* exists
inside the transferred ensemble, and is it transfer-ready?" — a diagnostic /
audit framing, not a new aggregation rule. The object of study is the
**checkpoint-internal critic anatomy**, not the algorithm.

## The honest ceiling

- As-is: workshop poster. Diagnostic is interesting; online is single-seed + mixed.
- The transient-outlier phenomenon is genuinely novel and visual (the heatmap).
- To break into main-track: need multi-seed + (multi-task OR theory OR a method
  that actually wins). See NEXT_STEPS.md.

## Naming history (avoid confusion)

- Project went BT-CCQ → ST-CCQ → **FOCUS** (paper name). GitHub repo renamed to FOCUS.
- Code internals still use `cfs` / `CFS-D` (Conservative-Footprint Selection, the
  earlier name). `cfs == FOCUS`. `btccq` in code = an earlier calibration variant,
  largely superseded; FOCUS online runs use `--config ...:antmaze_cql --use_cfs`.

## Definitions (so I don't re-derive)

- `ρ_h = p_h + 0.1·e_h` where `p_h` = mean one-sided pessimistic gap
  `(z_off − Q_h)_+`, `e_h` = mean squared Bellman inconsistency. `z_off` = frozen
  checkpoint-internal reference `r + γ·min_j Q_j(s',π(s'))`.
- `CV(ρ)` = std/mean over heads. <0.05 no-signal, >0.15 strong heterogeneity.
- `d_h` = P(head h wins the REDQ min over a random size-2 subset). `d̃_h = d_h/(1/H)`.
- `η_h = ρ_h · d̃_h`. FOCUS-Low = top-K lowest η; FOCUS-High = top-K highest η.
- Settings: H=10 heads, M=2 subsample, K=5 pool.
