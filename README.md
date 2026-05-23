# FOCUS: Footprint-based Offline-to-online Critic Selection

**Transfer-Ready Critics: Auditing Conservatism Footprints for Offline-to-Online RL**
*Submitted to the ICML 2026 Workshop on Decision-Making in Offline-to-Online RL.*

> ⚠️ **Naming note.** The method is called **FOCUS** in the paper. The code uses
> the earlier internal prefix **`cfs` / `CFS-D`** (Conservative-Footprint Selection)
> — e.g. `wsrl/cfs/`, `--use_cfs`, `cfs_mode`. **`cfs` == `FOCUS`**, same method.

---

## What is FOCUS?

In REDQ-style offline-to-online (O2O) fine-tuning, online Bellman targets are
built from a *randomly subsampled minimum* over an ensemble of critic heads.
WSRL transfers the whole offline REDQ ensemble and treats the heads as an
interchangeable target pool. **FOCUS asks a more specific question: after
conservative pretraining, which critic heads should be trusted to generate
online targets?**

FOCUS is a **transition-time audit** (applied once, before online updates) that
scores each critic head by:

- **Conservative footprint** `ρ_h` — one-sided pessimistic Bellman gap + Bellman
  inconsistency on an offline calibration batch.
- **Min-target exposure** `d_h` — how often head `h` wins the REDQ
  min-over-subset selection.
- **Combined score** `η_h = ρ_h · d̃_h`.

From the ranking it builds two stress-test pools (`FOCUS-Low` = lowest-η half,
`FOCUS-High` = highest-η half) and only changes *which heads are allowed to
generate early online targets* — actor, optimizer, replay, and warmup are
untouched.

## Key findings

- **Transient outlier phenomenon.** Across matched 550K–900K checkpoint windows
  on AntMaze **medium-** and **large-diverse** (8 + 8 checkpoints), the
  highest-footprint head identity changes at **every consecutive checkpoint pair
  (7/7 transitions in both envs)**, and **6 of 10** heads become the max at some
  point. Footprint heterogeneity is a *transient checkpoint property*, not a
  fixed "rogue head".
- **Offline success ⊥ critic transfer-readiness.** 550K and 850K both reach
  offline success ≈ 0.65 yet differ in CV(ρ) by ~2× (0.635 vs 0.334).
- **Risky-pool failure mode.** At the high-heterogeneity 850K checkpoint, forcing
  high-footprint heads into targets (`FOCUS-High`) cuts post-burn-in high-success
  occupancy from **0.892 → 0.514** and roughly doubles evaluation volatility.
- **Mixed, honestly reported.** `FOCUS-Low` matches WSRL at 850K but the
  predicted ordering reverses at 550K — FOCUS exposes a **risk–diversity
  tradeoff** rather than being a finished selector.

## Paper artifacts (reproducible from this repo)

| Artifact | Path |
|----------|------|
| Figure 1(a) — heterogeneity vs offline success (both envs) | `paper/figures/checkpoint_audit_figure.pdf` |
| Figure 1(b) — per-head ρ heatmap (medium 8 + large 8) | `paper/figures/rho_heatmap_figure.pdf` |
| Table 1 input — checkpoint audit sweep | `paper/data/seed0_checkpoint_sweep{,_large}.csv` |
| Table 2 input — online metrics | `paper/results/medium_seed0_metrics.csv` |
| Per-head diagnostic CSVs (medium / large) | `paper/results/{seed0_sweep,large_seed0_sweep}/cfs_stats_step*.csv` |
| Aligned 550K–900K summaries | `paper/results/*/_summary_550_900.csv` |

Regenerate figures:
```bash
python analysis/make_checkpoint_audit_figure.py   # Figure 1(a)
python analysis/make_rho_heatmap_figure.py        # Figure 1(b)
python analysis/analyze_rho_heatmap.py            # numerical claims + summary CSVs
```

## FOCUS code map (what's new vs the WSRL base)

| Component | Path | Origin |
|-----------|------|--------|
| **Footprint audit core** | `wsrl/cfs/` (`cfs_stats.py`, `cfs_calibration.py`, `cfs_head_selection.py`, `cfs_target_dominance.py`, `cfs_config.py`) | **FOCUS (new)** |
| **Diagnostic CLI** (per-checkpoint audit) | `analysis/cfs_compute_stats.py` | **FOCUS (new)** |
| **Figure / analysis scripts** | `analysis/make_*_figure.py`, `analysis/analyze_rho_heatmap.py` | **FOCUS (new)** |
| **Online integration** | `finetune.py` (`--use_cfs`, `--cfs_mode`, `--cfs_top_k`, `--cfs_stats_output` flags + transition-time block) | WSRL base + FOCUS hooks |
| **Idempotent patcher** | `apply_cfs_patch.py` | **FOCUS (new)** |
| Agents (CalQL / SAC / REDQ), envs, replay, eval, training loop | `wsrl/agents`, `wsrl/envs`, `wsrl/data`, `wsrl/common`, `wsrl/utils` | **WSRL (upstream)** |

### Key CLI flags (online run)
```bash
python finetune.py --agent calql --config experiments/configs/train_config.py:antmaze_cql \
  --use_redq --resume_path <ckpt> --env antmaze-medium-diverse-v2 \
  --use_cfs --cfs_mode low_eta --cfs_top_k 5 \
  --cfs_stats_output paper/results/<name>.csv \
  --num_offline_steps 0 --num_online_steps 200000 --warmup_steps 1250 --utd 4 --batch_size 1024
```
`--cfs_mode ∈ {low_eta, low_rho, high_eta, random_topk}`. Omit `--use_cfs` for the standard WSRL baseline.

## Setup & reproduction

```bash
conda create -n focus python=3.10 -y && conda activate focus
pip install -r requirements.txt
```
GPU runs use D4RL AntMaze (CalQL+REDQ checkpoints). Diagnostic and figure
scripts run CPU-only from the committed CSVs in `paper/`.

## Status / limitations

This is a workshop-stage study: **single seed (seed-0)**, online stress-tests on
**medium-diverse only** (large-diverse used as diagnostic replication). Planned
upgrades (multi-seed, cross-env online, baseline comparison, adaptive K) and the
full reproduction notes are tracked in the maintainers' private notes.

---

## Built on WSRL

This repository is a **fork of [WSRL](https://github.com/zhouzypaul/wsrl)**
(Zhou et al., *Efficient Online RL Fine-Tuning Need Not Retain Offline Data*,
2024). All base O2O infrastructure — CalQL/SAC/REDQ agents, D4RL dataset
loading, replay buffer, evaluation, and the training loop — is from WSRL. FOCUS
adds the conservatism-footprint audit and head-pool selection layer on top.
See the [upstream WSRL repository](https://github.com/zhouzypaul/wsrl) for the
base framework and its documentation.
