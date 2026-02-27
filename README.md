# CVCI with Causal Forests for Heterogeneous Treatment Effect Estimation

This project extends the **Cross-Validation for Causal Inference (CVCI)** framework of [Yang, Lin, Athey, Jordan, and Imbens (2025)](https://arxiv.org/abs/2307.16227) to **Causal Forests**, enabling cross-validated combination of experimental and observational data for estimating Conditional Average Treatment Effects (CATE).

## The Problem

Experimental data is unbiased but expensive — sample sizes are small and CATE estimates are noisy. Observational data is abundant but potentially biased due to confounding and treatment effect differences. How should a researcher optimally blend these two data sources?

## Our Approach

CVCI provides a principled answer: define a mixing parameter λ ∈ [0, 1] that controls the weight given to observational vs. experimental data, then use cross-validation on the experimental sample to select the optimal λ\*. The hybrid loss is:

```
L(θ, λ) = (1 − λ) · L_exp(θ) + λ · L_obs(θ)
```

When observational data is reliable (low bias, large sample), CVCI increases λ\* to borrow strength. When observational data is biased, CVCI reduces λ\* toward zero, falling back to the experiment. This happens automatically — no assumptions about the form of bias are needed.

We implement this using `econml.grf.CausalForest` (the Athey-Wager generalized random forest) with sample weights to implement the hybrid loss, and compare against:

- **Exp-only CF** (λ = 0): Causal Forest trained on experimental data only
- **Obs-only CF** (λ = 1): Causal Forest trained on observational data only
- **Pooled CF**: Naive concatenation of both datasets with uniform weights
- **CVCI-CF** (λ\*): Our method — cross-validated optimal mixing

## Key Methodological Choices

### Causal Forest: `econml.grf.CausalForest`, not `CausalForestDML`

We use the Athey-Wager generalized random forest (`econml.grf.CausalForest`), **not** the doubly-debiased `CausalForestDML`. This distinction is critical for CVCI:

| | `econml.grf.CausalForest` (current) | `CausalForestDML` (deprecated `_dml` files) |
|---|---|---|
| **Approach** | Solves local moment equations directly | Two-stage: nuisance estimation → residualization → forest on residuals |
| **Moment condition** | E[(Y − τ(x)A − μ₀(x)) · (A; 1) \| X=x] = 0 | R-loss on Robinson residuals Ỹ = Y − m̂(X), Ã = A − ê(X) |
| **Sample weights** | Flow directly into tree splits and leaf estimates | Must propagate through nuisance models, cross-fitting, and residualization |
| **Internal debiasing** | None — relies on the moment condition | Full double/debiased ML pipeline with propensity scores |

**Why this matters for CVCI:** The λ-weighted sample weights are the mechanism by which CVCI controls the experimental/observational trade-off. With `CausalForestDML`, weights must pass through multiple internal stages (nuisance model fitting, cross-fitting, residualization), and the internal debiasing can override or conflict with the weight signal. With `econml.grf.CausalForest`, weights directly control both tree structure and leaf estimates — exactly what CVCI requires.

**Empirical evidence:** CVCI with `CausalForestDML` systematically underperformed naive pooling across all 50-simulation experiments. Diagnostics revealed that sample weights were not propagating cleanly through the DML pipeline. Switching to `econml.grf.CausalForest` resolved this completely, with CVCI outperforming all baselines.

### Cross-Validation Loss: Outcome MSE

The CV criterion for selecting λ is **outcome MSE on held-out experimental data**:

```
Q(λ) = (1/K) Σ_k  mean_{i∈fold_k} ( Y_i − Ŷ_i(λ) )²
```

where `Ŷ_i = θ̂₀(Xᵢ) + Aᵢ · θ̂₁(Xᵢ)` uses both the baseline (θ₀) and treatment effect (θ₁) jointly estimated by the causal forest under λ-weights. This follows the spirit of the original CVCI paper, which uses prediction loss on experimental data.

**Why outcome MSE and not ATE-difference:** An earlier implementation used `Q(λ) = (ATE_val − mean(τ̂_val))²`, which collapses n_val validation observations to a single scalar comparison per fold. With K=5 folds and ~40 observations per fold, this gives ~5 noisy scalars to select λ — insufficient signal. Diagnostics confirmed a bimodal λ\* distribution (effectively a coin-flip between 0 and 1) despite the averaged Q curve having the correct shape. Outcome MSE provides n_val pointwise comparisons per fold, giving strong discriminative signal for λ selection.

### Sample Weight Scheme

For a given λ, experimental and observational data are combined with weights:

```
w_exp = (1 − λ) · n_total / n_exp     (for each experimental unit)
w_obs = λ       · n_total / n_obs     (for each observational unit)
```

Weights sum to n_total for sklearn compatibility. At λ=0, only experimental data is used; at λ=1, only observational data.

## Simulations

We evaluate CVCI-CF across three simulation axes, each with three CATE functions:

### Simulation Axes

| Axis | What varies | What's fixed | What we expect |
|------|-------------|--------------|----------------|
| **A: Treatment effect bias (ε)** | ε ∈ [0, 0.6] | n_obs = 1000, no confounding | λ\* decreases as bias grows |
| **B: Observational sample size** | n_obs ∈ [100, 5000] | ε = 0.1, no confounding | λ\* increases with more obs data |
| **C: Confounding strength** | γ ∈ [0, 2.0] | n_obs = 1000, ε = 0 | λ\* decreases as confounding grows |

In all simulations, the experimental sample (n_exp = 200) is held constant. The experimental data has randomized treatment assignment (propensity = 0.5).

### CATE Functions

| Function | Formula | Purpose |
|----------|---------|---------|
| **Constant** | τ(x) = 1.0 | Sanity check — no heterogeneity |
| **Step** | τ(x) = 2.0 if x₁ > 0, else 0.5 | Subgroup analysis use case |
| **Nonlinear** | τ(x) = 1.0 + 0.5x₁² + 0.3sin(x₂) + 0.2\|x₃\| | Tests CF's nonparametric advantage |

### Data Generating Process

**Experimental data** (unbiased):
```
X ~ N(0, I_d),  W ~ Bern(0.5),  Y = X'θ + W·τ(X) + ξ
```

**Observational data** (biased):
```
X ~ N(0, I_d),  W ~ Bern(σ(γ'X)),  Y = X'θ + W·(τ(X) + ε) + ξ
```

where θ is shared across datasets, ε is additive treatment effect bias, and γ controls confounding strength through the propensity score.

### Metrics

- **CATE MSE**: mean((τ̂(x) − τ(x))²) — pointwise accuracy of heterogeneous effects
- **ATE MSE**: (mean(τ̂(x)) − mean(τ(x)))² — average effect accuracy
- **Optimal λ\***: the cross-validated mixing parameter

Full τ̂(x) vectors are saved per simulation, enabling post-hoc computation of coverage, rank correlation, and other metrics without re-running.

## Repository Structure

```
CVCICF/
├── src/
│   ├── causal_forest_cv.py          # CVCI-CF wrapper (econml.grf.CausalForest)
│   ├── causal_forest_cv_dml.py      # [deprecated] CVCI-CF with CausalForestDML
│   ├── data_generation.py           # Synthetic data DGPs and CATE functions
│   └── dml_cv.py                    # DML-based CVCI (ATE estimation)
├── experiments/
│   ├── cf_simulations.py            # Main CF simulation runner (3 axes × 3 CATEs)
│   ├── cf_simulations_dml.py        # [deprecated] Simulations with CausalForestDML
│   ├── controlled_simulations_dml.py # DML controlled experiments
│   └── varying_nobs_dml.py          # DML varying n_obs experiments
├── sherlock/
│   ├── sherlock_setup.sh            # One-time environment setup on Sherlock
│   ├── sherlock_submit.sh           # SLURM array job submission
│   ├── sherlock_collect.py          # Post-run results collection and plotting
│   └── diagnose_results.py         # Diagnostic analysis of saved results
├── results/                         # Simulation outputs (auto-created)
├── notebooks/                       # Analysis notebooks
├── tests/
├── docs/
├── requirements.txt
├── setup.sh
└── README.md
```

## Getting Started

### Prerequisites

Python 3.9+ with: `numpy`, `scipy`, `scikit-learn`, `econml`, `matplotlib`, `pandas`

### Local Installation

```bash
git clone https://github.com/MelvinLiamPK/CVCICausalForest.git CVCICF
cd CVCICF
pip install -r requirements.txt
```

### Running Locally

The simulation runner supports four speed modes:

```bash
cd experiments

# Prototype — verify the pipeline doesn't crash (~1 min total)
python3 cf_simulations.py --prototype

# Ultra-quick — check that λ* adapts correctly (~5 min)
python3 cf_simulations.py --ultra-quick

# Quick — enough sims for rough trends (~30 min per axis × CATE)
python3 cf_simulations.py --quick

# Full — publication-quality results (50 sims, ~2-3 hours per axis × CATE)
python3 cf_simulations.py
```

Filter by axis or CATE function:

```bash
# Only vary epsilon, only step CATE
python3 cf_simulations.py --quick --axis epsilon --cate step

# All axes, constant CATE only
python3 cf_simulations.py --quick --cate constant
```

Override number of simulations:

```bash
python3 cf_simulations.py --quick --n-sims 25 --axis nobs
```

Re-plot saved results:

```bash
python3 cf_simulations.py --mode plot --results-dir ../results/cf_simulations/epsilon_constant_20260218_...
```

Results auto-save to `results/cf_simulations/` with incremental JSON checkpointing (safe to interrupt).

### Running on Stanford Sherlock

Sherlock parallelizes the 9 experiments (3 axes × 3 CATEs) as SLURM array jobs.

**One-time setup:**

```bash
ssh your_sunet@login.sherlock.stanford.edu
cd ~/CVCICF/sherlock
bash sherlock_setup.sh
```

**Run simulations:**

```bash
cd ~/CVCICF

# Quick sanity test
source ~/envs/cvci_cf/bin/activate
python3 experiments/cf_simulations.py --ultra-quick --axis epsilon --cate constant --n-sims 1

# Full run — 9 parallel jobs
sbatch sherlock/sherlock_submit.sh
```

**Collect results:**

```bash
# Check which experiments completed
python3 sherlock/sherlock_collect.py $SCRATCH/cvci_cf_results

# Run diagnostics (Q-curve shape, bias-variance decomposition)
python3 sherlock/diagnose_results.py $SCRATCH/cvci_cf_results
```

**Monitor jobs:**

```bash
squeue -u $USER                                         # Job status
tail -f logs/cvci_cf_*.out                              # Live output
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS   # Resource usage
```

### Adjusting Simulation Parameters

All parameters are centralized in `get_default_config()` at the top of `cf_simulations.py`:

```python
# Sample sizes
'n_exp': 200,              # Experimental sample size
'epsilon_n_obs': 1000,     # Obs sample size when varying epsilon

# Bias range
'epsilon_vals': np.linspace(0, 0.6, 13),

# Obs sample sizes to test
'nobs_vals': np.array([100, 200, 500, 1000, 2000, 5000]),

# Confounding range
'confounding_vals': np.linspace(0, 2.0, 9),

# CVCI settings
'lambda_bin': 21,          # Grid points for λ ∈ [0, 1]
'n_estimators': 200,       # Trees per causal forest
'n_sims': 50,              # Monte Carlo repetitions
```

## Output Format

Each experiment creates a timestamped directory:

```
results/cf_simulations/epsilon_constant_20260218_190244/
├── metadata.json        # Experiment configuration
├── results_all.json     # Full results: τ̂(x) vectors, MSEs, λ* per sim
├── summary.json         # Aggregated mean/SE across simulations
└── results_plot.png     # Three-panel figure: CATE MSE, ATE MSE, λ*
```

## References

- Yang, L., Lin, L., Athey, S., Jordan, M. I., & Imbens, G. W. (2025). Cross-Validation for Causal Inference. *arXiv:2307.16227*.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests. *Annals of Statistics*.
- Kallus, N., Puli, A. M., & Shalit, U. (2018). Removing Hidden Confounding by Experimental Grounding. *NeurIPS*.