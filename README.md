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

We implement this using `econml.CausalForestDML` with sample weights to approximate the hybrid loss, and compare against:

- **Exp-only CF** (λ = 0): Causal Forest trained on experimental data only
- **Obs-only CF** (λ = 1): Causal Forest trained on observational data only
- **Pooled CF**: Naive concatenation of both datasets with uniform weights
- **CVCI-CF** (λ\*): Our method — cross-validated optimal mixing

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
| **Constant** | τ(x) = 1.0 | Sanity check — no heterogeneity, DML should match CF |
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
│   ├── causal_forest_cv.py      # CausalForestCVCI class + cross_validation_cf()
│   ├── data_generation.py       # Synthetic data DGPs and CATE functions
│   └── dml_cv.py                # DML-based CVCI (ATE estimation)
├── experiments/
│   └── cf_simulations.py        # Main simulation runner (3 axes × 3 CATEs)
├── sherlock/
│   ├── sherlock_setup.sh         # One-time environment setup on Sherlock
│   ├── sherlock_submit.sh        # SLURM array job submission
│   └── sherlock_collect.py       # Post-run results collection and plotting
├── results/                      # Simulation outputs (auto-created)
├── notebooks/                    # Analysis notebooks
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
cd ~/CVCICF/sherlock

# 1. Prototype test — verify nothing crashes (~1 min)
sbatch sherlock_submit.sh --prototype

# 2. Check it worked
squeue -u $USER
cat logs/cvci_cf_*.out

# 3. Quick run — 9 parallel jobs, ~30 min each
sbatch sherlock_submit.sh --quick

# 4. Full run — 9 parallel jobs, ~2-3 hours each
sbatch sherlock_submit.sh
```

**Collect results:**

```bash
# Check which experiments completed
python3 sherlock_collect.py $SCRATCH/cvci_cf_results

# Create combined 3×3 figure
python3 sherlock_collect.py $SCRATCH/cvci_cf_results --plot
```

**Monitor jobs:**

```bash
squeue -u $USER                                         # Job status
tail -f logs/cvci_cf_*.out                              # Live output
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS   # Resource usage
```

**Customize Sherlock resources** by editing `sherlock_submit.sh`:

```bash
#SBATCH --partition=normal       # Change to 'owners' if you have access
#SBATCH --time=04:00:00          # Increase for full runs if needed
#SBATCH --cpus-per-task=8        # Cores per job
#SBATCH --mem=16G                # Memory per job
#SBATCH --account=your_pi_group  # Uncomment and set your PI group
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
