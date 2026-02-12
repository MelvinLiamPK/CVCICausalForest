# CVCI with Causal Forests

**Cross-Validation for Causal Inference with Heterogeneous Treatment Effects**

Extends the [Yang, Athey, and Imbens (2025)](https://arxiv.org/abs/2501.04908) CVCI framework from linear/DML models to **Causal Forests**, enabling cross-validated combination of experimental and observational data for estimating **Conditional Average Treatment Effects (CATE)**.

## Key Idea

When estimating heterogeneous treatment effects, researchers often have:
- **Small experimental data** (unbiased but noisy CATE estimates)
- **Large observational data** (potentially biased but more precise)

CVCI-CF uses cross-validation to learn a mixing parameter λ that optimally combines both data sources, automatically adapting to the quality of the observational data.

## Project Structure

```
CVCICF/
├── src/                    # Core implementation
│   ├── causal_forest_cv.py # CVCI with Causal Forests
│   ├── data_generation.py  # Synthetic data with heterogeneous effects
│   └── utils.py            # Shared utilities
├── experiments/            # Experiment scripts
├── results/                # Saved results (JSON)
├── notebooks/              # Analysis notebooks
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Key Dependencies

- `econml` - Microsoft's Causal ML library (Causal Forests)
- `scikit-learn` - Random Forests for nuisance estimation
- `numpy`, `pandas`, `matplotlib`

## References

- Yang, S., Athey, S., & Imbens, G. (2025). *Cross-Validation for Causal Inference.*
- Athey, S., Tibshirani, J., & Wager, S. (2019). *Generalized Random Forests.* Annals of Statistics.
- Wager, S. & Athey, S. (2018). *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests.* JASA.
