"""
Synthetic Data Generation for CVCI with Heterogeneous Treatment Effects

Generates data where:
- Experimental data has randomized treatment (unbiased CATE)
- Observational data has confounded treatment and potentially biased CATE
- Treatment effects vary across individuals (heterogeneous)

DGP designs:
1. Linear heterogeneity: τ(x) = τ₀ + x^T β_τ
2. Nonlinear heterogeneity: τ(x) = τ₀ + f(x) for nonlinear f
3. Step function: τ(x) differs by subgroup
"""

import numpy as np
from typing import Tuple, Optional, Callable


def generate_heterogeneous_data(
    n_exp: int,
    n_obs: int, 
    d: int,
    tau_func: Callable,
    epsilon: float = 0.0,
    sigma: float = 1.0,
    obs_propensity: float = 0.2,
    confounding_strength: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, ...]:
    """
    Generate synthetic data with heterogeneous treatment effects.
    
    Experimental:
        Z ~ N(0, I_d)
        W ~ Bern(0.5)                    [randomized]
        Y = Z^T θ + W × τ(Z) + ξ        [true CATE]
    
    Observational:
        Z ~ N(0, I_d)
        W ~ Bern(σ(γ^T Z))              [confounded propensity]
        Y = Z^T θ + W × (τ(Z) + ε) + ξ  [biased CATE]
    
    Args:
        n_exp: Number of experimental samples
        n_obs: Number of observational samples
        d: Covariate dimension
        tau_func: Function mapping covariates to true CATE, τ(x)
        epsilon: Additive bias in observational treatment effect
        sigma: Noise standard deviation
        obs_propensity: Base propensity in obs data (used if confounding_strength=0)
        confounding_strength: Strength of covariate-dependent confounding
        seed: Random seed
        
    Returns:
        X_exp, A_exp, Y_exp: Experimental data
        X_obs, A_obs, Y_obs: Observational data
        true_cate_exp: True CATE values for experimental units
        true_cate_obs: True CATE values for observational units
    """
    rng = np.random.default_rng(seed)
    
    # Outcome model coefficients (shared)
    theta = rng.normal(0, 1, size=d)
    
    # --- EXPERIMENTAL DATA ---
    X_exp = rng.normal(0, 1, size=(n_exp, d))
    A_exp = rng.binomial(1, 0.5, size=n_exp)  # Randomized
    true_cate_exp = tau_func(X_exp)
    Y_exp = (X_exp @ theta 
             + A_exp * true_cate_exp 
             + rng.normal(0, sigma, size=n_exp))
    
    # --- OBSERVATIONAL DATA ---
    X_obs = rng.normal(0, 1, size=(n_obs, d))
    
    # Confounded propensity
    if confounding_strength > 0:
        # Propensity depends on covariates via logistic model
        gamma = rng.normal(0, 1, size=d) * confounding_strength
        logit = X_obs @ gamma
        propensity = 1.0 / (1.0 + np.exp(-logit))
    else:
        propensity = np.full(n_obs, obs_propensity)
    
    A_obs = rng.binomial(1, propensity)
    true_cate_obs = tau_func(X_obs)
    
    # Biased treatment effect in obs data
    Y_obs = (X_obs @ theta 
             + A_obs * (true_cate_obs + epsilon) 
             + rng.normal(0, sigma, size=n_obs))
    
    return (X_exp, A_exp, Y_exp, 
            X_obs, A_obs, Y_obs, 
            true_cate_exp, true_cate_obs)


# ============================================================
# Pre-defined CATE functions
# ============================================================

def constant_cate(tau0=1.0):
    """Constant treatment effect τ(x) = τ₀."""
    def tau_func(X):
        return np.full(X.shape[0], tau0)
    tau_func.true_ate = tau0
    tau_func.name = f"constant(τ₀={tau0})"
    return tau_func


def linear_cate(tau0=1.0, beta_tau=None, d=5):
    """Linear heterogeneity: τ(x) = τ₀ + x^T β_τ."""
    if beta_tau is None:
        beta_tau = np.zeros(d)
        beta_tau[0] = 0.5  # Only first covariate matters
    
    def tau_func(X):
        return tau0 + X @ beta_tau
    tau_func.true_ate = tau0  # E[τ(X)] = τ₀ when X ~ N(0, I)
    tau_func.name = f"linear(τ₀={tau0})"
    return tau_func


def nonlinear_cate(tau0=1.0):
    """
    Nonlinear heterogeneity: 
    τ(x) = τ₀ + 0.5 * x₁² + 0.3 * sin(x₂) + 0.2 * |x₃|
    """
    def tau_func(X):
        base = tau0
        if X.shape[1] >= 1:
            base = base + 0.5 * X[:, 0] ** 2
        if X.shape[1] >= 2:
            base = base + 0.3 * np.sin(X[:, 1])
        if X.shape[1] >= 3:
            base = base + 0.2 * np.abs(X[:, 2])
        return base
    tau_func.true_ate = tau0 + 0.5  # E[X₁²] = 1 for X₁ ~ N(0,1)
    tau_func.name = f"nonlinear(τ₀={tau0})"
    return tau_func


def step_cate(tau_low=0.5, tau_high=2.0):
    """
    Step function heterogeneity:
    τ(x) = τ_high if x₁ > 0 else τ_low
    
    Two subgroups with different treatment effects.
    """
    def tau_func(X):
        return np.where(X[:, 0] > 0, tau_high, tau_low)
    tau_func.true_ate = (tau_low + tau_high) / 2  # Equal mass above/below 0
    tau_func.name = f"step(τ_low={tau_low}, τ_high={tau_high})"
    return tau_func


def interaction_cate(tau0=1.0):
    """
    Interaction heterogeneity:
    τ(x) = τ₀ + 0.5 * x₁ * x₂
    
    Treatment effect depends on interaction of two covariates.
    """
    def tau_func(X):
        if X.shape[1] >= 2:
            return tau0 + 0.5 * X[:, 0] * X[:, 1]
        return np.full(X.shape[0], tau0)
    tau_func.true_ate = tau0  # E[X₁X₂] = 0 for independent normals
    tau_func.name = f"interaction(τ₀={tau0})"
    return tau_func
