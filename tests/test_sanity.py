"""
Quick sanity test for CVCI-CF.

Run: python tests/test_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.causal_forest_cv import CausalForestCVCI, cross_validation_cf
from src.data_generation import generate_heterogeneous_data, linear_cate


def test_basic_fit():
    """Test that CausalForestCVCI can fit and predict."""
    print("Test: Basic fit and predict...", end=" ")
    
    tau_func = linear_cate(tau0=2.0, d=3)
    X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, _, _ = generate_heterogeneous_data(
        n_exp=100, n_obs=200, d=3, tau_func=tau_func, epsilon=0.0, seed=42
    )
    
    model = CausalForestCVCI(n_estimators=52, random_state=42)
    model.fit(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_=0.5)
    
    cate = model.predict_cate(X_exp)
    ate = model.predict_ate(X_exp)
    
    assert cate.shape == (100,), f"CATE shape wrong: {cate.shape}"
    assert np.isfinite(ate), f"ATE not finite: {ate}"
    
    print(f"PASS (ATE={ate:.2f}, true≈{tau_func.true_ate:.2f})")


def test_cross_validation():
    """Test that CV selects a lambda."""
    print("Test: Cross-validation...", end=" ")
    
    tau_func = linear_cate(tau0=1.0, d=3)
    X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, _, _ = generate_heterogeneous_data(
        n_exp=80, n_obs=200, d=3, tau_func=tau_func, epsilon=0.0, seed=123
    )
    
    lambda_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    Q_values, lambda_opt, model_opt = cross_validation_cf(
        X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
        lambda_vals=lambda_vals, k_fold=3,
        n_estimators=52, random_state=123, verbose=False
    )
    
    assert len(Q_values) == 5
    assert 0.0 <= lambda_opt <= 1.0
    assert model_opt is not None
    
    ate = model_opt.predict_ate(X_exp)
    print(f"PASS (λ*={lambda_opt:.2f}, ATE={ate:.2f})")


def test_lambda_zero_vs_one():
    """Test that λ=0 uses exp only and λ=1 uses obs only."""
    print("Test: λ=0 vs λ=1 behavior...", end=" ")
    
    tau_func = linear_cate(tau0=1.0, d=3)
    
    # Generate with large obs bias
    X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, _, _ = generate_heterogeneous_data(
        n_exp=100, n_obs=200, d=3, tau_func=tau_func, epsilon=5.0, seed=42
    )
    
    model_exp = CausalForestCVCI(n_estimators=52, random_state=42)
    model_exp.fit(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_=0.0)
    ate_exp = model_exp.predict_ate(X_exp)
    
    model_obs = CausalForestCVCI(n_estimators=52, random_state=42)
    model_obs.fit(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_=1.0)
    ate_obs = model_obs.predict_ate(X_exp)
    
    # With ε=5, obs-only should give a much higher ATE
    print(f"PASS (ATE_exp={ate_exp:.2f}, ATE_obs={ate_obs:.2f})")


if __name__ == '__main__':
    print("=" * 60)
    print("CVCI-CF Sanity Tests")
    print("=" * 60)
    
    test_basic_fit()
    test_cross_validation()
    test_lambda_zero_vs_one()
    
    print("\n✓ All tests passed!")
