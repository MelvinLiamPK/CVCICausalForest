"""
CVCI-CF Simulations: Causal Forest with Cross-Validated Mixing

Compares CATE estimation across methods:
  1. Exp-only CF (lambda=0): Causal Forest on experimental data only
  2. Obs-only CF (lambda=1): Causal Forest on observational data only
  3. Pooled CF: Naive concatenation with uniform weights
  4. CVCI-CF (lambda*): Cross-validated optimal mixing

Three simulation axes:
  A. Varying epsilon (treatment effect bias) with fixed n_obs, confounding=0
  B. Varying n_obs with fixed epsilon, confounding=0
  C. Varying confounding_strength with fixed n_obs, epsilon=0

Three CATE functions: constant, step, nonlinear

Reports both CATE MSE (pointwise) and ATE MSE.
Saves full tau_hat(x) vectors for post-hoc metric computation.

Usage:
    python cf_simulations.py                          # Full run
    python cf_simulations.py --quick                  # Quick test (~30 min)
    python cf_simulations.py --ultra-quick             # Ultra fast (~5 min)
    python cf_simulations.py --axis epsilon            # Only vary epsilon
    python cf_simulations.py --axis nobs               # Only vary n_obs
    python cf_simulations.py --axis confounding        # Only vary confounding
    python cf_simulations.py --cate step               # Only step CATE
    python cf_simulations.py --mode plot --results-dir X  # Plot saved results
"""

import numpy as np
import json
import os
import sys
import argparse
import time
from datetime import datetime

# ============================================================
# Import from project files
# Searches: ./src, ../src, /mnt/project, PYTHONPATH
# ============================================================
_import_paths = [
    os.path.join(os.path.dirname(__file__), '..', 'src'),  # ../src (from experiments/)
    os.path.join(os.path.dirname(__file__), 'src'),         # ./src (from root)
    os.path.dirname(__file__),                               # same directory
    '/mnt/project',                                          # Claude environment
]
for _p in _import_paths:
    _p = os.path.abspath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from data_generation import (
    generate_heterogeneous_data,
    constant_cate, step_cate, nonlinear_cate
)
from causal_forest_cv import (
    CausalForestCVCI, cross_validation_cf
)

RANDOM_SEED = 2024


# ============================================================
# Configuration: EASY TO CHANGE RANGES
# ============================================================

def get_default_config():
    """
    Central configuration -- change ranges here.
    All simulation parameters in one place.
    """
    return {
        # --- Sample sizes ---
        'n_exp': 200,           # Experimental sample size (fixed)
        'd': 5,                 # Covariate dimension
        'sigma': 1.0,          # Noise std

        # --- CVCI parameters ---
        'lambda_bin': 21,       # Number of lambda grid points (0, 0.05, ..., 1.0)
        'k_fold': 5,           # CV folds
        'n_estimators': 200,   # Trees per causal forest (must be divisible by 4)
        'min_samples_leaf': 5, # Min leaf size

        # --- Axis A: Varying epsilon (treatment effect bias) ---
        'epsilon_vals': np.linspace(0, 0.6, 13),  # 0, 0.05, ..., 0.6
        'epsilon_n_obs': 1000,                      # Fixed n_obs when varying epsilon
        'epsilon_confounding': 0.0,                 # No confounding when varying epsilon

        # --- Axis B: Varying n_obs ---
        'nobs_vals': np.array([100, 200, 500, 1000, 2000, 5000]),
        'nobs_epsilon': 0.1,                        # Fixed moderate bias when varying n_obs
        'nobs_confounding': 0.0,                    # No confounding when varying n_obs

        # --- Axis C: Varying confounding strength ---
        'confounding_vals': np.linspace(0, 2.0, 9),  # 0, 0.25, ..., 2.0
        'confounding_n_obs': 1000,                     # Fixed n_obs when varying confounding
        'confounding_epsilon': 0.0,                    # epsilon=0 to isolate confounding effect

        # --- Simulation repetitions ---
        'n_sims': 50,          # Number of Monte Carlo repetitions

        # --- CATE functions to test ---
        'cate_functions': {
            'constant': constant_cate(tau0=1.0),
            'step': step_cate(tau_low=0.5, tau_high=2.0),
            'nonlinear': nonlinear_cate(tau0=1.0),
        },
    }


def get_quick_config():
    """Quick test configuration."""
    cfg = get_default_config()
    cfg['n_sims'] = 10
    cfg['lambda_bin'] = 11
    cfg['n_estimators'] = 100  # Must be divisible by 4
    cfg['epsilon_vals'] = np.linspace(0, 0.6, 7)
    cfg['nobs_vals'] = np.array([200, 500, 1000, 2000])
    cfg['confounding_vals'] = np.linspace(0, 2.0, 5)
    return cfg


def get_ultra_quick_config():
    """Ultra-fast test configuration."""
    cfg = get_default_config()
    cfg['n_sims'] = 3
    cfg['lambda_bin'] = 6
    cfg['n_estimators'] = 100
    cfg['k_fold'] = 3
    cfg['epsilon_vals'] = np.linspace(0, 0.6, 4)
    cfg['nobs_vals'] = np.array([200, 1000])
    cfg['confounding_vals'] = np.array([0, 1.0, 2.0])
    return cfg


def get_prototype_config():
    """
    Prototype mode: absolute minimum to verify the full pipeline.
    Runs 1 sim with 2 parameter values and tiny forests.
    Should complete in ~30-60 seconds per axis per CATE function.
    Use this to catch bugs before committing to long runs.
    """
    cfg = get_default_config()
    cfg['n_sims'] = 1
    cfg['lambda_bin'] = 5           # lambda in {0, 0.25, 0.5, 0.75, 1.0}
    cfg['n_estimators'] = 40        # Divisible by 4, very small forest
    cfg['min_samples_leaf'] = 10    # Larger leaves = faster
    cfg['k_fold'] = 2               # Minimum CV folds
    cfg['n_exp'] = 100              # Smaller experiment
    cfg['epsilon_vals'] = np.array([0.0, 0.3, 0.6])        # 3 values
    cfg['epsilon_n_obs'] = 300
    cfg['nobs_vals'] = np.array([200, 1000])                # 2 values
    cfg['nobs_epsilon'] = 0.1
    cfg['confounding_vals'] = np.array([0.0, 1.0])          # 2 values
    cfg['confounding_n_obs'] = 300
    return cfg


# ============================================================
# JSON save/load utilities
# ============================================================

def convert_numpy(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(key): convert_numpy(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


def save_json(data, filepath):
    """Save data to JSON with numpy conversion."""
    with open(filepath, 'w') as f:
        json.dump(convert_numpy(data), f, indent=2)
    print(f"  > Saved: {filepath}")


def load_json(filepath):
    """Load JSON results."""
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================================
# Core simulation: run one (data, methods) comparison
# ============================================================

def run_single_comparison(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
                          true_cate_exp, tau_func, cfg, sim_seed):
    """
    Run all four methods on one dataset and return results.

    Methods:
      1. Exp-only CF (lambda=0)
      2. Obs-only CF (lambda=1)
      3. Pooled CF (uniform weights on concatenated data)
      4. CVCI-CF (lambda* via cross-validation)

    Returns dict with CATE predictions, MSEs, ATE estimates, and lambda*.
    Saves full tau_hat vectors for post-hoc metrics.
    """
    true_ate = tau_func.true_ate
    n_est = cfg['n_estimators']
    min_leaf = cfg['min_samples_leaf']
    lambda_vals = np.linspace(0, 1, cfg['lambda_bin'])

    results = {}

    # ----------------------------------------------------------
    # 1. Exp-only CF (lambda=0)
    # ----------------------------------------------------------
    try:
        model_exp = CausalForestCVCI(
            n_estimators=n_est, min_samples_leaf=min_leaf,
            random_state=sim_seed
        )
        model_exp.fit(X_exp, A_exp, Y_exp,
                      X_exp[:0], A_exp[:0], Y_exp[:0],  # Empty obs
                      lambda_=0.0)
        cate_exp = model_exp.predict_cate(X_exp)
        ate_exp = float(np.mean(cate_exp))

        results['exp_only'] = {
            'cate_pred': cate_exp.tolist(),
            'cate_mse': float(np.mean((cate_exp - true_cate_exp) ** 2)),
            'ate_est': ate_exp,
            'ate_mse': float((ate_exp - true_ate) ** 2),
        }
    except Exception as e:
        print(f"    [WARN] Exp-only failed: {e}")
        results['exp_only'] = {'cate_mse': float('nan'), 'ate_mse': float('nan'),
                               'ate_est': float('nan'), 'cate_pred': []}

    # ----------------------------------------------------------
    # 2. Obs-only CF (lambda=1)
    # ----------------------------------------------------------
    try:
        model_obs = CausalForestCVCI(
            n_estimators=n_est, min_samples_leaf=min_leaf,
            random_state=sim_seed
        )
        model_obs.fit(X_obs[:0], A_obs[:0], Y_obs[:0],  # Empty exp
                      X_obs, A_obs, Y_obs,
                      lambda_=1.0)
        cate_obs = model_obs.predict_cate(X_exp)  # Evaluate on exp data
        ate_obs = float(np.mean(cate_obs))

        results['obs_only'] = {
            'cate_pred': cate_obs.tolist(),
            'cate_mse': float(np.mean((cate_obs - true_cate_exp) ** 2)),
            'ate_est': ate_obs,
            'ate_mse': float((ate_obs - true_ate) ** 2),
        }
    except Exception as e:
        print(f"    [WARN] Obs-only failed: {e}")
        results['obs_only'] = {'cate_mse': float('nan'), 'ate_mse': float('nan'),
                               'ate_est': float('nan'), 'cate_pred': []}

    # ----------------------------------------------------------
    # 3. Pooled CF (naive concatenation, uniform weights)
    # ----------------------------------------------------------
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestRegressor

        X_all = np.vstack([X_exp, X_obs])
        A_all = np.concatenate([A_exp, A_obs])
        Y_all = np.concatenate([Y_exp, Y_obs])

        cf_pool = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=n_est, min_samples_leaf=max(min_leaf, 5),
                random_state=sim_seed, n_jobs=-1
            ),
            model_t=RandomForestRegressor(
                n_estimators=n_est, min_samples_leaf=max(min_leaf, 5),
                random_state=sim_seed, n_jobs=-1
            ),
            n_estimators=n_est,
            min_samples_leaf=min_leaf,
            random_state=sim_seed,
            cv=3,
        )
        cf_pool.fit(Y_all, A_all, X=X_all)
        cate_pool = cf_pool.effect(X_exp).flatten()
        ate_pool = float(np.mean(cate_pool))

        results['pooled'] = {
            'cate_pred': cate_pool.tolist(),
            'cate_mse': float(np.mean((cate_pool - true_cate_exp) ** 2)),
            'ate_est': ate_pool,
            'ate_mse': float((ate_pool - true_ate) ** 2),
        }
    except Exception as e:
        print(f"    [WARN] Pooled failed: {e}")
        results['pooled'] = {'cate_mse': float('nan'), 'ate_mse': float('nan'),
                             'ate_est': float('nan'), 'cate_pred': []}

    # ----------------------------------------------------------
    # 4. CVCI-CF (optimal lambda via cross-validation)
    # ----------------------------------------------------------
    try:
        Q_values, lambda_opt, model_cvci = cross_validation_cf(
            X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
            lambda_vals=lambda_vals,
            k_fold=cfg['k_fold'],
            exp_loss_method='difference',
            stratified=True,
            random_state=sim_seed,
            n_estimators=n_est,
            min_samples_leaf=min_leaf,
            verbose=False
        )
        cate_cvci = model_cvci.predict_cate(X_exp)
        ate_cvci = float(np.mean(cate_cvci))

        results['cvci'] = {
            'cate_pred': cate_cvci.tolist(),
            'cate_mse': float(np.mean((cate_cvci - true_cate_exp) ** 2)),
            'ate_est': ate_cvci,
            'ate_mse': float((ate_cvci - true_ate) ** 2),
            'lambda_opt': float(lambda_opt),
            'Q_values': Q_values.tolist(),
        }
    except Exception as e:
        print(f"    [WARN] CVCI failed: {e}")
        results['cvci'] = {'cate_mse': float('nan'), 'ate_mse': float('nan'),
                           'ate_est': float('nan'), 'lambda_opt': float('nan'),
                           'cate_pred': [], 'Q_values': []}

    # ----------------------------------------------------------
    # Store ground truth for this run
    # ----------------------------------------------------------
    results['true_cate'] = true_cate_exp.tolist()
    results['true_ate'] = float(true_ate)

    return results


# ============================================================
# Data generation helper (inline, matches data_generation.py DGP)
# ============================================================

def _generate_obs_data(n_obs, d, theta, tau_func, epsilon, confounding, sigma, rng):
    """Generate observational data with specified bias and confounding."""
    X_obs = rng.normal(0, 1, size=(n_obs, d))

    if confounding > 0:
        gamma = rng.normal(0, 1, size=d) * confounding
        logit = X_obs @ gamma
        propensity = 1.0 / (1.0 + np.exp(-logit))
    else:
        propensity = np.full(n_obs, 0.2)

    A_obs = rng.binomial(1, propensity)
    true_cate_obs = tau_func(X_obs)
    Y_obs = (X_obs @ theta
             + A_obs * (true_cate_obs + epsilon)
             + rng.normal(0, sigma, size=n_obs))

    return X_obs, A_obs, Y_obs


def _generate_exp_data(n_exp, d, theta, tau_func, sigma, rng):
    """Generate experimental data (randomized treatment)."""
    X_exp = rng.normal(0, 1, size=(n_exp, d))
    A_exp = rng.binomial(1, 0.5, size=n_exp)
    true_cate_exp = tau_func(X_exp)
    Y_exp = (X_exp @ theta
             + A_exp * true_cate_exp
             + rng.normal(0, sigma, size=n_exp))
    return X_exp, A_exp, Y_exp, true_cate_exp


# ============================================================
# Axis A: Varying epsilon (treatment effect bias)
# ============================================================

def simulate_varying_epsilon(cfg, cate_name, tau_func, save_dir):
    """
    Vary epsilon with fixed n_obs and confounding_strength.
    Experimental data is held constant across epsilon values within each sim.
    """
    n_exp = cfg['n_exp']
    n_obs = cfg['epsilon_n_obs']
    d = cfg['d']
    sigma = cfg['sigma']
    confounding = cfg['epsilon_confounding']
    epsilon_vals = cfg['epsilon_vals']
    n_sims = cfg['n_sims']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    axis_dir = os.path.join(save_dir, f'epsilon_{cate_name}_{timestamp}')
    os.makedirs(axis_dir, exist_ok=True)

    # Save metadata
    metadata = {
        'experiment': 'varying_epsilon',
        'cate_function': cate_name,
        'n_exp': n_exp, 'n_obs': n_obs, 'd': d, 'sigma': sigma,
        'confounding_strength': confounding,
        'epsilon_vals': epsilon_vals.tolist(),
        'n_sims': n_sims,
        'lambda_bin': cfg['lambda_bin'],
        'n_estimators': cfg['n_estimators'],
        'timestamp': timestamp,
    }
    save_json(metadata, os.path.join(axis_dir, 'metadata.json'))

    all_results = {str(eps): [] for eps in epsilon_vals}

    print(f"\n{'='*60}")
    print(f"AXIS A: Varying epsilon | CATE: {cate_name}")
    print(f"  n_exp={n_exp}, n_obs={n_obs}, confounding={confounding}")
    print(f"  epsilon range: {epsilon_vals.min():.2f} -> {epsilon_vals.max():.2f} "
          f"({len(epsilon_vals)} values)")
    print(f"  {n_sims} simulations")
    print(f"{'='*60}")

    for sim in range(n_sims):
        sim_start = time.time()
        sim_seed = RANDOM_SEED + sim

        # Generate theta + experimental data ONCE per simulation
        rng_exp = np.random.default_rng(sim_seed)
        theta = rng_exp.normal(0, 1, size=d)
        X_exp, A_exp, Y_exp, true_cate_exp = _generate_exp_data(
            n_exp, d, theta, tau_func, sigma, rng_exp)

        for eps_idx, eps in enumerate(epsilon_vals):
            obs_seed = RANDOM_SEED + sim * 1000 + int(eps * 10000)
            rng_obs = np.random.default_rng(obs_seed)
            X_obs, A_obs, Y_obs = _generate_obs_data(
                n_obs, d, theta, tau_func, eps, confounding, sigma, rng_obs)

            result = run_single_comparison(
                X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
                true_cate_exp, tau_func, cfg, sim_seed
            )
            result['epsilon'] = float(eps)
            result['sim_idx'] = sim
            all_results[str(eps)].append(result)

        elapsed = time.time() - sim_start
        lam_0 = all_results[str(epsilon_vals[0])][-1]['cvci'].get('lambda_opt', float('nan'))
        lam_hi = all_results[str(epsilon_vals[-1])][-1]['cvci'].get('lambda_opt', float('nan'))
        print(f"  Sim {sim+1}/{n_sims} ({elapsed:.0f}s) | "
              f"lam*(eps=0)={lam_0:.2f}, lam*(eps={epsilon_vals[-1]:.1f})={lam_hi:.2f}")

        # Incremental save
        if (sim + 1) % max(1, n_sims // 5) == 0 or sim == n_sims - 1:
            save_json(all_results, os.path.join(axis_dir, 'results_all.json'))

    save_json(all_results, os.path.join(axis_dir, 'results_all.json'))
    _save_summary(all_results, epsilon_vals, 'epsilon', axis_dir)
    return axis_dir


# ============================================================
# Axis B: Varying n_obs
# ============================================================

def simulate_varying_nobs(cfg, cate_name, tau_func, save_dir):
    """
    Vary n_obs with fixed epsilon and confounding_strength.
    Experimental data + theta held constant across n_obs values.
    """
    n_exp = cfg['n_exp']
    epsilon = cfg['nobs_epsilon']
    d = cfg['d']
    sigma = cfg['sigma']
    confounding = cfg['nobs_confounding']
    nobs_vals = cfg['nobs_vals']
    n_sims = cfg['n_sims']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    axis_dir = os.path.join(save_dir, f'nobs_{cate_name}_{timestamp}')
    os.makedirs(axis_dir, exist_ok=True)

    metadata = {
        'experiment': 'varying_nobs',
        'cate_function': cate_name,
        'n_exp': n_exp, 'epsilon': epsilon, 'd': d, 'sigma': sigma,
        'confounding_strength': confounding,
        'nobs_vals': nobs_vals.tolist(),
        'n_sims': n_sims,
        'lambda_bin': cfg['lambda_bin'],
        'n_estimators': cfg['n_estimators'],
        'timestamp': timestamp,
    }
    save_json(metadata, os.path.join(axis_dir, 'metadata.json'))

    all_results = {str(int(n)): [] for n in nobs_vals}

    print(f"\n{'='*60}")
    print(f"AXIS B: Varying n_obs | CATE: {cate_name}")
    print(f"  n_exp={n_exp}, epsilon={epsilon}, confounding={confounding}")
    print(f"  n_obs range: {nobs_vals.min()} -> {nobs_vals.max()} "
          f"({len(nobs_vals)} values)")
    print(f"  {n_sims} simulations")
    print(f"{'='*60}")

    for sim in range(n_sims):
        sim_start = time.time()
        sim_seed = RANDOM_SEED + sim

        rng_exp = np.random.default_rng(sim_seed)
        theta = rng_exp.normal(0, 1, size=d)
        X_exp, A_exp, Y_exp, true_cate_exp = _generate_exp_data(
            n_exp, d, theta, tau_func, sigma, rng_exp)

        for nobs_idx, n_obs in enumerate(nobs_vals):
            n_obs = int(n_obs)
            obs_seed = RANDOM_SEED + sim * 1000 + n_obs
            rng_obs = np.random.default_rng(obs_seed)
            X_obs, A_obs, Y_obs = _generate_obs_data(
                n_obs, d, theta, tau_func, epsilon, confounding, sigma, rng_obs)

            result = run_single_comparison(
                X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
                true_cate_exp, tau_func, cfg, sim_seed
            )
            result['n_obs'] = n_obs
            result['sim_idx'] = sim
            all_results[str(n_obs)].append(result)

        elapsed = time.time() - sim_start
        lambdas = [all_results[str(int(n))][-1]['cvci'].get('lambda_opt', float('nan'))
                    for n in nobs_vals]
        valid_lam = [l for l in lambdas if l == l]  # filter nan
        if valid_lam:
            print(f"  Sim {sim+1}/{n_sims} ({elapsed:.0f}s) | "
                  f"lam* range: {min(valid_lam):.2f}-{max(valid_lam):.2f}")
        else:
            print(f"  Sim {sim+1}/{n_sims} ({elapsed:.0f}s)")

        if (sim + 1) % max(1, n_sims // 5) == 0 or sim == n_sims - 1:
            save_json(all_results, os.path.join(axis_dir, 'results_all.json'))

    save_json(all_results, os.path.join(axis_dir, 'results_all.json'))
    _save_summary(all_results, nobs_vals, 'n_obs', axis_dir)
    return axis_dir


# ============================================================
# Axis C: Varying confounding strength
# ============================================================

def simulate_varying_confounding(cfg, cate_name, tau_func, save_dir):
    """
    Vary confounding_strength with fixed n_obs and epsilon.
    epsilon defaults to 0 to isolate confounding effect.
    """
    n_exp = cfg['n_exp']
    n_obs = cfg['confounding_n_obs']
    epsilon = cfg['confounding_epsilon']
    d = cfg['d']
    sigma = cfg['sigma']
    confounding_vals = cfg['confounding_vals']
    n_sims = cfg['n_sims']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    axis_dir = os.path.join(save_dir, f'confounding_{cate_name}_{timestamp}')
    os.makedirs(axis_dir, exist_ok=True)

    metadata = {
        'experiment': 'varying_confounding',
        'cate_function': cate_name,
        'n_exp': n_exp, 'n_obs': n_obs, 'epsilon': epsilon,
        'd': d, 'sigma': sigma,
        'confounding_vals': confounding_vals.tolist(),
        'n_sims': n_sims,
        'lambda_bin': cfg['lambda_bin'],
        'n_estimators': cfg['n_estimators'],
        'timestamp': timestamp,
    }
    save_json(metadata, os.path.join(axis_dir, 'metadata.json'))

    all_results = {str(c): [] for c in confounding_vals}

    print(f"\n{'='*60}")
    print(f"AXIS C: Varying confounding | CATE: {cate_name}")
    print(f"  n_exp={n_exp}, n_obs={n_obs}, epsilon={epsilon}")
    print(f"  confounding range: {confounding_vals.min():.2f} -> "
          f"{confounding_vals.max():.2f} ({len(confounding_vals)} values)")
    print(f"  {n_sims} simulations")
    print(f"{'='*60}")

    for sim in range(n_sims):
        sim_start = time.time()
        sim_seed = RANDOM_SEED + sim

        rng_exp = np.random.default_rng(sim_seed)
        theta = rng_exp.normal(0, 1, size=d)
        X_exp, A_exp, Y_exp, true_cate_exp = _generate_exp_data(
            n_exp, d, theta, tau_func, sigma, rng_exp)

        for c_idx, conf in enumerate(confounding_vals):
            obs_seed = RANDOM_SEED + sim * 1000 + int(conf * 10000)
            rng_obs = np.random.default_rng(obs_seed)
            X_obs, A_obs, Y_obs = _generate_obs_data(
                n_obs, d, theta, tau_func, epsilon, conf, sigma, rng_obs)

            result = run_single_comparison(
                X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
                true_cate_exp, tau_func, cfg, sim_seed
            )
            result['confounding_strength'] = float(conf)
            result['sim_idx'] = sim
            all_results[str(conf)].append(result)

        elapsed = time.time() - sim_start
        lambdas = [all_results[str(c)][-1]['cvci'].get('lambda_opt', float('nan'))
                    for c in confounding_vals]
        valid_lam = [l for l in lambdas if l == l]
        if valid_lam:
            print(f"  Sim {sim+1}/{n_sims} ({elapsed:.0f}s) | "
                  f"lam* range: {min(valid_lam):.2f}-{max(valid_lam):.2f}")
        else:
            print(f"  Sim {sim+1}/{n_sims} ({elapsed:.0f}s)")

        if (sim + 1) % max(1, n_sims // 5) == 0 or sim == n_sims - 1:
            save_json(all_results, os.path.join(axis_dir, 'results_all.json'))

    save_json(all_results, os.path.join(axis_dir, 'results_all.json'))
    _save_summary(all_results, confounding_vals, 'confounding', axis_dir)
    return axis_dir


# ============================================================
# Summary statistics
# ============================================================

def _save_summary(all_results, param_vals, param_name, axis_dir):
    """Compute and save summary statistics (mean/se of MSEs across sims)."""
    summary = []
    methods = ['exp_only', 'obs_only', 'pooled', 'cvci']

    for pv in param_vals:
        key = str(pv) if not isinstance(pv, (int, np.integer)) else str(int(pv))
        sims = all_results.get(key, [])
        if not sims:
            continue

        row = {param_name: float(pv), 'n_sims': len(sims)}

        for method in methods:
            cate_mses = []
            ate_mses = []
            for s in sims:
                cm = s.get(method, {}).get('cate_mse', float('nan'))
                am = s.get(method, {}).get('ate_mse', float('nan'))
                if cm == cm:  # not nan
                    cate_mses.append(cm)
                if am == am:
                    ate_mses.append(am)

            if cate_mses:
                row[f'{method}_cate_mse_mean'] = float(np.mean(cate_mses))
                row[f'{method}_cate_mse_se'] = float(
                    np.std(cate_mses) / np.sqrt(len(cate_mses)))
            if ate_mses:
                row[f'{method}_ate_mse_mean'] = float(np.mean(ate_mses))
                row[f'{method}_ate_mse_se'] = float(
                    np.std(ate_mses) / np.sqrt(len(ate_mses)))

        # lambda* statistics for CVCI
        lambdas = [s.get('cvci', {}).get('lambda_opt', float('nan')) for s in sims]
        lambdas = [l for l in lambdas if l == l]
        if lambdas:
            row['lambda_opt_mean'] = float(np.mean(lambdas))
            row['lambda_opt_se'] = float(
                np.std(lambdas) / np.sqrt(len(lambdas)))

        summary.append(row)

    save_json(summary, os.path.join(axis_dir, 'summary.json'))

    # Print table
    print(f"\n--- Summary: {param_name} ---")
    print(f"{'Param':>10} | {'Exp CATE':>10} | {'Obs CATE':>10} | "
          f"{'Pool CATE':>10} | {'CVCI CATE':>10} | {'lam*':>6}")
    print("-" * 72)
    for row in summary:
        pv = row[param_name]
        exp_c = row.get('exp_only_cate_mse_mean', float('nan'))
        obs_c = row.get('obs_only_cate_mse_mean', float('nan'))
        pool_c = row.get('pooled_cate_mse_mean', float('nan'))
        cvci_c = row.get('cvci_cate_mse_mean', float('nan'))
        lam = row.get('lambda_opt_mean', float('nan'))
        print(f"{pv:>10.3f} | {exp_c:>10.4f} | {obs_c:>10.4f} | "
              f"{pool_c:>10.4f} | {cvci_c:>10.4f} | {lam:>6.3f}")


# ============================================================
# Plotting
# ============================================================

def plot_results(axis_dir, show=True):
    """
    Generate publication-ready plots from saved results.
    Creates CATE MSE, ATE MSE, and optimal lambda panels.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    meta = load_json(os.path.join(axis_dir, 'metadata.json'))
    summary = load_json(os.path.join(axis_dir, 'summary.json'))

    experiment = meta['experiment']
    cate_name = meta['cate_function']

    if experiment == 'varying_epsilon':
        x_key, x_label = 'epsilon', 'Treatment Effect Bias (epsilon)'
    elif experiment == 'varying_nobs':
        x_key, x_label = 'n_obs', 'Observational Sample Size (N_obs)'
    elif experiment == 'varying_confounding':
        x_key, x_label = 'confounding', 'Confounding Strength'
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    x_vals = [row[x_key] for row in summary]
    methods = [
        ('exp_only', 'Exp-only CF', '#2ca02c', '--', 'o'),
        ('obs_only', 'Obs-only CF', '#d62728', '--', 's'),
        ('pooled', 'Pooled CF', '#9467bd', ':', '^'),
        ('cvci', 'CVCI-CF', '#1f77b4', '-', 'D'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: CATE MSE
    ax = axes[0]
    for method, label, color, ls, marker in methods:
        y = [row.get(f'{method}_cate_mse_mean', float('nan')) for row in summary]
        se = [row.get(f'{method}_cate_mse_se', 0) for row in summary]
        ax.errorbar(x_vals, y, yerr=se, label=label,
                    color=color, linestyle=ls, marker=marker,
                    markersize=5, capsize=3, linewidth=1.5)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('CATE MSE', fontsize=12)
    ax.set_title(f'CATE MSE - {cate_name}', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: ATE MSE
    ax = axes[1]
    for method, label, color, ls, marker in methods:
        y = [row.get(f'{method}_ate_mse_mean', float('nan')) for row in summary]
        se = [row.get(f'{method}_ate_mse_se', 0) for row in summary]
        ax.errorbar(x_vals, y, yerr=se, label=label,
                    color=color, linestyle=ls, marker=marker,
                    markersize=5, capsize=3, linewidth=1.5)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('ATE MSE', fontsize=12)
    ax.set_title(f'ATE MSE - {cate_name}', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: Optimal lambda
    ax = axes[2]
    lam_mean = [row.get('lambda_opt_mean', float('nan')) for row in summary]
    lam_se = [row.get('lambda_opt_se', 0) for row in summary]
    ax.errorbar(x_vals, lam_mean, yerr=lam_se,
                color='#1f77b4', marker='D', markersize=5,
                capsize=3, linewidth=1.5)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Optimal lambda*', fontsize=12)
    ax.set_title(f'Mixing Parameter - {cate_name}', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(axis_dir, 'results_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  > Plot saved: {plot_path}")

    if show:
        plt.show()
    plt.close()


# ============================================================
# Main entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='CVCI-CF Simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cf_simulations.py --ultra-quick              # Fast test, all axes
  python cf_simulations.py --quick --axis epsilon      # Quick, only vary epsilon
  python cf_simulations.py --cate step --axis nobs     # Full, step CATE, vary n_obs
  python cf_simulations.py --mode plot --results-dir simulation_results_cf/epsilon_step_...
        """)

    parser.add_argument('--quick', action='store_true',
                        help='Quick test (~30 min)')
    parser.add_argument('--ultra-quick', action='store_true',
                        help='Ultra fast test (~5 min)')
    parser.add_argument('--prototype', action='store_true',
                        help='Prototype mode: 1 sim, tiny forests, ~30-60s per experiment. '
                             'Use to verify pipeline before long runs.')
    parser.add_argument('--axis', choices=['epsilon', 'nobs', 'confounding', 'all'],
                        default='all', help='Which axis to simulate')
    parser.add_argument('--cate', choices=['constant', 'step', 'nonlinear', 'all'],
                        default='all', help='Which CATE function to use')
    parser.add_argument('--mode', choices=['run', 'plot'],
                        default='run', help='Run simulations or plot results')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory with results to plot')
    parser.add_argument('--save-dir', type=str,
                        default=None,
                        help='Directory for saving results (default: ../results/cf_simulations '
                             'when run from experiments/, or ./results/cf_simulations from root)')
    parser.add_argument('--n-sims', type=int, default=None,
                        help='Override number of simulations')

    args = parser.parse_args()

    # --- Plot mode ---
    if args.mode == 'plot':
        if args.results_dir is None:
            print("ERROR: --results-dir required for plot mode")
            sys.exit(1)
        plot_results(args.results_dir)
        return

    # --- Run mode ---
    if args.prototype:
        cfg = get_prototype_config()
        print("PROTOTYPE MODE (1 sim, tiny forests, ~30-60s per experiment)")
    elif args.ultra_quick:
        cfg = get_ultra_quick_config()
        print("ULTRA-QUICK MODE")
    elif args.quick:
        cfg = get_quick_config()
        print("QUICK MODE")
    else:
        cfg = get_default_config()
        print("FULL MODE")

    if args.n_sims is not None:
        cfg['n_sims'] = args.n_sims

    # Resolve save directory
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        # Auto-detect: if running from experiments/, go up to results/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(script_dir) == 'experiments':
            save_dir = os.path.join(script_dir, '..', 'results', 'cf_simulations')
        else:
            save_dir = os.path.join(script_dir, 'results', 'cf_simulations')
        save_dir = os.path.abspath(save_dir)

    # Select CATE functions
    if args.cate == 'all':
        cate_items = list(cfg['cate_functions'].items())
    else:
        cate_items = [(args.cate, cfg['cate_functions'][args.cate])]

    # Select axes
    axes_to_run = []
    if args.axis in ('all', 'epsilon'):
        axes_to_run.append('epsilon')
    if args.axis in ('all', 'nobs'):
        axes_to_run.append('nobs')
    if args.axis in ('all', 'confounding'):
        axes_to_run.append('confounding')

    os.makedirs(save_dir, exist_ok=True)

    # Print plan
    n_experiments = len(axes_to_run) * len(cate_items)
    print(f"\nPlan: {n_experiments} experiments "
          f"({len(axes_to_run)} axes x {len(cate_items)} CATE functions)")
    print(f"  Axes: {axes_to_run}")
    print(f"  CATE: {[name for name, _ in cate_items]}")
    print(f"  Sims per experiment: {cfg['n_sims']}")
    print(f"  n_exp: {cfg['n_exp']}, d: {cfg['d']}")
    print(f"  Results: {save_dir}/\n")

    # Run experiments
    completed_dirs = []
    total_start = time.time()

    for cate_name, tau_func in cate_items:
        for axis in axes_to_run:
            if axis == 'epsilon':
                d = simulate_varying_epsilon(cfg, cate_name, tau_func, save_dir)
            elif axis == 'nobs':
                d = simulate_varying_nobs(cfg, cate_name, tau_func, save_dir)
            elif axis == 'confounding':
                d = simulate_varying_confounding(cfg, cate_name, tau_func, save_dir)
            completed_dirs.append(d)

            # Plot immediately
            try:
                plot_results(d, show=False)
            except Exception as e:
                print(f"  [WARN] Plotting failed: {e}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"\nResults saved to:")
    for d in completed_dirs:
        print(f"  {d}")
    print(f"\nTo re-plot:")
    for d in completed_dirs:
        print(f"  python cf_simulations.py --mode plot --results-dir {d}")


if __name__ == '__main__':
    main()
