"""
Shared utilities for CVCI-CF project.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime


def save_results_json(results, filename):
    """Save results to JSON with numpy type conversion."""
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(convert(results), f, indent=2)


def load_results_json(filename):
    """Load results from JSON."""
    with open(filename, 'r') as f:
        return json.load(f)


def timestamp():
    """Current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def mse(estimates, truth):
    """Mean squared error."""
    return np.mean((np.asarray(estimates) - truth) ** 2)


def rmse(estimates, truth):
    """Root mean squared error."""
    return np.sqrt(mse(estimates, truth))


def pehe(cate_hat, cate_true):
    """
    Precision in Estimation of Heterogeneous Effects.
    
    PEHE = sqrt(E[(τ̂(x) - τ(x))²])
    
    Standard metric for CATE evaluation.
    """
    return np.sqrt(np.mean((cate_hat - cate_true) ** 2))
