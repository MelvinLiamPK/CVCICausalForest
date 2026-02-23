"""
CVCI with Causal Forests for Heterogeneous Treatment Effects

Extends Yang, Lin, Athey, Jordan, and Imbens (2025) CVCI framework to Causal Forests,
enabling cross-validated combination of experimental and observational data
for Conditional Average Treatment Effect (CATE) estimation.

Key components:
1. CausalForestCVCI: Main class that fits causal forests with hybrid loss
2. cross_validation_cf: Cross-validation to select optimal λ
3. Loss functions: L_exp (experimental) and L_obs (observational)

The hybrid loss is:
    L(θ, λ) = (1-λ) L_exp(θ) + λ L_obs(θ)

where:
    L_exp = MSE of CATE predictions vs experimental CATE estimates
    L_obs = MSE of outcome predictions on observational data
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class CausalForestCVCI:
    """
    Causal Forest model for the CVCI framework.
    
    Estimates CATE τ(x) using causal forests with sample weights
    derived from the hybrid loss mixing parameter λ.
    
    The key methodological innovation: we use sample weights to implement
    the hybrid loss L(θ,λ) = (1-λ)L_exp + λ L_obs when fitting
    the causal forest's nuisance models and final CATE estimator.
    """
    
    def __init__(self, n_estimators=200, min_samples_leaf=5,  # n_estimators must be divisible by 4 (econml subforest_size)
                 max_depth=None, random_state=None):
        """
        Args:
            n_estimators: Number of trees in the causal forest
            min_samples_leaf: Minimum samples per leaf
            max_depth: Maximum tree depth (None = unlimited)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.cf_model = None  # Fitted CausalForestDML
        self.propensity_model = None
        self.outcome_model = None
        
    def fit(self, X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_):
        """
        Fit causal forest using hybrid loss via sample weights.
        
        Strategy:
        1. Combine experimental + observational data with weights from λ
        2. Fit CausalForestDML on the weighted combined dataset
        
        Args:
            X_exp, A_exp, Y_exp: Experimental covariates, treatment, outcome
            X_obs, A_obs, Y_obs: Observational covariates, treatment, outcome
            lambda_: Mixing parameter in [0, 1]
        """
        if lambda_ < 0.01:
            # Experimental only
            X_train, A_train, Y_train = X_exp, A_exp, Y_exp
            sample_weight = None
        elif lambda_ > 0.99:
            # Observational only
            X_train, A_train, Y_train = X_obs, A_obs, Y_obs
            sample_weight = None
        else:
            # Combine with weights
            X_train = np.vstack([X_exp, X_obs])
            A_train = np.concatenate([A_exp, A_obs])
            Y_train = np.concatenate([Y_exp, Y_obs])
            
            n_exp = len(X_exp)
            n_obs = len(X_obs)
            n_total = n_exp + n_obs
            
            # Weights: (1-λ)/n_exp for exp, λ/n_obs for obs
            # Scaled so sum = n_total (sklearn compatibility)
            weights_exp = np.ones(n_exp) * (1 - lambda_) * n_total / n_exp
            weights_obs = np.ones(n_obs) * lambda_ * n_total / n_obs
            sample_weight = np.concatenate([weights_exp, weights_obs])
        
        # Fit CausalForestDML
        # NOTE: CausalForestDML requires regressors for both model_y and model_t
        # (it treats binary treatment as continuous internally for residualization)
        self.cf_model = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=max(self.min_samples_leaf, 5),
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            ),
            model_t=RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=max(self.min_samples_leaf, 5),
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            ),
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            cv=3,  # Internal cross-fitting for nuisance params
        )
        
        self.cf_model.fit(
            Y_train, A_train, X=X_train,
            sample_weight=sample_weight
        )
        
        return self
    
    def predict_cate(self, X):
        """
        Predict CATE τ(x) for given covariates.
        
        Args:
            X: Covariates (n_samples, d)
            
        Returns:
            tau_hat: Estimated CATE for each sample (n_samples,)
        """
        if self.cf_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.cf_model.effect(X).flatten()
    
    def predict_ate(self, X):
        """
        Predict ATE as the average of CATE predictions.
        
        Args:
            X: Covariates to average over
            
        Returns:
            Scalar ATE estimate
        """
        cate = self.predict_cate(X)
        return np.mean(cate)
    
    def predict_cate_interval(self, X, alpha=0.05):
        """
        Predict CATE with confidence intervals.
        
        Args:
            X: Covariates
            alpha: Significance level
            
        Returns:
            (tau_hat, tau_lower, tau_upper)
        """
        if self.cf_model is None:
            raise ValueError("Model not fitted yet.")
        
        inference = self.cf_model.effect_inference(X)
        tau_hat = inference.point_estimate.flatten()
        ci = inference.conf_int(alpha=alpha)
        tau_lower = ci[0].flatten()
        tau_upper = ci[1].flatten()
        
        return tau_hat, tau_lower, tau_upper


def compute_exp_cate(X_exp, A_exp, Y_exp, method='causal_forest'):
    """
    Compute CATE estimates from experimental data only.
    
    Args:
        X_exp, A_exp, Y_exp: Experimental data
        method: 'causal_forest' or 'difference' (ATE only)
        
    Returns:
        If 'causal_forest': CATE estimates τ̂(x) for each x in X_exp
        If 'difference': scalar ATE
    """
    if method == 'difference':
        return Y_exp[A_exp == 1].mean() - Y_exp[A_exp == 0].mean()
    
    elif method == 'causal_forest':
        # Fit causal forest on experimental data only
        cf = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                          random_state=42, n_jobs=-1),
            model_t=RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                          random_state=42, n_jobs=-1),
            n_estimators=100,
            min_samples_leaf=5,
            random_state=42,
            cv=2,
        )
        cf.fit(Y_exp, A_exp, X=X_exp)
        return cf.effect(X_exp).flatten()
    
    else:
        raise ValueError(f"Unknown method: {method}")


def L_exp_cf(tau_hat, X_exp, A_exp, Y_exp, tau_exp_precompute=None,
             method='difference'):
    """
    Experimental loss for CATE.
    
    Two modes:
    - method='difference': L_exp = (ATE_exp - ATE_hat)^2  [scalar comparison]
    - method='causal_forest': L_exp = mean((τ_exp(x) - τ_hat(x))^2)  [pointwise CATE]
    
    Args:
        tau_hat: CATE predictions from our model (array) or scalar ATE
        X_exp, A_exp, Y_exp: Experimental data
        tau_exp_precompute: Pre-computed experimental CATE/ATE
        method: 'difference' or 'causal_forest'
        
    Returns:
        Scalar loss
    """
    if tau_exp_precompute is None:
        tau_exp = compute_exp_cate(X_exp, A_exp, Y_exp, method=method)
    else:
        tau_exp = tau_exp_precompute
    
    if method == 'difference':
        # Compare ATEs
        ate_hat = np.mean(tau_hat) if isinstance(tau_hat, np.ndarray) else tau_hat
        ate_exp = tau_exp if np.isscalar(tau_exp) else np.mean(tau_exp)
        return (ate_exp - ate_hat) ** 2
    
    elif method == 'causal_forest':
        # Pointwise CATE comparison
        return np.mean((tau_exp - tau_hat) ** 2)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def L_obs_cf(model, X_obs, A_obs, Y_obs):
    """
    Observational loss: outcome prediction MSE.
    
    Uses the causal forest's underlying outcome model to predict Y,
    then computes MSE against observed outcomes.
    
    Args:
        model: Fitted CausalForestCVCI
        X_obs, A_obs, Y_obs: Observational data
        
    Returns:
        Scalar MSE loss
    """
    # Get CATE predictions
    tau_hat = model.predict_cate(X_obs)
    
    # For outcome prediction, we need μ(x) + A*τ(x)
    # Since we don't have direct access to μ(x) from CausalForestDML,
    # we approximate using the treatment effect structure:
    # E[Y|X,A=1] - E[Y|X,A=0] = τ(x)
    # So Y_pred ≈ E[Y|X,A=0] + A*τ(x)
    # We estimate E[Y|X,A=0] from the controls
    
    # Simple approach: use RF on controls for baseline, add CATE for treated
    from sklearn.ensemble import RandomForestRegressor
    
    # Combine exp and obs treated/control info
    X_ctrl = X_obs[A_obs == 0]
    Y_ctrl = Y_obs[A_obs == 0]
    
    if len(X_ctrl) > 5:
        mu0_model = RandomForestRegressor(n_estimators=50, min_samples_leaf=5,
                                           random_state=42, n_jobs=-1)
        mu0_model.fit(X_ctrl, Y_ctrl)
        mu0_hat = mu0_model.predict(X_obs)
    else:
        mu0_hat = np.mean(Y_obs[A_obs == 0]) if np.sum(A_obs == 0) > 0 else np.mean(Y_obs)
        mu0_hat = np.full(len(X_obs), mu0_hat)
    
    Y_pred = mu0_hat + A_obs * tau_hat
    
    return np.mean((Y_obs - Y_pred) ** 2)


def cross_validation_cf(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
                        lambda_vals, k_fold=5, exp_loss_method='difference',
                        stratified=True, random_state=None,
                        n_estimators=200, min_samples_leaf=5, verbose=False):
    """
    Cross-validation to select optimal λ for CVCI with Causal Forests.
    
    For each candidate λ:
    1. Split experimental data into K folds
    2. For each fold: fit CF on (train_exp + all_obs) with weights from λ
    3. Evaluate L_exp on validation fold
    4. Average across folds → Q(λ)
    
    Select λ* = argmin Q(λ)
    
    Args:
        X_exp, A_exp, Y_exp: Experimental data
        X_obs, A_obs, Y_obs: Observational data
        lambda_vals: Array of candidate λ values
        k_fold: Number of CV folds
        exp_loss_method: 'difference' (ATE comparison) or 'causal_forest' (CATE)
        stratified: Stratify folds by treatment
        random_state: Random seed
        n_estimators: Trees per forest
        min_samples_leaf: Min leaf size
        verbose: Print progress
        
    Returns:
        Q_values: CV error for each λ
        lambda_opt: Optimal λ
        model_opt: Fitted model with λ*
    """
    # Set up CV
    if stratified:
        cv = StratifiedKFold(n_splits=k_fold, shuffle=True,
                             random_state=random_state)
    else:
        cv = KFold(n_splits=k_fold, shuffle=True,
                   random_state=random_state)
    
    # Pre-compute experimental benchmark
    if exp_loss_method == 'difference':
        tau_exp = Y_exp[A_exp == 1].mean() - Y_exp[A_exp == 0].mean()
    else:
        tau_exp = compute_exp_cate(X_exp, A_exp, Y_exp, method=exp_loss_method)
    
    Q_values = np.zeros(len(lambda_vals))
    
    for i, lambda_ in enumerate(lambda_vals):
        if verbose:
            print(f"  λ = {lambda_:.2f}", end='\r')
        
        fold_losses = []
        
        splits = cv.split(X_exp, A_exp) if stratified else cv.split(X_exp)
        
        for train_idx, val_idx in splits:
            X_train = X_exp[train_idx]
            A_train = A_exp[train_idx]
            Y_train = Y_exp[train_idx]
            
            X_val = X_exp[val_idx]
            A_val = A_exp[val_idx]
            Y_val = Y_exp[val_idx]
            
            try:
                model = CausalForestCVCI(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
                model.fit(X_train, A_train, Y_train,
                         X_obs, A_obs, Y_obs, lambda_)
                
                # Predict CATE on validation set
                tau_hat_val = model.predict_cate(X_val)
                
                # Compute experimental loss
                loss = L_exp_cf(tau_hat_val, X_val, A_val, Y_val,
                               tau_exp_precompute=tau_exp,
                               method=exp_loss_method)
                
                fold_losses.append(loss)
                
            except Exception as e:
                if verbose:
                    print(f"    Fold failed: {e}")
                continue
        
        if fold_losses:
            Q_values[i] = np.mean(fold_losses)
        else:
            Q_values[i] = np.inf
    
    # Select optimal λ
    lambda_opt = lambda_vals[np.argmin(Q_values)]
    
    if verbose:
        print(f"  Optimal λ = {lambda_opt:.3f}")
    
    # Fit final model on all data
    model_opt = CausalForestCVCI(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model_opt.fit(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_opt)
    
    return Q_values, lambda_opt, model_opt
