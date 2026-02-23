"""
CVCI with Causal Forests for Heterogeneous Treatment Effects

Extends Yang, Lin, Athey, Jordan, and Imbens (2025) CVCI framework to
Athey-Wager Causal Forests (econml.grf.CausalForest), enabling cross-validated
combination of experimental and observational data for CATE estimation.

Key components:
1. CausalForestCVCI: Wraps CausalForest with lambda-weighted sample mixing
2. cross_validation_cf: Selects optimal lambda via outcome MSE on held-out
   experimental data

The causal forest solves the local moment equation:
    E[(Y - theta_1(x)*A - theta_0(x)) * (A; 1) | X=x] = 0

where theta_1(x) = tau(x) is the CATE and theta_0(x) = mu_0(x) is the baseline.
Sample weights from lambda directly affect tree splits and leaf estimates --
no intermediate nuisance estimation or residualization.

Cross-validation selects lambda* by evaluating outcome prediction quality
on held-out experimental data:
    Q(lambda) = mean( (Y_i - Y_hat_i)^2 )  over validation folds
where Y_hat_i = theta_0(X_i) + A_i * theta_1(X_i).
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from econml.grf import CausalForest
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class CausalForestCVCI:
    """
    Causal Forest wrapper for the CVCI framework.

    Uses econml.grf.CausalForest (Athey-Wager style) with sample weights
    to implement the hybrid loss mixing parameter lambda.

    With fit_intercept=True (default), the forest jointly estimates:
        theta_1(x) = tau(x)   (treatment effect / CATE)
        theta_0(x) = mu_0(x)  (baseline outcome under control)

    Sample weights flow directly into tree splitting and leaf estimation
    with no intermediate nuisance models.
    """

    def __init__(self, n_estimators=200, min_samples_leaf=5,
                 max_depth=None, random_state=None, inference=False):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.inference = inference

        self.cf_model = None

    def fit(self, X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_):
        """
        Fit causal forest on combined data with lambda-weights.

        Weight scheme:
            exp unit i gets weight  (1-lambda) * n_total / n_exp
            obs unit j gets weight  lambda     * n_total / n_obs
        Rescaled so sum(weights) = n_total for sklearn compatibility.

        Edge cases:
            lambda < 0.01: fit on experimental data only (no weights)
            lambda > 0.99: fit on observational data only (no weights)
        """
        if lambda_ < 0.01:
            X_train, A_train, Y_train = X_exp, A_exp, Y_exp
            sample_weight = None
        elif lambda_ > 0.99:
            X_train, A_train, Y_train = X_obs, A_obs, Y_obs
            sample_weight = None
        else:
            X_train = np.vstack([X_exp, X_obs])
            A_train = np.concatenate([A_exp, A_obs])
            Y_train = np.concatenate([Y_exp, Y_obs])

            n_exp = len(X_exp)
            n_obs = len(X_obs)
            n_total = n_exp + n_obs

            weights_exp = np.full(n_exp, (1 - lambda_) * n_total / n_exp)
            weights_obs = np.full(n_obs, lambda_ * n_total / n_obs)
            sample_weight = np.concatenate([weights_exp, weights_obs])

        self.cf_model = CausalForest(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            inference=self.inference,
        )
        self.cf_model.fit(X_train, A_train, Y_train, sample_weight=sample_weight)
        return self

    def predict_cate(self, X):
        """Predict CATE tau(x) = theta_1(x)."""
        if self.cf_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.cf_model.predict(X).flatten()

    def predict_full(self, X):
        """
        Predict full parameter vector [theta_1(x), theta_0(x)].

        Returns:
            theta1: CATE tau(x), shape (n,)
            theta0: baseline mu_0(x), shape (n,)
        """
        if self.cf_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        alpha, jac = self.cf_model.predict_alpha_and_jac(X)
        invjac = np.linalg.pinv(jac)
        params = np.einsum('ijk,ik->ij', invjac, alpha)
        theta1 = params[:, 0]  # treatment effect
        theta0 = params[:, 1]  # baseline (intercept)
        return theta1, theta0

    def predict_outcome(self, X, A):
        """
        Predict outcomes: Y_hat = theta_0(X) + A * theta_1(X).

        Used for the CV loss function (outcome MSE on experimental data).
        """
        theta1, theta0 = self.predict_full(X)
        return theta0 + A * theta1

    def predict_ate(self, X):
        """Predict ATE as mean of CATE predictions."""
        return float(np.mean(self.predict_cate(X)))


def cross_validation_cf(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs,
                        lambda_vals, k_fold=5, stratified=True,
                        random_state=None, n_estimators=200,
                        min_samples_leaf=5, verbose=False):
    """
    Cross-validation to select optimal lambda for CVCI-CF.

    For each candidate lambda:
    1. Split experimental data into K folds
    2. For each fold: fit CF on (train_exp + all_obs) with lambda-weights
    3. Evaluate outcome MSE on held-out experimental validation fold:
           loss = mean( (Y_val - theta_0(X_val) - A_val * theta_1(X_val))^2 )
    4. Average across folds -> Q(lambda)

    Select lambda* = argmin Q(lambda).

    The outcome MSE gives n_val pointwise comparisons per fold (vs 1 scalar
    with ATE-difference), providing strong signal for lambda selection.

    Args:
        X_exp, A_exp, Y_exp: Experimental data
        X_obs, A_obs, Y_obs: Observational data
        lambda_vals: Array of candidate lambda values
        k_fold: Number of CV folds
        stratified: Stratify folds by treatment assignment
        random_state: Random seed
        n_estimators: Trees per forest
        min_samples_leaf: Min leaf size
        verbose: Print progress

    Returns:
        Q_values: CV error for each lambda
        lambda_opt: Optimal lambda
        model_opt: Fitted model with lambda* on full data
    """
    if stratified:
        cv = StratifiedKFold(n_splits=k_fold, shuffle=True,
                             random_state=random_state)
    else:
        cv = KFold(n_splits=k_fold, shuffle=True,
                   random_state=random_state)

    Q_values = np.zeros(len(lambda_vals))

    for i, lambda_ in enumerate(lambda_vals):
        if verbose:
            print(f"  lambda = {lambda_:.2f}", end='\r')

        fold_losses = []
        splits = cv.split(X_exp, A_exp) if stratified else cv.split(X_exp)

        for train_idx, val_idx in splits:
            X_train, A_train, Y_train = X_exp[train_idx], A_exp[train_idx], Y_exp[train_idx]
            X_val, A_val, Y_val = X_exp[val_idx], A_exp[val_idx], Y_exp[val_idx]

            try:
                model = CausalForestCVCI(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    inference=False,  # Not needed for CV, saves time
                )
                model.fit(X_train, A_train, Y_train,
                          X_obs, A_obs, Y_obs, lambda_)

                # Outcome MSE on validation fold
                Y_hat_val = model.predict_outcome(X_val, A_val)
                loss = np.mean((Y_val - Y_hat_val) ** 2)
                fold_losses.append(loss)

            except Exception as e:
                if verbose:
                    print(f"    Fold failed: {e}")
                continue

        if fold_losses:
            Q_values[i] = np.mean(fold_losses)
        else:
            Q_values[i] = np.inf

    # Select optimal lambda
    lambda_opt = lambda_vals[np.argmin(Q_values)]

    if verbose:
        print(f"  Optimal lambda = {lambda_opt:.3f}")

    # Fit final model on all data with lambda*
    model_opt = CausalForestCVCI(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        inference=False,
    )
    model_opt.fit(X_exp, A_exp, Y_exp, X_obs, A_obs, Y_obs, lambda_opt)

    return Q_values, lambda_opt, model_opt
