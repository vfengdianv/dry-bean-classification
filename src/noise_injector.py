"""
Noise injection module with unified interface.

Three noise types, each at configurable levels:
  1. gaussian      — additive Gaussian noise on features (η ∈ {0.05, 0.15, 0.30})
  2. label_flip    — randomly reassign a fraction of labels (η ∈ {0.10, 0.20, 0.30})
  3. feature_missing — randomly mask features, mean-impute afterward (η ∈ {0.10, 0.20, 0.30})

Design principles:
  - Noise is injected ONLY into the training set.
  - Test/validation sets remain clean at all times.
  - A local np.random.RandomState is used to isolate randomness.
  - The caller is responsible for re-fitting a StandardScaler on the noisy training
    data (see train.py for the independent-scaler pipeline).
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def inject_noise(X_train, y_train, noise_type, level, random_seed=42):
    """
    Inject noise of a given type and intensity into the training set.

    Parameters
    ----------
    X_train : pd.DataFrame  (n_samples, n_features)
        Cleaned (but not yet scaled) training features.
    y_train : pd.Series  (n_samples,)
        Cleaned training labels.
    noise_type : str
        One of {'gaussian', 'label_flip', 'feature_missing'}.
    level : float
        Noise intensity level.
        - gaussian:       η ∈ {0.05, 0.15, 0.30} — multiplier on per-feature std.
        - label_flip:     η ∈ {0.10, 0.20, 0.30} — fraction of labels to flip.
        - feature_missing: η ∈ {0.10, 0.20, 0.30} — fraction of feature values to mask.
    random_seed : int
        Seed for the local random state (isolated from global state).

    Returns
    -------
    X_noisy : pd.DataFrame
        Noisy training features (same shape as input).
    y_noisy : pd.Series
        Noisy training labels (same shape as input).
    """
    rng = np.random.RandomState(random_seed)

    if noise_type == 'gaussian':
        X_noisy = _inject_gaussian_noise(X_train.copy(), level, rng)
        y_noisy = y_train.copy()

    elif noise_type == 'label_flip':
        X_noisy = X_train.copy()
        y_noisy = _inject_label_flip(y_train.copy(), level, rng)

    elif noise_type == 'feature_missing':
        X_noisy = _inject_feature_missing(X_train.copy(), level, rng)
        y_noisy = y_train.copy()

    else:
        raise ValueError(f"Unknown noise_type: '{noise_type}'. "
                         f"Use 'gaussian', 'label_flip', or 'feature_missing'.")

    print(f"[noise_injector] Applied {noise_type} noise (level={level}).")
    return X_noisy, y_noisy


# ── Private helpers ────────────────────────────────────────────────────────

def _inject_gaussian_noise(X, level, rng):
    """
    Add Gaussian noise: X_noisy = X + level * σ_j * N(0, 1)
    where σ_j is the per-feature standard deviation computed from X.
    """
    for col in X.columns:
        std = X[col].std()
        noise = rng.normal(0, std * level, size=len(X))
        X[col] = X[col] + noise
    return X


def _inject_label_flip(y, level, rng):
    """
    Randomly flip a fraction 'level' of labels to a different class.
    The new label is chosen uniformly from all OTHER classes.
    """
    n = len(y)
    n_flip = int(n * level)
    flip_indices = rng.choice(n, size=n_flip, replace=False)

    classes = np.unique(y)
    for idx in flip_indices:
        old_label = y.iloc[idx]
        other_classes = [c for c in classes if c != old_label]
        y.iloc[idx] = rng.choice(other_classes)

    return y


def _inject_feature_missing(X, level, rng):
    """
    Randomly mask a fraction 'level' of ALL feature values across the entire
    DataFrame, then impute the masked positions with the column mean (computed
    from the partially-masked training set).

    This simulates the realistic scenario where missing values are observed,
    and we impute with column statistics.
    """
    n_cells = X.size  # n_samples * n_features
    n_mask = int(n_cells * level)

    # Flatten the mask
    flat_indices = rng.choice(n_cells, size=n_mask, replace=False)

    # Convert to (row, col) pairs
    row_indices = flat_indices // X.shape[1]
    col_indices = flat_indices % X.shape[1]

    # Set the selected positions to NaN
    for row, col in zip(row_indices, col_indices):
        X.iat[row, col] = np.nan

    # Mean imputation using the (partially masked) training data
    imputer = SimpleImputer(strategy='mean')
    X_filled = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    return X_filled
