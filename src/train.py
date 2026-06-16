"""
Training orchestration module.

Responsibilities:
  1. Run baseline experiments (no noise) — fit scaler on clean train, train all models.
  2. Run noise experiments — inject noise → fit NEW scaler on noisy train → train.
  3. Collect all results and return them as a structured dict.

This is where the "independent scaler pipeline" for noise experiments lives:
  - Noise changes the distribution of X_train.
  - We MUST fit a fresh StandardScaler on X_train_noisy, not reuse the clean scaler.
  - The test set is always transformed with the scaler fitted on (noisy) train.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .models.knn_model import train_knn
from .models.svm_sgd_model import train_svm_sgd
from .models.mlp_model import train_mlp
from .models.xgb_model import train_xgb
from .noise_injector import inject_noise


def run_baseline_experiment(X_train_clean, X_test_clean, y_train_clean, y_test_clean,
                            X_val_clean=None, y_val_clean=None,
                            algorithms=None, config=None):
    """
    Run the baseline (no-noise) experiment.

    Pipeline:
      X_train_clean → StandardScaler.fit(X_train_clean) → transform(train, test, val)
      → train each algorithm → collect results.

    Parameters
    ----------
    X_train_clean, X_test_clean : pd.DataFrame
        Numerically cleaned but UNSCALED features.
    y_train_clean, y_test_clean : pd.Series
    X_val_clean, y_val_clean : pd.DataFrame/Series or None
    algorithms : list of str
        e.g. ['knn', 'svm', 'mlp', 'xgb']
    config : dict
        Full experiment config.

    Returns
    -------
    results : dict  {algo_name: result_dict}
    """
    if algorithms is None:
        algorithms = ['knn', 'svm', 'mlp', 'xgb']

    # Step 1: Fit scaler on CLEAN training set
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_clean),
        columns=X_train_clean.columns, index=X_train_clean.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_clean),
        columns=X_test_clean.columns, index=X_test_clean.index
    )
    X_val_scaled = None
    if X_val_clean is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val_clean),
            columns=X_val_clean.columns, index=X_val_clean.index
        )

    # Step 2: Train each algorithm
    results = {}
    random_seed = config.get('random_seed', 42)

    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"[train] Training {algo.upper()} (baseline, no noise)")
        print(f"{'='*60}")

        if algo == 'knn':
            results[algo] = train_knn(
                X_train_scaled, y_train_clean,
                X_test_scaled, y_test_clean,
                random_seed=random_seed
            )
        elif algo == 'svm':
            results[algo] = train_svm_sgd(
                X_train_scaled, y_train_clean,
                X_test_scaled, y_test_clean,
                X_val=X_val_scaled, y_val=y_val_clean,
                class_weight='balanced',
                random_seed=random_seed
            )
        elif algo == 'mlp':
            results[algo] = train_mlp(
                X_train_scaled, y_train_clean,
                X_test_scaled, y_test_clean,
                X_val=X_val_scaled, y_val=y_val_clean,
                use_sample_weight=True,
                random_seed=random_seed
            )
        elif algo == 'xgb':
            results[algo] = train_xgb(
                X_train_scaled, y_train_clean,
                X_test_scaled, y_test_clean,
                X_val=X_val_scaled, y_val=y_val_clean,
                use_sample_weight=True,
                random_seed=random_seed
            )
        else:
            print(f"[train] WARNING: Unknown algorithm '{algo}', skipping.")

    return results


def run_noise_experiment(X_train_clean, X_test_clean, y_train_clean, y_test_clean,
                         X_val_clean=None, y_val_clean=None,
                         noise_type=None, noise_level=0.0,
                         algorithms=None, config=None):
    """
    Run a single noise experiment with INDEPENDENT scaler pipeline.

    Pipeline:
      X_train_clean → inject_noise → X_train_noisy
      → StandardScaler.fit(X_train_noisy)  ← KEY: independent scaler
      → transform(X_train_noisy, X_test_clean)
      → train each algorithm → collect results.

    The test set is ALWAYS kept clean; the scaler is ALWAYS fit on
    the (potentially noisy) training set alone.
    """
    if algorithms is None:
        algorithms = ['knn', 'svm', 'mlp', 'xgb']

    random_seed = config.get('random_seed', 42)

    # Step 1: Inject noise into training set ONLY
    if noise_type is None or noise_level == 0.0:
        X_train_noisy = X_train_clean.copy()
        y_train_noisy = y_train_clean.copy()
    else:
        X_train_noisy, y_train_noisy = inject_noise(
            X_train_clean.copy(), y_train_clean.copy(),
            noise_type=noise_type, level=noise_level,
            random_seed=random_seed
        )

    # Step 2: INDEPENDENT scaler fitted on (noisy) training set
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_noisy),
        columns=X_train_noisy.columns, index=X_train_noisy.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_clean),
        columns=X_test_clean.columns, index=X_test_clean.index
    )
    X_val_scaled = None
    if X_val_clean is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val_clean),
            columns=X_val_clean.columns, index=X_val_clean.index
        )

    # Step 3: Train each algorithm
    results = {}
    for algo in algorithms:
        label = f"{noise_type}({noise_level})" if noise_type else "baseline"
        print(f"\n{'='*60}")
        print(f"[train] Training {algo.upper()} | noise={label}")
        print(f"{'='*60}")

        if algo == 'knn':
            results[algo] = train_knn(
                X_train_scaled, y_train_noisy,
                X_test_scaled, y_test_clean,
                random_seed=random_seed
            )
        elif algo == 'svm':
            results[algo] = train_svm_sgd(
                X_train_scaled, y_train_noisy,
                X_test_scaled, y_test_clean,
                X_val=X_val_scaled, y_val=y_val_clean,
                class_weight='balanced',
                random_seed=random_seed
            )
        elif algo == 'mlp':
            results[algo] = train_mlp(
                X_train_scaled, y_train_noisy,
                X_test_scaled, y_test_clean,
                X_val=X_val_scaled, y_val=y_val_clean,
                use_sample_weight=True,
                random_seed=random_seed
            )
        elif algo == 'xgb':
            results[algo] = train_xgb(
                X_train_scaled, y_train_noisy,
                X_test_scaled, y_test_clean,
                X_val=X_val_scaled, y_val=y_val_clean,
                use_sample_weight=True,
                random_seed=random_seed
            )
        else:
            print(f"[train] WARNING: Unknown algorithm '{algo}', skipping.")

    return results


def run_all_noise_levels(X_train_clean, X_test_clean, y_train_clean, y_test_clean,
                         X_val_clean=None, y_val_clean=None,
                         noise_type=None, noise_levels=None,
                         algorithms=None, config=None):
    """
    Run noise experiments across all noise levels and return a dict keyed by level.
    """
    all_results = {}
    for level in noise_levels:
        print(f"\n{'#'*60}")
        print(f"### NOISE EXPERIMENT: {noise_type} | level={level}")
        print(f"{'#'*60}")
        all_results[level] = run_noise_experiment(
            X_train_clean, X_test_clean, y_train_clean, y_test_clean,
            X_val_clean=X_val_clean, y_val_clean=y_val_clean,
            noise_type=noise_type, noise_level=level,
            algorithms=algorithms, config=config
        )
    return all_results
