"""
Numerical feature preprocessing module.

Handles all numerical cleaning steps:
  1. Compactness: strip ' cm' suffix → float
  2. Solidity: '?' → NaN → float
  3. Perimeter: median imputation (fit on train only)
  4. Solidity: median imputation (fit on train only)
  5. StandardScaler (fit on train only, transform all sets)

Design principle: ALL fit operations use ONLY the training set.
Imputer and Scaler are fitted on train, then applied to test/val.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def clean_numeric_features(X_train, X_test, X_val):
    """
    Clean numerical issues in feature DataFrames.

    Steps:
      1. Compactness: strip trailing ' cm' or 'cm' → float; NAs produced become NaN.
      2. Solidity: replace '?' with NaN → float.
      3. Perimeter: impute NaNs with train median.
      4. Solidity: impute NaNs with train median.

    Parameters
    ----------
    X_train, X_test, X_val : pd.DataFrame
        Raw feature DataFrames from data_loader.

    Returns
    -------
    X_train_c, X_test_c, X_val_c : pd.DataFrame
        Numerically cleaned DataFrames (not yet scaled).
    """
    X_train_c = X_train.copy()
    X_test_c = X_test.copy()
    X_val_c = X_val.copy()

    # --- Step 1: Compactness — strip ' cm' suffix ---
    for X_df in [X_train_c, X_test_c, X_val_c]:
        col = X_df['Compactness']
        if col.dtype == object:
            X_df['Compactness'] = (
                col.astype(str)
                .str.replace(r'\s*cm\s*$', '', regex=True)
                .str.strip()
            )
        X_df['Compactness'] = pd.to_numeric(X_df['Compactness'], errors='coerce')

    # --- Step 2: Solidity — replace '?' with NaN ---
    for X_df in [X_train_c, X_test_c, X_val_c]:
        col = X_df['Solidity']
        if col.dtype == object:
            X_df['Solidity'] = col.replace('?', np.nan)
        X_df['Solidity'] = pd.to_numeric(X_df['Solidity'], errors='coerce')

    # --- Step 3: Ensure all columns are numeric ---
    # (Perimeter may have empty strings → NaN)
    for X_df in [X_train_c, X_test_c, X_val_c]:
        for c in X_df.columns:
            if X_df[c].dtype == object:
                X_df[c] = pd.to_numeric(X_df[c], errors='coerce')

    # --- Step 4: Median imputation (fit on train only) ---
    # Perimeter almost certainly has NaNs; Solidity has NaNs from '?'.
    # We fit one imputer for all numeric columns — only Perimeter/Solidity have NaNs
    # so it's harmless for other columns.
    imputer = SimpleImputer(strategy='median')
    X_train_c = pd.DataFrame(
        imputer.fit_transform(X_train_c),
        columns=X_train_c.columns,
        index=X_train_c.index
    )
    X_test_c = pd.DataFrame(
        imputer.transform(X_test_c),
        columns=X_test_c.columns,
        index=X_test_c.index
    )
    X_val_c = pd.DataFrame(
        imputer.transform(X_val_c),
        columns=X_val_c.columns,
        index=X_val_c.index
    )

    print(f"[preprocessing] Numerical cleaning done. "
          f"NaNs after cleaning: train={X_train_c.isna().sum().sum()}, "
          f"test={X_test_c.isna().sum().sum()}, "
          f"val={X_val_c.isna().sum().sum()}")

    return X_train_c, X_test_c, X_val_c


def detect_outliers_3sigma(X_train):
    """
    Detect extreme outliers using the 3σ rule per feature.
    Returns a boolean mask DataFrame (True = outlier).
    Only *detects* — does not remove. The decision to keep or drop
    is made by the caller based on experiment configuration.

    Parameters
    ----------
    X_train : pd.DataFrame
        Cleaned training features.

    Returns
    -------
    outlier_mask : pd.DataFrame (bool)
    outlier_summary : dict  {feature_name: count}
    """
    outlier_mask = pd.DataFrame(False, index=X_train.index, columns=X_train.columns)
    summary = {}
    for col in X_train.columns:
        mean = X_train[col].mean()
        std = X_train[col].std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        mask = (X_train[col] < lower) | (X_train[col] > upper)
        outlier_mask[col] = mask
        summary[col] = int(mask.sum())
    return outlier_mask, summary


def apply_scaler(X_train, X_test, X_val):
    """
    Fit StandardScaler on training set ONLY, transform all three sets.

    Parameters
    ----------
    X_train, X_test, X_val : pd.DataFrame
        Numerically cleaned DataFrames (output of clean_numeric_features).

    Returns
    -------
    X_train_s, X_test_s, X_val_s : pd.DataFrame
        Scaled DataFrames (same shape, same column/index).
    scaler : StandardScaler
        The fitted scaler (can be reused or discarded).
    """
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    print(f"[preprocessing] StandardScaler fitted on train ({X_train.shape[0]} samples).")
    return X_train_s, X_test_s, X_val_s, scaler


def preprocess_pipeline(X_train, X_test, X_val, outlier_action='keep'):
    """
    Run the full preprocessing pipeline:
      clean_numeric → [outlier detection] → StandardScaler.

    Parameters
    ----------
    X_train, X_test, X_val : pd.DataFrame
        Raw features from data_loader.
    outlier_action : str
        'keep' — retain outliers (default).
        'drop' — remove rows with any 3σ outlier (train only).

    Returns
    -------
    X_train_s, X_test_s, X_val_s : pd.DataFrame
        Cleaned and scaled features.
    outlier_info : dict or None
        Outlier detection summary.
    """
    # Stage 1: Numerical cleaning
    X_train_c, X_test_c, X_val_c = clean_numeric_features(X_train, X_test, X_val)

    # Stage 2: Outlier detection
    outlier_info = None
    if outlier_action == 'drop':
        mask, outlier_info = detect_outliers_3sigma(X_train_c)
        # Remove rows where ANY feature is an outlier
        row_outlier = mask.any(axis=1)
        n_dropped = row_outlier.sum()
        X_train_c = X_train_c[~row_outlier].reset_index(drop=True)
        y_dropped_note = n_dropped  # caller must also drop y_train rows
        print(f"[preprocessing] 3σ outlier removal: dropped {n_dropped}/{len(mask)} rows "
              f"({100*n_dropped/len(mask):.1f}%)")
        # We return a note so the caller can drop corresponding y rows
        outlier_info = {'dropped_count': n_dropped, 'dropped_mask': row_outlier, 'per_feature': outlier_info}
    else:
        _, outlier_summary = detect_outliers_3sigma(X_train_c)
        outlier_info = outlier_summary
        # Don't drop, just log
        n_total = sum(outlier_summary.values())
        print(f"[preprocessing] 3σ outlier detection: {n_total} total outlier cells detected (kept).")

    # Stage 3: StandardScaler
    X_train_s, X_test_s, X_val_s, scaler = apply_scaler(X_train_c, X_test_c, X_val_c)

    return X_train_s, X_test_s, X_val_s, outlier_info
