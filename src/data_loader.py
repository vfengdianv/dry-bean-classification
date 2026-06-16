"""
Data loading module for the Dry Bean Dataset.

Reads the three pre-split CSV files (train/test/val) with UTF-8 BOM handling,
applies label cleaning via the centralized LABEL_MAP, and returns raw feature
DataFrames / label Series.
"""

import pandas as pd
from .label_mapping import clean_label


def load_data(train_path: str, test_path: str, val_path: str):
    """
    Load the three pre-split CSV files.

    - Uses encoding='utf-8-sig' to strip the BOM from column headers.
    - Applies label cleaning (LABEL_MAP lookup) at load time.
    - Returns raw feature DataFrames; numerical cleaning happens in preprocessing.py.

    Parameters
    ----------
    train_path, test_path, val_path : str
        Paths to the dirty CSV files.

    Returns
    -------
    X_train : pd.DataFrame    (n_train, 16) — raw features
    y_train : pd.Series       (n_train,)   — cleaned labels
    X_test  : pd.DataFrame    (n_test, 16)
    y_test  : pd.Series       (n_test,)
    X_val   : pd.DataFrame    (n_val, 16)
    y_val   : pd.Series       (n_val,)
    """
    def _read(path: str):
        df = pd.read_csv(path, encoding='utf-8-sig')
        # The Class column is the label; everything else is feature
        X = df.drop(columns=['Class'])
        y_raw = df['Class']
        # Clean labels
        y = y_raw.apply(clean_label)
        # Verify no None/NaN slipped through
        if y.isna().any():
            bad = y_raw[y.isna()]
            raise ValueError(f"Label cleaning produced NaN for: {bad.unique()}")
        return X, y

    X_train, y_train = _read(train_path)
    X_test, y_test = _read(test_path)
    X_val, y_val = _read(val_path)

    print(f"[data_loader] Loaded: train={X_train.shape}, test={X_test.shape}, val={X_val.shape}")
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)
