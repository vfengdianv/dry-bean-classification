"""
KNN model wrapper — non-iterative lazy learner.

Uses sklearn's KNeighborsClassifier with distance weighting to
mitigate class imbalance.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time


def train_knn(X_train, y_train, X_test, y_test, k=5, random_seed=42, **kwargs):
    """
    Train a KNN classifier and evaluate.

    Parameters
    ----------
    X_train, y_train : training data (scaled)
    X_test, y_test   : test data (scaled, always clean)
    k : int
        Number of neighbors.
    random_seed : int

    Returns
    -------
    result : dict
        Keys: model, train_acc, test_acc, infer_time_ms_per_sample,
              loss_curves=None (KNN is non-iterative)
    """
    model = KNeighborsClassifier(
        n_neighbors=k,
        algorithm='auto',
        weights='distance',  # Mitigates class imbalance
        n_jobs=-1
    )

    # Train (with timing)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # Test accuracy
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    # Inference speed: average over 10 runs
    start = time.perf_counter()
    for _ in range(10):
        model.predict(X_test)
    elapsed = time.perf_counter() - start
    infer_ms = (elapsed / 10 / len(X_test)) * 1000  # ms per sample

    # Model size (memory proxy: number of stored samples × features)
    model_size_kb = (X_train.shape[0] * X_train.shape[1] * 8) / 1024  # 8 bytes per float64

    print(f"[KNN] k={k} | Train acc={train_acc:.4f} | Test acc={test_acc:.4f} | "
          f"Infer={infer_ms:.4f} ms/sample | Train time={train_time:.2f}s")

    return {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_time_seconds': train_time,
        'infer_time_ms_per_sample': infer_ms,
        'model_size_kb': model_size_kb,
        'n_params': X_train.shape[0] * X_train.shape[1],  # stored feature values
        'loss_curves': None,  # Non-iterative — no loss curve
        'test_pred': test_pred,
        'train_pred': train_pred,
    }


def train_knn_multi_k(X_train, y_train, X_test, y_test, k_values=None, random_seed=42):
    """
    Train KNN for multiple k values and return results for each.
    Used for the overfitting (bias-variance) analysis.
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 10, 15, 20, 30, 50]

    results = {}
    for k in k_values:
        results[k] = train_knn(X_train, y_train, X_test, y_test, k=k, random_seed=random_seed)
    return results
