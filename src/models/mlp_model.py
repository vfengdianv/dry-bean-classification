"""
MLP (Multi-Layer Perceptron) model wrapper.

Uses sklearn's MLPClassifier with warm_start + manual epoch loop
to collect per-epoch training loss and validation error curves.

Architecture: (64, 32) two hidden layers — matches the plan specification.

Note about class balance:
  MLPClassifier (sklearn) does not support class_weight='balanced' or
  sample_weight. Unlike distance-based or margin-based methods, neural
  networks optimized with cross-entropy are less sensitive to moderate
  class imbalance. We retain default behavior and report macro F1 to
  fairly evaluate minority-class performance. This tradeoff is documented
  in the paper.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time
import warnings
import os

# Suppress ConvergenceWarning at module level for clean CLI output
os.environ['PYTHONWARNINGS'] = 'ignore'


def train_mlp(X_train, y_train, X_test, y_test,
              X_val=None, y_val=None,
              hidden_layer_sizes=(64, 32),
              n_epochs=100,
              use_sample_weight=True,
              random_seed=42, **kwargs):
    """
    Train an MLP with manual epoch control, collecting per-epoch metrics.

    Uses warm_start=True + max_iter=1 to control epoch count manually.
    Each call to .fit() advances exactly one SGD epoch.

    Parameters
    ----------
    X_train, y_train : training data (scaled)
    X_test, y_test   : test data (scaled, always clean)
    X_val, y_val     : optional validation set (scaled)
    hidden_layer_sizes : tuple
        (64, 32) by default.
    n_epochs : int
    use_sample_weight : bool
        Ignored for MLP (sklearn MLPClassifier does not support sample_weight).
        Kept for API consistency.
    random_seed : int
    """
    # Build the model once; warm_start=True means successive .fit() calls
    # continue from the previous solution.
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=1,
        warm_start=True,
        random_state=random_seed,
        early_stopping=False,
        verbose=False,          # Explicitly disable verbose output
    )

    train_losses = []
    val_errors = []

    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        # Shuffle each epoch for stochasticity
        idx = np.random.RandomState(random_seed + epoch).permutation(len(X_train))
        X_shuf = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
        y_shuf = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]

        # .fit() with max_iter=1 and warm_start=True = one epoch
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_shuf, y_shuf)

        # Record training loss
        if hasattr(model, 'loss_curve_') and len(model.loss_curve_) > 0:
            train_losses.append(model.loss_curve_[-1])

        # Validation error rate
        eval_set = X_val if X_val is not None else X_test
        eval_labels = y_val if y_val is not None else y_test
        if eval_set is not None:
            pred = model.predict(eval_set)
            err = 1.0 - accuracy_score(eval_labels, pred)
            val_errors.append(err)

    # Final evaluation
    train_time = time.perf_counter() - t0
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    # Inference speed
    start = time.perf_counter()
    for _ in range(100):
        model.predict(X_test)
    elapsed = time.perf_counter() - start
    infer_ms = (elapsed / 100 / len(X_test)) * 1000

    # Model size: sum of all weight matrices + biases
    n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    model_size_kb = n_params * 8 / 1024

    print(f"[MLP] layers={hidden_layer_sizes} epochs={n_epochs} | "
          f"Train acc={train_acc:.4f} | Test acc={test_acc:.4f} | "
          f"Infer={infer_ms:.5f} ms/sample | Train time={train_time:.1f}s")

    return {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_time_seconds': train_time,
        'infer_time_ms_per_sample': infer_ms,
        'model_size_kb': model_size_kb,
        'n_params': n_params,
        'loss_curves': {
            'train_loss': train_losses,
            'val_error': val_errors,
            'label': 'Cross-Entropy Loss (train) / Val Error Rate'
        },
        'test_pred': test_pred,
        'train_pred': train_pred,
    }
