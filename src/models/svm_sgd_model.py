"""
SVM (SGD approximation) model wrapper.

Uses sklearn's SGDClassifier with hinge loss and partial_fit loop
to obtain per-epoch training loss and validation error curves.

Key implementation detail: the FIRST call to partial_fit MUST receive
the `classes` parameter listing all unique labels.
"""

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import time


def train_svm_sgd(X_train, y_train, X_test, y_test,
                  X_val=None, y_val=None,
                  n_epochs=50, class_weight='balanced',
                  random_seed=42, **kwargs):
    """
    Train a linear SVM via SGD with partial_fit, collecting per-epoch metrics.

    Parameters
    ----------
    X_train, y_train : training data (scaled)
    X_test, y_test   : test data (scaled, always clean)
    X_val, y_val     : optional validation set (scaled)
    n_epochs : int
    class_weight : str or None
        'balanced' for inverse-frequency class weights.
    random_seed : int

    Returns
    -------
    result : dict
        Keys: model, train_acc, test_acc, infer_time_ms_per_sample,
              train_loss_curve, val_error_curve, test_pred, train_pred
    """
    classes = np.unique(y_train)

    # Compute class_weight as a dict (required for partial_fit compatibility)
    if class_weight == 'balanced':
        from sklearn.utils.class_weight import compute_class_weight
        cw_array = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {c: w for c, w in zip(classes, cw_array)}
    else:
        class_weight_dict = class_weight

    model = SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=0.0001,
        max_iter=1,             # We control epochs via manual loop
        tol=None,
        random_state=random_seed,
        class_weight=class_weight_dict,
        warm_start=True,
        early_stopping=False,
        learning_rate='optimal',
    )

    # Optional: compute sample_weight for class balance if class_weight is None
    sample_weight = None
    if class_weight == 'balanced':
        sample_weight = compute_sample_weight('balanced', y_train)

    train_losses = []
    val_errors = []

    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        # Shuffle training data each epoch to avoid learning-rate artifacts
        idx = np.random.RandomState(random_seed + epoch).permutation(len(X_train))
        X_shuf = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
        y_shuf = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
        sw_shuf = sample_weight[idx] if sample_weight is not None else None

        if epoch == 0:
            model.partial_fit(X_shuf, y_shuf, classes=classes, sample_weight=sw_shuf)
        else:
            model.partial_fit(X_shuf, y_shuf, sample_weight=sw_shuf)

        # Record training loss (average hinge loss from loss_curve_)
        # warm_start + max_iter=1 means loss_curve_ has one entry per partial_fit call
        if hasattr(model, 'loss_curve_') and len(model.loss_curve_) > 0:
            train_losses.append(model.loss_curve_[-1])

        # Validation error rate (1 - accuracy)
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

    # Model size: coef_ + intercept_ in float64
    n_params = model.coef_.size + model.intercept_.size
    model_size_kb = n_params * 8 / 1024

    print(f"[SVM-SGD] epochs={n_epochs} | Train acc={train_acc:.4f} | "
          f"Test acc={test_acc:.4f} | Infer={infer_ms:.5f} ms/sample | "
          f"Train time={train_time:.1f}s")

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
            'label': 'Hinge Loss (train) / Val Error Rate'
        },
        'test_pred': test_pred,
        'train_pred': train_pred,
    }
