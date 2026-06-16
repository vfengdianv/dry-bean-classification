"""
XGBoost model wrapper — the "not-taught-in-class" algorithm.

Uses xgboost.XGBClassifier with eval_set to extract train/val mlogloss
curves. Supports sample_weight for class imbalance mitigation.
"""

from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
import xgboost as xgb


def train_xgb(X_train, y_train, X_test, y_test,
              X_val=None, y_val=None,
              n_estimators=200,
              use_sample_weight=True,
              random_seed=42, **kwargs):
    """
    Train an XGBoost classifier with eval_set for loss curves.

    Parameters
    ----------
    X_train, y_train : training data (scaled)
    X_test, y_test   : test data (scaled, always clean)
    X_val, y_val     : optional validation set (scaled)
    n_estimators : int
        Number of boosting rounds.
    use_sample_weight : bool
        Compute balanced sample weights to mitigate class imbalance.
    random_seed : int

    Returns
    -------
    result : dict
    """
    # Encode string labels to integers for XGBoost
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    y_val_enc = le.transform(y_val) if y_val is not None else None

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
        n_jobs=-1,
        verbosity=0,
    )

    # Sample weights for class balance
    sample_weight = None
    sample_weight_eval = None
    if use_sample_weight:
        sample_weight = compute_sample_weight('balanced', y_train_enc)

    # Build eval_set
    eval_set = [(X_train, y_train_enc)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val_enc))
        if sample_weight is not None:
            sample_weight_eval = [sample_weight, None]
    else:
        eval_set.append((X_test, y_test_enc))
        if sample_weight is not None:
            sample_weight_eval = [sample_weight, None]

    # Train (with timing)
    t0 = time.perf_counter()
    model.fit(
        X_train, y_train_enc,
        eval_set=eval_set,
        sample_weight=sample_weight,
        sample_weight_eval_set=sample_weight_eval,
        verbose=False
    )
    train_time = time.perf_counter() - t0

    # Extract loss curves
    evals_result = model.evals_result()
    train_mlogloss = evals_result['validation_0']['mlogloss']
    val_key = 'validation_1' if len(evals_result) > 1 else 'validation_0'
    val_mlogloss = evals_result[val_key]['mlogloss']

    # Evaluation
    train_pred_enc = model.predict(X_train)
    train_acc = accuracy_score(y_train_enc, train_pred_enc)
    test_pred_enc = model.predict(X_test)
    test_acc = accuracy_score(y_test_enc, test_pred_enc)

    # Decode predictions back to labels
    test_pred = le.inverse_transform(test_pred_enc)
    train_pred = le.inverse_transform(train_pred_enc)

    # Inference speed
    start = time.perf_counter()
    for _ in range(100):
        model.predict(X_test)
    elapsed = time.perf_counter() - start
    infer_ms = (elapsed / 100 / len(X_test)) * 1000

    # Model size: estimate from number of trees × leaves (rough)
    # More accurate: sum of all tree node values
    booster = model.get_booster()
    import json
    dump_str = booster.get_dump(dump_format='json')
    n_params = sum(len(json.loads(t).get('split_indices', [])) for t in dump_str) * 2 + n_estimators * 2
    model_size_kb = booster.save_raw().__sizeof__() / 1024

    print(f"[XGBoost] n_estimators={n_estimators} | "
          f"Train acc={train_acc:.4f} | Test acc={test_acc:.4f} | "
          f"Infer={infer_ms:.5f} ms/sample | Train time={train_time:.1f}s")

    return {
        'model': model,
        'label_encoder': le,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_time_seconds': train_time,
        'infer_time_ms_per_sample': infer_ms,
        'model_size_kb': model_size_kb,
        'n_params': n_params,
        'loss_curves': {
            'train_loss': train_mlogloss,
            'val_loss': val_mlogloss,
            'label': 'mlogloss (train) / mlogloss (val)'
        },
        'test_pred': test_pred,
        'train_pred': train_pred,
    }
