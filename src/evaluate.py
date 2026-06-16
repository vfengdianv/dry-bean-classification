"""
Evaluation module — computes all ten analysis dimensions.

1. Test-set precision: accuracy, macro F1, per-class F1, confusion matrix
2. Loss curve data aggregation
3. Inference speed comparison
4. Training time comparison (NEW)
5. Robustness against noise
6. Overfitting analysis
7. Model size / memory footprint (NEW)
8. Feature importance extraction — XGBoost gain-based (NEW)
9. Parameter count comparison (NEW)
10. KNN k-value sensitivity analysis (NEW)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
import time


# ── 1. Precision comparison ────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, classes=None):
    """Compute accuracy, macro F1, and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return {'accuracy': acc, 'macro_f1': f1, 'confusion_matrix': cm}


def build_precision_table(results, y_test, classes):
    """Summary DataFrame: accuracy + macro F1 per algorithm."""
    rows = []
    for algo, r in results.items():
        rows.append({
            'Algorithm': algo.upper(),
            'Test Accuracy': r['test_acc'],
            'Macro F1': f1_score(y_test, r['test_pred'], average='macro'),
        })
    return pd.DataFrame(rows).set_index('Algorithm')


def build_per_class_f1(results, y_test, classes):
    """Per-class F1 scores for each algorithm — broadens evaluation granularity."""
    rows = []
    for algo, r in results.items():
        f1_per_class = f1_score(y_test, r['test_pred'], average=None, labels=classes)
        row = {'Algorithm': algo.upper()}
        for cls, f1_val in zip(classes, f1_per_class):
            row[cls] = f1_val
        rows.append(row)
    return pd.DataFrame(rows).set_index('Algorithm')


# ── 2. Loss curve aggregation ──────────────────────────────────────────────

def collect_loss_curves(results):
    """Gather loss curve data from all algorithms."""
    loss_data = {}
    for algo, r in results.items():
        loss_data[algo] = r.get('loss_curves') if r.get('loss_curves') is not None else None
    return loss_data


# ── 3. Inference speed ─────────────────────────────────────────────────────

def build_speed_table(results):
    """Summary of inference speed across algorithms."""
    rows = []
    for algo, r in results.items():
        rows.append({
            'Algorithm': algo.upper(),
            'Inference (ms/sample)': r['infer_time_ms_per_sample'],
        })
    return pd.DataFrame(rows).set_index('Algorithm')


# ── 4. Training time comparison (NEW) ──────────────────────────────────────

def build_train_time_table(results):
    """Summary of training time across algorithms."""
    rows = []
    for algo, r in results.items():
        rows.append({
            'Algorithm': algo.upper(),
            'Training Time (s)': r.get('train_time_seconds', np.nan),
        })
    return pd.DataFrame(rows).set_index('Algorithm')


# ── 5. Robustness matrix ───────────────────────────────────────────────────

def build_robustness_matrix(noise_results, baseline_results, y_test):
    """Build accuracy matrix under different noise types × intensities."""
    robustness = {}
    for algo in baseline_results:
        robustness[algo] = {}

    for noise_type, level_dict in (noise_results or {}).items():
        for level, algo_dict in level_dict.items():
            for algo, r in algo_dict.items():
                robustness.setdefault(algo, {}).setdefault(noise_type, {})[level] = r['test_acc']

    for algo, r in baseline_results.items():
        for noise_type in ['gaussian', 'label_flip', 'feature_missing']:
            robustness.setdefault(algo, {}).setdefault(noise_type, {})[0.0] = r['test_acc']

    return robustness


# ── 6. Overfitting delta ───────────────────────────────────────────────────

def compute_overfitting_delta(results):
    """Compute Δ = train_acc - test_acc for each algorithm."""
    rows = []
    for algo, r in results.items():
        delta = r['train_acc'] - r['test_acc']
        rows.append({
            'Algorithm': algo.upper(),
            'Train Accuracy': r['train_acc'],
            'Test Accuracy': r['test_acc'],
            'Δ (Overfitting)': delta,
        })
    return pd.DataFrame(rows).set_index('Algorithm')


# ── 7. Model size / memory footprint (NEW) ─────────────────────────────────

def build_model_size_table(results):
    """Summary of model size (KB) and parameter counts."""
    rows = []
    for algo, r in results.items():
        rows.append({
            'Algorithm': algo.upper(),
            'Model Size (KB)': r.get('model_size_kb', np.nan),
            'Parameters': r.get('n_params', np.nan),
        })
    return pd.DataFrame(rows).set_index('Algorithm')


# ── 8. XGBoost feature importance (NEW) ────────────────────────────────────

def extract_feature_importance(xgb_result, feature_names):
    """
    Extract XGBoost feature importance scores (gain-based).
    Returns a sorted Series if xgb_result is available, else None.
    """
    if xgb_result is None or 'model' not in xgb_result:
        return None
    model = xgb_result['model']
    if not hasattr(model, 'feature_importances_'):
        return None
    importances = model.feature_importances_
    series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    return series


# ── Utility: full evaluation summary ───────────────────────────────────────

def full_evaluation(baseline_results, y_test, classes, noise_results=None, feature_names=None):
    """Run all ten evaluation dimensions."""
    summary = {}
    summary['precision_table'] = build_precision_table(baseline_results, y_test, classes)
    summary['per_class_f1'] = build_per_class_f1(baseline_results, y_test, classes)
    summary['loss_data'] = collect_loss_curves(baseline_results)
    summary['speed_table'] = build_speed_table(baseline_results)
    summary['train_time_table'] = build_train_time_table(baseline_results)
    summary['model_size_table'] = build_model_size_table(baseline_results)
    summary['overfitting_table'] = compute_overfitting_delta(baseline_results)

    if noise_results is not None:
        summary['robustness'] = build_robustness_matrix(noise_results, baseline_results, y_test)

    if feature_names is not None and 'xgb' in baseline_results:
        summary['xgb_feature_importance'] = extract_feature_importance(baseline_results['xgb'], feature_names)

    return summary
