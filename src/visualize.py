"""
Visualization module — generates all 14 figures for the project.

All saved to disk via plt.savefig(). NO interactive UI.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,'font.size':10,'axes.titlesize':13,'axes.labelsize':11,'figure.titlesize':14})
sns.set_style('whitegrid')

ALGO_COLORS = {'knn':'#e74c3c','svm':'#3498db','mlp':'#2ecc71','xgb':'#e67e22'}


# ── 1. Correlation heatmap ─────────────────────────────────────────────────
def plot_correlation_heatmap(X_train, save_path):
    corr = X_train.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={'shrink':0.8}, ax=ax)
    ax.set_title('Pearson Correlation Matrix of Dry Bean Features')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 2. Class distribution ──────────────────────────────────────────────────
def plot_class_distribution(y_train, y_test, y_val, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    df = pd.DataFrame({'Train':y_train.value_counts(),'Test':y_test.value_counts(),'Val':y_val.value_counts()}).fillna(0).astype(int)
    df.plot(kind='bar', ax=ax, color=['#2c3e50','#3498db','#95a5a6'], edgecolor='white')
    ax.set_title('Class Distribution Across Train / Test / Val Splits')
    ax.set_xlabel('Bean Class'); ax.set_ylabel('Count'); ax.legend(title='Split')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 3. Loss curves ─────────────────────────────────────────────────────────
def plot_loss_curves(loss_data, save_path):
    iter_algos = {k:v for k,v in loss_data.items() if v is not None}
    if not iter_algos: return
    fig, axes = plt.subplots(1, len(iter_algos), figsize=(5*len(iter_algos), 4))
    if len(iter_algos)==1: axes=[axes]
    for ax, (algo, curves) in zip(axes, iter_algos.items()):
        color = ALGO_COLORS.get(algo, '#333')
        if 'train_loss' in curves and len(curves['train_loss'])>0:
            ax.plot(range(1,len(curves['train_loss'])+1), curves['train_loss'], color=color, lw=1.5, alpha=0.8, label='Train')
        if 'val_loss' in curves and len(curves['val_loss'])>0:
            ax.plot(range(1,len(curves['val_loss'])+1), curves['val_loss'], color='#e74c3c', lw=1.5, ls='--', alpha=0.8, label='Val')
        elif 'val_error' in curves and len(curves['val_error'])>0:
            ax.plot(range(1,len(curves['val_error'])+1), curves['val_error'], color='#e74c3c', lw=1.5, ls='--', alpha=0.8, label='Val Error')
        ax.set_title(algo.upper()); ax.set_xlabel('Epochs / Rounds')
        ax.set_ylabel(curves.get('label','Loss')); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Training & Validation Loss Curves', fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight'); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 4. Confusion matrices ──────────────────────────────────────────────────
def plot_confusion_matrices(results, y_test, classes, save_path):
    from sklearn.metrics import confusion_matrix
    algos = list(results.keys()); n = len(algos)
    cols = min(2, n); rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n==1: axes = np.array([axes])
    axes = axes.flatten()
    for i, algo in enumerate(algos):
        cm = results[algo].get('confusion_matrix', confusion_matrix(y_test, results[algo]['test_pred']))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                    square=True, linewidths=0.5, ax=axes[i], cbar_kws={'shrink':0.8})
        axes[i].set_title(algo.upper()); axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('True')
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle('Confusion Matrices — Test Set', fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight'); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 5. Precision comparison ────────────────────────────────────────────────
def plot_precision_comparison(results, y_test, save_path):
    from sklearn.metrics import f1_score
    algos = list(results.keys())
    accs = [results[a]['test_acc'] for a in algos]
    f1s = [f1_score(y_test, results[a]['test_pred'], average='macro') for a in algos]
    x = np.arange(len(algos)); width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x-width/2, accs, width, label='Test Accuracy', color='#3498db', edgecolor='white')
    b2 = ax.bar(x+width/2, f1s, width, label='Macro F1', color='#2ecc71', edgecolor='white')
    for b in b1+b2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel('Score'); ax.set_title('Test Accuracy & Macro F1'); ax.legend(); ax.set_ylim(0.85, 1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 6. Inference speed ─────────────────────────────────────────────────────
def plot_inference_speed(results, save_path):
    algos = list(results.keys())
    speeds = [results[a]['infer_time_ms_per_sample'] for a in algos]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([a.upper() for a in algos], speeds, color=[ALGO_COLORS.get(a,'#333') for a in algos], edgecolor='white')
    for bar, s in zip(bars, speeds):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1, f'{s:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('ms / sample'); ax.set_title('Inference Speed (lower=faster)')
    ax.set_yscale('log'); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 7. Robustness curves ───────────────────────────────────────────────────
def plot_robustness_curves(robustness, save_path):
    noise_types = ['gaussian','label_flip','feature_missing']
    noise_labels = {'gaussian':'Gaussian Noise (η)','label_flip':'Label Flip (fraction)','feature_missing':'Feature Missing (fraction)'}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, nt in zip(axes, noise_types):
        for algo, nt_dict in robustness.items():
            if nt in nt_dict:
                lv = sorted(nt_dict[nt].keys()); accs = [nt_dict[nt][l] for l in lv]
                ax.plot(lv, accs, marker='o', lw=2, ms=6, color=ALGO_COLORS.get(algo,'#333'), label=algo.upper())
        ax.set_title(noise_labels.get(nt,nt)); ax.set_xlabel('Noise Intensity')
        ax.set_ylabel('Test Accuracy'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Robustness: Accuracy Degradation Under Noise', fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight'); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 8. Overfitting delta ───────────────────────────────────────────────────
def plot_overfitting_delta(results, save_path):
    from src.evaluate import compute_overfitting_delta
    df = compute_overfitting_delta(results)
    algos = list(df.index); x = np.arange(len(algos)); width = 0.3
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x-width, df['Train Accuracy'], width, label='Train Acc', color='#3498db', edgecolor='white')
    ax.bar(x+width, df['Test Accuracy'], width, label='Test Acc', color='#2ecc71', edgecolor='white')
    for i, (_, row) in enumerate(df.iterrows()):
        d = row['Δ (Overfitting)']; top = max(row['Train Accuracy'], row['Test Accuracy'])
        ax.annotate(f'Δ={d:.3f}', xy=(x[i], top), xytext=(x[i], top+0.015),
                    ha='center', fontsize=9, color='#e74c3c', fontweight='bold', arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels(algos); ax.set_ylabel('Accuracy')
    ax.set_title('Overfitting: Train vs Test Accuracy'); ax.legend(); ax.set_ylim(0.86, 1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 9. Training time comparison (NEW) ──────────────────────────────────────
def plot_train_time(results, save_path):
    algos = list(results.keys())
    times = [results[a].get('train_time_seconds', 0) for a in algos]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([a.upper() for a in algos], times, color=[ALGO_COLORS.get(a,'#333') for a in algos], edgecolor='white')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(times)*0.02, f'{t:.1f}s', ha='center', fontsize=10)
    ax.set_ylabel('Time (seconds)'); ax.set_title('Training Time Comparison')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 10. Model size / memory (NEW) ──────────────────────────────────────────
def plot_model_size(results, save_path):
    algos = list(results.keys())
    sizes = [results[a].get('model_size_kb', 0) for a in algos]
    params = [results[a].get('n_params', 0) for a in algos]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [ALGO_COLORS.get(a,'#333') for a in algos]
    b1 = ax1.bar([a.upper() for a in algos], sizes, color=colors, edgecolor='white')
    for bar, s in zip(b1, sizes): ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(sizes)*0.02, f'{s:.1f}', ha='center', fontsize=9)
    ax1.set_ylabel('KB'); ax1.set_title('Model Size (KB)')
    b2 = ax2.bar([a.upper() for a in algos], params, color=colors, edgecolor='white')
    for bar, p in zip(b2, params): ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(params)*0.02, f'{p:,}', ha='center', fontsize=9)
    ax2.set_ylabel('Count'); ax2.set_title('Parameter Count')
    ax2.set_yscale('log')
    fig.suptitle('Model Complexity Comparison', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 11. Per-class F1 heatmap (NEW) ─────────────────────────────────────────
def plot_per_class_f1(per_class_f1_df, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(per_class_f1_df, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0.70, vmax=1.0,
                linewidths=0.5, ax=ax, cbar_kws={'shrink':0.8})
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_xlabel('Bean Class'); ax.set_ylabel('Algorithm')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 12. XGBoost feature importance (NEW) ───────────────────────────────────
def plot_xgb_feature_importance(importance_series, save_path):
    if importance_series is None: return
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_series.plot(kind='barh', color='#e67e22', edgecolor='white', ax=ax)
    ax.set_title('XGBoost Feature Importance (Gain-based)')
    ax.set_xlabel('Importance'); ax.invert_yaxis()
    for i, v in enumerate(importance_series.values):
        ax.text(v+0.002, i, f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 13. KNN k-value sensitivity (NEW) ─────────────────────────────────────
def plot_knn_sensitivity(k_results, save_path):
    k_values = sorted(k_results.keys())
    train_accs = [k_results[k]['train_acc'] for k in k_values]
    test_accs = [k_results[k]['test_acc'] for k in k_values]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, train_accs, marker='o', color='#3498db', lw=2, label='Train Accuracy')
    ax.plot(k_values, test_accs, marker='s', color='#e74c3c', lw=2, label='Test Accuracy')
    best_k = k_values[test_accs.index(max(test_accs))]
    ax.axvline(x=best_k, color='#2ecc71', ls='--', lw=1, label=f'Best k={best_k}')
    ax.set_xlabel('k (Number of Neighbors)'); ax.set_ylabel('Accuracy')
    ax.set_title('KNN: k-Value Sensitivity (Bias-Variance Tradeoff)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── 14. Radar chart — multi-dimensional comparison (NEW) ───────────────────
def plot_radar_comparison(results, y_test, save_path):
    """Normalized radar chart across 6 dimensions."""
    from sklearn.metrics import f1_score
    from math import pi
    algos = list(results.keys())
    # Raw values
    raw = {}
    for a in algos:
        raw[a] = {
            'Accuracy': results[a]['test_acc'],
            'Macro F1': f1_score(y_test, results[a]['test_pred'], average='macro'),
            'Speed': 1.0 / max(results[a]['infer_time_ms_per_sample'], 1e-9),  # higher = faster
            'Anti-Overfit': 1.0 - abs(results[a]['train_acc'] - results[a]['test_acc']),
            'Train Speed': 1.0 / max(results[a].get('train_time_seconds', 1), 1),
            'Compactness': 1.0 / max(results[a].get('model_size_kb', 1), 1),  # smaller = better
        }
    # Min-max normalize to [0,1] per dimension
    dims = ['Accuracy','Macro F1','Speed','Anti-Overfit','Train Speed','Compactness']
    norm = {a:{} for a in algos}
    for d in dims:
        vals = [raw[a][d] for a in algos]
        mn, mx = min(vals), max(vals)
        for a in algos:
            norm[a][d] = 1.0 if mx==mn else (raw[a][d]-mn)/(mx-mn)*0.7+0.3  # scale to [0.3,1.0] for readability

    angles = [n/float(len(dims))*2*pi for n in range(len(dims))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for a in algos:
        values = [norm[a][d] for d in dims] + [norm[a][dims[0]]]
        ax.plot(angles, values, 'o-', lw=2, color=ALGO_COLORS.get(a,'#333'), label=a.upper())
        ax.fill(angles, values, alpha=0.08, color=ALGO_COLORS.get(a,'#333'))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_title('Multi-Dimensional Algorithm Comparison', fontsize=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight'); plt.close(fig)
    print(f"[visualize] → {save_path}")


# ── Generate all baseline figures ──────────────────────────────────────────

def generate_all_figures(results, X_train, y_train, y_test, X_val, y_val,
                         classes, output_dir, robustness=None,
                         per_class_f1_df=None, xgb_importance=None,
                         knn_k_results=None):
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    from src.evaluate import collect_loss_curves

    plot_correlation_heatmap(X_train, os.path.join(fig_dir, 'corr_heatmap.png'))
    plot_class_distribution(y_train, y_test, y_val, os.path.join(fig_dir, 'class_distribution.png'))
    plot_loss_curves(collect_loss_curves(results), os.path.join(fig_dir, 'loss_curves.png'))
    plot_confusion_matrices(results, y_test, classes, os.path.join(fig_dir, 'confusion_matrices.png'))
    plot_precision_comparison(results, y_test, os.path.join(fig_dir, 'precision_comparison.png'))
    plot_inference_speed(results, os.path.join(fig_dir, 'inference_speed.png'))
    plot_overfitting_delta(results, os.path.join(fig_dir, 'overfitting_delta.png'))
    plot_train_time(results, os.path.join(fig_dir, 'train_time.png'))
    plot_model_size(results, os.path.join(fig_dir, 'model_size.png'))
    plot_radar_comparison(results, y_test, os.path.join(fig_dir, 'radar_comparison.png'))

    if per_class_f1_df is not None:
        plot_per_class_f1(per_class_f1_df, os.path.join(fig_dir, 'per_class_f1.png'))
    if robustness is not None:
        plot_robustness_curves(robustness, os.path.join(fig_dir, 'robustness_curves.png'))
    if xgb_importance is not None:
        plot_xgb_feature_importance(xgb_importance, os.path.join(fig_dir, 'xgb_feature_importance.png'))
    if knn_k_results is not None:
        plot_knn_sensitivity(knn_k_results, os.path.join(fig_dir, 'knn_k_sensitivity.png'))

    print(f"[visualize] All figures in {fig_dir}/")
