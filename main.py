#!/usr/bin/env python
"""
Dry Bean Classification — Command-Line Entry Point

Usage:
    python main.py --config configs/baseline.yaml
    python main.py --config configs/baseline.yaml --algorithms knn,svm,mlp,xgb
    python main.py --config configs/baseline.yaml --output_dir outputs/custom

All experiments run without any GUI; figures are saved to disk.
"""

import os, sys, argparse, warnings
import yaml
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.label_mapping import CLEAN_CLASSES
from src.preprocessing import clean_numeric_features, apply_scaler
from src.train import run_baseline_experiment, run_all_noise_levels
from src.models.knn_model import train_knn_multi_k
from src.evaluate import full_evaluation
from src.visualize import generate_all_figures


def parse_args():
    parser = argparse.ArgumentParser(description='Dry Bean Classification — Full Pipeline')
    parser.add_argument('--config', type=str, required=True, help='YAML config file path.')
    parser.add_argument('--algorithms', type=str, default=None, help='Comma-separated overrides.')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output dir.')
    parser.add_argument('--skip_noise', action='store_true', help='Skip noise experiments.')
    parser.add_argument('--skip_knn_k', action='store_true', help='Skip KNN k-sensitivity scan.')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.algorithms:
        config['algorithms'] = [a.strip() for a in args.algorithms.split(',')]
    if args.output_dir:
        config['output']['dir'] = args.output_dir

    output_dir = config['output']['dir']
    algorithms = config['algorithms']
    random_seed = config.get('random_seed', 42)
    noise_cfg = config.get('noise', {})
    feature_names = ['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRation',
                     'Eccentricity','ConvexArea','EquivDiameter','Extent','Solidity',
                     'roundness','Compactness','ShapeFactor1','ShapeFactor2','ShapeFactor3','ShapeFactor4']

    print("=" * 60)
    print(f"  Dry Bean Classification — {config['experiment']['name']}")
    print(f"  Algorithms: {algorithms}  |  Seed: {random_seed}  |  Output: {output_dir}")
    print("=" * 60)

    # ── Stage 1: Load ─────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    data_cfg = config['data']
    (X_train_raw, y_train), (X_test_raw, y_test), (X_val_raw, y_val) = load_data(
        data_cfg['train_path'], data_cfg['test_path'], data_cfg['val_path'])
    classes = sorted(CLEAN_CLASSES)

    # ── Stage 2: Clean ────────────────────────────────────────────────────
    print("\n[2/5] Cleaning numerical features...")
    X_train_clean, X_test_clean, X_val_clean = clean_numeric_features(X_train_raw, X_test_raw, X_val_raw)

    # ── Stage 3: Baseline ─────────────────────────────────────────────────
    print("\n[3/5] Running baseline experiment...")
    baseline_results = run_baseline_experiment(
        X_train_clean, X_test_clean, y_train, y_test,
        X_val_clean=X_val_clean, y_val_clean=y_val, algorithms=algorithms, config=config)

    # ── Stage 4: Noise ────────────────────────────────────────────────────
    noise_results = None
    noise_type = noise_cfg.get('type')
    noise_levels = noise_cfg.get('levels', [])
    if noise_type and noise_levels and not args.skip_noise:
        print(f"\n[4/5] Noise experiments: {noise_type} @ {noise_levels}...")
        noise_results_all = run_all_noise_levels(
            X_train_clean, X_test_clean, y_train, y_test,
            X_val_clean=X_val_clean, y_val_clean=y_val,
            noise_type=noise_type, noise_levels=noise_levels, algorithms=algorithms, config=config)
        noise_results = {noise_type: noise_results_all}
    else:
        print("\n[4/5] Skipping noise experiments.")

    # ── KNN k-sensitivity ─────────────────────────────────────────────────
    knn_k_results = None
    if 'knn' in algorithms and not args.skip_knn_k:
        print("\n     Running KNN k-sensitivity scan...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xts = pd.DataFrame(scaler.fit_transform(X_train_clean), columns=X_train_clean.columns)
        Xes = pd.DataFrame(scaler.transform(X_test_clean), columns=X_test_clean.columns)
        knn_k_results = train_knn_multi_k(Xts, y_train, Xes, y_test, random_seed=random_seed)

    # ── Stage 5: Evaluate & Visualize ─────────────────────────────────────
    print("\n[5/5] Evaluating and generating figures...")
    evaluation = full_evaluation(
        baseline_results, y_test, classes,
        noise_results=noise_results, feature_names=feature_names)

    generate_all_figures(
        baseline_results, X_train_clean, y_train, y_test,
        X_val_clean, y_val, classes, output_dir,
        robustness=evaluation.get('robustness'),
        per_class_f1_df=evaluation.get('per_class_f1'),
        xgb_importance=evaluation.get('xgb_feature_importance'),
        knn_k_results=knn_k_results)

    # Save CSVs
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    evaluation['precision_table'].to_csv(os.path.join(tables_dir, 'precision.csv'))
    evaluation['speed_table'].to_csv(os.path.join(tables_dir, 'inference_speed.csv'))
    evaluation['overfitting_table'].to_csv(os.path.join(tables_dir, 'overfitting_delta.csv'))
    evaluation['train_time_table'].to_csv(os.path.join(tables_dir, 'train_time.csv'))
    evaluation['model_size_table'].to_csv(os.path.join(tables_dir, 'model_size.csv'))
    evaluation['per_class_f1'].to_csv(os.path.join(tables_dir, 'per_class_f1.csv'))
    if evaluation.get('xgb_feature_importance') is not None:
        evaluation['xgb_feature_importance'].to_csv(os.path.join(tables_dir, 'xgb_feature_importance.csv'))

    # Console summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print("\n[Precision]"); print(evaluation['precision_table'].to_string())
    print("\n[Per-Class F1]"); print(evaluation['per_class_f1'].to_string())
    print("\n[Inference Speed]"); print(evaluation['speed_table'].to_string())
    print("\n[Training Time]"); print(evaluation['train_time_table'].to_string())
    print("\n[Model Size]"); print(evaluation['model_size_table'].to_string())
    print("\n[Overfitting Δ]"); print(evaluation['overfitting_table'].to_string())
    if evaluation.get('xgb_feature_importance') is not None:
        print("\n[XGBoost Top-5 Features]"); print(evaluation['xgb_feature_importance'].head(5).to_string())
    print(f"\nAll outputs → {output_dir}/\nDone.\n")


if __name__ == '__main__':
    main()
