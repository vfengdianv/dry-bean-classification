# 🫘 Dry Bean Classification — Full Pipeline ML Project

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-brightgreen?logo=xgboost)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**基于预划分 Dry Bean Dataset 的端到端机器学习工程**
<br>📊 数据分析 → 🧹 数据处理 → 🤖 4算法实验 → 🖼️ 14图表 → 📝 课程论文

</div>

---

## 📖 目录

- [数据集描述](#-数据集描述)
- [数据污染全景](#-数据污染全景)
- [算法矩阵](#-算法矩阵)
- [项目架构](#-项目架构)
- [快速开始](#-快速开始)
- [实验结果](#-实验结果)
- [图表展示](#-图表展示)
- [对比维度矩阵](#-对比维度矩阵)
- [可复现性](#-可复现性)

---

## 📊 数据集描述

**Dry Bean Dataset** — 7 种干豆，16 维形态学特征，13,611 条样本。

| 类别 | 训练集 | 测试集 | 验证集 | 占比 |
|------|:-----:|:-----:|:-----:|:---:|
| DERMASON | 2,503 | 658 | 331 | 36.1% |
| SIRA | 1,837 | 498 | 271 | 19.3% |
| SEKER | 1,408 | 403 | 194 | 14.8% |
| HOROZ | 1,340 | 376 | 176 | 14.1% |
| CALI | 1,151 | 318 | 150 | 12.1% |
| BARBUNYA | 927 | 267 | 118 | 9.7% |
| BOMBAY | 361 | 103 | 46 | 3.8% |
| **合计** | **9,527** | **2,737** | **1,347** | 100% |

**16 个数值特征**：`Area` `Perimeter` `MajorAxisLength` `MinorAxisLength` `AspectRation` `Eccentricity` `ConvexArea` `EquivDiameter` `Extent` `Solidity` `roundness` `Compactness` `ShapeFactor1` `ShapeFactor2` `ShapeFactor3` `ShapeFactor4`

> ⚠️ BOMBAY:DERMASON ≈ **1:7**，模型启用类别加权策略

---

## 🧹 数据污染全景

原始数据经过系统性审查，发现 **7 类污染**，在训练/测试/验证三集中均匀分布：

| # | 污染类型 | 影响列 | Train | Test | Val | 处理策略 |
|:--:|----------|--------|:-----:|:----:|:---:|----------|
| 1 | BOM 头 | Area | ✅ | ✅ | ✅ | `encoding='utf-8-sig'` |
| 2 | 大小写混淆 | Class | 6 变体 | 4 变体 | 4 变体 | 穷举 LABEL_MAP |
| 3 | 字符替换 (3→E,0→O) | Class | 4 变体 | 4 变体 | 4 变体 | 穷举 LABEL_MAP |
| 4 | 尾部空格 | Class | 8 变体 | 7 变体 | 6 变体 | `.strip()` + MAP |
| 5 | 缺失值 | Perimeter | 469 | 146 | 62 | 训练集中位数填补 |
| 6 | `?` 占位符 | Solidity | 202+272 | 46+75 | 22+41 | → NaN → 中位数填补 |
| 7 | `cm` 单位后缀 | Compactness | 258 | 92 | 38 | 正则剥离 → float |

```
标签清洗效果:
  原始标签: 25 种  ──LABEL_MAP──>  清洗后: 7 种 ✅
```

---

## 🤖 算法矩阵

| 算法 | 范式 | 训练方式 | Loss | 课堂 | 亮点 |
|------|------|----------|:---:|:---:|------|
| **KNN** (k=5) | 实例学习 | 惰性 | ❌ | 已学 | Bias-Variance 演示 |
| **SVM** (SGD) | 最大间隔 | hinge+SGD | ✅ | 已学 | 过拟合最小 Δ=0.0027 |
| **MLP** (64,32) | 神经网络 | 交叉熵 | ✅ | 已学 | 精度最高 92.73% |
| **XGBoost** | 梯度提升 | boosting | ✅ | **未学** | 鲁棒性之王 |

### 为什么不用 VGG？

> VGG 的 2D 卷积核依赖**空间局部相关性**（相邻像素有语义关联），而干豆数据的 16 个特征是无拓扑结构的异构表格属性。强行 reshape 为图像张量进行卷积缺乏物理依据。XGBoost 是更适合表格数据的先进算法。

---

## 🏗️ 项目架构

```
dry-bean-classification/
├── 📁 configs/                  YAML 配置（4个实验场景）
├── 📁 src/
│   ├── data_loader.py           CSV→DataFrame + BOM修复
│   ├── label_mapping.py         24键穷举标签映射表
│   ├── preprocessing.py         数值清洗 + StandardScaler
│   ├── noise_injector.py        3类噪声统一接口
│   ├── models/
│   │   ├── knn_model.py         KNN (weights='distance')
│   │   ├── svm_sgd_model.py     SVM-SGD (class_weight)
│   │   ├── mlp_model.py         MLP (warm_start epoch loop)
│   │   └── xgb_model.py         XGBoost (evals_result)
│   ├── train.py                 训练调度 + 噪声独立Scaler
│   ├── evaluate.py              10维度评估指标
│   └── visualize.py             14种图表 (plt.savefig)
├── main.py                      🚀 命令行入口
├── paper.md                     📝 课程论文
├── requirements.txt
├── README.md
└── outputs/                     🖼️ 自动生成（图表+CSV）
```

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/<username>/dry-bean-classification.git
cd dry-bean-classification
pip install -r requirements.txt
```

### 运行

```bash
# 基准实验（4算法全量对比）
python main.py --config configs/baseline.yaml

# 鲁棒性实验
python main.py --config configs/noise_gaussian.yaml
python main.py --config configs/noise_label.yaml
python main.py --config configs/noise_missing.yaml

# 灵活组合
python main.py --config configs/baseline.yaml --algorithms knn,xgb
python main.py --config configs/baseline.yaml --output_dir outputs/custom
```

> 🖥️ **纯 CLI 运行，无 UI 弹出** — 所有图表自动保存至 `outputs/<name>/figures/`

---

## 📈 实验结果

### 综合精度表

| 算法 | 准确率 | 宏 F1 | 推理(ms) | 训练(s) | 模型(KB) | Δ过拟合 | 高斯0.15 | 标签20% | 缺失20% |
|------|:-----:|:-----:|:--------:|:------:|:-------:|:------:|:------:|:------:|:------:|
| KNN | .9214 | .9305 | .0093 | 0.01 | 1191 | .0786 | .9160 | .8922 | .9116 |
| SVM | .9028 | .9172 | .0003 | 0.65 | **0.93** | **.0027** | .9090 | .7541 | .8126 |
| MLP | **.9273** | **.9357** | .0009 | 5.09 | 26.6 | .0143 | .9266 | **.9251** | .9222 |
| XGB  | .9236 | .9356 | .0043 | 1.69 | 2745 | .0750 | **.9262** | .9160 | **.9247** |

### 逐类 F1 热力图

| 算法 | BARBUNYA | BOMBAY | CALI | DERMASON | HOROZ | SEKER | SIRA |
|------|:------:|:------:|:----:|:------:|:-----:|:-----:|:----:|
| KNN | .914 | .986 | .916 | .911 | .957 | .954 | .875 |
| SVM | .914 | .986 | .910 | .890 | .947 | .937 | .836 |
| MLP | **.923** | .986 | **.929** | **.921** | **.961** | .949 | **.880** |
| XGB  | **.928** | **1.000** | **.934** | .907 | .959 | .944 | .878 |

### XGBoost Top-5 特征重要性

| 排名 | 特征 | 重要性 |
|:---:|------|:-----:|
| 🥇 | **ShapeFactor3** | 0.2400 |
| 🥈 | EquivDiameter | 0.1307 |
| 🥉 | Compactness | 0.1185 |
| 4 | ConvexArea | 0.1094 |
| 5 | Area | 0.1030 |

---

## 🖼️ 图表展示

运行后自动生成 **14 张图表**：

| # | 图表 | 文件名 | 分析维度 |
|:--:|------|--------|----------|
| 1 | 相关性热力图 | `corr_heatmap.png` | 数据分析 |
| 2 | 类别分布 | `class_distribution.png` | 不平衡检测 |
| 3 | Loss 曲线 ×3 | `loss_curves.png` | 收敛分析 |
| 4 | 混淆矩阵 ×4 | `confusion_matrices.png` | 误差分布 |
| 5 | 精度柱状图 | `precision_comparison.png` | 算法排名 |
| 6 | 推理速度 | `inference_speed.png` | 部署可行性 |
| 7 | 过拟合 Δ | `overfitting_delta.png` | 泛化能力 |
| 8 | 训练时间 | `train_time.png` | 计算成本 |
| 9 | 模型复杂度 | `model_size.png` | 存储成本 |
| 10 | 雷达图 | `radar_comparison.png` | 多维归一化 |
| 11 | 逐类 F1 | `per_class_f1.png` | 细粒度诊断 |
| 12 | XGBoost 重要性 | `xgb_feature_importance.png` | 可解释性 |
| 13 | KNN k 值敏感度 | `knn_k_sensitivity.png` | 超参影响 |
| 14 | 鲁棒性曲线 | `robustness_curves.png` | 噪声容忍度 |

---

## 📊 对比维度矩阵

本项目的分析维度超越评分要求的最低标准：

| 维度 | 评分要求 | 本项目实际 | 额外 |
|------|:------:|--------|:---:|
| 数据集划分 | 未指定 | 教师预划分+固化 | — |
| 算法数量 | ≥3 (含1未学) | **4 (含1未学)** | — |
| 精度对比 | ✅ | Accuracy + Macro F1 + **逐类F1** | ✅ |
| Loss 曲线 | ✅ | **3条** (SVM/MLP/XGB) | — |
| 推理速度 | ✅ | ms/样本 (log scale) | — |
| 鲁棒性 | ✅ | 3噪声×3强度×4算法 = **36组** | — |
| 过拟合 | ✅ | Δ 柱状图 + **KNN k扫描** | ✅ |
| **训练时间** | ❌ | 秒级 (4算法对比) | ✅ |
| **模型大小** | ❌ | KB + 参数量 (4算法) | ✅ |
| **特征重要性** | ❌ | XGBoost gain-based (Top-16) | ✅ |
| **雷达图** | ❌ | 6维归一化 (4算法轮廓) | ✅ |
| 图表数量 | ≥3 | **14张** | — |
| 论文 | 5% | 论文级 README + **完整 paper.md** | ✅ |
| GitHub | 30% | 表格式排版 + 徽章 + 代码块 | — |

---

## 🔬 核心发现

1. 🥇 **MLP 精度最高** (92.73%) — 神经网络的非线性拟合能力在 9,527 样本的中等规模上超越树模型
2. 🛡️ **XGBoost 鲁棒性最强** — 全噪声类型下精度损失 <0.008，特征缺失 30% 仍维持 0.9244
3. ⚡ **SVM 推理最快** (0.0003ms/样本) — 仅需 119 个参数，适合边缘部署
4. 💥 **标签噪声对 SVM 致命** — 20% 标签翻转下精度暴跌至 0.7541 (↓16.5%)
5. 📏 **ShapeFactor3 是最具判别力特征** — 重要性 0.24，且与其余特征相关性极低
6. 🎯 **BOMBAY 类 100% 可识别** — XGBoost 对该类 F1=1.0，特征边界最清晰

---

## 🔒 可复现性

| 措施 | 实现 |
|------|------|
| 随机种子 | 全局固定 `42` |
| 数据划分 | 教师提供 CSV (不自行拆分) |
| Scaler 防泄漏 | `fit(train)` → `transform(test)` |
| 噪声独立 Scaler | `fit(X_train_noisy)` (不复用干净scaler) |
| 噪声隔离 | 局部 `RandomState(42)` |
| 依赖版本 | requirements.txt (min 版本) |

> 论文 `paper.md` 中的全部图表和表格均可通过 `python main.py --config configs/baseline.yaml` 复现。

---

<div align="center">

**[📝 课程论文](paper.md)** &nbsp;|&nbsp; **[🚀 一键运行](#-快速开始)** &nbsp;|&nbsp; **[📦 GitHub](https://github.com/<username>/dry-bean-classification)**

*Made with ❤️ for Machine Learning Course Project*

</div>
