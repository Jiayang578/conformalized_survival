"""
config 包：CSA（Conformalized Survival Analysis）实现

子模块职责：
  data.py            – 数据生成（Weibull）与训练/校准/测试集分割
  models.py          – 生存模型拟合（KM/Cox/Weibull/RSF）、评估、中位数预测
  csa_base.py        – CSA 共享核心：删失权重、非一致性分数、加权分位数
  csa_traditional.py – 传统 CSA：单侧区间 [L, ∞)（Candès et al., JRSSB 2023）
  csa_two_sided.py   – 双侧 CSA：混合区间 [L, U] / [L, ∞)
  evaluation.py      – 覆盖率与区间宽度评估

所有符号统一从此处导出，notebooks 中的 `from config import *` 无需改动。
"""
import numpy as np

np.random.seed(2026)

from .data import (
    weibull_from_mu_sigma,
    generate_weibull_data,
    split_survival_data,
)

from .models import (
    fit_kaplan_meier,
    fit_cox,
    fit_weibull,
    fit_rsf,
    evaluate_model,
    predict_median_survival,
    predict_mean_survival_truncated,
)

from .csa_base import (
    estimate_c0_on_train,
    estimate_censoring_weights,
    csa_nonconformity_scores,
    calibrate_csa_quantile,
)

from .csa_traditional import fit_csa_intervals_traditional

from .csa_two_sided import (
    train_censoring_classifier,
    compute_upper_bounds_two_sided,
    classify_censoring_status_conformal,
    fit_csa_intervals_two_sided,
)

from .evaluation import (
    evaluate_interval_coverage_traditional,
    evaluate_interval_coverage_two_sided,
    evaluate_interval_coverage_two_sided_synthetic,
    evaluate_interval_coverage_two_sided_real,
)
