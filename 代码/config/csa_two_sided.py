"""
双侧 CSA（Two-sided Conformalized Survival Analysis）
按删失状态分类：未删失样本获得双侧区间 [L, U]，删失样本获得单侧区间 [L, ∞)。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .csa_base import csa_nonconformity_scores, calibrate_csa_quantile
from .models import predict_median_survival


def train_censoring_classifier(X_train, event_train, X_cal, event_cal):
    """
    训练删失状态分类器：预测P(Δ=1|X)

    该分类器用于识别哪些样本的生存时间被完全观测（未删失）。
    在两侧CSA中，未删失样本（Δ=1）可以获得两侧区间，删失样本(Δ=0)仅获得单侧。

    参数:
        X_train: numpy.ndarray, 训练集特征
        event_train: numpy.ndarray, 训练集事件指示（0=删失，1=事件）
        X_cal: numpy.ndarray, 校准集特征
        event_cal: numpy.ndarray, 校准集事件指示

    返回:
        classifier: 已拟合的分类器对象
        cal_probs: 校准集上Δ=1的预测概率，shape=(n_cal,)
        cal_scores: 校准集上的非一致性分数 ν(x,0)=1-π₀(x)，用于p-value计算
    """
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=2026)
    clf.fit(X_train, event_train)

    # 获取校准集上的预测概率
    cal_probs = clf.predict_proba(X_cal)  # shape = (n_cal, 2)，列对应[Δ=0, Δ=1]

    # 计算非一致性分数（Conformal中的关键量）
    # ν(x, 0) = 1 - π₀(x)，即在零假设Δ=0下的"异常程度"
    cal_scores = 1 - cal_probs[:, 0]  # P(Δ=1|X) = 1 - P(Δ=0|X)

    return clf, cal_probs[:, 1], cal_scores


def compute_upper_bounds_two_sided(model, X, q_value, model_type='cox'):
    """
    计算两侧CSA中未删失样本的上界

    上界通过反演生存函数得到：UPB = F̂⁻¹(0.5 + q)
    如果所求分位数超出支撑，则返回 np.inf

    参数:
        model: 已拟合的生存模型
        X: numpy.ndarray, 特征矩阵 (n_samples, n_features)
        q_value: float, 校准的置信区间半宽
        model_type: str, 模型类型

    返回:
        uppers: numpy.ndarray, 上界值，有限或np.inf，shape=(n_samples,)
    """
    n_samples = X.shape[0]
    uppers = np.full(n_samples, np.inf, dtype=float)

    if model_type == 'km':
        if 0.5 + q_value <= 0.99:
            try:
                times = model.survival_function_.index.values
                survs = model.survival_function_.values.ravel()
                idx = np.where(survs <= 0.5 - q_value)[0]
                if len(idx) > 0:
                    uppers[:] = times[idx[0]]
            except:
                pass

    elif model_type in ['cox', 'weibull']:
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        try:
            surv = model.predict_survival_function(df)
            for i in range(n_samples):
                sf = surv.iloc[:, i] if surv.shape[1] > 1 else surv.iloc[:, 0]
                times = sf.index.values
                values = sf.values
                if np.any(values <= 0.5 - q_value):
                    idx = np.where(values <= 0.5 - q_value)[0]
                    uppers[i] = times[idx[0]]
        except:
            pass

    elif model_type == 'rsf':
        try:
            surv_funcs = model.predict_survival_function(X)
            times = np.asarray(model.unique_times_)
            for i, sf in enumerate(surv_funcs):
                values = np.asarray(sf(times))
                if np.any(values <= 0.5 - q_value):
                    idx = np.where(values <= 0.5 - q_value)[0]
                    uppers[i] = times[idx[0]]
        except:
            pass

    return uppers


def classify_censoring_status_conformal(cal_scores, X_test, clf, alpha=0.1):
    """
    使用Conformal p-value进行删失状态分类

    参数:
        cal_scores: numpy.ndarray, 校准集的非一致性分数 ν(x,0)，shape=(n_cal,)
        X_test: numpy.ndarray, 测试集特征，shape=(n_test, n_features)
        clf: 已拟合的分类器
        alpha: float, 显著性水平，分类阈值为 α/2

    返回:
        pred_delta: numpy.ndarray, 预测的删失指示，1=未删失，0=删失，shape=(n_test,)
        test_pvalues: numpy.ndarray, 每个测试样本的p-value，shape=(n_test,)
    """
    # Step 1: 计算测试集的非一致性分数
    test_probs = clf.predict_proba(X_test)  # shape = (n_test, 2)
    test_scores = 1 - test_probs[:, 0]  # ν(X_test, 0) = 1 - π₀(X_test)

    # Step 2: 对每个测试样本计算p-value
    # p_i = (1 + #{校准样本的分数 ≥ 测试样本分数}) / (n_cal + 1)
    n_cal = len(cal_scores)
    test_pvalues = np.zeros(len(test_scores))

    for i, score_i in enumerate(test_scores):
        count = np.sum(cal_scores >= score_i)
        test_pvalues[i] = (count + 1.0) / (n_cal + 1.0)

    # Step 3: 基于p-value进行分类
    # 当 p-value < α/2 时，拒绝H₀（Δ=0），判为H₁（Δ=1，未删失）
    alpha_half = alpha / 2.0
    pred_delta = (test_pvalues < alpha_half).astype(int)

    return pred_delta, test_pvalues


def fit_csa_intervals_two_sided(model, X_train, time_train, event_train,
                                 X_cal, time_cal, event_cal, X_test,
                                 alpha=0.1, model_type='cox'):
    """
    两侧CSA区间构建：混合区间方法

    参数:
        model: 已拟合模型
        X_train, time_train, event_train: 训练集数据（用于分类器）
        X_cal, time_cal, event_cal: 校准集数据
        X_test: 测试集特征 (n_test, n_features)
        alpha: float，总体显著性水平，自动分成两部分各α/2使用
        model_type: str，模型类型

    返回:
        lower: numpy.ndarray, 下界，shape=(n_test,)
        upper: numpy.ndarray, 上界（有限或np.inf），shape=(n_test,)
        q_value: float, 使用的分位数
        classification: dict, 包含分类信息
            - 'pred_delta': 预测的Δ值（0或1，使用conformal p-value）
            - 'pvalues': Conformal p-value值
            - 'two_sided_mask': 获得两侧区间的样本掩码
    """
    # Step 1: 训练删失状态分类器
    clf, cal_probs, cal_scores = train_censoring_classifier(X_train, event_train, X_cal, event_cal)

    # Step 2: 计算非一致性分数
    pred_cal = predict_median_survival(model, X_cal, model_type=model_type)
    scores = csa_nonconformity_scores(pred_cal, time_cal, event_cal)

    # Step 3: 分别校准上下界的分位数
    alpha_half = alpha / 2.0
    q_lower = calibrate_csa_quantile(scores, alpha=alpha_half)
    q_upper = q_lower

    # Step 4: 使用Conformal p-value进行测试集分类
    pred_delta, test_pvalues = classify_censoring_status_conformal(
        cal_scores, X_test, clf, alpha=alpha
    )

    # Step 5: 在测试集上生成区间
    pred_test = predict_median_survival(model, X_test, model_type=model_type)
    lower = np.clip(pred_test - q_lower, 0.0, None)
    upper = np.full_like(lower, np.inf)

    # 对于预测为未删失的样本，尝试计算有限上界
    two_sided_mask = pred_delta == 1
    if np.any(two_sided_mask):
        X_two_sided = X_test[two_sided_mask]
        uppers_two = compute_upper_bounds_two_sided(model, X_two_sided, q_upper, model_type)
        upper[two_sided_mask] = uppers_two

    classification = {
        'pred_delta': pred_delta,
        'pvalues': test_pvalues,
        'two_sided_mask': two_sided_mask
    }

    return lower, upper, q_lower, classification
