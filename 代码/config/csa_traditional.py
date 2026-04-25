"""
传统 CSA（Conformalized Survival Analysis）
"""
import numpy as np

from .csa_base import (
    estimate_c0_on_train,
    estimate_censoring_weights,
    csa_nonconformity_scores,
    calibrate_csa_quantile
)
from .models import predict_mean_survival_truncated


def fit_csa_intervals_traditional(model, X_train, time_train, event_train,
                                   X_cal, time_cal, event_cal, X_test,
                                   alpha=0.1, model_type='cox', use_weights=True,
                                   cens_time_train=None, cens_time_cal=None,
                                   min_p_c0=0.3, max_p_c0=0.85,
                                   verbose=True):
    """
    生成单侧区间 [L, ∞)。
    参数:
        model: 已拟合模型对象
        X_train: numpy.ndarray, 训练集特征 (n_train, n_features) - 用于c0选择
        time_train: numpy.ndarray, 训练集观测时间 - 必须提供
        event_train: numpy.ndarray, 训练集事件指示
        X_cal: numpy.ndarray, 校准集特征 (n_cal, n_features)
        time_cal: numpy.ndarray, 校准集生存时间
        event_cal: numpy.ndarray, 校准集事件指示（0=删失，1=事件）
        X_test: numpy.ndarray, 测试集特征 (n_test, n_features)
        alpha: float, 显著性水平，目标覆盖率 = 1-alpha
        model_type: str, 模型类型 ('km', 'cox', 'weibull', 'rsf')
        use_weights: bool, 是否使用加权conformal推断（推荐True）
        cens_time_train: numpy.ndarray or None, 训练集删失时间 C
        cens_time_cal:   numpy.ndarray or None, 校准集删失时间 C
        min_p_c0: float, c0选择时要求的 P(C≥c₀) 最低阈值（默认0.3）；
                  合成数据（cens_time_train不为None）路径下该约束也会生效
        verbose: bool, 是否打印诊断信息

    返回:
        lower: numpy.ndarray, 区间下界，shape=(n_test,)
        upper: numpy.ndarray, 所有元素为 np.inf（表示无上界）
        q_value: float, 校准的平均分位数值
        weight_info: dict, 诊断信息字典
    """

    # Step 1: c0 自适应选择（Candès et al. Section 3.4 替代方案）
    # ĉ0 = argmax_{c0} (1/|Z_ca|) Σ L̂_{c0}(X_i)，用 Z_ca 一致性分数评估
    # P(C≥c0) 约束仍由训练集估计，避免约束检查引入 double-dipping
    if use_weights:
        c0, c0_scores = estimate_c0_on_train(
            X_train, time_train, event_train, model,
            X_cal, time_cal, event_cal,
            model_type=model_type,
            cens_time_train=cens_time_train,
            cens_time_cal=cens_time_cal,
            min_p_c0=min_p_c0, max_p_c0=max_p_c0, alpha=alpha, verbose=verbose
        )
    else:
        c0 = float(np.median(time_train))
        c0_scores = None
        if verbose:
            print(f"  无权重模式：c0 = 训练集中位数 {c0:.4f}")

    # Step 2: 从训练集估计边际概率 P(C≥c0)
    if cens_time_train is not None:
        # 精确 C：直接用真实删失时间
        p_c0_marginal = float(np.mean(cens_time_train >= c0))
    else:
        # 近似：用观测时间代理（未删失 T≥c0 保证 C>T≥c0，删失 C=T̃）
        p_c0_marginal = float(np.mean(time_train >= c0))

    if verbose:
        print(f"\n  边际概率 P(C≥c0) = {p_c0_marginal:.4f}（来自训练集）")

    # Step 3: 估计删失概率模型 P(C≥c0|X)，得到权重
    if use_weights:
        weights, censoring_probs, censor_model = estimate_censoring_weights(
            X_train, time_train, event_train,
            X_cal,
            c0_threshold=c0,
            p_c0_marginal=p_c0_marginal,
            cens_time_train=cens_time_train,
            verbose=verbose
        )
    else:
        weights = None
        censoring_probs = None
        censor_model = None

    # Step 4: 筛选校准子集 I'_ca
    if cens_time_cal is not None:
        # 精确 C：直接用真实删失时间
        cal_mask = cens_time_cal >= c0
    else:
        # 实际数据近似：由于 C 对未删失样本不可观测，采用激进 mask 作为近似：
        #   未删失（event=1）：C > T，理论上大概率满足 C ≥ c0（c0 = median 附近时尤其如此）
        #   删失且 T̃ ≥ c0：C = T̃ ≥ c0 ✓（严格成立）
        cal_mask = (event_cal == 1) | (time_cal >= c0)

    X_cal_prime    = X_cal[cal_mask]
    time_cal_prime = time_cal[cal_mask]
    weights_prime  = weights[cal_mask] if weights is not None else None

    if verbose:
        n_in_unc  = int(np.sum((event_cal == 1) & cal_mask))
        n_in_cens = int(np.sum((event_cal == 0) & cal_mask))
        mode_str  = "精确C" if cens_time_cal is not None else "近似"
        print(f"\n  I'_ca 筛选（{mode_str}）：{cal_mask.sum()}/{len(time_cal)} 个样本"
              f"（未删失 {n_in_unc} + 删失且C≥c0 {n_in_cens}）")

    # Step 5: 计算校准集非一致性分数 V = ŷ - min(T̃, c0)
    pred_cal_prime = predict_mean_survival_truncated(model, X_cal_prime, c0=c0, model_type=model_type)
    scores = csa_nonconformity_scores(pred_cal_prime, time_cal_prime, c0=c0)

    # Step 6: 计算测试集预测值和权重，校准分位数
    pred_test = predict_mean_survival_truncated(model, X_test, c0=c0, model_type=model_type)

    if use_weights:
        # 测试点权重：ŵ(x) = P(C≥c0) / P(C≥c0|X=x)
        epsilon = 0.01
        _tp = censor_model.predict_proba(X_test)
        if _tp.shape[1] == 1:
            _only = int(censor_model.classes_[0])
            test_censor_probs = np.full(len(X_test), float(_only))
        else:
            test_censor_probs = _tp[:, 1]
        test_censor_probs = np.clip(test_censor_probs, epsilon, 1.0 - epsilon)
        test_weights = p_c0_marginal / test_censor_probs  # shape=(n_test,)

        q_per_test = calibrate_csa_quantile(
            scores, alpha=alpha, weights=weights_prime, test_weight=test_weights)

        # 下界计算：L(x) = (ŷ - q) ∧ c0
        raw_lower = pred_test - q_per_test
        raw_lower[~np.isfinite(raw_lower)] = 0.0  # ∞-atom激活时下界为0
        lower = np.minimum(raw_lower, c0)  # 仅 clip 到上界c0
        lower = np.maximum(lower, 0.0)     # 然后保证非负

        q_value = float(np.nanmean(q_per_test[np.isfinite(q_per_test)]))

        if verbose:
            n_inf = int(np.sum(~np.isfinite(q_per_test)))
            print(f"\n✓ 加权分位数 q（均值）= {q_value:.4f}，∞-atom激活 {n_inf}/{len(q_per_test)} 个测试点")

    else:
        test_weights = np.ones(X_test.shape[0])
        q_per_test = calibrate_csa_quantile(
            scores, alpha=alpha, weights=None, test_weight=test_weights)

        raw_lower = pred_test - q_per_test
        raw_lower[~np.isfinite(raw_lower)] = 0.0
        lower = np.minimum(raw_lower, c0)
        lower = np.maximum(lower, 0.0)

        q_value = float(np.nanmean(q_per_test[np.isfinite(q_per_test)]))

        if verbose:
            n_inf = int(np.sum(~np.isfinite(q_per_test)))
            print(f"\n✓ 无权重分位数 q（均值）= {q_value:.4f}，∞-atom激活 {n_inf}/{len(q_per_test)} 个测试点")

    # Step 7: 构建区间（传统CSA都是单侧）
    upper = np.full_like(lower, np.inf)

    # 返回结果和诊断信息
    weight_info = {
        'use_weights': use_weights,
        'c0': c0,
        'c0_scores': c0_scores,
        'p_c0_marginal': p_c0_marginal,
        'weights': weights,
        'censoring_probs': censoring_probs,
        'censor_model': censor_model,
        'cal_mask': cal_mask,
        'test_weights': test_weights,
        'q_per_test': q_per_test,
    }

    return lower, upper, q_value, weight_info
