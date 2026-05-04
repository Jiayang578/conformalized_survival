"""
传统 CSA（Conformalized Survival Analysis）
"""
import numpy as np

from .csa_base import (
    estimate_c0_on_train,
    estimate_censoring_weights,
    should_fallback_to_unweighted,
    should_use_constant_censoring_weights,
    csa_nonconformity_scores,
    calibrate_csa_quantile
)
from .models import predict_mean_survival_truncated


def fit_csa_intervals_traditional(model, X_train, time_train, event_train,
                                   X_cal, time_cal, event_cal, X_test,
                                   alpha=0.1, model_type='cox', use_weights=True,
                                   cens_time_train=None, cens_time_cal=None,
                                   min_p_c0=0.3, max_p_c0=0.85,
                                   verbose=True,
                                   known_censoring_independent=None,
                                   adaptive_c0=True):
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
        use_weights: bool, 是否启用逆概率删失加权；
                     False 时仍保留 c0 选择与 I'_ca 筛选，只是令 w(x)=1
        cens_time_train: numpy.ndarray or None, 训练集删失时间 C
        cens_time_cal:   numpy.ndarray or None, 校准集删失时间 C
        min_p_c0: float, c0选择时要求的 P(C≥c₀) 最低阈值（默认0.3）；
                  合成数据（cens_time_train不为None）路径下该约束也会生效
        verbose: bool, 是否打印诊断信息
        known_censoring_independent: bool or None, 是否已知删失机制与 X 独立。
                     None 时，若提供了真实删失时间 cens_time_train，则默认视为
                     当前合成数据设置中的外生删失并直接使用常数权重 w(x)=1。
        adaptive_c0: bool, 是否使用训练内网格搜索自适应选择 c0

    返回:
        lower: numpy.ndarray, 区间下界，shape=(n_test,)
        upper: numpy.ndarray, 所有元素为 np.inf（表示无上界）
        q_value: float, 校准的平均分位数值
        weight_info: dict, 诊断信息字典
    """
    def _finalize_lower_bounds(pred_test_values, q_per_test_values, label):
        """统一处理 q 的诊断汇总与下界构造。"""
        raw_lower = pred_test_values - q_per_test_values
        raw_lower[~np.isfinite(raw_lower)] = 0.0
        lower_bounds = np.minimum(raw_lower, c0)
        lower_bounds = np.maximum(lower_bounds, 0.0)

        finite_q = q_per_test_values[np.isfinite(q_per_test_values)]
        q_summary = float(np.mean(finite_q)) if finite_q.size else np.inf

        if verbose:
            n_inf = int(np.sum(~np.isfinite(q_per_test_values)))
            q_str = f"{q_summary:.4f}" if np.isfinite(q_summary) else "inf"
            print(f"\n✓ {label}分位数 q（均值）= {q_str}，∞-atom激活 {n_inf}/{len(q_per_test_values)} 个测试点")

        return lower_bounds, q_summary

    # Step 1: c0 自适应选择（Candès et al. Section 3.4）
    # 在训练折内部再切 inner/holdout 选择 c0，保证 c0 独立于最终 calibration fold Z_ca。
    if adaptive_c0:
        c0, c0_scores = estimate_c0_on_train(
            X_train, time_train, event_train, model,
            model_type=model_type,
            cens_time_train=cens_time_train,
            min_p_c0=min_p_c0, max_p_c0=max_p_c0, alpha=alpha, verbose=verbose
        )
    else:
        c0 = float(np.median(time_train))
        c0_scores = None
        if verbose:
            print(f"  固定 c0 模式：c0 = 训练集中位数 {c0:.4f}")

    # Step 2: 从训练集估计边际概率 P(C≥c0)
    if cens_time_train is not None:
        # 精确 C：直接用真实删失时间
        p_c0_marginal = float(np.mean(cens_time_train >= c0))
    else:
        # 近似：用观测时间代理（未删失 T≥c0 保证 C>T≥c0，删失 C=T̃）
        p_c0_marginal = float(np.mean(time_train >= c0))

    if verbose:
        print(f"\n  边际概率 P(C≥c0) = {p_c0_marginal:.4f}（来自训练集）")

    if known_censoring_independent is None:
        # 默认仅在“可观测真实删失时间 + 经验上符合单一外生删失分布”的情形
        # 才跳过条件删失模型估计，避免把所有精确 C 场景都误判为独立删失。
        known_censoring_independent = should_use_constant_censoring_weights(
            cens_time_train, p_c0_marginal
        )

    # Step 3: 估计删失概率模型 P(C≥c0|X)，得到权重
    weight_fallback = False
    fallback_reason = None
    weight_strategy = 'disabled'

    if use_weights:
        if known_censoring_independent:
            weights = np.ones(len(X_cal), dtype=float)
            censoring_probs = np.full(len(X_cal), p_c0_marginal, dtype=float)
            censor_model = None
            weight_strategy = 'constant_one'
            if verbose:
                print("\n  已知删失机制与 X 独立：跳过删失模型，直接使用常数权重 w(x)=1")
        else:
            weights, censoring_probs, censor_model = estimate_censoring_weights(
                X_train, time_train, event_train,
                X_cal,
                c0_threshold=c0,
                p_c0_marginal=p_c0_marginal,
                cens_time_train=cens_time_train,
                verbose=verbose
            )
            weight_strategy = 'estimated'

            if should_fallback_to_unweighted(weights, censoring_probs, p_c0_marginal):
                weight_fallback = True
                fallback_reason = 'extreme_or_unstable_weights'
                weights = None
                censoring_probs = None
                censor_model = None
                weight_strategy = 'fallback_unweighted'
                if verbose:
                    print("\n  权重估计不稳定，自动回退到无权重 conformal 校准")
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

    if use_weights and weight_strategy == 'estimated' and (weights is not None):
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

        lower, q_value = _finalize_lower_bounds(pred_test, q_per_test, label='加权')

    elif use_weights and weight_strategy == 'constant_one':
        test_weights = np.ones(X_test.shape[0], dtype=float)
        q_per_test = calibrate_csa_quantile(
            scores, alpha=alpha, weights=weights_prime, test_weight=test_weights)
        lower, q_value = _finalize_lower_bounds(pred_test, q_per_test, label='常数权重')
    else:
        test_weights = np.ones(X_test.shape[0])
        q_per_test = calibrate_csa_quantile(
            scores, alpha=alpha, weights=None, test_weight=test_weights)

        lower, q_value = _finalize_lower_bounds(pred_test, q_per_test, label='无权重')

    # Step 7: 构建区间（传统CSA都是单侧）
    upper = np.full_like(lower, np.inf)

    # 返回结果和诊断信息
    weight_info = {
        'use_weights': use_weights,
        'adaptive_c0': adaptive_c0,
        'known_censoring_independent': known_censoring_independent,
        'weight_strategy': weight_strategy,
        'c0': c0,
        'c0_scores': c0_scores,
        'p_c0_marginal': p_c0_marginal,
        'weights': weights,
        'censoring_probs': censoring_probs,
        'censor_model': censor_model,
        'weight_fallback': weight_fallback,
        'fallback_reason': fallback_reason,
        'cal_mask': cal_mask,
        'test_weights': test_weights,
        'q_per_test': q_per_test,
    }

    return lower, upper, q_value, weight_info
