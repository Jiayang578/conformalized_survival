"""
传统 CSA（Conformalized Survival Analysis）
所有测试样本生成单侧区间 [L, ∞)，使用逆概率加权 conformal 推断。
"""
import numpy as np

from .csa_base import estimate_censoring_weights, csa_nonconformity_scores, calibrate_csa_quantile
from .models import predict_median_survival


def fit_csa_intervals_traditional(model, X_cal, time_cal, event_cal, X_test,
                                   alpha=0.1, model_type='cox', use_weights=True,
                                   time_train=None, verbose=True):
    """
    传统CSA区间构建：所有样本生成单侧区间 [L, ∞)（支持加权）

    参数:
        model: 已拟合模型对象
        X_cal: numpy.ndarray, 校准集特征 (n_cal, n_features)
        time_cal: numpy.ndarray, 校准集生存时间
        event_cal: numpy.ndarray, 校准集事件指示（0=删失，1=事件）
        X_test: numpy.ndarray, 测试集特征 (n_test, n_features)
        alpha: float, 显著性水平，目标覆盖率 = 1-alpha
        model_type: str, 模型类型 ('km', 'cox', 'weibull', 'rsf')
        use_weights: bool, 是否使用加权conformal推断（推荐True）
        time_train: numpy.ndarray or None, 训练集观测时间，用于自适应选取c0（论文要求）；
                    若为None则回退到校准集中位数
        verbose: bool, 是否打印诊断信息

    返回:
        lower: numpy.ndarray, 区间下界，shape=(n_test,)
        upper: numpy.ndarray, 所有元素为 np.inf（表示无上界）
        q_value: float, 校准的分位数值
        weight_info: dict, 权重信息（当use_weights=True时）
    """
    # Step 1: 在完整校准集上估计删失机制权重（训练分类器）
    # c0 从训练集自适应选取（论文要求，防止校准集数据泄漏）；无训练数据则回退到校准集中位数
    if use_weights:
        if time_train is not None:
            c0_from_train = float(np.median(time_train))
            if verbose:
                print(f"📌 自适应 c0：从训练集中位数选取 c0 = {c0_from_train:.4f}")
        else:
            c0_from_train = None  # 回退到 estimate_censoring_weights 内部的校准集中位数
        weights, c0, censoring_probs, censor_model = estimate_censoring_weights(
            X_cal, time_cal, event_cal, c0_threshold=c0_from_train, verbose=verbose
        )
    else:
        weights = None
        censoring_probs = None
        c0 = None
        censor_model = None
        if verbose:
            print("⚠️  使用无权重CSA（假设完全外生删失）")

    # Step 2: 筛选校准子集 I'_ca（论文核心要求）
    # 正确定义（见论文 Section 3）：
    #   未删失(Δ=1)：T̃ = T 已完整观测，Y = T ∧ c0 始终可计算，全部纳入
    #   删失(Δ=0)：T̃ = C，仅当 C ≥ c0 时 Y = c0 可从观测时间恢复，否则排除
    # 合并：I'_ca = {Δ=1} ∪ {Δ=0, C ≥ c0}
    if use_weights and c0 is not None:
        cal_mask = (event_cal == 1) | (time_cal >= c0)
        if verbose:
            n_unc = int(np.sum(event_cal == 1))
            n_cens = int(np.sum((event_cal == 0) & (time_cal >= c0)))
            print(f"\n📌 I'_ca 筛选：{cal_mask.sum()}/{len(time_cal)} 个样本"
                  f"（未删失 {n_unc} + 删失且C≥c0 {n_cens}）")
        X_cal_prime     = X_cal[cal_mask]
        time_cal_prime  = time_cal[cal_mask]
        event_cal_prime = event_cal[cal_mask]
        weights_prime   = weights[cal_mask]
    else:
        cal_mask        = np.ones(len(time_cal), dtype=bool)
        X_cal_prime     = X_cal
        time_cal_prime  = time_cal
        event_cal_prime = event_cal
        weights_prime   = weights

    # Step 3: 计算 I'_ca 上的 CMR 非一致性分数 V = ŷ - min(T̃, c0)
    # 始终传入 event，c0 不为 None 时走 CMR 路径（event 被忽略），
    # c0 为 None（无权重分支）时走旧行为路径（event 被使用）
    pred_cal = predict_median_survival(model, X_cal_prime, model_type=model_type)
    scores = csa_nonconformity_scores(pred_cal, time_cal_prime, event=event_cal_prime, c0=c0)

    # Step 4: 计算测试集预测值及各测试点的权重 ŵ(x_test)
    pred_test = predict_median_survival(model, X_test, model_type=model_type)

    if use_weights:
        # 测试点权重与校准点权重同公式：ŵ(x) = P(C≥c0) / P(C≥c0|X=x)
        p_c0_marginal = np.mean(time_cal >= c0)
        epsilon = 0.01
        test_censor_probs = censor_model.predict_proba(X_test)[:, 1]
        test_censor_probs = np.clip(test_censor_probs, epsilon, 1.0 - epsilon)
        test_weights = p_c0_marginal / test_censor_probs  # shape=(n_test,)

        # 预计算校准集的排序和累积权重（向量化，避免n_test次重复排序）
        sorted_idx      = np.argsort(scores)
        sorted_scores_v = scores[sorted_idx]
        cumsum_cal      = np.cumsum(weights_prime[sorted_idx])
        W_cal           = cumsum_cal[-1]

        # 每个测试点的分位数阈值（∞-atom分母含 ŵ(x_test)）
        thresholds = (1 - alpha) * (W_cal + test_weights)   # shape=(n_test,)

        # 二分查找：第一个累积权重 ≥ threshold 的位置
        pos      = np.searchsorted(cumsum_cal, thresholds, side='left')  # shape=(n_test,)
        n_prime  = len(sorted_scores_v)
        q_per_test = np.where(
            pos < n_prime,
            sorted_scores_v[np.minimum(pos, n_prime - 1)],
            np.inf   # ∞-atom激活
        )

        # 计算下界：L̂(x) = (ŷ - η) ∧ c0，clip到[0, c0]（论文公式）
        # ∞-atom激活时 q_i=inf，pred - inf = -inf，clip后为0
        raw_lower = pred_test - q_per_test
        raw_lower[~np.isfinite(raw_lower)] = 0.0
        lower = np.clip(raw_lower, 0.0, c0)

        q_value = float(np.mean(q_per_test[np.isfinite(q_per_test)])) \
                  if np.any(np.isfinite(q_per_test)) else np.inf
        if verbose:
            n_inf = int(np.sum(~np.isfinite(q_per_test)))
            print(f"\n✓ 校准分位数 q（测试点均值）= {q_value:.4f}，∞-atom激活：{n_inf}/{len(q_per_test)}")
    else:
        q_value    = calibrate_csa_quantile(scores, alpha=alpha)
        lower      = np.clip(pred_test - q_value, 0.0, None)  # 无权重时无c0截断
        q_per_test = None
        if verbose:
            print(f"\n✓ 校准分位数 q = {q_value:.4f}")

    # Step 5: 传统CSA中所有样本都没有有限上界
    upper = np.full_like(lower, np.inf)

    weight_info = {
        'use_weights': use_weights,
        'weights': weights,
        'censoring_probs': censoring_probs,
        'c0_threshold': c0,
        'censor_model': censor_model,
        'cal_mask': cal_mask,
        'q_per_test': q_per_test,
    }

    return lower, upper, q_value, weight_info
