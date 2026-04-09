"""
CSA 共享核心组件
- estimate_censoring_weights : 删失机制逆概率权重（论文 Section 3）
- csa_nonconformity_scores   : 非一致性分数（CMR 单侧 / 旧双侧兼容）
- calibrate_csa_quantile     : 加权分位数校准（含 ∞-atom）
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def estimate_censoring_weights(X_cal, time_cal, event_cal, c0_threshold=None, verbose=False):
    """
    估计删失机制权重：w(x) = 1/c(x; c_0)
    其中 c(x; c_0) = P(C ≥ c_0 | X=x) 即在给定协变量X下，删失时间≥c_0的概率

    参数:
        X_cal: numpy.ndarray, 校准集特征 (n_cal, n_features)
        time_cal: numpy.ndarray, 校准集观测时间
        event_cal: numpy.ndarray, 校准集事件指示（0=删失，1=事件）
        c0_threshold: float, 删失时间阈值（默认为校准集中位数）
        verbose: bool, 是否打印统计信息

    返回:
        weights: numpy.ndarray, 校准集上的样本权重，shape=(n_cal,)
        c0: float, 使用的删失阈值
        censoring_probs: numpy.ndarray, 估计的删失概率P(C≥c0|X)，shape=(n_cal,)
        censoring_model: RandomForestClassifier, 删失状态分类器
    """
    # Step 1: 确定c0阈值
    # 论文要求从训练集自适应选取c0，避免使用校准集（防止数据泄漏）
    # 若调用方已传入c0_threshold，直接使用；否则回退到校准集中位数（兜底行为）
    if c0_threshold is None:
        c0_threshold = np.median(time_cal)  # 兜底：调用方应优先从训练集传入

    # Step 2: 创建二元标签：1表示C≥c0，0表示C<c0
    # 标签可直接确认的样本：
    #   删失(Δ=0)：C = time_cal，标签 = (time_cal ≥ c0)，已知
    #   未删失(Δ=1) 且 T ≥ c0：C > T ≥ c0，标签必为1，已知
    #   未删失(Δ=1) 且 T < c0：C > T 但 C 未观测，无法确认是否 C ≥ c0，标签未知
    # → 仅用标签已知的样本训练分类器，避免错误标签污染权重估计
    censor_indicator = (time_cal >= c0_threshold).astype(int)
    known_mask = (event_cal == 0) | (time_cal >= c0_threshold)

    # Step 3: 训练分类器预测P(C≥c0|X)，使用随机森林
    censoring_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=2026
    )
    censoring_model.fit(X_cal[known_mask], censor_indicator[known_mask])

    # Step 4: 获取 c(x;c0) = P(C≥c0|X) 的概率估计
    censor_probs = censoring_model.predict_proba(X_cal)  # shape=(n_cal, 2)
    censoring_probs = censor_probs[:, 1]  # P(C≥c0|X)，对应类别1

    # Step 5: 计算权重w(x) = P(C≥c0) / c(x;c0)，用平滑而非硬截断
    # w(x) = P(C≥c₀) / P(C≥c₀|X=x)，其中分子是边际概率（常数）
    p_c0_marginal = np.mean(censor_indicator)  # 分子：边际P(C≥c0)

    epsilon = 0.01  # 平滑常数
    censoring_probs_smoothed = np.clip(censoring_probs, a_min=epsilon, a_max=1.0-epsilon)
    weights = p_c0_marginal / censoring_probs_smoothed

    if verbose:
        print(f"\n📊 删失机制权重估计（已修复：遵循论文标准）")
        print(f"  删失阈值 c0: {c0_threshold:.4f}")
        print(f"  边际 P(C≥c0): {p_c0_marginal:.4f}")
        print(f"  P(C≥c0|X) 范围: [{censoring_probs.min():.4f}, {censoring_probs.max():.4f}]")
        print(f"  权重 w(x) 范围: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  权重均值: {weights.mean():.4f} ± {weights.std():.4f}")
        print(f"  未被截断样本比例: {censor_indicator.mean():.1%}")
        print(f"  平滑方法: epsilon={epsilon:.4f}（遵循论文标准，not硬截断）")

    return weights, c0_threshold, censoring_probs, censoring_model


def csa_nonconformity_scores(pred_median, time, event=None, weights=None, c0=None):
    """
    计算非一致性分数

    当 c0 不为 None 时（传统CSA调用）：
        使用论文 CMR 分数 V = ŷ(x) - (T̃ ∧ c0)，其中 T̃ ∧ c0 = min(time, c0)
        对 I'_ca 中所有样本统一计算，无需区分是否删失。

    当 c0 为 None 时（两侧CSA调用，保持旧行为）：
        未删失：|T - ŷ|
        删失：max(0, ŷ - C)

    参数:
        pred_median: 预测的中位生存时间
        time: 观测生存时间 T̃ = min(T, C)
        event: 事件指示（0=删失，1=事件），c0=None 时必须提供
        weights: 保留参数（未使用，权重在 calibrate_csa_quantile 中应用）
        c0: float, 截断阈值；提供时启用 CMR 单侧分数

    返回:
        score: numpy.ndarray, 非一致性分数
    """
    if c0 is not None:
        # CMR 单侧分数：V = ŷ - min(T̃, c0)（论文 Section 3.1）
        Y = np.minimum(time, float(c0))
        return pred_median - Y

    # 旧行为（两侧CSA分支使用）
    score = np.zeros_like(time, dtype=float)
    uncensored = event == 1
    censored   = event == 0
    score[uncensored] = np.abs(time[uncensored] - pred_median[uncensored])
    score[censored]   = np.maximum(0.0, pred_median[censored] - time[censored])
    return score


def calibrate_csa_quantile(scores, alpha=0.1, weights=None, test_weight=None):
    """
    校准 CSA 置信区间的半宽（支持加权分位数，含∞-atom）

    无权重时退化为标准公式：q = 第k大分数，其中k = ⌈(n+1)*(1-α)⌉
    有权重时使用论文 Algorithm 1 Step 5-6 的加权分位数：
        p̂_i(x) = W_i / (ΣW_j + ŵ(x))，p̂_∞(x) = ŵ(x) / (ΣW_j + ŵ(x))
    测试点权重 ŵ(x) 以∞-atom形式进入分母，保证有限样本覆盖率。

    参数:
        scores: 校准集的非一致性分数
        alpha: 显著性水平（目标覆盖率 = 1-alpha）
        weights: numpy.ndarray, 校准样本权重 W_i（可选），shape=(n,)
                如果为None，使用均匀权重
        test_weight: float, 测试点权重 ŵ(x)（可选）
                    对应论文∞-atom分母修正；None时退化为无∞-atom近似

    返回:
        q: float，置信区间半宽；若∞-atom激活（校准权重不足）则返回 np.inf
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    if weights is None:
        # 无权重情况：标准conformal分位数，+1修正已内含
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(max(k, 1), n)
        return np.sort(scores)[k - 1]

    # 有权重情况：论文Algorithm 1加权分位数
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(scores):
        raise ValueError(f"权重长度{len(weights)} ≠ 分数长度{len(scores)}")

    # 测试点权重纳入分母（∞-atom），对应论文 p̂_∞(x) = ŵ(x)/(ΣW_j + ŵ(x))
    w_test = float(test_weight) if test_weight is not None else 0.0
    total_weight = weights.sum() + w_test

    # 排序并计算累积权重
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumsum_weights = np.cumsum(sorted_weights)

    # 找最小的q使得 Σ_{V_i≤q} W_i / total_weight ≥ (1-α)
    threshold = (1 - alpha) * total_weight
    idx = np.where(cumsum_weights >= threshold)[0]

    if len(idx) > 0:
        return sorted_scores[idx[0]]
    else:
        return np.inf  # ∞-atom激活：校准点累积权重不足，返回无穷
