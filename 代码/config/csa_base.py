"""
CSA 共享核心组件
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .models import predict_median_survival, predict_mean_survival_truncated


def estimate_c0_on_train(X_train, time_train, event_train, model, model_type='cox',
                          c0_grid=None, holdout_ratio=0.25, cens_time_train=None,
                          verbose=False):
    """
    c0 自适应选择：网格搜索 + 训练集holdout验证

    参数:
        X_train: numpy.ndarray, 训练集特征 (n_train, n_features)
        time_train: numpy.ndarray, 训练集观测时间
        event_train: numpy.ndarray, 训练集事件指示
        model: 已拟合的生存模型
        model_type: str, 模型类型
        c0_grid: numpy.ndarray or None, c0类值网格；若None则自动生成（从10%到90%分位数）
        holdout_ratio: float, 用于验证的holdout集比例（默认25%）
        cens_time_train: numpy.ndarray or None, 训练集删失时间C
                         提供时用精确 C 定义 I'_ca；否则用近似 time ≥ c0
        verbose: bool, 是否打印详细信息

    返回:
        c0_best: float, 选定的最优c0值
        c0_scores: dict, 网格搜索各候选的评分结果
    """
    from sklearn.model_selection import train_test_split

    # Step 1: 生成c0候选网格
    if c0_grid is None:
        percentiles = np.linspace(10, 90, 17)  # 10%, 15%, ..., 90%
        c0_grid = np.percentile(time_train, percentiles)

    # Step 2: 分割训练集 → c0_fit + c0_holdout
    arrays = [X_train, time_train, event_train]
    if cens_time_train is not None:
        arrays.append(cens_time_train)

    splits = train_test_split(*arrays, test_size=holdout_ratio, random_state=2026,
                               stratify=event_train)

    X_c0_fit,    X_c0_holdout    = splits[0],  splits[1]
    time_c0_fit, time_c0_holdout = splits[2],  splits[3]
    # splits[4]=event_c0_fit, splits[5]=event_c0_holdout（保守 mask 下不用 event）

    if cens_time_train is not None:
        cens_c0_fit = splits[6]  # train_test_split 返回 [train, test, ...] 交替

    c0_scores = {}
    best_score = -np.inf
    c0_best = None

    # Step 3: 对每个c0候选值，计算holdout集上的平均下界LPB
    for c0_cand in c0_grid:
        # I'_ca 定义（与 fit_csa_intervals_traditional 保持一致）：
        # 有精确 C：直接用 C_i ≥ c0
        # 无精确 C：未删失全部保留 + 删失且 T̃ ≥ c0
        #   理由：未删失样本 C > T，C 已知时几乎全部满足 C ≥ c0；
        #         此近似使分数含正值，目标函数可对 c0 做有意义的权衡
        if cens_time_train is not None:
            cal_mask = cens_c0_fit >= c0_cand
        else:
            # 合成数据但 C 未传入：保守近似
            cal_mask = time_c0_fit >= c0_cand

        if np.sum(cal_mask) < 2:
            c0_scores[float(c0_cand)] = np.nan
            continue

        # 计算非一致性分数（截断条件均值 E[T∧c0|X]）
        pred_fit = predict_mean_survival_truncated(model, X_c0_fit[cal_mask], c0=c0_cand, model_type=model_type)
        time_fit_prime = time_c0_fit[cal_mask]
        scores = pred_fit - np.minimum(time_fit_prime, c0_cand)

        # 在holdout集上计算LPB = mean(clip(ŷ - q, 0, c0))
        pred_holdout = predict_mean_survival_truncated(model, X_c0_holdout, c0=c0_cand, model_type=model_type)
        q_est = np.quantile(scores, 0.9)
        lpb_clipped = np.mean(np.clip(pred_holdout - q_est, 0.0, c0_cand))

        c0_scores[float(c0_cand)] = lpb_clipped

        if lpb_clipped > best_score:
            best_score = lpb_clipped
            c0_best = float(c0_cand)

    if verbose:
        print(f"\n  c0 网格搜索完成 ")
        print(f"  搜索范围: [{c0_grid.min():.4f}, {c0_grid.max():.4f}]")
        print(f"  最优 c0: {c0_best:.4f} (平均LPB: {best_score:.4f})")
        print(f"  候选值数: {len(c0_grid)}")

    return c0_best, c0_scores


def estimate_censoring_weights(X_train, time_train, event_train,
                               X_cal, c0_threshold,
                               p_c0_marginal=None, cens_time_train=None,
                               verbose=False):
    """
    估计删失机制权重：w(x) = P(C≥c₀) / P(C≥c₀|X=x)

    参数:
        X_train: numpy.ndarray, 训练集特征 (n_train, n_features) — 用于拟合删失模型
        time_train: numpy.ndarray, 训练集观测时间 — 用于构造训练标签
        event_train: numpy.ndarray, 训练集事件指示（0=删失，1=事件）
        X_cal: numpy.ndarray, 校准集特征 (n_cal, n_features) — 仅用于预测权重
        c0_threshold: float, 删失时间阈值（必须由调用方从训练集计算）
        p_c0_marginal: float, 边际概率 P(C≥c₀)（由训练集估计，必须提供）
        cens_time_train: numpy.ndarray or None, 训练集真实删失时间 C
                         提供时：用精确 C 对全部训练样本构造标签，分类器使用全量数据
                         未提供时：仅对标签已知样本（可确认 C≥c0 或 C<c0）拟合分类器
        verbose: bool, 是否打印统计信息

    返回:
        weights: numpy.ndarray, 校准集上的样本权重，shape=(n_cal,)
        censoring_probs: numpy.ndarray, 估计的删失概率P(C≥c0|X)，shape=(n_cal,)
        censoring_model: RandomForestClassifier, 删失状态分类器（在训练集上拟合）
    """
    if p_c0_marginal is None:
        raise ValueError("必须显式传入 p_c0_marginal（从训练集估计）！")

    # Step 1: 在训练集上构造二元标签：1表示Ci≥c0，0表示Ci<c0
    if cens_time_train is not None:
        #设置：C_i 完全可观测，所有训练样本标签已知
        censor_indicator_train = (cens_time_train >= c0_threshold).astype(int)
        fit_mask = np.ones(len(time_train), dtype=bool)  # 全量训练集
    else:
        # 实际数据近似：未删失(Δ=1) 且 T < c0 的样本 C 未观测，标签未知 → 排除
        censor_indicator_train = (time_train >= c0_threshold).astype(int)
        fit_mask = (event_train == 0) | (time_train >= c0_threshold)

    # Step 2: 用训练集拟合分类器 P(C≥c0|X)）
    censoring_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=2026
    )
    censoring_model.fit(X_train[fit_mask], censor_indicator_train[fit_mask])

    # Step 3: 在校准集上预测条件概率 P(C≥c0|X)
    censor_probs = censoring_model.predict_proba(X_cal)
    if censor_probs.shape[1] == 1:
        # 训练标签只有一类（全为 0 或全为 1），此时分类器无判别力
        only_class = int(censoring_model.classes_[0])
        censoring_probs = np.full(len(X_cal), float(only_class))
    else:
        censoring_probs = censor_probs[:, 1]  # P(C≥c0|X)

    # Step 4: 计算权重 w(x) = P(C≥c0) / P(C≥c0|X=x)
    epsilon = 0.01  # 平滑常数，防止权重发散
    censoring_probs_smoothed = np.clip(censoring_probs, a_min=epsilon, a_max=1.0-epsilon)
    weights = p_c0_marginal / censoring_probs_smoothed

    if verbose:
        mode = "精确C" if cens_time_train is not None else "近似（观测时间代理）"
        print(f"\n  删失机制权重估计（{mode}）")
        print(f"  删失阈值 c0: {c0_threshold:.4f}")
        print(f"  边际 P(C≥c0): {p_c0_marginal:.4f}")
        print(f"  分类器训练样本: {int(fit_mask.sum())}/{len(time_train)}")
        print(f"  P(C≥c0|X) 范围（校准集）: [{censoring_probs.min():.4f}, {censoring_probs.max():.4f}]")
        print(f"  权重 w(x) 范围: [{weights.min():.4f}, {weights.max():.4f}]")

    return weights, censoring_probs, censoring_model


def csa_nonconformity_scores(pred_median, time, event=None, weights=None, c0=None):
    """
    计算非一致性分数

    当 c0 不为 None 时（传统CSA调用）：
        使用 CMR 分数 V = ŷ(x) - (T̃ ∧ c0)，其中 T̃ ∧ c0 = min(time, c0)
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
        # CMR 单侧分数：V = ŷ - min(T̃, c0)
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
    CSA分位数校准
    
    分位数定义：
    - 无权重: q = sup{ z : Q{V_i ≤ z} < 1-α }
    - 有权重: q = sup{ z : Σ_i W_i · 1{V_i ≤ z} / Σ_j W_j ≥ 1-α }

    参数:
        scores: 校准集的非一致性分数 V = ŷ - min(T̃, c₀)
        alpha: 显著性水平（目标覆盖率 = 1-α）
        weights: numpy.ndarray, 校准样本权重（可选），若None使用均匀权重
        test_weight: float, 测试点权重ŵ(x) 

    返回:
        q: float 或 np.ndarray, 置信区间半宽
           - 若 test_weight 是标量 → 返回标量q
           - 若 test_weight 是数组 → 返回数组（每个测试点一个q）
           - 返回 np.inf 表示∞-atom激活
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    
    # 标准化inputs
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=float)
    
    if len(weights) != n:
        raise ValueError(f"权重数{len(weights)} ≠ 分数数{n}")
    
    # 处理test_weight（可能是标量或数组）
    if test_weight is None:
        test_weight = np.zeros(1)  # 默认无测试点权重
    else:
        test_weight = np.atleast_1d(test_weight)
    
    is_vectorized = len(test_weight) > 1
    
    # 排序一次
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumsum_weights = np.cumsum(sorted_weights)
    sum_weights = cumsum_weights[-1]
    
    # 分位数定义：sup{ z : Σ_{V_i ≤ z} W_i / (Σ_j W_j + ŵ) ≥ 1-α }
    def find_quantile_for_test_weight(w_test):
        total_weight = sum_weights + w_test  # ∞-atom分母
        threshold = (1 - alpha) * total_weight
        
        # 找第一个使累积权重 ≥ threshold 的位置
        idx = np.where(cumsum_weights >= threshold)[0]
        if len(idx) > 0:
            return sorted_scores[idx[0]]
        else:
            return np.inf  # ∞-atom激活
    
    if is_vectorized:
        # 向量化（多个测试点）
        q_values = np.array([find_quantile_for_test_weight(w) for w in test_weight])
        return q_values
    else:
        # 标量（单个测试点）
        return find_quantile_for_test_weight(float(test_weight[0]))
