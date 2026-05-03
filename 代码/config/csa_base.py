"""
CSA 共享核心组件
"""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from .models import predict_median_survival, predict_mean_survival_truncated
from .models import fit_kaplan_meier, fit_cox, fit_weibull, fit_rsf


def _fit_model_by_type(X, time, event, model_type):
    """按模型类型重新拟合生存模型。"""
    if model_type == 'km':
        return fit_kaplan_meier(time, event)
    if model_type == 'cox':
        return fit_cox(X, time, event)
    if model_type == 'weibull':
        return fit_weibull(X, time, event)
    if model_type == 'rsf':
        return fit_rsf(X, time, event)
    raise ValueError(f'未知 model_type: {model_type}')


def estimate_c0_on_train(X_train, time_train, event_train, model,
                          X_cal, time_cal, event_cal,
                          model_type='cox', c0_grid=None,
                          cens_time_train=None, cens_time_cal=None,
                          min_p_c0=0.3, max_p_c0=0.85, alpha=0.1, verbose=False,
                          holdout_frac=0.25, random_state=2026):
    """
    c0 自适应选择：只使用训练集内部数据，避免复用最终校准集。

        ĉ0 = argmax_{c0 ∈ C}  (1/|Z_ca|) Σ_{i∈Z_ca} L̂_{c0}(X_i)

    对每个 c0 候选值，按照 Algorithm 1 步骤计算校准集 Z_ca 上的平均 LPB：
      - Step 2: I'_ca = {i ∈ Z_ca : C_i ≥ c0}
      - Step 3: V_i = ŷ(X_i) - min(T̃_i, c0)，i ∈ I'_ca
      - Step 6: η = Quantile(1-α; ...) 带 ∞-atom 修正
      - 输出: L̂_{c0}(X) = clip(ŷ(X) - η, 0, c0)，对所有 X ∈ Z_ca

    P(C≥c0) 约束由训练集估计，防止权重极端化。

    参数:
        X_train, time_train, event_train: 训练集（用于估计 P(C≥c0) 约束）
        X_cal, time_cal, event_cal: 校准集 Z_ca（用于一致性分数和 LPB 评估）
        model: 已拟合的生存模型
        model_type: str, 模型类型
        c0_grid: c0 候选网格；None 则从 Z_ca 观测时间的分位数自动生成
        cens_time_train: 训练集真实删失时间 C（合成数据可用，用于 P(C≥c0) 约束估计）
        cens_time_cal: 校准集真实删失时间 C（合成数据可用，用于精确定义 I'_ca）
        min_p_c0, max_p_c0: P(C≥c0) 的合法范围（从训练集估计）
        alpha: 目标误覆盖率
        verbose: 是否打印诊断信息

    返回:
        c0_best: float, 选定的最优 c0
        c0_scores: dict, 各候选的平均 LPB（被约束跳过的值为 np.nan）
    """
    time_train = np.asarray(time_train)
    event_train = np.asarray(event_train)

    n_train = len(time_train)
    if n_train < 8:
        raise ValueError('训练集过小，无法稳定选择 c0')

    rng = np.random.default_rng(random_state)
    idx_all = np.arange(n_train)
    n_holdout = max(2, int(round(n_train * holdout_frac)))
    n_holdout = min(n_holdout, n_train - 2)
    holdout_idx = rng.choice(idx_all, size=n_holdout, replace=False)
    inner_idx = np.setdiff1d(idx_all, holdout_idx, assume_unique=False)

    X_inner = X_train[inner_idx]
    time_inner = time_train[inner_idx]
    event_inner = event_train[inner_idx]
    X_holdout = X_train[holdout_idx]
    time_holdout = time_train[holdout_idx]
    event_holdout = event_train[holdout_idx]

    if cens_time_train is not None:
        cens_inner = np.asarray(cens_time_train)[inner_idx]
        cens_holdout = np.asarray(cens_time_train)[holdout_idx]
    else:
        cens_inner = None
        cens_holdout = None

    # Step 1: 生成 c0 候选网格，仅基于训练集内部数据
    if c0_grid is None:
        percentiles = np.linspace(10, 90, 17)
        base_times = cens_inner if cens_inner is not None else time_inner
        c0_grid = np.percentile(base_times, percentiles)

    c0_scores = {}
    best_score = -np.inf
    c0_best = None
    n_skipped_constraint = 0

    for c0_cand in c0_grid:
        # 约束检查：从训练集估计 P(C≥c0) ∈ [min_p_c0, max_p_c0]
        # 使用训练集（非校准集）估计，避免约束检查本身引入 double-dipping
        if cens_time_train is not None:
            p_c0_est = float(np.mean(cens_time_train >= c0_cand))
        else:
            p_c0_est = float(np.mean(time_train >= c0_cand))

        if p_c0_est < min_p_c0 or p_c0_est > max_p_c0:
            c0_scores[float(c0_cand)] = np.nan
            n_skipped_constraint += 1
            continue

        # 用训练子集重新拟合模型，避免泄漏 holdout 信息
        inner_model = _fit_model_by_type(X_inner, time_inner, event_inner, model_type)

        # 用训练集内部 holdout 评估每个候选 c0 的效率
        if cens_holdout is not None:
            cal_mask = cens_holdout >= c0_cand
        else:
            cal_mask = (event_holdout == 1) | (time_holdout >= c0_cand)

        if cal_mask.sum() < 2:
            c0_scores[float(c0_cand)] = np.nan
            continue

        # Algorithm 1 Step 3: V_i = ŷ(X_i) - min(T̃_i, c0)，i ∈ I'_ca
        pred_cal_masked = predict_mean_survival_truncated(
            inner_model, X_holdout[cal_mask], c0=c0_cand, model_type=model_type)
        scores = pred_cal_masked - np.minimum(time_holdout[cal_mask], c0_cand)

        # 训练内的 c0 选择仅比较效率，使用无权重 split conformal 分位数。
        n_scores = len(scores)
        q_level = min(1.0, (1 - alpha) * (1 + 1.0 / n_scores))
        q_est = np.quantile(scores, q_level)

        # L̂_{c0}(X_i) = clip(ŷ(X_i) - η, 0, c0)，对 holdout 样本求平均
        pred_cal_all = predict_mean_survival_truncated(
            inner_model, X_holdout, c0=c0_cand, model_type=model_type)
        mean_lpb = float(np.mean(np.clip(pred_cal_all - q_est, 0.0, c0_cand)))

        c0_scores[float(c0_cand)] = mean_lpb

        if mean_lpb > best_score:
            best_score = mean_lpb
            c0_best = float(c0_cand)

    if verbose:
        print(f"\n  c0 网格搜索完成")
        print(f"  搜索范围: [{c0_grid.min():.4f}, {c0_grid.max():.4f}]")
        print(f"  训练内拆分: inner={len(inner_idx)} / holdout={len(holdout_idx)}")
        print(f"  约束 P(C≥c₀) ∈ [{min_p_c0}, {max_p_c0}]（训练集估计）："
              f"跳过 {n_skipped_constraint}/{len(c0_grid)} 个候选")
        print(f"  最优 c0: {c0_best:.4f}（平均 LPB: {best_score:.4f}）")
        print(f"  有效候选数: {len(c0_grid) - n_skipped_constraint}/{len(c0_grid)}")

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


def should_fallback_to_unweighted(weights, censoring_probs, p_c0_marginal,
                                  max_weight=10.0, min_prob_floor=0.05,
                                  max_prob_spread=0.25):
    """
    为稳定性决定是否回退到无权重版本。

    在独立删失或近独立删失下，理论上 w(x) 应接近常数 1。
    若估计出的删失概率过小、离散过大或权重过于极端，则回退到无权重。
    """
    weights = np.asarray(weights, dtype=float)
    censoring_probs = np.asarray(censoring_probs, dtype=float)

    if len(weights) == 0:
        return True

    if not np.isfinite(weights).all():
        return True

    if np.max(weights) > max_weight:
        return True

    if np.min(censoring_probs) < min_prob_floor:
        return True

    if np.max(censoring_probs) - np.min(censoring_probs) > max_prob_spread:
        return True

    if p_c0_marginal <= 0 or p_c0_marginal >= 1:
        return True

    return False


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
