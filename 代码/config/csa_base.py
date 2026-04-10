"""
CSA 共享核心组件（论文Candes et al. (2023) JRSSB Algorithm 1严格实现）
- estimate_c0_on_train          : c0 自适应选择（根据论文 Section 3）
- estimate_censoring_weights    : 删失机制逆概率权重（权重分子来自训练集）
- csa_nonconformity_scores      : 非一致性分数（CMR 单侧 / 旧双侧兼容）
- calibrate_csa_quantile        : 加权分位数校准（论文定义 + 无权重∞-atom）
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .models import predict_median_survival


def estimate_c0_on_train(X_train, time_train, event_train, model, model_type='cox',
                          c0_grid=None, holdout_ratio=0.25, verbose=False):
    """
    论文要求的 c0 自适应选择：网格搜索 + 训练集holdout验证
    
    根据Candes et al. (2023)论文 Section 3，c0不应固定为中位数，
    而应通过在训练集上网格搜索、最大化平均下界长度(LPB)来选择。
    
    参数:
        X_train: numpy.ndarray, 训练集特征 (n_train, n_features)
        time_train: numpy.ndarray, 训练集观测时间
        event_train: numpy.ndarray, 训练集事件指示
        model: 已拟合的生存模型
        model_type: str, 模型类型
        c0_grid: numpy.ndarray or None, c0类值网格；若None则自动生成（从10%到90%分位数）
        holdout_ratio: float, 用于验证的holdout集比例（默认25%）
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
    (X_c0_fit, X_c0_holdout,
     time_c0_fit, time_c0_holdout,
     event_c0_fit, event_c0_holdout) = train_test_split(
        X_train, time_train, event_train,
        test_size=holdout_ratio, random_state=2026, stratify=event_train
    )
    
    c0_scores = {}
    best_score = -np.inf
    c0_best = None
    
    # Step 3: 对每个c0候选值，计算holdout集上的平均下界LPB
    for c0_cand in c0_grid:
        # 在c0_fit上计算校准集子集I'_ca（论文定义：{Ci ≥ c0}）
        cal_mask = time_c0_fit >= c0_cand
        if np.sum(cal_mask) < 2:  # 需要至少2个样本
            c0_scores[float(c0_cand)] = np.nan
            continue
        
        # 计算非一致性分数
        pred_fit = predict_median_survival(model, X_c0_fit[cal_mask], model_type=model_type)
        time_fit_prime = time_c0_fit[cal_mask]
        scores = pred_fit - np.minimum(time_fit_prime, c0_cand)
        
        # 在holdout集上计算LPB = mean(ŷ - q) ∧ c0
        pred_holdout = predict_median_survival(model, X_c0_holdout, model_type=model_type)
        # 简单分位数估计（这里用90%分位数作为q的估计）
        q_est = np.quantile(scores, 0.9)
        lpb = np.mean(np.maximum(pred_holdout - q_est, 0.0)) 
        # 约束在[0, c0]
        lpb_clipped = np.mean(np.clip(pred_holdout - q_est, 0.0, c0_cand))
        
        c0_scores[float(c0_cand)] = lpb_clipped
        
        if lpb_clipped > best_score:
            best_score = lpb_clipped
            c0_best = float(c0_cand)
    
    if verbose:
        print(f"\n🔍 c0 网格搜索完成 (论文要求方法)")
        print(f"  搜索范围: [{c0_grid.min():.4f}, {c0_grid.max():.4f}]")
        print(f"  最优 c0: {c0_best:.4f} (平均LPB: {best_score:.4f})")
        print(f"  候选值数: {len(c0_grid)}")
    
    return c0_best, c0_scores


def estimate_censoring_weights(X_train, time_train, event_train,
                               X_cal, c0_threshold,
                               p_c0_marginal=None, verbose=False):
    """
    估计删失机制权重：w(x) = P(C≥c₀) / P(C≥c₀|X=x)

    论文要求（Candes Algorithm 1 Step 4）：权重函数 ŵ(·; Ztr) 必须在训练折上
    拟合，再对校准集/测试集评估，以保证校准集的独立性（data independence）。

    参数:
        X_train: numpy.ndarray, 训练集特征 (n_train, n_features) — 用于拟合删失模型
        time_train: numpy.ndarray, 训练集观测时间 — 用于构造训练标签
        event_train: numpy.ndarray, 训练集事件指示（0=删失，1=事件）
        X_cal: numpy.ndarray, 校准集特征 (n_cal, n_features) — 仅用于预测权重
        c0_threshold: float, 删失时间阈值（必须由调用方从训练集计算）
        p_c0_marginal: float, 边际概率 P(C≥c₀)（由训练集估计，必须提供）
        verbose: bool, 是否打印统计信息

    返回:
        weights: numpy.ndarray, 校准集上的样本权重，shape=(n_cal,)
        censoring_probs: numpy.ndarray, 估计的删失概率P(C≥c0|X)，shape=(n_cal,)
        censoring_model: RandomForestClassifier, 删失状态分类器（在训练集上拟合）
    """
    if p_c0_marginal is None:
        raise ValueError(
            "❌ 必须显式传入 p_c0_marginal（从训练集估计）！"
            "否则违反论文数据隔离原则。"
        )

    # Step 1: 在训练集上构造二元标签：1表示Ci≥c0，0表示Ci<c0
    # 标签可直接确认的训练样本：
    #   删失(Δ=0)：Ci = time_train，标签 = (time_train ≥ c0)，已知
    #   未删失(Δ=1) 且 T ≥ c0：Ci > T ≥ c0，标签必为1，已知
    #   未删失(Δ=1) 且 T < c0：Ci > T 但 Ci 未观测，标签未知 → 排除
    censor_indicator_train = (time_train >= c0_threshold).astype(int)
    known_mask_train = (event_train == 0) | (time_train >= c0_threshold)

    # Step 2: 用训练集标签已知样本拟合分类器 P(C≥c0|X)（论文 ŵ(·; Ztr)）
    censoring_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=2026
    )
    censoring_model.fit(X_train[known_mask_train], censor_indicator_train[known_mask_train])

    # Step 3: 在校准集上预测条件概率 c(x;c0) = P(C≥c0|X)
    censor_probs = censoring_model.predict_proba(X_cal)  # shape=(n_cal, 2)
    censoring_probs = censor_probs[:, 1]  # P(C≥c0|X)

    # Step 4: 计算权重 w(x) = P(C≥c0) / P(C≥c0|X=x)
    # 分子来自训练集（已通过参数传入），分母用训练集模型对校准集评估
    epsilon = 0.01  # 平滑常数
    censoring_probs_smoothed = np.clip(censoring_probs, a_min=epsilon, a_max=1.0-epsilon)
    weights = p_c0_marginal / censoring_probs_smoothed

    if verbose:
        print(f"\n📊 删失机制权重估计（论文 Algorithm 1 Step 4 严格实现）")
        print(f"  删失阈值 c0: {c0_threshold:.4f} ✓（来自训练集）")
        print(f"  边际 P(C≥c0): {p_c0_marginal:.4f} ✓（来自训练集）")
        print(f"  训练集标签已知样本: {np.sum(known_mask_train)}/{len(time_train)}")
        print(f"  P(C≥c0|X) 范围（校准集）: [{censoring_probs.min():.4f}, {censoring_probs.max():.4f}]")
        print(f"  权重 w(x) 范围: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  权重均值: {weights.mean():.4f} ± {weights.std():.4f}")
        print(f"  平滑参数: ε={epsilon:.4f}")

    return weights, censoring_probs, censoring_model


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
    论文定义的CSA分位数校准（Candes et al. Algorithm 1 Step 5-6）
    
    分位数定义（论文式2）：
    - 无权重: q = sup{ z : Q{V_i ≤ z} < 1-α }  （修正为论文的sup定义）
    - 有权重: q = sup{ z : Σ_i W_i · 1{V_i ≤ z} / Σ_j W_j ≥ 1-α }
    
    关键改动（严格遵循论文）：
    1. 无权重模式现在也加∞-atom（关键差异#6的修正）
    2. 使用论文的sup定义而非ceiling公式
    3. 返回 np.inf 时代表∞-atom激活（下界 → -∞ → clip到0）

    参数:
        scores: 校准集的非一致性分数 V = ŷ - min(T̃, c₀)
        alpha: 显著性水平（目标覆盖率 = 1-α）
        weights: numpy.ndarray, 校准样本权重（可选），若None使用均匀权重
        test_weight: float, 测试点权重ŵ(x) - 关键：无权重模式也要提供！
                    对应论文∞-atom分母修正 (Σ_j W_j + ŵ(x))

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
    
    # 论文分位数定义：sup{ z : Σ_{V_i ≤ z} W_i / (Σ_j W_j + ŵ) ≥ 1-α }
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
