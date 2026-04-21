"""
复现.py — 论文 "Conformalized survival analysis"（Candès, Lei & Ren, JRSSB 2023）专用辅助函数

本文件中每个函数均说明：
  (1) 函数用途
  (2) 为何需要新建（已有函数缺少什么）

函数列表
--------
generate_paper_lognormal_data    — 按论文 Table 1 生成 Log-Normal AFT 数据
predict_survival_at_time         — 在指定时间点评估生存函数 S(t|X)
predict_quantile_from_survival   — 从生存函数求任意分位数（CDR-LPB 需要）
compute_cdr_nonconformity_scores — CDR 非一致性分数 V = S(T̃∧c₀|X) − (1−α)
compute_cdr_lower_bound          — CDR-LPB 下界 F̂⁻¹(α − η|X) ∧ c₀
naive_split_conformal_lpb        — Naive 分裂共形（直接对 T̃ 做，无删失调整）
direct_model_lpb                 — 直接从模型取 α-分位数（无共形校准）
csa_cdr_lpb                      — Algorithm 1 + CDR 分数（完整流程）
run_single_trial                 — 运行一次模拟试验（所有方法）
evaluate_coverage_trial          — 计算本次试验的覆盖率
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────────────────────────
# 1. 数据生成
# ─────────────────────────────────────────────────────────────────

def generate_paper_lognormal_data(n, scenario, seed=None):
    """
    按论文 Table 1 的设定生成生存分析合成数据。

    数据生成过程（DGP）:
        logT | X ~ N(μ(X), σ²(X))，即 T = exp(μ(X) + σ(X)·ε)，ε ~ N(0,1)
        C ~ Exp(rate=0.4)，与 (X, T) 完全独立（Assumption 2）
        T̃ = min(T, C)，Δ = 1{T ≤ C}

    场景（scenario）参数对应 Table 1 四种设定：
        'uvt_homo'   — 单变量 + 同方差，p=1, σ=1.5
        'uvt_hetero' — 单变量 + 异方差，p=1, σ(x)=1+x/5
        'mvt_homo'   — 多变量 + 同方差，p=100, σ=1
        'mvt_hetero' — 多变量 + 异方差，p=100, σ(x)=|x₁₀|+1

    参数
    ----
    n        : 样本量
    scenario : str，四种设定之一（见上）
    seed     : 随机种子

    返回（均为 ndarray）
    --------------------
    X          : (n, p) 协变量矩阵
    time_obs   : (n,)   观测时间 T̃ = min(T, C)
    event      : (n,)   事件指示 Δ = 1{T ≤ C}
    time_true  : (n,)   真实生存时间 T（合成数据专用，用于覆盖率评估）
    cens_time  : (n,)   删失时间 C（Type I 删失下完全可观测）

    ── 为何新建本函数 ──
    已有 generate_weibull_data() 使用 **Weibull 分布**生成 T，与论文使用的
    **Log-Normal AFT**（logT|X ~ N(μ,σ²)）分布族不同：
      • 分布族不同：Weibull ≠ Log-Normal；论文模型更接近加速失效时间（AFT）
      • 协变量结构不同：单变量时论文用 X~U(0,4)、μ=2+0.37√x；已有代码用
        exp(2+0.8·x_std)，参数值完全不同
      • 多变量时论文用 p=100, X~U([-1,1]^100)；已有代码用 p=50 且协变量
        含 age/gender 等具体变量
      • 删失机制不同：论文直接用 C~Exp(0.4)（完全独立）；已有代码用二分搜索
        使删失率接近目标值，而非直接指定分布参数
    因此无法复用现有函数，需从头实现 Table 1 的精确 DGP。
    """
    rng = np.random.default_rng(seed)

    if scenario in ('uvt_homo', 'uvt_hetero'):
        # ── 单变量设定 ──────────────────────────────────────────────
        p = 1
        X = rng.uniform(0, 4, size=(n, 1))
        x = X[:, 0]                              # scalar covariate

        mu_x = 2.0 + 0.37 * np.sqrt(x)          # μ(x) = 2 + 0.37√x

        if scenario == 'uvt_homo':
            sigma_x = np.full(n, 1.5)            # σ = 1.5（同方差）
        else:
            sigma_x = 1.0 + x / 5.0             # σ(x) = 1 + x/5（异方差）

    elif scenario in ('mvt_homo', 'mvt_hetero'):
        # ── 多变量设定 ──────────────────────────────────────────────
        p = 100
        X = rng.uniform(-1, 1, size=(n, p))

        # 论文使用 1-indexed：x₁² − x₃x₅ → Python 0-indexed：X[:,0]² − X[:,2]·X[:,4]
        mu_x = (np.log(2) + 1.0
                + 0.55 * (X[:, 0] ** 2 - X[:, 2] * X[:, 4]))

        if scenario == 'mvt_homo':
            sigma_x = np.ones(n)                 # σ = 1（同方差）
        else:
            # 论文 x₁₀ → Python X[:,9]
            sigma_x = np.abs(X[:, 9]) + 1.0     # σ(x) = |x₁₀| + 1（异方差）
    else:
        raise ValueError(
            f"未知 scenario='{scenario}'。"
            f"可选值: 'uvt_homo', 'uvt_hetero', 'mvt_homo', 'mvt_hetero'"
        )

    # 生成真实生存时间 T：logT | X ~ N(μ(X), σ²(X))
    log_T = mu_x + sigma_x * rng.standard_normal(n)
    T = np.exp(log_T)

    # 生成删失时间 C ~ Exp(rate=0.4)（与 X, T 完全独立）
    C = rng.exponential(scale=1.0 / 0.4, size=n)

    # 构造观测数据
    event    = (T <= C).astype(int)             # Δ = 1{T ≤ C}
    time_obs = np.clip(np.minimum(T, C), 1e-8, None)
    time_true = np.clip(T, 1e-8, None)
    cens_time = np.clip(C, 1e-8, None)

    return X, time_obs, event, time_true, cens_time


# ─────────────────────────────────────────────────────────────────
# 2. 生存函数相关工具
# ─────────────────────────────────────────────────────────────────

def predict_survival_at_time(model, X, t, model_type='cox'):
    """
    对每个样本评估 S(t | X=x)（在指定时间点 t 的生存概率）。

    参数
    ----
    model      : 已拟合的生存模型
    X          : (n, p) 协变量矩阵
    t          : float，目标时间点
    model_type : 'km' / 'cox' / 'weibull' / 'rsf'

    返回
    ----
    ndarray，shape=(n,)，每个样本在时间 t 的生存概率 S(t|X=x)

    ── 为何新建本函数 ──
    已有 predict_mean_survival_truncated() 计算 ∫₀^c₀ S(t|X)dt（截断均值），
    predict_median_survival() 仅寻找 S(t)=0.5 的时间点（中位生存时间）。
    CDR 非一致性分数需要在**指定时间点** T̃∧c₀ 处评估生存概率 S(T̃∧c₀|X)，
    现有函数均不支持此操作，因此新建本函数。
    """
    t = float(t)
    n = X.shape[0]

    if model_type == 'km':
        times = model.survival_function_.index.values.astype(float)
        vals  = model.survival_function_['KM_estimate'].values.astype(float)
        s_val = float(_step_interp(times, vals, t))
        return np.full(n, s_val)

    if model_type in ('cox', 'weibull'):
        df    = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        sdf   = model.predict_survival_function(df)
        times = sdf.index.values.astype(float)
        result = np.zeros(n)
        for j, col in enumerate(sdf.columns):
            s = sdf[col].values.astype(float)
            result[j] = _step_interp(times, s, t)
        return result

    if model_type == 'rsf':
        surv_funcs = model.predict_survival_function(X)
        times = np.asarray(model.unique_times_, dtype=float)
        result = np.zeros(n)
        for j, fn in enumerate(surv_funcs):
            s = np.asarray(fn(times), dtype=float)
            result[j] = _step_interp(times, s, t)
        return result

    raise ValueError(f'未知 model_type: {model_type}')


def _step_interp(times, vals, t):
    """阶梯函数（右连续）在时间 t 处的值：S(t) = S(t−) 为左极限约定。"""
    if t < times[0]:
        return 1.0
    if t >= times[-1]:
        return float(vals[-1])
    idx = np.searchsorted(times, t, side='right') - 1
    return float(vals[idx])


def predict_survival_at_times_vectorized(model, X, t_array, model_type='cox'):
    """
    对每个样本评估 S(t_i | X=x_i)，其中 t_i 随样本不同（向量化）。

    参数
    ----
    model      : 已拟合的生存模型
    X          : (n, p) 协变量矩阵
    t_array    : (n,) 每个样本对应的时间点
    model_type : 'km' / 'cox' / 'weibull' / 'rsf'

    返回
    ----
    ndarray，shape=(n,)，S(t_i | X_i)

    ── 为何新建本函数 ──
    predict_survival_at_time() 只支持所有样本共享一个时间点 t，
    CDR 分数需要 S(T̃_i ∧ c₀ | X_i)，每个样本的截断时间不同，需逐样本查询。
    """
    t_array = np.asarray(t_array, dtype=float)
    n = X.shape[0]
    result = np.zeros(n)

    if model_type == 'km':
        times = model.survival_function_.index.values.astype(float)
        vals  = model.survival_function_['KM_estimate'].values.astype(float)
        for i in range(n):
            result[i] = _step_interp(times, vals, t_array[i])
        return result

    if model_type in ('cox', 'weibull'):
        df   = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        sdf  = model.predict_survival_function(df)
        times = sdf.index.values.astype(float)
        for j, col in enumerate(sdf.columns):
            s = sdf[col].values.astype(float)
            result[j] = _step_interp(times, s, t_array[j])
        return result

    if model_type == 'rsf':
        surv_funcs = model.predict_survival_function(X)
        times = np.asarray(model.unique_times_, dtype=float)
        for j, fn in enumerate(surv_funcs):
            s = np.asarray(fn(times), dtype=float)
            result[j] = _step_interp(times, s, t_array[j])
        return result

    raise ValueError(f'未知 model_type: {model_type}')


def predict_quantile_from_survival(model, X, survival_level, model_type='cox'):
    """
    求满足 S(t|X) ≤ survival_level 的最小时间 t（即 CDF 分位数的反演）。

    等价关系：
        S(t|X) = survival_level  ←→  F(t|X) = 1 - survival_level
        本函数返回的是 (1 - survival_level) 的 CDF 分位数。

    CDR-LPB 使用：
        L̂(x) = F̂⁻¹(α − η | X) ∧ c₀
             = S̃⁻¹(1 − (α − η) | X) ∧ c₀
        → 传入 survival_level = 1 − (α − η)

    参数
    ----
    model          : 已拟合的生存模型
    X              : (n, p) 协变量矩阵
    survival_level : float 或 (n,) 数组，目标生存概率水平
    model_type     : 'km' / 'cox' / 'weibull' / 'rsf'

    返回
    ----
    ndarray，shape=(n,)

    ── 为何新建本函数 ──
    已有 predict_median_survival() 固定寻找 S(t)=0.5 的点（中位生存时间）。
    CDR-LPB 需要对任意水平（因每个测试点的 η 不同）进行 CDF 反演，
    且 survival_level 可以是向量（每个测试点不同），现有函数不支持。
    """
    survival_level = np.atleast_1d(np.asarray(survival_level, dtype=float))
    n = X.shape[0]
    assert len(survival_level) == 1 or len(survival_level) == n, \
        "survival_level 长度须为 1 或 n"
    if len(survival_level) == 1:
        survival_level = np.full(n, survival_level[0])

    result = np.zeros(n)

    def _invert_one(times, s_vals, s_target):
        """在生存曲线上找第一个 S(t) ≤ s_target 的 t。"""
        if s_target >= 1.0:
            return 0.0          # S(0)=1 ≥ target，无法更早
        if s_target <= s_vals[-1]:
            return float(times[-1])   # 超出观测范围，返回最大时间
        idx = np.where(s_vals <= s_target)[0]
        if len(idx) == 0:
            return float(times[-1])
        return float(times[idx[0]])

    if model_type == 'km':
        times = model.survival_function_.index.values.astype(float)
        vals  = model.survival_function_['KM_estimate'].values.astype(float)
        for i in range(n):
            result[i] = _invert_one(times, vals, survival_level[i])
        return result

    if model_type in ('cox', 'weibull'):
        df   = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        sdf  = model.predict_survival_function(df)
        times = sdf.index.values.astype(float)
        for j, col in enumerate(sdf.columns):
            s = sdf[col].values.astype(float)
            result[j] = _invert_one(times, s, survival_level[j])
        return result

    if model_type == 'rsf':
        surv_funcs = model.predict_survival_function(X)
        times = np.asarray(model.unique_times_, dtype=float)
        for j, fn in enumerate(surv_funcs):
            s = np.asarray(fn(times), dtype=float)
            result[j] = _invert_one(times, s, survival_level[j])
        return result

    raise ValueError(f'未知 model_type: {model_type}')


# ─────────────────────────────────────────────────────────────────
# 3. CDR 非一致性分数
# ─────────────────────────────────────────────────────────────────

def compute_cdr_nonconformity_scores(model, X_cal, time_cal, c0, alpha, model_type='cox'):
    """
    计算 CDR（Conformalized Distribution Regression）非一致性分数：

        V_i = α − F̂_{T∧c₀|X_i}(T̃_i ∧ c₀)
            = S(T̃_i ∧ c₀ | X_i) − (1 − α)

    其中 F̂(y|x) = 1 − S(y|x)，S 为模型估计的条件生存函数。
    此分数定义见论文 Section 3.1（"CDR scores"）和 Algorithm 1 Step 3。

    参数
    ----
    model      : 已拟合的生存模型（用于估计 S(·|x)）
    X_cal      : (n_cal, p) 校准集协变量（已筛选 I'_ca，即 C_i ≥ c₀）
    time_cal   : (n_cal,) 校准集观测时间 T̃_i（已筛选）
    c0         : float，截断阈值
    alpha      : float，显著性水平（目标覆盖率 = 1 − α）
    model_type : 'km' / 'cox' / 'weibull' / 'rsf'

    返回
    ----
    scores : (n_cal,) CDR 非一致性分数

    ── 为何新建本函数 ──
    已有 csa_nonconformity_scores() 只实现了 **CMR 分数**：
        V_CMR = m̂(x) − min(T̃, c₀)，其中 m̂(x) = E[T∧c₀|X=x]
    而 CDR 分数使用**条件 CDF** 的值：
        V_CDR = α − F̂(T̃∧c₀|X) = S(T̃∧c₀|X) − (1−α)
    两者的计算方式、量纲（CMR 为时间差，CDR 为无量纲概率差）和数值范围均不同，
    无法通过改参数方式复用；且 CDR 需要在指定时间点评估生存函数，
    而现有函数不支持此操作。
    """
    c0 = float(c0)
    y_cal = np.minimum(time_cal, c0)  # T̃_i ∧ c₀（I'_ca 中 C_i ≥ c₀）

    # S(T̃_i ∧ c₀ | X_i)：对每个样本在其各自的 y_i 处评估生存函数
    s_vals = predict_survival_at_times_vectorized(model, X_cal, y_cal, model_type)

    # V_i = S(T̃∧c₀|X) − (1−α) = α − F̂(T̃∧c₀|X)
    scores = s_vals - (1.0 - alpha)
    return scores


# ─────────────────────────────────────────────────────────────────
# 4. CDR-LPB 下界计算
# ─────────────────────────────────────────────────────────────────

def compute_cdr_lower_bound(model, X_test, eta_per_test, c0, alpha, model_type='cox'):
    """
    计算 CDR-LPB（Algorithm 1 Output）：

        L̂(x) = F̂⁻¹_{T∧c₀|X}(α − η(x)) ∧ c₀
              = S̃⁻¹(1 − (α − η(x)) | X) ∧ c₀

    其中 S̃⁻¹(s|X) = inf{t : S(t|X) ≤ s} 为条件生存函数的反函数。
    η(x) 是由 calibrate_csa_quantile() 得到的加权分位数（每个测试点不同）。

    特殊情况处理：
      • η = ∞（∞-atom 激活）：α − ∞ ≤ 0，F̂⁻¹(≤0) = 0，L̂ = 0
      • α − η ≤ 0：同上，L̂ = 0
      • α − η ≥ 1：S̃⁻¹(0) = sup(支撑集) ≈ max 观测时间，L̂ = c₀

    参数
    ----
    model        : 已拟合的生存模型
    X_test       : (n_test, p) 测试集协变量
    eta_per_test : (n_test,) 每个测试点的校准分位数 η(x)
    c0           : float，截断阈值
    alpha        : float，显著性水平
    model_type   : 'km' / 'cox' / 'weibull' / 'rsf'

    返回
    ----
    lower : (n_test,) CDR 下界

    ── 为何新建本函数 ──
    已有 fit_csa_intervals_traditional() 的下界计算是 CMR 形式：
        L̂_CMR(x) = (m̂(x) − η(x)) ∧ c₀
    CDR 需要对条件 CDF 做反演：
        L̂_CDR(x) = S̃⁻¹(1 − (α − η(x)) | X) ∧ c₀
    两者计算方式完全不同；且每个测试点的 η 不同，需逐点 CDF 反演，
    现有代码没有此能力。
    """
    c0 = float(c0)
    eta_per_test = np.asarray(eta_per_test, dtype=float)
    n_test = len(eta_per_test)

    # α − η(x)：目标 CDF 水平
    cdf_level = alpha - eta_per_test   # shape=(n_test,)

    # 对应生存函数水平 S = 1 − (α − η) = 1 − cdf_level
    survival_level = 1.0 - cdf_level   # shape=(n_test,)

    # 裁剪到 [0, 1]
    # • survival_level ≥ 1（η ≤ 0）：L̂ = 0（S̃⁻¹(1) = 0）
    # • survival_level ≤ 0（η ≥ α，∞-atom 激活）：L̂ = max 时间 → clip to c₀
    # 实际处理：先计算，最后 clip 到 [0, c₀]

    # 处理 ∞（∞-atom 激活）→ α − ∞ = -∞ → cdf_level = -∞ → survival_level = +∞
    inf_mask = ~np.isfinite(eta_per_test)
    survival_level[inf_mask] = 2.0    # 超出范围，将返回 0 后被 clip

    # 计算各测试点的分位数时间
    t_pred = predict_quantile_from_survival(
        model, X_test, survival_level, model_type
    )

    # 对 ∞-atom 激活的测试点：L̂ = 0（论文 Step 6 说明 w(x)=∞ 时 L̂=-∞，即 0）
    t_pred[inf_mask] = 0.0

    # 最终下界：clip 到 [0, c₀]
    lower = np.clip(t_pred, 0.0, c0)
    return lower


# ─────────────────────────────────────────────────────────────────
# 5. Naive 分裂共形（论文对比基准）
# ─────────────────────────────────────────────────────────────────

def naive_split_conformal_lpb(model, X_cal, time_cal, X_test, alpha, model_type='cox'):
    """
    Naive CQR/CMR（论文 Section 4 的对比基准）：直接对 T̃ 用 CMR 分数做
    标准分裂共形（无 c₀ 筛选、无删失权重）。

    步骤：
      1. 用已拟合模型计算校准集预测值 m̂(X_cal)（中位生存时间代替均值，保持简单）
      2. 非一致性分数：V_i = m̂(X_cal_i) − T̃_i
      3. 无权重分位数：η = Quantile_{1−α}({V_i})
      4. L̂(x) = max(m̂(x) − η, 0)（无 c₀ 截断）

    参数
    ----
    model      : 已拟合的生存模型（在训练集上拟合）
    X_cal      : (n_cal, p) 校准集协变量
    time_cal   : (n_cal,) 校准集观测时间 T̃
    X_test     : (n_test, p) 测试集协变量
    alpha      : 显著性水平
    model_type : 'km' / 'cox' / 'weibull' / 'rsf'

    返回
    ----
    lower : (n_test,) 下界

    ── 为何新建本函数 ──
    已有 fit_csa_intervals_traditional() 是**加权**共形推断，包含：
      (a) 自适应选择 c₀（截断阈值）
      (b) 仅保留 C_i ≥ c₀ 的校准样本（I'_ca 筛选）
      (c) 逆概率删失权重 IPCW
    Naive 方法的核心是**不**做任何删失调整：全量校准集、均匀权重、无截断。
    论文用 Naive CQR 作为保守基准，说明忽略删失机制会导致过于保守的区间。
    现有函数无法通过改参数退化为真正的 Naive（即使 use_weights=False 仍会
    做 c₀ 筛选），因此需单独实现。
    """
    from config.models import predict_median_survival

    # 步骤 1：预测校准集和测试集的中位生存时间（作为 CMR 的 m̂(x)）
    pred_cal  = predict_median_survival(model, X_cal,  model_type)
    pred_test = predict_median_survival(model, X_test, model_type)

    # 处理 inf 预测（高删失率时 Cox 模型中位生存时间可能超出观测范围）
    # 将 inf 替换为最大观测时间，保证分位数计算不因 inf-inf=nan 而失败
    max_obs = float(np.max(time_cal))
    pred_cal  = np.where(np.isfinite(pred_cal),  pred_cal,  max_obs)
    pred_test = np.where(np.isfinite(pred_test), pred_test, max_obs)

    # 步骤 2：非一致性分数 V_i = m̂(X_i) − T̃_i
    scores = pred_cal - time_cal

    # 步骤 3：标准（无权重）(1−α)-分位数
    # 论文定义：Quantile(1−α; Q) = sup{z: Q(Z≤z) < 1−α}
    # 等价于 ceil[(1−α)(n+1)]/n 经验分位数
    n_cal = len(scores)
    level  = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    level  = min(level, 1.0)
    eta    = float(np.quantile(scores, level))

    # 步骤 4：下界 L̂(x) = max(m̂(x) − η, 0)
    lower = np.maximum(pred_test - eta, 0.0)
    return lower


# ─────────────────────────────────────────────────────────────────
# 6. 直接模型分位数（无共形校准）
# ─────────────────────────────────────────────────────────────────

def direct_model_lpb(model, X_test, alpha, model_type='cox'):
    """
    直接取模型的 α-分位数作为下界（无共形校准）。

    对应论文 Section 4 中的对比基准：
      • "Cox model"  — Cox 模型的 α-条件分位数
      • "AFT model"  — Weibull AFT 模型的 α-条件分位数

    计算方式：
        L̂(x) = F̂⁻¹(α | X=x) = S̃⁻¹(1 − α | X=x)

    参数
    ----
    model      : 已拟合的生存模型
    X_test     : (n_test, p) 测试集协变量
    alpha      : 显著性水平（LPB 对应 α-分位数）
    model_type : 'km' / 'cox' / 'weibull' / 'rsf'

    返回
    ----
    lower : (n_test,) 下界

    ── 为何新建本函数 ──
    已有代码总是先做共形校准再输出下界（fit_csa_intervals_traditional 等）。
    论文比较的基准方法之一是直接取模型 α-分位数，**不做共形校准**。
    这体现了论文的核心贡献——共形校准保证覆盖率，而直接取分位数不能。
    现有函数没有此直接返回路径，故需新建。
    """
    survival_level = 1.0 - alpha   # S = 1 − α 对应 α-分位数
    lower = predict_quantile_from_survival(
        model, X_test, survival_level, model_type
    )
    lower = np.maximum(lower, 0.0)
    return lower


# ─────────────────────────────────────────────────────────────────
# 7. Algorithm 1 with CDR scores（完整流程）
# ─────────────────────────────────────────────────────────────────

def csa_cdr_lpb(model, X_train, time_train, event_train, cens_time_train,
                X_cal, time_cal, event_cal, cens_time_cal,
                X_test, alpha=0.1, model_type='cox', verbose=False):
    """
    Algorithm 1（论文 Section 3）的 CDR 分数版本。

    步骤严格对应论文 Algorithm 1：
      1. c₀ 选择（在训练集上网格搜索 + holdout 验证）
      2. I'_ca = {i ∈ I_ca : C_i ≥ c₀}
      3. V_i = α − F̂_{T∧c₀|X_i}(T̃_i ∧ c₀) = S(T̃∧c₀|X_i) − (1−α)
      4. W_i = ŵ(X_i) = P(C≥c₀) / P(C≥c₀|X=x_i)
      5. 加权分位数 η(x)（含 ∞-atom）
      6. L̂(x) = F̂⁻¹(α − η(x)|X) ∧ c₀

    参数
    ----
    model          : 已拟合的生存模型
    X_train        : (n_tr,  p) 训练集协变量
    time_train     : (n_tr,)    训练集观测时间
    event_train    : (n_tr,)    训练集事件指示
    cens_time_train: (n_tr,)    训练集删失时间 C（Type I: 完全可观测）
    X_cal          : (n_cal, p) 校准集协变量
    time_cal       : (n_cal,)   校准集观测时间
    event_cal      : (n_cal,)   校准集事件指示
    cens_time_cal  : (n_cal,)   校准集删失时间 C
    X_test         : (n_test,p) 测试集协变量
    alpha          : 显著性水平（目标覆盖率 = 1−α）
    model_type     : 'km' / 'cox' / 'weibull' / 'rsf'
    verbose        : 是否打印诊断信息

    返回
    ----
    lower      : (n_test,) CDR 下界
    upper      : (n_test,) 全部为 np.inf（单侧区间）
    q_mean     : float，η(x) 的均值（诊断用）
    info       : dict，中间结果（c0、权重等）

    ── 为何新建本函数 ──
    已有 fit_csa_intervals_traditional() 固定使用 **CMR 分数**（V = m̂(x) − T̃∧c₀）；
    CDR 分数（V = S(T̃∧c₀|X) − (1−α)）的计算和下界反演完全不同（见
    compute_cdr_nonconformity_scores 和 compute_cdr_lower_bound 的说明）。
    两种分数对应论文中两个不同的方法，必须分别实现。
    """
    from config.csa_base import (
        estimate_c0_on_train,
        estimate_censoring_weights,
        calibrate_csa_quantile,
    )

    # Step 1: 自适应 c₀ 选择（网格搜索 + holdout）
    c0, c0_scores = estimate_c0_on_train(
        X_train, time_train, event_train, model,
        model_type=model_type,
        cens_time_train=cens_time_train,
        verbose=verbose,
    )
    if verbose:
        print(f"  [CDR] c₀ = {c0:.4f}")

    # Step 2: 筛选校准子集 I'_ca = {i : C_i ≥ c₀}
    cal_mask        = cens_time_cal >= c0
    X_cal_prime     = X_cal[cal_mask]
    time_cal_prime  = time_cal[cal_mask]
    event_cal_prime = event_cal[cal_mask]
    if verbose:
        print(f"  [CDR] |I'_ca| = {cal_mask.sum()} / {len(time_cal)}")

    # 边际概率 P(C ≥ c₀)（从训练集估计）
    p_c0_marginal = float(np.mean(cens_time_train >= c0))

    # Step 3-4: 权重估计 W_i = P(C≥c₀) / P(C≥c₀|X_i)
    weights, cens_probs, censor_model = estimate_censoring_weights(
        X_train, time_train, event_train,
        X_cal_prime,
        c0_threshold=c0,
        p_c0_marginal=p_c0_marginal,
        cens_time_train=cens_time_train,
        verbose=verbose,
    )

    # Step 3: CDR 非一致性分数 V_i = S(T̃∧c₀|X_i) − (1−α)
    scores = compute_cdr_nonconformity_scores(
        model, X_cal_prime, time_cal_prime, c0, alpha, model_type
    )

    # 测试点权重 ŵ(x) = P(C≥c₀) / P(C≥c₀|X=x)
    epsilon = 0.01
    _tp = censor_model.predict_proba(X_test)
    if _tp.shape[1] == 1:
        _only = int(censor_model.classes_[0])
        test_cens_probs = np.full(len(X_test), float(_only))
    else:
        test_cens_probs = _tp[:, 1]
    test_cens_probs = np.clip(test_cens_probs, epsilon, 1.0 - epsilon)
    test_weights    = p_c0_marginal / test_cens_probs

    # Step 5-6: 加权分位数校准 η(x)（含 ∞-atom）
    eta_per_test = calibrate_csa_quantile(
        scores,
        alpha=alpha,
        weights=weights,
        test_weight=test_weights,
    )

    # Step 6 Output: L̂(x) = F̂⁻¹(α − η(x)|X) ∧ c₀
    lower = compute_cdr_lower_bound(
        model, X_test, eta_per_test, c0, alpha, model_type
    )

    upper  = np.full_like(lower, np.inf)
    q_mean = float(np.nanmean(eta_per_test[np.isfinite(eta_per_test)]))

    info = {
        'c0': c0,
        'p_c0_marginal': p_c0_marginal,
        'cal_mask': cal_mask,
        'weights': weights,
        'test_weights': test_weights,
        'eta_per_test': eta_per_test,
        'scores_cdr': scores,
    }
    return lower, upper, q_mean, info


# ─────────────────────────────────────────────────────────────────
# 8. 单次试验（所有方法）
# ─────────────────────────────────────────────────────────────────

def run_single_trial(seed, scenario, alpha=0.1,
                     n_total=2000, fit_frac=0.5, test_frac=0.5,
                     model_type='cox', run_cdr=True):
    """
    运行一次模拟试验，比较论文中的所有方法。

    数据划分（对应论文 Section 4）：
      总样本 n_total，其中：
        test set      : n_total × test_frac    （论文: 3000）
        train+cal set : 剩余样本
          fit set     : 剩余 × fit_frac        （论文: 1500 = 50%）
          cal set     : 剩余 × (1 − fit_frac)  （论文: 1500 = 50%）

    方法（对应论文 Section 4 及 Figure 1）：
      1. direct_cox / direct_weibull — 直接 α-分位数（无共形校准）
      2. naive               — Naive split conformal on T̃（论文 "Naive CQR"）
      3. csa_cmr             — Algorithm 1 + CMR 分数（对应 CMR-LPB）
      4. csa_cdr（可选）    — Algorithm 1 + CDR 分数（对应 CDR-LPB）

    参数
    ----
    seed       : 随机种子
    scenario   : 论文 Table 1 场景名
    alpha      : 显著性水平（目标覆盖率 1−α）
    n_total    : 总样本量（论文: 6000）
    fit_frac   : 训练（拟合）集占训练+校准集的比例（论文: 0.5）
    test_frac  : 测试集占总样本的比例（论文: 0.5）
    model_type : 基础生存模型（'cox' / 'weibull' / 'rsf'）
    run_cdr    : 是否运行 CDR-LPB（计算较慢）

    返回
    ----
    dict：各方法的覆盖率 P(T ≥ L̂(X)) 和均值 LPB
    """
    from config.models import fit_cox, fit_weibull, fit_rsf, fit_kaplan_meier
    from config.csa_traditional import fit_csa_intervals_traditional
    import importlib, sys

    # 生成数据
    X, time_obs, event, time_true, cens_time = generate_paper_lognormal_data(
        n=n_total, scenario=scenario, seed=seed
    )

    # 数据划分
    n_test = int(n_total * test_frac)
    n_rest = n_total - n_test
    n_fit  = int(n_rest * fit_frac)
    n_cal  = n_rest - n_fit

    rng  = np.random.default_rng(seed + 10000)
    idx  = rng.permutation(n_total)
    idx_fit  = idx[:n_fit]
    idx_cal  = idx[n_fit:n_fit + n_cal]
    idx_test = idx[n_fit + n_cal:]

    X_fit,  time_fit,  event_fit,  tt_fit,  ct_fit  = (
        X[idx_fit], time_obs[idx_fit], event[idx_fit],
        time_true[idx_fit], cens_time[idx_fit]
    )
    X_cal,  time_cal,  event_cal,  tt_cal,  ct_cal  = (
        X[idx_cal], time_obs[idx_cal], event[idx_cal],
        time_true[idx_cal], cens_time[idx_cal]
    )
    X_test, time_test, event_test, tt_test, ct_test = (
        X[idx_test], time_obs[idx_test], event[idx_test],
        time_true[idx_test], cens_time[idx_test]
    )

    # 拟合生存模型（在 fit 集上）
    if model_type == 'cox':
        model = fit_cox(X_fit, time_fit, event_fit)
    elif model_type == 'weibull':
        model = fit_weibull(X_fit, time_fit, event_fit)
    elif model_type == 'rsf':
        model = fit_rsf(X_fit, time_fit, event_fit)
    elif model_type == 'km':
        model = fit_kaplan_meier(time_fit, event_fit)
    else:
        raise ValueError(f"未知 model_type: {model_type}")

    results = {'scenario': scenario, 'seed': seed,
               'n_total': n_total, 'model_type': model_type,
               'cens_rate': float(1 - np.mean(event))}

    def coverage(lower, time_true_arr):
        """精确覆盖率 P(T ≥ L̂(X))（合成数据专用）"""
        return float(np.mean(time_true_arr >= lower))

    # ── 方法 1：直接模型 α-分位数（无共形校准）──────────────────────
    lower_direct = direct_model_lpb(model, X_test, alpha, model_type)
    results['direct_coverage'] = coverage(lower_direct, tt_test)
    results['direct_mean_lpb'] = float(np.mean(lower_direct))

    # ── 方法 2：Naive split conformal on T̃ ──────────────────────────
    lower_naive = naive_split_conformal_lpb(
        model, X_cal, time_cal, X_test, alpha, model_type
    )
    results['naive_coverage'] = coverage(lower_naive, tt_test)
    results['naive_mean_lpb'] = float(np.mean(lower_naive))

    # ── 方法 3：Algorithm 1 + CMR 分数（论文 CMR-LPB）──────────────
    lower_cmr, _, q_cmr, _ = fit_csa_intervals_traditional(
        model,
        X_fit, time_fit, event_fit,
        X_cal, time_cal, event_cal,
        X_test,
        alpha=alpha,
        model_type=model_type,
        use_weights=True,
        cens_time_train=ct_fit,
        cens_time_cal=ct_cal,
        verbose=False,
    )
    results['cmr_coverage'] = coverage(lower_cmr, tt_test)
    results['cmr_mean_lpb'] = float(np.mean(lower_cmr))

    # ── 方法 4：Algorithm 1 + CDR 分数（论文 CDR-LPB）──────────────
    if run_cdr:
        lower_cdr, _, q_cdr, _ = csa_cdr_lpb(
            model,
            X_fit, time_fit, event_fit, ct_fit,
            X_cal, time_cal, event_cal, ct_cal,
            X_test,
            alpha=alpha,
            model_type=model_type,
            verbose=False,
        )
        results['cdr_coverage'] = coverage(lower_cdr, tt_test)
        results['cdr_mean_lpb'] = float(np.mean(lower_cdr))

    return results


# ─────────────────────────────────────────────────────────────────
# 9. 批量模拟（多次试验）
# ─────────────────────────────────────────────────────────────────

def run_simulation_study(scenarios, n_trials=10, alpha=0.1,
                         n_total=2000, model_type='cox',
                         run_cdr=True, verbose=True):
    """
    运行完整模拟研究（对应论文 Section 4）。

    论文原始设定：n_trials=200, n_total=6000（3000 train + 3000 test）。
    受计算资源限制，默认使用 n_trials=10, n_total=2000；
    增大两者可使结果更接近论文图 Figure 1。

    参数
    ----
    scenarios  : list of str，要测试的场景
    n_trials   : 每个场景运行的独立试验次数（论文: 200）
    alpha      : 显著性水平
    n_total    : 每次试验的总样本量（论文: 6000）
    model_type : 基础生存模型
    run_cdr    : 是否运行 CDR-LPB
    verbose    : 是否显示进度

    返回
    ----
    pd.DataFrame：所有结果
    """
    all_results = []
    for scenario in scenarios:
        if verbose:
            print(f"\n=== 场景: {scenario} ===")
        for trial in range(n_trials):
            seed = trial * 1000 + hash(scenario) % 1000
            res  = run_single_trial(
                seed=seed,
                scenario=scenario,
                alpha=alpha,
                n_total=n_total,
                model_type=model_type,
                run_cdr=run_cdr,
            )
            all_results.append(res)
            if verbose:
                line = (f"  Trial {trial+1:3d}/{n_trials}  "
                        f"cens={res['cens_rate']:.2f}  "
                        f"direct={res['direct_coverage']:.3f}  "
                        f"naive={res['naive_coverage']:.3f}  "
                        f"cmr={res['cmr_coverage']:.3f}")
                if run_cdr:
                    line += f"  cdr={res['cdr_coverage']:.3f}"
                print(line)

    return pd.DataFrame(all_results)
