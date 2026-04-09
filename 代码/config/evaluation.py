"""
CSA 区间覆盖率与宽度评估
- evaluate_interval_coverage_traditional        : 传统 CSA（单侧）
- evaluate_interval_coverage_two_sided          : 双侧 CSA 分发器（按 data_type 路由）
- evaluate_interval_coverage_two_sided_synthetic: 合成数据专用（真实 T 已知）
- evaluate_interval_coverage_two_sided_real     : 真实数据专用（覆盖率上下界估计）
"""
import numpy as np


def evaluate_interval_coverage_traditional(lower, upper, time, event):
    """
    评估传统CSA的覆盖率和宽度指标

    传统CSA所有样本都是 [L, ∞)，所以：
    - 覆盖标准：未删失 T ∈ [L, U]，删失 T ≥ L
    - 宽度：所有区间都是"无穷"，不计算有限宽度

    参数:
        lower: 下界，shape=(n,)
        upper: 上界（所有值应为np.inf），shape=(n,)
        time: 观测生存时间
        event: 事件指示（0=删失，1=事件）

    返回:
        dict，包含：
        - 'coverage': 总覆盖率
        - 'coverage_uncensored': 未删失样本覆盖率
        - 'coverage_censored': 删失样本覆盖率
        - 'coverage_detail': 详细字符串描述
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    time = np.asarray(time)
    event = np.asarray(event)

    covered = np.zeros_like(time, dtype=bool)
    uncensored = event == 1
    censored = event == 0

    # 覆盖标准：对于[L, ∞)区间
    covered[uncensored] = (lower[uncensored] <= time[uncensored])
    covered[censored] = (lower[censored] <= time[censored])

    cov_uncensored = covered[uncensored].mean() if np.any(uncensored) else np.nan
    cov_censored = covered[censored].mean() if np.any(censored) else np.nan

    return {
        'coverage': covered.mean(),
        'coverage_uncensored': cov_uncensored,
        'coverage_censored': cov_censored,
        'num_uncensored': np.sum(uncensored),
        'num_censored': np.sum(censored),
        'coverage_detail': f'Overall: {covered.mean():.4f} | Uncensored: {cov_uncensored:.4f} | Censored: {cov_censored:.4f}'
    }


def evaluate_interval_coverage_two_sided(lower, upper, time, event, classification=None, data_type='real'):
    """
    评估两侧CSA的覆盖率和宽度指标

    参数:
        lower: 下界，shape=(n,)
        upper: 上界（有限或np.inf），shape=(n,)
        time: 真实生存时间（合成数据）或观测时间（真实数据）
        event: 事件指示（0=删失，1=事件）
        classification: dict，包含 'pred_delta' 或 'two_sided_mask'
        data_type: str，'synthetic'或'real'，决定覆盖度计算策略

    返回:
        dict，包含覆盖率及宽度指标（结构因 data_type 而异）
    """
    if data_type == 'synthetic':
        return evaluate_interval_coverage_two_sided_synthetic(lower, upper, time, event, classification)
    else:
        return evaluate_interval_coverage_two_sided_real(lower, upper, time, event, classification)


def evaluate_interval_coverage_two_sided_synthetic(lower, upper, time, event, classification=None):
    """
    【合成数据专用】覆盖度计算

    关键特性：
    - 所有样本有真实生存时间T
    - 无论真实Δ和预测Δ̂，都用真实T判断覆盖成功/失败
    - 按预测分组（单侧/双侧）展示覆盖率

    覆盖规则：
    - 两侧区间（Δ̂=1）：T ∈ [LB₁, UB₁]
    - 单侧区间（Δ̂=0）：T ∈ [L̂B₀, +∞)

    参数:
        lower: 下界数组，shape=(n,)
        upper: 上界数组（有限或np.inf），shape=(n,)
        time: 真实生存时间，shape=(n,)
        event: 事件指示/删失指示，shape=(n,)；此参数在合成数据中仅用于记录真实Δ，不影响覆盖度计算
        classification: dict，包含'two_sided_mask'标记两侧区间样本

    返回:
        dict，覆盖度及宽度指标
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    time = np.asarray(time)

    # 判断预测分组：两侧（有限上界）vs 单侧（无限上界）
    two_sided_mask = np.isfinite(upper)
    one_sided_mask = ~two_sided_mask

    # 两侧区间：T ∈ [L, U]
    covered_two_sided = (lower[two_sided_mask] <= time[two_sided_mask]) & \
                        (time[two_sided_mask] <= upper[two_sided_mask])

    # 单侧区间：T ≥ L
    covered_one_sided = lower[one_sided_mask] <= time[one_sided_mask]

    cov_two_sided = covered_two_sided.mean() if np.any(two_sided_mask) else np.nan
    cov_one_sided = covered_one_sided.mean() if np.any(one_sided_mask) else np.nan
    overall_coverage = np.concatenate([covered_two_sided, covered_one_sided]).mean() if len(time) > 0 else np.nan

    if np.any(two_sided_mask):
        widths_two = upper[two_sided_mask] - lower[two_sided_mask]
        mean_width_two = np.mean(widths_two)
        std_width_two = np.std(widths_two)
    else:
        mean_width_two = np.nan
        std_width_two = np.nan

    return {
        'coverage': overall_coverage,
        'coverage_two_sided': cov_two_sided,
        'coverage_one_sided': cov_one_sided,
        'mean_width_two_sided': mean_width_two,
        'std_width_two_sided': std_width_two,
        'num_two_sided': np.sum(two_sided_mask),
        'num_one_sided': np.sum(one_sided_mask),
        'coverage_detail': f'Overall: {overall_coverage:.4f} | Two-sided: {cov_two_sided:.4f} ({np.sum(two_sided_mask)}) | One-sided: {cov_one_sided:.4f} ({np.sum(one_sided_mask)})'
    }


def evaluate_interval_coverage_two_sided_real(lower, upper, time, event, classification=None):
    """
    【真实数据专用】覆盖度下界/上界计算

    关键特性：
    - 未删失样本（Δ=1）有真实T，可精确计算覆盖成功/失败
    - 删失样本（Δ=0）仅有截尾时间T̃，无法精确验证T是否覆盖
      无覆盖概率的下界/上界估计：
      * 下界（cov_lo）：保守估计，假设删失样本都未覆盖（最差情况）
      * 上界（cov_up）：宽松估计，仅当T̃超出上界时判为未覆盖（最优情况）
    - 按预测分组（单侧/双侧）计算覆盖率
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    time = np.asarray(time)
    event = np.asarray(event)

    # 判断预测分组：两侧（有限上界）vs 单侧（无限上界）
    two_sided_mask = np.isfinite(upper)
    one_sided_mask = ~two_sided_mask
    uncensored = event == 1
    censored = event == 0

    # ========== 两侧区间（Δ̂=1）==========
    two_censored = two_sided_mask & censored
    two_uncensored = two_sided_mask & uncensored

    two_cens_uncovered_lo = np.sum(two_censored)  # 保守：全部未覆盖
    two_cens_uncovered_up = np.sum((time[two_censored] >= upper[two_censored]) if np.any(two_censored) else False)

    two_uncens_covered = np.sum((lower[two_uncensored] <= time[two_uncensored]) &
                                (time[two_uncensored] <= upper[two_uncensored])) if np.any(two_uncensored) else 0
    two_uncens_total = np.sum(two_uncensored)

    n_two = np.sum(two_sided_mask)
    uncovered_lo_two = two_cens_uncovered_lo + (two_uncens_total - two_uncens_covered)
    uncovered_up_two = two_cens_uncovered_up + (two_uncens_total - two_uncens_covered)
    cov_lo_two = 1 - uncovered_lo_two / n_two if n_two > 0 else np.nan
    cov_up_two = 1 - uncovered_up_two / n_two if n_two > 0 else np.nan

    # ========== 单侧区间（Δ̂=0）==========
    one_censored = one_sided_mask & censored
    one_uncensored = one_sided_mask & uncensored

    one_uncens_covered = np.sum(lower[one_uncensored] <= time[one_uncensored]) if np.any(one_uncensored) else 0
    one_uncens_total = np.sum(one_uncensored)

    one_cens_uncovered_lo = np.sum(time[one_censored] < lower[one_censored]) if np.any(one_censored) else 0
    one_cens_uncovered_up = 0  # 宽松假设所有覆盖成功

    n_one = np.sum(one_sided_mask)
    uncovered_lo_one = (one_uncens_total - one_uncens_covered) + one_cens_uncovered_lo
    uncovered_up_one = (one_uncens_total - one_uncens_covered) + one_cens_uncovered_up
    cov_lo_one = 1 - uncovered_lo_one / n_one if n_one > 0 else np.nan
    cov_up_one = 1 - uncovered_up_one / n_one if n_one > 0 else np.nan

    # 总体覆盖度
    n_total = len(time)
    uncovered_lo_total = uncovered_lo_two + uncovered_lo_one
    uncovered_up_total = uncovered_up_two + uncovered_up_one
    cov_lo_total = 1 - uncovered_lo_total / n_total if n_total > 0 else np.nan
    cov_up_total = 1 - uncovered_up_total / n_total if n_total > 0 else np.nan

    if np.any(two_sided_mask):
        widths_two = upper[two_sided_mask] - lower[two_sided_mask]
        mean_width_two = np.mean(widths_two)
        std_width_two = np.std(widths_two)
        two_sided_ratio = np.sum(two_sided_mask) / n_total
    else:
        mean_width_two = np.nan
        std_width_two = np.nan
        two_sided_ratio = 0.0

    return {
        'cov_lo': cov_lo_total,
        'cov_lo_two_sided': cov_lo_two,
        'cov_lo_one_sided': cov_lo_one,
        'cov_up': cov_up_total,
        'cov_up_two_sided': cov_up_two,
        'cov_up_one_sided': cov_up_one,
        'mean_width_two_sided': mean_width_two,
        'std_width_two_sided': std_width_two,
        'two_sided_ratio': two_sided_ratio,
        'num_two_sided': np.sum(two_sided_mask),
        'num_one_sided': np.sum(one_sided_mask),
        'num_uncensored': np.sum(uncensored),
        'num_censored': np.sum(censored),
    }
