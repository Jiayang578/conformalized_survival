import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter

np.random.seed(2026)

def weibull_from_mu_sigma(mu, sigma, size, random_state=None):
    """
    mu:weibull分布的均值
    sigma:weibull分布的标准层
    size:生成数据的大小
    random_state:随机种子
    return：weibull分布的随机样本
    该函数通过输入均值和标准差，返回符合weibull分布的样本
    """
    rng = np.random.default_rng(random_state)
    samples = np.zeros(size)

    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)

    for i in range(size):
        mu_i = mu[i] if len(mu) > 1 else mu[0]
        sigma_i = sigma[i] if len(sigma) > 1 else sigma[0]
        cv = sigma_i / mu_i
        cv2 = cv ** 2

        def equation(k):
            g1 = gamma(1 + 1 / k)
            g2 = gamma(1 + 2 / k)
            return g2 / (g1 ** 2) - 1 - cv2

        k = fsolve(equation, 1.0, maxfev=100)[0]
        lam = mu_i / gamma(1 + 1 / k)
        u = rng.uniform(low=1e-10, high=1 - 1e-10)
        samples[i] = lam * (-np.log(1 - u)) ** (1 / k)

    return samples

def generate_weibull_data(n, p=1, hetero=False, cens_rate=0.4):
    """
    该函数能够生成指定条件的，服从weibull分布的生存分析数据
    n: 样本量
    p: 协变量维度
    hetero: True=异方差，False=同方差
    cens_rate: 删失率参数（指数分布的rate参数λ）
    """
    # 生成协变量X
    if p == 1:
        X = np.random.uniform(0, 4, size=(n, 1))
    else:
        age = np.random.uniform(18, 80, size=n)
        gender = np.random.binomial(1, 0.5, size=n)
        other_X = np.random.uniform(-1, 1, size=(n, p-2))
        X = np.column_stack([age, gender, other_X])

    # 生成均值μ(X)，单维度情况下仅与单协变量x相关，多维度情况下与年龄性别相关
    if p == 1:
        mu = 2 + 0.37 * np.sqrt(X[:, 0])
    else:
        age = X[:, 0]
        gender = X[:, 1]
        mu = np.log(2) + 1 + 0.6 * (age**2 / 1000 - age * gender / 10)

    # 生成方差σ(X)，同方差时固定，异方差时与某些协变量相关
    if p == 1:
        sigma = 1.5 if not hetero else 1 + X[:, 0] / 5
    else:
        sigma = 1.0 if not hetero else (np.abs(X[:, 9]) + 1) if X.shape[1] >= 10 else 1.0

    # 生成真实生存时间
    T = weibull_from_mu_sigma(mu, sigma, size=n, random_state=2026)

    # 二分搜索找到合适的λ，使得删失率接近目标
    def get_cens_rate_for_lambda(lam):
        C_temp = np.random.exponential(scale=1/lam, size=n)
        return np.mean(C_temp < T)

    lam_low, lam_high = 0.001, 10.0
    for _ in range(20):  # 20次迭代足够精确
        lam_mid = (lam_low + lam_high) / 2
        actual_rate = get_cens_rate_for_lambda(lam_mid)
        if actual_rate < cens_rate:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    # 生成最终删失时间
    optimal_lambda = (lam_low + lam_high) / 2
    C = np.random.exponential(scale=1/optimal_lambda, size=n)

    # 构造观测数据
    time = np.minimum(T, C)
    time = np.clip(time, 1e-8, None)
    event = (T <= C).astype(int)

    # 封装生存数据
    surv = Surv.from_arrays(event=event, time=time)

    return X, surv, time, event, np.mean(1 - event)

#  模型拟合函数 
def fit_kaplan_meier(time, event):
    """拟合Kaplan-Meier模型"""
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=event)
    return kmf

def fit_cox(X, time, event):
    """拟合Cox比例风险模型"""
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    df['time'] = time
    df['event'] = event
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col='time', event_col='event')
    return cph

def fit_weibull(X, time, event):
    """拟合Weibull AFT模型"""
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    df['time'] = time
    df['event'] = event
    wf = WeibullAFTFitter(penalizer=0.1)
    wf._scipy_fit_method = "SLSQP"
    wf.fit(df, duration_col='time', event_col='event', fit_options={'maxiter': 1000})
    return wf

def fit_rsf(X, time, event):
    """拟合随机生存森林"""
    y = Surv.from_arrays(event=event.astype(bool), time=time)
    rsf = RandomSurvivalForest(n_estimators=100, random_state=2026)
    rsf.fit(X, y)
    return rsf

# 模型评估函数 

def evaluate_model(model, X, time, event, model_type='cox'):
    """评估模型性能，返回C-index"""
    if model_type == 'km':
        return None  # KM无法计算C-index

    y = Surv.from_arrays(event=event.astype(bool), time=time)

    if model_type == 'cox':
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        risk_scores = -model.predict_partial_hazard(df).values
    elif model_type == 'weibull':
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        risk_scores = -model.predict_median(df).values
    else:  # rsf
        risk_scores = model.predict(X)

    c_index = concordance_index_censored(event.astype(bool), time, risk_scores)[0]
    return c_index


#  保形生存预测辅助函数 
def predict_median_survival(model, X, model_type='cox'):
    """预测每个样本的中位生存时间
    
    参数:
        model: 已拟合的生存模型
        X: 协变量矩阵，shape=(n_samples, n_features)
        model_type: 模型类型 ('km', 'cox', 'weibull', 'rsf')
    
    返回:
        每个样本的中位生存时间预测值，shape=(n_samples,)
    """
    if model_type == 'km':
        # KM 模型没有协变量，所有样本中位生存时间相同
        return np.full(X.shape[0], model.median_survival_time_, dtype=float)

    if model_type in ['cox', 'weibull']:
        # 使用模型内置的 predict_median 方法
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        med = np.asarray(model.predict_median(df)).reshape(-1)
        return med.astype(float)

    if model_type == 'rsf':
        # RSF 从生存函数反推中位生存时间
        surv_funcs = model.predict_survival_function(X)
        times = np.asarray(model.unique_times_)
        medians = []
        
        for fn in surv_funcs:
            values = np.asarray(fn(times))
            # 找第一个生存概率 <= 0.5 的时间点（对应中位生存时间）
            idx = np.where(values <= 0.5)[0]
            med_time = times[idx[0]] if len(idx) > 0 else times[-1]
            medians.append(med_time)
        
        return np.asarray(medians, dtype=float)

    raise ValueError(f'未知 model_type: {model_type}')


def csa_nonconformity_scores(pred_median, time, event):
    """计算非一致性分数
    
    参数:
        pred_median: 预测的中位生存时间
        time: 观测生存时间
        event: 事件指示（0=删失，1=事件）
    
    返回:
        非一致性分数（衡量预测与观测的差异）
    """
    score = np.zeros_like(time, dtype=float)
    uncensored = event == 1
    censored = event == 0
    
    # 未删失样本：绝对误差
    score[uncensored] = np.abs(time[uncensored] - pred_median[uncensored])
    
    # 删失样本：单侧惩罚（只在预测高估时有分数）
    score[censored] = np.maximum(0.0, pred_median[censored] - time[censored])
    
    return score


def calibrate_csa_quantile(scores, alpha=0.1):
    """校准 CSA 置信区间的半宽
    
    参数:
        scores: 校准集的非一致性分数
        alpha: 显著性水平（目标覆盖率 = 1-alpha）
    
    返回:
        置信区间的半宽 q（保证至少 1-alpha 的校准样本不一致性 < q）
    """
    n = len(scores)
    # 计算第 k 大分数，其中 k = ceil((n+1)*(1-alpha))
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return np.sort(scores)[k - 1]




def split_survival_data(X, time, event, test_size=0.2, cal_size=0.25, random_state=None):
    """按比例拆分生存数据为训练、校准和测试集。

    参数:
        X: numpy.ndarray, 协变量矩阵。
        time: numpy.ndarray, 生存时间。
        event: numpy.ndarray, 事件指示。
        test_size: float, 测试集比例。
        cal_size: float, 校准集占剩余训练集的比例。
        random_state: int or None。

    返回:
        X_train, time_train, event_train,
        X_cal, time_cal, event_cal,
        X_test, time_test, event_test
    """
    X_temp, X_test, time_temp, time_test, event_temp, event_test = train_test_split(
        X, time, event, test_size=test_size, random_state=random_state, stratify=event
    )
    X_train, X_cal, time_train, time_cal, event_train, event_cal = train_test_split(
        X_temp, time_temp, event_temp,
        test_size=cal_size, random_state=random_state, stratify=event_temp
    )
    return X_train, time_train, event_train, X_cal, time_cal, event_cal, X_test, time_test, event_test


def fit_csa_intervals(model, X_cal, time_cal, event_cal, X_test, alpha=0.1, model_type='cox'):
    """基于预测中位生存时间的 CSA 区间构建。

    参数:
        model: 已拟合模型对象。
        X_cal: numpy.ndarray, 校准集特征。
        time_cal: numpy.ndarray, 校准集生存时间。
        event_cal: numpy.ndarray, 校准集事件指示。
        X_test: numpy.ndarray, 测试集特征。
        alpha: float, 1-alpha 为目标覆盖概率。
        model_type: str, 模型类型。

    返回:
        lower: numpy.ndarray, 区间下界。
        upper: numpy.ndarray, 区间上界。
        q_value: float, 半宽 q。
    """
    pred_cal = predict_median_survival(model, X_cal, model_type=model_type)
    scores = csa_nonconformity_scores(pred_cal, time_cal, event_cal)
    q = calibrate_csa_quantile(scores, alpha=alpha)
    pred_test = predict_median_survival(model, X_test, model_type=model_type)
    lower = np.clip(pred_test - q, 0.0, None)
    upper = pred_test + q
    return lower, upper, q


def evaluate_interval_coverage(lower, upper, time, event):
    """计算预测区间覆盖率和区间宽度指标。"""
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    time = np.asarray(time)
    event = np.asarray(event)

    covered = np.zeros_like(time, dtype=bool)
    uncensored = event == 1
    censored = event == 0
    covered[uncensored] = (lower[uncensored] <= time[uncensored]) & (time[uncensored] <= upper[uncensored])
    covered[censored] = lower[censored] <= time[censored]

    widths = upper - lower
    half_widths = widths / 2.0
    mean_width = np.nanmean(widths)
    return {
        'coverage': covered.mean(),
        'coverage_uncensored': covered[uncensored].mean() if np.any(uncensored) else np.nan,
        'coverage_censored': covered[censored].mean() if np.any(censored) else np.nan,
        'mean_width': mean_width,
        'width_cv': np.nanstd(widths, ddof=0) / mean_width if mean_width > 0 else np.nan,
        'q_value': np.nanmean(half_widths)
    }