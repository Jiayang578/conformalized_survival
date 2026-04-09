import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter


# ── 模型拟合 ──────────────────────────────────────────────────────────────────

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
    cph = CoxPHFitter(penalizer=0.1)
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


# ── 模型评估 ──────────────────────────────────────────────────────────────────

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


# ── 生存预测 ──────────────────────────────────────────────────────────────────

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
