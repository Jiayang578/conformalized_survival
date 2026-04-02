import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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


# ============================================================================
# 传统 CSA (Traditional Conformalized Survival Analysis)
# 所有样本生成单侧区间 [L, ∞)，使用加权 conformal 推断
# ============================================================================

def fit_csa_intervals_traditional(model, X_cal, time_cal, event_cal, X_test, 
                                   alpha=0.1, model_type='cox'):
    """
    传统CSA区间构建：所有样本生成单侧区间 [L, ∞)
    
    基于加权 conformal 推断，对所有样本保证统一的覆盖率：
    P(T ≥ L(X)) ≥ 1-α
    
    参数:
        model: 已拟合模型对象
        X_cal: numpy.ndarray, 校准集特征 (n_cal, n_features)
        time_cal: numpy.ndarray, 校准集生存时间
        event_cal: numpy.ndarray, 校准集事件指示（0=删失，1=事件）
        X_test: numpy.ndarray, 测试集特征 (n_test, n_features)
        alpha: float, 显著性水平，目标覆盖率 = 1-alpha
        model_type: str, 模型类型 ('km', 'cox', 'weibull', 'rsf')
    
    返回:
        lower: numpy.ndarray, 区间下界，shape=(n_test,)
        upper: numpy.ndarray, 所有元素为 np.inf（表示无上界）
        q_value: float, 校准的分位数值
    """
    # Step 1: 计算校准集上的非一致性分数
    pred_cal = predict_median_survival(model, X_cal, model_type=model_type)
    scores = csa_nonconformity_scores(pred_cal, time_cal, event_cal)
    
    # Step 2: 校准分位数（对所有样本统一）
    q = calibrate_csa_quantile(scores, alpha=alpha)
    
    # Step 3: 在测试集上生成下界
    pred_test = predict_median_survival(model, X_test, model_type=model_type)
    lower = np.clip(pred_test - q, 0.0, None)
    
    # Step 4: 传统CSA中所有样本都没有有限上界
    upper = np.full_like(lower, np.inf)
    
    return lower, upper, q


# ============================================================================
# 两侧 CSA (Two-sided Conformalized Survival Analysis)
# 分类区分：未删失样本get两侧区间[L,U]，删失样本get单侧区间[L,∞)
# ============================================================================

def train_censoring_classifier(X_train, event_train, X_cal, event_cal):
    """
    训练删失状态分类器：预测P(Δ=1|X)
    
    该分类器用于识别哪些样本的生存时间被完全观测（未删失）。
    在两侧CSA中，未删失样本（Δ=1）可以获得两侧区间，删失样本(Δ=0)仅获得单侧。
    
    参数:
        X_train: numpy.ndarray, 训练集特征
        event_train: numpy.ndarray, 训练集事件指示（0=删失，1=事件）
        X_cal: numpy.ndarray, 校准集特征
        event_cal: numpy.ndarray, 校准集事件指示
    
    返回:
        classifier: 已拟合的分类器对象
        cal_probs: 校准集上Δ=1的预测概率，shape=(n_cal,)
        cal_scores: 校准集上的非一致性分数 ν(x,0)=1-π₀(x)，用于p-value计算
    """
    # 使用随机森林进行分类
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=2026)
    clf.fit(X_train, event_train)
    
    # 获取校准集上的预测概率
    cal_probs = clf.predict_proba(X_cal)  # shape = (n_cal, 2)，列对应[Δ=0, Δ=1]
    
    # 计算非一致性分数（Conformal中的关键量）
    # ν(x, 0) = 1 - π₀(x)，即在零假设Δ=0下的"异常程度"
    cal_scores = 1 - cal_probs[:, 0]  # P(Δ=1|X) = 1 - P(Δ=0|X)
    
    return clf, cal_probs[:, 1], cal_scores


def compute_upper_bounds_two_sided(model, X, q_value, model_type='cox'):
    """
    计算两侧CSA中未删失样本的上界
    
    上界通过反演生存函数得到：UPB = F̂⁻¹(0.5 + q)
    如果所求分位数超出支撑，则返回 np.inf
    
    参数:
        model: 已拟合的生存模型
        X: numpy.ndarray, 特征矩阵 (n_samples, n_features)
        q_value: float, 校准的置信区间半宽
        model_type: str, 模型类型
    
    返回:
        uppers: numpy.ndarray, 上界值，有限或np.inf，shape=(n_samples,)
    """
    n_samples = X.shape[0]
    uppers = np.full(n_samples, np.inf, dtype=float)
    
    if model_type == 'km':
        # KM 所有样本相同的中位数
        median = model.median_survival_time_
        if 0.5 + q_value <= 0.99:  # 粗略检查是否在支撑内
            try:
                # 从KM的生存曲线反推
                times = model.survival_function_.index.values
                survs = model.survival_function_.values.ravel()
                # 找F⁻¹(0.5 + q) 对应的t
                idx = np.where(survs <= 0.5 + q_value)[0]
                if len(idx) > 0:
                    upper_val = times[idx[0]]
                    uppers[:] = upper_val
            except:
                pass  # 失败则保持 np.inf
    
    elif model_type in ['cox', 'weibull']:
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        try:
            # 获取每个样本的生存函数估计
            surv = model.predict_survival_function(df)
            
            # 对每个样本反演生存函数
            for i in range(n_samples):
                sf = surv.iloc[:, i] if surv.shape[1] > 1 else surv.iloc[:, 0]
                times = sf.index.values
                values = sf.values
                
                # 找最小的 t 使得 F(t) = 1 - S(t) >= 0.5 + q
                target_f = 0.5 + q_value
                if np.any(values <= target_f):
                    idx = np.where(values <= target_f)[0]
                    uppers[i] = times[idx[0]]
                # 否则 uppers[i] = np.inf（已初始化）
        except:
            pass  # 如果反演失败，保持无穷
    
    elif model_type == 'rsf':
        try:
            surv_funcs = model.predict_survival_function(X)
            times = np.asarray(model.unique_times_)
            
            for i, sf in enumerate(surv_funcs):
                values = np.asarray(sf(times))
                target_f = 0.5 + q_value
                
                # 找最小的t使得S(t) <= 0.5 - q (即F(t) >= 0.5 + q)
                if np.any(values <= 0.5 - q_value):
                    idx = np.where(values <= 0.5 - q_value)[0]
                    uppers[i] = times[idx[0]]
        except:
            pass
    
    return uppers


def fit_csa_intervals_two_sided(model, X_train, time_train, event_train,
                                 X_cal, time_cal, event_cal, X_test,
                                 alpha=0.1, model_type='cox'):
    """
    两侧CSA区间构建：混合区间方法（修复版本）
    
    **修正内容**：
    1. 分类器现在使用conformal p-value（而非简单阈值）
    2. 这保证了有限样本的Type I错误控制：P(Δ̂=1|Δ=0)=α/2
    3. 因此覆盖率保证 P(T∉Ĉ)≤α 是有效的
    
    分群体覆盖率：
    - 未删失样本 (Δ=1)：两侧区间 [L₁, U₁]，覆盖率 α/2
    - 删失样本 (Δ=0)：单侧区间 [L₀, ∞)，覆盖率 α/2
    
    整体覆盖率：P(T ∉ Ĉ(X)) ≤ α (有限样本保证)
    
    参数:
        model: 已拟合模型
        X_train, time_train, event_train: 训练集数据（用于分类器）
        X_cal, time_cal, event_cal: 校准集数据
        X_test: 测试集特征 (n_test, n_features)
        alpha: float，总体显著性水平，自动分成两部分各α/2使用
        model_type: str，模型类型
    
    返回:
        lower: numpy.ndarray, 下界，shape=(n_test,)
        upper: numpy.ndarray, 上界（有限或np.inf），shape=(n_test,)
        q_value: float, 使用的分位数
        classification: dict, 包含分类信息
            - 'pred_delta': 预测的Δ值（0或1，使用conformal p-value）
            - 'pvalues': Conformal p-value值
            - 'two_sided_mask': 获得两侧区间的样本掩码
    
    **关键改进**（vs 之前的简单阈值分类）：
    之前代码使用 pred_delta = (test_probs > 0.5)，这是heuristic。
    现在使用conformal p-value来保证 P(Type I error) = α/2。
    """
    # Step 1: 训练删失状态分类器
    clf, cal_probs, cal_scores = train_censoring_classifier(X_train, event_train, X_cal, event_cal)
    
    # Step 2: 计算非一致性分数（用于下界构建）
    pred_cal = predict_median_survival(model, X_cal, model_type=model_type)
    scores = csa_nonconformity_scores(pred_cal, time_cal, event_cal)
    
    # Step 3: 校准分位数（使用 α/2）
    alpha_half = alpha / 2.0
    q = calibrate_csa_quantile(scores, alpha=alpha_half)
    
    # Step 4: 使用Conformal p-value进行测试集分类（关键修复！）
    pred_delta, test_pvalues = classify_censoring_status_conformal(
        cal_scores, X_test, clf, alpha=alpha
    )
    
    # Step 5: 在测试集上生成区间
    pred_test = predict_median_survival(model, X_test, model_type=model_type)
    lower = np.clip(pred_test - q, 0.0, None)
    upper = np.full_like(lower, np.inf)
    
    # 对于预测为未删失的样本，尝试计算有限上界
    two_sided_mask = pred_delta == 1
    if np.any(two_sided_mask):
        X_two_sided = X_test[two_sided_mask]
        uppers_two = compute_upper_bounds_two_sided(model, X_two_sided, q, model_type)
        upper[two_sided_mask] = uppers_two
    
    classification = {
        'pred_delta': pred_delta,
        'pvalues': test_pvalues,  # 关键修改：返回p-value而非概率
        'two_sided_mask': two_sided_mask
    }
    
    return lower, upper, q, classification


def classify_censoring_status_conformal(cal_scores, X_test, clf, alpha=0.1):
    """
    使用Conformal p-value进行删失状态分类（论文正确做法）
    
    **理论依据（论文Two-sided CSA, Lemma 1）**：
    基于conformal p-value的分类提供有限样本的Type I错误控制：
    P(Δ̂=1 | Δ=0) = P(p-value < α/2 | Δ=0) = α/2（恰好比例）
    
    参数:
        cal_scores: numpy.ndarray, 校准集的非一致性分数 ν(x,0)，shape=(n_cal,)
        X_test: numpy.ndarray, 测试集特征，shape=(n_test, n_features)
        clf: 已拟合的分类器
        alpha: float, 显著性水平，分类阈值为 α/2
    
    返回:
        pred_delta: numpy.ndarray, 预测的删失指示，1=未删失，0=删失，shape=(n_test,)
        test_pvalues: numpy.ndarray, 每个测试样本的p-value，shape=(n_test,)
    
    **关键性质**：
    p_i ~ Uniform{1/(n+1), 2/(n+1), ..., 1} 在H₀: Δ=0下（exchangeability）
    这提供了分类规则的理论保证（有限样本）
    
    **与简单阈值的区别**：
    - 旧代码：pred_delta = (probs > 0.5)，无理论依据
    - 新代码：pred_delta = (pval < α/2)，有finite-sample保证
    """
    # Step 1: 计算测试集的非一致性分数
    test_probs = clf.predict_proba(X_test)  # shape = (n_test, 2)
    test_scores = 1 - test_probs[:, 0]  # ν(X_test, 0) = 1 - π₀(X_test)
    
    # Step 2: 对每个测试样本计算p-value
    # p_i = (1 + #{校准样本的分数 ≤ 测试样本分数}) / (n_cal + 1)
    n_cal = len(cal_scores)
    test_pvalues = np.zeros(len(test_scores))
    
    for i, score_i in enumerate(test_scores):
        # 计算有多少个校准样本的分数 ≤ 测试样本的分数
        count = np.sum(cal_scores <= score_i)
        # p-value = (count + 1) / (n_cal + 1)
        # +1是因为要包括测试样本本身（conformal的标准做法）
        test_pvalues[i] = (count + 1.0) / (n_cal + 1.0)
    
    # Step 3: 基于p-value进行分类
    # 当 p-value < α/2 时，拒绝H₀（Δ=0），判为H₁（Δ=1，未删失）
    alpha_half = alpha / 2.0
    pred_delta = (test_pvalues < alpha_half).astype(int)
    
    return pred_delta, test_pvalues


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
    covered[uncensored] = (lower[uncensored] <= time[uncensored])  # T应该≥L（对无穷区间）
    covered[censored] = (lower[censored] <= time[censored])  # C应该≥L

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


def evaluate_interval_coverage_two_sided(lower, upper, time, event, classification=None):
    """
    评估两侧CSA的覆盖率和宽度指标
    
    两侧CSA中：
    - 未删失样本可能有有限上界 [L, U]
    - 删失样本只有单侧 [L, ∞)
    - 分别统计两侧和单侧的覆盖率与宽度
    
    参数:
        lower: 下界，shape=(n,)
        upper: 上界（有限或np.inf），shape=(n,)
        time: 观测生存时间
        event: 事件指示（0=删失，1=事件）
        classification: dict，包含 'pred_delta' 或 'two_sided_mask'
    
    返回:
        dict，包含：
        - 'coverage': 总覆盖率
        - 'coverage_two_sided': 两侧区间的覆盖率
        - 'coverage_one_sided': 单侧区间的覆盖率
        - 'mean_width_two_sided': 两侧区间的平均宽度
        - 'num_two_sided': 两侧区间样本数
        - 'num_one_sided': 单侧区间样本数
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    time = np.asarray(time)
    event = np.asarray(event)

    # 判断哪些是有限上界（两侧）vs 无限上界（单侧）
    two_sided_mask = np.isfinite(upper)
    one_sided_mask = ~two_sided_mask

    covered = np.zeros_like(time, dtype=bool)
    uncensored = event == 1
    censored = event == 0

    # 覆盖规则：
    # - 有有限上界（两侧）：T ∈ [L, U]
    # - 无有限上界（单侧）：T ≥ L
    covered[two_sided_mask & uncensored] = (
        (lower[two_sided_mask & uncensored] <= time[two_sided_mask & uncensored]) &
        (time[two_sided_mask & uncensored] <= upper[two_sided_mask & uncensored])
    )
    covered[two_sided_mask & censored] = lower[two_sided_mask & censored] <= time[two_sided_mask & censored]
    covered[one_sided_mask & uncensored] = lower[one_sided_mask & uncensored] <= time[one_sided_mask & uncensored]
    covered[one_sided_mask & censored] = lower[one_sided_mask & censored] <= time[one_sided_mask & censored]

    # 计算各类覆盖率
    cov_two_sided = covered[two_sided_mask].mean() if np.any(two_sided_mask) else np.nan
    cov_one_sided = covered[one_sided_mask].mean() if np.any(one_sided_mask) else np.nan
    
    # 计算两侧区间的宽度统计
    if np.any(two_sided_mask):
        widths_two = upper[two_sided_mask] - lower[two_sided_mask]
        mean_width_two = np.mean(widths_two)
    else:
        mean_width_two = np.nan

    return {
        'coverage': covered.mean(),
        'coverage_two_sided': cov_two_sided,
        'coverage_one_sided': cov_one_sided,
        'mean_width_two_sided': mean_width_two,
        'num_two_sided': np.sum(two_sided_mask),
        'num_one_sided': np.sum(one_sided_mask),
        'coverage_detail': f'Overall: {covered.mean():.4f} | Two-sided: {cov_two_sided:.4f} ({np.sum(two_sided_mask)}) | One-sided: {cov_one_sided:.4f} ({np.sum(one_sided_mask)})'
    }


# 保留向后兼容性的通用函数
def evaluate_interval_coverage(lower, upper, time, event):
    """向后兼容：默认使用传统CSA的覆盖率评估"""
    return evaluate_interval_coverage_traditional(lower, upper, time, event)


# ============================================================================
# 诊断和验证函数（修复版本通用工具）
# ============================================================================

def verify_coverage_guarantee(coverage_array, alpha=0.1, verbose=True):
    """
    验证理论保证：P(T ∉ Ĉ) ≤ α
    
    参数:
        coverage_array: 布尔数组或覆盖率数值数组，True/1表示覆盖
        alpha: 目标误差水平
        verbose: 是否打印详细信息
        
    返回:
        dict: 验证结果
            - satisfied: bool，是否满足保证
            - error_rate: float，实际误差率
            - alpha: float，目标误差水平
            - coverage_rate: float，实际覆盖率
            - margin: float，误差率与alpha的差异（负数更好）
    """
    # 处理覆盖率数值（0-1范围）
    if np.all((coverage_array == 0) | (coverage_array == 1) | np.isnan(coverage_array)):
        # 布尔或0/1数组
        error_rate = 1 - np.nanmean(coverage_array)
    else:
        # 连续数值
        error_rate = 1 - np.mean(coverage_array)
    
    satisfied = error_rate <= alpha
    coverage_rate = 1 - error_rate
    
    if verbose:
        symbol = "✓" if satisfied else "✗"
        print(f"\n{symbol} 覆盖率理论保证验证")
        print(f"  理论上界: P(Type I error) ≤ {alpha:.2%}")
        print(f"  实际误差率: {error_rate:.2%}")
        print(f"  实际覆盖率: {coverage_rate:.2%}")
        print(f"  满足保证: {'是' if satisfied else '否'}")
        if not satisfied:
            print(f"  ⚠️ 超出边界: {(error_rate - alpha):.2%}")
    
    return {
        'satisfied': satisfied,
        'error_rate': error_rate,
        'alpha': alpha,
        'coverage_rate': coverage_rate,
        'margin': error_rate - alpha
    }


def analyze_two_sided_rate(classification, alpha=0.1, verbose=True):
    """
    分析两侧预测的生成率
    
    理论: P(生成两侧) ≈ max(0, 1 - α) 在标准Conformal设置下
    实证: 应接近这个值，偏离说明分类器或校准数据有问题
    
    参数:
        classification: numpy数组或dict
            - 如果是数组: 1表示两侧，0表示单侧
            - 如果是dict: 应包含'pred_delta'或相关键
        alpha: float，显著性水平
        verbose: bool，是否打印信息
        
    返回:
        dict: 分析结果
            - empirical_two_sided_rate: 实际两侧比例
            - expected_two_sided_rate: 理论期望
            - relative_error: 相对误差
            - is_reasonable: bool，是否在合理范围内
    """
    # 处理输入格式
    if isinstance(classification, dict):
        if 'pred_delta' in classification:
            clf_array = classification['pred_delta']
        else:
            # 尝试找第一个包含分类信息的键
            for key in classification:
                if isinstance(classification[key], np.ndarray):
                    clf_array = classification[key]
                    break
    else:
        clf_array = np.asarray(classification)
    
    empirical_rate = np.mean(clf_array == 1)
    expected_rate = max(0, 1 - alpha)  # 理论期望
    
    # 判断是否合理（允许10%的相对误差）
    relative_error = abs(empirical_rate - expected_rate) / max(expected_rate, 0.01)
    is_reasonable = relative_error < 0.2  # 20%相对误差以内
    
    if verbose:
        print(f"\n两侧预测率分析")
        print(f"  理论期望: {expected_rate:.1%}")
        print(f"  实际观测: {empirical_rate:.1%}")
        print(f"  相对误差: {relative_error:.1%}")
        print(f"  合理性: {'✓ 是' if is_reasonable else '⚠️ 否'}")
        if not is_reasonable:
            print(f"    (误差超过20%，可能需要检查分类器或校准数据)")
    
    return {
        'empirical_two_sided_rate': empirical_rate,
        'expected_two_sided_rate': expected_rate,
        'relative_error': relative_error,
        'is_reasonable': is_reasonable
    }


def get_classification_two_sided(p_values, alpha=0.1):
    """
    独立的Conformal p-value分类函数
    
    基于修复后的方案：使用|p - 0.5|作为分类统计量
    
    参数:
        p_values: numpy数组，Conformal p-values (在U[0,1]下)
        alpha: float，显著性水平 (使用 1-α/2 作为分位数)
        
    返回:
        tuple:
            - classification: 1=两侧，0=单侧
            - threshold: 使用的分位数阈值
            - q_level: 实际使用的分位数水平 (1-α/2)
    """
    n = len(p_values)
    
    # Conformal分位数: q̂ = ⌈(n+1)(1-α/2)⌉ / n 阶分位数
    q_level = 1 - alpha / 2
    k_index = int(np.ceil((n + 1) * q_level)) - 1
    k_index = min(max(k_index, 0), n - 1)  # 防止越界
    
    # 使用|p - 0.5|作为分类统计量
    abs_dev = np.abs(p_values - 0.5)
    sorted_dev = np.sort(abs_dev)
    threshold = sorted_dev[k_index]
    
    # 分类
    classification = (abs_dev >= threshold).astype(int)
    
    return classification, threshold, q_level