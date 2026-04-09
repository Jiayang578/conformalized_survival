import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split
from sksurv.util import Surv


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
    # time_obs：观测时间 min(T,C)，用于模型拟合与CSA校准
    # time_true：真实事件时间 T，仅用于合成数据的覆盖率评估
    event    = (T <= C).astype(int)
    time_obs  = np.clip(np.minimum(T, C), 1e-8, None)
    time_true = np.clip(T, 1e-8, None)

    # 封装生存数据（使用正确的观测时间）
    surv = Surv.from_arrays(event=event, time=time_obs)

    return X, surv, time_obs, event, np.mean(1 - event), time_true


def split_survival_data(X, time, event, time_true=None, test_size=0.2, cal_size=0.25, random_state=None):
    """按比例拆分生存数据为训练、校准和测试集。

    参数:
        X: numpy.ndarray, 协变量矩阵。
        time: numpy.ndarray, 观测时间 min(T, C)，用于模型拟合/CSA。
        event: numpy.ndarray, 事件指示。
        time_true: numpy.ndarray or None, 真实事件时间 T（仅合成数据可用），
                   用于覆盖率评估。若提供，返回值额外包含 ttrue_train/cal/test。
        test_size: float, 测试集比例。
        cal_size: float, 校准集占剩余训练集的比例。
        random_state: int or None。

    返回:
        若 time_true is None（默认）:
            X_train, time_train, event_train,
            X_cal, time_cal, event_cal,
            X_test, time_test, event_test          (9 values)
        若 time_true is not None:
            X_train, time_train, event_train,
            X_cal, time_cal, event_cal,
            X_test, time_test, event_test,
            ttrue_train, ttrue_cal, ttrue_test      (12 values)
    """
    if time_true is not None:
        (X_temp, X_test,
         time_temp, time_test,
         event_temp, event_test,
         ttrue_temp, ttrue_test) = train_test_split(
            X, time, event, time_true,
            test_size=test_size, random_state=random_state, stratify=event
        )
        (X_train, X_cal,
         time_train, time_cal,
         event_train, event_cal,
         ttrue_train, ttrue_cal) = train_test_split(
            X_temp, time_temp, event_temp, ttrue_temp,
            test_size=cal_size, random_state=random_state, stratify=event_temp
        )
        return (X_train, time_train, event_train,
                X_cal, time_cal, event_cal,
                X_test, time_test, event_test,
                ttrue_train, ttrue_cal, ttrue_test)
    else:
        X_temp, X_test, time_temp, time_test, event_temp, event_test = train_test_split(
            X, time, event, test_size=test_size, random_state=random_state, stratify=event
        )
        X_train, X_cal, time_train, time_cal, event_train, event_cal = train_test_split(
            X_temp, time_temp, event_temp,
            test_size=cal_size, random_state=random_state, stratify=event_temp
        )
        return X_train, time_train, event_train, X_cal, time_cal, event_cal, X_test, time_test, event_test
