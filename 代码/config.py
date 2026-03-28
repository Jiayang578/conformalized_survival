import pandas as pd
import numpy as np
# 生成单组生存模拟数据
import pandas as pd
import numpy as np

# 生成单组生存模拟数据 【已修复：协变量真正生效 + 删失正确 + 无偏】
def gen_surv(
    n,
    shape=1.2, scale=365,
    age_mean=65, age_std=10, age_clip=(40,90),
    gender_p=0.5,
    target_censor=0.1
):
    """
    生成服从weibull分布的生存分析数据
    shape:weibull分布的形状参数，为1时则为指数分布
    scale:weibull分布的尺度参数
    age_mean:年龄的平均值
    age_std:年龄的标准差
    age_clip:年龄限制范围
    gender_p:男性概率
    target_censor:删失概率
    """
    # 协变量（完全保留你原来的写法）
    age = np.random.normal(age_mean, age_std, n)
    age = np.clip(age, *age_clip).round(0)
    gender = np.random.binomial(1, gender_p, n)

    beta_age    = -0.015   # 年龄越大，生存时间越短
    beta_gender = 0.25     # 性别1 比 性别0 生存更长（可正可负）
    
    # 线性项 → 影响每个人的尺度参数
    linear = beta_age * (age - age_mean) + beta_gender * gender
    # 每个人的真实生存时间
    t_true = np.random.weibull(shape, n) * scale * np.exp(linear)
    t_true = t_true.round(0)

    # 删失时间（根据目标删失率随机生成）
    max_follow = 4 * scale  # 最长随访时间（独立于T）
    for _ in range(30):
        censor_time = np.random.uniform(0, max_follow, n).round(0)
        t_obs = np.minimum(t_true, censor_time)
        event = (t_obs == t_true).astype(int)
        current_censor = 1 - event.mean()
        if abs(current_censor - target_censor) < 0.005:
            break
        if current_censor > target_censor:
            max_follow *= 1.1
        else:
            max_follow *= 0.9

    return pd.DataFrame({
        'age': age,
        'gender': gender,
        'survival_time': t_obs,
        'event': event
    })