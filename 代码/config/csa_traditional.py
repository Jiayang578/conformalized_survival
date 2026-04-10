"""
传统 CSA（Conformalized Survival Analysis）
"""
import numpy as np

from .csa_base import (
    estimate_c0_on_train,
    estimate_censoring_weights,
    csa_nonconformity_scores,
    calibrate_csa_quantile
)
from .models import predict_mean_survival_truncated


def fit_csa_intervals_traditional(model, X_train, time_train, event_train,
                                   X_cal, time_cal, event_cal, X_test,
                                   alpha=0.1, model_type='cox', use_weights=True,
                                   verbose=True):
    """
    生成单侧区间 [L, ∞)。
    参数:
        model: 已拟合模型对象
        X_train: numpy.ndarray, 训练集特征 (n_train, n_features) - 用于c0选择
        time_train: numpy.ndarray, 训练集观测时间 - 必须提供
        event_train: numpy.ndarray, 训练集事件指示
        X_cal: numpy.ndarray, 校准集特征 (n_cal, n_features)
        time_cal: numpy.ndarray, 校准集生存时间
        event_cal: numpy.ndarray, 校准集事件指示（0=删失，1=事件）
        X_test: numpy.ndarray, 测试集特征 (n_test, n_features)
        alpha: float, 显著性水平，目标覆盖率 = 1-alpha
        model_type: str, 模型类型 ('km', 'cox', 'weibull', 'rsf')
        use_weights: bool, 是否使用加权conformal推断（推荐True）
        verbose: bool, 是否打印诊断信息

    返回:
        lower: numpy.ndarray, 区间下界，shape=(n_test,)
        upper: numpy.ndarray, 所有元素为 np.inf（表示无上界）
        q_value: float, 校准的平均分位数值
        weight_info: dict, 诊断信息字典
    """
    
    # Step 1: c0 自适应选择

    if use_weights:
        c0, c0_scores = estimate_c0_on_train(
            X_train, time_train, event_train, model, 
            model_type=model_type, verbose=verbose
        )
    else:
        c0 = float(np.median(time_train))
        c0_scores = None
        if verbose:
            print(f"⚠️  无权重模式：c0 = 训练集中位数 {c0:.4f}")
    
    # Step 2: 从训练集估计边际概率 P(C≥c0)
    censor_indicator_train = (time_train >= c0).astype(int)
    p_c0_marginal = float(np.mean(censor_indicator_train))
    
    if verbose:
        print(f"\n✓ 边际概率 P(C≥c0) = {p_c0_marginal:.4f}（来自训练集）")
    
    # Step 3: 在校准集上估计删失概率模型 P(C≥c0|X)，得到权重
    if use_weights:
        weights, censoring_probs, censor_model = estimate_censoring_weights(
            X_train, time_train, event_train,  
            X_cal,                             # 校准集仅用于预测权重
            c0_threshold=c0,
            p_c0_marginal=p_c0_marginal,
            verbose=verbose
        )
    else:
        weights = None
        censoring_probs = None
        censor_model = None
    
    # Step 4: 筛选校准子集 I'_ca
    cal_mask = time_cal >= c0
    
    X_cal_prime     = X_cal[cal_mask]
    time_cal_prime  = time_cal[cal_mask]
    event_cal_prime = event_cal[cal_mask]
    weights_prime   = weights[cal_mask] if weights is not None else None
    
    if verbose:
        n_unc = int(np.sum(event_cal == 1))
        n_cens = int(np.sum((event_cal == 0) & (time_cal >= c0)))
        print(f"\n✓ I'_ca 筛选：{cal_mask.sum()}/{len(time_cal)} 个样本"
              f"（未删失 {n_unc} + 删失且C≥c0 {n_cens}）")
    
    # Step 5: 计算校准集非一致性分数 V = ŷ - min(T̃, c0)
    # 论文 CMR 分数：V = m̂(x) - (T̃ ∧ c0)，其中 m̂(x) = E[T∧c0 | X=x]（条件均值）
    pred_cal_prime = predict_mean_survival_truncated(model, X_cal_prime, c0=c0, model_type=model_type)
    scores = csa_nonconformity_scores(pred_cal_prime, time_cal_prime, c0=c0)
    
    # Step 6: 计算测试集预测值和权重，校准分位数
    pred_test = predict_mean_survival_truncated(model, X_test, c0=c0, model_type=model_type)
    
    if use_weights:
        # 测试点权重：ŵ(x) = P(C≥c0) / P(C≥c0|X=x)
        epsilon = 0.01
        test_censor_probs = censor_model.predict_proba(X_test)[:, 1]
        test_censor_probs = np.clip(test_censor_probs, epsilon, 1.0 - epsilon)
        test_weights = p_c0_marginal / test_censor_probs  # shape=(n_test,)
        
        # 调用新的分位数校准函数（支持向量化和∞-atom）
        q_per_test = calibrate_csa_quantile(
            scores, 
            alpha=alpha,
            weights=weights_prime,  # 校准集权重
            test_weight=test_weights  # 测试点权重（向量形式）
        )
        
        # 下界计算：L(x) = (ŷ - q) ∧ c0
        raw_lower = pred_test - q_per_test
        raw_lower[~np.isfinite(raw_lower)] = 0.0  # ∞-atom激活时下界为0
        lower = np.minimum(raw_lower, c0)  # 仅 clip 到上界c0
        lower = np.maximum(lower, 0.0)     # 然后保证非负
        
        q_value = float(np.nanmean(q_per_test[np.isfinite(q_per_test)]))
        
        if verbose:
            n_inf = int(np.sum(~np.isfinite(q_per_test)))
            print(f"\n✓ 加权分位数 q（均值）= {q_value:.4f}，∞-atom激活 {n_inf}/{len(q_per_test)} 个测试点")
    
    else:
        # 无权重模式：仍加∞-atom
        test_weights = np.ones(X_test.shape[0])  # 默认权重1
        q_per_test = calibrate_csa_quantile(
            scores,
            alpha=alpha,
            weights=None,  # 均匀权重
            test_weight=test_weights  # 【关键】无权重也要加∞-atom
        )
        
        raw_lower = pred_test - q_per_test
        raw_lower[~np.isfinite(raw_lower)] = 0.0
        lower = np.minimum(raw_lower, c0)
        lower = np.maximum(lower, 0.0)
        
        q_value = float(np.nanmean(q_per_test[np.isfinite(q_per_test)]))
        
        if verbose:
            n_inf = int(np.sum(~np.isfinite(q_per_test)))
            print(f"\n✓ 无权重分位数 q（均值）= {q_value:.4f}，∞-atom激活 {n_inf}/{len(q_per_test)} 个测试点")
    
    # Step 7: 构建区间（传统CSA都是单侧）
    upper = np.full_like(lower, np.inf)
    
    # 返回结果和诊断信息
    weight_info = {
        'use_weights': use_weights,
        'c0': c0,
        'c0_scores': c0_scores,
        'p_c0_marginal': p_c0_marginal,
        'weights': weights,
        'censoring_probs': censoring_probs,
        'censor_model': censor_model,
        'cal_mask': cal_mask,
        'test_weights': test_weights,
        'q_per_test': q_per_test,
    }
    
    return lower, upper, q_value, weight_info
