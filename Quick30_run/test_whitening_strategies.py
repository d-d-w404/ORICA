#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同白化策略的效果
"""

import numpy as np
from ORICA import ORICA

def test_whitening_strategies():
    """测试不同白化策略的效果"""
    print("=== 测试不同白化策略的效果 ===")
    
    # 模拟EEG数据
    n_channels = 25
    n_samples = 2500
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_channels)
    
    # 添加一些非高斯成分来模拟真实EEG
    X[:, 0] = np.sign(X[:, 0]) * np.abs(X[:, 0])**1.5  # 超高斯
    X[:, 1] = np.tanh(X[:, 1])  # 次高斯
    
    strategies = [
        ("传统SVD白化", {"use_rls_whitening": False}),
        ("纯RLS白化", {"use_rls_whitening": True, "hybrid_whitening": False}),
        ("混合白化", {"use_rls_whitening": True, "hybrid_whitening": True}),
    ]
    
    results = {}
    
    for name, params in strategies:
        print(f"\n--- 测试 {name} ---")
        try:
            # 创建ORICA实例
            orica = ORICA(
                n_components=n_channels,
                learning_rate=0.001,
                ortho_every=10,
                nonlinearity='tanh',
                **params
            )
            
            # 初始化
            orica.initialize(X)
            
            # 在线处理
            sources_list = []
            for i in range(n_samples):
                x_t = X[i, :]
                result = orica.partial_fit(x_t)
                sources_list.append(result)
            
            sources = np.array(sources_list).T
            
            # 评估结果
            kurtosis = orica.evaluate_separation(sources)
            kurtosis_mean = np.mean(np.abs(kurtosis))
            
            # 计算互信息
            from sklearn.metrics import mutual_info_score
            mi_values = []
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    mi = mutual_info_score(
                        np.digitize(sources[i], np.histogram(sources[i], bins=10)[1]),
                        np.digitize(sources[j], np.histogram(sources[j], bins=10)[1])
                    )
                    mi_values.append(mi)
            mi_mean = np.mean(mi_values)
            
            results[name] = {
                'kurtosis_mean': kurtosis_mean,
                'mi_mean': mi_mean,
                'sources': sources
            }
            
            print(f"✅ {name} 完成")
            print(f"  峰度均值: {kurtosis_mean:.3f}")
            print(f"  互信息均值: {mi_mean:.3f}")
            
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            results[name] = None
    
    # 比较结果
    print("\n=== 结果比较 ===")
    print("策略\t\t峰度均值\t互信息均值\t评价")
    print("-" * 50)
    
    for name, result in results.items():
        if result is not None:
            kurt = result['kurtosis_mean']
            mi = result['mi_mean']
            
            # 评价标准
            if kurt > 2.0 and mi < 0.02:
                rating = "优秀"
            elif kurt > 1.0 and mi < 0.05:
                rating = "良好"
            elif kurt > 0.5 and mi < 0.1:
                rating = "一般"
            else:
                rating = "较差"
            
            print(f"{name:<12}\t{kurt:.3f}\t\t{mi:.3f}\t\t{rating}")
    
    return results

def test_parameter_sensitivity():
    """测试参数敏感性"""
    print("\n=== 测试RLS参数敏感性 ===")
    
    n_channels = 25
    n_samples = 1000
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_channels)
    
    # 测试不同的遗忘因子
    forgetting_factors = [0.95, 0.97, 0.98, 0.99, 0.995]
    
    for ff in forgetting_factors:
        print(f"\n--- 测试遗忘因子 {ff} ---")
        try:
            orica = ORICA(
                n_components=n_channels,
                use_rls_whitening=True,
                hybrid_whitening=True,
                forgetting_factor=ff,
                nonlinearity='tanh'
            )
            
            orica.initialize(X)
            
            sources_list = []
            for i in range(n_samples):
                x_t = X[i, :]
                result = orica.partial_fit(x_t)
                sources_list.append(result)
            
            sources = np.array(sources_list).T
            
            kurtosis = orica.evaluate_separation(sources)
            kurtosis_mean = np.mean(np.abs(kurtosis))
            
            print(f"  峰度均值: {kurtosis_mean:.3f}")
            
        except Exception as e:
            print(f"  失败: {e}")

def main():
    """主函数"""
    print("白化策略效果比较测试")
    print("=" * 50)
    
    # 测试不同白化策略
    results = test_whitening_strategies()
    
    # 测试参数敏感性
    test_parameter_sensitivity()
    
    print("\n=== 建议 ===")
    print("1. 如果RLS效果不好，建议使用混合白化策略")
    print("2. 遗忘因子建议设置在0.97-0.99之间")
    print("3. 确保有足够的数据进行训练（至少1000个样本）")

if __name__ == "__main__":
    main() 