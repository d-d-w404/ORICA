#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复熵计算问题并改进ORICA
"""

import numpy as np
from scipy.stats import kurtosis, entropy
import matplotlib.pyplot as plt

def fix_entropy_calculation(y_t):
    """修复熵计算问题"""
    # 方法1: 使用scipy的entropy函数
    hist, bin_edges = np.histogram(y_t, bins=20, density=True)
    # 移除零概率，避免log(0)
    hist = hist[hist > 1e-10]
    if len(hist) > 0:
        entropy_scipy = entropy(hist)
    else:
        entropy_scipy = 0
    
    # 方法2: 使用差分熵估计
    # 对于连续变量，使用差分熵
    var_y = np.var(y_t)
    if var_y > 0:
        # 假设高斯分布的差分熵
        diff_entropy = 0.5 * np.log(2 * np.pi * np.e * var_y)
    else:
        diff_entropy = 0
    
    # 方法3: 使用k-近邻熵估计
    from sklearn.neighbors import NearestNeighbors
    if len(y_t) > 10:
        y_reshaped = y_t.reshape(-1, 1)
        nbrs = NearestNeighbors(n_neighbors=2).fit(y_reshaped)
        distances, _ = nbrs.kneighbors(y_reshaped)
        # 使用最近邻距离估计熵
        knn_entropy = np.mean(np.log(distances[:, 1] + 1e-10))
    else:
        knn_entropy = 0
    
    return {
        'scipy_entropy': entropy_scipy,
        'diff_entropy': diff_entropy,
        'knn_entropy': knn_entropy
    }

def analyze_entropy_issue():
    """分析熵计算问题"""
    print("=== 分析熵计算问题 ===")
    
    # 生成不同分布的测试数据
    n_samples = 1000
    
    # 1. 高斯分布
    gaussian_data = np.random.randn(n_samples)
    
    # 2. 均匀分布
    uniform_data = np.random.uniform(-2, 2, n_samples)
    
    # 3. 拉普拉斯分布
    laplace_data = np.random.laplace(0, 1, n_samples)
    
    # 4. 混合分布（模拟ICA输出）
    mixed_data = np.random.randn(n_samples)
    mixed_data[:500] = np.sign(mixed_data[:500]) * np.abs(mixed_data[:500])**1.5
    
    test_data = {
        'Gaussian': gaussian_data,
        'Uniform': uniform_data,
        'Laplace': laplace_data,
        'Mixed': mixed_data
    }
    
    print("\n📊 不同分布的熵计算结果:")
    for name, data in test_data.items():
        entropy_results = fix_entropy_calculation(data)
        print(f"\n{name}分布:")
        print(f"  Scipy熵: {entropy_results['scipy_entropy']:.3f}")
        print(f"  差分熵: {entropy_results['diff_entropy']:.3f}")
        print(f"  KNN熵: {entropy_results['knn_entropy']:.3f}")
        print(f"  方差: {np.var(data):.3f}")
        print(f"  峰度: {kurtosis(data):.3f}")
    
    # 绘制分布图
    plt.figure(figsize=(15, 10))
    
    for i, (name, data) in enumerate(test_data.items()):
        plt.subplot(2, 2, i+1)
        plt.hist(data, bins=30, alpha=0.7, density=True)
        plt.title(f'{name}分布')
        plt.xlabel('值')
        plt.ylabel('密度')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('entropy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def improved_orica_entropy():
    """改进的ORICA熵计算"""
    class ORICAEntropyFixed:
        def __init__(self, n_components, learning_rate=0.01, ortho_every=10):
            self.n_components = n_components
            self.learning_rate = learning_rate
            self.ortho_every = ortho_every
            self.W = np.eye(n_components)
            self.mean = None
            self.whitening_matrix = None
            self.whitened = False
            self.update_count = 0
            
            # 熵计算历史
            self.entropy_history = []
            self.spatial_concentration_history = []
        
        def _compute_entropy_fixed(self, y_t):
            """修复的熵计算"""
            # 使用差分熵估计
            var_y = np.var(y_t)
            if var_y > 1e-10:
                # 差分熵
                diff_entropy = 0.5 * np.log(2 * np.pi * np.e * var_y)
                
                # 添加非高斯性修正
                kurt = kurtosis(y_t)
                non_gaussian_correction = 0.1 * np.abs(kurt)
                
                corrected_entropy = diff_entropy + non_gaussian_correction
            else:
                corrected_entropy = 0
            
            return corrected_entropy
        
        def _compute_spatial_concentration(self):
            """计算空间集中度"""
            if not hasattr(self, 'W') or self.W is None:
                return 1.0
            
            # 计算W矩阵的空间集中度
            W_abs = np.abs(self.W)
            max_weights = np.max(W_abs, axis=1)
            mean_weights = np.mean(W_abs, axis=1)
            
            # 避免除零
            mean_weights = np.maximum(mean_weights, 1e-10)
            concentration = np.mean(max_weights / mean_weights)
            
            return concentration
        
        def initialize(self, X_init):
            """初始化"""
            self.mean = np.mean(X_init, axis=0)
            X_centered = X_init - self.mean
            
            # 白化
            cov = np.cov(X_centered, rowvar=False)
            d, E = np.linalg.eigh(cov)
            D_inv = np.diag(1.0 / np.sqrt(d + 1e-8))
            self.whitening_matrix = E @ D_inv @ E.T
            
            self.whitened = True
            print("✅ 初始化完成")
        
        def partial_fit(self, x_t):
            """单个样本更新"""
            if not self.whitened:
                raise ValueError("Must initialize first")
            
            # 去均值
            x_t = x_t - self.mean
            
            # 白化
            x_whitened = self.whitening_matrix @ x_t
            
            # ICA更新
            y_t = self.W @ x_whitened
            
            # 非线性函数
            g_y = y_t * np.exp(-y_t**2/2)
            
            # 更新规则
            I = np.eye(self.n_components)
            delta_W = self.learning_rate * ((I - g_y @ y_t.T) @ self.W)
            self.W += delta_W
            
            # 正交化
            self.update_count += 1
            if self.update_count % self.ortho_every == 0:
                U, _, Vt = np.linalg.svd(self.W)
                self.W = U @ Vt
                
                # 计算并记录指标
                entropy_val = self._compute_entropy_fixed(y_t)
                spatial_conc = self._compute_spatial_concentration()
                
                self.entropy_history.append(entropy_val)
                self.spatial_concentration_history.append(spatial_conc)
                
                # 打印监控信息
                if self.update_count % 100 == 0:
                    print(f"  步骤 {self.update_count}: 熵={entropy_val:.3f}, "
                          f"空间集中度={spatial_conc:.3f}")
            
            return y_t
        
        def get_metrics(self):
            """获取当前指标"""
            if len(self.entropy_history) == 0:
                return None
            
            return {
                'entropy': self.entropy_history[-1],
                'spatial_concentration': self.spatial_concentration_history[-1],
                'entropy_trend': np.mean(self.entropy_history[-10:]) if len(self.entropy_history) >= 10 else 0,
                'concentration_trend': np.mean(self.spatial_concentration_history[-10:]) if len(self.spatial_concentration_history) >= 10 else 0
            }
    
    return ORICAEntropyFixed

def test_improved_entropy():
    """测试改进的熵计算"""
    print("\n=== 测试改进的熵计算 ===")
    
    # 生成测试数据
    n_channels = 25
    n_samples = 10000
    
    # 生成混合信号
    sources = np.random.randn(n_channels, n_samples)
    sources[0, :] = np.sign(sources[0, :]) * np.abs(sources[0, :])**1.5
    sources[1, :] = np.tanh(sources[1, :])
    
    # 生成混合矩阵
    A = np.random.randn(n_channels, n_channels)
    A = A / np.linalg.norm(A, axis=0)
    
    # 混合信号
    X = A @ sources
    
    # 创建改进的ORICA实例
    ORICAFixed = improved_orica_entropy()
    orica = ORICAFixed(
        n_components=n_channels,
        learning_rate=0.01,
        ortho_every=10
    )
    
    # 初始化
    orica.initialize(X[:1000].T)
    
    # 在线学习
    print("开始在线学习...")
    for i in range(1000, len(X), 100):
        batch = X[:, i:i+100].T
        
        for x_t in batch:
            result = orica.partial_fit(x_t)
        
        if i % 2000 == 0:
            metrics = orica.get_metrics()
            if metrics:
                print(f"  步骤 {i}: 熵={metrics['entropy']:.3f}, "
                      f"空间集中度={metrics['spatial_concentration']:.3f}")
    
    print("✅ 学习完成")
    
    # 分析最终结果
    final_metrics = orica.get_metrics()
    if final_metrics:
        print(f"\n📊 最终指标:")
        print(f"  熵: {final_metrics['entropy']:.3f}")
        print(f"  空间集中度: {final_metrics['spatial_concentration']:.3f}")
        print(f"  熵趋势: {final_metrics['entropy_trend']:.3f}")
        print(f"  集中度趋势: {final_metrics['concentration_trend']:.3f}")
    
    # 绘制结果
    plt.figure(figsize=(15, 5))
    
    # 熵变化
    plt.subplot(1, 3, 1)
    plt.plot(orica.entropy_history)
    plt.title('熵变化')
    plt.xlabel('更新步骤')
    plt.ylabel('熵')
    plt.grid(True)
    
    # 空间集中度变化
    plt.subplot(1, 3, 2)
    plt.plot(orica.spatial_concentration_history)
    plt.title('空间集中度变化')
    plt.xlabel('更新步骤')
    plt.ylabel('空间集中度')
    plt.grid(True)
    
    # 熵分布
    plt.subplot(1, 3, 3)
    plt.hist(orica.entropy_history, bins=20, alpha=0.7)
    plt.title('熵分布')
    plt.xlabel('熵值')
    plt.ylabel('频次')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_entropy_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 分析熵计算问题
    analyze_entropy_issue()
    
    # 测试改进的熵计算
    test_improved_entropy() 