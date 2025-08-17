#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版ORICA - 解决IC集中在channel的问题
"""

import numpy as np
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_regression

class ORICAW:
    def __init__(self, n_components, learning_rate=0.01, ortho_every=10, 
                 use_rls_whitening=True, forgetting_factor=0.98, 
                 nonlinearity='gaussian', enhanced_init=True):
        """
        增强版ORICA
        
        Args:
            n_components: 独立成分数量
            learning_rate: 学习率
            ortho_every: 正交化频率
            use_rls_whitening: 是否使用RLS白化
            forgetting_factor: RLS遗忘因子
            nonlinearity: 非线性函数类型
            enhanced_init: 是否使用增强初始化
        """
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.ortho_every = ortho_every
        self.W = np.eye(n_components)
        self.mean = None
        self.whitening_matrix = None
        self.whitened = False
        self.update_count = 0
        
        # RLS白化参数
        self.use_rls_whitening = use_rls_whitening
        self.forgetting_factor = forgetting_factor
        self.nonlinearity = nonlinearity
        
        # 增强功能参数
        self.enhanced_init = enhanced_init
        self.spatial_diversity_weight = 0.1  # 空间多样性权重
        self.entropy_weight = 0.05  # 熵权重
        
        if self.use_rls_whitening:
            self.C = None
            self.t = 0
        
        # 性能监控
        self.spatial_concentration_history = []
        self.entropy_history = []
        
    def _enhanced_initialization(self, X_init):
        """增强初始化策略"""
        print("🔧 使用增强初始化策略...")
        
        n_samples, n_channels = X_init.shape
        
        # 1. 基于PCA的初始化
        # 计算协方差矩阵
        cov_matrix = np.cov(X_init.T)
        
        # 特征值分解
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # 选择前n_components个主成分
        sorted_indices = np.argsort(eigenvals)[::-1]
        selected_indices = sorted_indices[:self.n_components]
        
        # 初始化W矩阵为主成分方向
        W_pca = eigenvecs[:, selected_indices].T
        
        # 2. 添加随机扰动以增加多样性
        noise = np.random.randn(*W_pca.shape) * 0.1
        W_enhanced = W_pca + noise
        
        # 3. 正交化
        U, _, Vt = np.linalg.svd(W_enhanced)
        self.W = U @ Vt
        
        print(f"✅ 增强初始化完成，使用PCA + 随机扰动策略")
        
    def _compute_spatial_diversity(self):
        """计算空间多样性"""
        # 计算W矩阵的条件数
        condition_number = np.linalg.cond(self.W)
        
        # 计算W矩阵的奇异值分布
        singular_values = np.linalg.svd(self.W, compute_uv=False)
        sv_ratio = np.min(singular_values) / np.max(singular_values)
        
        # 计算空间集中度
        spatial_concentration = np.mean(np.max(np.abs(self.W), axis=1) / np.mean(np.abs(self.W), axis=1))
        
        return {
            'condition_number': condition_number,
            'sv_ratio': sv_ratio,
            'spatial_concentration': spatial_concentration
        }
    
    def _compute_entropy(self, y_t):
        """计算熵"""
        # 使用直方图估计熵
        hist, _ = np.histogram(y_t, bins=20, density=True)
        hist = hist[hist > 0]  # 移除零概率
        entropy = -np.sum(hist * np.log(hist))
        return entropy
    
    def _spatial_diversity_loss(self):
        """空间多样性损失"""
        # 鼓励W矩阵的列向量更加分散
        W_normalized = self.W / np.linalg.norm(self.W, axis=1, keepdims=True)
        
        # 计算列向量间的相似度
        similarity_matrix = W_normalized @ W_normalized.T
        
        # 对角线元素设为0（排除自身相似度）
        np.fill_diagonal(similarity_matrix, 0)
        
        # 多样性损失：最小化相似度
        diversity_loss = np.mean(similarity_matrix**2)
        
        return diversity_loss
    
    def _center(self, X):
        """去均值"""
        self.mean = np.mean(X, axis=0)
        return X - self.mean

    def _whiten(self, X):
        """改进的白化策略"""
        cov = np.cov(X, rowvar=False)
        
        # 添加正则化以提高数值稳定性
        reg_factor = 1e-6
        cov_reg = cov + reg_factor * np.eye(cov.shape[0])
        
        d, E = np.linalg.eigh(cov_reg)
        
        # 确保特征值为正
        d = np.maximum(d, 1e-8)
        
        D_inv = np.diag(1.0 / np.sqrt(d))
        self.whitening_matrix = E @ D_inv @ E.T
        
        return X @ self.whitening_matrix.T

    def _rls_whiten_initialize(self, X):
        """初始化RLS白化"""
        n_channels = X.shape[1]
        self.C = np.eye(n_channels)
        self.whitening_matrix = np.eye(n_channels)
        self.t = 0

    def _rls_whiten_update(self, x_t):
        """RLS白化单步更新"""
        if self.C is None:
            raise ValueError("RLS whitening not initialized")
        
        lambda_t = self.forgetting_factor
        self.t += 1
        
        # 计算增益向量
        k = self.C @ x_t
        denominator = lambda_t + x_t.T @ k
        
        # 数值稳定性检查
        if denominator < 1e-10:
            return self.whitening_matrix @ x_t
        
        k = k / denominator
        
        # 更新协方差矩阵的逆
        self.C = (self.C - k @ x_t.T @ self.C) / lambda_t
        
        # 确保对称性
        self.C = (self.C + self.C.T) / 2
        
        # 更新白化矩阵
        try:
            eigenvals, eigenvecs = np.linalg.eigh(self.C)
            eigenvals = np.maximum(eigenvals, 1e-6)
            self.whitening_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
        except np.linalg.LinAlgError:
            pass
        
        return self.whitening_matrix @ x_t

    def _g(self, y):
        """非线性函数及其导数"""
        if self.nonlinearity == 'gaussian':
            g_y = y * np.exp(-y**2/2)
            g_prime = (1 - y**2) * np.exp(-y**2/2)
        elif self.nonlinearity == 'tanh':
            g_y = np.tanh(y)
            g_prime = 1 - np.tanh(y)**2
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
        return g_y, g_prime

    def initialize(self, X_init):
        """初始化"""
        print(f"🔧 初始化增强版ORICA: 成分数={self.n_components}")
        
        X_init = self._center(X_init)
        
        # 白化
        if self.use_rls_whitening:
            self._rls_whiten_initialize(X_init)
            X_init = self._whiten(X_init)
        else:
            X_init = self._whiten(X_init)
        
        # 增强初始化
        if self.enhanced_init:
            self._enhanced_initialization(X_init)
        
        self.whitened = True
        print(f"✅ 初始化完成")

    def partial_fit(self, x_t):
        """单个样本在线更新"""
        if not self.whitened:
            raise ValueError("Must call `initialize` with initial batch before `partial_fit`.")
        
        # 去均值
        if self.mean is not None:
            x_t = x_t - self.mean
        
        # 白化
        if self.use_rls_whitening:
            x_t_whitened = self._rls_whiten_update(x_t.reshape(-1, 1)).ravel()
        else:
            x_t_whitened = self.whitening_matrix @ x_t
        
        # ICA更新
        y_t = self.W @ x_t_whitened
        g_y, _ = self._g(y_t)
        
        # 标准ORICA更新规则
        I = np.eye(self.n_components)
        delta_W_standard = self.learning_rate * ((I - g_y @ y_t.T) @ self.W)
        
        # 增强更新：添加空间多样性约束
        diversity_loss = self._spatial_diversity_loss()
        diversity_gradient = self._compute_diversity_gradient()
        
        # 计算熵
        entropy = self._compute_entropy(y_t)
        
        # 组合更新
        delta_W_enhanced = delta_W_standard + self.spatial_diversity_weight * diversity_gradient
        
        self.W += delta_W_enhanced
        
        # 正交化
        self.update_count += 1
        if self.update_count % self.ortho_every == 0:
            U, _, Vt = np.linalg.svd(self.W)
            self.W = U @ Vt
            
            # 记录性能指标
            spatial_metrics = self._compute_spatial_diversity()
            self.spatial_concentration_history.append(spatial_metrics['spatial_concentration'])
            self.entropy_history.append(entropy)
            
            # 打印监控信息
            # if self.update_count % 100 == 0:
            #     print(f"  步骤 {self.update_count}: 空间集中度={spatial_metrics['spatial_concentration']:.3f}, "
            #           f"熵={entropy:.3f}, 多样性损失={diversity_loss:.4f}")
        
        return y_t
    
    def _compute_diversity_gradient(self):
        """计算多样性约束的梯度"""
        W_normalized = self.W / np.linalg.norm(self.W, axis=1, keepdims=True)
        
        # 计算相似度矩阵
        similarity_matrix = W_normalized @ W_normalized.T
        np.fill_diagonal(similarity_matrix, 0)
        
        # 计算梯度
        gradient = 2 * similarity_matrix @ W_normalized
        
        return gradient
    
    def get_spatial_metrics(self):
        """获取空间分布指标"""
        if len(self.spatial_concentration_history) == 0:
            return None
        
        return {
            'spatial_concentration': self.spatial_concentration_history[-1],
            'entropy': self.entropy_history[-1] if len(self.entropy_history) > 0 else 0,
            'concentration_trend': np.mean(self.spatial_concentration_history[-10:]) if len(self.spatial_concentration_history) >= 10 else 0
        }
    
    def evaluate_separation(self, Y):
        """评估分离效果"""
        return kurtosis(Y, axis=0, fisher=False)
    
    def transform(self, X):
        """变换数据"""
        if not self.whitened:
            raise ValueError("Model must be initialized first.")
        
        X = X - self.mean
        X_whitened = X @ self.whitening_matrix.T
        Y = (self.W @ X_whitened.T).T
        return Y
    
    def inverse_transform(self, Y):
        """逆变换"""
        if not self.whitened:
            raise ValueError("Model must be initialized first.")
        
        X_whitened = np.linalg.pinv(self.W) @ Y.T
        X = X_whitened.T @ np.linalg.pinv(self.whitening_matrix)
        X = X + self.mean
        return X
    
    def get_W(self):
        """获取解混矩阵"""
        return self.W.copy()
    
    def get_whitening_matrix(self):
        """获取白化矩阵"""
        return self.whitening_matrix.copy()

# def test_enhanced_orica():
#     """测试增强版ORICA"""
#     print("=== 测试增强版ORICA ===")
    
#     # 生成测试数据
#     n_channels = 25
#     n_samples = 10000
    
#     # 生成混合信号
#     sources = np.random.randn(n_channels, n_samples)
    
#     # 添加非高斯成分
#     sources[0, :] = np.sign(sources[0, :]) * np.abs(sources[0, :])**1.5
#     sources[1, :] = np.tanh(sources[1, :])
#     sources[2, :] = sources[2, :] * np.exp(-sources[2, :]**2/2)
    
#     # 生成混合矩阵
#     A = np.random.randn(n_channels, n_channels)
#     A = A / np.linalg.norm(A, axis=0)
    
#     # 混合信号
#     X = A @ sources
    
#     # 创建增强版ORICA实例
#     orica = ORICAEnhanced(
#         n_components=n_channels,
#         learning_rate=0.01,
#         ortho_every=10,
#         use_rls_whitening=False,
#         enhanced_init=True
#     )
    
#     # 初始化
#     orica.initialize(X[:1000].T)
    
#     # 在线学习
#     performance_history = []
#     spatial_metrics_history = []
    
#     print("开始在线学习...")
#     for i in range(1000, len(X), 100):
#         batch = X[:, i:i+100].T
        
#         # 处理批次
#         sources_batch = []
#         for x_t in batch:
#             result = orica.partial_fit(x_t)
#             sources_batch.append(result)
        
#         # 评估性能
#         sources_array = np.array(sources_batch)
#         kurt_vals = kurtosis(sources_array, axis=0, fisher=True)
#         kurt_mean = np.mean(np.abs(kurt_vals))
#         performance_history.append(kurt_mean)
        
#         # 获取空间指标
#         spatial_metrics = orica.get_spatial_metrics()
#         if spatial_metrics:
#             spatial_metrics_history.append(spatial_metrics)
        
#         if len(performance_history) % 20 == 0:
#             print(f"  步骤 {len(performance_history)*100}: Kurtosis={kurt_mean:.3f}")
#             if spatial_metrics:
#                 print(f"    空间集中度: {spatial_metrics['spatial_concentration']:.3f}")
    
#     print(f"\n✅ 学习完成")
    
#     # 分析最终结果
#     final_sources = orica.transform(X.T)
#     final_mixing_matrix = np.linalg.pinv(orica.get_W())
    
#     # 使用分析工具
#     from analyze_ica_separation import analyze_ica_separation_quality
#     analyze_ica_separation_quality(final_sources.T, final_mixing_matrix, 
#                                  [f'Ch{i}' for i in range(n_channels)])
    
#     # 绘制性能变化
#     import matplotlib.pyplot as plt
    
#     plt.figure(figsize=(15, 5))
    
#     # 性能变化
#     plt.subplot(1, 3, 1)
#     plt.plot(performance_history, label='Kurtosis', linewidth=2)
#     plt.title('增强ORICA性能变化')
#     plt.xlabel('更新步骤 (x100)')
#     plt.ylabel('Kurtosis Mean')
#     plt.legend()
#     plt.grid(True)
    
#     # 空间集中度变化
#     if spatial_metrics_history:
#         plt.subplot(1, 3, 2)
#         concentrations = [m['spatial_concentration'] for m in spatial_metrics_history]
#         plt.plot(concentrations, label='空间集中度', linewidth=2)
#         plt.title('空间集中度变化')
#         plt.xlabel('更新步骤 (x100)')
#         plt.ylabel('空间集中度')
#         plt.legend()
#         plt.grid(True)
        
#         # 熵变化
#         plt.subplot(1, 3, 3)
#         entropies = [m['entropy'] for m in spatial_metrics_history]
#         plt.plot(entropies, label='熵', linewidth=2)
#         plt.title('熵变化')
#         plt.xlabel('更新步骤 (x100)')
#         plt.ylabel('熵')
#         plt.legend()
#         plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig('enhanced_orica_performance.png', dpi=300, bbox_inches='tight')
#     plt.show()

