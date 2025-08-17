#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版本的ORICA，包含自适应学习率和更好的参数设置
"""

import numpy as np
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_regression

class ORICAOptimized:
    def __init__(self, n_components, learning_rate=0.005, ortho_every=5, 
                 use_rls_whitening=False, forgetting_factor=0.98, 
                 nonlinearity='gaussian', adaptive_lr=True):
        """
        优化版本的ORICA
        
        Args:
            n_components: 独立成分数量
            learning_rate: 初始学习率
            ortho_every: 每隔多少次迭代正交化
            use_rls_whitening: 是否使用RLS白化
            forgetting_factor: RLS遗忘因子
            nonlinearity: 非线性函数类型
            adaptive_lr: 是否使用自适应学习率
        """
        self.n_components = n_components
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.ortho_every = ortho_every
        self.W = np.eye(n_components)
        self.mean = None
        self.whitening_matrix = None
        self.whitened = False
        self.update_count = 0
        
        # 自适应学习率参数
        self.adaptive_lr = adaptive_lr
        self.lr_decay_factor = 0.9995
        self.min_lr = 0.0001
        self.max_lr = 0.1
        
        # RLS白化参数
        self.use_rls_whitening = use_rls_whitening
        self.forgetting_factor = forgetting_factor
        self.nonlinearity = nonlinearity
        
        if self.use_rls_whitening:
            self.C = None
            self.t = 0
        
        # 收敛监控
        self.convergence_history = []
        self.w_norm_history = []
        
    def _update_learning_rate(self):
        """自适应学习率更新"""
        if not self.adaptive_lr:
            return
            
        # 基于更新次数衰减
        if self.update_count < 1000:
            # 初期保持较大学习率
            self.learning_rate = self.initial_learning_rate
        elif self.update_count < 5000:
            # 中期逐渐衰减
            decay = self.lr_decay_factor ** (self.update_count - 1000)
            self.learning_rate = max(self.initial_learning_rate * decay, self.min_lr)
        else:
            # 后期使用最小学习率
            self.learning_rate = self.min_lr
            
        # 确保在合理范围内
        self.learning_rate = np.clip(self.learning_rate, self.min_lr, self.max_lr)
    
    def _center(self, X):
        """去均值"""
        self.mean = np.mean(X, axis=0)
        return X - self.mean

    def _whiten(self, X):
        """传统批量白化"""
        cov = np.cov(X, rowvar=False)
        d, E = np.linalg.eigh(cov)
        D_inv = np.diag(1.0 / np.sqrt(d + 1e-2))
        self.whitening_matrix = E @ D_inv @ E.T
        return X @ self.whitening_matrix.T

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
        if X_init.shape[1] != self.n_components:
            print(f"⚠️ 调整n_components: {self.n_components} -> {X_init.shape[1]}")
            self.n_components = X_init.shape[1]
            self.W = np.eye(self.n_components)
        
        X_init = self._center(X_init)
        X_init = self._whiten(X_init)
        self.whitened = True
        
        print(f"✅ ORICA初始化完成: n_components={self.n_components}")
        return X_init

    def partial_fit(self, x_t):
        """单个样本在线更新"""
        x_t = x_t.reshape(-1, 1)
        
        if not self.whitened:
            raise ValueError("Must call `initialize` first.")
        
        # 维度检查
        if x_t.shape[0] != self.n_components:
            print(f"⚠️ 维度不匹配，重新初始化: {self.n_components} -> {x_t.shape[0]}")
            self.n_components = x_t.shape[0]
            self.W = np.eye(self.n_components)
            self.whitening_matrix = np.eye(self.n_components)
        
        # 去均值
        if self.mean is not None and self.mean.shape[0] == x_t.shape[0]:
            x_t = x_t - self.mean.reshape(-1, 1)
        else:
            self.mean = np.zeros(x_t.shape[0])
        
        # 白化
        x_t_whitened = self.whitening_matrix @ x_t
        
        # ICA更新
        y_t = self.W @ x_t_whitened
        g_y, _ = self._g(y_t)
        
        # 更新学习率
        self._update_learning_rate()
        
        # ORICA更新规则
        I = np.eye(self.n_components)
        delta_W = self.learning_rate * ((I - g_y @ y_t.T) @ self.W)
        self.W += delta_W

        # 正交化
        self.update_count += 1
        if self.update_count % self.ortho_every == 0:
            U, _, Vt = np.linalg.svd(self.W)
            self.W = U @ Vt
            
            # 记录收敛历史
            w_norm = np.linalg.norm(self.W)
            self.w_norm_history.append(w_norm)
            
            if len(self.w_norm_history) > 10:
                # 计算W矩阵的稳定性
                recent_norms = self.w_norm_history[-10:]
                stability = np.std(recent_norms)
                self.convergence_history.append(stability)

        return y_t.ravel()
    
    def get_learning_rate(self):
        """获取当前学习率"""
        return self.learning_rate
    
    def set_learning_rate(self, lr):
        """设置学习率"""
        self.learning_rate = np.clip(lr, self.min_lr, self.max_lr)
    
    def get_convergence_status(self):
        """获取收敛状态"""
        if len(self.convergence_history) < 5:
            return "初始化中"
        
        recent_stability = np.mean(self.convergence_history[-5:])
        if recent_stability < 0.01:
            return "已收敛"
        elif recent_stability < 0.1:
            return "接近收敛"
        else:
            return "未收敛"
    
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

def test_optimized_orica():
    """测试优化版本的ORICA"""
    print("=== 测试优化版本ORICA ===")
    
    # 模拟数据
    n_channels = 25
    n_samples = 2000
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_channels)
    X[:, 0] = np.sign(X[:, 0]) * np.abs(X[:, 0])**1.5
    X[:, 1] = np.tanh(X[:, 1])
    
    # 测试不同配置
    configs = [
        ("固定小学习率", {"learning_rate": 0.0001, "adaptive_lr": False}),
        ("固定标准学习率", {"learning_rate": 0.005, "adaptive_lr": False}),
        ("自适应学习率", {"learning_rate": 0.01, "adaptive_lr": True}),
    ]
    
    for name, config in configs:
        print(f"\n--- 测试 {name} ---")
        
        orica = ORICAOptimized(
            n_components=n_channels,
            ortho_every=5,
            **config
        )
        
        # 初始化
        orica.initialize(X)
        
        # 在线处理
        sources_list = []
        lr_history = []
        
        for i in range(n_samples):
            x_t = X[i, :]
            result = orica.partial_fit(x_t)
            sources_list.append(result)
            
            if i % 100 == 0:
                lr_history.append(orica.get_learning_rate())
        
        # 评估结果
        sources = np.array(sources_list).T
        kurtosis = orica.evaluate_separation(sources)
        kurtosis_mean = np.mean(np.abs(kurtosis))
        
        convergence_status = orica.get_convergence_status()
        
        print(f"✅ {name} 完成")
        print(f"  最终峰度均值: {kurtosis_mean:.3f}")
        print(f"  收敛状态: {convergence_status}")
        print(f"  最终学习率: {orica.get_learning_rate():.6f}")
        print(f"  学习率变化: {lr_history[0]:.6f} -> {lr_history[-1]:.6f}")

if __name__ == "__main__":
    test_optimized_orica() 