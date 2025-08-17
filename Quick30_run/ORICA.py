import numpy as np
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_regression
from ORICA_calibration import ORICACalibration

class ORICA:
    def __init__(self, n_components, learning_rate=0.001, ortho_every=10, 
                 use_rls_whitening=True, forgetting_factor=0.98, 
                 nonlinearity='gaussian'):
        """
        ORICA with RLS whitening support
        
        Args:
            n_components: 独立成分数量
            learning_rate: 学习率
            ortho_every: 每隔多少次迭代正交化
            use_rls_whitening: 是否使用RLS白化
            forgetting_factor: RLS遗忘因子 (0 < λ < 1)
            nonlinearity: 非线性函数类型 ('gaussian', 'tanh')
        """
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.W = np.eye(n_components)  # 解混矩阵
        self.mean = None
        self.whitening_matrix = None
        self.whitened = False
        self.update_count = 0
        self.ortho_every = ortho_every  # 每隔多少次迭代正交化
        
        # RLS白化参数
        self.use_rls_whitening = use_rls_whitening
        self.forgetting_factor = forgetting_factor
        self.nonlinearity = nonlinearity
        
        # RLS白化相关变量
        if self.use_rls_whitening:
            self.C = None  # 协方差矩阵的逆
            self.t = 0     # 时间步计数器

    def _center(self, X):
        """去均值"""
        self.mean = np.mean(X, axis=0)
        return X - self.mean

    def _whiten(self, X):
        """传统批量白化 - 使用特征值分解"""
        cov = np.cov(X, rowvar=False)
        d, E = np.linalg.eigh(cov)
        D_inv = np.diag(1.0 / np.sqrt(d + 1e-2))  # 防止除0
        self.whitening_matrix = E @ D_inv @ E.T
        return X @ self.whitening_matrix.T

    def _rls_whiten_initialize(self, X):
        """初始化RLS白化"""
        # X的形状是 (samples, channels)，需要转置为 (channels, samples)
        # if X.shape[0] < X.shape[1]:  # 如果第一个维度小于第二个维度，说明是 (samples, channels)
        #     n_channels = X.shape[1]
        # else:
        n_channels = X.shape[1]
        
        # 初始化协方差矩阵的逆为单位矩阵
        self.C = np.eye(n_channels)
        self.whitening_matrix = np.eye(n_channels)
        self.t = 0
        print(f"🔧 RLS白化初始化: 通道数={n_channels}, C矩阵形状={self.C.shape}")

    def _rls_whiten_update(self, x_t):
        """
        RLS白化单步更新
        
        Args:
            x_t: 单个时间点的数据 (n_channels, 1)
        """
        if self.C is None:
            raise ValueError("RLS whitening not initialized. Call initialize() first.")
        
        # 检查维度匹配
        expected_channels = self.C.shape[0]
        actual_channels = x_t.shape[0]
        
        if expected_channels != actual_channels:
            print(f"⚠️ RLS白化维度不匹配: 期望{expected_channels}通道，实际{actual_channels}通道")
            # 重新初始化以匹配新的维度
            self.C = np.eye(actual_channels)
            self.whitening_matrix = np.eye(actual_channels)
            self.t = 0
            print(f"✅ 重新初始化RLS白化，新维度: {actual_channels}")
        
        # RLS更新规则
        lambda_t = self.forgetting_factor
        self.t += 1
        
        # 计算增益向量
        k = self.C @ x_t
        denominator = lambda_t + x_t.T @ k
        k = k / denominator
        
        # 更新协方差矩阵的逆
        self.C = (self.C - k @ x_t.T @ self.C) / lambda_t
        
        # 更新白化矩阵
        # 白化矩阵是协方差矩阵逆的平方根
        eigenvals, eigenvecs = np.linalg.eigh(self.C)
        eigenvals = np.maximum(eigenvals, 1e-6)  # 防止负值
        self.whitening_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
        
        # 返回白化后的数据
        return self.whitening_matrix @ x_t

    def _g(self, y):
        """
        非线性函数
        
        Args:
            y: 输入信号
            
        Returns:
            g_y: 非线性函数值
            g_prime: 导数
        """
        if self.nonlinearity == 'gaussian':
            # 高斯非线性函数
            g_y = y * np.exp(-0.5 * y**2)
            g_prime = (1 - y**2) * np.exp(-0.5 * y**2)
        elif self.nonlinearity == 'tanh':
            # 双曲正切非线性函数
            g_y = np.tanh(y)
            g_prime = 1 - np.tanh(y)**2
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
        
        return g_y, g_prime

    def initialize(self, X_init):
        """初始化ORICA"""
        # 检查并调整n_components以匹配数据维度
        if X_init.shape[1] != self.n_components:  # X_init是 (samples, channels) 格式
            print(f"⚠️ 初始化维度不匹配: 期望{self.n_components}通道，实际{X_init.shape[1]}通道")
            self.n_components = X_init.shape[1]
            # 重新创建W矩阵以匹配新的维度
            self.W = np.eye(self.n_components)
            print(f"✅ 调整n_components为{self.n_components}")
        
        # 去均值
        X_init = self._center(X_init)
        
        # 白化
        if self.use_rls_whitening:
            # 使用RLS白化初始化
            self._rls_whiten_initialize(X_init)
            # 对初始数据进行批量白化
            X_init = self._whiten(X_init)
        else:
            # 使用传统批量白化
            X_init = self._whiten(X_init)
        
        self.whitened = True
        print(f"✅ ORICA初始化完成: n_components={self.n_components}, 数据形状={X_init.shape}")
        #test (2500, 25)
        return X_init

    def partial_fit(self, x_t):
        """
        单个样本在线更新
        
        Args:
            x_t: 单个时间点的数据 (n_channels,)
        """
        x_t = x_t.reshape(-1, 1)# 从 (25,) 变为 (25, 1)
        if not self.whitened:
            raise ValueError("Must call `initialize` with initial batch before `partial_fit`.")
        
        # 检查输入维度是否与当前模型匹配
        if x_t.shape[0] != self.n_components:
            print(f"⚠️ partial_fit维度不匹配: 期望{self.n_components}通道，实际{x_t.shape[0]}通道")
            # 重新初始化以匹配新的维度
            self.n_components = x_t.shape[0]
            self.W = np.eye(self.n_components)
            # 重新初始化白化
            if self.use_rls_whitening:
                self.C = np.eye(self.n_components)
                self.whitening_matrix = np.eye(self.n_components)
                self.t = 0
            else:
                self.whitening_matrix = np.eye(self.n_components)
            print(f"✅ 重新初始化ORICA，新维度: {self.n_components}")
        
        # 去均值
        if self.mean is not None and self.mean.shape[0] == x_t.shape[0]:
            x_t = x_t - self.mean.reshape(-1, 1)
        else:
            # 如果mean维度不匹配，重新计算
            print(f"⚠️ 均值维度不匹配，重新计算")
            self.mean = np.zeros(x_t.shape[0])
        
        # 白化
        if self.use_rls_whitening:
            # RLS白化更新
            x_t_whitened = self._rls_whiten_update(x_t)
        else:
            # 传统白化
            x_t_whitened = self.whitening_matrix @ x_t
        
        # ICA更新
        y_t = self.W @ x_t_whitened
        g_y, _ = self._g(y_t)
        
        # ORICA更新规则
        I = np.eye(self.n_components)
        delta_W = self.learning_rate * ((I - g_y @ y_t.T) @ self.W)
        self.W += delta_W

        # 正交化
        self.update_count += 1
        if self.update_count % self.ortho_every == 0:
            U, _, Vt = np.linalg.svd(self.W)
            self.W = U @ Vt

        return y_t.ravel()

    def transform(self, X):
        """变换数据"""
        if not self.whitened:
            raise ValueError("Model must be initialized first with `initialize()`.")
        
        # 去均值
        X = X - self.mean
        
        # 白化
        if self.use_rls_whitening:
            # 使用当前的白化矩阵
            X_whitened = X @ self.whitening_matrix.T
        else:
            # 传统白化
            X_whitened = X @ self.whitening_matrix.T
        
        # ICA变换
        Y = (self.W @ X_whitened.T).T
        return Y

    def inverse_transform(self, Y):
        """逆变换"""
        Xw = np.linalg.pinv(self.W) @ Y.T
        X = Xw.T @ np.linalg.pinv(self.whitening_matrix).T + self.mean
        return X

    def get_W(self):
        """获取解混矩阵"""
        return self.W

    def get_whitening_matrix(self):
        """获取白化矩阵"""
        return self.whitening_matrix

    def evaluate_separation(self, Y):
        """评估分离效果 - 使用峰度"""
        k = kurtosis(Y, axis=0, fisher=False)
        return k

    def rank_components_by_kurtosis(self, Y):
        """按峰度排序成分"""
        k = self.evaluate_separation(Y)
        indices = np.argsort(-np.abs(k))
        return indices, k

    def calc_mutual_info_matrix(self, sources):
        """计算互信息矩阵"""
        n = sources.shape[0]
        MI = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    MI[i, j] = mutual_info_regression(
                        sources[i, :].reshape(-1, 1), sources[j, :]
                    )[0]
        return MI

    def set_nonlinearity(self, nonlinearity):
        """设置非线性函数类型"""
        if nonlinearity not in ['gaussian', 'tanh']:
            raise ValueError("nonlinearity must be 'gaussian' or 'tanh'")
        self.nonlinearity = nonlinearity

    def set_forgetting_factor(self, forgetting_factor):
        """设置RLS遗忘因子"""
        if not (0 < forgetting_factor < 1):
            raise ValueError("forgetting_factor must be between 0 and 1")
        self.forgetting_factor = forgetting_factor

    

