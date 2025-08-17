import numpy as np
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_regression


class ORICA1:
    def __init__(self, n_components, learning_rate=0.001, ortho_every=10, 
                 use_rls_whitening=True, forgetting_factor=0.98, 
                 nonlinearity='gaussian', adaptive_ff='cooling', 
                 sample_rate=500, block_size=8, eval_convergence=True):
        """
        ORICA with RLS whitening support - 完整版本
        
        Args:
            n_components: 独立成分数量
            learning_rate: 学习率
            ortho_every: 每隔多少次迭代正交化
            use_rls_whitening: 是否使用RLS白化
            forgetting_factor: RLS遗忘因子 (0 < λ < 1)
            nonlinearity: 非线性函数类型 ('gaussian', 'tanh')
            adaptive_ff: 遗忘因子策略 ('cooling', 'constant', 'adaptive')
            sample_rate: 采样率
            block_size: 块大小
            eval_convergence: 是否评估收敛性
        """
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.W = np.eye(n_components)  # 解混矩阵
        self.mean = None
        self.whitening_matrix = None
        self.whitened = False
        self.update_count = 0
        self.ortho_every = ortho_every
        
        # RLS白化参数
        self.use_rls_whitening = use_rls_whitening
        self.forgetting_factor = forgetting_factor
        self.nonlinearity = nonlinearity
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # 自适应遗忘因子参数
        self.adaptive_ff = adaptive_ff
        self.gamma = 0.6  # 冷却策略参数
        self.lambda_0 = 0.995  # 初始遗忘因子
        self.tau_const = 3  # 常数策略参数
        self.counter = 0  # 时间计数器
        
        # 自适应策略参数
        self.decay_rate_alpha = 0.02
        self.upper_bound_beta = 0.001
        self.trans_band_width = 1
        self.trans_band_center = 5
        self.min_norm_rn = None
        
        # 收敛性评估
        self.eval_convergence = eval_convergence
        self.leaky_avg_delta = 0.01
        self.leaky_avg_delta_var = 1e-3
        self.Rn = None
        self.Var = None
        self.norm_rn = None
        
        # RLS白化相关变量
        if self.use_rls_whitening:
            self.C = None  # 协方差矩阵的逆
            self.t = 0     # 时间步计数器

    def _center(self, X):
        """去均值"""
        if self.mean is None:
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

    def _get_kurtosis_sign(self, y):
        """计算峰度符号"""
        k = kurtosis(y, axis=0, fisher=False)
        return k > 0  # True for supergaussian, False for subgaussian

    def _g(self, y):
        """
        改进的非线性函数，根据峰度符号选择
        
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
        elif self.nonlinearity == 'adaptive':
            # 自适应非线性函数 - 根据峰度符号选择
            kurt_sign = self._get_kurtosis_sign(y)
            
            g_y = np.zeros_like(y)
            g_prime = np.zeros_like(y)
            
            # Supergaussian components
            super_idx = kurt_sign
            g_y[super_idx] = -2 * np.tanh(y[super_idx])
            g_prime[super_idx] = -2 * (1 - np.tanh(y[super_idx])**2)
            
            # Subgaussian components  
            sub_idx = ~kurt_sign
            g_y[sub_idx] = np.tanh(y[sub_idx]) - y[sub_idx]
            g_prime[sub_idx] = 1 - np.tanh(y[sub_idx])**2 - 1
            
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
        
        return g_y, g_prime

    def _compute_forgetting_factor(self, n_samples):
        """计算遗忘因子"""
        if self.adaptive_ff == 'cooling':
            # 冷却策略: lambda = lambda_0 / t^gamma
            t_range = np.arange(self.counter + 1, self.counter + n_samples + 1)
            lambda_k = self.lambda_0 / (t_range ** self.gamma)
            lambda_const = 1 - np.exp(-1 / (self.tau_const * self.sample_rate))
            lambda_k = np.maximum(lambda_k, lambda_const)
            
        elif self.adaptive_ff == 'constant':
            # 常数策略
            lambda_const = 1 - np.exp(-1 / (self.tau_const * self.sample_rate))
            lambda_k = np.full(n_samples, lambda_const)
            
        elif self.adaptive_ff == 'adaptive':
            # 自适应策略
            if self.min_norm_rn is None:
                self.min_norm_rn = self.norm_rn if self.norm_rn is not None else 1.0
            self.min_norm_rn = max(min(self.min_norm_rn, self.norm_rn), 1)
            ratio = self.norm_rn / self.min_norm_rn
            lambda_k = self._adaptive_forgetting_factor(n_samples, ratio)
            
        else:
            lambda_k = np.full(n_samples, self.forgetting_factor)
        
        return lambda_k

    def _adaptive_forgetting_factor(self, n_samples, ratio):
        """自适应遗忘因子计算"""
        gain = self.upper_bound_beta * 0.5 * (1 + np.tanh((ratio - self.trans_band_center) / self.trans_band_width))
        
        lambda_k = np.zeros(n_samples)
        for n in range(n_samples):
            lambda_k[n] = ((1 + gain) ** (n + 1)) * self.forgetting_factor - \
                          self.decay_rate_alpha * (((1 + gain) ** (2 * n + 1)) - ((1 + gain) ** n)) / gain * (self.forgetting_factor ** 2)
        
        return lambda_k

    def _update_convergence_metrics(self, y, f, x_whitened, n_samples):
        """更新收敛性指标"""
        n_channels = y.shape[0]
        
        # 模型拟合度
        model_fitness = np.eye(n_channels) + y @ f.T / n_samples
        
        # 方差计算
        variance = x_whitened * x_whitened
        
        if self.Rn is None:
            self.Rn = model_fitness
            self.Var = np.sum(variance, axis=1) / (n_samples - 1)
        else:
            # 泄漏平均更新
            self.Rn = (1 - self.leaky_avg_delta) * self.Rn + self.leaky_avg_delta * model_fitness
            
            # 方差更新
            decay_factors = (1 - self.leaky_avg_delta_var) ** np.arange(n_samples, 0, -1)
            self.Var = (1 - self.leaky_avg_delta_var) ** n_samples * self.Var + \
                       np.sum(self.leaky_avg_delta_var * variance * decay_factors.reshape(1, -1), axis=1)
        
        # 非平稳性指数
        self.norm_rn = np.linalg.norm(self.Rn, 'fro')

    def initialize(self, X_init):
        """初始化ORICA"""
        # 检查并调整n_components以匹配数据维度
        if X_init.shape[1] != self.n_components:
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
        return X_init

    def partial_fit(self, x_t):
        """
        单个样本在线更新
        
        Args:
            x_t: 单个时间点的数据 (n_channels,)
        """
        x_t = x_t.reshape(-1, 1)  # 从 (25,) 变为 (25, 1)
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

    def partial_fit_block(self, X_block, time_perm=True):
        """
        块更新 - 更接近MATLAB实现
        
        Args:
            X_block: 数据块 (n_channels, n_samples)
            time_perm: 是否进行时间排列
        """
        if not self.whitened:
            raise ValueError("Must call `initialize` first.")
        
        n_channels, n_samples = X_block.shape
        
        # 检查维度匹配
        if n_channels != self.n_components:
            print(f"⚠️ 块更新维度不匹配: 期望{self.n_components}通道，实际{n_channels}通道")
            self.n_components = n_channels
            self.W = np.eye(self.n_components)
            if self.use_rls_whitening:
                self.C = np.eye(self.n_components)
                self.whitening_matrix = np.eye(self.n_components)
                self.t = 0
            else:
                self.whitening_matrix = np.eye(self.n_components)
            print(f"✅ 重新初始化ORICA，新维度: {self.n_components}")
        
        # 去均值
        if self.mean is not None and self.mean.shape[0] == n_channels:
            X_block = X_block - self.mean.reshape(-1, 1)
        else:
            print(f"⚠️ 均值维度不匹配，重新计算")
            self.mean = np.zeros(n_channels)
        
        # 白化
        if self.use_rls_whitening:
            X_whitened = self.whitening_matrix @ X_block
        else:
            X_whitened = self.whitening_matrix @ X_block
        
        # 时间排列
        if time_perm:
            perm_idx = np.random.permutation(n_samples)
            X_whitened = X_whitened[:, perm_idx]
        
        # 计算源激活
        y = self.W @ X_whitened
        
        # 非线性函数
        f, _ = self._g(y)
        
        # 计算遗忘因子
        lambda_k = self._compute_forgetting_factor(n_samples)
        
        # 计算收敛性指标
        if self.eval_convergence:
            self._update_convergence_metrics(y, f, X_whitened, n_samples)
        
        # ORICA块更新规则
        lambda_prod = np.prod(1.0 / (1.0 - lambda_k))
        Q = 1.0 + lambda_k * (np.sum(f * y, axis=0) - 1.0)
        
        # 更新权重矩阵
        delta_W = np.zeros_like(self.W)
        for i in range(n_samples):
            delta_W += (y[:, i:i+1] * (lambda_k[i] / Q[i]) * f[:, i:i+1].T) @ self.W
        
        self.W = lambda_prod * (self.W - delta_W)
        
        # 正交化
        U, _, Vt = np.linalg.svd(self.W)
        self.W = U @ Vt
        
        self.counter += n_samples
        return y

    def fit_online(self, X_stream):
        """
        在线拟合 - 处理数据流
        
        Args:
            X_stream: 数据流 (n_samples, n_channels)
        """
        results = []
        
        # 分块处理
        for i in range(0, len(X_stream), self.block_size):
            block = X_stream[i:i+self.block_size].T  # 转置为 (n_channels, n_samples)
            y = self.partial_fit_block(block)
            results.append(y.T)  # 转置回 (n_samples, n_channels)
        
        return np.vstack(results)

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

    def get_convergence_metrics(self):
        """获取收敛性指标"""
        return {
            'norm_rn': self.norm_rn,
            'Rn': self.Rn,
            'Var': self.Var
        }

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
        if nonlinearity not in ['gaussian', 'tanh', 'adaptive']:
            raise ValueError("nonlinearity must be 'gaussian', 'tanh', or 'adaptive'")
        self.nonlinearity = nonlinearity

    def set_forgetting_factor(self, forgetting_factor):
        """设置RLS遗忘因子"""
        if not (0 < forgetting_factor < 1):
            raise ValueError("forgetting_factor must be between 0 and 1")
        self.forgetting_factor = forgetting_factor

    def set_adaptive_ff(self, adaptive_ff):
        """设置自适应遗忘因子策略"""
        if adaptive_ff not in ['cooling', 'constant', 'adaptive']:
            raise ValueError("adaptive_ff must be 'cooling', 'constant', or 'adaptive'")
        self.adaptive_ff = adaptive_ff


