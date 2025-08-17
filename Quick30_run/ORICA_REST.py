import numpy as np
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_regression
from ORICA_calibration import ORICACalibration

class ORICAZ:
    def __init__(self, n_components, learning_rate=0.001, ortho_every=10, 
                 use_rls_whitening=False, forgetting_factor=0.98, 
                 nonlinearity='gaussian', block_size_ica=8, block_size_white=8,
                 ff_profile='cooling', tau_const=np.inf, gamma=0.6, lambda_0=0.995,
                 num_subgaussian=0, eval_convergence=True, verbose=False):
        """
        ORICA with RLS whitening support - 基于MATLAB orica.m实现
        
        Args:
            n_components: 独立成分数量
            learning_rate: 学习率
            ortho_every: 每隔多少次迭代正交化
            use_rls_whitening: 是否使用RLS白化
            forgetting_factor: RLS遗忘因子 (0 < λ < 1)
            nonlinearity: 非线性函数类型 ('gaussian', 'tanh')
            block_size_ica: ICA块大小
            block_size_white: 白化块大小
            ff_profile: 遗忘因子策略 ('cooling', 'constant', 'adaptive')
            tau_const: 局部平稳性参数
            gamma: 冷却策略参数
            lambda_0: 初始遗忘因子
            num_subgaussian: 次高斯源数量
            eval_convergence: 是否评估收敛性
            verbose: 是否输出详细信息
        """
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.W = np.eye(n_components)  # 解混矩阵 (icaweights)
        self.mean = None
        self.whitening_matrix = None  # icasphere
        self.whitened = False
        self.update_count = 0
        self.ortho_every = ortho_every
        
        # 块更新参数
        self.block_size_ica = block_size_ica
        self.block_size_white = block_size_white
        
        # 遗忘因子参数
        self.ff_profile = ff_profile
        self.tau_const = tau_const
        self.gamma = gamma
        self.lambda_0 = lambda_0
        self.lambda_const = 1 - np.exp(-1/tau_const) if tau_const != np.inf else 0.98
        
        # 次高斯源参数
        self.num_subgaussian = num_subgaussian
        self.kurtosis_sign = np.ones(n_components, dtype=bool)  # True为超高斯
        if num_subgaussian > 0:
            self.kurtosis_sign[:num_subgaussian] = False
        
        # 收敛性评估
        self.eval_convergence = eval_convergence
        self.leaky_avg_delta = 0.01
        self.leaky_avg_delta_var = 1e-3
        self.Rn = None
        self.non_stat_idx = None
        self.min_non_stat_idx = None
        
        # 状态变量
        self.lambda_k = np.zeros(block_size_ica)
        self.counter = 0
        
        # RLS白化参数
        self.use_rls_whitening = use_rls_whitening
        self.forgetting_factor = forgetting_factor
        self.nonlinearity = nonlinearity
        
        # RLS白化相关变量
        if self.use_rls_whitening:
            self.C = None  # 协方差矩阵的逆
            self.t = 0     # 时间步计数器
            
        self.verbose = verbose

    def _center(self, X):
        """去均值"""
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
        return X - self.mean

    def _whiten(self, X):
        """传统批量白化 - 使用特征值分解"""
        # 检查数据长度是否足够
        if X.shape[0] < 2:
            print(f"⚠️ 白化数据长度不足: {X.shape[0]}，跳过白化")
            return X
        
        try:
            cov = np.cov(X, rowvar=False)
            d, E = np.linalg.eigh(cov)
            D_inv = np.diag(1.0 / np.sqrt(d + 1e-2))  # 防止除0
            self.whitening_matrix = E @ D_inv @ E.T
            return X @ self.whitening_matrix.T
        except Exception as e:
            print(f"⚠️ 白化失败: {e}，返回原始数据")
            return X

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

    def _gen_cooling_ff(self, t):
        """生成冷却遗忘因子 - 对应MATLAB的genCoolingFF"""
        return self.lambda_0 / (t ** self.gamma)
    
    def _gen_adaptive_ff(self, data_range, ratio_of_norm_rn):
        """生成自适应遗忘因子 - 对应MATLAB的genAdaptiveFF"""
        # 自适应策略参数
        decay_rate_alpha = 0.02
        upper_bound_beta = 1e-3
        trans_band_width_gamma = 1
        trans_band_center = 5
        
        # 检查lambda_k是否为空
        if len(self.lambda_k) == 0:
            print("⚠️ lambda_k为空，使用默认值")
            return np.full(len(data_range), self.lambda_const)
        
        gain_for_errors = upper_bound_beta * 0.5 * (1 + np.tanh((ratio_of_norm_rn - trans_band_center) / trans_band_width_gamma))
        
        def f(n):
            return ((1 + gain_for_errors) ** n) * self.lambda_k[-1] - \
                   decay_rate_alpha * (((1 + gain_for_errors) ** (2*n-1)) - ((1 + gain_for_errors) ** (n-1))) / gain_for_errors * (self.lambda_k[-1] ** 2)
        
        return np.array([f(n) for n in range(1, len(data_range) + 1)])

    def initialize(self, X_init):
        """初始化ORICA"""
        # 检查数据长度是否足够
        if X_init.shape[0] < 2:
            print(f"⚠️ 初始化数据长度不足: {X_init.shape[0]}，跳过初始化")
            return X_init
        
        # 检查并调整n_components以匹配数据维度
        if X_init.shape[1] != self.n_components:  # X_init是 (samples, channels) 格式
            print(f"⚠️ 初始化维度不匹配: 期望{self.n_components}通道，实际{X_init.shape[1]}通道")
            self.n_components = X_init.shape[1]
            # 重新创建W矩阵以匹配新的维度
            self.W = np.eye(self.n_components)
            print(f"✅ 调整n_components为{self.n_components}")
        
        try:
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
        except Exception as e:
            print(f"⚠️ ORICA初始化失败: {e}")
            self.whitened = False
        
        return X_init

    def partial_fit(self, x_t):
        """
        单个样本在线更新
        
        Args:
            x_t: 单个时间点的数据 (n_channels,)
        """
        try:
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
        except Exception as e:
            print(f"⚠️ partial_fit失败: {e}")
            return x_t.ravel() if hasattr(x_t, 'ravel') else x_t



    def fit_online_stream(self, data_stream, block_size=None):
        """
        在线流处理 - 使用逐样本更新而非批量处理
        
        Args:
            data_stream: 数据流 (samples, channels)
            block_size: 块大小，如果为None则使用默认值
        """
        try:
            if block_size is None:
                block_size = self.block_size_ica
            
        # 检查数据长度
            if data_stream.shape[0] < 1 or data_stream.shape[1] < 1:
                print(f"⚠️ 数据流长度不足: {data_stream.shape}，返回原始数据")
                return data_stream
            

            # 检查是否需要初始化
            if not self.whitened or self.whitening_matrix is None:
                print("⚠️ ORICA未初始化，尝试初始化...")
                # 使用前几个样本进行初始化
                init_samples = min(block_size*2, data_stream.shape[0])
                if init_samples >= 2:
                    init_data = data_stream[:init_samples, :]
                    self.initialize(init_data)
                else:
                    print("⚠️ 初始化数据不足，返回原始数据")
                    return data_stream
            
            # 使用逐样本的方式处理数据流，就像partial_fit一样
            sources = []

            for i in range(data_stream.shape[0]):
                x_t = data_stream[i, :]  # 获取单个样本 (n_channels,)
                y_t = self.partial_fit(x_t)  # 进行在线学习并返回源信号
                sources.append(y_t)
            
            # 转换为numpy数组并转置以匹配期望的输出格式
            sources = np.array(sources)  # shape: (samples, components)
            # 转置以匹配partial_fit方式的输出格式: (components, samples)
            sources = sources.T  # shape: (components, samples)
            return sources
                
        except Exception as e:
            print(f"⚠️ fit_online_stream失败: {e}")
            return data_stream




    def transform(self, X):
        """变换数据"""
        try:
            if not self.whitened:
                raise ValueError("Model must be initialized first with `initialize()`.")
            
            # 检查数据长度
            if X.shape[0] < 1:
                print(f"⚠️ 变换数据长度不足: {X.shape[0]}")
                return X
            
            # 检查维度匹配
            if self.mean is not None and self.mean.shape[0] != X.shape[1]:
                print(f"⚠️ 均值维度不匹配: 期望{X.shape[1]}，实际{self.mean.shape[0]}")
                # 重新计算均值
                self.mean = np.mean(X, axis=0)
            
            # 去均值
            if self.mean is not None:
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
        except Exception as e:
            print(f"⚠️ transform失败: {e}")
            return X

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

    def _dynamic_whitening(self, block_data):
        """动态白化 - 对应MATLAB的dynamicWhitening"""
        n_pts = block_data.shape[1]
        
        # 检查数据长度是否足够
        if n_pts < 2:
            print(f"⚠️ 白化数据长度不足: {n_pts}，跳过白化更新")
            return
        
        # 定义自适应遗忘率
        if self.ff_profile == 'cooling':
            lambda_vals = self._gen_cooling_ff(self.counter + np.arange(1, n_pts + 1))
            if lambda_vals[0] < self.lambda_const:
                lambda_vals = np.full(n_pts, self.lambda_const)
        elif self.ff_profile == 'constant':
            lambda_vals = np.full(n_pts, self.lambda_const)
        elif self.ff_profile == 'adaptive':
            lambda_vals = np.full(n_pts, self.lambda_k[-1] if len(self.lambda_k) > 0 else self.lambda_const)
        
        # 使用在线RLS白化块更新规则更新球化矩阵
        v = self.whitening_matrix @ block_data  # 预白化数据
        lambda_avg = 1 - lambda_vals[n_pts // 2]  # 中位数lambda
        Q_white = lambda_avg / (1 - lambda_avg) + np.trace(v.T @ v) / n_pts
        self.whitening_matrix = (1 / lambda_avg) * (self.whitening_matrix - 
                                                   v @ v.T / n_pts / Q_white @ self.whitening_matrix)

    def _dynamic_orica(self, block_data):
        """动态ORICA - 对应MATLAB的dynamicOrica"""
        n_chs, n_pts = block_data.shape
        
        # 检查数据长度是否足够
        if n_pts < 2:
            print(f"⚠️ ORICA数据长度不足: {n_pts}，跳过ORICA更新")
            return
        
        f = np.zeros((n_chs, n_pts))
        
        # 使用先前的权重矩阵计算源激活
        y = self.W @ block_data
        
        # 为超高斯和次高斯选择非线性函数
        f[self.kurtosis_sign, :] = -2 * np.tanh(y[self.kurtosis_sign, :])  # 超高斯
        f[~self.kurtosis_sign, :] = 2 * np.tanh(y[~self.kurtosis_sign, :])  # 次高斯
        
        # 计算非平稳性指数和源动态方差
        if self.eval_convergence:
            model_fitness = np.eye(n_chs) + y @ f.T / n_pts
            variance = block_data * block_data
            if self.Rn is None:
                self.Rn = model_fitness
            else:
                self.Rn = (1 - self.leaky_avg_delta) * self.Rn + self.leaky_avg_delta * model_fitness
            self.non_stat_idx = np.linalg.norm(self.Rn, 'fro')
        
        # 计算遗忘率
        data_range = np.arange(1, n_pts + 1)
        if self.ff_profile == 'cooling':
            self.lambda_k = self._gen_cooling_ff(self.counter + data_range)
            if len(self.lambda_k) > 0 and self.lambda_k[0] < self.lambda_const:
                self.lambda_k = np.full(n_pts, self.lambda_const)
            self.counter += n_pts
        elif self.ff_profile == 'constant':
            self.lambda_k = np.full(n_pts, self.lambda_const)
        elif self.ff_profile == 'adaptive':
            if self.min_non_stat_idx is None:
                self.min_non_stat_idx = self.non_stat_idx
            self.min_non_stat_idx = max(min(self.min_non_stat_idx, self.non_stat_idx), 1)
            ratio_of_norm_rn = self.non_stat_idx / self.min_non_stat_idx
            self.lambda_k = self._gen_adaptive_ff(data_range, ratio_of_norm_rn)
        
        # 使用在线递归ICA块更新规则更新权重矩阵
        lambda_prod = np.prod(1.0 / (1.0 - self.lambda_k))
        Q = 1 + self.lambda_k * (np.sum(f * y, axis=0) - 1)
        self.W = lambda_prod * (self.W - y @ np.diag(self.lambda_k / Q) @ f.T @ self.W)
        
        # 正交化权重矩阵
        eigenvals, eigenvecs = np.linalg.eigh(self.W @ self.W.T)
        self.W = eigenvecs @ np.diag(1/np.sqrt(eigenvals)) @ eigenvecs.T @ self.W

    def fit_block(self, data, num_passes=1):
        """
        块更新拟合 - 对应MATLAB orica.m的主要逻辑
        
        Args:
            data: 输入数据 (channels, samples)
            num_passes: 数据遍历次数
        """
        n_chs, n_pts = data.shape
        
        if self.verbose:
            print(f"使用{'在线' if self.use_rls_whitening else '离线'}白化方法")
            print(f"运行ORICA，遗忘因子策略: {self.ff_profile}")
        
        # 初始化白化
        if not self.use_rls_whitening:
            if self.verbose:
                print("使用预白化方法")
            # 预白化
            cov_matrix = np.cov(data.T)
            self.whitening_matrix = 2.0 * np.linalg.inv(np.sqrtm(cov_matrix))
        
        # 白化数据
        data = self.whitening_matrix @ data
        
        # 将数据分成块进行在线块更新
        min_block_size = min(self.block_size_ica, self.block_size_white)
        num_blocks = n_pts // min_block_size
        
        if self.verbose:
            import time
            start_time = time.time()
        
        for it in range(num_passes):
            for bi in range(num_blocks):
                # 计算数据范围
                start_idx = bi * n_pts // num_blocks
                end_idx = min(n_pts, (bi + 1) * n_pts // num_blocks)
                data_range = slice(start_idx, end_idx)
                block_data = data[:, data_range]
                
                # 在线白化
                if self.use_rls_whitening:
                    self._dynamic_whitening(block_data)
                    block_data = self.whitening_matrix @ block_data
                
                # 动态ORICA
                self._dynamic_orica(block_data)
                
                if self.verbose and bi % (num_blocks // 10) == 0:
                    progress = (it * num_blocks + bi) / (num_passes * num_blocks) * 100
                    print(f" 进度: {progress:.0f}%")
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            print(f"完成。耗时: {elapsed_time:.2f} 秒")
        
        return self.W, self.whitening_matrix
