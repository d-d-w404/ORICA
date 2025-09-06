import numpy as np
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_regression
from ORICA_calibration import ORICACalibration
from scipy.linalg import sqrtm
import scipy

class ORICA_final:
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
        
        print("initialize done")
        # try:
        #     # 去均值
        #     #X_init = self._center(X_init)
        #     #似乎np.cov自带中心化
            
        #     # 白化
        #     if self.use_rls_whitening:
        #         # 使用RLS白化初始化
        #         self._rls_whiten_initialize(X_init)
        #         # 对初始数据进行批量白化
        #         X_init = self._whiten(X_init)
        #     else:
        #         # 使用传统批量白化
        #         X_init = self._whiten(X_init)
            
        #     self.whitened = True
        #     print(f"✅ ORICA初始化完成: n_components={self.n_components}, 数据形状={X_init.shape}")
        # except Exception as e:
        #     print(f"⚠️ ORICA初始化失败: {e}")
        #     self.whitened = False
        
        return X_init




    def dynamic_whitening(self,blockdata, data_range, state, lambda_const,gamma,lambda_0):
        """
        RLS在线白化算法 - 与MATLAB源码完全一致
        
        参数:
        blockdata: 当前数据块 [nChs × nPts]
        data_range: 数据范围索引
        state: 状态字典，包含icasphere等
        adaptive_ff: 自适应遗忘因子参数
        lambda_const: 遗忘因子下限常数
        
        返回:
        state: 更新后的状态
        """
        nPts = blockdata.shape[1]

        



        # 计算遗忘因子 - 完全按照MATLAB的逻辑
        # MATLAB: lambda = genCoolingFF(state.counter+dataRange, adaptiveFF.gamma, adaptiveFF.lambda_0);

        lambda_values = self.gen_cooling_ff(state['counter'] + data_range, gamma, lambda_0)
        lambda_const=1 - np.exp(-1 / np.inf)

        # MATLAB: if lambda(1) < adaptiveFF.lambda_const
        #         lambda = repmat(adaptiveFF.lambda_const,1,nPts);
                
        if lambda_values[0] < lambda_const:
            lambda_values = np.full(len(data_range), lambda_const)



            

        # save_txt("23.txt",lambda_values.reshape(1, -1))
        # 注意：Lambda下限检查应该在dynamic_whitening函数内部进行，与MATLAB一致
        # 这里只传递原始lambda值
        adaptive_ff = {'lambda': lambda_values}


        #save_txt("201.txt",blockdata)

        # 1. 使用当前白化矩阵预处理数据
        v = state['icasphere'] @ blockdata  # 预白化数据
        v = self.snap_to_kbits(v, k=38)


        #print("blockdata",blockdata)
        np.set_printoptions(precision=16, suppress=False, linewidth=200)


        # save_txt("21.txt",v)

        
        
        # 2. 计算遗忘因子 - 修复：使用中间值，与MATLAB完全一致
        # MATLAB: lambda_avg = 1 - lambda(ceil(end/2));
        #lambda_avg = 1 - adaptive_ff['lambda'][len(adaptive_ff['lambda'])//2]
        lambda_avg = 1 - lambda_values[int(np.ceil(len(lambda_values) / 2)) - 1]

        # save_txt("28.txt",lambda_avg)
        

        # 3. RLS更新规则
        #QWhite = lambda_avg/(1-lambda_avg) + np.trace(v.T @ v) / nPts
        # 方式 1：显式共轭转置
        #QWhite = lambda_avg/(1 - lambda_avg) + (np.trace(v.conj().T @ v).real) / nPts
        # 方式 2：用 vdot（对第一个参数做共轭，再做内积；等价于 Frobenius 范数平方）
        #QWhite = lambda_avg/(1 - lambda_avg) + (np.vdot(v, v).real) / nPts
        # 方式 3：直接用 Frobenius 范数
        QWhite = lambda_avg/(1 - lambda_avg) + (np.linalg.norm(v, 'fro')**2) / nPts




        QWhite = self.snap_to_kbits(QWhite, k=38)

        # save_txt("22.txt",QWhite)

        # 4. 递归更新白化矩阵 - 与MATLAB完全一致

        #这里出了问题。。



        
        # MATLAB: state.icasphere = 1/lambda_avg * (state.icasphere - v * v' / nPts / QWhite * state.icasphere);
        update_term = (v @ v.T) / nPts / QWhite @ state['icasphere']


        
        state['icasphere'] = (1/lambda_avg) * (state['icasphere'] - update_term)








        

        
        return state

    def gen_cooling_ff(self,t, gamma, lambda_0):
        """
        生成冷却遗忘因子 - 与MATLAB源码完全一致
        MATLAB: lambda = lambda_0 ./ (t .^ gamma)
        
        参数:
        t: 时间点或数组
        gamma: 衰减率
        lambda_0: 初始遗忘因子
        
        返回:
        lambda: 遗忘因子值或数组
        """
        # 确保t不为0，避免除零错误
        # save_txt("24.txt",t.reshape(1, -1))
        # save_txt("25.txt",gamma)
        # save_txt("26.txt",lambda_0)
        t_safe = np.maximum(t, 1e-10)
        #lambda_values = lambda_0 / (t ** gamma)
        lambda_values = lambda_0 / np.power(t, gamma)


        # 用法
        #lambda_ = lambda_0 / np.power(t, gamma)
        lambda_values = self.snap_to_kbits(lambda_values, k=50)


        # save_txt("27.txt",lambda_values.reshape(1, -1))
        return lambda_values

    def snap_to_kbits(self,x, k=50):  # k < 52
        k=10
        x = np.asarray(x, dtype=np.float64)
        m, e = np.frexp(x)                     # x = m * 2**e，m∈[-1, -0.5)∪[0.5, 1)
        m = np.round(m * (1 << k)) / float(1 << k)  # 只保留 k 位尾数（纯2的幂，二进制精确）
        return np.ldexp(m, e)

    def dynamic_orica_cooling(self,blockdata, data_range, state=None, gamma=0.5, lambda_0=1.0):
        """
        极简 ORICA（cooling 版，lambda_const=0）
        blockdata: np.ndarray, shape=(n_chs, n_pts)

        state: dict，可为空；缺省时自动初始化：
        - icaweights: 解混矩阵 (n_chs, n_chs)
        - kurtsign  : True=super-gaussian, False=sub-gaussian（默认全 True）
        - counter   : 已处理样本计数
        - lambda_k  : 上次 λ（仅记录）

        返回：更新后的 state
        """
        X = np.asarray(blockdata, dtype=np.float64)
        n_chs, n_pts = X.shape

        # --- 初始化 state（若未给） ---
        if state is None:
            state = {}
        if "icaweights" not in state:
            state["icaweights"] = np.eye(n_chs, dtype=np.float32)
        if "kurtsign" not in state:
            state["kurtsign"] = np.ones(n_chs, dtype=bool)  # 全部按超高斯处理
        if "counter" not in state:
            state["counter"] = 0
        if "lambda_k" not in state:
            state["lambda_k"] = np.array([0.0], dtype=np.float32)

        W = state["icaweights"]

        # (1) 源激活
        Y = W @ X  # (n_chs, n_pts)
        # save_txt("5.txt",Y)
        #print("blockdata",blockdata)
        #print("Y",Y)

        # (2) 非线性（extended-Infomax 的符号）
        F = np.empty_like(Y)
        idx_sg  = state["kurtsign"]           # super-gaussian
        idx_sub = ~state["kurtsign"]          # sub-gaussian
        F[idx_sg, :]  = -2.0 * np.tanh(Y[idx_sg, :])
        F[idx_sub, :] =  2.0 * np.tanh(Y[idx_sub, :])

        # print("F",F.shape)
        # print("F",F)
        # print("Y",Y.shape)
        # print("Y",Y)
        # print("Y @ F.T",(Y @ F.T) / nPts)
        # print("nChs,n_pts",nChs,n_pts)


        evalConvergence = {}
        evalConvergence["profile"] = True
        evalConvergence["leakyAvgDelta"] = 0.01
        evalConvergence["leakyAvgDeltaVar"] = 1e-3


        if evalConvergence["profile"]:
            # modelFitness = I + (y @ f.T) / nPts
            modelFitness = np.eye(n_chs) + (Y @ F.T) / n_pts
            
            # variance = blockdata .* blockdata
            variance = blockdata * blockdata   # element-wise square
            
            if state.get("Rn") is None:
                state["Rn"] = modelFitness
                #print("state['Rn']",state["Rn"])
            else:
                delta = evalConvergence["leakyAvgDelta"]
                state["Rn"] = (1 - delta) * state["Rn"] + delta * modelFitness
                #print("state2['Rn']",state["Rn"])
            
            # Frobenius norm
            state["nonStatIdx"] = np.linalg.norm(state["Rn"], 'fro')

        




        # (3) cooling 遗忘因子（lambda_const=0，不设下限）

        lambda_k = self.gen_cooling_ff(state['counter'] + data_range, gamma, lambda_0)
        # print("state['counter']",state['counter'])
        # print("data_range",data_range)
        state['counter'] += n_pts
        lambda_const=1 - np.exp(-1 / np.inf)

        # MATLAB: if lambda(1) < adaptiveFF.lambda_const
        #         lambda = repmat(adaptiveFF.lambda_const,1,nPts);
                
        if lambda_k[0] < lambda_const:
            lambda_k = np.full(len(data_range), lambda_const)


        
        #print("lambda_k",lambda_k)



        # update weight matrix using online recursive ICA block update rule
        lambda_prod = np.prod(1.0 / (1 - lambda_k))
        Q = 1.0 + lambda_k * (np.sum(F * Y, axis=0) - 1.0)
        F=self.snap_to_kbits(F, k=44)



        # save_txt("31.txt",Y)
        # save_txt("32.txt",np.diag(lambda_k / Q))
        # save_txt("33.txt",F.T)
        # save_txt("34.txt",state['icaweights'])
        # save_txt("35.txt",lambda_prod)
        

        state['icaweights'] = lambda_prod * (state['icaweights'] - Y @ np.diag(lambda_k / Q) @ F.T @ state['icaweights'])


        # save_txt("36.txt",state['icaweights'])
        #print("state['icaweights']",state['icaweights'])



        # orthogonalize weight matrix 
        # V, D = np.linalg.eig(state['icaweights'] @ state['icaweights'].T)
        # D_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(D)))
        #state['icaweights'] = V @ D_sqrt_inv @ V.T @ state['icaweights']



        # save_txt("391.txt",state["icaweights"].T)
        # save_txt("392.txt",state["icaweights"] @ state["icaweights"].T)
        #D, V = np.linalg.eigh(state["icaweights"] @ state["icaweights"].T)
        #D, V = np.linalg.eigh(state["icaweights"] @ state["icaweights"].T)
        D, V = scipy.linalg.eigh(state["icaweights"] @ state["icaweights"].T)



        D = np.diag(D)

        D=self.snap_to_kbits(D, k=32)
        V=self.snap_to_kbits(V, k=32)




        # save_txt("38.txt",D)

        # save_txt("39.txt",np.abs(V))

        # print("V",V)
        # print("D",D)
        # print("np.diag(1.0/np.sqrt(D))",V @ np.diag(1.0/np.sqrt(D)))
        # print("state['icaweights']",state["icaweights"])



        # state["icaweights"] = V @ np.diag(1.0/np.sqrt(D)) @ V.T @ state["icaweights"]
        # state["icaweights"] = snap_to_kbits(state["icaweights"], k=40)

        # save_txt("37.txt",state["icaweights"])






        def inv_sqrt_from_D(D):
            """D 可以是 (n,) 的特征值向量 或 (n,n) 的对角矩阵。返回 inv(sqrt(D)) 的对角矩阵 (n,n)。"""
            D = np.asarray(D)
            if D.ndim == 1:              # w: 特征值向量
                d = D
            elif D.ndim == 2:            # 对角矩阵
                d = np.diag(D)
            else:
                raise ValueError("D 必须是一维(特征值向量)或二维(对角矩阵)")
            inv_sqrt = 1.0 / np.sqrt(d)  # 注意 0 会变成 inf，建议加一个极小阈值防零
            return np.diag(inv_sqrt)

        # === 代入你的式子 ===
        W = state["icaweights"]
        # 如果 V,D 来自 np.linalg.eigh(W @ W.conj().T):
        #   w, V = np.linalg.eigh(W @ W.conj().T)     # w: (n,), V: (n,n)
        #   D = w  # 此时 D 就用向量，更方便
        # 否则如果你已经有对角矩阵 D = np.diag(w)，也没问题。

        M = V @ inv_sqrt_from_D(D) @ V.conj().T      # 等价于 MATLAB 的 V / sqrt(D) * V'
        state["icaweights"] = M @ W                  # 再与原 W 相乘

        # snap & 保存（注意 state["icaweights"] 必须是 2D）
        state["icaweights"] = self.snap_to_kbits(state["icaweights"], k=40)
        assert state["icaweights"].ndim == 2 and state["icaweights"].shape == (V.shape[0], V.shape[0])
        # save_txt("37.txt", state["icaweights"])



        



        
        

        
        


        
        #     t = counter + (1..n_pts)
        # t = state["counter"] + np.arange(1, n_pts + 1, dtype=float)
        # lam = lambda_0 / (t ** gamma)                   # shape (n_pts,)
        # state["lambda_k"] = lam
        # state["counter"] += n_pts

        # # (4) 递归块更新
        # # s_j = sum_i F_ij * Y_ij
        # s = np.sum(F * Y, axis=0)                      # (n_pts,)
        # Q = 1.0 + lam * (s - 1.0)                      # (n_pts,)
        # col_scale = (lam / Q)[None, :]                 # (1, n_pts)
        # M = (Y * col_scale) @ F.T                      # (n_chs, n_chs)
        # lambda_prod = float(np.prod(1.0 / (1.0 - lam)))  # 标量
        # W = lambda_prod * (W - M @ W)

        # # (5) 对称去相关（orthogonalize）
        # WWt = W @ W.T
        # d, E = np.linalg.eigh(WWt)                     # WWt = E diag(d) E^T
        # eps = 1e-12
        # Dm12 = E @ np.diag(1.0 / np.sqrt(d + eps)) @ E.T
        # W = Dm12 @ W

        # # 写回
        # state["icaweights"] = W
        return state


    def gen_adaptive_ff(self,data_range,
                        lambda_vec,
                        decayRateAlpha,
                        upperBoundBeta,
                        transBandWidthGamma,
                        transBandCenter,
                        ratioOfNormRn):
        """
        复现 MATLAB: genAdaptiveFF
        参数
        ----
        data_range : 1D array-like
            本 block 的样本索引（只需长度）
        lambda_vec : 1D array-like
            历史 λ 序列；只使用最后一个元素 lambda_vec[-1]
        decayRateAlpha : float
        upperBoundBeta : float
        transBandWidthGamma : float
        transBandCenter : float
        ratioOfNormRn : float
            非平稳性指标相对最小值的比值

        返回
        ----
        lambdas : np.ndarray, shape = (len(data_range),)
            本 block 内逐样本的 λ 序列
        """
        n_pts = len(data_range)
        lam0 = float(np.asarray(lambda_vec)[-1])  # lambda(end)
        # gainForErrors = upperBoundBeta * 0.5 * (1 + tanh((ratio - center)/bandwidth))
        gain = upperBoundBeta * 0.5 * (1.0 + np.tanh((ratioOfNormRn - transBandCenter) / transBandWidthGamma))

        n = np.arange(1, n_pts + 1, dtype=np.float32)     # n = 1..N
        one_plus_g = 1.0 + gain

        # term1 = (1+g)^n * lam0
        term1 = (one_plus_g ** n) * lam0

        # term2 = decayRateAlpha * ((1+g)^(2n-1) - (1+g)^(n-1)) / g * lam0^2
        # 需要处理 g -> 0 的极限：当 g≈0 时，分子 ~ n*g，因此整体极限为 n
        eps = 1e-12
        if abs(gain) < eps:
            frac = n  # 极限：((1+g)^(2n-1) - (1+g)^(n-1)) / g  ->  n
        else:
            frac = ((one_plus_g ** (2.0 * n - 1.0)) - (one_plus_g ** (n - 1.0))) / gain

        term2 = decayRateAlpha * frac * (lam0 ** 2)

        lambdas = term1 - term2
        return np.asarray(lambdas, dtype=np.float32)


    def mmul_strict(self,A, B):
        A = np.asarray(A, dtype=np.float64, order='F')  # 列主序视图
        B = np.asarray(B, dtype=np.float64, order='F')
        m, k = A.shape
        k2, n = B.shape
        assert k == k2
        C = np.empty((m, n), dtype=np.float64, order='F')
        for j in range(n):            # 先列后行（匹配 MATLAB 线性索引）
            for i in range(m):
                s = 0.0
                for p in range(k):
                    s += A[i, p] * B[p, j]
                C[i, j] = s
        return C


    def orica_rls_whitening(self,data, block_size_white=8, num_pass=1, 
                            lambda_0=0.995, gamma=0.6, lambda_const=0.95, verbose=True):
        """
        ORICA RLS在线白化主函数 - 与MATLAB源码完全一致
        
        参数:
        data: 输入数据 [nChs × nPts]
        block_size_white: 白化块大小
        num_pass: 数据遍历次数
        lambda_0: 初始遗忘因子
        gamma: 衰减率
        lambda_const: 遗忘因子下限常数
        verbose: 是否显示进度
        
        返回:
        weights: ICA权重矩阵
        sphere: 最终白化矩阵
        """
        nChs, nPts = data.shape
        
        # 初始化状态
        state = {
            'icasphere': np.eye(nChs),  # 初始白化矩阵为单位矩阵
            'icaweights': np.eye(nChs),  # 初始ICA权重矩阵
            'counter': 0
        }

        # save_txt("13.txt", data)
        # 预白化整个数据集
        data = state['icasphere'] @ data  # 对应MATLAB: data = state.icasphere * data;
        # save_txt("14.txt", data)  # 保存预白化后的数据
        
        # 数据分块
        num_block = int(np.floor(nPts / min(block_size_white, block_size_white)))
        
        if verbose:

            import time
            start_time = time.time()
        
        for it in range(num_pass):
            # print("dog")
            # print(data[:3])
            #range(1)就只有一个数
            #for bi in range(11):
            for bi in range(num_block):
                # 计算当前数据块范围
                start_idx = int(np.floor(bi * nPts / num_block))
                end_idx = min(nPts, int(np.floor((bi + 1) * nPts / num_block)))
                data_range = np.arange(start_idx, end_idx)

                # save_txt("11.txt",data)
                # 提取当前数据块
                blockdata = data[:, data_range]
                # save_txt("12.txt",blockdata)
                nPts_block = blockdata.shape[1]



                #print("blockdataxxxxxxxxxxx",blockdata)
                
                # # 计算遗忘因子 - 完全按照MATLAB的逻辑
                # # MATLAB: lambda = genCoolingFF(state.counter+dataRange, adaptiveFF.gamma, adaptiveFF.lambda_0);
                # lambda_values = gen_cooling_ff(state['counter'] + data_range, gamma, lambda_0)
                # lambda_const=1 - np.exp(-1 / np.inf)

                # # MATLAB: if lambda(1) < adaptiveFF.lambda_const
                # #         lambda = repmat(adaptiveFF.lambda_const,1,nPts);
                
                # if lambda_values[0] < lambda_const:
                #     lambda_values = np.full(nPts_block, lambda_const)
                #     print(f"动态白化 - 应用下限: {lambda_const}")

                # print("shit")
                # print(state['counter'])
                # print(data_range)
                # print(lambda_values)
                # print(lambda_const)
                # print()
                
                # # 注意：Lambda下限检查应该在dynamic_whitening函数内部进行，与MATLAB一致
                # # 这里只传递原始lambda值
                #adaptive_ff = {'lambda': lambda_values}
                
                # 执行RLS白化更新
                state = self.dynamic_whitening(blockdata, data_range+1, state,  lambda_const, gamma,lambda_0)
                # save_txt("2.txt",state['icasphere'])




                # 更新计数器
                #state['counter'] += len(data_range)
                
                if verbose and bi % 10 == 0:
                    pass

                data[:, data_range] = state['icasphere'] @ data[:, data_range]
                #data[:, data_range] = mmul_strict(state['icasphere'], data[:, data_range])
                data[:, data_range] = self.snap_to_kbits(data[:, data_range], k=44)
                # save_txt("3.txt",data[:, data_range])
                
                state=self.dynamic_orica_cooling(data[:, data_range], data_range+1, state, gamma, lambda_0)
                



                # Xx = np.asarray(data[:, data_range], dtype=np.float64)
                # n_chsxxx, n_ptsxxx = Xx.shape
                # state['counter'] += n_ptsxxx

                
        
                # save_txt("4.txt",state['icaweights'])
                # print("finalweights",state['icaweights'])
                # print("finalsphere",state['icasphere'])





        if verbose:
            elapsed_time = time.time() - start_time

        
        return state['icaweights'], state['icasphere']





    def fit(self,data,
            block_size_white=8,
            num_pass=1,
            lambda_0=0.995,
            gamma=0.6,
            lambda_const=0.95,
            verbose=False):
        """
        用 ORICA_final.py 中的 orica_rls_whitening 进行白化+ORICA，并返回源信号（sources）。
        - 输入:
        data: np.ndarray, shape = (samples, channels)
        - 返回:
        sources: np.ndarray, shape = (samples, components)  # 与输入 samples 对齐
        weights: np.ndarray, shape = (components, components)  # icaweights
        sphere:  np.ndarray, shape = (components, components)  # icasphere
        """
        assert isinstance(data, np.ndarray) and data.ndim == 2, "data必须是(samples, channels)的二维ndarray"

        # 统一到 (channels, samples)
        X = data.T.astype(np.float64, copy=False)

        # 调用你文件内的白化+ORICA主流程，得到权重和白化矩阵

        weights, sphere = self.orica_rls_whitening(
            X,
            block_size_white=block_size_white,
            num_pass=num_pass,
            lambda_0=lambda_0,
            gamma=gamma,
            lambda_const=lambda_const,
            verbose=verbose
        )


        # y_orica = weights @ sphere @ X
        X_whitened = sphere @ X
        sources = weights @ X_whitened

        self.whitening_matrix = sphere
        
        self.W = weights




        # 返回为 (samples, components)
        return sources, weights, sphere


    def transform(self, X):
        """变换数据"""
        X_whitened = X @ self.whitening_matrix.T
        Y = (self.W @ X_whitened.T).T
        return Y

    
    def inverse_transform(self, Y):
        """逆变换"""
        Xw = np.linalg.pinv(self.W) @ Y.T
        X = Xw.T @ np.linalg.pinv(self.whitening_matrix).T 
        return X


    def transform_old(self, X):
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

    def inverse_transform_old(self, Y):
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

