import numpy as np
from paths import DEFAULT_DEMO_SET, TEMP_TXT_ROOT
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_regression
import scipy


class ORICA_final_new:
    def __init__(self, n_components, learning_rate=0.001, ortho_every=10,
                 use_rls_whitening=False, forgetting_factor=0.98,
                 nonlinearity='gaussian', block_size_ica=1, block_size_white=8,
                 ff_profile='cooling', tau_const=3, gamma=0.6, lambda_0=0.995,
                 num_subgaussian=0, eval_convergence=True, verbose=False, srate=500,
                 time_perm=False):
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
            time_perm: 是否对数据进行时间打乱（减少时间相关性）
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

        # 时间打乱参数
        self.time_perm = time_perm

        # 遗忘因子参数
        self.ff_profile = ff_profile
        self.srate = srate
        self.tau_const = tau_const
        self.gamma = gamma
        self.lambda_0 = lambda_0
        self.lambda_const = 1 - np.exp(-1 / (self.tau_const * self.srate)) if tau_const != np.inf else 0.98

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
        self.counter = 7681

        # RLS白化参数
        self.use_rls_whitening = use_rls_whitening
        self.forgetting_factor = forgetting_factor
        self.nonlinearity = nonlinearity

        # RLS白化相关变量
        if self.use_rls_whitening:
            self.C = None  # 协方差矩阵的逆
            self.t = 0     # 时间步计数器

        self.verbose = verbose
        self.record = None

    def initialize(self, X_init):
        """初始化ORICA

        Args:
            X_init: np.ndarray, shape = (channels, samples)，与 MNE/LSL 一致
        """
        if X_init.shape[1] < 2:
            return X_init

        if X_init.shape[0] != self.n_components:
            self.n_components = X_init.shape[0]
            self.W = np.eye(self.n_components)

        self.whitening_matrix = np.eye(self.n_components)
        return X_init

    def dynamic_whitening(self, blockdata, data_range, state, lambda_const, gamma, lambda_0):
        """
        RLS在线白化算法 - 与MATLAB源码完全一致

        参数:
        blockdata: 当前数据块 [nChs × nPts]
        data_range: 数据范围索引
        state: 状态字典，包含 icasphere 等
        lambda_const: 遗忘因子下限常数

        返回:
        state: 更新后的状态
        """
        nPts = blockdata.shape[1]

        # MATLAB: lambda = genCoolingFF(state.counter+dataRange, adaptiveFF.gamma, adaptiveFF.lambda_0)
        lambda_values = self.gen_cooling_ff(state['counter'] + data_range, gamma, lambda_0)

        # Quick30 使用固定 lambda_const，而非 cooling 下限判断
        if True:
            lambda_values = np.full(len(data_range), self.lambda_const)

        # 1. 预白化当前块
        v = state['icasphere'] @ blockdata
        v = self.snap_to_kbits(v, k=38)

        # 2. MATLAB: lambda_avg = 1 - lambda(ceil(end/2))
        lambda_avg = 1 - lambda_values[int(np.ceil(len(lambda_values) / 2)) - 1]

        # 3. RLS 白化更新
        QWhite = lambda_avg / (1 - lambda_avg) + (np.linalg.norm(v, 'fro') ** 2) / len(data_range)
        QWhite = self.snap_to_kbits(QWhite, k=38)

        # MATLAB: state.icasphere = 1/lambda_avg * (state.icasphere - v * v' / nPts / QWhite * state.icasphere)
        update_term = (v @ v.T) / nPts / QWhite @ state['icasphere']
        state['icasphere'] = (1 / lambda_avg) * (state['icasphere'] - update_term)

        return state

    def gen_cooling_ff(self, t, gamma, lambda_0):
        """
        生成冷却遗忘因子 - 与MATLAB源码完全一致
        MATLAB: lambda = lambda_0 ./ (t .^ gamma)
        """
        t_safe = np.maximum(t, 1e-10)
        lambda_values = lambda_0 / np.power(t, gamma)
        return self.snap_to_kbits(lambda_values, k=50)

    def snap_to_kbits(self, x, k=50):
        """保留 k 位尾数的 MATLAB 数值对齐占位；当前直接返回原值。"""
        return x

    def dynamic_orica_cooling(self, blockdata, data_range, state=None, gamma=0.5, lambda_0=1.0):
        """
        极简 ORICA（cooling 版）

        blockdata: np.ndarray, shape=(n_chs, n_pts)
        state 字段: icaweights, kurtsign, counter, lambda_k
        """
        X = np.asarray(blockdata, dtype=np.float64)
        n_chs, n_pts = X.shape

        if state is None:
            state = {}
        if "icaweights" not in state:
            state["icaweights"] = np.eye(n_chs, dtype=np.float32)
        if "kurtsign" not in state:
            state["kurtsign"] = np.ones(n_chs, dtype=bool)
        if "counter" not in state:
            state["counter"] = 0
        if "lambda_k" not in state:
            state["lambda_k"] = np.array([0.0], dtype=np.float32)

        W = state["icaweights"]

        # (1) 源激活
        Y = W @ X

        # (2) Extended-Infomax 非线性
        F = np.empty_like(Y)
        idx_sg = state["kurtsign"]
        idx_sub = ~state["kurtsign"]
        F[idx_sg, :] = -2.0 * np.tanh(Y[idx_sg, :])
        F[idx_sub, :] = np.tanh(Y[idx_sub, :]) - Y[idx_sub, :]

        evalConvergence = {
            "profile": True,
            "leakyAvgDelta": 0.01,
            "leakyAvgDeltaVar": 1e-3,
        }

        if evalConvergence["profile"]:
            modelFitness = np.eye(n_chs) + (Y @ F.T) / n_pts
            state['Rn'] = self.Rn
            if state.get("Rn") is None:
                state["Rn"] = modelFitness
            else:
                delta = evalConvergence["leakyAvgDelta"]
                state["Rn"] = (1 - delta) * state["Rn"] + delta * modelFitness
            state["nonStatIdx"] = np.linalg.norm(state["Rn"], 'fro')

        # (3) 遗忘因子 — Quick30 固定为 lambda_const
        lambda_k = self.gen_cooling_ff(state['counter'] + data_range, gamma, lambda_0)
        state['counter'] += n_pts
        if True:
            lambda_k = np.full(len(data_range), self.lambda_const)

        lambda_prod = np.prod(1.0 / (1 - lambda_k))
        Q = 1.0 + lambda_k * (np.sum(F * Y, axis=0) - 1.0)
        F = self.snap_to_kbits(F, k=44)

        state['icaweights'] = lambda_prod * (
            state['icaweights'] - Y @ np.diag(lambda_k / Q) @ F.T @ state['icaweights']
        )

        # 对称去相关 (orthogonalize)
        D, V = scipy.linalg.eigh(state["icaweights"] @ state["icaweights"].T)
        D = np.diag(D)
        D = self.snap_to_kbits(D, k=32)
        V = self.snap_to_kbits(V, k=32)

        def inv_sqrt_from_D(d_mat):
            """返回 inv(sqrt(D)) 的对角矩阵。"""
            d_arr = np.asarray(d_mat)
            if d_arr.ndim == 1:
                d = d_arr
            elif d_arr.ndim == 2:
                d = np.diag(d_arr)
            else:
                raise ValueError("D 必须是一维(特征值向量)或二维(对角矩阵)")
            inv_sqrt = 1.0 / np.sqrt(d)
            return np.diag(inv_sqrt)

        W = state["icaweights"]
        M = V @ inv_sqrt_from_D(D) @ V.conj().T  # 等价 MATLAB: V / sqrt(D) * V'
        state["icaweights"] = self.snap_to_kbits(M @ W, k=40)
        assert state["icaweights"].ndim == 2 and state["icaweights"].shape == (V.shape[0], V.shape[0])

        return state

    def gen_adaptive_ff(self, data_range,
                        lambda_vec,
                        decayRateAlpha,
                        upperBoundBeta,
                        transBandWidthGamma,
                        transBandCenter,
                        ratioOfNormRn):
        """复现 MATLAB: genAdaptiveFF"""
        n_pts = len(data_range)
        lam0 = float(np.asarray(lambda_vec)[-1])
        gain = upperBoundBeta * 0.5 * (1.0 + np.tanh((ratioOfNormRn - transBandCenter) / transBandWidthGamma))

        n = np.arange(1, n_pts + 1, dtype=np.float32)
        one_plus_g = 1.0 + gain
        term1 = (one_plus_g ** n) * lam0

        eps = 1e-12
        if abs(gain) < eps:
            frac = n
        else:
            frac = ((one_plus_g ** (2.0 * n - 1.0)) - (one_plus_g ** (n - 1.0))) / gain

        term2 = decayRateAlpha * frac * (lam0 ** 2)
        return np.asarray(term1 - term2, dtype=np.float32)

    def mmul_strict(self, A, B):
        """列主序矩阵乘，匹配 MATLAB 线性索引顺序。"""
        A = np.asarray(A, dtype=np.float64, order='F')
        B = np.asarray(B, dtype=np.float64, order='F')
        m, k = A.shape
        k2, n = B.shape
        assert k == k2
        C = np.empty((m, n), dtype=np.float64, order='F')
        for j in range(n):
            for i in range(m):
                s = 0.0
                for p in range(k):
                    s += A[i, p] * B[p, j]
                C[i, j] = s
        return C

    def orica_rls_whitening(self, data, block_size_white=8, num_pass=1,
                            lambda_0=0.995, gamma=0.6, lambda_const=0.95, verbose=True):
        """
        ORICA RLS 在线白化 + ORICA 主流程

        data: 输入数据 [nChs × nPts]
        返回: (icaweights, icasphere)
        """
        nChs, nPts = data.shape

        # center 仅用于 online whitening；mixtures 仍用原始 data
        data_center = data.copy()
        data_center -= data.mean(axis=1, keepdims=True)

        if self.whitening_matrix is not None and self.W is not None:
            state = {
                'icasphere': self.whitening_matrix,
                'icaweights': self.W,
                'counter': self.counter,
            }
        else:
            # MATLAB: [U,~,~] = svd(rand(nChs)); state.icasphere = U
            rand_mat = np.random.randn(nChs, nChs)
            Q, R = np.linalg.qr(rand_mat)
            signs = np.sign(np.diag(R))
            signs[signs == 0] = 1.0
            Q = Q * signs
            state = {
                'icasphere': Q,
                'icaweights': np.eye(nChs),
                'counter': 0,
            }

        numsplits = nPts // block_size_white

        if verbose:
            import time
            start_time = time.time()

        for _it in range(num_pass):
            for bi in range(numsplits):
                start = int(bi * nPts / numsplits)
                end = min(nPts, int((bi + 1) * nPts / numsplits))
                data_range = np.arange(start, end)
                blockdata = data_center[:, data_range]
                state = self.dynamic_whitening(
                    blockdata, data_range + 1, state, lambda_const, gamma, lambda_0
                )

        mixtures = state['icasphere'] @ data

        # 时间打乱 — 对应 MATLAB options.timeperm
        if self.time_perm:
            perm_idx = np.random.permutation(nPts)
        else:
            perm_idx = np.arange(nPts)

        block_size_orica = 32
        num_block_orica = int(np.floor(nPts / block_size_orica))

        for _it in range(num_pass):
            for bi in range(num_block_orica):
                start = int(bi * nPts / numsplits)
                end = min(nPts, int((bi + 1) * nPts / numsplits))
                data_range = np.arange(start, end)
                perm_data_range = perm_idx[data_range]
                state = self.dynamic_orica_cooling(
                    mixtures[:, perm_data_range], data_range + 1, state, gamma, lambda_0
                )

        if verbose:
            _ = time.time() - start_time

        self.record = state['icaweights']
        self.counter = state['counter']
        self.Rn = state['Rn']
        self.lambda_k = state['lambda_k']

        return state['icaweights'], state['icasphere']

    def fit(self, data,
            block_size_white=32,
            num_pass=1,
            lambda_0=0.5,
            gamma=0.6,
            lambda_const=0.95,
            verbose=False):
        """
        白化 + ORICA，返回源信号与权重矩阵。

        data: (channels, samples)
        返回: sources (components, samples), weights, sphere
        """
        assert isinstance(data, np.ndarray) and data.ndim == 2, "data必须是(channels, samples)的二维ndarray"

        X = data.astype(np.float64, copy=False)
        weights, sphere = self.orica_rls_whitening(
            X,
            block_size_white=block_size_white,
            num_pass=num_pass,
            lambda_0=lambda_0,
            gamma=gamma,
            lambda_const=lambda_const,
            verbose=verbose,
        )

        X_whitened = sphere @ X
        sources = weights @ X_whitened

        self.whitening_matrix = sphere
        self.W = weights

        return sources, weights, sphere

    def transform(self, X):
        """变换数据: (channels, samples) -> (components, samples)"""
        X_whitened = self.whitening_matrix @ X
        return self.W @ X_whitened

    def inverse_transform(self, Y):
        """逆变换: (components, samples) -> (channels, samples)"""
        Xw = np.linalg.pinv(self.W) @ Y
        return np.linalg.pinv(self.whitening_matrix) @ Xw

    def get_W(self):
        """获取解混矩阵"""
        return self.W

    def get_whitening_matrix(self):
        """获取白化矩阵"""
        return self.whitening_matrix

    def get_icawinv(self):
        """获取ICA逆矩阵"""
        return self.W @ self.whitening_matrix

    def get_sources(self):
        """获取源信号"""
        return self.sources

    def evaluate_separation(self, Y):
        """评估分离效果 - 使用峰度"""
        return kurtosis(Y, axis=0, fisher=False)

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


if __name__ == "__main__":
    import os
    from scipy.io import loadmat

    set_path = DEFAULT_DEMO_SET
    S = loadmat(set_path, squeeze_me=True, struct_as_record=False)
    EEG = S["EEG"]

    def _get(obj, name):
        return getattr(obj, name) if hasattr(obj, name) else obj[name]

    nbchan = int(_get(EEG, 'nbchan'))
    pnts = int(_get(EEG, 'pnts'))
    data_f = _get(EEG, 'data')

    if isinstance(data_f, (str, bytes, np.str_)):
        fdt_path = data_f if os.path.isabs(data_f) else os.path.join(os.path.dirname(set_path), data_f)
        X = np.fromfile(fdt_path, dtype='<f4', count=nbchan * pnts).reshape((nbchan, pnts), order='F')
    else:
        X = np.asarray(data_f, dtype=np.float32, order='F')

    X = X.astype(np.float64, copy=False)

    srate = float(EEG.srate)
    xmin = float(getattr(EEG, "xmin", 0.0))
    t0, t1 = 0.0, 60.0
    start = max(0, int(np.floor((t0 - xmin) * srate)))
    end = min(int(np.floor((t1 - xmin) * srate)), X.shape[1])
    X = X[:, start:end]

    orica = ORICA_final_new(
        n_components=X.shape[0],
        use_rls_whitening=True,
        block_size_white=8,
        block_size_ica=1,
        gamma=0.6,
        lambda_0=0.995,
    )
    sources, weights, sphere = orica.fit(
        X,
        block_size_white=8,
        num_pass=1,
        lambda_0=0.995,
        gamma=0.6,
        lambda_const=0.95,
    )

    output_file = TEMP_TXT_ROOT / "orica_results_demo.mat"
    scipy.io.savemat(output_file, {
        'sources': sources,
        'weights': weights,
        'sphere': sphere,
        'X_original': X,
    })
