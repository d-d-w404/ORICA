from tkinter.constants import YES
from mne.viz.backends.renderer import VALID_3D_BACKENDS
import numpy as np
import mne
import numpy as np
from scipy.linalg import sqrtm
import scipy


np.set_printoptions(precision=17, floatmode='maxprec', suppress=False, linewidth=200)













import numpy as np

def dynamic_whitening(blockdata, data_range, state, lambda_const,gamma,lambda_0):
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

    lambda_values = gen_cooling_ff(state['counter'] + data_range, gamma, lambda_0)
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
    v = snap_to_kbits(v, k=38)


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




    QWhite = snap_to_kbits(QWhite, k=38)

    # save_txt("22.txt",QWhite)

    # 4. 递归更新白化矩阵 - 与MATLAB完全一致

    #这里出了问题。。



    
    # MATLAB: state.icasphere = 1/lambda_avg * (state.icasphere - v * v' / nPts / QWhite * state.icasphere);
    update_term = (v @ v.T) / nPts / QWhite @ state['icasphere']


    
    state['icasphere'] = (1/lambda_avg) * (state['icasphere'] - update_term)








    

    
    return state

def gen_cooling_ff(t, gamma, lambda_0):
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
    lambda_values = snap_to_kbits(lambda_values, k=50)


    # save_txt("27.txt",lambda_values.reshape(1, -1))
    return lambda_values




def snap_to_kbits(x, k=50):  # k < 52
    k=10
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)                     # x = m * 2**e，m∈[-1, -0.5)∪[0.5, 1)
    m = np.round(m * (1 << k)) / float(1 << k)  # 只保留 k 位尾数（纯2的幂，二进制精确）
    return np.ldexp(m, e)






def dynamic_orica_cooling(blockdata, data_range, state=None, gamma=0.5, lambda_0=1.0):
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
        modelFitness = np.eye(nChs) + (Y @ F.T) / n_pts
        
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

    lambda_k = gen_cooling_ff(state['counter'] + data_range, gamma, lambda_0)
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
    F=snap_to_kbits(F, k=44)



    # save_txt("31.txt",Y)
    # save_txt("32.txt",np.diag(lambda_k / Q))
    # save_txt("33.txt",F.T)
    # save_txt("34.txt",state['icaweights'])
    # save_txt("35.txt",lambda_prod)
    

    state['icaweights'] = lambda_prod * (state['icaweights'] - Y @ np.diag(lambda_k / Q) @ F.T @ state['icaweights'])


    # save_txt("36.txt",state['icaweights'])
    #print("state['icaweights']",state['icaweights'])

# 修复后的代码
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

    D=snap_to_kbits(D, k=32)
    V=snap_to_kbits(V, k=32)




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
    state["icaweights"] = snap_to_kbits(state["icaweights"], k=40)
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





def gen_adaptive_ff(data_range,
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



def save_txt_old(filename, X, folder="temp_txt"):
    """
    Save a numeric matrix to a txt file with:
      1) Decimal values printed with '%.17g' (round-trip for float64)
      2) IEEE754 hex (big-endian, 64-bit) per element for exact bit-level fidelity
    
    参数:
    filename: 文件名
    X: 要保存的数据矩阵
    folder: 保存文件夹，默认为 "temp_txt"
    """
    import numpy as np
    import struct
    import os

    # 创建文件夹（如果不存在）
    os.makedirs(folder, exist_ok=True)
    
    # 构建完整的文件路径
    filepath = os.path.join(folder, filename)
    
    X = np.asarray(X, dtype=np.float64)  # 强制为 float64 与 MATLAB 对齐
    nrows, ncols = X.shape

    with open(filepath, "w", encoding="utf-8") as f:
        # 元信息
        f.write(f"# rows={nrows} cols={ncols} dtype=float64\n")

        # 十进制（%.17g）
        for i in range(nrows):
            f.write("\t".join(f"{v:.17g}" for v in X[i]) + "\n")

        # IEEE754 十六进制（大端、64位）
        f.write("# IEEE754 hex (big-endian, 64-bit)\n")
        for i in range(nrows):
            hexrow = [struct.pack(">d", float(v)).hex() for v in X[i]]
            f.write("\t".join(hexrow) + "\n")
    



def save_txt(filename, X):
    """
    只需传 (filename, X)。兼容:
      - 标量 -> 1x1
      - 1D 向量 -> Nx1 (与 MATLAB 常见列向量一致)
      - 2D 矩阵 -> 原样
    十进制使用 Python 格式 '.17g'；HEX 为逐元素行序，big-endian 64-bit。
    输出目录固定到 DEFAULT_DIR（自动创建）。
    """
    import os, struct
    import numpy as np

    # === 内置参数（可按需改）===
    DEFAULT_DIR = r"D:\work\Python_Project\ORICA\temp_txt"
    DECIMAL_FMT = ".17g"         # Python 的格式，不带 %
    ROUND_NDEC  = None           # 例如 10：np.round(X, 10)
    HEADER_LINE = "class=double" # 与 MATLAB 风格对齐
    HEX_HEADER  = "# IEEE754 hex (big-endian logical order, 64-bit)\n"

    # 路径
    if os.path.isabs(filename):
        filepath = filename
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    else:
        os.makedirs(DEFAULT_DIR, exist_ok=True)
        filepath = os.path.join(DEFAULT_DIR, filename)

    # 数据准备：兼容 0D/1D/2D
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 0:        # 标量 -> 1x1
        X = X.reshape(1, 1)
    elif X.ndim == 1:      # 1D -> Nx1（列向量，贴近 MATLAB）
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        raise ValueError(f"X 必须是 0/1/2 维，当前 ndim={X.ndim}, shape={X.shape}")

    if ROUND_NDEC is not None:
        X = np.round(X, ROUND_NDEC)  # 十进制与 HEX 都来自同一份量化后的数组

    nrows, ncols = X.shape

    # 写文件
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# rows={nrows} cols={ncols} {HEADER_LINE}\n")

        # 十进制（逐元素按行）
        for i in range(nrows):
            line = "\t".join(f"{float(v):{DECIMAL_FMT}}" for v in X[i])
            f.write(line + "\n")

        # IEEE754 HEX（逐元素按行，big-endian, 64-bit）
        f.write(HEX_HEADER)
        for i in range(nrows):
            hexrow = [struct.pack(">d", float(v)).hex() for v in X[i]]
            f.write("\t".join(hexrow) + "\n")





def mmul_strict(A, B):
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



def orica_rls_whitening(data, block_size_white=8, num_pass=1, 
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
            state = dynamic_whitening(blockdata, data_range+1, state,  lambda_const, gamma,lambda_0)
            # save_txt("2.txt",state['icasphere'])




            # 更新计数器
            #state['counter'] += len(data_range)
            
            if verbose and bi % 10 == 0:
                pass

            data[:, data_range] = state['icasphere'] @ data[:, data_range]
            #data[:, data_range] = mmul_strict(state['icasphere'], data[:, data_range])
            data[:, data_range] = snap_to_kbits(data[:, data_range], k=44)
            # save_txt("3.txt",data[:, data_range])
            
            state=dynamic_orica_cooling(data[:, data_range], data_range+1, state, gamma, lambda_0)
            



            # Xx = np.asarray(data[:, data_range], dtype=np.float64)
            # n_chsxxx, n_ptsxxx = Xx.shape
            # state['counter'] += n_ptsxxx

            
    
            # save_txt("4.txt",state['icaweights'])
            # print("finalweights",state['icaweights'])
            # print("finalsphere",state['icasphere'])





    if verbose:
        elapsed_time = time.time() - start_time

    
    return state['icaweights'], state['icasphere']















# 使用示例
if __name__ == "__main__":
    # # 生成测试数据
    # np.random.seed(42)
    # nChs, nPts = 16, 1000
    # test_data = np.random.randn(nChs, nPts)

    # 读入 .set（同目录下若有 .fdt 会自动配对）


    raw = mne.io.read_raw_eeglab('D:\work\matlab_project\orica-master\orica-master\SIM_STAT_16ch_3min.set', preload=True, verbose='error')


    # 基本信息
    sfreq      = raw.info['sfreq']          # 采样率
    ch_names   = raw.info['ch_names']       # 通道名列表
    n_channels = raw.info['nchan']
    n_times    = raw.n_times

    # # 取 numpy 数组：形状 = (n_channels, n_times)
    # #X = raw.get_data()
    # # X = raw.get_data()
    # # print_full("X",X[0:3,0:3])
    


    # X = raw.get_data().astype(np.float64)
    # X = X * 1e6   # 转换成 µV
    # #X = np.round(X, 10)   # 保留小数点后 5 位



    # 替代你那几行（不要再用 raw.get_data() * 1e6 了）
    import os
    import numpy as np
    from scipy.io import loadmat

    set_path = r"D:\work\matlab_project\orica-master\orica-master\SIM_STAT_16ch_3min.set"

    S = loadmat(set_path, squeeze_me=True, struct_as_record=False)
    EEG = S['EEG']

    def _get(obj, name):
        # 兼容 scipy 加载成对象或字典的两种情况
        return getattr(obj, name) if hasattr(obj, name) else obj[name]

    nbchan = int(_get(EEG, 'nbchan'))
    pnts   = int(_get(EEG, 'pnts'))
    data_f = _get(EEG, 'data')  # 外部 .fdt 文件名或直接内嵌矩阵

    if isinstance(data_f, (str, bytes, np.str_)):
        fdt_path = data_f if os.path.isabs(data_f) else os.path.join(os.path.dirname(set_path), data_f)
        # 直接读取 .fdt 的原始 float32（EEGLAB 按列写入）→ 重塑为 (channels, time)
        X = np.fromfile(fdt_path, dtype='<f4', count=nbchan * pnts).reshape((nbchan, pnts), order='F')
    else:
        # 少见：数据内嵌在 .set
        X = np.asarray(data_f, dtype=np.float32, order='F')

    X = X.astype(np.float64, copy=False)   # float32→float64 是精确映射















    # save_txt("1.txt", X)#[:,0:3])








 



    n = 20
    #X = X * 1e6   # 转换成 µV

    
    # 获取数据的通道数和时间点数，使用与之前随机数据相同的变量名
    nChs, nPts = X.shape

    
    # 运行RLS白化 - 使用实际的EEG数据X，而不是test_data

    weights, sphere= orica_rls_whitening(
        X,  # 使用实际的EEG数据X
        block_size_white=8, 
        num_pass=1, 
        lambda_0=0.995, 
        gamma=0.6,
        lambda_const=0.95,  # 添加遗忘因子下限常数
        verbose=True
    )
    

    
    # 验证白化效果

    # save_txt("40.txt",X)
    # save_txt("41.txt",sphere)
    # save_txt("42.txt",weights)
    
    # 计算白化后的数据（对应MATLAB: X_whitened = EEG.icasphere * EEG.data）
    X_whitened = sphere @ X
    # save_txt("4x.txt",X_whitened)
    
    # 计算最终的ORICA结果（对应MATLAB: y_orica = EEG.icaweights * EEG.icasphere * EEG.data）
    y_orica = weights @ sphere @ X
    # save_txt("43.txt",y_orica)
    
    # 应用snap_to_kbits（对应MATLAB中的snap_to_kbits调用）
    y_orica = snap_to_kbits(y_orica, k=48)
    # save_txt("44.txt",y_orica)
    
    # 计算白化后数据的协方差矩阵（用于验证白化效果）
    cov_whitened = np.cov(X_whitened)
    

    
    # # #保存结果

    try:
        import os
        from datetime import datetime
        
        # 创建输出目录
        output_dir = "./ORICA_results_onlinewhitening"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = "SIM_STAT_16ch_3min"
        
        # 保存白化后的数据为txt（对应MATLAB的X_whitened）
        txt_w_filename = f"{base_filename}_online_whitened_{timestamp}.txt"
        txt_w_path = os.path.join(output_dir, txt_w_filename)
        
        with open(txt_w_path, 'w', encoding='utf-8') as f:
            f.write("通道名称: " + " ".join(ch_names) + "\n")
            f.write(f"采样率: {sfreq}\n")
            f.write("在线白化后数据 (X_whitened = sphere @ X):\n")
            # 每行一个通道（源1、源2...），每列一个样本
            for i in range(X_whitened.shape[0]):
                row_str = f"源{i+1:2d}\t" + "\t".join([f"{val:.6f}" for val in X_whitened[i, :]])
                f.write(row_str + "\n")
        

        
        # 保存最终的ORICA结果（对应MATLAB的y_orica）
        txt_orica_filename = f"{base_filename}_online_orica_{timestamp}.txt"
        txt_orica_path = os.path.join(output_dir, txt_orica_filename)
        
        with open(txt_orica_path, 'w', encoding='utf-8') as f:
            f.write("通道名称: " + " ".join(ch_names) + "\n")
            f.write(f"采样率: {sfreq}\n")
            f.write("ORICA独立成分结果 (y_orica = weights @ sphere @ X):\n")
            # 每行一个独立成分，每列一个样本
            for i in range(y_orica.shape[0]):
                row_str = f"源{i+1:2d}\t" + "\t".join([f"{val:.6f}" for val in y_orica[i, :]])
                f.write(row_str + "\n")
        

        
        # 保存白化矩阵为txt
        txt_s_filename = f"{base_filename}_online_sphere_{timestamp}.txt"
        txt_s_path = os.path.join(output_dir, txt_s_filename)
        
        with open(txt_s_path, 'w') as f:
            f.write(f"在线白化矩阵 (形状: {sphere.shape})\n")
            f.write("通道名称: " + " ".join(ch_names) + "\n")
            f.write("矩阵数据:\n")
            for i in range(sphere.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in sphere[i, :]])
                f.write(row_str + "\n")
        

        
        # 保存权重矩阵为txt
        txt_w_filename = f"{base_filename}_online_weights_{timestamp}.txt"
        txt_w_path = os.path.join(output_dir, txt_w_filename)
        
        with open(txt_w_path, 'w') as f:
            f.write(f"ICA权重矩阵 (形状: {weights.shape})\n")
            f.write("矩阵数据:\n")
            for i in range(weights.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in weights[i, :]])
                f.write(row_str + "\n")
        
        
        

        
    except Exception as e:
        pass

"""
源信号数据 (完整23041个样本):
源\样本	  1	  2	  3	  4	  5	  
"""