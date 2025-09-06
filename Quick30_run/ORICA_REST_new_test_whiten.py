import mne
import numpy as np
from scipy.linalg import sqrtm
# 读入 .set（同目录下若有 .fdt 会自动配对）
raw = mne.io.read_raw_eeglab('D:\work\matlab_project\orica-master\orica-master\SIM_STAT_16ch_3min.set', preload=True, verbose='error')

# 基本信息
sfreq      = raw.info['sfreq']          # 采样率
ch_names   = raw.info['ch_names']       # 通道名列表
n_channels = raw.info['nchan']
n_times    = raw.n_times

# 取 numpy 数组：形状 = (n_channels, n_times)
X = raw.get_data()
print(X.dtype) 

print(sfreq, n_channels, n_times, X.shape)

n = 20
X = X * 1e6   # 转换成 µV
for ch_idx, ch_name in enumerate(raw.info["ch_names"]):
    print(f"{ch_name:>8}: " + " ".join(f"{v:8.4f}" for v in X[ch_idx, :n]))



def whiten(X):
    """传统批量白化 - 使用特征值分解"""
    # 检查数据长度是否足够
    print(f"\n🔄 开始白化过程...")
    print(f"   输入数据形状: {X.shape}")
    print(f"   数据类型: {X.dtype}")
    
    # 计算协方差矩阵
    print(f"\n📊 步骤1: 计算协方差矩阵")
    cov = np.cov(X, rowvar=False)
    print(f"   协方差矩阵形状: {cov.shape}")
    
    # 显示完整的协方差矩阵
    print(f"\n🔢 完整协方差矩阵:")
    print("-" * 120)
    # 打印表头
    header = "通道\\通道"
    for j in range(cov.shape[1]):
        header += f"{j+1:>10}"
    print(header)
    print("-" * 120)
    
    # 打印矩阵内容
    for i in range(cov.shape[0]):
        row_str = f"{i+1:>8}"
        for j in range(cov.shape[1]):
            row_str += f"{cov[i][j]:}"
        print(row_str)
    print("-" * 120)
    print(cov[0][1])
    
    # 特征值分解
    print(f"\n📊 步骤2: 特征值分解")
    d, E = np.linalg.eigh(cov)
    print(f"   特征值: {d}")
    print(f"   特征值范围: [{np.min(d):.6f}, {np.max(d):.6f}]")
    print(f"   特征向量矩阵形状: {E.shape}")
    
    # 显示完整的特征向量矩阵
    print(f"\n🔢 完整特征向量矩阵:")
    print("-" * 120)
    # 打印表头
    header = "通道\\通道"
    for j in range(E.shape[1]):
        header += f"{j+1:>10}"
    print(header)
    print("-" * 120)
    
    # 打印矩阵内容
    for i in range(E.shape[0]):
        row_str = f"{i+1:>8}"
        for j in range(E.shape[1]):
            row_str += f"{E[i, j]:10.6f}"
        print(row_str)
    print("-" * 120)
    
    # 计算逆平方根对角矩阵
    print(f"\n📊 步骤3: 计算逆平方根对角矩阵")
    D_inv = np.diag(1.0 / np.sqrt(d + 1e-2))  # 防止除0
    print(f"   逆平方根对角矩阵范围: [{np.min(np.diag(D_inv)):.6f}, {np.max(np.diag(D_inv)):.6f}]")
    
    # 显示完整的逆平方根对角矩阵
    print(f"\n🔢 完整逆平方根对角矩阵:")
    print("-" * 120)
    # 打印表头
    header = "通道\\通道"
    for j in range(D_inv.shape[1]):
        header += f"{j+1:>10}"
    print(header)
    print("-" * 120)
    
    # 打印矩阵内容
    for i in range(D_inv.shape[0]):
        row_str = f"{i+1:>8}"
        for j in range(D_inv.shape[1]):
            row_str += f"{D_inv[i, j]:10.6f}"
        print(row_str)
    print("-" * 120)
    
    # 构建白化矩阵
    print(f"\n📊 步骤4: 构建白化矩阵")
    whitening_matrix =2* E @ D_inv @ E.T
    print(f"   白化矩阵形状: {whitening_matrix.shape}")
    
    # 显示完整的白化矩阵
    print(f"\n🔢 完整白化矩阵:")
    print("-" * 120)
    # 打印表头
    header = "通道\\通道"
    for j in range(whitening_matrix.shape[1]):
        header += f"{j+1:>10}"
    print(header)
    print("-" * 120)
    
    # 打印矩阵内容
    for i in range(whitening_matrix.shape[0]):
        row_str = f"{i+1:>8}"
        for j in range(whitening_matrix.shape[1]):
            row_str += f"{whitening_matrix[i, j]:10.6f}"
        print(row_str)
    print("-" * 120)









    print(f"\n📊 步骤5: 构建白化矩阵")
    x=2.0 *np.linalg.inv(sqrtm(cov))
    print(f"   xxxx白化矩阵形状: {x.shape}")
    
    # 显示完整的白化矩阵
    print(f"\n🔢 xxxx完整白化矩阵:")
    print("-" * 120)
    # 打印表头
    header = "xxxx通道\\通道"
    for j in range(x.shape[1]):
        header += f"{j+1:>10}"
    print(header)
    print("-" * 120)

    # 打印矩阵内容
    for i in range(x.shape[0]):
        row_str = f"{i+1:>8}"
        for j in range(x.shape[1]):
            row_str += f"{x[i, j]:10.6f}"
        print(row_str)
    print("-" * 120)


    




    
    # 应用白化矩阵
    print(f"\n📊 步骤5: 应用白化矩阵")
    Xwhtie = X @ whitening_matrix.T
    print(f"   白化后数据形状: {Xwhtie.shape}")
    print(f"   白化后数据均值: {np.mean(Xwhtie):.6f}")
    print(f"   白化后数据标准差: {np.std(Xwhtie):.6f}")
    
    # 验证白化效果
    print(f"\n🔍 步骤6: 验证白化效果")
    cov_whitened = np.cov(Xwhtie.T, rowvar=True)
    print(f"   白化后协方差矩阵形状: {cov_whitened.shape}")
    
    # 显示完整的白化后协方差矩阵
    print(f"\n🔢 完整白化后协方差矩阵:")
    print("-" * 120)
    # 打印表头
    header = "通道\\通道"
    for j in range(cov_whitened.shape[1]):
        header += f"{j+1:>10}"
    print(header)
    print("-" * 120)
    
    # 打印矩阵内容
    for i in range(cov_whitened.shape[0]):
        row_str = f"{i+1:>8}"
        for j in range(cov_whitened.shape[1]):
            row_str += f"{cov_whitened[i, j]:10.6f}"
        print(row_str)
    print("-" * 120)


        # 应用白化矩阵
    print(f"\n📊 步骤5: 应用白化矩阵")

    Xwhtie = X @ x.T
    print(f"   xxx白化后数据形状: {Xwhtie.shape}")
    print(f"   白化后数据均值: {np.mean(Xwhtie):.6f}")
    print(f"   白化后数据标准差: {np.std(Xwhtie):.6f}")
    
    # 验证白化效果
    print(f"\n🔍 步骤6: 验证白化效果")
    cov_whitened = np.cov(Xwhtie.T, rowvar=True)
    print(f"   白化后协方差矩阵形状: {cov_whitened.shape}")
    
    # 显示完整的白化后协方差矩阵
    print(f"\n🔢 完整白化后协方差矩阵:")
    print("-" * 120)
    # 打印表头
    header = "通道\\通道"
    for j in range(cov_whitened.shape[1]):
        header += f"{j+1:>10}"
    print(header)
    print("-" * 120)
    
    # 打印矩阵内容
    for i in range(cov_whitened.shape[0]):
        row_str = f"{i+1:>8}"
        for j in range(cov_whitened.shape[1]):
            row_str += f"{cov_whitened[i, j]:10.6f}"
        print(row_str)
    print("-" * 120)






    print(f"\n🎉 白化过程完成！")
    return Xwhtie, whitening_matrix
    #state.icasphere = 2.0*inv(sqrtm(double(cov(data'))));




def whiten_easy(X):
    #协方差矩阵
    cov = np.cov(X, rowvar=False)
    #白化矩阵
    x=2.0 *np.linalg.inv(sqrtm(cov))
    #白化后的数据
    Xwhtie = X @ x.T
    #验证白化后的数据是否cov为I 这里是4I
    cov_whitened = np.cov(Xwhtie.T, rowvar=True)

# 调用白化函数
print(f"\n🚀 开始执行白化...")
X_whitened, whitening_matrix = whiten(X.T)

print(f"\n" + "="*80)
print("📊 白化结果摘要")
print("="*80)
print(f"📈 输入数据: {X.shape}")
print(f"📈 白化后数据: {X_whitened.shape}")
print(f"🔢 白化矩阵: {whitening_matrix.shape}")
print(f"✅ 白化完成！")