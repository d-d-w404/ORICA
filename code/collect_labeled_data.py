import numpy as np
import time
from scipy.signal import welch

# ========== 配置参数 ==========
FS = 250  # 采样率
DURATION = 60  # 每次采集时长（秒）
N_COLLECT = 4  # 采集次数
SAVE_PATH = 'labeled_eeg_data.npz'  # 保存文件名

# ========== 特征提取函数 ==========
def extract_bandpower_features(data, fs=FS):
    """
    data: shape [channels, samples]
    返回每个通道的delta, theta, alpha, beta, gamma能量
    """
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    features = []
    for ch in data:
        f, Pxx = welch(ch, fs=fs, nperseg=fs*2)
        for band in bands.values():
            idx = np.logical_and(f >= band[0], f < band[1])
            features.append(np.sum(Pxx[idx]))
    return np.array(features)

# ========== 主采集流程 ==========
def main(stream_receiver):
    X_list = []
    y_list = []
    for i in range(N_COLLECT):
        print(f"\n第{i+1}/{N_COLLECT}次采集：请保持目标情绪状态，{DURATION}秒后自动采集...")
        time.sleep(DURATION)
        buffer_data = stream_receiver.get_buffer_data(data_type='processed')
        if buffer_data is None:
            print("❌ 未获取到数据，跳过本次采集")
            continue
        features = extract_bandpower_features(buffer_data, fs=FS)
        print("请输入本次采集的标签（如：高唤醒高愉悦输入 1 1）：")
        label = input().strip().split()
        label = [int(x) for x in label]
        X_list.append(features)
        y_list.append(label)
        print("✅ 本次采集完成！")
    X = np.array(X_list)
    y = np.array(y_list)
    np.savez(SAVE_PATH, X=X, y=y)
    print(f"\n所有数据已保存到 {SAVE_PATH}")

# ========== 用法示例 ==========
if __name__ == '__main__':
    from stream_receiver import StreamReceiver
    # 请根据你的实际情况初始化stream_receiver
    stream_receiver = StreamReceiver()
    main(stream_receiver) 