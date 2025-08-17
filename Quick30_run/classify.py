import numpy as np
from sklearn.linear_model import SGDClassifier
from scipy.signal import welch
import time

def extract_bandpower_features(data, fs=250):
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

X_list = []
y_list = []

def collect_labeled_data(stream_receiver, fs=250, duration=60):
    print(f"请保持当前情绪状态，采集{duration}秒数据...")
    time.sleep(duration)  # 实际应用中应为采集循环
    buffer_data = stream_receiver.get_buffer_data(data_type='processed')
    features = extract_bandpower_features(buffer_data, fs=fs)
    print("请输入当前标签（arousal, valence），高为1，低为0，例如高唤醒高愉悦输入 1 1：")
    label = input().strip().split()
    label = [int(x) for x in label]
    X_list.append(features)
    y_list.append(label)
    print("采集完成！")


def train_initial_model(X_list, y_list):
    X = np.array(X_list)
    y = np.array(y_list)
    clf = SGDClassifier()
    # 只支持一维标签，这里以arousal为例
    clf.fit(X, y[:, 0])
    return clf


def online_update_and_predict(stream_receiver, clf, fs=250, duration=10):
    print(f"采集{duration}秒新数据用于实时分类...")
    time.sleep(duration)
    buffer_data = stream_receiver.get_buffer_data(data_type='processed')
    features = extract_bandpower_features(buffer_data, fs=fs)
    pred = clf.predict([features])[0]
    print(f"当前预测arousal标签：{pred}")
    print("如需用新标签增量训练请输入（否则回车跳过）：")
    label = input().strip()
    if label:
        label = int(label)
        clf.partial_fit([features], [label])
        print("模型已增量更新。")
    else:
        print("未更新模型。")

# 假设你有stream_receiver对象
fs = 250  # 采样率
for i in range(4):
    collect_labeled_data(stream_receiver, fs=fs, duration=60)

clf = train_initial_model(X_list, y_list)

while True:
    online_update_and_predict(stream_receiver, clf, fs=fs, duration=10)