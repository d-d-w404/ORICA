import numpy as np
from scipy.signal import butter, filtfilt, welch
import time

class EEGSignalProcessor:
    @staticmethod
    #先于callback，用于receiver的预处理部分
    def eeg_filter(data, srate, cutoff=0.5, order=2):
        nyq = 0.5 * srate
        if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
            low, high = cutoff
            if low == 0:
                mode = 'low'
                normal_cutoff = high / nyq
            elif high == 0:
                mode = 'high'
                normal_cutoff = low / nyq
            else:
                mode = 'band'
                normal_cutoff = [low / nyq, high / nyq]
        else:
            return data  # 不处理非法 cutoff 格式

        b, a = butter(order, normal_cutoff, btype=mode, analog=False)
        return filtfilt(b, a, data, axis=1)

    @staticmethod
    def clean_bad_channels(chunk, labels=None, threshold_uv=200):
        stds = np.std(chunk, axis=1)
        bad_indices = np.where(stds > threshold_uv)[0]

        if len(bad_indices) > 0:
            good_channels = [i for i in range(chunk.shape[0]) if i not in bad_indices]
            if good_channels:
                mean_signal = np.mean(chunk[good_channels, :], axis=0)
                for i in bad_indices:
                    chunk[i, :] = mean_signal

                if labels:
                    bad_names = [labels[i] for i in bad_indices]
                    print(f"⚠️ 替换了异常通道: {bad_names}")
                else:
                    print(f"⚠️ 替换了异常通道: 索引 {bad_indices}")
        return chunk

    @staticmethod
    def heavy_analysis( chunk, raw, srate, labels):
        t0 = time.time()
        try:
            if not isinstance(chunk, np.ndarray) or chunk.ndim != 2:
                print("❗ chunk 非法，跳过分析。shape x:", np.shape(chunk))
                return
            if not isinstance(raw, np.ndarray) or raw.ndim != 2:
                print("❗ raw 非法，跳过分析。shape:", np.shape(raw))
                return
            if chunk.shape[1] < srate:
                print("⚠️ 数据不足 1 秒，跳过")
                return

            for data_name, data in zip(['cleaned', 'raw'], [chunk, raw]):
                freqs, psd = welch(data, fs=srate, nperseg=srate, axis=1)
                for band, (fmin, fmax) in {
                    'delta': (1, 4),
                    'theta': (4, 8),
                    'alpha': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, 45)
                }.items():
                    idx = (freqs >= fmin) & (freqs <= fmax)
                    if not np.any(idx):
                        print(f"⚠️ Band {band} not found in freqs, skipping.")
                        continue
                    band_power = np.mean(psd[:, idx], axis=1)

            def compute_hjorth(data):
                d1 = np.diff(data, axis=1)
                d2 = np.diff(d1, axis=1)
                activity = np.var(data, axis=1)
                mobility = np.sqrt(np.var(d1, axis=1) / activity)
                complexity = np.sqrt(np.var(d2, axis=1) / np.var(d1, axis=1))
                return activity, mobility, complexity

            hjorth_act, hjorth_mob, hjorth_comp = compute_hjorth(chunk)

            cov = np.cov(chunk)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.sort(eigvals)[::-1]

            dummy_features = np.concatenate([hjorth_act, eigvals[:10]])
            dummy_prediction = int(np.sum(dummy_features) % 3)

            t1 = time.time()
            # print(f"✅ [重计算完成] 耗时: {(t1 - t0) * 1000:.1f} ms，预测类: {dummy_prediction}")

        except Exception as e:
            print("❌ heavy_analysis 错误:", e)
