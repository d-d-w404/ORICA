import numpy as np
from scipy.signal import welch
import time


def analyze_bandpower(chunk, raw, srate, labels, gui=None):
    try:
        freqs, psd = welch(chunk, fs=srate, nperseg=srate, axis=1)
        band_dict = {}
        for band, (fmin, fmax) in {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }.items():
            idx = (freqs >= fmin) & (freqs <= fmax)
            if np.any(idx):
                band_dict[band] = float(np.mean(psd[:, idx]))

        if gui and gui.bandpower_plot:
            gui.bandpower_plot.update_bandpower(band_dict)

    except Exception as e:
        print("❌ analyze_bandpower 错误:", e)



def heavy_analysis(chunk, raw, srate, labels):
    t0 = time.time()



    #time.sleep(2)  # 等待 2 秒

    #print("🧪 [重计算开始]")

    try:
        # === Step 0: 输入检查 ===
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 2:
            print("❗ chunk 非法，跳过分析。shape:", np.shape(chunk))
            return
        if not isinstance(raw, np.ndarray) or raw.ndim != 2:
            print("❗ raw 非法，跳过分析。shape:", np.shape(raw))
            return
        if chunk.shape[1] < srate:
            print("⚠️ 数据不足 1 秒，跳过")
            return


        # === Step 1: bandpower ===
        for data_name, data in zip(['cleaned', 'raw'], [chunk, raw]):

            #print("data.shape =",data.shape)#data.shape = (29, 2500)
            #print(srate)
            freqs,psd  = welch(data, fs=srate, nperseg=srate, axis=1)
            #print("psd.shape =", psd.shape)  # psd.shape = (29,251)
            #代表了29个通道，在251个频率上的功率密度值
            #后续直接对 psd[:, 8:13] → 求 alpha 波段的能量
            #print(psd)

            #print("freqs.shape =", freqs.shape)  # freqs.shape = (251,)
            #print(freqs)


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
                band_power = np.mean(psd[:, idx], axis=1)#计算每个通道在该频段的平均功率
                #print(f"{data_name} | {band} power: {np.mean(band_power):.2f}")

        # === Step 2: Hjorth 参数 ===
        def compute_hjorth(data):
            d1 = np.diff(data, axis=1)#求导（axis=1，对行） 信号变化速度
            d2 = np.diff(d1, axis=1)#信号变化加速度
            activity = np.var(data, axis=1)#振幅高 → 活动强；比如觉醒时脑电 activity 较大。
            mobility = np.sqrt(np.var(d1, axis=1) / activity)#越高的 mobility，表示脑电越活跃于高频段，如 beta、gamma。
            complexity = np.sqrt(np.var(d2, axis=1) / np.var(d1, axis=1))#高复杂度可能表示注意力转移、思维活跃、感知突变等。
            return activity, mobility, complexity

        hjorth_act, hjorth_mob, hjorth_comp = compute_hjorth(chunk)

        # === Step 3: 协方差矩阵特征值分解 ===
        cov = np.cov(chunk)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)[::-1]

        # === Step 4: 构造特征 + 假分类 ===
        dummy_features = np.concatenate([hjorth_act, eigvals[:10]])
        dummy_prediction = int(np.sum(dummy_features) % 3)  # 假装有个分类器

        t1 = time.time()
        #print(f"✅ [重计算完成] 耗时: {(t1 - t0) * 1000:.1f} ms，预测类: {dummy_prediction}")

    except Exception as e:
        print("❌ heavy_analysis 错误:", e)