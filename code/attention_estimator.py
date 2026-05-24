import numpy as np
from scipy.signal import welch
from scipy.special import expit

class RealTimeAttentionEstimator:
    def __init__(self, gui=None,receiver=None):
        self.gui = gui
        self.history = []
        self.max_history = 30  # 平滑用的历史窗口
        self.receiver=receiver

    def extract_attention_score(self, chunk, srate):
        # ✅ Step 0: 选择特定通道进行 bandpower（如 AF7=0, AF8=1）

        selected_channels_name=['F7','F8','T7','T8','Fpz']
        selected_channels = self.receiver.channel_manager.get_indices_by_labels(selected_channels_name)
        #print(selected_channels)# 替换成你想用的通道索引
        chunk = chunk[selected_channels, :]  # 只保留感兴趣通道
        if chunk.shape[0] != 5:
            
            raise ValueError("❌ 找不到所有指定通道：Fpz, F7, F8, T7, T8")

        ref = chunk[0, :]  # Fpz 是第一个通道
        chunk = chunk[1:, :] - ref  # 将其余通道减去 Fpz，实现 re-referencing




        # Step 1: Hjorth 参数
        d1 = np.diff(chunk, axis=1)
        d2 = np.diff(d1, axis=1)
        activity = np.mean(np.var(chunk, axis=1))
        complexity = np.mean(np.sqrt(np.var(d2, axis=1) / np.var(d1, axis=1)))

        # Step 2: 频段功率
        nperseg = min(srate, chunk.shape[1])
        freqs, psd = welch(chunk, fs=srate, nperseg=nperseg, axis=1)
        def band_power(fmin, fmax):
            idx = (freqs >= fmin) & (freqs <= fmax)
            return np.mean(psd[:, idx])

        alpha = band_power(8, 13)
        theta = band_power(4, 8)
        beta = band_power(13, 30)
        gamma = band_power(30, 45)

        # #Step 3: Attention Score
        # score = (
        #     -alpha * 0.6 +   # alpha ↓ 表示集中
        #     +theta * 0.2    # theta ↑
        #     +beta * 0.5
        #     #+gamma * 0.3
        #     # +activity * 0.1 -
        #     # complexity * 0.2
        # )
        #
        # print(score)
        #
        # normalized = float(expit(score))  # sigmoid(score)
        # print(normalized)
        # return normalized


        epsilon = 1e-6
        # Step 1: Engagement Index
        engagement = (beta + epsilon) / (alpha + theta + epsilon)

        # Step 2: 平滑处理（滑动窗口）
        self.history.append(engagement)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        engagement_avg = np.mean(self.history)

        # Step 3: 动态归一化（历史最大最小范围）
        E_min = min(self.history)
        E_max = max(self.history)
        range_ = max(E_max - E_min, epsilon)  # 避免除以 0
        normalized = (engagement_avg - E_min) / range_
        normalized = float(np.clip(normalized, 0.0, 1.0))

        # print(
        #     f"[Attention] α={alpha:.2f}, θ={theta:.2f}, β={beta:.2f} | Engagement_avg={engagement_avg:.2f}, Norm={normalized:.2f}")
        return normalized

    def callback(self, chunk, raw, srate, labels):
        try:
            score = self.extract_attention_score(chunk, srate)
            #print(score)
            #self.history.append(score)
            if len(self.history) > self.max_history:
                self.history.pop(0)

            smoothed = np.mean(self.history)

            if self.gui:
                self.gui.update_attention_circle(smoothed)

        except Exception as e:
            print("❌ 注意力评分错误:", e)