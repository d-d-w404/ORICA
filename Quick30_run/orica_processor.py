from ORICA import ORICA
import numpy as np


class ORICAProcessor:
    def __init__(self, n_components=None, max_samples=1000, srate=None):
        self.n_components = n_components
        self.max_samples = max_samples
        self.srate = srate  # ✅ 保存采样率
        self.ica = None
        self.data_buffer = None
        self.eog_indices = []  # indices of components identified as eye artifacts

    def fit(self, data):
        """Fit ICA on the buffered EEG data"""
        assert data.shape[0] == self.n_components, f"Expected {self.n_components} channels, got {data.shape[0]}"

        if data.shape[1] < self.max_samples:
            return False  # Not enough data yet

        print()
        # ica = FastICA(n_components=self.n_components or data.shape[0],#都是29一样的
        #               max_iter=2000, tol=1e-3, random_state=42)
        #
        # sources = ica.fit_transform(data.T)

        self.ica = ORICA(n_components=min(self.n_components, data.shape[0]), learning_rate=0.001)
        sources = self.ica.fit_transform(data.T).T  # shape: (components, samples)（29,1500）
        # print("source")
        # print(sources.shape)
        self.identify_eye_artifacts(sources, self.srate)
        return True

    # def identify_eye_artifacts(self, components):
    #     """Heuristic: identify eye components as those with high frontal power and low frequency"""
    #     self.eog_indices = []
    #     for i, comp in enumerate(components):
    #         power = np.sum(comp ** 2)
    #         if power > np.percentile([np.sum(c ** 2) for c in components], 90):
    #             self.eog_indices.append(i)
    #
    #     print("EOG artifact:",self.eog_indices)

    def identify_eye_artifacts(self, components, srate):

        self.eog_indices = []

        for i, comp in enumerate(components):

            fft_vals = np.abs(np.fft.rfft(comp))
            freqs = np.fft.rfftfreq(comp.shape[0], 1 / srate)
            low_freq_power = np.sum(fft_vals[(freqs >= 0.1) & (freqs <= 4)])#0.1-4hz的低频信号
            total_power = np.sum(fft_vals)
            ratio = low_freq_power / (total_power + 1e-10)


            if ratio > 0.2:  # 如果低频占比超过阈值，认为是 EOG
                self.eog_indices.append(i)

        print("EOG artifact indices (low-freq based):", self.eog_indices)

    def transform(self, new_data):
        #去除伪影的独立成分后重新映射回原来的通道。
        if self.ica is None:
            return new_data

        sources = self.ica.transform(new_data.T)
        sources[:, self.eog_indices] = 0  # Zero out EOG components
        cleaned = self.ica.inverse_transform(sources)
        return cleaned.T

    def update_buffer(self, new_chunk):
        if self.data_buffer is None:
            self.data_buffer = new_chunk
        else:
            self.data_buffer = np.concatenate([self.data_buffer, new_chunk], axis=1)
            #当数据足够的时候才生成self.data_buffer,之后一直保持1500的长度，并且不断移动窗口
            if self.data_buffer.shape[1] > self.max_samples:
                self.data_buffer = self.data_buffer[:, -self.max_samples:]

        print("data_buffer")
        print(np.array(self.data_buffer).shape)

        return self.data_buffer.shape[1] >= self.max_samples