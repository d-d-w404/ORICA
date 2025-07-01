from ORICA import ORICA
import numpy as np


class ORICAProcessor:
    def __init__(self, n_components=None, max_samples=1000):
        self.n_components = n_components
        self.max_samples = max_samples
        self.ica = None
        self.data_buffer = None
        self.eog_indices = []  # indices of components identified as eye artifacts

    def fit(self, data):
        """Fit ICA on the buffered EEG data"""
        assert data.shape[0] == self.n_components, f"Expected {self.n_components} channels, got {data.shape[0]}"

        if data.shape[1] < self.max_samples:
            return False  # Not enough data yet


        # ica = FastICA(n_components=self.n_components or data.shape[0],#都是29一样的
        #               max_iter=2000, tol=1e-3, random_state=42)
        #
        # sources = ica.fit_transform(data.T)

        print("dog")
        self.ica = ORICA(n_components=min(self.n_components, data.shape[0]), learning_rate=0.001)

        print("dog")
        sources = self.ica.fit_transform(data.T).T  # shape: (components, samples)

        print("dog")

        self.identify_eye_artifacts(sources.T)
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

    def identify_eye_artifacts(self, components):
        powers = np.sum(components ** 2, axis=1)  # 每个分量的总能量
        threshold = np.percentile(powers, 90)
        self.eog_indices = [i for i, p in enumerate(powers) if p > threshold]

        print("EOG artifact indices (filtered):", self.eog_indices)

    def transform(self, new_data):
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
            if self.data_buffer.shape[1] > self.max_samples:
                self.data_buffer = self.data_buffer[:, -self.max_samples:]

        print(np.array(self.data_buffer).shape)

        return self.data_buffer.shape[1] >= self.max_samples