import numpy as np
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_regression


class ORICA:
    def __init__(self, n_components, learning_rate=0.001):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.W = np.eye(n_components)  # 解混矩阵
        self.mean = None
        self.whitening_matrix = None
        self.whitened = False

    def _center(self, X):
        self.mean = np.mean(X, axis=0)
        return X - self.mean

    def _whiten(self, X):
        cov = np.cov(X, rowvar=False)
        d, E = np.linalg.eigh(cov)
        #D_inv = np.diag(1.0 / np.sqrt(d + 1e-5))
        D_inv = np.diag(1.0 / np.sqrt(d + 1e-2))
        self.whitening_matrix = E @ D_inv @ E.T
        return X @ self.whitening_matrix.T

    def _gx(self, y):
        g_y = np.tanh(y)
        g_prime = 1 - g_y ** 2
        return g_y, g_prime

    def _g(self, y):
        # Gaussian nonlinearity
        g_y = y * np.exp(-0.5 * y**2)
        g_prime = (1 - y**2) * np.exp(-0.5 * y**2)
        return g_y, g_prime

    def partial_fit(self, x_t):
        x_t = x_t.reshape(-1, 1)
        y_t = self.W @ x_t
        g_y, _ = self._g(y_t)
        I = np.eye(self.n_components)
        delta_W = self.learning_rate * ((I - g_y @ y_t.T) @ self.W)
        self.W += delta_W
        return y_t.ravel()

    def fit_transform(self, X):
        X = self._center(X)
        X = self._whiten(X)
        self.whitened = True
        Y = []
        for x_t in X:
            y_t = self.partial_fit(x_t)
            Y.append(y_t)
        return np.array(Y)

    def transform(self, X):
        if not self.whitened:
            raise ValueError("Model must be fitted first with `fit_transform()`.")
        X = X - self.mean
        X = X @ self.whitening_matrix.T
        Y = (self.W @ X.T).T
        return Y

    def inverse_transform(self, Y):
        # 将独立成分还原为 EEG 空间（仅当需要合成 EEG 时使用）
        Xw = np.linalg.pinv(self.W) @ Y.T
        X = Xw.T @ np.linalg.pinv(self.whitening_matrix).T + self.mean
        return X


    def get_W(self):
        """
        获取当前的解混矩阵W。
        返回:
        - W: ICA分离矩阵，形状为(n_components, n_components)
        """
        return self.W

    def evaluate_separation(self, Y):
        """
        返回每个成分的峰度，用于评估是否成功分离
        """
        k = kurtosis(Y, axis=0, fisher=False)
        return k

    def rank_components_by_kurtosis(self, Y):
        """
        返回根据 |kurtosis| 排序后的成分索引
        """
        k = self.evaluate_separation(Y)
        indices = np.argsort(-np.abs(k))
        return indices, k

    def calc_mutual_info_matrix(self, sources):
        """
        计算互信息矩阵
        参数:
        - sources: 分离后的源信号，形状为 (n_components, n_samples)
        返回:
        - MI: 互信息矩阵，形状为 (n_components, n_components)
        """
        n = sources.shape[0]
        MI = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    MI[i, j] = mutual_info_regression(sources[i, :].reshape(-1, 1), sources[j, :])[0]
        return MI