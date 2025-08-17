import numpy as np
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_regression


class ORICA:
    def __init__(self, n_components, learning_rate=0.001, ortho_every=10):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.W = np.eye(n_components)  # 解混矩阵
        self.mean = None
        self.whitening_matrix = None
        self.whitened = False
        self.update_count = 0
        self.ortho_every = ortho_every  # 每隔多少次迭代正交化

    def _center(self, X):
        self.mean = np.mean(X, axis=0)
        return X - self.mean

    def _whiten(self, X):
        cov = np.cov(X, rowvar=False)
        d, E = np.linalg.eigh(cov)
        D_inv = np.diag(1.0 / np.sqrt(d + 1e-2))  # 防止除0
        self.whitening_matrix = E @ D_inv @ E.T
        return X @ self.whitening_matrix.T

    def _g(self, y):
        # Gaussian nonlinearity
        g_y = y * np.exp(-0.5 * y**2)
        g_prime = (1 - y**2) * np.exp(-0.5 * y**2)
        return g_y, g_prime

    def initialize(self, X_init):
        X_init = self._center(X_init)
        X_init = self._whiten(X_init)
        self.whitened = True
        return X_init

    def partial_fit(self, x_t):
        x_t = x_t.reshape(-1, 1)
        if not self.whitened:
            raise ValueError("Must call `initialize` with initial batch before `partial_fit`.")
        x_t = x_t - self.mean.reshape(-1, 1)
        x_t = self.whitening_matrix @ x_t
        y_t = self.W @ x_t
        g_y, _ = self._g(y_t)
        I = np.eye(self.n_components)
        delta_W = self.learning_rate * ((I - g_y @ y_t.T) @ self.W)
        self.W += delta_W

        self.update_count += 1
        if self.update_count % self.ortho_every == 0:
            U, _, Vt = np.linalg.svd(self.W)
            self.W = U @ Vt

        return y_t.ravel()

    def transform(self, X):
        if not self.whitened:
            raise ValueError("Model must be initialized first with `initialize()`.")
        X = X - self.mean
        X = X @ self.whitening_matrix.T
        Y = (self.W @ X.T).T
        return Y

    def inverse_transform(self, Y):
        Xw = np.linalg.pinv(self.W) @ Y.T
        X = Xw.T @ np.linalg.pinv(self.whitening_matrix).T + self.mean
        return X

    def get_W(self):
        return self.W


    def evaluate_separation(self, Y):
        k = kurtosis(Y, axis=0, fisher=False)
        return k

    def rank_components_by_kurtosis(self, Y):
        k = self.evaluate_separation(Y)
        indices = np.argsort(-np.abs(k))
        return indices, k

    def calc_mutual_info_matrix(self, sources):
        n = sources.shape[0]
        MI = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    MI[i, j] = mutual_info_regression(
                        sources[i, :].reshape(-1, 1), sources[j, :]
                    )[0]
        return MI

    

