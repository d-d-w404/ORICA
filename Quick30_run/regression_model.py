import threading

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


class RealTimeRegressor:
    def __init__(self, gui=None):
        self.model = SGDRegressor(learning_rate='adaptive', eta0=0.01)
        self.scaler = StandardScaler()
        self.first_fit_done = False
        self.lock = threading.Lock()

        self.latest_prediction = 0.0
        self.gui = gui  # ⬅️ GUI 实例，用于更新预测显示
        self.last_input_x = None

        self.feature_buffer = []  # 特征缓存
        self.max_pretrain_samples = 20  # 采样阈值

    def extract_features(self, chunk, srate):
        d1 = np.diff(chunk, axis=1)
        d2 = np.diff(d1, axis=1)
        activity = np.var(chunk, axis=1)
        mobility = np.sqrt(np.var(d1, axis=1) / activity)
        complexity = np.sqrt(np.var(d2, axis=1) / np.var(d1, axis=1))

        freqs, psd = welch(chunk, fs=srate, nperseg=srate, axis=1)
        features = list(activity) + list(mobility) + list(complexity)

        for fmin, fmax in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]:
            idx = (freqs >= fmin) & (freqs <= fmax)
            bandpower = np.mean(psd[:, idx], axis=1)
            features.extend(bandpower)

        return np.array(features).reshape(1, -1)

    def callback(self, chunk, raw, srate, labels):
        try:
            x = self.extract_features(chunk, srate)
            self.last_input_x = x

            if not self.first_fit_done:
                self.feature_buffer.append(x)
                print(f"📦 收集中: {len(self.feature_buffer)}/{self.max_pretrain_samples}")

                # 通知 GUI 激活评分输入
                if len(self.feature_buffer) >= self.max_pretrain_samples and self.gui:
                    self.gui.enable_initial_rating_ui(True)
                return

            # === 实时预测 ===
            x_scaled = self.scaler.transform(x)
            pred = self.model.predict(x_scaled)[0]
            self.latest_prediction = pred
            if self.gui:
                self.gui.update_prediction_display(pred)

        except Exception as e:
            print("❌ 实时回归错误:", e)

    def init_model_with_label(self, y_init):
        """由 GUI 提交初始评分后调用"""
        if len(self.feature_buffer) < self.max_pretrain_samples:
            print("❌ 特征不足，无法初始化模型")
            return

        X = np.vstack(self.feature_buffer)
        y = np.full((X.shape[0],), y_init)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.partial_fit(X_scaled, y)

        self.first_fit_done = True
        self.feature_buffer.clear()
        print("✅ 初始模型已完成训练")

    def update_with_feedback(self, y):
        if not self.first_fit_done:
            print("⚠️ 模型未初始化")
            return
        if self.last_input_x is None:
            print("⚠️ 尚无最新特征输入")
            return
        x_scaled = self.scaler.transform(self.last_input_x)
        self.model.partial_fit(x_scaled, [y])
        print("✅ 模型已通过反馈值更新")