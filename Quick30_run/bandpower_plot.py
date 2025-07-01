from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import matplotlib.pyplot as plt

class BandpowerStreamVisualizer(QWidget):
    def __init__(self, bands=('delta', 'theta', 'alpha', 'beta', 'gamma'), history_length=300):
        super().__init__()
        self.bands = bands
        self.history_length = history_length
        self.history = {band: [0] * history_length for band in self.bands}
        self.timestamps = list(range(-history_length + 1, 1))

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # 初始化 line 和 legend label
        self.offsets = np.arange(len(self.bands)) * 30
        self.lines = {}
        self.text_labels = {}

        for i, band in enumerate(self.bands):
            line, = self.ax.plot([], [], label=f"{band}: 0.00", linewidth=1.5)
            self.lines[band] = line

        # 设置图像基本信息
        self.ax.set_xlim(-self.history_length + 1, 0)
        self.ax.set_xlabel("Time (chunks)")
        self.ax.set_title("Real-time Bandpower Waveform")

        # 外部图例
        self.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        self.fig.tight_layout()

    def update_bandpower(self, band_values):
        for band in self.bands:
            if band in band_values:
                self.history[band].append(band_values[band])
                if len(self.history[band]) > self.history_length:
                    self.history[band].pop(0)

        self.ax.clear()

        # 1. 计算中位数 + 高百分位（抗异常）
        all_vals = np.concatenate([np.array(self.history[band]) for band in self.bands])
        sorted_vals = np.sort(all_vals)
        low_percentile = int(0.05 * len(sorted_vals))
        high_percentile = int(0.95 * len(sorted_vals))
        safe_vals = sorted_vals[low_percentile:high_percentile]
        safe_max = np.max(safe_vals) if len(safe_vals) > 0 else 1.0

        # 2. 添加偏移
        safe_max += 40

        # 3. 限制最大高度（防炸）
        safe_max = min(safe_max, 200)

        # ✅ 绘制每个频段曲线并标注当前值
        for i, band in enumerate(self.bands):
            data = np.array(self.history[band])

            recent_window = data[-100:] if len(data) >= 100 else data
            band_min = np.min(recent_window)
            band_max = np.max(recent_window)
            range_ = band_max - band_min
            if range_ < 1e-3:
                range_ = 1.0
            norm_data = (data - band_min) / range_

            scaled_data = norm_data * 20 + self.offsets[i]

            self.ax.plot(self.timestamps[-len(data):], scaled_data,
                         label=f"{band}: {data[-1]:.2f}", linewidth=1.5)

        # 4. 设置 Y 轴等属性
        self.ax.set_ylim(-10, safe_max)
        self.ax.set_yticks(self.offsets)
        self.ax.set_yticklabels(self.bands)
        self.ax.set_xlabel("Time (chunks)")
        self.ax.set_title("Real-time Bandpower Waveform")

        # ✅ 图例放右侧，并避免警告
        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        self.fig.tight_layout()
        self.canvas.draw()