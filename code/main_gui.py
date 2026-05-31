import os
import threading
import time

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from receiver import LSLStreamReceiver
from viewer import LSLStreamVisualizer
from attention_ball import AttentionBallWindow
from bandpower_plot import BandpowerStreamVisualizer
from ica_component_window import ICAComponentWindow
from topomap_visualizer import TopomapWindow
from bandpower_analyzer import BandpowerAnalyzer
from attention_analyzer import AttentionAnalyzer
from channel_selector_gui import ChannelSelectorDialog
from pretrain_online_learning import PretrainOnlineLearning


class EEGGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.receiver = LSLStreamReceiver()
        self.viewer = LSLStreamVisualizer(self.receiver)

        instance_id_str = os.environ.get("EEG_GUI_INSTANCE", "1")
        instance_id = int(instance_id_str)

        if instance_id == 1:
            window_title = "Real-time EEG Viewer - Method 1 (IIR)"
        elif instance_id == 2:
            window_title = "Real-time EEG Viewer - Method 2 (IIR+ASR)"
        elif instance_id == 3:
            window_title = "Real-time EEG Viewer - Method 3 (IIR+ASR+ORICA)"
        else:
            method_tag = os.environ.get("IIR_FILTER_METHOD", str(instance_id))
            window_title = f"Real-time EEG Viewer - Instance {instance_id} (method={method_tag})"

        self.setWindowTitle(window_title)

        x_pos = 100 + (instance_id - 1) * 50
        y_pos = 100 + (instance_id - 1) * 50
        self.setGeometry(x_pos, y_pos, 1000, 700)

        print(f"✅ 窗口标题: {window_title}")
        print(f"✅ 实例ID: {instance_id}, 窗口位置: ({x_pos}, {y_pos})")

        self.tabs = QTabWidget()
        self.tab_main = QWidget()
        self.tab_bandpower = QWidget()
        self.tabs.addTab(self.tab_main, "EEG Viewer")

        main_layout = QVBoxLayout()
        self.canvas = FigureCanvas(self.viewer.fig)
        main_layout.addWidget(self.canvas)

        self.cutoff_input1 = QLineEdit("1")
        self.cutoff_input2 = QLineEdit("50")
        cutoff_layout = QHBoxLayout()
        cutoff_layout.addWidget(QLabel("Lower Cutoff:"))
        cutoff_layout.addWidget(self.cutoff_input1)
        cutoff_layout.addWidget(QLabel("Upper Cutoff:"))
        cutoff_layout.addWidget(self.cutoff_input2)
        main_layout.addLayout(cutoff_layout)

        self.start_btn = QPushButton("Start Stream")
        self.start_btn.clicked.connect(self.start_stream)
        main_layout.addWidget(self.start_btn)

        self.update_btn = QPushButton("Update Filter")
        self.update_btn.clicked.connect(self.update_filter_params)
        main_layout.addWidget(self.update_btn)

        self.asr_checkbox = QCheckBox("Enable ASR (pyPREP)")
        main_layout.addWidget(self.asr_checkbox)

        self.att_label = QLabel("attention_level")
        self.att_circle = QLabel()
        self.att_circle.setFixedSize(100, 100)
        self.att_circle.setStyleSheet("border-radius: 50px; background-color: green;")
        main_layout.addWidget(self.att_label)
        main_layout.addWidget(self.att_circle)

        self.attention_ball_window = AttentionBallWindow()
        self.attention_ball_window.show()

        self.channel_select_btn = QPushButton("Select Channels")
        self.channel_select_btn.clicked.connect(self.open_channel_selector)
        main_layout.addWidget(self.channel_select_btn)

        self.ica_btn = QPushButton("Show ICA Components")
        self.ica_btn.clicked.connect(self.show_ica_window)
        main_layout.addWidget(self.ica_btn)
        self.ica_window = None

        self.iclabel_label = QLabel("ICLabel: (暂无)")
        main_layout.addWidget(self.iclabel_label)

        self.topomap_btn = QPushButton("Show Topomap")
        self.topomap_btn.clicked.connect(self.show_topomap_window)
        main_layout.addWidget(self.topomap_btn)
        self.topomap_window = None

        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("input label，like 1 1")
        main_layout.addWidget(self.label_input)

        self.collect_btn = QPushButton("collect data")
        self.collect_btn.clicked.connect(self.collect_labeled_data_from_gui)
        main_layout.addWidget(self.collect_btn)

        self.collect_raw_btn = QPushButton("collect raw data")
        self.collect_raw_btn.clicked.connect(self.collect_raw_labeled_data_from_gui)
        main_layout.addWidget(self.collect_raw_btn)

        online_learning_layout = QHBoxLayout()
        self.online_learn_btn = QPushButton("Start Online Learning")
        self.online_learn_btn.clicked.connect(self.online_learning_from_gui)
        online_learning_layout.addWidget(self.online_learn_btn)

        self.stop_online_learn_btn = QPushButton("Stop Online Learning")
        self.stop_online_learn_btn.clicked.connect(self.stop_online_learning_from_gui)
        self.stop_online_learn_btn.setEnabled(False)
        online_learning_layout.addWidget(self.stop_online_learn_btn)
        main_layout.addLayout(online_learning_layout)

        online_help_label = QLabel(
            "💡 Tips: input label(like 1 or 0), system will use this label for online learning"
        )
        online_help_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        main_layout.addWidget(online_help_label)

        self.online_result_label = QLabel("online learning result: waiting for start...")
        self.online_result_label.setStyleSheet(
            "QLabel { background-color: lightblue; padding: 5px; border-radius: 3px; }"
        )
        main_layout.addWidget(self.online_result_label)

        online_label_layout = QHBoxLayout()
        online_label_layout.addWidget(QLabel("Online Learning Label:"))
        self.online_label_input = QLineEdit()
        self.online_label_input.setPlaceholderText("Input current label (like 1 or 0)")
        online_label_layout.addWidget(self.online_label_input)

        self.clear_label_btn = QPushButton("Clear Label")
        self.clear_label_btn.clicked.connect(self.clear_online_label)
        online_label_layout.addWidget(self.clear_label_btn)
        main_layout.addLayout(online_label_layout)

        self.tab_main.setLayout(main_layout)

        self.bandpower_plot = BandpowerStreamVisualizer()
        bp_layout = QVBoxLayout()
        bp_layout.addWidget(self.bandpower_plot)
        self.tab_bandpower.setLayout(bp_layout)
        self.tabs.addTab(self.tab_bandpower, "Bandpower Waveform")

        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self.tabs)
        self.setLayout(outer_layout)

    def start_stream(self):
        self.update_filter_params()
        self.receiver.start()
        self.viewer.start()
        self.canvas.draw()

        self.bandpower_analyzer = BandpowerAnalyzer(
            receiver=self.receiver,
            gui=self,
            update_interval=1.0,
        )
        self.bandpower_analyzer.start()

        self.attention_analyzer = AttentionAnalyzer(
            receiver=self.receiver,
            gui=self,
            update_interval=1.0,
        )
        self.attention_analyzer.start()

    def update_filter_params(self):
        try:
            val1 = float(self.cutoff_input1.text())
            val2 = float(self.cutoff_input2.text())
            self.receiver.cutoff = (float(val1), float(val2))
            print(f"✅ 已更新滤波参数: cutoff = {self.receiver.cutoff}")
        except ValueError:
            print("❌ Cutoff 值无效，使用默认值 (1, 50)")
            self.receiver.cutoff = (1, 50)

        if hasattr(self, "asr_checkbox"):
            self.receiver.use_asr = self.asr_checkbox.isChecked()
            print(f"{'✅ 启用' if self.receiver.use_asr else '❌ 关闭'} ASR 处理")

    def update_attention_circle(self, score):
        size = int(100 * score)
        self.att_circle.setFixedSize(size, size)
        color = "green" if score > 0.6 else "orange" if score > 0.3 else "red"
        self.att_circle.setStyleSheet(f"border-radius: {size // 2}px; background-color: {color};")
        self.attention_ball_window.update_attention(score)

    def open_channel_selector(self):
        if self.receiver.chan_labels:
            dlg = ChannelSelectorDialog(self, self.receiver)
            dlg.exec_()
        else:
            print("⚠️ 通道标签尚未加载，无法选择通道")

    def show_ica_window(self):
        sources = self.receiver.latest_sources
        if sources is None:
            print("❌ 当前没有 ICA 成分可视化")
            return

        if self.ica_window is None:
            self.ica_window = ICAComponentWindow(ica_sources=sources)
        else:
            self.ica_window.update_sources(sources)

        if hasattr(self.receiver, "latest_eog_indices"):
            self.ica_window.set_eog_indices(self.receiver.latest_eog_indices)

        ic_probs = getattr(self.receiver, "latest_ic_probs", None)
        ic_labels = getattr(self.receiver, "latest_ic_labels", None)
        if ic_labels is not None and ic_probs is not None:
            try:
                lines = []
                for i, (label, probs) in enumerate(zip(ic_labels, ic_probs)):
                    p = float(np.max(probs)) if hasattr(probs, "__len__") else float(probs)
                    lines.append(f"IC{i}: {label} ({p:.2f})")
                self.iclabel_label.setText("ICLabel: " + "; ".join(lines))
            except Exception:
                self.iclabel_label.setText("ICLabel: (解析失败)")
        else:
            self.iclabel_label.setText("ICLabel: (暂无)")

        self.ica_window.show()
        self.ica_window.raise_()

    def show_topomap_window(self):
        if self.topomap_window is None:
            self.topomap_window = TopomapWindow(self.receiver)
        else:
            self.topomap_window.set_receiver(self.receiver)

        self.topomap_window.show()
        self.topomap_window.raise_()

    def collect_labeled_data_from_gui(self):
        threading.Thread(target=self._collect_labeled_data_worker, daemon=True).start()

    def collect_raw_labeled_data_from_gui(self):
        threading.Thread(target=self._collect_raw_labeled_data_worker, daemon=True).start()

    def _collect_labeled_data_worker(self):
        from scipy.signal import welch

        FS = 500
        WINDOW_DURATION = 2
        TOTAL_DURATION = 60
        SAVE_PATH = "./Quick30/labeled_eeg_data_listen_features.npz"
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

        label_text = self.label_input.text().strip()
        if not label_text:
            print("❌ 请输入标签")
            return
        try:
            label = [int(x) for x in label_text.split()]
        except ValueError:
            print("❌ 标签格式错误，应为如 '1 0'")
            return

        print(
            f"🔄 开始收集 {TOTAL_DURATION} 秒数据，分为 "
            f"{TOTAL_DURATION // WINDOW_DURATION} 个 {WINDOW_DURATION} 秒窗口..."
        )
        start_time = time.time()
        collected_windows = []
        collection_start = time.time()

        while time.time() - collection_start < TOTAL_DURATION:
            buffer = self.receiver.get_buffer_data(data_type="processed")
            if buffer is not None and buffer.shape[1] >= FS * WINDOW_DURATION:
                window = buffer[:, -FS * WINDOW_DURATION :]
                collected_windows.append(window)
                time.sleep(WINDOW_DURATION)
            else:
                print("⚠️ 等待足够的数据...")
                time.sleep(0.5)

        if len(collected_windows) < TOTAL_DURATION // WINDOW_DURATION:
            print(
                f"❌ 收集的窗口不足，期望 {TOTAL_DURATION // WINDOW_DURATION} 个，"
                f"实际 {len(collected_windows)} 个"
            )
            return

        print(f"✅ 成功收集 {len(collected_windows)} 个窗口")

        def extract_bandpower_features(data, fs=FS):
            bands = {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 45),
            }
            features = []
            for ch in data:
                f, Pxx = welch(ch, fs=fs, nperseg=fs * 2)
                for band in bands.values():
                    idx = np.logical_and(f >= band[0], f < band[1])
                    features.append(np.sum(Pxx[idx]))
            return np.array(features)

        all_features = []
        for i, window in enumerate(collected_windows):
            features = extract_bandpower_features(window)
            all_features.append(features)
            print(f"窗口 {i + 1}: 特征维度 {features.shape}")

        try:
            old = np.load(SAVE_PATH)
            X_list = list(old["X"])
            y_list = list(old["y"])
        except FileNotFoundError:
            X_list, y_list = [], []

        for features in all_features:
            X_list.append(features)
            y_list.append(label)

        np.savez(SAVE_PATH, X=np.array(X_list), y=np.array(y_list))

        elapsed_time = time.time() - start_time
        print(f"✅ 已保存 {len(all_features)} 个样本，路径：{SAVE_PATH}")
        print(f"⏱️ 收集耗时: {elapsed_time:.1f} 秒")

        print("\n" + "=" * 50)
        print("📋 数据结构展示")
        print("=" * 50)

        feature_df = pd.DataFrame(
            X_list, columns=[f"Feature_{i}" for i in range(len(X_list[0]))]
        )
        print("🔍 特征数据 (X):")
        print(f"形状: {feature_df.shape}")

        label_df = pd.DataFrame(
            y_list, columns=[f"Label_{i}" for i in range(len(y_list[0]))]
        )
        print("\n🏷️ 标签数据 (y):")
        print(f"形状: {label_df.shape}")
        print("所有数据:")
        print(label_df.to_string())

        print("\n📊 数据统计:")
        print(f"总样本数: {len(X_list)}")
        print(f"特征数: {len(X_list[0])}")
        print(f"标签数: {len(y_list[0])}")
        print(f"窗口时长: {WINDOW_DURATION} 秒")
        print(f"总收集时长: {TOTAL_DURATION} 秒")

        print("\n📈 特征统计:")
        print(
            f"特征均值范围: [{np.mean(feature_df, axis=0).min():.6f}, "
            f"{np.mean(feature_df, axis=0).max():.6f}]"
        )
        print(
            f"特征标准差范围: [{np.std(feature_df, axis=0).min():.6f}, "
            f"{np.std(feature_df, axis=0).max():.6f}]"
        )

        unique_labels, counts = np.unique(y_list, return_counts=True)
        print("\n🏷️ 标签分布:")
        for lbl, count in zip(unique_labels, counts):
            print(f"  标签 {lbl}: {count} 个样本 ({count / len(y_list) * 100:.1f}%)")

        print("=" * 50)

    def _collect_raw_labeled_data_worker(self):
        FS = 500
        WINDOW_DURATION = 2
        TOTAL_DURATION = 60
        SAVE_PATH = "./Quick30/labeled_raw_eeg_data_listen_processed.npz"
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

        label_text = self.label_input.text().strip()
        if not label_text:
            print("❌ 请输入标签")
            return
        try:
            label = [int(x) for x in label_text.split()]
        except ValueError:
            print("❌ 标签格式错误，应为如 '1 0'")
            return

        print(
            f"🔄 开始收集原始EEG数据 {TOTAL_DURATION} 秒，分为 "
            f"{TOTAL_DURATION // WINDOW_DURATION} 个 {WINDOW_DURATION} 秒窗口..."
        )
        start_time = time.time()
        collected_windows = []
        collection_start = time.time()

        while time.time() - collection_start < TOTAL_DURATION:
            buffer = self.receiver.get_buffer_data(data_type="processed")
            if buffer is not None and buffer.shape[1] >= FS * WINDOW_DURATION:
                window = buffer[:, -FS * WINDOW_DURATION :]
                collected_windows.append(window)
                time.sleep(WINDOW_DURATION)
            else:
                print("⚠️ 等待足够的数据...")
                time.sleep(0.5)

        if len(collected_windows) < TOTAL_DURATION // WINDOW_DURATION:
            print(
                f"❌ 收集的窗口不足，期望 {TOTAL_DURATION // WINDOW_DURATION} 个，"
                f"实际 {len(collected_windows)} 个"
            )
            return

        print(f"✅ 成功收集 {len(collected_windows)} 个原始数据窗口")

        try:
            old = np.load(SAVE_PATH)
            X_list = list(old["X"])
            y_list = list(old["y"])
            print(f"📂 加载现有数据: {len(X_list)} 个样本")
        except FileNotFoundError:
            X_list, y_list = [], []
            print("📂 创建新的数据文件")

        for i, window in enumerate(collected_windows):
            X_list.append(window)
            y_list.append(label)
            print(f"窗口 {i + 1}: 原始数据形状 {window.shape}")

        np.savez(SAVE_PATH, X=np.array(X_list), y=np.array(y_list))

        elapsed_time = time.time() - start_time
        print(f"✅ 已保存 {len(collected_windows)} 个原始数据样本，路径：{SAVE_PATH}")
        print(f"⏱️ 收集耗时: {elapsed_time:.1f} 秒")

        print("\n" + "=" * 50)
        print("📋 原始数据结构展示")
        print("=" * 50)

        print("\n📊 数据统计:")
        print(f"总样本数: {len(X_list)}")
        print(f"每个样本形状: {X_list[0].shape if X_list else 'N/A'}")
        print(f"标签数: {len(y_list[0]) if y_list else 0}")
        print(f"窗口时长: {WINDOW_DURATION} 秒")
        print(f"总收集时长: {TOTAL_DURATION} 秒")
        print(f"采样率: {FS} Hz")
        print(f"通道数: {X_list[0].shape[0] if X_list else 'N/A'}")
        print(f"每个窗口数据点数: {X_list[0].shape[1] if X_list else 'N/A'}")

        if X_list:
            all_data = np.array(X_list)
            print("\n📈 原始数据统计:")
            print(f"数据均值: {np.mean(all_data):.6f}")
            print(f"数据标准差: {np.std(all_data):.6f}")
            print(f"数据最小值: {np.min(all_data):.6f}")
            print(f"数据最大值: {np.max(all_data):.6f}")

        if y_list:
            unique_labels, counts = np.unique(y_list, return_counts=True)
            print("\n🏷️ 标签分布:")
            for lbl, count in zip(unique_labels, counts):
                print(f"  标签 {lbl}: {count} 个样本 ({count / len(y_list) * 100:.1f}%)")

        print("=" * 50)

    def online_learning_from_gui(self):
        if not hasattr(self, "online_learning_manager"):
            self.online_learning_manager = PretrainOnlineLearning(
                receiver=self.receiver,
                gui=self,
            )

        self.online_learning_manager.start_online_learning()
        self.online_learn_btn.setEnabled(False)
        self.stop_online_learn_btn.setEnabled(True)
        print("✅ 在线学习已启动")
        self.online_result_label.setText("在线学习运行中...")
        self.online_result_label.setStyleSheet(
            "QLabel { background-color: lightblue; padding: 5px; border-radius: 3px; }"
        )

    def stop_online_learning_from_gui(self):
        if hasattr(self, "online_learning_manager") and self.online_learning_manager:
            self.online_learning_manager.stop_online_learning()
            self.online_learn_btn.setEnabled(True)
            self.stop_online_learn_btn.setEnabled(False)
            self.online_result_label.setText("Online learning stopped, results saved")
            self.online_result_label.setStyleSheet(
                "QLabel { background-color: lightgreen; padding: 5px; border-radius: 3px; }"
            )
            print("🛑 Online learning stopped")
        else:
            print("⚠️ No online learning is running")

    def clear_online_label(self):
        self.online_label_input.clear()
        print("✅ Label input cleared")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    gui = EEGGUI()
    gui.show()
    sys.exit(app.exec_())
