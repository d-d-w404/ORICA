import threading
import time
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication


from PyQt5.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QGroupBox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


from stream_receiver import LSLStreamReceiver, ChannelManager
from bandpower_analysis import analyze_bandpower
from attention_estimator import RealTimeAttentionEstimator
from viewer import LSLStreamVisualizer
from regression_model import RealTimeRegressor
from attention_ball import AttentionBallWindow
from bandpower_plot import BandpowerStreamVisualizer
from filter_utils import EEGSignalProcessor
from ica_component_window import ICAComponentWindow
from orica_processor import ORICAProcessor
from topomap_visualizer import TopomapWindow

# ✅ 新增：导入新的分析器
from bandpower_analyzer import BandpowerAnalyzer
from attention_analyzer import AttentionAnalyzer

from channel_selector import ChannelSelectorDialog
from pretrain_online_learning import PretrainOnlineLearning

import mne
class EEGGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time EEG Viewer")
        self.setGeometry(100, 100, 1000, 700)

        # ========== Function Parts ==========
        self.receiver = LSLStreamReceiver()
        self.viewer = LSLStreamVisualizer(self.receiver)
        self.regressor = RealTimeRegressor(gui=self)


        # ========== 创建 Tab 界面 ==========
        self.tabs = QTabWidget()
        self.tab_main = QWidget()
        self.tab_bandpower = QWidget()
        self.tabs.addTab(self.tab_main, "EEG Viewer")
        self.tabs.addTab(self.tab_bandpower, "Bandpower Plot")

        # ========== 主界面 Tab ==========
        main_layout = QVBoxLayout()
        self.canvas = FigureCanvas(self.viewer.fig)
        main_layout.addWidget(self.canvas)

        self.cutoff_input1 = QLineEdit("0.5")
        self.cutoff_input2 = QLineEdit("45")
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

        # attention display
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

        self.tab_main.setLayout(main_layout)

        # ========== Bandpower Tab ==========
        # 新增 bandpower 可视化器
        self.bandpower_plot = BandpowerStreamVisualizer()
        bp_layout = QVBoxLayout()
        bp_layout.addWidget(self.bandpower_plot)
        self.tab_bandpower.setLayout(bp_layout)
        self.tabs.addTab(self.tab_bandpower, "Bandpower Waveform")




        # ========== ICA window ==========
        self.ica_btn = QPushButton("Show ICA Components")
        self.ica_btn.clicked.connect(self.show_ica_window)
        main_layout.addWidget(self.ica_btn)

        self.ica_window = None  # 初始化为空

        # ========== topomap ==========
        self.topomap_btn = QPushButton("Show Topomap")
        self.topomap_btn.clicked.connect(self.show_topomap_window)
        main_layout.addWidget(self.topomap_btn)

        self.topomap_window = None  # 初始化为空


        # ... 其他控件 ...
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("input label，like 1 1")
        main_layout.addWidget(self.label_input)

        self.collect_btn = QPushButton("collect data")
        self.collect_btn.clicked.connect(self.collect_labeled_data_from_gui)
        main_layout.addWidget(self.collect_btn)
        
        self.collect_raw_btn = QPushButton("collect raw data")
        self.collect_raw_btn.clicked.connect(self.collect_raw_labeled_data_from_gui)
        main_layout.addWidget(self.collect_raw_btn)
        
        # 在线学习按钮组
        online_learning_layout = QHBoxLayout()
        
        # 开始在线学习按钮
        self.online_learn_btn = QPushButton("Start Online Learning")
        self.online_learn_btn.clicked.connect(self.online_learning_from_gui)
        online_learning_layout.addWidget(self.online_learn_btn)
        
        # 停止在线学习按钮
        self.stop_online_learn_btn = QPushButton("Stop Online Learning")
        self.stop_online_learn_btn.clicked.connect(self.stop_online_learning_from_gui)
        self.stop_online_learn_btn.setEnabled(False)  # 初始状态禁用
        online_learning_layout.addWidget(self.stop_online_learn_btn)
        
        main_layout.addLayout(online_learning_layout)
        
        # 在线学习提示
        online_help_label = QLabel("💡 Tips: input label(like 1 or 0), system will use this label for online learning")
        online_help_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        main_layout.addWidget(online_help_label)
        
        # 在线学习结果显示
        self.online_result_label = QLabel("online learning result: waiting for start...")
        self.online_result_label.setStyleSheet("QLabel { background-color: lightblue; padding: 5px; border-radius: 3px; }")
        main_layout.addWidget(self.online_result_label)
        
        # 在线学习标签输入
        online_label_layout = QHBoxLayout()
        online_label_layout.addWidget(QLabel("Online Learning Label:"))
        self.online_label_input = QLineEdit()
        self.online_label_input.setPlaceholderText("Input current label (like 1 or 0)")
        online_label_layout.addWidget(self.online_label_input)
        
        # 清除标签按钮
        self.clear_label_btn = QPushButton("Clear Label")
        self.clear_label_btn.clicked.connect(self.clear_online_label)
        online_label_layout.addWidget(self.clear_label_btn)
        
        main_layout.addLayout(online_label_layout)


        # ========== 外层 Layout ==========
        outer_layout = QVBoxLayout() 
        outer_layout.addWidget(self.tabs)
        self.setLayout(outer_layout)



        #通过线程展示通道的电压，rMs值
        # 添加这个函数到类中或文件外部
        def periodic_print(receiver, interval_sec=1):
            while True:
                time.sleep(interval_sec)
                receiver.print_latest_channel_values()
        # 延迟启动，避免在初始化时就开始运行
        # threading.Thread(target=periodic_print, args=(self.receiver,), daemon=True).start()

    def start_stream(self):
        self.update_filter_params()
        self.viewer.start()
        self.canvas.draw()

        # ✅ 使用新的分析器类，而不是callback机制
        # 创建并启动频段功率分析器
        self.bandpower_analyzer = BandpowerAnalyzer(
            receiver=self.receiver,
            gui=self,
            update_interval=1.0
        )
        self.bandpower_analyzer.start()
        
        # 创建并启动注意力分析器
        self.attention_analyzer = AttentionAnalyzer(
            receiver=self.receiver,
            gui=self,
            update_interval=1.0
        )
        self.attention_analyzer.start()
        
        # 保留原有的回归模型（如果需要的话）
        # self.regressor = RealTimeRegressor(gui=self)
        # self.receiver.register_analysis_callback(self.regressor.callback)

    def update_filter_params(self):
        try:
            # 获取输入的上下截止频率
            val1 = float(self.cutoff_input1.text())
            val2 = float(self.cutoff_input2.text())

            # 设置到 receiver 中
            self.receiver.cutoff = (float(val1), float(val2))
            print(f"✅ 已更新滤波参数: cutoff = {self.receiver.cutoff}")

        except ValueError:
            print("❌ Cutoff 值无效，使用默认值 (0.5, 45)")

        # 设置是否启用 ASR（从复选框读取）
        if hasattr(self, 'asr_checkbox'):
            self.receiver.use_asr = self.asr_checkbox.isChecked()
            print(f"{'✅ 启用' if self.receiver.use_asr else '❌ 关闭'} ASR 处理")

    def update_prediction_display(self, pred):
        self.pred_label.setText(f"🎯 预测情绪强度: {pred:.3f}")

    def update_model_from_gui(self):
        text = self.feedback_input.text()
        try:
            y = float(text)
            if 0 <= y <= 1:
                self.regressor.update_with_feedback(y)  # ✅ GUI 不再处理 scaler 或模型
            else:
                print("⚠️ 输入应在 [0, 1] 之间")
        except Exception as e:
            print("❌ 非法输入", e)

        print(text)

    def enable_initial_rating_ui(self, enable=True):
        self.init_label.setVisible(enable)
        self.init_input.setVisible(enable)
        self.init_button.setVisible(enable)

    def submit_initial_rating(self):
        text = self.init_input.text()
        try:
            y = float(text)
            if 0 <= y <= 1:
                self.regressor.init_model_with_label(y)
                print("✅ 初始评分已用于模型初始化")
                self.enable_initial_rating_ui(False)
            else:
                print("⚠️ 输入应在 [0, 1] 之间")
        except Exception as e:
            print("❌ 初始评分输入无效", e)

    def update_attention_circle(self, score):
        #size = int(30 + 70 * score)
        size = int(100*score)
        self.att_circle.setFixedSize(size, size)
        color = "green" if score > 0.6 else "orange" if score > 0.3 else "red"
        self.att_circle.setStyleSheet(f"border-radius: {size // 2}px; background-color: {color};")

        # 👇 同步更新到漂移小球窗口
        self.attention_ball_window.update_attention(score)

    def open_channel_selector(self):
        if self.receiver.chan_labels:
            dlg = ChannelSelectorDialog(self, self.receiver)
            dlg.exec_()
        else:
            print("⚠️ 通道标签尚未加载，无法选择通道")

    def show_ica_window(self):
        sources = self.receiver.latest_sources  # 从 stream_receiver 获取 ICA 分量
        if sources is None:
            print("❌ 当前没有 ICA 成分可视化")
            return

        if self.ica_window is None:
            self.ica_window = ICAComponentWindow(ica_sources=sources)
        else:
            self.ica_window.update_sources(sources)

        # ✅ 设置高亮的 EOG 成分（红色显示）
        if hasattr(self.receiver, 'latest_eog_indices'):
            self.ica_window.set_eog_indices(self.receiver.latest_eog_indices)

        self.ica_window.show()
        self.ica_window.raise_()

    def show_topomap_window(self):
        """显示topomap窗口，类似show_ica_window的实现"""
        if self.topomap_window is None:
            self.topomap_window = TopomapWindow(self.receiver)
        else:
            # 如果窗口已存在，设置receiver（会触发更新）
            self.topomap_window.set_receiver(self.receiver)

        self.topomap_window.show()
        self.topomap_window.raise_()

    def collect_labeled_data_from_gui(self):
        # 启动新线程采集，避免阻塞主线程
        threading.Thread(target=self._collect_labeled_data_worker, daemon=True).start()
    
    def collect_raw_labeled_data_from_gui(self):
        # 启动新线程采集原始数据，避免阻塞主线程
        threading.Thread(target=self._collect_raw_labeled_data_worker, daemon=True).start()
    

    def _collect_labeled_data_worker(self):
        import numpy as np
        from scipy.signal import welch
        import time, os, pandas as pd

        FS = 500
        WINDOW_DURATION = 2  # 每个窗口2秒
        TOTAL_DURATION = 60  # 总收集时长20秒
        SAVE_PATH = './Quick30/labeled_eeg_data_listen_features.npz'
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

        label_text = self.label_input.text().strip()
        if not label_text:
            print("❌ 请输入标签")
            return
        try:
            label = [int(x) for x in label_text.split()]
        except:
            print("❌ 标签格式错误，应为如 '1 0'")
            return

        print(f"🔄 开始收集 {TOTAL_DURATION} 秒数据，分为 {TOTAL_DURATION//WINDOW_DURATION} 个 {WINDOW_DURATION} 秒窗口...")
        start_time = time.time()
        
        # 收集指定时长的数据
        collected_windows = []
        collection_start = time.time()
        
        while time.time() - collection_start < TOTAL_DURATION:
            buffer = self.receiver.get_buffer_data(data_type='processed')
            if buffer is not None and buffer.shape[1] >= FS * WINDOW_DURATION:
                # 获取最新的2秒数据作为一个窗口
                window = buffer[:, -FS*WINDOW_DURATION:]
                collected_windows.append(window)
                time.sleep(WINDOW_DURATION)  # 等待2秒再收集下一个窗口
            else:
                print("⚠️ 等待足够的数据...")
                time.sleep(0.5)
        
        if len(collected_windows) < TOTAL_DURATION // WINDOW_DURATION:
            print(f"❌ 收集的窗口不足，期望 {TOTAL_DURATION//WINDOW_DURATION} 个，实际 {len(collected_windows)} 个")
            return
            
        print(f"✅ 成功收集 {len(collected_windows)} 个窗口")

        def extract_bandpower_features(data, fs=FS):
            bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
            features = []
            for ch in data:
                f, Pxx = welch(ch, fs=fs, nperseg=fs*2)
                for band in bands.values():
                    idx = np.logical_and(f >= band[0], f < band[1])
                    features.append(np.sum(Pxx[idx]))
            return np.array(features)

        # 为每个窗口提取特征
        all_features = []
        for i, window in enumerate(collected_windows):
            features = extract_bandpower_features(window)
            all_features.append(features)
            print(f"窗口 {i+1}: 特征维度 {features.shape}")

        try:
            old = np.load(SAVE_PATH)
            X_list = list(old['X'])
            y_list = list(old['y'])
        except FileNotFoundError:
            X_list, y_list = [], []

        # 添加所有窗口的特征和标签
        for features in all_features:
            X_list.append(features)
            y_list.append(label)  # 每个窗口使用相同的标签

        np.savez(SAVE_PATH, X=np.array(X_list), y=np.array(y_list))

        elapsed_time = time.time() - start_time
        print(f"✅ 已保存 {len(all_features)} 个样本，路径：{SAVE_PATH}")
        print(f"⏱️ 收集耗时: {elapsed_time:.1f} 秒")
        
        # 使用pandas展示数据结构
        print("\n" + "="*50)
        print("📋 数据结构展示")
        print("="*50)
        
        # 创建特征DataFrame
        feature_df = pd.DataFrame(X_list, columns=[f'Feature_{i}' for i in range(len(X_list[0]))])
        print("🔍 特征数据 (X):")
        print(f"形状: {feature_df.shape}")
        print(f"所有数据:")
        #print(feature_df.to_string())
        
        # 创建标签DataFrame
        label_df = pd.DataFrame(y_list, columns=[f'Label_{i}' for i in range(len(y_list[0]))])
        print(f"\n🏷️ 标签数据 (y):")
        print(f"形状: {label_df.shape}")
        print(f"所有数据:")
        print(label_df.to_string())
        
        # 简要统计
        print(f"\n📊 数据统计:")
        print(f"总样本数: {len(X_list)}")
        print(f"特征数: {len(X_list[0])}")
        print(f"标签数: {len(y_list[0])}")
        print(f"窗口时长: {WINDOW_DURATION} 秒")
        print(f"总收集时长: {TOTAL_DURATION} 秒")
        
        # 显示特征统计信息
        print(f"\n📈 特征统计:")
        print(f"特征均值范围: [{np.mean(feature_df, axis=0).min():.6f}, {np.mean(feature_df, axis=0).max():.6f}]")
        print(f"特征标准差范围: [{np.std(feature_df, axis=0).min():.6f}, {np.std(feature_df, axis=0).max():.6f}]")
        
        # 显示标签统计
        unique_labels, counts = np.unique(y_list, return_counts=True)
        print(f"\n🏷️ 标签分布:")
        for label, count in zip(unique_labels, counts):
            print(f"  标签 {label}: {count} 个样本 ({count/len(y_list)*100:.1f}%)")
        
        print("="*50)

    def _collect_raw_labeled_data_worker(self):
        """收集原始EEG数据，不提取特征"""
        import numpy as np
        import time, os, pandas as pd

        FS = 500
        WINDOW_DURATION = 2  # 每个窗口2秒
        TOTAL_DURATION = 60  # 总收集时长60秒
        SAVE_PATH = './Quick30/labeled_raw_eeg_data_listen_processed.npz'
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

        label_text = self.label_input.text().strip()
        if not label_text:
            print("❌ 请输入标签")
            return
        try:
            label = [int(x) for x in label_text.split()]
        except:
            print("❌ 标签格式错误，应为如 '1 0'")
            return

        print(f"🔄 开始收集原始EEG数据 {TOTAL_DURATION} 秒，分为 {TOTAL_DURATION//WINDOW_DURATION} 个 {WINDOW_DURATION} 秒窗口...")
        start_time = time.time()
        
        # 收集指定时长的原始数据
        collected_windows = []
        collection_start = time.time()
        
        while time.time() - collection_start < TOTAL_DURATION:
            # 获取原始数据（不经过特征提取）
            buffer = self.receiver.get_buffer_data(data_type='processed')
            if buffer is not None and buffer.shape[1] >= FS * WINDOW_DURATION:
                # 获取最新的2秒原始数据作为一个窗口
                window = buffer[:, -FS*WINDOW_DURATION:]
                collected_windows.append(window)
                time.sleep(WINDOW_DURATION)  # 等待2秒再收集下一个窗口
            else:
                print("⚠️ 等待足够的数据...")
                time.sleep(0.5)
        
        if len(collected_windows) < TOTAL_DURATION // WINDOW_DURATION:
            print(f"❌ 收集的窗口不足，期望 {TOTAL_DURATION//WINDOW_DURATION} 个，实际 {len(collected_windows)} 个")
            return
            
        print(f"✅ 成功收集 {len(collected_windows)} 个原始数据窗口")

        try:
            # 尝试加载现有数据
            old = np.load(SAVE_PATH)
            X_list = list(old['X'])
            y_list = list(old['y'])
            print(f"📂 加载现有数据: {len(X_list)} 个样本")
        except FileNotFoundError:
            X_list, y_list = [], []
            print("📂 创建新的数据文件")

        # 添加所有窗口的原始数据和标签
        for i, window in enumerate(collected_windows):
            X_list.append(window)  # 直接保存原始EEG数据
            y_list.append(label)   # 每个窗口使用相同的标签
            print(f"窗口 {i+1}: 原始数据形状 {window.shape}")

        # 保存原始数据
        np.savez(SAVE_PATH, X=np.array(X_list), y=np.array(y_list))

        elapsed_time = time.time() - start_time
        print(f"✅ 已保存 {len(collected_windows)} 个原始数据样本，路径：{SAVE_PATH}")
        print(f"⏱️ 收集耗时: {elapsed_time:.1f} 秒")
        
        # 使用pandas展示数据结构
        print("\n" + "="*50)
        print("📋 原始数据结构展示")
        print("="*50)
        
        # 显示数据统计信息
        print(f"\n📊 数据统计:")
        print(f"总样本数: {len(X_list)}")
        print(f"每个样本形状: {X_list[0].shape if X_list else 'N/A'}")
        print(f"标签数: {len(y_list[0]) if y_list else 0}")
        print(f"窗口时长: {WINDOW_DURATION} 秒")
        print(f"总收集时长: {TOTAL_DURATION} 秒")
        print(f"采样率: {FS} Hz")
        print(f"通道数: {X_list[0].shape[0] if X_list else 'N/A'}")
        print(f"每个窗口数据点数: {X_list[0].shape[1] if X_list else 'N/A'}")
        
        # 显示原始数据统计信息
        if X_list:
            all_data = np.array(X_list)
            print(f"\n📈 原始数据统计:")
            print(f"数据均值范围: [{np.mean(all_data):.6f}, {np.mean(all_data):.6f}]")
            print(f"数据标准差: {np.std(all_data):.6f}")
            print(f"数据最小值: {np.min(all_data):.6f}")
            print(f"数据最大值: {np.max(all_data):.6f}")
        
        # 显示标签统计
        if y_list:
            unique_labels, counts = np.unique(y_list, return_counts=True)
            print(f"\n🏷️ 标签分布:")
            for label, count in zip(unique_labels, counts):
                print(f"  标签 {label}: {count} 个样本 ({count/len(y_list)*100:.1f}%)")
        
        print("="*50)

    def online_learning_from_gui(self):
        """从GUI启动在线学习"""
        # 创建在线学习管理器
        if not hasattr(self, 'online_learning_manager'):
            self.online_learning_manager = PretrainOnlineLearning(
                receiver=self.receiver,
                gui=self
            )
        
        # 启动在线学习
        self.online_learning_manager.start_online_learning()
        
        # 更新按钮状态
        self.online_learn_btn.setEnabled(False)
        self.stop_online_learn_btn.setEnabled(True)
        
        print("✅ 在线学习已启动")
        
        # 更新结果显示
        self.online_result_label.setText("在线学习运行中...")
        self.online_result_label.setStyleSheet("QLabel { background-color: lightblue; padding: 5px; border-radius: 3px; }")

    def stop_online_learning_from_gui(self):
        """从GUI停止在线学习"""
        if hasattr(self, 'online_learning_manager') and self.online_learning_manager:
            # 停止在线学习
            self.online_learning_manager.stop_online_learning()
            
            # 更新按钮状态
            self.online_learn_btn.setEnabled(True)
            self.stop_online_learn_btn.setEnabled(False)
            
            # 更新结果显示
            self.online_result_label.setText("Online learning stopped, results saved")
            self.online_result_label.setStyleSheet("QLabel { background-color: lightgreen; padding: 5px; border-radius: 3px; }")
            
            print("🛑 Online learning stopped")
        else:
            print("⚠️ No online learning is running")

    def clear_online_label(self):
        """清除在线学习标签输入"""
        self.online_label_input.clear()
        print("✅ Label input cleared")

    def _get_current_attention_level(self):
        """获取当前注意力水平"""
        try:
            # 这里可以根据实际的注意力分析器获取注意力水平
            # 暂时返回一个随机值作为示例
            import random
            return random.uniform(0.3, 0.8)
        except:
            return 0.5


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    gui = EEGGUI()
    gui.show()
    sys.exit(app.exec_())
