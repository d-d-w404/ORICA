import threading
import time

from PyQt5.QtWidgets import QApplication


from PyQt5.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QGroupBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


from stream_receiver import LSLStreamReceiver, ChannelManager
from bandpower_analysis import analyze_bandpower
from attention_estimator import RealTimeAttentionEstimator
from stream_receiver import LSLStreamReceiver
from viewer import LSLStreamVisualizer
from regression_model import RealTimeRegressor
from attention_ball import AttentionBallWindow
from bandpower_plot import BandpowerStreamVisualizer
from filter_utils import EEGSignalProcessor
from ica_component_window import ICAComponentWindow
from orica_processor import ORICAProcessor

from channel_selector import ChannelSelectorDialog

class EEGGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time EEG Viewer")
        self.setGeometry(100, 100, 1000, 700)

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
        self.att_label = QLabel("🎯 注意力水平")
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

        # ========== 外层 Layout ==========
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self.tabs)
        self.setLayout(outer_layout)




        # 添加这个函数到类中或文件外部
        def periodic_print(receiver, interval_sec=1):
            while True:
                time.sleep(interval_sec)
                receiver.print_latest_channel_values()

        # 然后在 EEGGUI.__init__ 末尾添加：
        threading.Thread(target=periodic_print, args=(self.receiver,), daemon=True).start()

    def start_stream(self):
        self.update_filter_params()
        self.viewer.start()
        self.canvas.draw()

        #self.receiver.register_analysis_callback(analyze_bandpower)
        #self.receiver.register_analysis_callback(self.regressor.callback)

        self.attention_estimator = RealTimeAttentionEstimator(gui=self,receiver=self.receiver)
        self.receiver.register_analysis_callback(self.attention_estimator.callback)


        self.receiver.register_analysis_callback(lambda **kwargs: analyze_bandpower(gui=self, **kwargs))

        self.receiver.register_analysis_callback(EEGSignalProcessor.heavy_analysis)
        #这里注册的回调函数必须先实例化，然后通过实例化引用这个方法。但是我已经把这个函数作为static状态了，后续可以研究一下

    def update_filter_params(self):
        try:
            # 获取输入的上下截止频率
            val1 = float(self.cutoff_input1.text())
            val2 = float(self.cutoff_input2.text())

            # 设置到 receiver 中
            self.receiver.cutoff = (val1, val2)
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


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    gui = EEGGUI()
    gui.show()
    sys.exit(app.exec_())
