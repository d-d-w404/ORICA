# vis_stream_orica.py
# Re-implementation of vis_stream_ORICA in Python using pylsl and matplotlib

import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
from matplotlib.animation import FuncAnimation
import time
from scipy.signal import butter, lfilter
from scipy.signal import butter, filtfilt
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QComboBox, QLabel, QLineEdit, QHBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys

class LSLStreamReceiver:
    def __init__(self, stream_type='EEG', time_range=5):
        self.stream_type = stream_type
        self.time_range = time_range
        self.inlet = None
        self.srate = None
        self.nbchan = None
        self.buffer = None
        self.chan_labels = []
        self.channel_range = []
        self.cutoff = (0.5, 45)

        #ASR
        self.asr = SimpleASR(sfreq=self.srate, cutoff=5.0)
        self.asr_calibrated = False
        self.use_asr = False  # 默认不启用 ASR
        self.asr_calibration_buffer = None

    def find_and_open_stream(self):
        print(f"Searching for LSL stream with type = '{self.stream_type}'...")
        streams = resolve_byprop('type', self.stream_type, timeout=5)

        if not streams:
            raise RuntimeError(f"No LSL stream with type '{self.stream_type}' found.")

        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()

        print("=== StreamInfo XML description ===")
        print(self.inlet.info().as_xml())

        self.srate = int(info.nominal_srate())
        self.nbchan = info.channel_count()

        chs = info.desc().child('channels').child('channel')
        all_labels = []
        for _ in range(self.nbchan):
            label = chs.child_value('label')
            all_labels.append(label if label else f"Ch {_+1}")
            chs = chs.next_sibling()

        exclude_keywords = ['TRIGGER', 'ACC', 'ExG', 'Packet', 'A2']
        for i, label in enumerate(all_labels):
            if not any(keyword in label for keyword in exclude_keywords):
                self.chan_labels.append(label)
                self.channel_range.append(i)

        self.nbchan = len(self.channel_range)
        self.buffer = np.zeros((info.channel_count(), self.srate * self.time_range))

        print(f"Stream opened: {info.channel_count()} channels at {self.srate} Hz")
        print(f"Using {self.nbchan} EEG channels: {self.chan_labels}")

    def pull_and_update_buffer(self):
        samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
        if timestamps:
            chunk = np.array(samples).T  # shape: (channels, samples)

            # Step 1: Bandpass or highpass filter
            chunk = eeg_filter(chunk, self.srate, cutoff=self.cutoff)

            self.last_unclean_chunk = chunk.copy()  # 保存未ASR前的滤波数据

            #Step 2: ASR - calibration or cleaning
            if self.use_asr:
                if not self.asr_calibrated:
                    # 累加到校准缓存
                    if self.asr_calibration_buffer is None:
                        self.asr_calibration_buffer = chunk.copy()
                    else:
                        self.asr_calibration_buffer = np.concatenate((self.asr_calibration_buffer, chunk), axis=1)

                    if self.asr_calibration_buffer.shape[1] >= self.srate * 20:  # 满足 2 秒
                        self.asr.calibrate(self.asr_calibration_buffer)
                        self.asr_calibrated = True
                        print("✅ ASR calibrated with", self.asr_calibration_buffer.shape[1], "samples.")
                        self.asr_calibration_buffer = None  # 清空缓存
                else:
                    chunk_before_clean = chunk.copy()
                    chunk = self.asr.clean(chunk)
                    diff = chunk - chunk_before_clean

                    for i, ch in enumerate(self.channel_range):
                        label = self.chan_labels[i]
                        last_diff = diff[ch, -1]
                        rms_diff = np.sqrt(np.mean(diff[ch] ** 2))
                        print(f"{label}: Δlast = {last_diff:.2f} µV, ΔRMS = {rms_diff:.2f} µV")

            # chunk=self.apply_pyprep_asr(chunk)


            # Step 3: Update ring buffer
            num_new = chunk.shape[1]
            self.buffer = np.roll(self.buffer, -num_new, axis=1)
            self.buffer[:, -num_new:] = chunk

    def print_latest_channel_values(self):
        print("--- EEG Channel Values (last column) ---")
        for i, ch in enumerate(self.channel_range):
            label = self.chan_labels[i]
            value = self.buffer[ch, -1]
            rms = np.sqrt(np.mean(self.buffer[ch]**2))
            print(f"{label}: {value:.2f} (RMS: {rms:.2f})")

    # def print_latest_channel_values(self):
    #     print("--- EEG Channel Values (last sample, RMS) ---")
    #     for i, ch in enumerate(self.channel_range):
    #         label = self.chan_labels[i]
    #         # 已清洗数据（当前 buffer 中）
    #         clean_val = self.buffer[ch, -1]
    #         clean_rms = np.sqrt(np.mean(self.buffer[ch] ** 2))
    #
    #         # 原始数据（保存的 last_unclean_chunk）
    #         if hasattr(self, 'last_unclean_chunk') and self.last_unclean_chunk is not None:
    #             raw_val = self.last_unclean_chunk[ch, -1]
    #             raw_rms = np.sqrt(np.mean(self.last_unclean_chunk[ch] ** 2))
    #         else:
    #             raw_val, raw_rms = np.nan, np.nan
    #
    #         if self.use_asr:
    #             print(
    #                 f"{label}: Clean = {clean_val:.2f} (RMS: {clean_rms:.2f}) | Raw = {raw_val:.2f} (RMS: {raw_rms:.2f}) ✅ ASR")
    #         else:
    #             print(f"{label}: Raw  = {clean_val:.2f} (RMS: {clean_rms:.2f}) ❌ ASR Disabled")

    def apply_pyprep_asr(self, chunk):
        if not self.asr_calibrated:
            # 累加到校准缓存
            if self.asr_calibration_buffer is None:
                self.asr_calibration_buffer = chunk.copy()
            else:
                self.asr_calibration_buffer = np.concatenate((self.asr_calibration_buffer, chunk), axis=1)

            if self.asr_calibration_buffer.shape[1] >= self.srate * 10:  # 满足 10 秒
                try:
                    import mne
                    from pyprep.prep_pipeline import PrepPipeline

                    # 创建临时 RawArray 用于 pyprep 处理
                    info = mne.create_info(
                        ch_names=[self.chan_labels[i] for i in range(len(self.channel_range))],
                        sfreq=self.srate,
                        ch_types=["eeg"] * len(self.channel_range)
                    )
                    raw = mne.io.RawArray(self.asr_calibration_buffer[self.channel_range, :], info)
                    raw.set_montage("standard_1020")  # ✅ 添加标准电极位置，用于PrepPipeline

                    prep_params = {
                        "ref_chs": raw.ch_names,
                        "reref_chs": raw.ch_names,
                        "line_freqs": [50]
                    }

                    prep = PrepPipeline(raw, prep_params, montage="standard_1020")  # ✅ 加上必需的 montage 参数
                    prep.fit()

                    self.prep_reference = prep  # 保存处理器实例
                    self.asr_calibrated = True
                    print("✅ pyPREP calibrated with", self.asr_calibration_buffer.shape[1], "samples.")
                    self.asr_calibration_buffer = None
                except Exception as e:
                    print("❌ pyPREP calibration failed:", e)
        else:
            try:
                import mne
                from pyprep.prep_pipeline import PrepPipeline

                info = mne.create_info(
                    ch_names=[self.chan_labels[i] for i in range(len(self.channel_range))],
                    sfreq=self.srate,
                    ch_types=["eeg"] * len(self.channel_range)
                )
                raw_chunk = mne.io.RawArray(chunk[self.channel_range, :], info)
                raw_chunk.set_montage("standard_1020")  # ✅ 为 chunk 设置 montage

                prep = PrepPipeline(raw_chunk, {
                    "ref_chs": raw_chunk.ch_names,
                    "reref_chs": raw_chunk.ch_names,
                    "line_freqs": [50]
                }, montage="standard_1020")
                prep.fit()

                clean_data = prep.raw.get_data()

                diff = clean_data - chunk[self.channel_range, :]
                for i, ch in enumerate(self.channel_range):
                    label = self.chan_labels[i]
                    last_diff = diff[i, -1]
                    rms_diff = np.sqrt(np.mean(diff[i] ** 2))
                    print(f"{label}: Δlast = {last_diff:.2f} µV, ΔRMS = {rms_diff:.2f} µV")

                chunk[self.channel_range, :] = clean_data
            except Exception as e:
                print("❌ pyPREP ASR cleaning failed:", e)

        return chunk


class LSLStreamVisualizer:
    def __init__(self, receiver: LSLStreamReceiver,
                 data_scale=150,
                 sampling_rate=100,
                 refresh_rate=10,
                 reref=False):

        self.receiver = receiver
        self.data_scale = data_scale
        self.sampling_rate = sampling_rate
        self.refresh_rate = refresh_rate
        self.reref = reref

        self.fig, self.ax = plt.subplots()
        self.lines = []
        self.last_print_time = time.time()

    def update_plot(self, frame):
        self.receiver.pull_and_update_buffer()

        plotdata = self.receiver.buffer[self.receiver.channel_range, ::int(self.receiver.srate/self.sampling_rate)]
        plotdata = plotdata - np.mean(plotdata, axis=1, keepdims=True)
        plotoffsets = np.arange(len(self.receiver.channel_range))[:, None] * self.data_scale
        plotdata += plotoffsets

        self.ax.clear()
        self.ax.set_title(f"LSL Stream Type: {self.receiver.stream_type}")
        self.ax.set_xlabel("Time (samples)")
        self.ax.set_ylabel("Channels")
        self.ax.set_yticks(plotoffsets[:, 0])
        ylabels = [self.receiver.chan_labels[i] for i in range(len(self.receiver.channel_range))]
        self.ax.set_yticklabels(ylabels)
        self.ax.set_ylim(-self.data_scale, plotoffsets[-1][0] + self.data_scale)
        self.ax.plot(plotdata.T, linewidth=0.5)

        current_time = time.time()
        if current_time - self.last_print_time >= 5.0:
            self.receiver.print_latest_channel_values()
            self.last_print_time = current_time

    def start(self):
        self.receiver.find_and_open_stream()
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000/self.refresh_rate)






class SimpleASR:
    def __init__(self, sfreq, cutoff=5.0):
        self.sfreq = sfreq
        self.cutoff = cutoff
        self.calibrated = False
        self.ref_cov = None

    def calibrate(self, data):  # data: (channels, samples)
        self.ref_cov = np.cov(data)
        self.calibrated = True

    def clean(self, data):
        if not self.calibrated:
            raise RuntimeError("ASR not calibrated.")

        # 当前协方差矩阵
        cov = np.cov(data)

        # EVD of current cov
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-10, None)
        inv_sqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        whitened = inv_sqrt_cov @ data

        # EVD of reference cov
        ref_eigvals, ref_eigvecs = np.linalg.eigh(self.ref_cov)
        ref_eigvals = np.clip(ref_eigvals, 1e-10, None)
        sqrt_ref_cov = ref_eigvecs @ np.diag(np.sqrt(ref_eigvals)) @ ref_eigvecs.T

        # Project back
        cleaned = sqrt_ref_cov @ whitened

        # 均值对齐，防止低频跳变
        if cleaned.shape == data.shape:
            cleaned += np.mean(data - cleaned, axis=1, keepdims=True)

        return cleaned


# EEG滤波函数


#IIR
def eeg_filter(data, srate, cutoff=0.5, order=2):
    nyq = 0.5 * srate
    if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
        low, high = cutoff
        if low == 0:
            mode = 'low'
            normal_cutoff = high / nyq
        elif high == 0:
            mode = 'high'
            normal_cutoff = low / nyq
        else:
            mode = 'band'
            normal_cutoff = [low / nyq, high / nyq]
    else:
        return data  # 不处理非法 cutoff 格式

    b, a = butter(order, normal_cutoff, btype=mode, analog=False)
    return filtfilt(b, a, data, axis=1)


# GUI界面嵌入Matplotlib绘图
class EEGGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time EEG Viewer")
        self.setGeometry(100, 100, 1000, 700)

        self.receiver = LSLStreamReceiver()
        self.viewer = LSLStreamVisualizer(self.receiver)

        layout = QVBoxLayout()
        self.canvas = FigureCanvas(self.viewer.fig)
        layout.addWidget(self.canvas)

        self.cutoff_input1 = QLineEdit("0.5")
        self.cutoff_input2 = QLineEdit("45")
        cutoff_layout = QHBoxLayout()
        cutoff_layout.addWidget(QLabel("Lower Cutoff:"))
        cutoff_layout.addWidget(self.cutoff_input1)
        cutoff_layout.addWidget(QLabel("Upper Cutoff:"))
        cutoff_layout.addWidget(self.cutoff_input2)
        layout.addLayout(cutoff_layout)

        self.start_btn = QPushButton("Start Stream")
        self.start_btn.clicked.connect(self.start_stream)
        layout.addWidget(self.start_btn)

        self.update_btn = QPushButton("Update Filter")
        self.update_btn.clicked.connect(self.update_filter_params)
        layout.addWidget(self.update_btn)

        # 添加 ASR 开关按钮
        self.asr_btn = QPushButton("Enable ASR")
        self.asr_btn.setCheckable(True)
        self.asr_btn.setChecked(False)
        self.asr_btn.clicked.connect(self.toggle_asr)
        layout.addWidget(self.asr_btn)

        self.setLayout(layout)

    def start_stream(self):
        self.update_filter_params()
        self.viewer.start()
        self.canvas.draw()

    def update_filter_params(self):
        try:
            val1 = float(self.cutoff_input1.text())
            val2 = float(self.cutoff_input2.text())
            self.receiver.cutoff = (val1, val2)
            print(f"已更新滤波参数: cutoff={self.receiver.cutoff}")
        except ValueError:
            print("Cutoff 值无效，使用默认值")

    def toggle_asr(self):
        self.receiver.use_asr = self.asr_btn.isChecked()
        status = "启用" if self.receiver.use_asr else "禁用"
        print(f"🧠 ASR 已{status}")
        self.asr_btn.setText("Disable ASR" if self.receiver.use_asr else "Enable ASR")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EEGGUI()
    gui.show()
    sys.exit(app.exec_())
