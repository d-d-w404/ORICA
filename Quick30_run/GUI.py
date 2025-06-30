# vis_stream_orica.py with OnlineASR integrated for real-time EEG cleaning

import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QLineEdit, QLabel, QHBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import time


class OnlineASR:
    def __init__(self, calibration_data, threshold=5.0):
        from scipy.linalg import eigh
        self.threshold = threshold
        self.mean = np.mean(calibration_data, axis=1, keepdims=True)
        centered = calibration_data - self.mean
        cov = centered @ centered.T / centered.shape[1]
        self.eigvals, self.eigvecs = eigh(cov)
        self.inv_sqrt_cov = self.eigvecs @ np.diag(1.0 / np.sqrt(self.eigvals + 1e-10)) @ self.eigvecs.T

    def clean(self, chunk):
        centered = chunk - self.mean
        z = self.inv_sqrt_cov @ centered
        mask = np.abs(z) > self.threshold
        cleaned = chunk.copy()
        cleaned[mask] = self.mean[mask]
        return cleaned


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

        self.use_asr = False
        self.asr_model = None
        self.asr_calibrated = False
        self.asr_calibration_buffer = None

    def find_and_open_stream(self):
        streams = resolve_byprop('type', self.stream_type, timeout=5)
        if not streams:
            raise RuntimeError(f"No LSL stream with type '{self.stream_type}' found.")

        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.srate = int(info.nominal_srate())
        self.nbchan = info.channel_count()

        chs = info.desc().child('channels').child('channel')
        for i in range(self.nbchan):
            label = chs.child_value('label') or f"Ch{i+1}"
            if not any(k in label for k in ['TRIGGER', 'ACC', 'ExG', 'Packet', 'A2']):
                self.chan_labels.append(label)
                self.channel_range.append(i)
            chs = chs.next_sibling()

        self.nbchan = len(self.channel_range)
        self.buffer = np.zeros((info.channel_count(), self.srate * self.time_range))

    def pull_and_update_buffer(self):
        samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
        if timestamps:
            chunk = np.array(samples).T
            chunk = eeg_filter(chunk, self.srate, cutoff=self.cutoff)

            if self.use_asr:
                chunk = self.apply_online_asr(chunk)

            num_new = chunk.shape[1]
            self.buffer = np.roll(self.buffer, -num_new, axis=1)
            self.buffer[:, -num_new:] = chunk

    def apply_online_asr(self, chunk):
        if not self.asr_calibrated:
            if self.asr_calibration_buffer is None:
                self.asr_calibration_buffer = chunk.copy()
            else:
                self.asr_calibration_buffer = np.concatenate((self.asr_calibration_buffer, chunk), axis=1)

            if self.asr_calibration_buffer.shape[1] >= self.srate * 10:
                try:
                    self.asr_model = OnlineASR(self.asr_calibration_buffer[self.channel_range, :])
                    self.asr_calibrated = True
                    print("✅ Online ASR calibrated with", self.asr_calibration_buffer.shape[1], "samples.")
                    self.asr_calibration_buffer = None
                except Exception as e:
                    print("❌ ASR calibration failed:", e)
        else:
            try:
                raw_data = chunk[self.channel_range, :]
                clean_data = self.asr_model.clean(raw_data)
                chunk[self.channel_range, :] = clean_data
            except Exception as e:
                print("❌ ASR cleaning failed:", e)

        return chunk


def eeg_filter(data, srate, cutoff=(0.5, 45), order=2):
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
        return data
    b, a = butter(order, normal_cutoff, btype=mode, analog=False)
    return filtfilt(b, a, data, axis=1)


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

        cutoff_layout = QHBoxLayout()
        self.cutoff_input1 = QLineEdit("0.5")
        self.cutoff_input2 = QLineEdit("45")
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

        self.asr_btn = QPushButton("Enable ASR")
        self.asr_btn.setCheckable(True)
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
        except ValueError:
            print("Invalid cutoff values")

    def toggle_asr(self):
        self.receiver.use_asr = self.asr_btn.isChecked()
        self.asr_btn.setText("Disable ASR" if self.receiver.use_asr else "Enable ASR")


class LSLStreamVisualizer:
    def __init__(self, receiver):
        self.receiver = receiver
        self.fig, self.ax = plt.subplots()
        self.last_print_time = time.time()

    def update_plot(self, frame):
        self.receiver.pull_and_update_buffer()
        data = self.receiver.buffer[self.receiver.channel_range, ::int(self.receiver.srate/100)]
        data = data - np.mean(data, axis=1, keepdims=True)
        offsets = np.arange(data.shape[0])[:, None] * 150
        data += offsets

        self.ax.clear()
        self.ax.set_title("EEG Stream")
        self.ax.plot(data.T, linewidth=0.5)

    def start(self):
        self.receiver.find_and_open_stream()
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EEGGUI()
    gui.show()
    sys.exit(app.exec_())
