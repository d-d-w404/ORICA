from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time
from stream_receiver import LSLStreamReceiver
import numpy as np

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

    # def update_plot(self, frame):
    #     self.receiver.pull_and_update_buffer()
    #
    #     plotdata = self.receiver.buffer[self.receiver.channel_range, ::int(self.receiver.srate/self.sampling_rate)]
    #     plotdata = plotdata - np.mean(plotdata, axis=1, keepdims=True)
    #     plotoffsets = np.arange(len(self.receiver.channel_range))[:, None] * self.data_scale
    #     plotdata += plotoffsets
    #
    #     self.ax.clear()
    #     self.ax.set_title(f"LSL Stream Type: {self.receiver.stream_type}")
    #     self.ax.set_xlabel("Time (samples)")
    #     self.ax.set_ylabel("Channels")
    #     self.ax.set_yticks(plotoffsets[:, 0])
    #     ylabels = [self.receiver.chan_labels[i] for i in range(len(self.receiver.channel_range))]
    #     self.ax.set_yticklabels(ylabels)
    #     self.ax.set_ylim(-self.data_scale, plotoffsets[-1][0] + self.data_scale)
    #     self.ax.plot(plotdata.T, linewidth=0.5)
    #
    #     current_time = time.time()
    #     if current_time - self.last_print_time >= 5.0:
    #         self.receiver.print_latest_channel_values()
    #         self.last_print_time = current_time

    def update_plot(self, frame):
        self.receiver.pull_and_update_buffer()

        # === Step 1: 获取 ASR 清洗后的数据
        clean_data = self.receiver.buffer[self.receiver.channel_range, ::int(self.receiver.srate / self.sampling_rate)]
        clean_data = clean_data - np.mean(clean_data, axis=1, keepdims=True)

        # === Step 2: 获取 bandpass-only 数据
        raw_data = self.receiver.raw_buffer[self.receiver.channel_range,
                   ::int(self.receiver.srate / self.sampling_rate)]
        raw_data = raw_data - np.mean(raw_data, axis=1, keepdims=True)

        # ✅ Step 2.1: 对齐两个数据长度（防止红线太短）
        min_len = min(clean_data.shape[1], raw_data.shape[1])
        clean_data = clean_data[:, -min_len:]
        raw_data = raw_data[:, -min_len:]

        # === Step 3: 添加垂直偏移量
        offsets = np.arange(len(self.receiver.channel_range))[:, None] * self.data_scale
        clean_data += offsets
        raw_data += offsets

        # === Step 4: 绘图
        self.ax.clear()
        self.ax.set_title(f"LSL Stream Type: {self.receiver.stream_type}")
        self.ax.set_xlabel("Time (samples)")
        self.ax.set_ylabel("Channels")
        self.ax.set_yticks(offsets[:, 0])
        ylabels = [self.receiver.chan_labels[i] for i in range(len(self.receiver.channel_range))]
        self.ax.set_yticklabels(ylabels)
        self.ax.set_ylim(-self.data_scale, offsets[-1][0] + self.data_scale)

        # 蓝色线：ASR 清洗后的 EEG
        self.ax.plot(clean_data.T, color='blue', linewidth=0.6)

        # 红色虚线：只经过 bandpass 的 EEG
        self.ax.plot(raw_data.T, color='red', linewidth=0.4, linestyle='--')



        # 可选：定期打印最新值
        current_time = time.time()
        if current_time - self.last_print_time >= 5.0:
            self.receiver.print_latest_channel_values()
            self.last_print_time = current_time

    def start(self):
        self.receiver.find_and_open_stream()
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000/self.refresh_rate)