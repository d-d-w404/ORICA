'''
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time
from stream_receiver import LSLStreamReceiver
import numpy as np

from matplotlib.collections import LineCollection

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
        # ✅ 使用receiver的数据接口获取数据
        if not self.receiver.is_data_available():
            return  # 数据不可用时跳过绘图
        
        # 获取通道信息
        channel_info = self.receiver.get_channel_info()
        if not channel_info['labels']:
            return
        
        # === Step 1: 获取 ASR 清洗后的数据
        clean_data = self.receiver.get_buffer_data('processed')
        if clean_data is None:
            return
            
        # 降采样
        clean_data = clean_data[:, ::int(self.receiver.srate / self.sampling_rate)]
        clean_data = clean_data - np.mean(clean_data, axis=1, keepdims=True)

        # === Step 2: 获取 bandpass-only 数据
        raw_data = self.receiver.get_buffer_data('raw')
        if raw_data is None:
            return
            
        raw_data = raw_data[:, ::int(self.receiver.srate / self.sampling_rate)]
        raw_data = raw_data - np.mean(raw_data, axis=1, keepdims=True)

        # ✅ Step 2.1: 对齐两个数据长度（防止红线太短）
        min_len = min(clean_data.shape[1], raw_data.shape[1])
        clean_data = clean_data[:, -min_len:]
        raw_data = raw_data[:, -min_len:]

        # === Step 3: 添加垂直偏移量
        offsets = np.arange(len(channel_info['labels']))[:, None] * self.data_scale
        clean_data += offsets
        raw_data += offsets

        # === Step 4: 绘图
        self.ax.clear()
        self.ax.set_title(f"LSL Stream Type: {self.receiver.stream_type}")
        self.ax.set_xlabel("Time (samples)")
        self.ax.set_ylabel("Channels")
        self.ax.set_yticks(offsets[:, 0])
        self.ax.set_yticklabels(channel_info['labels'])
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


    def update_plot_synced(self, frame):
        # ✅ 使用receiver的数据接口获取数据
        if not self.receiver.is_data_available():
            return  # 数据不可用时跳过绘图
        
        # 获取通道信息
        channel_info = self.receiver.get_channel_info()
        if not channel_info['labels']:
            return
        
        # === Step 1: 获取 ASR 清洗后的数据

        raw_data, clean_data = self.receiver.get_pair_data()
        #clean_data = self.receiver.get_pair_data('processed')
        #clean_data = self.receiver.get_buffer_data('processed')
        if clean_data is None:
            return
            
        # 降采样
        clean_data = clean_data[:, ::int(self.receiver.srate / self.sampling_rate)]
        clean_data = clean_data - np.mean(clean_data, axis=1, keepdims=True)

        # === Step 2: 获取 bandpass-only 数据
        #raw_data = self.receiver.get_pair_data('raw')
        #raw_data = self.receiver.get_buffer_data('raw')
        if raw_data is None:
            return
            
        raw_data = raw_data[:, ::int(self.receiver.srate / self.sampling_rate)]
        raw_data = raw_data - np.mean(raw_data, axis=1, keepdims=True)

        # ✅ Step 2.1: 对齐两个数据长度（防止红线太短）
        min_len = min(clean_data.shape[1], raw_data.shape[1])
        clean_data = clean_data[:, -min_len:]
        raw_data = raw_data[:, -min_len:]

        # === Step 3: 添加垂直偏移量
        offsets = np.arange(len(channel_info['labels']))[:, None] * self.data_scale
        clean_data += offsets
        raw_data += offsets



        # === Step 4: 绘图
        self.ax.clear()
        self.ax.set_title(f"LSL Stream Type: {self.receiver.stream_type}")
        self.ax.set_xlabel("Time (samples)")
        self.ax.set_ylabel("Channels")
        self.ax.set_yticks(offsets[:, 0])
        self.ax.set_yticklabels(channel_info['labels'])
        self.ax.set_ylim(-self.data_scale, offsets[-1][0] + self.data_scale)

        
        # 红色虚线：只经过 bandpass 的 EEG
        self.ax.plot(raw_data.T, color='red', linewidth=0.4, linestyle='--')

        # 蓝色线：ASR 清洗后的 EEG
        self.ax.plot(clean_data.T, color='blue', linewidth=0.6)

        # 可选：定期打印最新值
        current_time = time.time()
        if current_time - self.last_print_time >= 5.0:
            self.receiver.print_latest_channel_values()
            self.last_print_time = current_time




    








    def start(self):
        # 只需启动receiver的数据流
        self.receiver.start()
        # 启动绘图动画
        #self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000/self.refresh_rate)
        #self.ani = FuncAnimation(self.fig, self.update_plot_synced, interval=1000/self.refresh_rate)
        self.ani = FuncAnimation(self.fig, self.update_plot_synced, interval = 1)


        
    def stop(self):
        """停止数据更新和绘图"""
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        self.receiver.stop()



'''





from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time
from stream_receiver import LSLStreamReceiver
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.collections import LineCollection
import numpy as np
import time


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
        self.last_print_time = time.time()

        # 运行时缓存
        self.channel_labels = []
        self.n_ch = 0
        self.offsets = None  # (n_ch,)
        self.x = None        # 时间轴（样本序号或秒）
        self.ds = 1          # 降采样因子
        self.lc_raw = None
        self.lc_clean = None


    def _make_segments(self, x, Y):
        # Y: (n_ch, n_pts) -> (n_ch, n_pts, 2)
        X = np.tile(x, (Y.shape[0], 1))
        return np.stack([X, Y], axis=2)

    def init_plot(self):
        channel_info = self.receiver.get_channel_info()
        self.channel_labels = channel_info['labels']
        self.n_ch = len(self.channel_labels)
        if self.n_ch == 0:
            return []

        # 降采样因子
        self.ds = max(1, int(self.receiver.srate / self.sampling_rate))

        # 时间轴占位（1点）
        self.x = np.array([0.0], dtype=float)

        # 垂直偏移
        self.offsets = (np.arange(self.n_ch) * self.data_scale).astype(float)

        # 轴设定（只做一次）
        self.ax.set_title(f"LSL Stream Type: {self.receiver.stream_type}")
        self.ax.set_xlabel("Time (samples)")
        self.ax.set_ylabel("Channels")
        self.ax.set_yticks(self.offsets)
        self.ax.set_yticklabels(self.channel_labels)
        self.ax.set_ylim(-self.data_scale, self.offsets[-1] + self.data_scale)
        self.ax.set_xlim(0, 1)

        # 先用零数据占位：2*n_ch 段（前 n_ch: raw，后 n_ch: clean）
        zeros = np.zeros((self.n_ch, 1), dtype=float)
        seg_raw_init   = self._make_segments(self.x, zeros + self.offsets[:, None])
        seg_clean_init = self._make_segments(self.x, zeros + self.offsets[:, None])
        segments_init  = np.concatenate([seg_raw_init, seg_clean_init], axis=0)

        # 每段的样式
        colors      = (['red']  * self.n_ch) + (['blue'] * self.n_ch)
        linestyles  = (['--']   * self.n_ch) + (['solid']* self.n_ch)
        linewidths  = ([0.4]    * self.n_ch) + ([0.6]   * self.n_ch)

        # 关键：只用一个 LineCollection（一个艺术家）
        self.lc = LineCollection(
            segments_init,
            colors=colors,
            linestyles=linestyles,
            linewidths=linewidths,
            animated=True  # 配合 blit
        )
        self.lc.set_rasterized(True)
        self.ax.add_collection(self.lc)

        return [self.lc]

    def update_plot_synced(self, frame):
        if not self.receiver.is_data_available():
            return []

        channel_info = self.receiver.get_channel_info()
        labels = channel_info['labels']
        if not labels:
            return []

        # 通道数变化则重建
        if len(labels) != self.n_ch:
            self.ax.cla()
            return self.init_plot()

        # 数据获取
        # clean_data = self.receiver.get_buffer_data('processed')
        # raw_data   = self.receiver.get_buffer_data('raw')

        # 2) update_plot_synced 里，原子读取（替换原来的两次 get_buffer_data 调用）
        raw_data,clean_data  = self.receiver.get_pair_data()

        # print("clean_data",clean_data.shape)
        # print("raw_data",raw_data.shape)



        if clean_data is None or raw_data is None:
            return [self.lc]

        # 降采样与去均值
        ds = max(1, int(self.receiver.srate / self.sampling_rate))
        if ds != getattr(self, 'ds', ds):
            self.ds = ds
        clean_data = clean_data[:, ::self.ds]
        raw_data   = raw_data[:,   ::self.ds]
        clean_data = clean_data - np.mean(clean_data, axis=1, keepdims=True)
        raw_data   = raw_data   - np.mean(raw_data,   axis=1, keepdims=True)

        # 对齐长度
        min_len = min(clean_data.shape[1], raw_data.shape[1])
        if min_len <= 1:
            return [self.lc]
        clean_data = clean_data[:, -min_len:]
        raw_data   = raw_data[:,   -min_len:]

        # 添加偏移（副本）
        clean_disp = clean_data + self.offsets[:, None]
        raw_disp   = raw_data   + self.offsets[:, None]

        # 时间轴（样本编号；若用秒：/self.sampling_rate）
        x = np.arange(min_len, dtype=float)

        # 构造两组段并合并为一个集合（顺序：raw 段 + clean 段）
        seg_raw   = self._make_segments(x, raw_disp)
        seg_clean = self._make_segments(x, clean_disp)
        segments  = np.concatenate([seg_raw, seg_clean], axis=0)

        # 同一帧内只更新这“一位艺术家”
        self.lc.set_segments(segments)

        # x 轴范围
        self.ax.set_xlim(x[0], x[-1])

        # （可选）定期打印
        current_time = time.time()
        if current_time - self.last_print_time >= 5.0:
            self.receiver.print_latest_channel_values()
            self.last_print_time = current_time

        return [self.lc]


    def start(self):
        self.receiver.start()
        # 用同步更新器 + blit
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot_synced,
            init_func=self.init_plot,
            interval=max(10, int(1000 / self.refresh_rate)),  # 20~50ms 更稳
            blit=True
        )


    def stop(self):
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        self.receiver.stop()
