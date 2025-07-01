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

from PyQt5.QtWidgets import QGroupBox
import sys
import numpy as np
import mne
from pyprep.prep_pipeline import PrepPipeline
from PyQt5.QtWidgets import QCheckBox
from asrpy import ASR
import numpy as np
import mne

from scipy.signal import welch
import threading

from scipy.special import expit
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import joblib


from sklearn.decomposition import FastICA


#this is the main project currently


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

        self.channel_manager=None
        self.cutoff = (0.5, 45)

        # ASR
        self.use_asr = False
        self.asr_calibrated = False
        self.asr_calibration_buffer = None
        self.prep_reference = None


        self.raw_buffer = None  # 存放未 ASR 的 bandpass-only 历史数据

        self.analysis_callbacks = []  # 存放所有回调分析函数

        # 在线回归模型（情绪强度）
        # self.online_model = SGDRegressor(learning_rate='adaptive', eta0=0.01)
        # self.scaler = StandardScaler()
        # self.first_fit_done = False
        # self.first_fit_lock = threading.Lock()  # 🔒 加锁


        #ORICA
        self.orica = None

    def register_analysis_callback(self, callback_fn):
        """注册一个函数用于处理每次更新后的数据段 chunk"""
        self.analysis_callbacks.append(callback_fn)

    def find_and_open_stream(self):
        print(f"Searching for LSL stream with type = '{self.stream_type}'...")
        streams = resolve_byprop('type', self.stream_type, timeout=5)

        if not streams:
            raise RuntimeError(f"No LSL stream with type '{self.stream_type}' found.")

        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()

        print("=== StreamInfo XML description ===")
        print(self.inlet.info().as_xml())

        info = self.inlet.info()
        self.channel_manager = ChannelManager(info)

        self.srate = int(info.nominal_srate())
        self.nbchan = info.channel_count()

        # chs = info.desc().child('channels').child('channel')
        # all_labels = []
        # for _ in range(self.nbchan):
        #     label = chs.child_value('label')
        #     all_labels.append(label if label else f"Ch {_+1}")
        #     chs = chs.next_sibling()
        #
        # # self.chan_labels = all_labels.copy()
        # # self.enabled = [True] * len(self.chan_labels)  # ← 添加这行，标记每个通道是否启用
        #
        # exclude_keywords = ['TRIGGER', 'ACC', 'ExG', 'Packet', 'A2','O2','Oz']
        # for i, label in enumerate(all_labels):
        #     if not any(keyword in label for keyword in exclude_keywords):
        #         self.chan_labels.append(label)
        #         self.channel_range.append(i)


        # 或者自定义排除某些关键词
        exclude = ['TRIGGER', 'ACC34','ACC33','ACC32', 'Packet Counter', 'ExG 2','ExG 1','A2']
        self.chan_labels = self.channel_manager.get_labels_excluding_keywords(exclude)
        self.channel_range = self.channel_manager.get_indices_excluding_keywords(exclude)

        # print(self.chan_labels)
        # print(self.channel_range)
        #这里的self.channel_range 对应了每一个self.chan_labels标签的序号
        #假如我在上面的exclude中去掉了O2,那么O2这个label以及他的序号都会被删除。

        self.nbchan = len(self.channel_range)
        self.buffer = np.zeros((info.channel_count(), self.srate * self.time_range))

        #for the comparing stream
        self.raw_buffer = np.zeros((info.channel_count(), self.srate * self.time_range))

        print(f"Stream opened: {info.channel_count()} channels at {self.srate} Hz")
        print(f"Using {self.nbchan} EEG channels: {self.chan_labels}")



        # ✅ 初始化 ORICA
        self.orica = ORICAProcessor(
            n_components=len(self.channel_range),
            max_samples=self.srate * 3
        )
        print("✅ ORICA processor initialized.")

    def pull_and_update_buffer(self):
        samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
        if timestamps:
            chunk = np.array(samples).T  # shape: (channels, samples)


            # #step 0: replace woring channels with means
            # print("before")
            # print(np.array(chunk).shape)
            # chunk = clean_bad_channels(chunk, labels=self.chan_labels)
            # print("after")
            # print(np.array(chunk).shape)

            # Step 1: Bandpass or highpass filter
            chunk = eeg_filter(chunk, self.srate, cutoff=self.cutoff)


            # ✅ 更新原始滤波后的 buffer（raw_buffer）
            self.last_unclean_chunk = chunk.copy()
            if self.raw_buffer is not None:
                self.raw_buffer = np.roll(self.raw_buffer, -chunk.shape[1], axis=1)
                self.raw_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk



            # Step X: ORICA 去眼动伪影
            #print(np.array(chunk[self.channel_range, :]).shape)#(29, 64)
            #print(np.array(chunk).shape)#(37, 64)
            # if self.orica.update_buffer(chunk[self.channel_range, :]):#输出true的同时，更新了窗口
            #     if self.orica.fit(self.orica.data_buffer):
            #         chunk[self.channel_range, :] = self.orica.transform(chunk[self.channel_range, :])





            # Step 2
            if self.use_asr:
                chunk = self.apply_pyprep_asr(chunk)

            # Step 3: Update ring buffer
            num_new = chunk.shape[1]
            self.buffer = np.roll(self.buffer, -num_new, axis=1)
            self.buffer[:, -num_new:] = chunk

            # ✅ Step 4: 回调分析函数，输入是当前最新的 chunk 数据
            #当执行到这里的时候就会触发回调函数，运行try下面的内容
            # for fn in self.analysis_callbacks:
            #     try:
            #         fn(chunk=self.buffer[self.channel_range, :],  # 清洗后的
            #            raw=self.raw_buffer[self.channel_range, :],  # 仅 bandpass
            #            srate=self.srate,
            #            labels=self.chan_labels)
            #     except Exception as e:
            #         print(f"❌ 回调分析函数错误: {e}")



            # 在你的 update_plot 或 pull_and_update_buffer 之后：
            for fn in self.analysis_callbacks:
                try:
                    thread = threading.Thread(
                        target=fn,
                        kwargs=dict(
                            chunk=self.buffer[self.channel_range, :],
                            raw=self.raw_buffer[self.channel_range, :],
                            srate=self.srate,
                            labels=self.chan_labels
                        )
                    )
                    thread.start()
                except Exception as e:
                    print(f"❌ 回调分析函数错误: {e}")

    def print_latest_channel_values(self):
        pass
        # print("--- EEG Channel Values (last column) ---")
        # for i, ch in enumerate(self.channel_range):
        #     label = self.chan_labels[i]
        #     value = self.buffer[ch, -1]
        #     rms = np.sqrt(np.mean(self.buffer[ch]**2))
        #     print(f"{label}: {value:.2f} (RMS: {rms:.2f})")

    def apply_pyprep_asr(self, chunk):
        try:
            if not self.asr_calibrated:
                # 🔄 Step 1: 收集静息数据进行校准
                if self.asr_calibration_buffer is None:
                    self.asr_calibration_buffer = chunk.copy()
                else:
                    self.asr_calibration_buffer = np.concatenate(
                        (self.asr_calibration_buffer, chunk), axis=1
                    )

                if self.asr_calibration_buffer.shape[1] >= self.srate * 20:
                    print("⏳ Calibrating ASR...")

                    # ➕ 创建 Raw 对象用于校准
                    info = mne.create_info(
                        ch_names=self.channel_manager.get_labels_by_indices(self.channel_range),
                        sfreq=self.srate,
                        ch_types=["eeg"] * len(self.channel_range)
                    )
                    x=self.channel_manager.get_labels_by_indices(self.channel_range)
                    # print("X" * 30)
                    # print(x)
                    # print("X" * 30)

                    raw = mne.io.RawArray(
                        self.asr_calibration_buffer[self.channel_range, :], info
                    )

                    #
                    # print(self.asr_calibration_buffer[self.channel_range, :])
                    # print("Selected shape:", self.buffer[self.channel_range, :].shape)
                    # print("Selected labels:", self.chan_labels)

                    raw.set_montage("standard_1020")

                    # 🔧 初始化并校准 ASR 实例
                    self.asr_instance = ASR(
                        sfreq=self.srate,

                        cutoff=3,
                        win_len=0.5,
                        win_overlap=0.66,
                        blocksize=self.srate
                    )

                    self.asr_instance.fit(raw)

                    self.asr_calibrated = True
                    self.asr_calibration_buffer = None
                    print("✅ ASRpy calibrated successfully.")

            else:
                # 🔄 Step 2: 实时清洗数据
                info = mne.create_info(
                    ch_names=self.channel_manager.get_labels_by_indices(self.channel_range),
                    sfreq=self.srate,
                    ch_types=["eeg"] * len(self.channel_range)
                )
                raw_chunk = mne.io.RawArray(chunk[self.channel_range, :], info)
                raw_chunk.set_montage("standard_1020")

                cleaned_raw = self.asr_instance.transform(raw_chunk)
                chunk[self.channel_range, :] = cleaned_raw.get_data()

        except Exception as e:
            print("❌ Error in apply_pyprep_asr:", e)

        return chunk

    # def apply_pyprep_asr(self, chunk):
    #     if not self.asr_calibrated:
    #         if self.asr_calibration_buffer is None:
    #             self.asr_calibration_buffer = chunk.copy()
    #         else:
    #             self.asr_calibration_buffer = np.concatenate((self.asr_calibration_buffer, chunk), axis=1)
    #
    #         if self.asr_calibration_buffer.shape[1] >= self.srate * 10:
    #             try:
    #                 info = mne.create_info(
    #                     ch_names=[self.chan_labels[i] for i in range(len(self.channel_range))],
    #                     sfreq=self.srate,
    #                     ch_types=["eeg"] * len(self.channel_range)
    #                 )
    #                 raw = mne.io.RawArray(self.asr_calibration_buffer[self.channel_range, :], info)
    #                 raw.set_montage("standard_1020")
    #
    #                 prep = PrepPipeline(raw, {
    #                     "ref_chs": raw.ch_names,
    #                     "reref_chs": raw.ch_names,
    #                     "line_freqs": [50]
    #                 }, montage="standard_1020")
    #
    #                 prep.fit()
    #                 self.prep_reference = prep
    #                 self.asr_calibrated = True
    #                 print("✅ pyPREP ASR calibrated.")
    #                 self.asr_calibration_buffer = None
    #             except Exception as e:
    #                 print("❌ ASR calibration failed:", e)
    #     else:
    #         try:
    #             info = mne.create_info(
    #                 ch_names=[self.chan_labels[i] for i in range(len(self.channel_range))],
    #                 sfreq=self.srate,
    #                 ch_types=["eeg"] * len(self.channel_range)
    #             )
    #             raw_chunk = mne.io.RawArray(chunk[self.channel_range, :], info)
    #             raw_chunk.set_montage("standard_1020")
    #
    #             prep = PrepPipeline(raw_chunk, {
    #                 "ref_chs": raw_chunk.ch_names,
    #                 "reref_chs": raw_chunk.ch_names,
    #                 "line_freqs": [50]
    #             }, montage="standard_1020")
    #
    #             prep.fit()
    #             clean_data = prep.raw.get_data()
    #             chunk[self.channel_range, :] = clean_data
    #         except Exception as e:
    #             print("❌ pyPREP ASR cleaning failed:", e)
    #
    #     return chunk


from ORICA import ORICA


class ORICAProcessor:
    def __init__(self, n_components=None, max_samples=1000):
        self.n_components = n_components
        self.max_samples = max_samples
        self.ica = None
        self.data_buffer = None
        self.eog_indices = []  # indices of components identified as eye artifacts

    def fit(self, data):
        """Fit ICA on the buffered EEG data"""
        assert data.shape[0] == self.n_components, f"Expected {self.n_components} channels, got {data.shape[0]}"

        if data.shape[1] < self.max_samples:
            return False  # Not enough data yet


        # ica = FastICA(n_components=self.n_components or data.shape[0],#都是29一样的
        #               max_iter=2000, tol=1e-3, random_state=42)
        #
        # sources = ica.fit_transform(data.T)

        print("dog")
        self.ica = ORICA(n_components=min(self.n_components, data.shape[0]), learning_rate=0.001)

        print("dog")
        sources = self.ica.fit_transform(data.T).T  # shape: (components, samples)

        print("dog")

        self.identify_eye_artifacts(sources.T)
        return True

    # def identify_eye_artifacts(self, components):
    #     """Heuristic: identify eye components as those with high frontal power and low frequency"""
    #     self.eog_indices = []
    #     for i, comp in enumerate(components):
    #         power = np.sum(comp ** 2)
    #         if power > np.percentile([np.sum(c ** 2) for c in components], 90):
    #             self.eog_indices.append(i)
    #
    #     print("EOG artifact:",self.eog_indices)

    def identify_eye_artifacts(self, components):
        powers = np.sum(components ** 2, axis=1)  # 每个分量的总能量
        threshold = np.percentile(powers, 90)
        self.eog_indices = [i for i, p in enumerate(powers) if p > threshold]

        print("EOG artifact indices (filtered):", self.eog_indices)

    def transform(self, new_data):
        if self.ica is None:
            return new_data

        sources = self.ica.transform(new_data.T)
        sources[:, self.eog_indices] = 0  # Zero out EOG components
        cleaned = self.ica.inverse_transform(sources)
        return cleaned.T

    def update_buffer(self, new_chunk):
        if self.data_buffer is None:
            self.data_buffer = new_chunk
        else:
            self.data_buffer = np.concatenate([self.data_buffer, new_chunk], axis=1)
            if self.data_buffer.shape[1] > self.max_samples:
                self.data_buffer = self.data_buffer[:, -self.max_samples:]

        print(np.array(self.data_buffer).shape)

        return self.data_buffer.shape[1] >= self.max_samples




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




def clean_bad_channels(chunk, labels=None, threshold_uv=200):
    """检测并替换掉可能断触的通道（过大波动）"""
    stds = np.std(chunk, axis=1)
    bad_indices = np.where(stds > threshold_uv)[0]

    if len(bad_indices) > 0:
        good_channels = [i for i in range(chunk.shape[0]) if i not in bad_indices]
        if good_channels:
            mean_signal = np.mean(chunk[good_channels, :], axis=0)
            for i in bad_indices:
                chunk[i, :] = mean_signal

            # 输出通道名而不是索引
            if labels:
                bad_names = [labels[i] for i in bad_indices]
                print(f"⚠️ 替换了异常通道: {bad_names}")
            else:
                print(f"⚠️ 替换了异常通道: 索引 {bad_indices}")
    return chunk





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


def analyze_bandpower(chunk, raw, srate, labels, gui=None):
    try:
        freqs, psd = welch(chunk, fs=srate, nperseg=srate, axis=1)
        band_dict = {}
        for band, (fmin, fmax) in {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }.items():
            idx = (freqs >= fmin) & (freqs <= fmax)
            if np.any(idx):
                band_dict[band] = float(np.mean(psd[:, idx]))

        if gui and gui.bandpower_plot:
            gui.bandpower_plot.update_bandpower(band_dict)

    except Exception as e:
        print("❌ analyze_bandpower 错误:", e)







import time



def heavy_analysis(chunk, raw, srate, labels):
    t0 = time.time()



    #time.sleep(2)  # 等待 2 秒

    #print("🧪 [重计算开始]")

    try:
        # === Step 0: 输入检查 ===
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 2:
            print("❗ chunk 非法，跳过分析。shape:", np.shape(chunk))
            return
        if not isinstance(raw, np.ndarray) or raw.ndim != 2:
            print("❗ raw 非法，跳过分析。shape:", np.shape(raw))
            return
        if chunk.shape[1] < srate:
            print("⚠️ 数据不足 1 秒，跳过")
            return


        # === Step 1: bandpower ===
        for data_name, data in zip(['cleaned', 'raw'], [chunk, raw]):

            #print("data.shape =",data.shape)#data.shape = (29, 2500)
            #print(srate)
            freqs,psd  = welch(data, fs=srate, nperseg=srate, axis=1)
            #print("psd.shape =", psd.shape)  # psd.shape = (29,251)
            #代表了29个通道，在251个频率上的功率密度值
            #后续直接对 psd[:, 8:13] → 求 alpha 波段的能量
            #print(psd)

            #print("freqs.shape =", freqs.shape)  # freqs.shape = (251,)
            #print(freqs)


            for band, (fmin, fmax) in {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }.items():
                idx = (freqs >= fmin) & (freqs <= fmax)
                if not np.any(idx):
                    print(f"⚠️ Band {band} not found in freqs, skipping.")
                    continue
                band_power = np.mean(psd[:, idx], axis=1)#计算每个通道在该频段的平均功率
                #print(f"{data_name} | {band} power: {np.mean(band_power):.2f}")

        # === Step 2: Hjorth 参数 ===
        def compute_hjorth(data):
            d1 = np.diff(data, axis=1)#求导（axis=1，对行） 信号变化速度
            d2 = np.diff(d1, axis=1)#信号变化加速度
            activity = np.var(data, axis=1)#振幅高 → 活动强；比如觉醒时脑电 activity 较大。
            mobility = np.sqrt(np.var(d1, axis=1) / activity)#越高的 mobility，表示脑电越活跃于高频段，如 beta、gamma。
            complexity = np.sqrt(np.var(d2, axis=1) / np.var(d1, axis=1))#高复杂度可能表示注意力转移、思维活跃、感知突变等。
            return activity, mobility, complexity

        hjorth_act, hjorth_mob, hjorth_comp = compute_hjorth(chunk)

        # === Step 3: 协方差矩阵特征值分解 ===
        cov = np.cov(chunk)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)[::-1]

        # === Step 4: 构造特征 + 假分类 ===
        dummy_features = np.concatenate([hjorth_act, eigvals[:10]])
        dummy_prediction = int(np.sum(dummy_features) % 3)  # 假装有个分类器

        t1 = time.time()
        #print(f"✅ [重计算完成] 耗时: {(t1 - t0) * 1000:.1f} ms，预测类: {dummy_prediction}")

    except Exception as e:
        print("❌ heavy_analysis 错误:", e)


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



class RealTimeAttentionEstimator:
    def __init__(self, gui=None,receiver=None):
        self.gui = gui
        self.history = []
        self.max_history = 30  # 平滑用的历史窗口
        self.receiver=receiver

    def extract_attention_score(self, chunk, srate):
        # ✅ Step 0: 选择特定通道进行 bandpower（如 AF7=0, AF8=1）

        selected_channels_name=['F7','F8','T7','T8','Fpz']
        selected_channels = self.receiver.channel_manager.get_indices_by_labels(selected_channels_name)
        #print(selected_channels)# 替换成你想用的通道索引
        chunk = chunk[selected_channels, :]  # 只保留感兴趣通道
        if chunk.shape[0] != 5:
            raise ValueError("❌ 找不到所有指定通道：Fpz, F7, F8, T7, T8")

        ref = chunk[0, :]  # Fpz 是第一个通道
        chunk = chunk[1:, :] - ref  # 将其余通道减去 Fpz，实现 re-referencing




        # Step 1: Hjorth 参数
        d1 = np.diff(chunk, axis=1)
        d2 = np.diff(d1, axis=1)
        activity = np.mean(np.var(chunk, axis=1))
        complexity = np.mean(np.sqrt(np.var(d2, axis=1) / np.var(d1, axis=1)))

        # Step 2: 频段功率
        freqs, psd = welch(chunk, fs=srate, nperseg=srate, axis=1)
        def band_power(fmin, fmax):
            idx = (freqs >= fmin) & (freqs <= fmax)
            return np.mean(psd[:, idx])

        alpha = band_power(8, 13)
        theta = band_power(4, 8)
        beta = band_power(13, 30)
        gamma = band_power(30, 45)

        # #Step 3: 归一化 Attention Score（可调权重）
        # score = (
        #     -alpha * 0.6 +   # alpha ↓ 表示集中
        #     +theta * 0.2    # theta ↑
        #     +beta * 0.5
        #     #+gamma * 0.3
        #     # +activity * 0.1 -
        #     # complexity * 0.2
        # )
        #
        # print(score)
        #
        # normalized = float(expit(score))  # sigmoid(score)
        # print(normalized)
        # return normalized

        # Step 3: 注意力打分逻辑
        epsilon = 1e-6
        # NeuroChat 核心公式：Engagement Index = β / (α + θ)
        engagement = (beta + epsilon) / (alpha + theta + epsilon)

        # 添加滑动窗口平均（用于平滑注意力）
        self.history.append(engagement)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        # engagement_avg = np.mean(self.history)
        #
        # # Normalize with assumed calibration range (建议之后替换为实际的 E_min 和 E_max)
        # E_min = 0.2
        # E_max = 1.2
        # normalized = (engagement_avg - E_min) / (E_max - E_min + epsilon)
        # normalized = float(np.clip(normalized, 0.0, 1.0))

        engagement = (beta + epsilon) / (alpha + theta + epsilon)

        self.history.append(engagement)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        engagement_avg = np.mean(self.history)

        # ⚠️ 用历史最大最小值代替静态 E_min/E_max（更灵敏）
        E_min = min(self.history)
        E_max = max(self.history)
        range_ = max(E_max - E_min, epsilon)  # 避免除以 0
        normalized = (engagement_avg - E_min) / range_
        normalized = float(np.clip(normalized, 0.0, 1.0))

        print(
            f"[Attention] alpha={alpha:.2f}, theta={theta:.2f}, beta={beta:.2f}, engagement_avg={engagement_avg:.2f}, norm={normalized:.2f}")
        return normalized

    def callback(self, chunk, raw, srate, labels):
        try:
            score = self.extract_attention_score(chunk, srate)
            #print(score)
            #self.history.append(score)
            if len(self.history) > self.max_history:
                self.history.pop(0)

            smoothed = np.mean(self.history)

            if self.gui:
                self.gui.update_attention_circle(smoothed)

        except Exception as e:
            print("❌ 注意力评分错误:", e)







from PyQt5.QtCore import QTimer, QRect, QPoint, QPointF, Qt
from PyQt5.QtGui import QColor, QPainter, QFont
import numpy as np
import random

class AttentionBallWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎯 Attention Ball View")
        self.resize(800, 600)
        self.ball_pos = QPoint(400, 300)
        self.ball_radius = 30
        self.color = QColor("gray")
        self.score = 0.0
        self.velocity = QPointF(0, 0)

        # 初始化表达式与显示控制
        self.current_expression = ""
        self.current_result = ""
        self.showing_result = False

        # 小球漂移计时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.move_ball)
        self.timer.start(16)



    def show_result(self):
        self.showing_result = True
        self.update()

    def move_ball(self):
        jitter = 0.5*(0.1+abs(max(0,(1.0 - self.score))))
        ax = np.random.uniform(-jitter, jitter) * 5
        ay = np.random.uniform(-jitter, jitter) * 5
        self.velocity += QPointF(ax, ay)
        max_speed = 12.0
        self.velocity.setX(np.clip(self.velocity.x(), -max_speed, max_speed))
        self.velocity.setY(np.clip(self.velocity.y(), -max_speed, max_speed))
        self.ball_pos += QPoint(int(self.velocity.x()), int(self.velocity.y()))
        margin = self.ball_radius
        self.ball_pos.setX(np.clip(self.ball_pos.x(), margin, self.width() - margin))
        self.ball_pos.setY(np.clip(self.ball_pos.y(), margin, self.height() - margin))
        self.velocity *= 0.92
        self.update()

    def update_attention(self, score):
        self.score = float(score)
        self.ball_radius = int(40 + 50 * self.score)

        # 🎨 将注意力分数映射到红→黄→绿的渐变
        # 0.0 → 红 (255, 0, 0)
        # 0.5 → 黄 (255, 255, 0)
        # 1.0 → 绿 (0, 255, 0)
        if self.score <= 0.5:
            # 红 → 黄 线性插值
            r = 255
            g = int(255 * (self.score / 0.5))  # 0→255
            b = 0
        else:
            # 黄 → 绿 线性插值
            r = int(255 * (1 - (self.score - 0.5) / 0.5))  # 255→0
            g = 255
            b = 0

        self.color = QColor(r, g, b)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.ball_pos, self.ball_radius, self.ball_radius)

        # ✏️ 显示注意力得分（保留两位小数）
        painter.setPen(Qt.black)
        font = QFont("Arial", 14)
        font.setBold(True)
        painter.setFont(font)

        text = f"{self.score:.2f}"
        text_rect = QRect(self.ball_pos.x() - self.ball_radius,
                          self.ball_pos.y() - 10,
                          self.ball_radius * 2,
                          20)
        painter.drawText(text_rect, Qt.AlignCenter, text)




from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QCheckBox, QScrollArea, QWidget

# class ChannelSelectorDialog(QDialog):
#     def __init__(self, parent_gui, receiver):
#         super().__init__()
#         self.setWindowTitle("Select EEG Channels")
#         self.receiver = receiver
#         self.parent_gui = parent_gui
#         self.checkboxes = []
#
#         layout = QVBoxLayout()
#
#         scroll = QScrollArea()
#         scroll_widget = QWidget()
#         scroll_layout = QVBoxLayout(scroll_widget)
#
#         for i, label in enumerate(receiver.chan_labels):
#             cb = QCheckBox(label)
#             cb.setChecked(i in receiver.channel_range)
#             self.checkboxes.append(cb)
#             scroll_layout.addWidget(cb)
#
#         scroll.setWidget(scroll_widget)
#         scroll.setWidgetResizable(True)
#         layout.addWidget(scroll)
#
#         confirm_btn = QPushButton("Confirm Selection")
#         confirm_btn.clicked.connect(self.apply_selection)
#         layout.addWidget(confirm_btn)
#
#         self.setLayout(layout)
#         self.resize(300, 400)
#
#     def apply_selection(self):
#         selected_indices = [
#             i for i, cb in enumerate(self.checkboxes) if cb.isChecked()
#         ]
#         self.receiver.channel_range = selected_indices
#         print(f"✅ Selected channel indices: {selected_indices}")
#         print(f"✅ Selected channel labels: {[self.receiver.chan_labels[i] for i in selected_indices]}")
#         self.accept()



class ChannelSelectorDialog(QDialog):
    def __init__(self, parent_gui, receiver):
        super().__init__()
        self.setWindowTitle("Select EEG Channels")
        self.receiver = receiver
        self.parent_gui = parent_gui
        self.checkboxes = []

        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 从 ChannelManager 获取全部通道
        self.channel_info = self.receiver.channel_manager.channels

        # 当前选中的 index 列表
        current_range = set(self.receiver.channel_range)

        for ch in self.channel_info:
            cb = QCheckBox(ch["label"])
            # 勾选当前在 range 中的通道
            cb.setChecked(ch["index"] in current_range)
            self.checkboxes.append(cb)
            scroll_layout.addWidget(cb)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        confirm_btn = QPushButton("Confirm Selection")
        confirm_btn.clicked.connect(self.apply_selection)
        layout.addWidget(confirm_btn)

        self.setLayout(layout)
        self.resize(300, 400)

    def apply_selection(self):
        selected_indices = []
        selected_labels = []

        for cb, ch in zip(self.checkboxes, self.channel_info):
            if cb.isChecked():
                selected_indices.append(ch["index"])
                selected_labels.append(ch["label"])

        # 更新 receiver 的通道范围和标签
        self.receiver.channel_range = selected_indices
        self.receiver.chan_labels = selected_labels

        print(f"✅ Selected channel indices: {selected_indices}")
        print(f"✅ Selected channel labels: {selected_labels}")
        self.accept()



class ChannelManager:
    def __init__(self, lsl_info):
        """
        从 pylsl.StreamInfo 提取并保存所有有意义的信息
        包括全局属性和通道结构
        """
        # === 全局流信息 ===
        self.name = lsl_info.name()
        self.type = lsl_info.type()
        self.channel_count = lsl_info.channel_count()
        self.srate = int(lsl_info.nominal_srate())
        self.uid = lsl_info.uid()
        self.hostname = lsl_info.hostname()
        self.source_id = lsl_info.source_id()
        self.version = lsl_info.version()
        self.created_at = lsl_info.created_at()

        # === 所有通道信息 ===
        #这个类的实例中有着所有通道的信息，包括一些非EEG，但是它有所有通道的信息，后续还能使用。
        self.channels = []  # 每个元素是 dict：label, index, type, unit

        ch = lsl_info.desc().child('channels').child('channel')
        index = 0
        while ch.name() == "channel":
            label = ch.child_value("label")
            ch_type = ch.child_value("type")
            unit = ch.child_value("unit")

            self.channels.append({
                "label": label,
                "index": index,
                "type": ch_type,
                "unit": unit
            })

            ch = ch.next_sibling()
            index += 1

    # === 通道筛选方法 ===
    def get_all_labels(self):
        return [ch["label"] for ch in self.channels]

    def get_all_indices(self):
        return [ch["index"] for ch in self.channels]

    def get_labels_by_type(self, desired_type="EEG"):
        return [ch["label"] for ch in self.channels if ch["type"] == desired_type]

    def get_indices_by_type(self, desired_type="EEG"):
        return [ch["index"] for ch in self.channels if ch["type"] == desired_type]

    def get_indices_by_labels(self, labels):
        label_set = set(labels)
        return [ch["index"] for ch in self.channels if ch["label"] in label_set]

    def get_labels_excluding_keywords(self, keywords):
        return [ch["label"] for ch in self.channels
                if not any(kw in ch["label"] for kw in keywords)]

    def get_indices_excluding_keywords(self, keywords):
        return [ch["index"] for ch in self.channels
                if not any(kw in ch["label"] for kw in keywords)]

    def get_labels_by_indices(self, indices):
        """
        根据索引列表获取对应的通道名列表
        """
        index_set = set(indices)
        return [ch["label"] for ch in self.channels if ch["index"] in index_set]

    # === 信息打印方法 ===
    def print_summary(self):
        print("=== Stream Info ===")
        print(f"Name      : {self.name}")
        print(f"Type      : {self.type}")
        print(f"Channels  : {self.channel_count}")
        print(f"Sampling  : {self.srate} Hz")
        print(f"UID       : {self.uid}")
        print(f"Hostname  : {self.hostname}")
        print(f"Source ID : {self.source_id}")
        print(f"Version   : {self.version}")
        print(f"Created   : {self.created_at}")
        print("\n=== Channel List ===")
        for ch in self.channels:
            print(f"[{ch['index']:02}] {ch['label']} | Type: {ch['type']} | Unit: {ch['unit']}")

#在LSLStreamReceiver.find_and_open_stream()中
# info = self.inlet.info()
# self.channel_manager = ChannelManager(info)
#
# # 只获取 EEG 类型
# self.chan_labels = self.channel_manager.get_labels_by_type("EEG")
# self.channel_range = self.channel_manager.get_indices_by_type("EEG")
#
# # 或者自定义排除某些关键词
# exclude = ['TRIGGER', 'ACC', 'Packet', 'ExG']
# self.chan_labels = self.channel_manager.get_labels_excluding_keywords(exclude)
# self.channel_range = self.channel_manager.get_indices_excluding_keywords(exclude)
#
# # 打印概况
# self.channel_manager.print_summary()
#
# # 获取全局参数（可选）
# print("采样率为：", self.channel_manager.srate)


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import numpy as np

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


# GUI界面嵌入Matplotlib绘图
from PyQt5.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QGroupBox

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

        # ========== 外层 Layout ==========
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self.tabs)
        self.setLayout(outer_layout)


    def start_stream(self):
        self.update_filter_params()
        self.viewer.start()
        self.canvas.draw()

        self.receiver.register_analysis_callback(analyze_bandpower)
        self.receiver.register_analysis_callback(heavy_analysis)
        #self.receiver.register_analysis_callback(self.regressor.callback)

        self.attention_estimator = RealTimeAttentionEstimator(gui=self,receiver=self.receiver)
        self.receiver.register_analysis_callback(self.attention_estimator.callback)


        self.receiver.register_analysis_callback(lambda **kwargs: analyze_bandpower(gui=self, **kwargs))

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EEGGUI()
    gui.show()
    sys.exit(app.exec_())
