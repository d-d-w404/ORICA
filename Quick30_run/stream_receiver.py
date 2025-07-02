import threading
import time

from pylsl import StreamInlet, resolve_byprop
import numpy as np
from filter_utils import EEGSignalProcessor

from orica_processor import ORICAProcessor
from asrpy import ASR
import mne


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
        self.latest_sources = None
        self.latest_eog_indices = None

        #当我在切换通道的过程中，会让ic的个数发生改变，但是此时buffer还在运行，会导致卡死，
        #所以我需要把通道切换过程锁住
        self.lock = threading.Lock()

    #Selected channel indices: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    #Selected channel labels: ['AF7', 'Fpz', 'F7', 'Fz', 'T7', 'FC6', 'F4', 'C4', 'Oz', 'CP6', 'Cz', 'PO8', 'CP5', 'O2', 'O1', 'P3', 'P4', 'P7', 'P8', 'Pz', 'PO7', 'T8', 'C3', 'Fp2', 'F3', 'F8', 'FC5', 'AF8']
    #上面就是channel_range和channel_labels的格式，要调用函数就传入这样的list
    def set_channel_range_and_labels(self, new_range, new_labels):
        with self.lock:
            self.channel_range = new_range
            self.chan_labels = new_labels
            self.nbchan = len(new_range)
            self.reinitialize_orica()
            print(f"🔁 通道更新: {self.chan_labels}")

    def register_analysis_callback(self, callback_fn):
        """注册一个函数用于处理每次更新后的数据段 chunk"""
        self.analysis_callbacks.append(callback_fn)

    def reinitialize_orica(self):
        self.orica = ORICAProcessor(
            n_components=len(self.channel_range),
            max_samples=self.srate * 3,
            srate=self.srate
        )
        print("🔁 ORICA processor re-initialized with new channel range.")

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
        self.reinitialize_orica()




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
            chunk = EEGSignalProcessor.eeg_filter(chunk, self.srate, cutoff=self.cutoff)



            # ✅ 更新原始滤波后的 buffer（raw_buffer）
            self.last_unclean_chunk = chunk.copy()
            if self.raw_buffer is not None:
                self.raw_buffer = np.roll(self.raw_buffer, -chunk.shape[1], axis=1)
                self.raw_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk



            #✅ Step X: ORICA 去眼动伪影
            if self.orica.update_buffer(chunk[self.channel_range, :]):
                if self.orica.fit(self.orica.data_buffer):
                    cleaned = self.orica.transform(chunk[self.channel_range, :])
                    chunk[self.channel_range, :] = cleaned

                    # ✅ 新增：保存当前 ICA sources 用于可视化
                    self.latest_sources = self.orica.ica.transform(
                        self.orica.data_buffer.T).T  # (components, samples)

                    # ✅ 可选：也保存 EOG 伪影成分索引
                    self.latest_eog_indices = self.orica.eog_indices

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

    # def pull_and_update_buffer(self):
    #     samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
    #     if timestamps:
    #         chunk = np.array(samples).T  # shape: (channels, samples)
    #
    #         chunk = EEGSignalProcessor.eeg_filter(chunk, self.srate, cutoff=self.cutoff)
    #
    #         with self.lock:
    #             self.last_unclean_chunk = chunk.copy()
    #             if self.raw_buffer is not None:
    #                 self.raw_buffer = np.roll(self.raw_buffer, -chunk.shape[1], axis=1)
    #                 self.raw_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk
    #
    #             # ORICA 去伪影
    #             if self.orica.update_buffer(chunk[self.channel_range, :]):
    #                 if self.orica.fit(self.orica.data_buffer):
    #                     cleaned = self.orica.transform(chunk[self.channel_range, :])
    #                     chunk[self.channel_range, :] = cleaned
    #                     self.latest_sources = self.orica.ica.transform(self.orica.data_buffer.T).T
    #                     self.latest_eog_indices = self.orica.eog_indices
    #
    #             if self.use_asr:
    #                 chunk = self.apply_pyprep_asr(chunk)
    #
    #             num_new = chunk.shape[1]
    #             self.buffer = np.roll(self.buffer, -num_new, axis=1)
    #             self.buffer[:, -num_new:] = chunk
    #
    #         # 回调函数（异步线程）
    #         for fn in self.analysis_callbacks:
    #             try:
    #                 thread = threading.Thread(
    #                     target=fn,
    #                     kwargs=dict(
    #                         chunk=self.buffer[self.channel_range, :],
    #                         raw=self.raw_buffer[self.channel_range, :],
    #                         srate=self.srate,
    #                         labels=self.chan_labels
    #                     )
    #                 )
    #                 thread.start()
    #             except Exception as e:
    #                 print(f"❌ 回调分析函数错误: {e}")

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