import threading
import time

from pylsl import StreamInlet, resolve_byprop
import numpy as np
from filter_utils import EEGSignalProcessor
from pylsl import resolve_streams
from orica_processor import ORICAProcessor
from asrpy import ASR
import mne
from scipy.signal import medfilt

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

        # ✅ 移除callback机制，改为纯数据接口模式
        # self.analysis_callbacks = []  # 存放所有回调分析函数

        #ORICA
        self.orica = None
        self.latest_sources = None
        self.latest_eog_indices = None

        #当我在切换通道的过程中，会让ic的个数发生改变，但是此时buffer还在运行，会导致卡死，
        #所以我需要把通道切换过程锁住
        self.lock = threading.Lock()
        
        # ✅ 新增：数据更新线程控制
        self.data_update_thread = None
        self.is_running = False
        self.update_interval = 0.1  # 100ms更新间隔
        
        # ✅ 新增：数据接口相关
        self.last_unclean_chunk = None  # 最新的原始数据块
        self.last_processed_chunk = None  # 最新的处理后数据块
        self.data_timestamp = 0  # 数据时间戳，用于检测数据更新

        #用于画图时，保证处理后的数据和处理前的能够在时间上吻合
        self.chunk_pairs = []  # [(timestamp, unclean, processed)]

    def find_and_open_stream(self):

        #check the whole stream
        streams = resolve_streams()
        print("🔍 当前可用的 LSL 流：")
        for i, stream in enumerate(streams):
            print(
                f"[{i}] Name: {stream.name()}, Type: {stream.type()}, Channels: {stream.channel_count()}, ID: {stream.source_id()}")
        #--------------------------------------

        print(f"Searching for LSL stream with type = '{self.stream_type}'...")
        #streams = resolve_byprop('type', self.stream_type, timeout=5)

        #暂时使用name筛选stream
        stream_name = 'mybrain'
        streams = resolve_byprop('name', stream_name, timeout=5)

        if not streams:
            raise RuntimeError(f"No LSL stream with type '{self.stream_type}' found.")

        #我使用REST做LSL的时候有两个lsl,都是eeg类型，这里应该会默认选择第一个，但是第一个不是lsl output的，会卡死
        #这里暂时使用1，因为0用不了老是卡死
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
#         channel_all = [
#     'AF7',      # 0
#     'Fpz',      # 1
#     'F7',       # 2
#     'Fz',       # 3
#     'T7',       # 4
#     'FC6',      # 5
#     'Fp1',      # 6
#     'F4',       # 7
#     'C4',       # 8
#     'Oz',       # 9
#     'CP6',      # 10
#     'Cz',       # 11
#     'PO8',      # 12
#     'CP5',      # 13
#     'O2',       # 14
#     'O1',       # 15
#     'P3',       # 16
#     'P4',       # 17
#     'P7',       # 18
#     'P8',       # 19
#     'Pz',       # 20
#     'PO7',      # 21
#     'T8',       # 22
#     'C3',       # 23
#     'Fp2',      # 24
#     'F3',       # 25
#     'F8',       # 26
#     'FC5',      # 27
#     'AF8',      # 28
#     'A2',       # 29
#     'ExG 1',    # 30
#     'ExG 2'     # 31
# ]

        exclude = [
            'AF7',      # 0
            'Fpz',      # 1
            'F7',       # 2
            #'Fz',       # 3
            #'T7',       # 4
            'FC6',      # 5
            'Fp1',      # 6
            #'F4',       # 7
            'C4',       # 8
            'Oz',       # 9
            'CP6',      # 10
            'Cz',       # 11
            'PO8',      # 12
            'CP5',      # 13
            #'O2',       # 14
            #'O1',       # 15
            'P3',       # 16
            'P4',       # 17
            'P7',       # 18
            'P8',       # 19
            #'Pz',       # 20
            'PO7',      # 21
            #'T8',       # 22
            'C3',       # 23
            'Fp2',      # 24
            #'F3',       # 25
            'F8',       # 26
            'FC5',      # 27
            'AF8',      # 28
            'A2',       # 29
            'ExG 1',    # 30
            'ExG 2'     # 31
            'TRIGGER',
            'ACC34',
            'ACC33',
            'ACC32',
            'Packet Counter',
            'ACC',
        ]
        
        #前额的5个通道
        #exclude = ['TRIGGER', 'ACC34','ACC33','ACC32', 'Packet Counter', 'ExG 2','ExG 1','ACC','A2','Oz','P3','F8','PO8','F7','Fz', 'T7', 'FC6', 'F4', 'C4', 'CP6', 'Cz', 'CP5', 'O2', 'O1', 'P4', 'P7', 'P8', 'Pz', 'PO7', 'T8', 'C3', 'F3', 'FC5']

        #exclude = ['TRIGGER', 'ACC34','ACC33','ACC32', 'Packet Counter', 'ExG 2','ExG 1','ACC','A2','Oz','P3','F8','PO8','F7']
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



    def process_orica(self, chunk):
        """
        对输入的chunk进行ORICA伪影去除处理。
        输入：chunk（shape: 通道数, 样本数），只处理self.channel_range对应的通道。
        输出：
            cleaned_chunk: 伪影去除后的chunk（只对self.channel_range部分做了修改，其余通道不变）
            ica_sources: ICA源信号（components, samples），可用于可视化
            eog_indices: EOG伪影成分索引
            A: ICA mixing matrix (通道数, 成分数)
            spectrum: dict，包含所有IC分量的频谱（'freqs': 频率, 'powers': shape=(n_components, n_freqs)）
        """
        import numpy as np
        from scipy.signal import welch
        cleaned_chunk = chunk.copy()
        ica_sources = None
        eog_indices = None
        A = None
        spectrum = None
        # ORICA处理
        if self.orica is not None:
            if self.orica.update_buffer(chunk[self.channel_range, :]):
                if self.orica.fit(self.orica.data_buffer, self.channel_range, self.chan_labels, self.srate):
                    #classify
                    # ic_probs, ic_labels = self.orica.classify(chunk[self.channel_range, :],self.chan_labels, self.srate)
                    # if ic_probs is not None and ic_labels is not None:
                    #     print('ICLabel概率:', ic_probs)
                    #     print('ICLabel标签:', ic_labels)


                    cleaned = self.orica.transform(chunk[self.channel_range, :])
                    cleaned_chunk[self.channel_range, :] = cleaned
                    ica_sources = self.orica.ica.transform(self.orica.data_buffer.T).T  # (components, samples)
                    eog_indices = self.orica.eog_indices
                    # 获取mixing matrix A
                    try:
                        A = np.linalg.pinv(self.orica.ica.W)
                    except Exception:
                        A = None
                    # 获取所有IC分量的spectrum
                    if ica_sources is not None:
                        powers = []
                        freqs = None
                        for ic in range(ica_sources.shape[0]):
                            f, Pxx = welch(ica_sources[ic], fs=self.srate)
                            if freqs is None:
                                freqs = f
                            powers.append(Pxx)
                        powers = np.array(powers)  # shape: (n_components, n_freqs)
                        spectrum = {'freqs': freqs, 'powers': powers}
        return cleaned_chunk, ica_sources, eog_indices, A, spectrum

    def pull_and_update_buffer(self):
        samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
        if timestamps:
            chunk = np.array(samples).T  # shape: (channels, samples)
            #print("test",chunk.shape) # 

            # Step 1: Bandpass or highpass filter
            chunk = EEGSignalProcessor.eeg_filter(chunk, self.srate, cutoff=self.cutoff)

            # ✅ 更新原始滤波后的数据接口
            self.last_unclean_chunk = chunk.copy()
            if self.raw_buffer is not None:
                self.raw_buffer = np.roll(self.raw_buffer, -chunk.shape[1], axis=1)
                self.raw_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk


            #✅ Step X: ORICA 去眼动伪影（重构为独立函数）
            chunk, ica_sources, eog_indices,A,spectrum = self.process_orica(chunk)
            if ica_sources is not None:
                self.latest_sources = ica_sources
            if eog_indices is not None:
                self.latest_eog_indices = eog_indices

            # Step 2: ASR处理
            if self.use_asr:
                chunk = self.apply_pyprep_asr(chunk)

            # Step 3: Update ring buffer
            num_new = chunk.shape[1]
            self.buffer = np.roll(self.buffer, -num_new, axis=1)
            self.buffer[:, -num_new:] = chunk
            
            # ✅ 更新处理后数据接口
            self.last_processed_chunk = chunk.copy()
            self.data_timestamp = time.time()


            
                # 3. 存成一对
            timestamp = time.time()
            self.chunk_pairs.append((timestamp, self.last_unclean_chunk, self.last_processed_chunk))
            # 只保留最近N对
            if len(self.chunk_pairs) > 1:
                self.chunk_pairs.pop(0)

            # ✅ Step 4: 回调分析函数
            # for fn in self.analysis_callbacks: # 移除此行
            #     try: # 移除此行
            #         thread = threading.Thread( # 移除此行
            #             target=fn, # 移除此行
            #             kwargs=dict( # 移除此行
            #                 chunk=self.buffer[self.channel_range, :], # 移除此行
            #                 raw=self.raw_buffer[self.channel_range, :], # 移除此行
            #                 srate=self.srate, # 移除此行
            #                 labels=self.chan_labels # 移除此行
            #             ) # 移除此行
            #         ) # 移除此行
            #         thread.start() # 移除此行
            #     except Exception as e: # 移除此行
            #         print(f"❌ 回调分析函数错误: {e}") # 移除此行


    


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
        # self.analysis_callbacks.append(callback_fn) # 移除此行
        pass # 移除此行

    def reinitialize_orica(self):
        self.orica = ORICAProcessor(
            n_components=len(self.channel_range),
            max_samples=self.srate * 10,
            srate=self.srate
        )
        print("🔁 ORICA processor re-initialized with new channel range.")

    def start(self):
        """启动数据流和数据更新线程"""
        if hasattr(self, 'is_running') and self.is_running:
            print("⚠️ 数据流已在运行")
            return
        self.find_and_open_stream()
        self.is_running = True
        self.data_update_thread = threading.Thread(target=self._data_update_loop, daemon=True)
        self.data_update_thread.start()
        print("✅ 数据流和数据更新线程已启动")

    def stop(self):
        """停止数据更新线程"""
        self.is_running = False
        if hasattr(self, 'data_update_thread') and self.data_update_thread and self.data_update_thread.is_alive():
            self.data_update_thread.join(timeout=1.0)
        print("🛑 数据更新线程已停止")

    def _data_update_loop(self):
        """数据更新循环 - 在独立线程中运行"""
        while self.is_running:
            try:
                self.pull_and_update_buffer()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"❌ 数据更新错误: {e}")
                time.sleep(0.1)  # 错误时短暂等待

    # ✅ 新增：数据接口方法
    def get_raw_data(self):
        """获取最新的原始数据（仅带通滤波）"""
        return self.last_unclean_chunk.copy() if self.last_unclean_chunk is not None else None
    
    def get_processed_data(self):
        """获取最新的处理后数据（ORICA + ASR）"""
        return self.last_processed_chunk.copy() if self.last_processed_chunk is not None else None

    def get_pair_data(self, data_type='processed'):
        if data_type == 'raw':
            return self.chunk_pairs.copy()[0][1][self.channel_range, :] if self.chunk_pairs is not None else None
        else:
            return self.chunk_pairs.copy()[0][2][self.channel_range, :] if self.chunk_pairs is not None else None
        #return self.chunk_pairs.copy() if self.chunk_pairs is not None else None
    
    def get_buffer_data(self, data_type='processed'):
        """获取缓冲区数据
        
        Args:
            data_type: 'raw' 或 'processed'
        """
        if data_type == 'raw':
            return self.raw_buffer[self.channel_range, :] if self.raw_buffer is not None else None
        else:
            return self.buffer[self.channel_range, :] if self.buffer is not None else None
    
    
    def get_ica_sources(self):
        """获取最新的ICA源信号"""
        return self.latest_sources.copy() if self.latest_sources is not None else None
    
    def get_eog_indices(self):
        """获取EOG伪影成分索引"""
        return self.latest_eog_indices.copy() if self.latest_eog_indices is not None else None
    
    def get_channel_info(self):
        """获取通道信息"""
        return {
            'labels': self.chan_labels.copy() if self.chan_labels else [],
            'indices': self.channel_range.copy() if self.channel_range else [],
            'count': len(self.channel_range) if self.channel_range else 0,
            'sampling_rate': self.srate
        }
    
    def is_data_available(self):
        """检查是否有可用数据"""
        return (self.last_unclean_chunk is not None and 
                self.last_processed_chunk is not None and 
                self.buffer is not None)
    
    def get_data_timestamp(self):
        """获取数据时间戳，用于检测数据更新"""
        return self.data_timestamp





    def print_latest_channel_values(self):
        if self.buffer is None:
            print("⚠️ Buffer 尚未初始化，无法打印通道值")
            return

        # print("--- EEG Channel Values (Last Sample) ---")
        # for i, ch_idx in enumerate(self.channel_range):
        #     label = self.chan_labels[i]
        #     last_value = self.buffer[ch_idx, -1]
        #     rms = np.sqrt(np.mean(self.buffer[ch_idx] ** 2))
        #     print(f"{label:>4}: {last_value:>8.2f} μV | RMS: {rms:.2f}")



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

                        cutoff=20,
                        win_len=2,
                        win_overlap=0.8,
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



                # 在 ASR 后加个中值滤波处理，平滑
                cleaned_chunk = cleaned_raw.get_data()
                cleaned_chunk = medfilt(cleaned_chunk, kernel_size=(1, 5))  # 保通道不变，仅时间平滑
                chunk[self.channel_range, :] = cleaned_chunk


        except Exception as e:
            print("❌ Error in apply_pyprep_asr:", e)

        return chunk



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
                if not any(kw == ch["label"] for kw in keywords)]

    def get_indices_excluding_keywords(self, keywords):
        return [ch["index"] for ch in self.channels
                if not any(kw == ch["label"] for kw in keywords)]

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