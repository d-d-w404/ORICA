import threading
import time

from pylsl import StreamInlet, resolve_byprop
import numpy as np
from filter_utils import EEGSignalProcessor

from filter_utils_fir import example_usage
from pylsl import resolve_streams
from orica_processor import ORICAProcessor
from asrpy import ASR
import mne
from scipy.signal import medfilt
from meegkit import asr
import scipy.io

import numpy as np
from mne.filter import filter_data
from FirFilter import rest_fir_filter

class LSLStreamReceiver:
    def __init__(self, stream_type='EEG', time_range=5):
        self.stream_type = stream_type
        self.time_range = time_range
        self.inlet = None
        self.srate = None
        self.nbchan = None
        self.fixed_chunk_len = None  # 你想要的固定样本数
        self._stash = None

        self.buffer = None#这个buffer暂时只有view里面使用用来绘图
        self.chan_labels = []
        self.channel_range = []

        self.channel_manager=None
        self.cutoff = (1, 50)

        # ASR
        self.use_asr = False
        self.asr_calibrated = False
        self.asr_calibration_buffer = None
        self.prep_reference = None
        self.asr_filter = None  # ✅ 存储已校准的ASR实例

        self.raw_buffer = None  # 存放未 ASR 的 bandpass-only 历史数据
        self.buffer_real = None

        self.samples_buffer = None


        self.pair_buffer = None

        # ✅ 移除callback机制，改为纯数据接口模式
        # self.analysis_callbacks = []  # 存放所有回调分析函数

        # ✅ 新增：CAR设置
        self.use_car = False  # 是否启用CAR

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

        # #暂时使用name筛选stream
        stream_name = 'mybrain'
        streams = resolve_byprop('name', stream_name, timeout=60)

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

        
        #前额的5个通道
        #exclude = ['TRIGGER', 'ACC34','ACC33','ACC32', 'Packet Counter', 'ExG 2','ExG 1','ACC','A2','Oz','P3','F8','PO8','F7','Fz', 'T7', 'FC6', 'F4', 'C4', 'CP6', 'Cz', 'CP5', 'O2', 'O1', 'P4', 'P7', 'P8', 'Pz', 'PO7', 'T8', 'C3', 'F3', 'FC5']

        exclude = ['TRIGGER', 'ACC34','ACC33','ACC32', 'Packet Counter', 'ExG 2','ExG 1','ACC']#,'F7','F8'
        #exclude =[]
        self.chan_labels = self.channel_manager.get_labels_excluding_keywords(exclude)
        self.channel_range = self.channel_manager.get_indices_excluding_keywords(exclude)

        # print(self.chan_labels)
        # print(self.channel_range)
        #这里的self.channel_range 对应了每一个self.chan_labels标签的序号
        #假如我在上面的exclude中去掉了O2,那么O2这个label以及他的序号都会被删除。

        self.nbchan = len(self.channel_range)

        self._stash = np.empty((info.channel_count(), 0))
        self.fixed_chunk_len = 100  # 你想要的固定 chunk 长度

        self.buffer = np.zeros((info.channel_count(), self.srate * self.time_range))

        #for the comparing stream
        self.raw_buffer = np.zeros((info.channel_count(), self.srate * self.time_range))
        self.buffer_real = np.zeros((info.channel_count(), self.srate * self.time_range))

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
            #这一句话实际上就，这个updata_buffer就保证了一个稳定长度的buffer用于orica的处理，虽然chunk大小不一，但是没有关系
                if self.orica.fit(self.orica.data_buffer, self.channel_range, self.chan_labels, self.srate):
                    # ✅ 从 ORICAProcessor 取出 ICLabel 结果，供 GUI 使用
                    ic_probs, ic_labels = self.orica.get_iclabel_results()
                    self.latest_ic_probs = ic_probs
                    self.latest_ic_labels = ic_labels


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
        # 目标长度
        # target = self.fixed_chunk_len

        # # 拉取尽量接近目标长度的一次数据
        # samples, timestamps = self.inlet.pull_chunk(max_samples=target, timeout=0.0)
        # if timestamps:
        #     new = np.asarray(samples, dtype=float).T  # (channels, samples)
        #     print("test1",new.shape)


        #     # 拼接进缓存
        #     buf = np.concatenate([self._stash, new], axis=1)

        #     if buf.shape[1] >= target:
        #         # 截取固定长度作为本次 chunk
        #         chunk = buf[:, :target]
        #         # 剩余留作下次
        #         self._stash = buf[:, target:]
        #     else:
        #         # 不足则零填充到固定长度
        #         pad = np.zeros((self._stash.shape[0], target - buf.shape[1]), dtype=buf.dtype)
        #         chunk = np.concatenate([buf, pad], axis=1)
        #         self._stash = np.empty((self._stash.shape[0], 0), dtype=buf.dtype)

        #     # 后续处理继续用 chunk（shape: channels x target）
        #     # chunk = EEGSignalProcessor.eeg_filter(chunk, self.srate, cutoff=self.cutoff)
        #     # ... 你的后续逻辑
        #     print("test",chunk.shape)
        # 采集（保持低延迟）


        # samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
        # if timestamps:
        #     new = np.asarray(samples, dtype=float).T
        #     if new.shape[0] != 37:
        #         new = new[:37, :]
            
        #     # 累积到 stash
        #     if self._stash is None:
        #         self._stash = new
        #     else:
        #         self._stash = np.concatenate([self._stash, new], axis=1)
            
        #     # 只有当 stash 足够时才处理
        #     if self._stash.shape[1] >= self.fixed_chunk_len:
        #         # 取固定长度处理
        #         chunk = self._stash[:, :self.fixed_chunk_len]
        #         # 剩余部分保留
        #         self._stash = self._stash[:, self.fixed_chunk_len:]
        #         print("chunk",chunk.shape)
        #     else:
        #         print("nothing")
        #         return

        '''
        暂时先不用这个，因为我发现在orica_processor.py中，updata_buffer会保持一个稳定的长度
        所以虽然chunk长度不一致，但是我能够保证我后续在使用orica的时候能够从稳定长度的buffer中取数据
        
        '''






        # samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
        # if timestamps:

        #     new = np.asarray(samples, dtype=float).T  # (channels, samples)
        #     print("new1",new.shape)
        #     if new.shape[0] != 37:
        #         new = new[:37, :]

        #     print("new",new.shape)

        #     # 写入原始环形缓冲（实时）
        #     self.buffer_real = np.roll(self.buffer_real, -new.shape[1], axis=1)
        #     #self.buffer_real[:, -new.shape[1]:] = EEGSignalProcessor.eeg_filter(new, self.srate, cutoff=self.cutoff)
        #     self.buffer_real[:, -new.shape[1]:] = new

        #     print("buffer_real",self.buffer_real.shape)
        #     # 处理/绘图用固定帧长（不等待）：直接从环形缓冲取“最近 target 列”
        #     target = self.fixed_chunk_len  # 例如 int(self.srate / self.refresh_rate) 或 50
        #     print("target",target)
        #     take = min(target, self.buffer_real.shape[1])
        #     print("take",take)
        #     chunk = self.buffer_real[:, -take:]  # chunk 列数<=target，但不中断、不填充
        #     print("chunk",chunk.shape)

        #     # 如果你“必须”喂固定列数给算法：仅在足够时才处理；不够时跳过本帧
        #     if take < target:
        #         print("take < target")
        #         return  # 本帧跳过处理，继续保持实时采集与显示

        #samples, timestamps = self.inlet.pull_chunk(timeout=0.0)
        samples_random, timestamps = self.inlet.pull_chunk(timeout=0.0)
        #samples, timestamps = self.inlet.pull_chunk(max_samples=100, timeout=0.0)#

        # 检查是否有数据
        if not timestamps:
            return
            
        # 转换samples_random为正确的格式 (channels, samples)
        samples_random = np.array(samples_random).T
        
        if samples_random is not None:
            print("y"*20)
            print(samples_random.shape)
            print(samples_random)
            print("y"*20) 

        # #这个是用于验证的时候，保证每次传入都是固定的size
        # print("samples_random",samples_random.shape)
        # if self.samples_buffer is None:
        #     self.samples_buffer = samples_random.copy()
        # else:
        #     self.samples_buffer = np.concatenate([self.samples_buffer, samples_random], axis=1)
        
        # if self.samples_buffer.shape[1] >= 20:
        #     print("samples已经满了",self.samples_buffer.shape)

        #     # 取前面的100个样本作为samples
        #     samples = self.samples_buffer[:, :20]
            
        #     # 把剩余的样本放到新的samples_buffer的最前面
        #     remaining_samples = self.samples_buffer[:, 20:]
        #     if remaining_samples.shape[1] > 0:
        #         self.samples_buffer = remaining_samples
        #     else:
        #         print("合理=============================")
        #         self.samples_buffer = None

        # else:
        #     #print("samples还没有满",self.samples_buffer.shape)
        #     return
    
        #这里是不在乎samples大小的，实际中就是这样的
        samples=samples_random
        print("xxx")
        print("samplesxxxxx",samples.shape)
        '''
        timestampes 有的时候可能是(0,),这种情况就是没有数据，下面的if判定就不会执行。
        timestampes 有数据的时候就会是(samples的size,0)这样的数据，代表了每一个sample都有一个时间戳
        '''
        if timestamps:
            chunk = np.array(samples)  # shape: (channels, samples)这里的samples的大小是不固定的
            print("FIR filter in")
            print("testin",chunk.shape) # test (14, 36)
            print("testin",chunk[0:3,0:3])




            # # ✅ 更新原始滤波后的数据接口
            # self.last_unclean_chunk = chunk.copy()
            # if self.raw_buffer is not None:
            #     self.raw_buffer = np.roll(self.raw_buffer, -chunk.shape[1], axis=1)
            #     self.raw_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk


            # Step 1: 使用 MNE 的专业 FIR 滤波器
            chunk = self.apply_mne_iir_filter(chunk)
            #chunk = self.apply_scipy_fir_filter(chunk)
            # chunk = EEGSignalProcessor.eeg_filter(chunk, self.srate, cutoff=self.cutoff)
            #chunk = rest_fir_filter(chunk, srate=self.srate, cutoff=self.cutoff)

            
            # filtered = filter_data(
            #     data=chunk,
            #     sfreq=self.srate,
            #     l_freq=1.0,
            #     h_freq=50.0,
            #     method='fir',
            #     fir_design='firwin',
            #     fir_window='hamming',       # 更易满足阻带衰减需求
            #     phase='minimum',           # 最小相位，对应 flt_fir 'minimum-phase'
            #     l_trans_bandwidth=0.5,     # 对应 0.5→1 Hz 的低端过渡带
            #     h_trans_bandwidth=5.0,     # 对应 50→55 Hz 的高端过渡带
            #     filter_length='auto',      # 若需更强阻带，可手动加长，如 '20s' 或 8191
            #     verbose=False
            # )
            # chunk = filtered

            # print("testout",chunk.shape)
            # print("testout",chunk[0:3,0:3])


            # print("FIR filter out")

            # # ✅ 更新原始滤波后的数据接口
            # self.last_unclean_chunk = chunk.copy()
            # if self.raw_buffer is not None:
            #     self.raw_buffer = np.roll(self.raw_buffer, -chunk.shape[1], axis=1)
            #     self.raw_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk


            # # # ✅ 新增：CAR处理
            # # if self.use_car:  # 需要添加这个标志
            # #     chunk = self.apply_car(chunk)

            # # Step 2: ASR处理
            # # if self.use_asr:
            # #     chunk = self.apply_pyprep_asr(chunk)

            # print("ASR in")
            # # pip install asrpy
            # from asrpy import ASR
            # # 假设：
            # # calib: (n_channels, n_samples) 校准数据（尽量≥60秒），与在线数据同高通策略
            # # srate: 采样率 (Hz)
            # # stream() 产出在线分块 chunk: (n_channels, n_chunk)
            # raw_cali = mne.io.read_raw_eeglab(r'D:\work\Python_Project\ORICA\temp_txt\Demo_EmotivEPOC_EyeOpen.set', preload=True)
            # # 只保留 EEG 通道（去掉 EOG/stim/misc）
            # raw_cali.pick_types(eeg=True, eog=False, stim=False, misc=False)

            # # 2) 获取校准数据 calib 和采样率
            # calib = raw_cali.get_data()      
            #         # 形状 (n_channels, n_samples)
            # print("raw calib")
            # print(calib[0:3,0:3])

            
            # #去掉坏通道数据
            # calib=self.select_clean_reference(calib,self.srate)
            # print(calib[0:3,0:3])

            # # # 1) 初始化 ASR（参数映射自 flt_repair_bursts）
            # # asr = ASR(
            # #     sfreq=self.srate,
            # #     cutoff=10.0,           # stddev_cutoff
            # #     win_len=0.5,           # window_len
            # #     step_size=0.3333,      # block_size -> stats update step
            # #     lookahead=0.125,       # processing_delay
            # #     max_dims_ratio=0.66,   # max_dimensions (比例)
            # #     spectral_weighting=False,  # 若要复刻频谱加权，这里改成 True 并提供IIR
            # #     use_gpu=False
            # #     # 如果库支持：decim=10  # 对应 calib_precision
            # # )
            # asr = ASR(
            #     sfreq=self.srate,
            #     cutoff=10.0,        # 典型：stddev_cutoff
            #     win_len=0.5,        # 典型：window_len
            # )
            # print("done")

            # # 2) 标定：用 MNE RawArray 包装 calib 并拟合
            # cal_C, cal_S = calib.shape
            # cal_ch_names = [f"EEG{i+1}" for i in range(cal_C)]
            # info_cal = mne.create_info(ch_names=cal_ch_names, sfreq=self.srate, ch_types='eeg')
            # ref_raw = mne.io.RawArray(calib, info_cal, verbose=False)
            # asr.fit(ref_raw)

            # # 3) 在线处理：同样用 RawArray 包装 chunk 再 transform
            # print("ok")
            # ch_C, ch_S = chunk.shape
            # ch_ch_names = [f"EEG{i+1}" for i in range(ch_C)]
            # info_chunk = mne.create_info(ch_names=ch_ch_names, sfreq=self.srate, ch_types='eeg')
            # chunk_raw = mne.io.RawArray(chunk, info_chunk, verbose=False)
            # clean_raw = asr.transform(chunk_raw)
            # chunk = clean_raw.get_data()   # (channels, samples)
            # # 如果库支持分离噪声，可：
            # # clean_chunk, noise_chunk = asr.transform(chunk, return_noise=True)
            # # 你的后续处理...

            #step 2: ASR
            # ✅ 使用封装的 ASR 初始化函数（只在第一次调用时校准，之后直接复用）
            if self.asr_filter is None:
                self.initialize_asr_from_mat()
            
            # 应用 ASR 清理（如果已校准）
            if self.asr_filter is not None:
                chunk = self.asr_filter.transform(chunk)


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

            self.pair_buffer = (self.raw_buffer, self.buffer)

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
    
    def apply_mne_fir_filter(self, data):
        """
        使用 MNE-Python 的专业 FIR 滤波器
        专为 EEG 数据设计，效果更好
        """
        try:
            from mne.filter import filter_data
            
            # 使用 MNE 的专业 FIR 滤波器（优化参数）
            filtered_data = filter_data(
                data=data,
                sfreq=self.srate,
                l_freq=self.cutoff[0],      # 低频截止
                h_freq=self.cutoff[1],      # 高频截止
                method='fir',               # 使用 FIR 滤波器
                phase='zero-double',        # 零相位，双向滤波减少延迟
                l_trans_bandwidth=0.25,     # 更窄的低频过渡带
                h_trans_bandwidth=2.5,      # 更窄的高频过渡带
                filter_length='10s',        # 固定滤波器长度，避免过长
                fir_window='hamming',       # 使用 Hamming 窗
                verbose=False
            )
            
            print(f"✅ MNE FIR 滤波完成: {self.cutoff[0]}-{self.cutoff[1]} Hz")
            return filtered_data
            
        except Exception as e:
            print(f"❌ MNE FIR 滤波失败: {e}")
            print("⚠️ 回退到原始数据")
            return data

    def apply_mne_iir_filter(self, data):
        """
        使用 MNE-Python 的 IIR 滤波器（备选方案）
        延迟更小，适合实时处理
        """
        try:
            from mne.filter import filter_data
            
            # 使用 IIR 滤波器，延迟更小
            filtered_data = filter_data(
                data=data,
                sfreq=self.srate,
                l_freq=self.cutoff[0],      # 低频截止
                h_freq=self.cutoff[1],      # 高频截止
                method='iir',               # 使用 IIR 滤波器
                iir_params={'order': 4, 'ftype': 'butter'},  # 4阶 Butterworth
                verbose=False
            )
            
            print(f"✅ MNE IIR 滤波完成: {self.cutoff[0]}-{self.cutoff[1]} Hz")
            return filtered_data
            
        except Exception as e:
            print(f"❌ MNE IIR 滤波失败: {e}")
            print("⚠️ 回退到原始数据")
            return data

    def apply_scipy_fir_filter(self, data):
        """
        使用 SciPy 的 FIR 滤波器（备选方案）
        更轻量级，适合实时处理
        """
        try:
            from scipy import signal
            
            # 设计 FIR 带通滤波器
            nyquist = self.srate / 2
            low = self.cutoff[0] / nyquist
            high = self.cutoff[1] / nyquist
            
            # 使用 window 方法设计 FIR 滤波器
            taps = signal.firwin(
                numtaps=101,           # 滤波器长度
                cutoff=[low, high],    # 截止频率
                window='hann',         # 窗函数
                pass_zero=False,       # 带通滤波器
                scale=True
            )
            
            # 应用滤波器
            filtered_data = np.array([
                signal.lfilter(taps, 1.0, ch) for ch in data
            ])
            
            print(f"✅ SciPy FIR 滤波完成: {self.cutoff[0]}-{self.cutoff[1]} Hz")
            return filtered_data
            
        except Exception as e:
            print(f"❌ SciPy FIR 滤波失败: {e}")
            print("⚠️ 回退到原始数据")
            return data

    def initialize_asr_from_mat(self, mat_file_path=r"D:\work\Python_Project\ORICA\temp_txt\cleaned_data_quick30.mat"):
        """
        从 MATLAB 文件加载校准数据并初始化 ASR（只执行一次）
        
        Args:
            mat_file_path: 校准数据的 .mat 文件路径
        """
        if self.asr_filter is not None:
            print("⏩ ASR 已校准，跳过重复初始化")
            return self.asr_filter
        
        try:
            from meegkit import asr
            import scipy.io
            
            # 加载 MATLAB 文件
            mat_data = scipy.io.loadmat(mat_file_path)
            
            # 提取校准数据（EEGLAB 格式）
            calibration_data = None
            if 'cleaned_data' in mat_data:
                eeg_struct = mat_data['cleaned_data'][0, 0]
                if 'data' in eeg_struct.dtype.names:
                    calibration_data = eeg_struct['data']
            elif 'data' in mat_data:
                calibration_data = mat_data['data']
            
            if calibration_data is None:
                print(f"❌ 无法提取校准数据，可用字段: {mat_data.keys()}")
                return None
            
            # 转换为标准数组
            calibration_data = np.asarray(calibration_data, dtype=np.float64)
            print(f"✅ 校准数据加载成功 - 原始形状: {calibration_data.shape}")
            
            # 只选择当前使用的通道
            if calibration_data.shape[0] != len(self.channel_range):
                print(f"⚠️ 通道数不匹配：校准 {calibration_data.shape[0]} 通道，在线 {len(self.channel_range)} 通道")
                calibration_data = calibration_data[self.channel_range, :]
                print(f"✅ 已调整校准数据形状: {calibration_data.shape}")
            
            # 初始化并拟合 ASR
            self.asr_filter = asr.ASR(
                sfreq=self.srate,
                cutoff=5,
            )
            self.asr_filter.fit(calibration_data)
            print(f"✅ ASR 已校准完成，通道数: {calibration_data.shape[0]}")
            
            return self.asr_filter
            
        except Exception as e:
            print(f"❌ ASR 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return None

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

    def get_pair_data(self):
        return self.pair_buffer[0][self.channel_range, :], self.pair_buffer[1][self.channel_range, :] if self.pair_buffer is not None else None

    def get_pair_data_old(self, data_type='processed'):
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




    def apply_car(self, chunk):
        """使用MNE包实现CAR"""
        try:
            # 创建临时的MNE Raw对象
            info = mne.create_info(
                ch_names=self.channel_manager.get_labels_by_indices(self.channel_range),
                sfreq=self.srate,
                ch_types=["eeg"] * len(self.channel_range)
            )
            
            raw = mne.io.RawArray(chunk[self.channel_range, :], info)      
            # 应用CAR
            raw.set_eeg_reference('average')
            
            # 获取处理后的数据
            chunk[self.channel_range, :] = raw.get_data()
            
            return chunk
        
        except Exception as e:
            print(f"⚠️ MNE CAR处理失败，使用简单实现: {e}")
            return self.apply_car(chunk)  # 回退到简单实现

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

    def select_clean_reference(
        self,
        calib: np.ndarray,
        srate: float,
        window_len: float = 0.5,          # 窗长（秒）
        window_overlap: float = 0.66,     # 重叠比例
        zthresholds: tuple = (-3.5, 5.0), # Z 分数阈值 [下限, 上限]
        max_bad_channels: float = 0.15    # 单窗允许的坏通道比例（或绝对数）
    ):

        """
        基于 REST/BCILAB flt_clean_windows 的默认逻辑筛选“干净参考段”。
        输入:
        - calib: (n_channels, n_samples) 校准数据（已做与在线一致的高通）
        - srate: 采样率(Hz)
        参数:
        - window_len: 窗长(秒)，默认 0.5
        - window_overlap: 窗重叠比例，默认 0.66（步长 ≈ 0.17s）
        - zthresholds: 窗口 RMS 的稳健 Z 阈值（相对“干净 EEG”分布），默认 [-3.5, 5]
        - max_bad_channels: 单窗允许的“坏通道”上限（比例或绝对数），默认 0.15
        返回:
        - ref: (n_channels, n_kept_samples) 拼接的参考段
        - sample_mask: (n_samples,) 布尔掩码，True 表示保留
        - kept_slices: list[ slice ] 保留的窗口切片列表
        """
        C, S = calib.shape
        N = int(round(window_len * srate))
        if N <= 1 or N > S:
            raise ValueError("window_len 导致窗口大小异常，请检查 window_len 与 srate。")
        # 计算步长（和 BCILAB 一致：round(N*(1-overlap))）
        step = int(round(N * (1 - window_overlap)))
        step = max(1, step)
        # 生成窗口起点（和 BCILAB 一致，上限为 S-N）
        offsets = np.arange(0, max(1, S - N + 1), step, dtype=int)
        if len(offsets) == 0:
            offsets = np.array([0], dtype=int)
        W = len(offsets)

        # 每通道每窗口 RMS: (C, W)
        rms = np.empty((C, W), dtype=float)
        for wi, st in enumerate(offsets):
            seg = calib[:, st:st + N]
            rms[:, wi] = np.sqrt(np.mean(seg * seg, axis=1) + 1e-12)

        # 每通道做稳健 Z（用 median/MAD 近似 flt_clean_windows 的稳健拟合）
        med = np.median(rms, axis=1, keepdims=True)
        mad = np.median(np.abs(rms - med), axis=1, keepdims=True) + 1e-12
        # 将 MAD 转换为类似标准差的尺度（常用 1.4826）
        robust_std = 1.4826 * mad
        wz = (rms - med) / robust_std  # (C, W)

        # 窗口层面的“坏通道计数”
        bad_low = wz < zthresholds[0]
        bad_high = wz > zthresholds[1]
        bad_any = np.logical_or(bad_low, bad_high)  # (C, W)
        bad_count = bad_any.sum(axis=0)             # (W,)

        # 允许的坏通道数（比例 → 绝对数）
        if 0 < max_bad_channels < 1:
            max_bad_abs = int(np.round(C * max_bad_channels))
        else:
            max_bad_abs = int(max_bad_channels)
        max_bad_abs = max(0, min(C - 1, max_bad_abs))  # 不允许等于或超过 C

        # 需要移除的窗口：坏通道数 > 上限
        remove_mask = bad_count > max_bad_abs
        removed_windows = np.where(remove_mask)[0]
        kept_windows = np.where(~remove_mask)[0]

        # 生成样本掩码：移除坏窗覆盖的样本
        sample_mask = np.ones(S, dtype=bool)
        for wi in removed_windows:
            st = offsets[wi]
            sample_mask[st:st + N] = False

        # 保留窗的切片列表
        kept_slices = [slice(offsets[wi], offsets[wi] + N) for wi in kept_windows]

        # 拼接参考段
        ref = calib[:, sample_mask]

        #return ref, sample_mask, kept_slices
        return ref

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
            print(self.channels)

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