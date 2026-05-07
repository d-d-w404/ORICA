import os
import threading
import time
import numpy as np
from scipy.signal import welch
from pylsl import StreamInlet, resolve_byprop
import numpy as np
from filter_utils import EEGSignalProcessor

from filter_utils_fir import example_usage
from pylsl import resolve_streams
from orica_processor import ORICAProcessor
import mne
from scipy.signal import medfilt
from meegkit import asr
import scipy.io


from mne.filter import filter_data
from FirFilter import rest_fir_filter
from scipy.signal import butter, lfilter, lfilter_zi

# 在线 ASR：默认 meegkit；环境变量 EEG_ASR_BACKEND=asrpy 时使用 asrpy（asr_process 跨缓冲保持 R/Zi/cov）。
#this class to some extent is like a container which includs all of the paraments(orica, icalabel)
class LSLStreamReceiver:
    def __init__(self, stream_type='EEG', time_range=5, stream_name='mybrain'):
    #def __init__(self, stream_type='EEG', time_range=5, stream_name='CGX Quick-32r Q32r-0584'):
        # inoput LSL stream info
        self.stream_type = stream_type
        self.stream_name = stream_name
        self.inlet = None
        self.srate = None
        self.nbchan = None
        self.chan_labels = []
        self.chan_range = []
        self.channel_manager=None

        # buffer 
        self.time_range = time_range #control the buffer size by sec
        self.raw_buffer = None  
        self.buffer = None
        self.asr_buffer = None
        self.pair_buffer = None

        # chunk (currently I use chunk for vis, because it's more small than buffer, making the vis smooth)
        # chunk size is not a fixed value
        self.last_unclean_chunk = None  # this chunk currently only used for vis(before orica)
        self.last_processed_chunk = None  #this chunk currently only used for vis(after orica)
        self.chunk_pairs = []  # conbining two chunk above


        self.raw_chunk=None
        self.iir_chunk=None
        self.asr_chunk=None
        self.orica_chunk=None

        #IIR
        self.cutoff = (1, 50)
        # 在线IIR滤波器状态（用于状态保持）
        self.iir_filter_b = None  # 分子系数
        self.iir_filter_a = None  # 分母系数
        self.iir_filter_zi = None  # 滤波器初始状态 (filter_order, n_channels)

        # ASR
        self.use_asr = False
        self.asr_filter = None
        # values for online asr calibration
        self.asr_calibration_data = None 
        self.asr_calibration_size = 0
        # 累积用于ASR处理的缓冲区（按时间拼接，channels x samples）
        self.asr_accum_buffer = None
        self.asr_backend = os.environ.get("EEG_ASR_BACKEND", "meegkit").strip().lower()
        if self.asr_backend not in ("meegkit", "asrpy"):
            print(
                f"[WARN] EEG_ASR_BACKEND={self.asr_backend!r} 无效（仅 meegkit / asrpy），已回退 meegkit"
            )
            self.asr_backend = "meegkit"
        self._asrpy_proc_R = None
        self._asrpy_proc_Zi = None
        self._asrpy_proc_cov = None
        print(f"[ASR] 在线 EEG_ASR_BACKEND={self.asr_backend}")

        #ORICA
        self.orica = None
        self.latest_sources = None

        #ICLabel
        self.latest_ic_probs = None
        self.latest_ic_labels = None
        self.latest_eog_indices = None
        # ICLabel artifact判定阈值（可由 run_two_instances.py 通过环境变量控制）
        self.icalabel_threshold = 0.7
        
        # 简单的10秒数据收集
        self.calibration_data = None
        self.calibration_size = 0
        self.calibration_duration = 10  # 10秒
        self.calibration_collected = False
        
        # 多阶段数据保存相关变量（raw / iir / asr / orica）
        self.save_processed_data = True  # 是否启用保存
        self.raw_save_file = None
        self.iir_save_file = None
        self.asr_save_file = None
        self.orica_save_file = None
        self.raw_data_list = []
        self.iir_data_list = []
        self.asr_data_list = []
        self.orica_data_list = []
        self.last_data_time = None  # 最后一次收到数据的时间
        self.data_empty_wait_time = 5  # 数据为空后等待时间（秒）
        self.save_monitor_thread = None  # 监控线程
        
        # 固定长度 chunk 累计缓冲（按时间拼接，后续从中切出等长 chunk）
        self.stream_buffer = None
        self.chunk_size = None


        #当我在切换通道的过程中，会让ic的个数发生改变，但是此时buffer还在运行，会导致卡死，
        #所以我需要把通道切换过程锁住
        self.lock = threading.Lock()
        
        # ✅ 新增：数据更新线程控制
        self.data_update_thread = None
        self.is_running = False
        self.update_interval = 0.1  # 100ms更新间隔




    def find_and_open_stream(self):
        # ========================1) check all available streams===============================
        streams = resolve_streams()
        print("Current available LSL streams:")
        for i, stream in enumerate(streams):
            print(
                f"[{i}] Name: {stream.name()}, Type: {stream.type()}, Channels: {stream.channel_count()}, ID: {stream.source_id()}")
        print(f"Searching for LSL stream with type = '{self.stream_type}'...")

        # ========================2) select the stream by type or name===============================
        '''
        # useing the stream_type to filter the stream, 
        # but sometimes there are serveral streams with the same type, 
        # so we need to use the stream_name to filter the stream

        '''
        #streams = resolve_byprop('type', self.stream_type, timeout=5)
        # using the stream_name to filter the stream
        streams = resolve_byprop('name', self.stream_name, timeout=60)

        if not streams:
            raise RuntimeError(f"No LSL stream with type '{self.stream_type}' and name '{self.stream_name}' found.")


        # ========================3)generate basic info of the stream===============================
        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.channel_manager = ChannelManager(info)
        self.srate = int(info.nominal_srate())
        self.nbchan = info.channel_count()

        # remove some of the useless or broken channels
        #exclude = ['TRIGGER', 'ACC34','ACC33','ACC32', 'Packet Counter', 'ExG 2','ExG 1','ACC','Oz','PO8',"F7","T8",'FC5','F8']#,'F7','F8'
        exclude = ['TRIGGER', 'ACC34','ACC33','ACC32', 'Packet Counter', 'ExG 2','ExG 1','ACC']#,'F7','F8'
        self.chan_labels = self.channel_manager.get_labels_excluding_keywords(exclude)
        self.chan_range = self.channel_manager.get_indices_excluding_keywords(exclude)
        self.nbchan = len(self.chan_range)

        # 4) buffer for visualization
        # self.buffer = np.zeros((info.channel_count(), self.srate * self.time_range))
        # self.raw_buffer = np.zeros((info.channel_count(), self.srate * self.time_range))
        # self.asr_buffer = np.zeros((info.channel_count(), self.srate * self.time_range))
        self.buffer = np.zeros((self.nbchan, self.srate * self.time_range))
        self.raw_buffer = np.zeros((self.nbchan, self.srate * self.time_range))
        self.asr_buffer = np.zeros((self.nbchan, self.srate * self.time_range))
        #buffer for asr calibration
        #self.asr_calibration_data = np.zeros((info.channel_count(), self.srate * 40))
        self.asr_calibration_data = np.zeros((len(self.chan_range), self.srate * 40))
        
        # 初始化10秒校准数据缓冲区
        self.calibration_data = np.zeros((len(self.chan_range), self.srate * self.calibration_duration))

        # 初始化固定长度 chunk 缓冲：例如每 0.1 秒一个 chunk
        self.chunk_size = max(int(self.srate * 1), 1)
        self.stream_buffer = np.empty((info.channel_count(), 0))

        print(f"Stream opened: {info.channel_count()} channels at {self.srate} Hz")
        print(f"Using {self.nbchan} EEG channels: {self.chan_labels}")

        # ========================5) init ORICA===============================
        self.reinitialize_orica()
        
        # ========================6) 如果启用保存，自动初始化保存文件===============================
        if self.save_processed_data and self.orica_save_file is None:
            self.enable_processed_data_saving()


    #input chunk, return modeify chunk and sources , artifact index
    def process_orica(self, chunk, threshold=None):
        """
        use the orica_processor.py to modify the chunk data
        """
        if threshold is None:
            threshold = self.icalabel_threshold
        cleaned_chunk = chunk.copy()
        if self.orica is not None:
            if self.orica.update_buffer(chunk[self.chan_range, :]):
                sources, eog_indices, ic_probs, ic_labels= self.orica.fit(
                    self.orica.data_buffer,
                    self.chan_range,
                    self.chan_labels,
                    self.srate,
                    threshold=float(threshold),
                )
                if sources is not None:
                    cleaned = self.orica.transform(chunk[self.chan_range, :])
                    cleaned_chunk[self.chan_range, :] = cleaned  
                    self.latest_ic_probs = ic_probs
                    self.latest_ic_labels = ic_labels       
                    
        return cleaned_chunk, sources, eog_indices

    def pull_and_update_buffer(self):
        # ========================1) pull data from the stream===============================
        # this samples_random mean the number of samples is random, which depends on the system situation(no control)
        # this timestamps are the time of every random samples
        samples_random, timestamps = self.inlet.pull_chunk(timeout=0)
        #samples, timestamps = self.inlet.pull_chunk(max_samples=100, timeout=0.0)#

        # check if there is data
        if not timestamps:
            # 数据为空，如果启用了保存功能，开始监控等待
            if self.save_processed_data:
                self._handle_data_empty()
            return
        
        # 有数据，更新最后数据时间
        if self.save_processed_data:
            self.last_data_time = time.time()
            
        # convert samples_random to the correct format (channels, samples)
        samples_random = np.array(samples_random).T
        samples=samples_random


        # ========================2) accumulate into fixed-length stream buffer===============================
        if timestamps:
            # new = np.array(samples)  # shape: (samples, channels) or (channels, samples)?
            # new = new if new.ndim == 2 else new.reshape(-1, 1)
            # #new = new.T  # (channels, samples)

            # if self.stream_buffer is None:
            #     self.stream_buffer = new
            # else:
            #     self.stream_buffer = np.concatenate([self.stream_buffer, new], axis=1)

            # # 如果当前缓冲还不够一个 chunk，先返回，等下次再处理
            # if self.stream_buffer is None or self.stream_buffer.shape[1] < (self.chunk_size or 1):
            #     return

            # # 从缓冲最左侧切出一个固定长度 chunk，剩余部分保留到下次
            # chunk = self.stream_buffer[:, :self.chunk_size]
            # self.stream_buffer = self.stream_buffer[:, self.chunk_size:]

            # raw_chunk = chunk.copy()
            # raw_chunk = raw_chunk[self.chan_range, :]






            chunk = np.array(samples)  # shape: (channels, samples)samples size is not a fixed value


            chunk = chunk[self.chan_range, :]






            raw_chunk = chunk.copy()
            raw_chunk = raw_chunk[self.chan_range, :]
            # 每帧初始化四路阶段数据（后续分支会覆盖）
            self.raw_chunk = chunk.copy()
            self.iir_chunk = chunk.copy()
            self.asr_chunk = chunk.copy()
            self.orica_chunk = chunk.copy()


            # # 简单的10秒数据收集
            # if not self.calibration_collected:
            #     self._collect_10s_data(raw_chunk)
            #     return  # 收集期间跳过后续处理

            # Step 1: Apply Common Average Reference (CAR)
            #chunk = self.apply_car_rereference(chunk)
            
            # Step 2: use the professional FIR filter from MNE
            #chunk = self.apply_mne_iir_filter(chunk)








            # # 简单的10秒数据收集
            # if not self.calibration_collected:
            #     self._collect_10s_data(raw_chunk)
            #     return  # 收集期间跳过后续处理

            # Step 1: Apply Common Average Reference (CAR)
            #chunk = self.apply_car_rereference(chunk)

            #Step 2: use the professional FIR filter from MNE
            #chunk = self.apply_mne_iir_filter(chunk)
            
            # Step 2: Apply online IIR filter (状态保持版本，适用于实时流处理)
            # 检查环境变量，决定使用哪种滤波方法（用于对比）
            # import os
            # filter_method = os.environ.get('IIR_FILTER_METHOD', '1')  # 默认使用方法1
            
            # if filter_method == '2':
            #     chunk = self.apply_online_iir_filter1(chunk)
            # else:
            #     #chunk = self.apply_online_iir_filter(chunk)
            #     pass
   





                

            # Step 3: ORICA artifact removal
            # update chunk for visualization before ORICA
            # self.last_unclean_chunk = chunk.copy()

            # # update buffer before ORICA
            # if self.raw_buffer is not None:
            #     self.raw_buffer = np.roll(self.raw_buffer, -chunk.shape[1], axis=1)
            #     self.raw_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk



            filter_method = os.environ.get('IIR_FILTER_METHOD', '1')  # 默认使用方法1
            th_env = os.environ.get("EEG_ICALABEL_THRESHOLD")
            if th_env is not None and str(th_env).strip() != "":
                self.icalabel_threshold = float(th_env)



            
            if filter_method == '1':
                # 仅 IIR：红线 = raw_buffer / last_unclean_chunk（IIR 前）；蓝线 = buffer / last_processed_chunk（IIR 后，在下方统一 roll 写入）
                chunk_pre_iir = chunk.copy()
                chunk = self.apply_online_iir_filter(chunk)
                self.raw_chunk = chunk_pre_iir.copy()
                self.iir_chunk = chunk.copy()
                self.asr_chunk = chunk.copy()
                self.orica_chunk = chunk.copy()
                self.last_unclean_chunk = chunk_pre_iir.copy()
                self.last_processed_chunk = chunk.copy()
                if self.raw_buffer is not None:
                    n = chunk_pre_iir.shape[1]
                    self.raw_buffer = np.roll(self.raw_buffer, -n, axis=1)
                    self.raw_buffer[:, -n:] = chunk_pre_iir
            
            elif filter_method == '2':
                # 本帧入口样本数（LSL chunk 宽）。raw_buffer 与 buffer 必须用同一 n roll，
                # 否则两缓冲推进的列数不一致，会越滚越错位；“卡一下”常出现在 ASR 首次输出或
                # asr_out 长度 < 本帧 chunk 时，原先先写了 raw 再改短 chunk，蓝线比红线少推几列。
                n_in = int(chunk.shape[1])

                chunk_pre_iir = chunk.copy()
                chunk = self.apply_online_iir_filter(chunk)
                chunk_after_iir = chunk.copy()
                self.raw_chunk = chunk_pre_iir.copy()
                self.iir_chunk = chunk_after_iir.copy()

                if self.asr_filter is None:
                    self.initialize_asr_from_mat(cutoff=5)

                if self.asr_filter is not None:
                    print("using asr")
                    target_len = int(self.srate * 0.5) if self.srate is not None else n_in

                    if self.asr_accum_buffer is None:
                        self.asr_accum_buffer = chunk.copy()
                    else:
                        self.asr_accum_buffer = np.concatenate(
                            [self.asr_accum_buffer, chunk], axis=1
                        )

                    if self.asr_accum_buffer.shape[1] >= target_len:
                        asr_out = self._apply_asr_accumulated_numpy(self.asr_accum_buffer)

                        if asr_out.shape[1] >= n_in:
                            chunk = asr_out[:, -n_in:]
                        else:
                            pad = n_in - asr_out.shape[1]
                            chunk = np.pad(asr_out, ((0, 0), (pad, 0)), mode="edge")

                        if asr_out.shape[1] > target_len:
                            self.asr_accum_buffer = asr_out[:, -target_len:]
                        else:
                            self.asr_accum_buffer = asr_out.copy()
                    else:
                        chunk = chunk_after_iir.copy()

                if chunk.shape[1] != n_in:
                    if chunk.shape[1] > n_in:
                        chunk = chunk[:, -n_in:]
                    else:
                        chunk = np.pad(
                            chunk,
                            ((0, 0), (n_in - chunk.shape[1], 0)),
                            mode="edge",
                        )

                self.last_unclean_chunk = chunk_after_iir.copy()
                self.last_processed_chunk = chunk.copy()
                self.asr_chunk = chunk.copy()
                self.orica_chunk = chunk.copy()
                if self.raw_buffer is not None:
                    self.raw_buffer = np.roll(self.raw_buffer, -n_in, axis=1)
                    self.raw_buffer[:, -n_in:] = chunk_after_iir



            elif filter_method == '3':
                # 本帧入口样本数（LSL chunk 宽）。raw_buffer 与 buffer 必须用同一 n roll，
                # 否则两缓冲推进的列数不一致，会越滚越错位；“卡一下”常出现在 ASR 首次输出或
                # asr_out 长度 < 本帧 chunk 时，原先先写了 raw 再改短 chunk，蓝线比红线少推几列。
                n_in = int(chunk.shape[1])

                chunk_pre_iir = chunk.copy()
                chunk = self.apply_online_iir_filter(chunk)
                chunk_after_iir = chunk.copy()
                self.raw_chunk = chunk_pre_iir.copy()
                self.iir_chunk = chunk_after_iir.copy()

                self.last_unclean_chunk = chunk.copy()
                # update buffer before ORICA
                if self.raw_buffer is not None:
                    self.raw_buffer = np.roll(self.raw_buffer, -chunk.shape[1], axis=1)
                    self.raw_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk


                if self.asr_filter is None:
                    self.initialize_asr_from_mat(cutoff=10)

                if self.asr_filter is not None:
                    print("using asr")
                    target_len = int(self.srate * 0.5) if self.srate is not None else n_in

                    if self.asr_accum_buffer is None:
                        self.asr_accum_buffer = chunk.copy()
                    else:
                        self.asr_accum_buffer = np.concatenate(
                            [self.asr_accum_buffer, chunk], axis=1
                        )

                    if self.asr_accum_buffer.shape[1] >= target_len:
                        asr_out = self._apply_asr_accumulated_numpy(self.asr_accum_buffer)

                        if asr_out.shape[1] >= n_in:
                            chunk = asr_out[:, -n_in:]
                        else:
                            pad = n_in - asr_out.shape[1]
                            chunk = np.pad(asr_out, ((0, 0), (pad, 0)), mode="edge")

                        if asr_out.shape[1] > target_len:
                            self.asr_accum_buffer = asr_out[:, -target_len:]
                        else:
                            self.asr_accum_buffer = asr_out.copy()
                    else:
                        chunk = chunk_after_iir.copy()


                if chunk.shape[1] != n_in:
                    if chunk.shape[1] > n_in:
                        chunk = chunk[:, -n_in:]
                    else:
                        chunk = np.pad(
                            chunk,
                            ((0, 0), (n_in - chunk.shape[1], 0)),
                            mode="edge",
                        )




                # self.last_unclean_chunk = chunk.copy()
                # # update buffer before ORICA
                # if self.raw_buffer is not None:
                #     self.raw_buffer = np.roll(self.raw_buffer, -chunk.shape[1], axis=1)
                #     self.raw_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk
                        
                print("Before",chunk.shape)
                # # ORICA processing
                # chunk, ica_sources, eog_indices = self.process_orica(chunk)
                # if ica_sources is not None:
                #     self.latest_sources = ica_sources
                # if eog_indices is not None:
                #     self.latest_eog_indices = eog_indices
                print("After",chunk.shape)


                    

                # if chunk.shape[1] != n_in:
                #     if chunk.shape[1] > n_in:
                #         chunk = chunk[:, -n_in:]
                #     else:
                #         chunk = np.pad(
                #             chunk,
                #             ((0, 0), (n_in - chunk.shape[1], 0)),
                #             mode="edge",
                #         )

                #self.last_unclean_chunk = chunk_after_iir.copy()
                self.last_processed_chunk = chunk.copy()
                self.asr_chunk = chunk.copy()
                self.orica_chunk = chunk.copy()
                # if self.raw_buffer is not None:
                #     self.raw_buffer = np.roll(self.raw_buffer, -n_in, axis=1)
                #     self.raw_buffer[:, -n_in:] = chunk_after_iir



            elif filter_method == '4':
                # 本帧入口样本数（LSL chunk 宽）。raw_buffer 与 buffer 必须用同一 n roll，
                # 否则两缓冲推进的列数不一致，会越滚越错位；“卡一下”常出现在 ASR 首次输出或
                # asr_out 长度 < 本帧 chunk 时，原先先写了 raw 再改短 chunk，蓝线比红线少推几列。
                print("1"*100)
                n_in = int(chunk.shape[1])

                print("2"*100)

                self.raw_chunk=chunk.copy()

                chunk = self.apply_online_iir_filter(chunk)
                chunk_after_iir = chunk.copy()

                self.iir_chunk=chunk.copy()


                print("3"*100)
                # update buffer after ORICA
                num_new0 = chunk.shape[1]
                self.raw_buffer = np.roll(self.raw_buffer, -num_new0, axis=1)
                self.raw_buffer[:, -num_new0:] = chunk



                print("x"*100)
                if self.asr_filter is None:
                    # 可由启动脚本按实例注入：
                    # EEG_ASR_CALIB_NPZ / EEG_ASR_CUTOFF
                    asr_npz_path = os.environ.get("EEG_ASR_CALIB_NPZ")
                    if not asr_npz_path:
                        raise ValueError("缺少环境变量 EEG_ASR_CALIB_NPZ。")
                    asr_cutoff_env = os.environ.get("EEG_ASR_CUTOFF")
                    if not asr_cutoff_env:
                        raise ValueError("缺少环境变量 EEG_ASR_CUTOFF。")
                    asr_cutoff = float(asr_cutoff_env)
                    print("c"*100)
                    print("asr_npz_path",asr_npz_path)
                    print("asr_cutoff",asr_cutoff)
                    print("c"*100)
                    self.initialize_asr_from_npz_1(
                        npz_file_path=asr_npz_path,
                        cutoff=asr_cutoff,
                    )
                    #self.initialize_asr_from_mat_1(mat_file_path=r"D:\work\Python_Project\ORICA\temp_txt\cleaned_data_quick30.mat",cutoff=10)
                    #self.initialize_asr_from_npz_1(npz_file_path=r"D:\work\Python_Project\ORICA\temp_txt\cleaned_data_quick30.npz",cutoff=10)

                if self.asr_filter is not None:
                    print("using asr")
                    target_len = int(self.srate * 0.5) if self.srate is not None else n_in


                    if self.asr_accum_buffer is None:
                        self.asr_accum_buffer = chunk.copy()
                    else:
                        self.asr_accum_buffer = np.concatenate(
                            [self.asr_accum_buffer, chunk], axis=1
                        )
                    print("self.asr_accum_buffer.shape[1]",self.asr_accum_buffer.shape)
                    if self.asr_accum_buffer.shape[1] >= target_len:
                        print("6"*100)
                        asr_out = self._apply_asr_accumulated_numpy(self.asr_accum_buffer)
                        print("7"*100)
                        if asr_out.shape[1] >= n_in:
                            chunk = asr_out[:, -n_in:]
                        else:
                            pad = n_in - asr_out.shape[1]
                            chunk = np.pad(asr_out, ((0, 0), (pad, 0)), mode="edge")
                        print("8"*100)
                        if asr_out.shape[1] > target_len:
                            self.asr_accum_buffer = asr_out[:, -target_len:]
                        else:
                            self.asr_accum_buffer = asr_out.copy()
                    else:
                        chunk = chunk_after_iir.copy()

                print("4x"*100)
                if chunk.shape[1] != n_in:
                    if chunk.shape[1] > n_in:
                        chunk = chunk[:, -n_in:]
                    else:
                        chunk = np.pad(
                            chunk,
                            ((0, 0), (n_in - chunk.shape[1], 0)),
                            mode="edge",
                        )

                self.asr_chunk=chunk.copy()


                self.last_unclean_chunk = chunk.copy()
                # update buffer before ORICA
                if self.asr_buffer is not None:
                    self.asr_buffer = np.roll(self.asr_buffer, -chunk.shape[1], axis=1)
                    self.asr_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk

                print("Before",chunk.shape)

                # ORICA processing
                chunk, ica_sources, eog_indices = self.process_orica(
                    chunk,
                    threshold=self.icalabel_threshold,
                )
                if ica_sources is not None:
                    self.latest_sources = ica_sources
                if eog_indices is not None:
                    self.latest_eog_indices = eog_indices

                print("After",chunk.shape)
                
                self.orica_chunk=chunk.copy()

                    

                # if chunk.shape[1] != n_in:
                #     if chunk.shape[1] > n_in:
                #         chunk = chunk[:, -n_in:]
                #     else:
                #         chunk = np.pad(
                #             chunk,
                #             ((0, 0), (n_in - chunk.shape[1], 0)),
                #             mode="edge",
                #         )

                #self.last_unclean_chunk = chunk_after_iir.copy()
                self.last_processed_chunk = chunk.copy()
                # if self.raw_buffer is not None:
                #     self.raw_buffer = np.roll(self.raw_buffer, -n_in, axis=1)
                #     self.raw_buffer[:, -n_in:] = chunk_after_iir



            elif filter_method == '41':
                # 本帧入口样本数（LSL chunk 宽）。raw_buffer 与 buffer 必须用同一 n roll，
                # 否则两缓冲推进的列数不一致，会越滚越错位；“卡一下”常出现在 ASR 首次输出或
                # asr_out 长度 < 本帧 chunk 时，原先先写了 raw 再改短 chunk，蓝线比红线少推几列。
                n_in = int(chunk.shape[1])

                self.raw_chunk=chunk.copy()

                chunk = self.apply_online_iir_filter(chunk)
                chunk_after_iir = chunk.copy()

                self.iir_chunk=chunk.copy()



                # update buffer after ORICA
                num_new0 = chunk.shape[1]
                self.raw_buffer = np.roll(self.raw_buffer, -num_new0, axis=1)
                self.raw_buffer[:, -num_new0:] = chunk




                if self.asr_filter is None:
                    # 可由启动脚本按实例注入：
                    # EEG_ASR_CALIB_NPZ / EEG_ASR_CUTOFF
                    asr_npz_path = os.environ.get("EEG_ASR_CALIB_NPZ")
                    if not asr_npz_path:
                        raise ValueError("缺少环境变量 EEG_ASR_CALIB_NPZ。")
                    asr_cutoff_env = os.environ.get("EEG_ASR_CUTOFF")
                    if not asr_cutoff_env:
                        raise ValueError("缺少环境变量 EEG_ASR_CUTOFF。")
                    asr_cutoff = float(asr_cutoff_env)
                    self.initialize_asr_from_npz_1(
                        npz_file_path=asr_npz_path,
                        cutoff=asr_cutoff,
                    )
                    #self.initialize_asr_from_npz_1(npz_file_path=r"D:\work\Python_Project\ORICA\Quick30_run\artifact_removal_verify\set_npz\npz_data\cali\laparoscopic_1311_EEGmerged_iir_filtered.npz",cutoff=10)

                if self.asr_filter is not None:
                    print("using asr")
                    target_len = int(self.srate * 0.5) if self.srate is not None else n_in

                    if self.asr_accum_buffer is None:
                        self.asr_accum_buffer = chunk.copy()
                    else:
                        self.asr_accum_buffer = np.concatenate(
                            [self.asr_accum_buffer, chunk], axis=1
                        )

                    if self.asr_accum_buffer.shape[1] >= target_len:
                        asr_out = self._apply_asr_accumulated_numpy(self.asr_accum_buffer)

                        if asr_out.shape[1] >= n_in:
                            chunk = asr_out[:, -n_in:]
                        else:
                            pad = n_in - asr_out.shape[1]
                            chunk = np.pad(asr_out, ((0, 0), (pad, 0)), mode="edge")

                        if asr_out.shape[1] > target_len:
                            self.asr_accum_buffer = asr_out[:, -target_len:]
                        else:
                            self.asr_accum_buffer = asr_out.copy()
                    else:
                        chunk = chunk_after_iir.copy()


                if chunk.shape[1] != n_in:
                    if chunk.shape[1] > n_in:
                        chunk = chunk[:, -n_in:]
                    else:
                        chunk = np.pad(
                            chunk,
                            ((0, 0), (n_in - chunk.shape[1], 0)),
                            mode="edge",
                        )

                self.asr_chunk=chunk.copy()


                self.last_unclean_chunk = chunk.copy()
                # update buffer before ORICA
                if self.asr_buffer is not None:
                    self.asr_buffer = np.roll(self.asr_buffer, -chunk.shape[1], axis=1)
                    self.asr_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk

                print("Before",chunk.shape)
                        
                # ORICA processing
                chunk, ica_sources, eog_indices = self.process_orica(
                    chunk,
                    threshold=self.icalabel_threshold,
                )
                if ica_sources is not None:
                    self.latest_sources = ica_sources
                if eog_indices is not None:
                    self.latest_eog_indices = eog_indices

                print("After",chunk.shape)
                
                self.orica_chunk=chunk.copy()

                    

                # if chunk.shape[1] != n_in:
                #     if chunk.shape[1] > n_in:
                #         chunk = chunk[:, -n_in:]
                #     else:
                #         chunk = np.pad(
                #             chunk,
                #             ((0, 0), (n_in - chunk.shape[1], 0)),
                #             mode="edge",
                #         )

                #self.last_unclean_chunk = chunk_after_iir.copy()
                self.last_processed_chunk = chunk.copy()
                # if self.raw_buffer is not None:
                #     self.raw_buffer = np.roll(self.raw_buffer, -n_in, axis=1)
                #     self.raw_buffer[:, -n_in:] = chunk_after_iir




            elif filter_method == '5xw':
                # 本帧入口样本数（LSL chunk 宽）。raw_buffer 与 buffer 必须用同一 n roll，
                # 否则两缓冲推进的列数不一致，会越滚越错位；“卡一下”常出现在 ASR 首次输出或
                # asr_out 长度 < 本帧 chunk 时，原先先写了 raw 再改短 chunk，蓝线比红线少推几列。
                n_in = int(chunk.shape[1])

                chunk_pre_iir = chunk.copy()
                chunk = self.apply_online_iir_filter(chunk)
                chunk_after_iir = chunk.copy()
                self.raw_chunk = chunk_pre_iir.copy()
                self.iir_chunk = chunk_after_iir.copy()



                # update buffer after ORICA
                num_new0 = chunk.shape[1]
                self.raw_buffer = np.roll(self.raw_buffer, -num_new0, axis=1)
                self.raw_buffer[:, -num_new0:] = chunk




                if self.asr_filter is None:
                    self.initialize_asr_from_npz_1(cutoff=5)

                if self.asr_filter is not None:
                    print("using asr")
                    target_len = int(self.srate * 0.5) if self.srate is not None else n_in

                    if self.asr_accum_buffer is None:
                        self.asr_accum_buffer = chunk.copy()
                    else:
                        self.asr_accum_buffer = np.concatenate(
                            [self.asr_accum_buffer, chunk], axis=1
                        )

                    if self.asr_accum_buffer.shape[1] >= target_len:
                        asr_out = self._apply_asr_accumulated_numpy(self.asr_accum_buffer)

                        if asr_out.shape[1] >= n_in:
                            chunk = asr_out[:, -n_in:]
                        else:
                            pad = n_in - asr_out.shape[1]
                            chunk = np.pad(asr_out, ((0, 0), (pad, 0)), mode="edge")

                        if asr_out.shape[1] > target_len:
                            self.asr_accum_buffer = asr_out[:, -target_len:]
                        else:
                            self.asr_accum_buffer = asr_out.copy()
                    else:
                        chunk = chunk_after_iir.copy()


                if chunk.shape[1] != n_in:
                    if chunk.shape[1] > n_in:
                        chunk = chunk[:, -n_in:]
                    else:
                        chunk = np.pad(
                            chunk,
                            ((0, 0), (n_in - chunk.shape[1], 0)),
                            mode="edge",
                        )




                self.asr_chunk = chunk.copy()
                self.last_unclean_chunk = chunk.copy()
                # update buffer before ORICA
                if self.asr_buffer is not None:
                    self.asr_buffer = np.roll(self.asr_buffer, -chunk.shape[1], axis=1)
                    self.asr_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk

                print("Before",chunk.shape)
                        
                # ORICA processing
                chunk, ica_sources, eog_indices = self.process_orica(
                    chunk,
                    threshold=self.icalabel_threshold,
                )
                if ica_sources is not None:
                    self.latest_sources = ica_sources
                if eog_indices is not None:
                    self.latest_eog_indices = eog_indices

                print("After",chunk.shape)
                self.orica_chunk = chunk.copy()


                    

                # if chunk.shape[1] != n_in:
                #     if chunk.shape[1] > n_in:
                #         chunk = chunk[:, -n_in:]
                #     else:
                #         chunk = np.pad(
                #             chunk,
                #             ((0, 0), (n_in - chunk.shape[1], 0)),
                #             mode="edge",
                #         )

                #self.last_unclean_chunk = chunk_after_iir.copy()
                self.last_processed_chunk = chunk.copy()
                # if self.raw_buffer is not None:
                #     self.raw_buffer = np.roll(self.raw_buffer, -n_in, axis=1)
                #     self.raw_buffer[:, -n_in:] = chunk_after_iir
















            

            # #step 2: ASR with offline calibration
            # #asr calibration
            # if self.asr_filter is None:
            #     #self.initialize_asr_from_mat()
            #     self.initialize_asr_from_npz()

            # # asr usage：先在时间上累计到约0.5秒，再统一做一次ASR处理
            # if self.asr_filter is not None:
            #     print("using asr")
            #     # 目标累计长度：约0.5秒
            #     target_len = int(self.srate * 0.5) if self.srate is not None else chunk.shape[1]

            #     # 初始化/更新ASR累积缓冲区（在时间维度拼接）
            #     if self.asr_accum_buffer is None:
            #         self.asr_accum_buffer = chunk.copy()
            #     else:
            #         self.asr_accum_buffer = np.concatenate(
            #             [self.asr_accum_buffer, chunk], axis=1
            #         )

            #     # 当累计长度达到0.5秒时，再统一进行ASR处理
            #     if self.asr_accum_buffer.shape[1] >= target_len:
            #         # 对累积缓冲区整体做一次ASR，以利用前后文
            #         asr_out = self.asr_filter.transform(self.asr_accum_buffer)

            #         # 当前这一帧输出：取最后一个chunk长度对应的部分
            #         # 这样保证对外的chunk长度不变，但ASR使用了更长的上下文
            #         if asr_out.shape[1] >= chunk.shape[1]:
            #             chunk = asr_out[:, -chunk.shape[1]:]
            #         else:
            #             # 理论上不会发生，仅作保护
            #             chunk = asr_out

            #         # 为防止缓冲区无限增长，只保留最后 target_len 作为上下文
            #         if asr_out.shape[1] > target_len:
            #             self.asr_accum_buffer = asr_out[:, -target_len:]
            #         else:
            #             self.asr_accum_buffer = asr_out.copy()
            #     else:
            #         # 累计长度尚不足0.5秒：先不做ASR，直接沿用 IIR 结果
            #         # （此时 chunk 保持为 IIR 之后的信号）
            #         pass

            




            # # ORICA processing
            # chunk, ica_sources, eog_indices = self.process_orica(chunk)
            # if ica_sources is not None:
            #     self.latest_sources = ica_sources
            # if eog_indices is not None:
            #     self.latest_eog_indices = eog_indices

            # #ASR with online calibration
            # if not self.use_asr:
            #     # 如果正在进行在线校准，收集数据
            #     if self.asr_calibration_size < self.srate * 20:
            #         print("collecting asr data")
            #         if self.asr_calibration_data is not None:
            #             self.asr_calibration_data = np.roll(self.asr_calibration_data, -raw_chunk.shape[1], axis=1)
            #             self.asr_calibration_data[:, -raw_chunk.shape[1]:] = raw_chunk
            #             self.asr_calibration_size += raw_chunk.shape[1]
            #     # 如果ASR已校准且启用，则使用ASR处理
            #     elif self.asr_calibration_size >= self.srate * 40:
            #         print("init asr")
            #         self.initialize_asr_online(self.asr_calibration_data)
            #         self.use_asr = True
            # elif self.use_asr:
            #     print("using asr")
            #     chunk[self.chan_range, :] = self.asr_filter.transform(chunk[self.chan_range, :])


            # # update chunk for visualization after ORICA
            # self.last_processed_chunk = chunk.copy()
            
            # 保存处理后的数据到文件（每次追加）
            if self.save_processed_data:
                self._append_processed_data_to_list(chunk)
            
            # update buffer after ORICA
            num_new = chunk.shape[1]
            self.buffer = np.roll(self.buffer, -num_new, axis=1)
            self.buffer[:, -num_new:] = chunk


            #Step 4: update chunk pairs for visualization
            # this step is used for keeping the vis looks synced
            # becasue the ORICA takes some time which would cause the data before and after ORICA looks not synced
            timestamp = time.time()
            self.chunk_pairs.append((timestamp, self.last_unclean_chunk, self.last_processed_chunk))
            if len(self.chunk_pairs) > 1:
                self.chunk_pairs.pop(0)
            self.pair_buffer = (self.raw_buffer,self.asr_buffer,self.buffer )



    


    #Selected channel indices: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    #Selected channel labels: ['AF7', 'Fpz', 'F7', 'Fz', 'T7', 'FC6', 'F4', 'C4', 'Oz', 'CP6', 'Cz', 'PO8', 'CP5', 'O2', 'O1', 'P3', 'P4', 'P7', 'P8', 'Pz', 'PO7', 'T8', 'C3', 'Fp2', 'F3', 'F8', 'FC5', 'AF8']
    #上面就是channel_range和channel_labels的格式，要调用函数就传入这样的list
    #channel_selector.py use this function to update the channel range and labels
    def set_channel_range_and_labels(self, new_range, new_labels):
        with self.lock:
            self.chan_range = new_range
            self.chan_labels = new_labels
            self.nbchan = len(new_range)
            self.reinitialize_orica()
            print(f"channel range and labels updated: {self.chan_labels}")


    def reinitialize_orica(self):
        self.orica = ORICAProcessor(
            n_components=len(self.chan_range),
            srate=self.srate
        )
        print("🔁 ORICA processor re-initialized with new channel range.")
    
    def apply_car_rereference(self, data):
        """
        应用平均参考 (Common Average Reference)
        
        Args:
            data: EEG数据 (channels, samples)
        
        Returns:
            重参考后的数据 (channels, samples) - 通道数不变
        """
        try:
            # 计算所有通道的平均值作为参考
            ref_signal = np.mean(data, axis=0, keepdims=True)
            # 每个通道减去参考信号
            reref_data = data - ref_signal
            print(f"✅ CAR重参考完成: {data.shape[0]} 通道")
            return reref_data
            
        except Exception as e:
            print(f"❌ CAR重参考失败: {e}")
            print("⚠️ 返回原始数据")
            return data

    def apply_mne_iir_filter(self, data):
        """
        应用 IIR 滤波（原始MNE版本，使用零相位滤波filtfilt）
        适用于离线批处理，不适用于在线实时处理
        """
        try:     
            filtered_data = filter_data(
                data=data,
                sfreq=self.srate,
                l_freq=self.cutoff[0],      # low frequency cutoff
                h_freq=self.cutoff[1],      # high frequency cutoff
                # l_freq=15,
                # h_freq=30,
                
                method='iir',               # apply IIR filter
                iir_params={'order': 4, 'ftype': 'butter'},  # 4th order Butterworth
                verbose=False
            )
            print(f"✅ MNE IIR filter finished: {self.cutoff[0]}-{self.cutoff[1]} Hz")
            return filtered_data
        except Exception as e:
            print(f"❌ MNE IIR filter failed: {e}")
            print("⚠️ revert to original data")
            return data

    def _initialize_online_iir_filter(self, n_channels):
        """
        初始化在线IIR滤波器（只在第一次或通道数变化时调用）
        
        Args:
            n_channels: 通道数
        
        Returns:
            bool: 初始化是否成功
        """
        if self.srate is None:
            print("⚠️ 采样率未设置，无法初始化IIR滤波器")
            return False
        
        try:
            # 设计4阶Butterworth带通滤波器
            nyq = 0.5 * self.srate
            low = self.cutoff[0] / nyq
            high = self.cutoff[1] / nyq
            
            # 设计滤波器系数
            b, a = butter(4, [low, high], btype='band')
            
            self.iir_filter_b = b
            self.iir_filter_a = a
            
            # 为每个通道初始化滤波器状态
            # lfilter_zi 返回单个通道的初始状态，形状为 (filter_order,)
            zi_single = lfilter_zi(b, a)
            
            # 为所有通道创建状态矩阵，形状为 (filter_order, n_channels)
            self.iir_filter_zi = np.tile(zi_single, (n_channels, 1)).T
            
            print(f"✅ 在线IIR滤波器初始化: {self.cutoff[0]}-{self.cutoff[1]} Hz, {n_channels} 通道")
            return True
        except Exception as e:
            print(f"❌ IIR滤波器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def apply_online_iir_filter(self, data):
        """
        应用在线IIR滤波器（状态保持版本，适用于实时流处理）
        
        这个方法使用 scipy.signal.lfilter 进行单向滤波，并在chunk之间保持滤波器状态。
        与 MNE 的 filter_data（零相位滤波）不同，这个方法适合在线实时处理。
        
        Args:
            data: EEG数据 (channels, samples)
        
        Returns:
            滤波后的数据 (channels, samples)
        """
        if data is None or data.size == 0:
            return data
        
        try:
            n_channels, n_samples = data.shape
            
            # 检查是否需要重新初始化滤波器（第一次调用或通道数变化）
            if (self.iir_filter_b is None or 
                self.iir_filter_a is None or 
                self.iir_filter_zi is None or 
                self.iir_filter_zi.shape[1] != n_channels):
                if not self._initialize_online_iir_filter(n_channels):
                    # 初始化失败，回退到原始数据
                    return data
            
            # 检查数据长度是否足够（至少需要滤波器阶数）
            filter_order = len(self.iir_filter_a) - 1
            if n_samples < filter_order:
                # 数据太短，使用MNE的filter_data作为后备（会有边界效应）
                print(f"⚠️ Chunk太小({n_samples} < {filter_order})，使用MNE filter_data作为后备")
                filtered_data = filter_data(
                    data=data,
                    sfreq=self.srate,
                    l_freq=self.cutoff[0],
                    h_freq=self.cutoff[1],
                    method='iir',
                    iir_params={'order': 4, 'ftype': 'butter'},
                    verbose=False
                )
                return filtered_data
            
            # 在线滤波：使用保持的状态
            filtered_data_list = []
            final_zi_list = []
            
            for ch in range(n_channels):
                channel_data = data[ch, :]
                # 使用当前通道的滤波器状态
                zi_ch = self.iir_filter_zi[:, ch]
                
                # 应用滤波器，并获取新的状态
                filtered_ch, final_zi_ch = lfilter(
                    self.iir_filter_b, 
                    self.iir_filter_a, 
                    channel_data, 
                    zi=zi_ch
                )
                
                filtered_data_list.append(filtered_ch)
                final_zi_list.append(final_zi_ch)
            
            # 更新滤波器状态（为下一次chunk准备）
            # final_zi_list 的每个元素形状为 (filter_order,)
            # 转置后得到 (filter_order, n_channels)
            self.iir_filter_zi = np.array(final_zi_list).T
            
            return np.array(filtered_data_list)
            
        except Exception as e:
            print(f"❌ 在线IIR滤波失败: {e}")
            import traceback
            traceback.print_exc()
            return data




    import numpy as np
    from scipy.signal import butter, lfilter, lfilter_zi

    def _initialize_online_iir_filter1(self, n_channels: int) -> bool:
        if self.srate is None:
            print("⚠️ 采样率未设置，无法初始化IIR滤波器")
            return False
        try:
            l_freq, h_freq = float(self.cutoff[0]), float(self.cutoff[1])

            # 用 fs= 更稳、更不容易写错
            b, a = butter(
                N=4,
                Wn=[l_freq, h_freq],
                btype="bandpass",
                fs=float(self.srate),
            )

            self.iir_filter_b = b
            self.iir_filter_a = a

            # 先不创建 zi（因为需要 x0 缩放），等第一块数据进来再初始化
            self.iir_filter_zi = None

            print(f"✅ 在线IIR滤波器初始化: {l_freq}-{h_freq} Hz, {n_channels} 通道")
            return True
        except Exception as e:
            print(f"❌ IIR滤波器初始化失败: {e}")
            import traceback; traceback.print_exc()
            self.iir_filter_b, self.iir_filter_a, self.iir_filter_zi = None, None, None
            return False


    def apply_online_iir_filter1(self, data: np.ndarray):
        if data is None:
            return data
        data = np.asarray(data)
        if data.size == 0:
            return data

        try:
            if data.ndim != 2:
                raise ValueError(f"data must be (n_channels, n_samples), got {data.shape}")

            n_channels, n_samples = data.shape
            if n_samples == 0:
                return data

            # 初始化 b/a
            if self.iir_filter_b is None or self.iir_filter_a is None:
                if not self._initialize_online_iir_filter(n_channels):
                    return data

            # 如果通道数变了：重置 zi（让下一步重新用 x0 初始化）
            if self.iir_filter_zi is not None and self.iir_filter_zi.shape[0] != n_channels:
                self.iir_filter_zi = None

            # order 更严谨
            order = max(len(self.iir_filter_a), len(self.iir_filter_b)) - 1

            # 首次/重置后：用 lfilter_zi * x0 初始化每个通道的状态
            if self.iir_filter_zi is None:
                zi1 = lfilter_zi(self.iir_filter_b, self.iir_filter_a)  # (order,)
                x0 = data[:, 0].astype(np.float64, copy=False)          # (n_channels,)
                self.iir_filter_zi = zi1[None, :] * x0[:, None]         # (n_channels, order)

            # 用 float64 做滤波更稳
            x = data.astype(np.float64, copy=False)

            # 一次滤所有通道（axis=1 是时间）
            y, zf = lfilter(
                self.iir_filter_b,
                self.iir_filter_a,
                x,
                axis=1,
                zi=self.iir_filter_zi
            )
            self.iir_filter_zi = zf

            return y.astype(data.dtype, copy=False)

        except Exception as e:
            print(f"❌ 在线IIR滤波失败: {e}")
            import traceback; traceback.print_exc()
            return data



    def enable_processed_data_saving(self, save_file=None):
        """
        启用处理后数据保存功能
        
        Args:
            save_file: 处理后数据保存文件路径，如果为None则自动生成
        """
        self.save_processed_data = True
        self.raw_data_list = []
        self.iir_data_list = []
        self.asr_data_list = []
        self.orica_data_list = []
        self.last_data_time = None
        import os
        # 支持通过环境变量为多实例分别配置保存标签/目录
        # 例如：
        #   EEG_SAVE_FILE_TAG=b84
        #   EEG_SAVE_DIR=processed_data_saves_threshold/asr20_2min_70/runA
        file_tag = os.environ.get("EEG_SAVE_FILE_TAG")
        if not file_tag:
            raise ValueError(
                "缺少环境变量 EEG_SAVE_FILE_TAG（例如 b84a / b84b）。"
            )
        
        if save_file is None:
            from datetime import datetime
            from pathlib import Path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir_env = os.environ.get("EEG_SAVE_DIR")
            if not save_dir_env:
                raise ValueError(
                    "缺少环境变量 EEG_SAVE_DIR（例如 processed_data_saves_threshold/asr20_2min_70/runA）。"
                )
            save_dir = Path(__file__).parent / save_dir_env
            save_dir.mkdir(parents=True, exist_ok=True)
            # 生成四个阶段文件路径
            raw_file = save_dir / f"{file_tag}eeg_raw1.npz"
            iir_file = save_dir / f"{file_tag}eeg_iir1.npz"
            asr_file = save_dir / f"{file_tag}eeg_asr1.npz"
            orica_file = save_dir / f"{file_tag}eeg_orica1.npz"
            '''
              iclablethreshold   chunk(timeout=1)
            x 0.8     0
            y 0.8     1
            z 0.9     1
            a 0.9     0
            b 0.7     0
            '''
        else:
            # 如果提供了文件路径，基于它生成两个文件路径
            save_path = Path(save_file)
            save_dir = save_path.parent
            raw_file = save_dir / f"{file_tag}eeg_raw1.npz"
            iir_file = save_dir / f"{file_tag}eeg_iir1.npz"
            asr_file = save_dir / f"{file_tag}eeg_asr1.npz"
            orica_file = save_dir / f"{file_tag}eeg_orica1.npz"
        
        self.raw_save_file = Path(raw_file)
        self.iir_save_file = Path(iir_file)
        self.asr_save_file = Path(asr_file)
        self.orica_save_file = Path(orica_file)
        print(f"✅ 已启用数据保存")
        print(f"   file_tag: {file_tag}")
        print(f"   Raw文件: {self.raw_save_file}")
        print(f"   IIR文件: {self.iir_save_file}")
        print(f"   ASR文件: {self.asr_save_file}")
        print(f"   ORICA文件: {self.orica_save_file}")
        print(f"   数据为空后将等待 {self.data_empty_wait_time} 秒后完成保存")
    
    def disable_processed_data_saving(self):
        """禁用处理后数据保存功能"""
        if self.save_processed_data and len(self.orica_data_list) > 0:
            print("💾 保存已累积的数据...")
            self._save_processed_data_to_file()
        self.save_processed_data = False
        self.raw_data_list = []
        self.iir_data_list = []
        self.asr_data_list = []
        self.orica_data_list = []
        print("🛑 已禁用处理后数据保存")
    
    def _select_channels_for_save(self, chunk):
        """统一将 chunk 转为选中通道 (len(chan_range), samples)。"""
        if chunk is None:
            return None
        if chunk.shape[0] == len(self.chan_range):
            return chunk.copy()
        return chunk[self.chan_range, :].copy()

    def _append_processed_data_to_list(self, _processed_chunk_unused=None):
        """将 raw/iir/asr/orica 四个阶段 chunk 追加到内存列表。"""
        raw_to_save = self._select_channels_for_save(self.raw_chunk)
        iir_to_save = self._select_channels_for_save(self.iir_chunk)
        asr_to_save = self._select_channels_for_save(self.asr_chunk)
        orica_to_save = self._select_channels_for_save(self.orica_chunk)

        # 若某阶段缺失，自动回退到前一阶段，保证四路时长一致
        if raw_to_save is None:
            return
        if iir_to_save is None:
            iir_to_save = raw_to_save.copy()
        if asr_to_save is None:
            asr_to_save = iir_to_save.copy()
        if orica_to_save is None:
            orica_to_save = asr_to_save.copy()

        self.raw_data_list.append(raw_to_save)
        self.iir_data_list.append(iir_to_save)
        self.asr_data_list.append(asr_to_save)
        self.orica_data_list.append(orica_to_save)
    
    def _handle_data_empty(self):
        """
        处理数据为空的情况：启动监控线程，等待10秒后保存文件
        """
        # 如果监控线程已经在运行，不重复启动
        if self.save_monitor_thread is not None and self.save_monitor_thread.is_alive():
            return
        
        # 启动监控线程
        self.save_monitor_thread = threading.Thread(
            target=self._monitor_and_save_data,
            daemon=True
        )
        self.save_monitor_thread.start()
    
    def _monitor_and_save_data(self):
        """
        监控线程：等待10秒，如果期间没有新数据，则保存文件
        """
        empty_start_time = time.time()
        print(f"📊 检测到数据为空，等待 {self.data_empty_wait_time} 秒确认数据流结束...")
        
        # 等待10秒，期间检查是否有新数据
        while True:
            time.sleep(0.5)  # 每0.5秒检查一次
            elapsed = time.time() - empty_start_time
            
            # 检查是否有新数据到达
            if self.last_data_time is not None:
                time_since_last_data = time.time() - self.last_data_time
                if time_since_last_data < 0.5:  # 0.5秒内有新数据
                    print(f"   检测到新数据，继续接收... (已等待 {elapsed:.1f}秒)")
                    return  # 有新数据，停止监控
            
            # 等待时间到
            if elapsed >= self.data_empty_wait_time:
                print(f"✅ 已等待 {self.data_empty_wait_time} 秒，确认数据流结束")
                break
        
        # 保存文件
        if len(self.orica_data_list) > 0:
            self._save_processed_data_to_file()
        else:
            print("⚠️ 没有数据需要保存")
    
    def _save_processed_data_to_file(self):
        """将累积的四路数据（raw/iir/asr/orica）保存为四个 npz 文件。"""
        if len(self.orica_data_list) == 0:
            print("⚠️ 没有数据需要保存")
            return
        
        try:
            all_raw = np.concatenate(self.raw_data_list, axis=1)
            all_iir = np.concatenate(self.iir_data_list, axis=1)
            all_asr = np.concatenate(self.asr_data_list, axis=1)
            all_orica = np.concatenate(self.orica_data_list, axis=1)

            def _save_one(path_obj, arr, stage_name):
                save_dict = {
                    'data': arr,
                    'channels': self.chan_labels,
                    'channel_indices': self.chan_range,
                    'sampling_rate': self.srate,
                    'duration': arr.shape[1] / self.srate,
                    'total_samples': arr.shape[1],
                    'stage': stage_name,
                }
                np.savez(path_obj, **save_dict)
                print(f"\n💾 {stage_name} 数据保存完成！")
                print(f"   文件: {path_obj}")
                print(f"   数据形状: {arr.shape}")
                print(f"   总时长: {arr.shape[1]/self.srate:.2f}秒")

            _save_one(self.raw_save_file, all_raw, 'raw')
            _save_one(self.iir_save_file, all_iir, 'iir')
            _save_one(self.asr_save_file, all_asr, 'asr')
            _save_one(self.orica_save_file, all_orica, 'orica')

            # 清空列表
            self.raw_data_list = []
            self.iir_data_list = []
            self.asr_data_list = []
            self.orica_data_list = []
            
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            import traceback
            traceback.print_exc()

    def _collect_10s_data(self, raw_chunk):
        """
        收集10秒数据
        
        Args:
            raw_chunk: 原始数据块 (channels, samples)
        """
        chunk_size = raw_chunk.shape[1]
        
        # 检查是否还有空间存储更多数据
        if self.calibration_size + chunk_size <= self.calibration_data.shape[1]:
            # 将新数据添加到校准缓冲区
            self.calibration_data[:, self.calibration_size:self.calibration_size + chunk_size] = raw_chunk
            self.calibration_size += chunk_size
            
            # 显示进度
            progress = self.calibration_size / (self.srate * self.calibration_duration)
            print(f"📊 收集进度: {progress:.1%} ({self.calibration_size / self.srate:.1f}/{self.calibration_duration}秒)")
            
            # 检查是否收集完成
            if self.calibration_size >= self.srate * self.calibration_duration:
                print("✅ 10秒数据收集完成！")
                self.calibration_collected = True
        else:
            print("⚠️ 校准数据缓冲区已满")

    def _asrpy_channel_names(self):
        n = len(self.chan_range)
        labels = list(self.chan_labels) if self.chan_labels else []
        if len(labels) >= n:
            return [str(labels[i]) for i in range(n)]
        return [f"EEG{i+1:03d}" for i in range(n)]

    def _fit_asrpy_on_calibration_numpy(self, calibration_data, cutoff):
        import asrpy

        x = np.asarray(calibration_data, dtype=np.float64)
        ch_names = self._asrpy_channel_names()
        if x.shape[0] != len(ch_names):
            ch_names = [f"EEG{i+1:03d}" for i in range(x.shape[0])]
        info = mne.create_info(
            ch_names=ch_names, sfreq=float(self.srate), ch_types="eeg"
        )
        raw = mne.io.RawArray(x, info, verbose=False)
        try:
            raw.set_montage("standard_1020", on_missing="ignore")
        except Exception:
            pass
        asr_inst = asrpy.ASR(sfreq=float(self.srate), cutoff=float(cutoff))
        asr_inst.fit(raw, picks="eeg")
        self.asr_filter = asr_inst
        self._asrpy_proc_R = None
        self._asrpy_proc_Zi = None
        self._asrpy_proc_cov = None

    def _apply_asr_accumulated_numpy(self, accum):
        if self.asr_backend == "meegkit":
            return self.asr_filter.transform(np.asarray(accum, dtype=np.float64))
        from asrpy.asr import asr_process

        asr_inst = self.asr_filter
        sfreq = float(self.srate)
        lookahead = 0.25
        stepsize = 32
        maxdims = 0.66
        mem_splits = 1
        x = np.asarray(accum, dtype=np.float64)
        n_ch, Lp = x.shape
        ls = int(float(sfreq) * lookahead)
        X_in = np.concatenate([x, np.zeros((n_ch, ls), dtype=np.float64)], axis=1)
        out, st = asr_process(
            X_in,
            sfreq,
            asr_inst.M,
            asr_inst.T,
            asr_inst.win_len,
            float(lookahead),
            int(stepsize),
            float(maxdims),
            (asr_inst.A, asr_inst.B),
            self._asrpy_proc_R,
            self._asrpy_proc_Zi,
            self._asrpy_proc_cov,
            None,
            True,
            asr_inst.method,
            int(mem_splits),
        )
        self._asrpy_proc_R = st["R"]
        self._asrpy_proc_Zi = st["Zi"]
        self._asrpy_proc_cov = st["cov"]
        out = np.asarray(out[:, ls:], dtype=np.float64)
        if out.shape[1] > Lp:
            out = out[:, -Lp:]
        elif out.shape[1] < Lp:
            out = np.pad(out, ((0, 0), (Lp - out.shape[1], 0)), mode="edge")
        return out

    def initialize_asr_from_mat(self, mat_file_path=r"D:\work\Python_Project\ORICA\temp_txt\cleaned_data_quick30.mat",cutoff=5):
        """
        从 MATLAB 文件加载校准数据并初始化 ASR（只执行一次）
        
        Args:
            mat_file_path: 校准数据的 .mat 文件路径
        """
        if self.asr_filter is not None:
            print("⏩ ASR 已校准，跳过重复初始化")
            return self.asr_filter
        
        try:


            
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
            if calibration_data.shape[0] != len(self.chan_range):
                print(f"⚠️ 通道数不匹配：校准 {calibration_data.shape[0]} 通道，在线 {len(self.chan_range)} 通道")
                calibration_data = calibration_data[self.chan_range, :]
                print(f"✅ 已调整校准数据形状: {calibration_data.shape}")
            
            if self.asr_backend == "asrpy":
                print(f"🔧 初始化 ASR (asrpy, cutoff={cutoff})...")
                self._fit_asrpy_on_calibration_numpy(calibration_data, cutoff)
            else:
                self.asr_filter = asr.ASR(
                    sfreq=self.srate,
                    cutoff=cutoff,
                )
                self.asr_filter.fit(calibration_data)
            print(f"✅ ASR 已校准完成，通道数: {calibration_data.shape[0]}")
            
            return self.asr_filter
            
        except Exception as e:
            print(f"❌ ASR 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    #D:\work\Python_Project\ORICA\temp_txt\cleaned_data_quick30.npz
    #D:\work\Python_Project\ORICA\Quick30_run\artifact_removal_verify\IIR_filter2\cali\laparoscopic_1309_EEGmerged_861_981.npz
    def initialize_asr_from_npz_1(self, npz_file_path=r"D:\work\Python_Project\ORICA\Quick30_run\artifact_removal_verify\IIR_filter2\cali\laparoscopic_1309_EEGmerged_845_1025.npz",cutoff=5):

    #def initialize_asr_from_npz(self, npz_file_path=r"D:\work\Python_Project\ORICA\Quick30_run\calibration\asr_calibration_20260104_231043.npz", cutoff=90):
        """
        从 .npz 文件加载校准数据并初始化 ASR（由receiver_new_cali.py生成）
        与 initialize_asr_from_mat 功能相同，但加载的是 .npz 格式文件
        
        Args:
            npz_file_path: 校准数据的 .npz 文件路径（由receiver_new_cali.py生成）
            cutoff: ASR截止阈值（标准差倍数），默认5。值越小越激进，值越大越保守
        
        Returns:
            asr_filter: 初始化并拟合好的ASR滤波器对象，失败返回None
        """
        if self.asr_filter is not None:
            print("⏩ ASR 已校准，跳过重复初始化")
            return self.asr_filter
        
        try:
            from pathlib import Path
            file_path = Path(npz_file_path)
            
            if not file_path.exists():
                print(f"❌ 校准文件不存在: {npz_file_path}")
                return None
            
            print(f"📂 加载ASR校准数据: {file_path.name}")
            
            # 加载NPZ文件
            if file_path.suffix.lower() != '.npz':
                print(f"❌ 不支持的文件格式，需要.npz文件")
                return None
            
            npz_data = np.load(file_path, allow_pickle=True)
            
            # 提取校准数据（优先使用 calibration_data 键）
            calibration_data = None
            if 'calibration_data' in npz_data:
                calibration_data = npz_data['calibration_data']
                print(f"   使用键: calibration_data")
            elif 'data' in npz_data:
                calibration_data = npz_data['data']
                print(f"   使用键: data")
            elif 'eeg_data' in npz_data:
                calibration_data = npz_data['eeg_data']
                print(f"   使用键: eeg_data")
            
            if calibration_data is None:
                print(f"❌ 无法从文件中提取校准数据，可用键: {list(npz_data.keys())}")
                return None
            
            # 转换为标准数组格式 (channels, samples)
            calibration_data = np.asarray(calibration_data, dtype=np.float64)
            
            # 确保数据格式为 (channels, samples)
            # 判断逻辑：如果第一个维度远大于第二个维度（比如样本数远大于通道数），说明是 (samples, channels)，需要转置
            if calibration_data.ndim == 2:
                if calibration_data.shape[0] > calibration_data.shape[1] * 10:
                    # 如果第一个维度远大于第二个维度（比如 1000000 > 100），说明是 (samples, channels)，需要转置
                    print(f"   ⚠️ 检测到数据形状 {calibration_data.shape}，可能是 (samples, channels)，将转置")
                    calibration_data = calibration_data.T
                    print(f"   转置后形状: {calibration_data.shape} (channels, samples)")
            
            print(f"✅ 校准数据加载成功 - 形状: {calibration_data.shape} (channels, samples)")
            
            # 检查通道数匹配
            if calibration_data.shape[0] != len(self.chan_range):
                print(f"⚠️ 通道数不匹配：校准数据 {calibration_data.shape[0]} 通道，当前使用 {len(self.chan_range)} 通道")
                
                # 如果校准数据通道数更多，尝试选择对应的通道
                if calibration_data.shape[0] > len(self.chan_range):
                    # 假设前N个通道对应
                    calibration_data = calibration_data[:len(self.chan_range), :]
                    print(f"   已选择前 {len(self.chan_range)} 个通道")
                else:
                    print(f"   ⚠️ 校准数据通道数不足，可能影响ASR效果")
            
            # 获取采样率（如果文件中有保存）
            file_srate = self.srate
            if 'sampling_rate' in npz_data:
                file_srate = int(npz_data['sampling_rate'])
                if file_srate != self.srate:
                    print(f"⚠️ 采样率不匹配：文件 {file_srate} Hz，当前 {self.srate} Hz")
                    print(f"   使用当前采样率: {self.srate} Hz")
            elif 'srate' in npz_data:
                file_srate = int(npz_data['srate'])
                if file_srate != self.srate:
                    print(f"⚠️ 采样率不匹配：文件 {file_srate} Hz，当前 {self.srate} Hz")
                    print(f"   使用当前采样率: {self.srate} Hz")
            
            print(f"🔧 初始化ASR滤波器 ({self.asr_backend}, cutoff={cutoff}, sfreq={self.srate} Hz)...")
            print(f"   注意: cutoff是标准差倍数，不是频率！值越小越激进，值越大越保守")
            if self.asr_backend == "asrpy":
                self._fit_asrpy_on_calibration_numpy(calibration_data, cutoff)
            else:
                self.asr_filter = asr.ASR(
                    sfreq=self.srate,
                    cutoff=cutoff,
                )
                self.asr_filter.fit(calibration_data)
            
            print(f"✅ ASR 校准完成！")
            print(f"   通道数: {calibration_data.shape[0]}")
            print(f"   数据长度: {calibration_data.shape[1]} 样本 ({calibration_data.shape[1]/self.srate:.2f} 秒)")
            print(f"   采样率: {self.srate} Hz")
            print(f"   cutoff阈值: {cutoff} (标准差倍数)")
            
            return self.asr_filter
            
        except Exception as e:
            print(f"❌ ASR 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def initialize_asr_from_mat_1(self, mat_file_path=r"D:\work\Python_Project\ORICA\temp_txt\cleaned_data_quick30.mat", cutoff=5):
        """
        从 .mat 文件加载校准数据并初始化 ASR（与 initialize_asr_from_npz_1 同逻辑）。
        唯一区别：输入来源是 MATLAB 文件。

        Args:
            mat_file_path: 校准数据的 .mat 文件路径
            cutoff: ASR截止阈值（标准差倍数），默认5。值越小越激进，值越大越保守

        Returns:
            asr_filter: 初始化并拟合好的ASR滤波器对象，失败返回None
        """
        if self.asr_filter is not None:
            print("⏩ ASR 已校准，跳过重复初始化")
            return self.asr_filter

        try:
            from pathlib import Path
            file_path = Path(mat_file_path)

            if not file_path.exists():
                print(f"❌ 校准文件不存在: {mat_file_path}")
                return None

            print(f"📂 加载ASR校准数据: {file_path.name}")

            # 加载MAT文件
            if file_path.suffix.lower() != ".mat":
                print(f"❌ 不支持的文件格式，需要.mat文件")
                return None

            mat_data = scipy.io.loadmat(file_path)

            # 提取校准数据（优先 cleaned_data.data，再 data）
            calibration_data = None
            if "cleaned_data" in mat_data:
                try:
                    eeg_struct = mat_data["cleaned_data"][0, 0]
                    if hasattr(eeg_struct, "dtype") and "data" in eeg_struct.dtype.names:
                        calibration_data = eeg_struct["data"]
                        print("   使用字段: cleaned_data.data")
                except Exception:
                    calibration_data = None
            if calibration_data is None and "data" in mat_data:
                calibration_data = mat_data["data"]
                print("   使用字段: data")

            if calibration_data is None:
                print(f"❌ 无法从文件中提取校准数据，可用字段: {list(mat_data.keys())}")
                return None

            # 转换为标准数组格式 (channels, samples)
            calibration_data = np.asarray(calibration_data, dtype=np.float64)

            # 确保数据格式为 (channels, samples)
            # 判断逻辑：如果第一个维度远大于第二个维度（比如样本数远大于通道数），说明是 (samples, channels)，需要转置
            if calibration_data.ndim == 2:
                if calibration_data.shape[0] > calibration_data.shape[1] * 10:
                    # 如果第一个维度远大于第二个维度（比如 1000000 > 100），说明是 (samples, channels)，需要转置
                    print(f"   ⚠️ 检测到数据形状 {calibration_data.shape}，可能是 (samples, channels)，将转置")
                    calibration_data = calibration_data.T
                    print(f"   转置后形状: {calibration_data.shape} (channels, samples)")

            print(f"✅ 校准数据加载成功 - 形状: {calibration_data.shape} (channels, samples)")

            # 检查通道数匹配
            if calibration_data.shape[0] != len(self.chan_range):
                print(f"⚠️ 通道数不匹配：校准数据 {calibration_data.shape[0]} 通道，当前使用 {len(self.chan_range)} 通道")

                # 如果校准数据通道数更多，尝试选择对应的通道
                if calibration_data.shape[0] > len(self.chan_range):
                    # 假设前N个通道对应
                    calibration_data = calibration_data[:len(self.chan_range), :]
                    print(f"   已选择前 {len(self.chan_range)} 个通道")
                else:
                    print(f"   ⚠️ 校准数据通道数不足，可能影响ASR效果")

            # 获取采样率（如果文件中有保存）
            file_srate = self.srate
            if "sampling_rate" in mat_data:
                file_srate = int(np.asarray(mat_data["sampling_rate"]).squeeze())
                if file_srate != self.srate:
                    print(f"⚠️ 采样率不匹配：文件 {file_srate} Hz，当前 {self.srate} Hz")
                    print(f"   使用当前采样率: {self.srate} Hz")
            elif "srate" in mat_data:
                file_srate = int(np.asarray(mat_data["srate"]).squeeze())
                if file_srate != self.srate:
                    print(f"⚠️ 采样率不匹配：文件 {file_srate} Hz，当前 {self.srate} Hz")
                    print(f"   使用当前采样率: {self.srate} Hz")

            print(f"🔧 初始化ASR滤波器 ({self.asr_backend}, cutoff={cutoff}, sfreq={self.srate} Hz)...")
            print(f"   注意: cutoff是标准差倍数，不是频率！值越小越激进，值越大越保守")
            if self.asr_backend == "asrpy":
                self._fit_asrpy_on_calibration_numpy(calibration_data, cutoff)
            else:
                self.asr_filter = asr.ASR(
                    sfreq=self.srate,
                    cutoff=cutoff,
                )
                self.asr_filter.fit(calibration_data)

            print(f"✅ ASR 校准完成！")
            print(f"   通道数: {calibration_data.shape[0]}")
            print(f"   数据长度: {calibration_data.shape[1]} 样本 ({calibration_data.shape[1]/self.srate:.2f} 秒)")
            print(f"   采样率: {self.srate} Hz")
            print(f"   cutoff阈值: {cutoff} (标准差倍数)")

            return self.asr_filter

        except Exception as e:
            print(f"❌ ASR 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def initialize_asr_online(self,calibration_data_raw):
        """
        从 MATLAB 文件加载校准数据并初始化 ASR（只执行一次）
        
        Args:
            mat_file_path: 校准数据的 .mat 文件路径
        """
        if self.asr_filter is not None:
            print("⏩ ASR 已校准，跳过重复初始化")
            return self.asr_filter
        
        try:


            
            
            # 转换为标准数组
            print("will get in")
            calibration_data = np.asarray(calibration_data_raw, dtype=np.float64)
            print(f"✅ 校准数据加载成功 - 原始形状: {calibration_data.shape}")
            
            # 只选择当前使用的通道
            if calibration_data.shape[0] != len(self.chan_range):
                print(f"⚠️ 通道数不匹配：校准 {calibration_data.shape[0]} 通道，在线 {len(self.chan_range)} 通道")
                calibration_data = calibration_data[self.chan_range, :]
                print(f"✅ 已调整校准数据形状: {calibration_data.shape}")
            
            cutoff_online = 5
            if self.asr_backend == "asrpy":
                print(f"🔧 初始化 ASR (asrpy, cutoff={cutoff_online})...")
                self._fit_asrpy_on_calibration_numpy(calibration_data, cutoff_online)
            else:
                self.asr_filter = asr.ASR(
                    sfreq=self.srate,
                    cutoff=cutoff_online,
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
        
        # 如果启用了数据保存，保存剩余数据
        if self.save_processed_data and len(self.orica_data_list) > 0:
            print("💾 停止数据流，保存已累积的数据...")
            self._save_processed_data_to_file()
        
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
        if self.pair_buffer is None:
            return None, None, None
        pb = self.pair_buffer
        return pb[0][self.chan_range, :], pb[1][self.chan_range, :], pb[2][self.chan_range, :]

    def get_pair_data_old(self, data_type='processed'):
        if data_type == 'raw':
            return self.chunk_pairs.copy()[0][1][self.chan_range, :] if self.chunk_pairs is not None else None
        else:
            return self.chunk_pairs.copy()[0][2][self.chan_range, :] if self.chunk_pairs is not None else None
        #return self.chunk_pairs.copy() if self.chunk_pairs is not None else None
    
    def get_buffer_data(self, data_type='processed'):
        """
        this function is used for the visualization
        Args:
            data_type: 'raw' or 'processed'
        """
        if data_type == 'raw':
            return self.raw_buffer[self.chan_range, :] if self.raw_buffer is not None else None
        else:
            return self.buffer[self.chan_range, :] if self.buffer is not None else None
    
    
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
            'indices': self.chan_range.copy() if self.chan_range else [],
            'count': len(self.chan_range) if self.chan_range else 0,
            'sampling_rate': self.srate
        }
    
    def is_data_available(self):
        """检查是否有可用数据"""
        return (self.last_unclean_chunk is not None and 
                self.last_processed_chunk is not None and 
                self.buffer is not None)
    
    def get_calibration_data(self):
        """获取收集到的10秒校准数据"""
        if self.calibration_collected:
            return self.calibration_data[:, :self.calibration_size].copy()
        return None
    
    def is_calibration_completed(self):
        """检查10秒数据收集是否完成"""
        return self.calibration_collected
    
    def process_calibration_data(self):
        """
        对收集到的10秒数据依次执行IIR、CAR、ASR、ORICA处理
        返回每一步处理后的数据和最终的icaweight、icasphere
        """
        if not self.calibration_collected:
            print("❌ 校准数据收集未完成，无法处理")
            return None
        
        try:
            print("🔄 开始处理10秒校准数据...")
            
            # 获取收集到的原始数据
            raw_data = self.calibration_data[:, :self.calibration_size]
            print(f"📊 原始数据形状: {raw_data.shape}")
            
            # Step 1: CAR (Common Average Reference)
            print("🔄 执行CAR重参考...")
            car_data = self.apply_car_rereference(raw_data)
            print(f"✅ CAR完成，数据形状: {car_data.shape}")
            
            # Step 2: IIR滤波
            print("🔄 执行IIR滤波...")
            iir_data = self.apply_mne_iir_filter(car_data)
            print(f"✅ IIR滤波完成，数据形状: {iir_data.shape}")
            
            # Step 3: ASR处理
            print("🔄 执行ASR处理...")
            asr_data = self._apply_asr_to_calibration_data(iir_data)
            print(f"✅ ASR处理完成，数据形状: {asr_data.shape}")
            
            # Step 4: ORICA处理
            print("🔄 执行ORICA处理...")
            orica_results = self._apply_orica_to_calibration_data(asr_data)
            
            if orica_results is not None:
                ica_weight, ica_sphere, sources, eog_indices = orica_results
                print(f"✅ ORICA处理完成")
                print(f"📊 ICA Weight形状: {ica_weight.shape}")
                print(f"📊 ICA Sphere形状: {ica_sphere.shape}")
                print(f"📊 源信号形状: {sources.shape}")
                print(f"📊 识别到 {len(eog_indices) if eog_indices else 0} 个伪影成分")
                
                # 返回所有处理结果
                return {
                    'raw_data': raw_data,
                    'car_data': car_data,
                    'iir_data': iir_data,
                    'asr_data': asr_data,
                    'ica_weight': ica_weight,
                    'ica_sphere': ica_sphere,
                    'sources': sources,
                    'eog_indices': eog_indices
                }
            else:
                print("❌ ORICA处理失败")
                return None
                
        except Exception as e:
            print(f"❌ 校准数据处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_asr_to_calibration_data(self, iir_data):
        """
        对IIR滤波后的数据应用ASR处理
        
        Args:
            iir_data: IIR滤波后的数据 (channels, samples)
        
        Returns:
            asr_data: ASR处理后的数据 (channels, samples)
        """
        try:
            # 创建ASR滤波器
            asr_filter = asr.ASR(
                sfreq=self.srate,
                cutoff=5,
            )
            
            # 使用IIR数据拟合ASR
            asr_filter.fit(iir_data)
            
            # 应用ASR处理
            asr_data = asr_filter.transform(iir_data)
            
            return asr_data
            
        except Exception as e:
            print(f"❌ ASR处理失败: {e}")
            return iir_data  # 如果ASR失败，返回原始IIR数据
    
    def _apply_orica_to_calibration_data(self, asr_data):
        """
        对ASR处理后的数据应用ORICA处理
        
        Args:
            asr_data: ASR处理后的数据 (channels, samples)
        
        Returns:
            tuple: (ica_weight, ica_sphere, sources, eog_indices) 或 None
        """
        try:
            # 创建临时ORICA处理器
            temp_orica = ORICAProcessor(
                n_components=len(self.chan_range),
                srate=self.srate
            )
            
            # 执行ORICA拟合
            sources, eog_indices, ic_probs, ic_labels = temp_orica.fit(
                asr_data, self.chan_range, self.chan_labels, self.srate
            )
            
            if sources is not None and temp_orica.ica is not None:
                # 获取ICA weight和sphere参数
                ica_weight = temp_orica.ica.get_W()
                ica_sphere = temp_orica.ica.get_sphere()
                
                return ica_weight, ica_sphere, sources, eog_indices
            else:
                print("⚠️ ORICA未产生有效结果")
                return None
                
        except Exception as e:
            print(f"❌ ORICA处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    

    def print_latest_channel_values(self):
        if self.buffer is None:
            print("⚠️ Buffer 尚未初始化，无法打印通道值")
            return



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