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
from asrpy import ASR
import mne
from scipy.signal import medfilt
from meegkit import asr
import scipy.io


from mne.filter import filter_data
from FirFilter import rest_fir_filter

#this class to some extent is like a container which includs all of the paraments(orica, icalabel)
class LSLStreamReceiver:
    def __init__(self, stream_type='EEG', time_range=5, stream_name='mybrain'):
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
        self.raw_buffer = None  # this buffer currently only used for vis(before orica)
        self.buffer = None#this buffer currently only used for vis(after orica)
        self.pair_buffer = None# conbining two bufer above
        

        #IIR
        self.cutoff = (1, 50)

        # ASR
        self.use_asr = False
        self.asr_calibrated = False
        self.asr_calibration_buffer = None
        self.prep_reference = None
        self.asr_filter = None  # ✅ 存储已校准的ASR实例

        

        #ORICA
        #ICLabel
        self.latest_ic_probs = None
        self.latest_ic_labels = None


        

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

        #用于画图时，保证处理后的数据和处理前的能够在时间上吻合
        self.chunk_pairs = []  # [(timestamp, unclean, processed)]


    def find_and_open_stream(self):
        # 1) check all available streams
        streams = resolve_streams()
        print("Current available LSL streams:")
        for i, stream in enumerate(streams):
            print(
                f"[{i}] Name: {stream.name()}, Type: {stream.type()}, Channels: {stream.channel_count()}, ID: {stream.source_id()}")
        print(f"Searching for LSL stream with type = '{self.stream_type}'...")

        # 2) select the stream by type or name
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


        # 3)generate basic info of the stream
        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.channel_manager = ChannelManager(info)
        self.srate = int(info.nominal_srate())
        self.nbchan = info.channel_count()

        # remove some of the useless or broken channels
        exclude = ['TRIGGER', 'ACC34','ACC33','ACC32', 'Packet Counter', 'ExG 2','ExG 1','ACC']#,'F7','F8'
        self.chan_labels = self.channel_manager.get_labels_excluding_keywords(exclude)
        self.chan_range = self.channel_manager.get_indices_excluding_keywords(exclude)

        self.nbchan = len(self.chan_range)

        # 4) buffer for visualization
        self.buffer = np.zeros((info.channel_count(), self.srate * self.time_range))
        self.raw_buffer = np.zeros((info.channel_count(), self.srate * self.time_range))

        print(f"Stream opened: {info.channel_count()} channels at {self.srate} Hz")
        print(f"Using {self.nbchan} EEG channels: {self.chan_labels}")

        # 5) init ORICA
        self.reinitialize_orica()


    #input chunk, return modeify chunk and sources , artifact index
    def process_orica(self, chunk):
        """
        use the orica_processor.py to modify the chunk data
        """
        cleaned_chunk = chunk.copy()
        # ORICA处理
        if self.orica is not None:
            if self.orica.update_buffer(chunk[self.chan_range, :]):
                sources, eog_indices, ic_probs, ic_labels= self.orica.fit(self.orica.data_buffer, self.chan_range, self.chan_labels, self.srate)
                if sources is not None:
                    cleaned = self.orica.transform(chunk[self.chan_range, :])
                    cleaned_chunk[self.chan_range, :] = cleaned  
                    self.latest_ic_probs = ic_probs
                    self.latest_ic_labels = ic_labels       
                    
        return cleaned_chunk, sources, eog_indices

    def pull_and_update_buffer(self):

        # this samples_random mean the number of samples is random, which depends on the system situation(no control)
        # this timestamps are the time of every random samples
        samples_random, timestamps = self.inlet.pull_chunk(timeout=0.0)
        #samples, timestamps = self.inlet.pull_chunk(max_samples=100, timeout=0.0)#

        # check if there is data
        if not timestamps:
            return
            
        # convert samples_random to the correct format (channels, samples)
        samples_random = np.array(samples_random).T
        samples=samples_random

        '''
        timestampes sometime is (0,), which means there is no data, the if will not execute.
        timestampes sometime is (samples_size,0), which means there is data, the if will execute.
        '''
        if timestamps:
            chunk = np.array(samples)  # shape: (channels, samples)这里的samples的大小是不固定的




            # Step 1: use the professional FIR filter from MNE
            chunk = self.apply_mne_iir_filter(chunk)

            
            #step 2: ASR
            # asr calibration
            if self.asr_filter is None:
                self.initialize_asr_from_mat()
            # asr usage
            if self.asr_filter is not None:
                chunk = self.asr_filter.transform(chunk)


            # Step 3: ORICA artifact removal
            # update chunk for visualization before ORICA
            self.last_unclean_chunk = chunk.copy()

            # update buffer before ORICA
            if self.raw_buffer is not None:
                self.raw_buffer = np.roll(self.raw_buffer, -chunk.shape[1], axis=1)
                self.raw_buffer[:, -chunk.shape[1]:] = self.last_unclean_chunk

            # ORICA processing
            chunk, ica_sources, eog_indices = self.process_orica(chunk)
            if ica_sources is not None:
                self.latest_sources = ica_sources
            if eog_indices is not None:
                self.latest_eog_indices = eog_indices



            # update chunk for visualization after ORICA
            self.last_processed_chunk = chunk.copy()
            
            # update buffer after ORICA
            num_new = chunk.shape[1]
            self.buffer = np.roll(self.buffer, -num_new, axis=1)
            self.buffer[:, -num_new:] = chunk

            



            
            # update chunk pairs for visualization
            # this step is used for keeping the vis looks synced
            # becasue the ORICA takes some time which would cause the data before and after ORICA looks not synced
            timestamp = time.time()
            self.chunk_pairs.append((timestamp, self.last_unclean_chunk, self.last_processed_chunk))
            if len(self.chunk_pairs) > 1:
                self.chunk_pairs.pop(0)

            self.pair_buffer = (self.raw_buffer, self.buffer)



    


    #Selected channel indices: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    #Selected channel labels: ['AF7', 'Fpz', 'F7', 'Fz', 'T7', 'FC6', 'F4', 'C4', 'Oz', 'CP6', 'Cz', 'PO8', 'CP5', 'O2', 'O1', 'P3', 'P4', 'P7', 'P8', 'Pz', 'PO7', 'T8', 'C3', 'Fp2', 'F3', 'F8', 'FC5', 'AF8']
    #上面就是channel_range和channel_labels的格式，要调用函数就传入这样的list
    def set_channel_range_and_labels(self, new_range, new_labels):
        with self.lock:
            self.chan_range = new_range
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
            n_components=len(self.chan_range),
            srate=self.srate
        )
        print("🔁 ORICA processor re-initialized with new channel range.")
    
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
            if calibration_data.shape[0] != len(self.chan_range):
                print(f"⚠️ 通道数不匹配：校准 {calibration_data.shape[0]} 通道，在线 {len(self.chan_range)} 通道")
                calibration_data = calibration_data[self.chan_range, :]
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
        return self.pair_buffer[0][self.chan_range, :], self.pair_buffer[1][self.chan_range, :] if self.pair_buffer is not None else None

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