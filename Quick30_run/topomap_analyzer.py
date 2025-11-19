from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import mne
from scipy.signal import welch

class TopomapDataWorker(QThread):
    data_ready = pyqtSignal(object, object, object, int, object, object, object, object, object, object)  # 10参数，最后一个为rv_list

    def __init__(self, receiver, interval_sec=2, n_components=5, psd_window=512):
        super().__init__()
        self.receiver = receiver
        self.interval_sec = interval_sec
        self.n_components = n_components
        self.running = True
        self.psd_window = psd_window
        self.psd_history = []  # 动态power spectrum缓存
        self.psd_maxlen = 10   # 滑动窗口长度


    def run(self):
        import time
        while self.running:
            try:
                if self.receiver is None or self.receiver.orica is None or self.receiver.orica.ica is None or self.receiver.channel_manager is None:
                    time.sleep(self.interval_sec)
                    continue
                

                #mixing_matrix = np.linalg.pinv(self.receiver.orica.ica.get_W())
                
                #对W按照能量强度排序。
                #W_sorted= self.receiver.orica.sorted_W
                W_sorted = self.receiver.orica.ica.get_icawinv()
                mixing_matrix = np.linalg.pinv(W_sorted)

                mixing_matrix2 = np.linalg.pinv(self.receiver.orica.ica.get_W())
                sorted_idx=self.receiver.orica.sorted_idx



                # 比较mixing_matrix和mixing_matrix2每一列的最大最小值
                # for i in range(mixing_matrix.shape[1]):
                #     col_sorted = mixing_matrix[:, i]
                #     col_orig = mixing_matrix2[:, sorted_idx[i]]
                #     print(f"IC {i} (能量排序后) vs 原始IC {sorted_idx[i]}:")
                #     print(f"  排序后max/min/mean: {col_sorted.max():.4f}/{col_sorted.min():.4f}/{col_sorted.mean():.4f}")
                #     print(f"  原始max/min/mean:   {col_orig.max():.4f}/{col_orig.min():.4f}/{col_orig.mean():.4f}")
                #     print(f"  差异范数: {np.linalg.norm(col_sorted - col_orig):.6f}")









                
                ch_names = self.receiver.channel_manager.get_labels_by_indices(self.receiver.chan_range)
                print("ch_names:", ch_names)
                #ch_names: ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 'FC6', 'T7', 'T8', 'C3', 'Cz', 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 'Pz', 'P4', 'PO8', 'PO3', 'PO4', 'O1', 'O2']

                print("self.receiver.srate:", self.receiver.srate)
                info = mne.create_info(ch_names=ch_names, sfreq=self.receiver.srate, ch_types='eeg')
                #info.set_montage('standard_1020')
                info.set_montage('standard_1020', on_missing='warn', match_case=False)


                


                #self._setup_emotiv_epoc_montage(info, ch_names)
                n_components = 20
                #n_components = mixing_matrix.shape[1]
                # 显示全部成分
                n_to_plot = n_components
                eog_indices = getattr(self.receiver, 'latest_eog_indices', None)
                


                # 添加缺失的参数，使用None或默认值
                ecd_locs = None
                power_spectra = None
                freqs = None
                rv_list = None
                self.data_ready.emit(mixing_matrix, ch_names, info, n_to_plot, eog_indices, sorted_idx ,ecd_locs, power_spectra, freqs, rv_list)
                
            except Exception as e:
                print(f"❌ Topomap数据线程错误: {e}")
                import traceback
                traceback.print_exc()
            time.sleep(self.interval_sec)

    def stop(self):
        self.running = False
        self.quit()
        self.wait() 

    def _setup_emotiv_epoc_montage(self, info, chan_names):
        """为 Emotiv EPOC 设备设置自定义 montage"""
        # Emotiv EPOC 的电极位置（基于实际设备布局）
        emotiv_positions = {
            'AF3': [0.0, 0.5, 0.0],      # 前额
            'F7': [-0.3, 0.3, 0.0],      # 左前额
            'F3': [-0.2, 0.4, 0.0],      # 左前额
            'FC5': [-0.4, 0.2, 0.0],     # 左前中央
            'T7': [-0.5, 0.0, 0.0],      # 左颞
            'P7': [-0.4, -0.2, 0.0],     # 左后颞
            'O1': [-0.2, -0.4, 0.0],     # 左枕
            'O2': [0.2, -0.4, 0.0],      # 右枕
            'P8': [0.4, -0.2, 0.0],      # 右后颞
            'T8': [0.5, 0.0, 0.0],       # 右颞
            'FC6': [0.4, 0.2, 0.0],      # 右前中央
            'F4': [0.2, 0.4, 0.0],       # 右前额
            'F8': [0.3, 0.3, 0.0],       # 右前额
            'AF4': [0.0, 0.5, 0.0]       # 前额
        }
        
        # 如果通道名称不匹配，使用圆形排列
        if len(chan_names) == 14:
            # 创建圆形排列的电极位置
            angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
            positions = np.column_stack([
                np.cos(angles) * 0.4,  # x坐标
                np.sin(angles) * 0.4,  # y坐标
                np.zeros(14)           # z坐标
            ])
            
            # 创建自定义 montage
            montage = mne.channels.make_dig_montage(
                ch_pos=dict(zip(chan_names, positions)),
                coord_frame='head'
            )
            info.set_montage(montage)
            print("✅ 已为 Emotiv EPOC 设置自定义圆形电极布局")
        else:
            print(f"⚠️ 通道数量 {len(chan_names)} 与 Emotiv EPOC 不匹配，使用默认布局")
