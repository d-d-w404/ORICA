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
                W_sorted= self.receiver.orica.sorted_W
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









                
                ch_names = self.receiver.channel_manager.get_labels_by_indices(self.receiver.channel_range)
                info = mne.create_info(ch_names=ch_names, sfreq=self.receiver.srate, ch_types='eeg')
                info.set_montage('standard_1020')
                n_components = mixing_matrix.shape[1]
                n_to_plot = min(self.n_components, n_components)
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