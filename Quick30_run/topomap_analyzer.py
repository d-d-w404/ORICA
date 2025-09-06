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
                
                # 过滤掉非EEG通道（如TRIGGER）
                eeg_ch_names = []
                for ch_name in ch_names:
                    if ch_name not in ['TRIGGER', 'ExG 1', 'ExG 2'] and not ch_name.startswith('ExG'):
                        eeg_ch_names.append(ch_name)
                
                if len(eeg_ch_names) == 0:
                    print("⚠️ 没有有效的EEG通道，跳过topomap生成")
                    time.sleep(self.interval_sec)
                    continue
                
                info = mne.create_info(ch_names=eeg_ch_names, sfreq=self.receiver.srate, ch_types='eeg')
                
                try:
                    info.set_montage('standard_1020')
                except ValueError as e:
                    print(f"⚠️ 设置导联失败: {e}")
                    # 如果设置导联失败，尝试使用默认位置
                    try:
                        info.set_montage('standard_1020', on_missing='warn')
                    except Exception as e2:
                        print(f"⚠️ 使用默认导联也失败: {e2}")
                        # 创建简单的圆形导联
                        pos = np.random.rand(len(eeg_ch_names), 3) * 0.1
                        info.set_montage(mne.channels.make_dig_montage(
                            ch_pos=dict(zip(eeg_ch_names, pos)),
                            coord_frame='head'
                        ))
                # 根据过滤后的通道数量调整mixing_matrix
                if len(eeg_ch_names) != mixing_matrix.shape[0]:
                    # 如果通道数量不匹配，需要重新计算mixing_matrix
                    print(f"⚠️ 通道数量不匹配: mixing_matrix={mixing_matrix.shape}, 有效通道={len(eeg_ch_names)}")
                    # 这里可以添加逻辑来重新计算mixing_matrix
                    # 暂时跳过这次更新
                    time.sleep(self.interval_sec)
                    continue
                
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