from ORICA import ORICA
from ORICA_none_rls import ORICAX
from ORICA1 import ORICA1
from ORICA_enhanced import ORICAW
#from ORICA_REST import ORICAZ
from ORICA_REST_new import ORICAZ
from ORICA_final import ORICA_final
#from ORICA_final_new import ORICA_final_new
#from ORICA_final_no_print import ORICA_final_new
from ORICA_final_no_print_quick30 import ORICA_final_new
# from ORICA_old import ORICA
import numpy as np
from scipy.signal import welch
import mne
from mne_icalabel import label_components
from mne.preprocessing import ICA
import matplotlib

from scipy.stats import kurtosis
from sklearn.metrics import mutual_info_score
import numpy as np
import json
import os
from datetime import datetime
import csv
# 更安全的GUI后端设置
try:
    matplotlib.use('Agg')  # 优先使用非GUI后端
except:
    try:
        matplotlib.use('TkAgg')  # 备用方案
    except:
        matplotlib.use('Qt5Agg')  # 最后备用方案

import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import seaborn as sns


class ORICAProcessor:
    def __init__(self, n_components=None, max_samples=10000, srate=None):
        self.n_components = n_components
        self.max_samples = max_samples
        self.srate = srate  # ✅ 保存采样率
        self.ica = None
        self.data_buffer = None
        

        self.sorted_W = None
        self.sorted_idx = None

        # ✅ 保存最近一次的 ICLabel 结果，供 GUI 显示
        self.latest_ic_probs = None
        self.latest_ic_labels = None
        self.eog_indices = [] 


    # evaluate the ORICA sources, but I would not use it now
    def evaluate_orica_sources(self, sources, n_bins=10):
        from scipy.stats import kurtosis
        from sklearn.metrics import mutual_info_score
        import numpy as np

        # 峭度
        kurt_vals = kurtosis(sources, axis=1, fisher=True)
        # print("【ORICA评估】各IC的峭度（越大越非高斯，分离效果越好）:")
        # print(kurt_vals)
        kurtosis_mean = np.mean(np.abs(kurt_vals))
        print(f"kurtosis mean: {kurtosis_mean:.3f}")

        # 互信息（先离散化）
        n = sources.shape[0]
        mi_matrix = np.zeros((n, n))
        # 离散化
        digitized = np.array([np.digitize(s, np.histogram(s, bins=n_bins)[1]) for s in sources])
        for i in range(n):
            for j in range(n):
                if i != j:
                    mi_matrix[i, j] = mutual_info_score(digitized[i], digitized[j])
        #print("【ORICA评估】IC之间的互信息矩阵（越接近0越独立）:")
        np.set_printoptions(precision=3, suppress=True)
        #print(mi_matrix)
        # 只统计非对角线元素的均值
        mi_mean = np.sum(mi_matrix) / (n * (n - 1))
        print(f"MI mean: {mi_mean:.3f}")

        # 保存评估结果到文件
        self._save_evaluation_results(kurtosis_mean, mi_mean, sources.shape[0], sources.shape[1])

    def _save_evaluation_results(self, kurtosis_mean, mi_mean, n_components, n_samples):
        """持续保存ORICA评估结果到文件"""
        timestamp = datetime.now().isoformat()
        
        # 准备数据
        evaluation_data = {
            'timestamp': timestamp,
            'kurtosis_mean': float(kurtosis_mean),
            'mi_mean': float(mi_mean),
            'n_components': n_components,
            'n_samples': n_samples
        }
        
        # 确保Results目录存在
        os.makedirs('./Results', exist_ok=True)
        
        # 保存到JSON文件（追加模式）
        json_filename = './Results/orica_evaluation_continuous.json'
        self._append_to_json(json_filename, evaluation_data)
        
        # 保存到CSV文件（追加模式）
        csv_filename = './Results/orica_evaluation_continuous.csv'
        self._append_to_csv(csv_filename, evaluation_data)
        
        print(f"✅ 评估结果已保存: kurtosis_mean={kurtosis_mean:.3f}, mi_mean={mi_mean:.3f}")

    def _append_to_json(self, filename, data):
        """将数据追加到JSON文件"""
        try:
            # 如果文件不存在，创建新文件
            if not os.path.exists(filename):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump([data], f, ensure_ascii=False, indent=2)
            else:
                # 读取现有数据
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
                
                # 追加新数据
                if isinstance(existing_data, list):
                    existing_data.append(data)
                else:
                    existing_data = [existing_data, data]
                
                # 写回文件
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                    
        except Exception as e:
            print(f"❌ 保存JSON文件失败: {e}")

    def _append_to_csv(self, filename, data):
        """将数据追加到CSV文件"""
        try:
            # 如果文件不存在，创建新文件并写入表头
            if not os.path.exists(filename):
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    writer.writeheader()
                    writer.writerow(data)
            else:
                # 追加数据
                with open(filename, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    writer.writerow(data)
                    
        except Exception as e:
            print(f"❌ 保存CSV文件失败: {e}")



    def fit(self, data, channel_range, chan_labels, srate):
        """Fit ICA on the buffered EEG data
        返回：
            sources: ICA分离出的独立成分信号（components, samples）
            A: ICA mixing matrix (通道数, 成分数)
            spectrum: dict，包含所有IC分量的频谱（'freqs': 频率, 'powers': shape=(n_components, n_freqs)）
        """

        #1 ORICA initialization and running
        assert data.shape[0] == self.n_components, f"Expected {self.n_components} channels, got {data.shape[0]}"
        # only create ORICA instance once, avoid repeated initialization
        if self.ica is None:
            print("Create ORICA instance")
            self.ica = ORICA_final_new(n_components=min(self.n_components, data.shape[0]),srate=self.srate)
            self.ica.initialize(data.T)
            sources,weight,sphere = self.ica.fit(data.T)
        else:
            sources,weight,sphere = self.ica.fit(data.T)
            # print('data.T.shape',data.T.shape)
            # print("sources",sources.shape)#(22,5000)

        # print("源结果对比")
        # print("sources.shape",sources.shape)
        # print("sources",sources[0:3,0:3])
        # print("evalshit")

        # evaluate the ORICA sources
        #self.evaluate_orica_sources(sources)
        #actually this is not necessary and accurate, maybe the chunk is too small.




        #2 use ICLabel to classify the ORICA sources
        A =self.ica.get_icawinv()
        ic_probs, ic_labels, eog_indices = None, None, None
        if sources is not None and A is not None:
            try:
                ic_probs, ic_labels,eog_indices = self.use_icalabel_online(data,sources,self.ica.get_icawinv() ,A, chan_labels, srate,n_comp=self.n_components)
            except Exception as e:
                print(f"ICLabel classification failed: {e}")

        self.latest_ic_probs = ic_probs
        self.latest_ic_labels = ic_labels
        self.eog_indices = eog_indices
        print(f"Total artifacts: {self.eog_indices}")

  

        # 获取mixing matrix A
        try:
            A = np.linalg.pinv(self.ica.W)
        except Exception:
            A = None


        # 获取所有IC分量的spectrum
        spectrum = None
        if sources is not None and self.srate is not None:
            powers = []
            freqs = None
            for ic in range(sources.shape[0]):
                f, Pxx = welch(sources[ic], fs=float(self.srate))
                if freqs is None:
                    freqs = f
                powers.append(Pxx)
            powers = np.array(powers)  # shape: (n_components, n_freqs)
            spectrum = {'freqs': freqs, 'powers': powers}


        # --- 新逻辑：保持原始IC顺序 ---
        if sources is not None:
            self.sorted_idx = np.arange(sources.shape[0])
            if hasattr(self.ica, 'get_W'):
                W = self.ica.get_W()
                self.sorted_W = W


        return sources, eog_indices, ic_probs, ic_labels


    def get_iclabel_results(self):
        return self.latest_ic_probs, self.latest_ic_labels
    
    # use icalabel online and return the ic_probs, ic_labels, eog_indices
    def use_icalabel_online(self, data, sources, W, A, ch_names, srate,
                                threshold=0.8, n_comp=None, montage="standard_1020"
                                ):

        n_channels = len(ch_names)
        assert data.shape[0] == n_channels
        n_components = A.shape[1]
        assert A.shape[0] == n_channels
        
        # 0) Raw + preprocessing
        info = mne.create_info(ch_names=list(ch_names), sfreq=float(srate), ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        #without car and filter, becasue which would not be corresponding to the sources

        # 1) set montage
        raw.set_montage(montage)


        # 2) confirm the shape of W
        W_use = np.asarray(W, dtype=float)
        if W_use.shape == (n_channels, n_channels):
            W_use = W_use[:n_components, :]
        assert W_use.shape == (n_components, n_channels)

        # 3) construct the "fitted" ICA container and inject
        # because we use the ORICA instead of the mne-ica
        ica = ICA(n_components=n_components, method='picard',
                fit_params=dict(extended=True, ortho=False), random_state=97)
        A_use = np.asarray(A, dtype=float)

        # —— set the public properties + private fields (compatible with different versions)
        ica.mixing_matrix_   = A_use.copy()
        ica.unmixing_matrix_ = W_use.copy()
        ica._mixing          = A_use.copy()
        ica._unmixing        = W_use.copy()

        ica.n_components_ = n_components
        ica.ch_names = list(ch_names)
        ica._ica_names = [f"IC {k:03d}" for k in range(n_components)]
        ica.picks_ = np.arange(n_channels)
        ica._ica_channel_names = list(ch_names)
        ica.current_fit = 'raw'

        # —— fill the PCA/whitening placeholder, avoid the attribute check triggered by version differences
        ica.pca_mean_ = np.zeros(n_channels)
        ica.pca_components_ = np.eye(n_channels)
        ica.pca_explained_variance_ = np.ones(n_channels)
        ica.pca_explained_variance_ratio_ = (
            ica.pca_explained_variance_ / ica.pca_explained_variance_.sum()
        )
        ica._pre_whitener = np.ones((n_channels, 1))
        ica._whitener = np.eye(n_channels)

        # 4) ICLabel: return (labels, proba)
        labels = label_components(raw, ica, method='iclabel')

        
        # get the classification results
        ic_probs = labels.get('y_pred_proba', None)
        ic_labels = labels.get('y_pred', None)
        # print("ic_labels",ic_labels)
        # print("ic_probs",ic_probs)
        if ic_labels is None and 'labels' in labels:
            ic_labels = labels['labels']
        
        # identify the artifacts
        eog_indices = []
        if ic_labels is not None:
            for i, label in enumerate(ic_labels):
                if label not in ['brain', 'other']:  # keep brain and other
                    eog_indices.append(i)

        
        print(f"ICLabel identified {len(eog_indices)} artifacts: {eog_indices}")
        
        return ic_probs, ic_labels, eog_indices
       
    #remove the artifacts and map back to the original channels, not used in the main code
    def transform(self, new_data):
        if self.ica is None:
            return new_data
        sources = self.ica.transform(new_data.T)
        sources[:, self.eog_indices] = 0  # Zero out EOG components
        cleaned = self.ica.inverse_transform(sources)
        return cleaned.T


    def update_buffer(self, new_chunk):
        self.data_buffer = new_chunk
        return True