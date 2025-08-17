from ORICA import ORICA
import numpy as np
import numpy as np
from scipy.signal import welch
import mne
from mne_icalabel import label_components
from mne.preprocessing import ICA
import matplotlib
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
        self.eog_indices = []  # indices of components identified as eye artifacts
        self.artifact = []


    def fit(self, data, channel_range, chan_labels, srate):
        """Fit ICA on the buffered EEG data
        返回：
            sources: ICA分离出的独立成分信号（components, samples）
            A: ICA mixing matrix (通道数, 成分数)
            spectrum: dict，包含所有IC分量的频谱（'freqs': 频率, 'powers': shape=(n_components, n_freqs)）
        """

        assert data.shape[0] == self.n_components, f"Expected {self.n_components} channels, got {data.shape[0]}"

        if data.shape[1] < self.max_samples:
            return None, None, None  # Not enough data yet



        self.ica = ORICA(n_components=min(self.n_components, data.shape[0]), learning_rate=0.001)
        sources = self.ica.fit_transform(data.T).T  # shape: (components, samples)




        # 峰度评估
        k = self.ica.evaluate_separation(sources)
        #print("Kurtosis of components:", k)

        # 排序成分（高非高斯性优先）
        sorted_idx, k = self.ica.rank_components_by_kurtosis(sources)
        Y_sorted = sources[:, sorted_idx]

        #print("shwo",Y_sorted)



        #ic_probs, ic_labels = self.classify(data[channel_range, :],chan_labels, srate)
        #ic_probs, ic_labels = self.classify_with_mne_ica(data[channel_range, :],chan_labels, srate)
        
        #if ic_probs is not None and ic_labels is not None:
        #     print('ICLabel概率:', ic_probs)
        #     print('ICLabel标签:', ic_labels)

        self.identify_eye_artifacts(sources, self.srate)
        #self.identify_artifacts_by_iclabel(ic_labels, ic_probs, threshold=0.8)
        
        
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

    
        return sources, A, spectrum

    # def identify_eye_artifacts(self, components):
    #     """Heuristic: identify eye components as those with high frontal power and low frequency"""
    #     self.eog_indices = []
    #     for i, comp in enumerate(components):
    #         power = np.sum(comp ** 2)
    #         if power > np.percentile([np.sum(c ** 2) for c in components], 90):
    #             self.eog_indices.append(i)
    #
    #     print("EOG artifact:",self.eog_indices)




    def classify(self, data, chan_names, srate, montage='standard_1020'):
        """
        用 mne-icalabel 对当前窗口的ICA结果进行分类。
        输入:
            data: shape=(n_channels, n_samples)，原始EEG窗口数据
            chan_names: 通道名list
            srate: 采样率
            montage: 电极布局
        输出:
            ic_probs: shape=(n_components, 7)，每个IC属于各类别的概率
            ic_labels: shape=(n_components,)，每个IC的类别标签
        """



        # 1. 构造Raw对象
        info = mne.create_info(chan_names, srate, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        raw.set_montage(montage)
        raw.set_eeg_reference('average', projection=False)
        raw.filter(1., 100., fir_design='firwin')

        # 2. 用ORICA分离结果伪造ICA对象
        # A: mixing matrix (n_channels, n_components)
        # W: unmixing matrix (n_components, n_channels)
        W = self.ica.get_W()  # (n_components, n_channels)
        A = np.linalg.pinv(W) # (n_channels, n_components)
        n_components = W.shape[0]
        n_channels = A.shape[0]

        xica = ICA(n_components=n_components, fit_params=dict(extended=True), method='infomax', random_state=97, max_iter='auto')
        xica.current_fit = 'ica'
        xica.n_components_ = n_components
        xica.mixing_matrix_ = A
        xica.unmixing_matrix_ = W
        setattr(xica, 'pca_explained_variance_', np.ones(n_components))
        setattr(xica, 'pca_mean_', np.zeros(n_channels))
        setattr(xica, 'pca_components_', np.eye(n_components, n_channels))

        # 3. 调用ICLabel
        labels = label_components(raw, xica, method='iclabel')
        print("ICLabel返回内容：", labels)
        ic_probs = labels.get('y_pred_proba', None)
        ic_labels = labels.get('y_pred', None)
        if ic_labels is None and 'labels' in labels:
            ic_labels = labels['labels']
        return ic_probs, ic_labels


    def classify_with_mne_ica(self, data, chan_names, srate, montage='standard_1020'):
        """
        用MNE自带的ICA分解+ICLabel分类，便于和ORICA hack结果对比。
        输入:
            data: shape=(n_channels, n_samples)
            chan_names: 通道名list
            srate: 采样率
            montage: 电极布局
        输出:
            ic_probs: shape=(n_components, 7)
            ic_labels: shape=(n_components,)
        """
        import mne
        from mne.preprocessing import ICA
        from mne_icalabel import label_components

        info = mne.create_info(chan_names, srate, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        raw.set_montage(montage)
        raw.set_eeg_reference('average', projection=False)
        raw.filter(1., 100., fir_design='firwin')

        # 用MNE自带ICA分解
        ica = ICA(n_components=data.shape[0], fit_params=dict(extended=True), method='infomax', random_state=97, max_iter='auto')
        ica.fit(raw)

        # ICLabel分类
        labels = label_components(raw, ica, method='iclabel')
        print("[MNE ICA] ICLabel返回内容：", labels)
        ic_probs = labels.get('y_pred_proba', None)
        ic_labels = labels.get('y_pred', None)
        if ic_labels is None and 'labels' in labels:
            ic_labels = labels['labels']
        return ic_probs, ic_labels



    def identify_eye_artifacts(self, components, srate):

        self.eog_indices = []

        for i, comp in enumerate(components):

            fft_vals = np.abs(np.fft.rfft(comp))
            freqs = np.fft.rfftfreq(comp.shape[0], 1 / srate)
            low_freq_power = np.sum(fft_vals[(freqs >= 0.1) & (freqs <= 4)])#0.1-4hz的低频信号
            total_power = np.sum(fft_vals)
            ratio = low_freq_power / (total_power + 1e-10)


            if ratio > 0.19:  # 如果低频占比超过阈值，认为是 EOG
                self.eog_indices.append(i)

        #print("shitinggggggggggggggg:",self.eog_indices)

        #print("EOG artifact indices (low-freq based):", self.eog_indices)

    def identify_artifacts_by_iclabel(self, ic_labels, ic_probs, threshold=0.8):
        """
        根据ICLabel分类结果自动识别伪影IC。
        只要不是'brain'，且概率大于阈值（默认0.8），就加入self.artifact。
        """
        self.artifact = []
        for i, (label, prob) in enumerate(zip(ic_labels, ic_probs)):
            if label == 'brain':
                continue
            if label == 'other':
                if prob > threshold:
                    self.artifact.append(i)
            else:
                self.artifact.append(i)

            # if label != 'brain' and prob > threshold:
            #     self.artifact.append(i)

        #print("shitinggggggggggggggg:",self.artifact)



    def transform(self, new_data):
        #去除伪影的独立成分后重新映射回原来的通道。
        if self.ica is None:
            return new_data

        sources = self.ica.transform(new_data.T)
        sources[:, self.eog_indices] = 0  # Zero out EOG components
        #sources[:, self.eog_indices] = 0  # Zero out EOG components
        cleaned = self.ica.inverse_transform(sources)
        return cleaned.T

    def update_buffer(self, new_chunk):
        if self.data_buffer is None:
            self.data_buffer = new_chunk
        else:
            self.data_buffer = np.concatenate([self.data_buffer, new_chunk], axis=1)
            #当数据足够的时候才生成self.data_buffer,之后一直保持1500的长度，并且不断移动窗口
            if self.data_buffer.shape[1] > self.max_samples:
                self.data_buffer = self.data_buffer[:, -self.max_samples:]

        #print("data_buffer")
        #print(np.array(self.data_buffer).shape)

        return self.data_buffer.shape[1] >= self.max_samples
        