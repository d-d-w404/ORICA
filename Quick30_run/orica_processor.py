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
        self.icax = None
        self.data_buffer = None
        self.eog_indices = []  # indices of components identified as eye artifacts
        self.artifact = []

        self.sorted_W = None
        self.sorted_idx = None

        # ✅ 保存最近一次的 ICLabel 结果，供 GUI 显示
        self.latest_ic_probs = None
        self.latest_ic_labels = None





    def assess_orica_success(self,k_values, threshold_strong=5.0, threshold_good=3.0, verbose=True):
        k = np.array(k_values)
        count_strong = np.sum(k > threshold_strong)
        count_good = np.sum((k > threshold_good) & (k <= threshold_strong))
        mean_k = np.mean(k)
        std_k = np.std(k)
        max_k = np.max(k)

        if verbose:
            print(f"🔎 ORICA 全局评估：")
            print(f"  ▶️ 极强非高斯源数 (> {threshold_strong}): {count_strong}")
            print(f"  ▶️ 可接受脑源成分数 ({threshold_good}~{threshold_strong}): {count_good}")
            print(f"  ▶️ 峰度均值: {mean_k:.2f}, 标准差: {std_k:.2f}, 最大峰度: {max_k:.2f}")

        # 判断成功条件：至少一个 strong，或多个 good 且峰度分布拉开
        if count_strong >= 1:
            return True
        elif count_good >= 2 and std_k > 0.5:
            return True
        else:
            return False




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

    def evaluate_orica_sourcesx(self,sources, n_bins=10):
        from scipy.stats import kurtosis
        from sklearn.metrics import mutual_info_score
        import numpy as np

        # 峭度
        kurt_vals = kurtosis(sources, axis=1, fisher=True)
        # print("【ORICA评估】各IC的峭度（越大越非高斯，分离效果越好）:")
        # print(kurt_vals)
        print(f"kurtosis mean{np.mean(np.abs(kurt_vals)):.3f}")

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
        print(f"MI mean：{mi_mean:.3f}")



    def fit(self, data, channel_range, chan_labels, srate):
        """Fit ICA on the buffered EEG data
        返回：
            sources: ICA分离出的独立成分信号（components, samples）
            A: ICA mixing matrix (通道数, 成分数)
            spectrum: dict，包含所有IC分量的频谱（'freqs': 频率, 'powers': shape=(n_components, n_freqs)）
        """

        assert data.shape[0] == self.n_components, f"Expected {self.n_components} channels, got {data.shape[0]}"

        # if data.shape[1] < self.max_samples:
        #     return None, None, None  # Not enough data yet







        # self.ica = ORICA(n_components=min(self.n_components, data.shape[0]), learning_rate=0.001)
        # #self.ica = ORICA(n_components=min(self.n_components, data.shape[0]), learning_rate=0.001)
        # sources = self.ica.fit_transform(data.T).T  # shape: (components, samples)


        # self.ica = ORICA(n_components=min(self.n_components, data.shape[0]))
        # self.ica.initialize(data.T)  # only once, when data is available

        # # 在数据流中逐个输入样本（或者小批次）
        # sources = np.array([self.ica.partial_fit(x_t) for x_t in data.T]).T



        # 只在第一次创建ORICA实例，避免重复初始化
        if self.ica is None:
            print("🔄 首次创建ORICA实例bigshit")
            print("srate是啥呢",self.srate)
            self.ica = ORICA_final_new(n_components=min(self.n_components, data.shape[0]),srate=self.srate)
            self.ica.initialize(data.T)
            sources,x,y = self.ica.fit(data.T)
            #print("sources1",sources.shape)#(22,5000)
            # print("srate1",self.srate)#500
            #
        else:
            sources,x,y = self.ica.fit(data.T)
            print('data.T.shape',data.T.shape)
            print("sources",sources.shape)#(22,5000)
            #data.T.shape (49, 12)
            #sources (12, 49)
            # print("sources",sources.shape)#(22,5000)
            # print("srate",self.srate)#500
        print("源结果对比")
        print("sources.shape",sources.shape)
        print("sources",sources[0:3,0:3])
        print("evalshit")
        self.evaluate_orica_sources(sources)




        '''
            self.ica = ORICAZ(n_components=min(self.n_components, data.shape[0]))
            self.ica.initialize(data.T)  # 只在第一次初始化
            #print("test",data.T.shape)#test (2500, 25)
            # 首次初始化后，使用partial_fit处理数据
            #x_t (25,)

            #sources = np.array([self.ica.partial_fit(x_t) for x_t in data.T]).T
            # 使用fit_online_stream处理整个数据流
            #sources = self.ica.fit_online_stream(data.T)
            sources = self.ica.fit_block_stream(data.T)


            # #用于测试rls有无的差别
            # self.icax = ORICA1(n_components=min(self.n_components, data.shape[0]))
            # self.icax.initialize(data.T)  # 只在第一次初始化
            # #print("test",data.T.shape)#test (2500, 25)
            # # 首次初始化后，使用partial_fit处理数据
            # #x_t (25,)
            # sourcesx = np.array([self.icax.partial_fit(x_t) for x_t in data.T]).T

        else:
            #print("📈 继续使用现有ORICA实例进行学习")
            # 直接使用partial_fit进行在线学习，不重新初始化

            #sources = np.array([self.ica.partial_fit(x_t) for x_t in data.T]).T#sources (25, 5000)
            # 使用fit_online_stream处理整个数据流
            #sources = self.ica.fit_online_stream(data.T)#sources (5000, 25)
            sources = self.ica.fit_block_stream(data.T)

            #sourcesx = np.array([self.icax.partial_fit(x_t) for x_t in data.T]).T
            # for x_t in data.T:
            #     self.ica.partial_fit(x_t)
        # print("-"*50)
        #print("sources",sources.shape)
        #self.evaluate_orica_sources(sources)
        
        self.evaluate_orica_sources(sourcesx)
        print("sources",sources.shape)
        print("srate",self.srate)
        '''

        # 获取分离后的源信号
        #sources = self.ica.transform(data.T).T  # shape: (components, samples)

        # 峰度评估
        # k = self.ica.evaluate_separation(sources)
        # np.set_printoptions(threshold=np.inf, precision=4)  # 显示所有元素，保留4位小数
        # print("Kurtosis of components:", k)
        # np.set_printoptions()  # 恢复默认设置

        # success=self.assess_orica_success(k)
        # if success:
        #     print("✅ ORICA 分离成功（全局判断）")
        # else:
        #     print("❌ ORICA 分离失败或无显著非高斯源")

        # 排序成分（高非高斯性优先）
        # sorted_idx, k = self.ica.rank_components_by_kurtosis(sources)
        # Y_sorted = sources[:, sorted_idx]

        #print("shwo",Y_sorted)
        # mi_matrix = self.ica.calc_mutual_info_matrix(sources)
        # np.set_printoptions(threshold=np.inf, precision=4)  # 显示所有元素，保留4位小数
        # print("互信息矩阵:")
        # print(mi_matrix)
        # print("最大互信息:", np.max(mi_matrix))
        # print("平均互信息:", np.mean(mi_matrix))
        # np.set_printoptions()  # 恢复默认设置


        #ic_probs, ic_labels = self.classify(data[channel_range, :],chan_labels, srate)
        #ic_probs, ic_labels = self.classify_with_mne_ica(data[channel_range, :],chan_labels, srate)
        
        #if ic_probs is not None and ic_labels is not None:
        #     print('ICLabel概率:', ic_probs)
        #     print('ICLabel标签:', ic_labels)

        #self.identify_eye_artifacts(sources, self.srate)
        #self.identify_artifacts_by_iclabel(ic_labels, ic_probs, threshold=0.8)

        # 在 fit 方法中：
        # 使用ORICA的sources进行ICLabel分类并识别伪影

        
        # A =self.ica.get_icawinv()
        # #A = np.linalg.pinv(self.ica.W)

        # ic_probs, ic_labels = None, None
        # if sources is not None and A is not None:
        #     try:
        #         ic_probs, ic_labels = self.classify_sources_directly(data,sources,self.ica.get_icawinv() ,A, chan_labels, srate,n_comp=self.n_components)

        #         #ic_probs, ic_labels = self.classify_sources_directly(data,sources ,A, chan_labels, srate,n_comp=self.n_components)
        #     except Exception as e:
        #         print(f"ICLabel分类失败: {e}")

        # # 现在 self.eog_indices 已经包含了ICLabel识别的伪影
        # print(f"总伪影成分: {self.eog_indices}")

        # # ✅ 记录最新的 ICLabel 结果
        # self.latest_ic_probs = ic_probs
        # self.latest_ic_labels = ic_labels



        

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

        #print("2")
        # --- IC能量排序（旧逻辑，现注释保留备用） ---
        # if sources is not None:
        #     # 计算每个IC的低频占比
        #     ratios = []
        #     for comp in sources:
        #         fft_vals = np.abs(np.fft.rfft(comp))
        #         freqs = np.fft.rfftfreq(comp.shape[0], 1 / self.srate)
        #         low_freq_power = np.sum(fft_vals[(freqs >= 0.1) & (freqs <= 4)])
        #         total_power = np.sum(fft_vals)
        #         ratio = low_freq_power / (total_power + 1e-10)
        #         ratios.append(ratio)
        #     ratios = np.array(ratios)
        #     # 按低频占比从大到小排序
        #     self.sorted_idx = np.argsort(-ratios)
        #     sources = sources[self.sorted_idx, :]
        #     if A is not None:
        #         A = A[:, self.sorted_idx]
        #     # 对W排序并保存
        #     if hasattr(self.ica, 'get_W'):
        #         W = self.ica.get_W()
        #         self.sorted_W = W[self.sorted_idx, :]

        # --- 新逻辑：保持原始IC顺序 ---
        if sources is not None:
            self.sorted_idx = np.arange(sources.shape[0])
            if hasattr(self.ica, 'get_W'):
                W = self.ica.get_W()
                self.sorted_W = W



        return sources, A, spectrum

    # ✅ 提供获取 ICLabel 结果的便捷方法
    def get_iclabel_results(self):
        return self.latest_ic_probs, self.latest_ic_labels
    
    def _setup_emotiv_epoc_montage(self, raw, chan_names):
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
            raw.set_montage(montage)
            print("✅ 已为 Emotiv EPOC 设置自定义圆形电极布局")
        else:
            print(f"⚠️ 通道数量 {len(chan_names)} 与 Emotiv EPOC 不匹配，使用默认布局")

    # def identify_eye_artifacts(self, components):
    #     """Heuristic: identify eye components as those with high frontal power and low frequency"""
    #     self.eog_indices = []
    #     for i, comp in enumerate(components):
    #         power = np.sum(comp ** 2)
    #         if power > np.percentile([np.sum(c ** 2) for c in components], 90):
    #             self.eog_indices.append(i)
    #
    #     print("EOG artifact:",self.eog_indices)


    def classify_sources_directly_x(self, sources, mixing_matrix, chan_names, srate, threshold=0.8,n_comp=None):
        """直接对sources进行ICLabel分类，不依赖MNE ICA，并识别伪影"""
        from mne_icalabel import label_components
        
        # 创建Raw对象
        info = mne.create_info(chan_names, srate, ch_types='eeg')
        raw = mne.io.RawArray(sources, info)
        #raw.set_montage(mne.channels.make_standard_montage("standard_1020"))  # 或自定义 montage
        

        # 2) 构建 ICA 容器，并“注入” A/W（不需要再 fit）
        ica = ICA(n_components=n_comp, method='infomax')   # method 无关紧要
        ica.n_components_ = n_comp
        ica.current_fit = 'unfitted'
        ica.ch_names = chan_names
        ica._ica_names = [f'IC {k:03d}' for k in range(n_comp)]

        ica.mixing_matrix_   = mixing_matrix
        ica.unmixing_matrix_ = np.linalg.pinv(mixing_matrix)  # 若你有 W，就直接用你的 W，mixing_matrix_=pinv(W)




        # 直接使用ICLabel
        labels = label_components(raw, ica, method='iclabel')
        
        # 获取分类结果
        ic_probs = labels.get('y_pred_proba', None)
        ic_labels = labels.get('y_pred', None)
        if ic_labels is None and 'labels' in labels:
            ic_labels = labels['labels']
        
        # 识别伪影
        self.eog_indices = []
        if ic_labels is not None:
            for i, label in enumerate(ic_labels):
                if label != 'brain':  # 只要不是大脑信号，就认为是伪影
                    self.eog_indices.append(i)
        
        print(f"ICLabel识别到 {len(self.eog_indices)} 个伪影成分: {self.eog_indices}")
        
        return ic_probs, ic_labels


    def classify_sources_directly(self, data, sources, W, A, ch_names, srate,
                                threshold=0.8, n_comp=None, montage="standard_1020",
                                do_car=True, do_filter=True):

        print("classify")
        print(data.shape)
        print(srate)


        n_channels = len(ch_names)
        assert data.shape[0] == n_channels
        n_components = A.shape[1]
        assert A.shape[0] == n_channels
        
        # 0) Raw + 预处理
        info = mne.create_info(ch_names=list(ch_names), sfreq=float(srate), ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        #without car and filter, becasue which would not be corresponding to the sources

        # 1) 设置 montage
        raw.set_montage(montage)


        # 1) 规范 W 形状
        W_use = np.asarray(W, dtype=float)
        if W_use.shape == (n_channels, n_channels):
            W_use = W_use[:n_components, :]
        assert W_use.shape == (n_components, n_channels)

        # 2) 构造“已拟合”ICA 容器并注入
        ica = ICA(n_components=n_components, method='picard',
                fit_params=dict(extended=True, ortho=False), random_state=97)
        A_use = np.asarray(A, dtype=float)

        # —— 同时设置公开属性 + 私有字段（兼容不同版本）
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

        # —— 补齐 PCA/白化占位，避免版本差异触发属性检查
        ica.pca_mean_ = np.zeros(n_channels)
        ica.pca_components_ = np.eye(n_channels)
        ica.pca_explained_variance_ = np.ones(n_channels)
        ica.pca_explained_variance_ratio_ = (
            ica.pca_explained_variance_ / ica.pca_explained_variance_.sum()
        )
        ica._pre_whitener = np.ones((n_channels, 1))
        ica._whitener = np.eye(n_channels)

        # 3) ICLabel：返回 (labels, proba)
        labels = label_components(raw, ica, method='iclabel')
        print("labels",labels)
        
        # 获取分类结果
        ic_probs = labels.get('y_pred_proba', None)
        ic_labels = labels.get('y_pred', None)
        print("ic_labels",ic_labels)
        print("ic_probs",ic_probs)
        if ic_labels is None and 'labels' in labels:
            ic_labels = labels['labels']
        
        # 识别伪影
        self.eog_indices = []
        if ic_labels is not None:
            for i, label in enumerate(ic_labels):
                # 只要不是brain或other信号，就认为是伪影
                #if label not in ['brain', 'other']:
                #if label not in ['brain']:
                if label not in ['brain', 'other']:  # 保留 brain 和 other
                    self.eog_indices.append(i)


        # ✅ 强制去除指定IC（例如：IC 0）
        # 如需可配置，可改为读取外部列表或GUI参数
        # if 4 not in self.eog_indices:
        #     self.eog_indices.append(4)

        # if 5 not in self.eog_indices:
        #     self.eog_indices.append(5)

        
        #self.eog_indices.append(1)


        
        print(f"ICLabel识别到 {len(self.eog_indices)} 个伪影成分: {self.eog_indices}")
        
        return ic_probs, ic_labels
       







        
        # # 构建 ICA 容器，并"注入" A
        # print("n_comp",n_comp)
        # ica = ICA(n_components=n_comp, method='infomax')
        # ica.n_components_ = n_comp
        # ica.current_fit = 'raw'  # ✅ 关键：设置为 'raw' 而不是 'unfitted'
        # ica.ch_names = chan_names
        # ica._ica_names = [f'IC {k:03d}' for k in range(n_comp)]
        
        # # 设置混合矩阵和分离矩阵
        # ica.mixing_matrix_ = mixing_matrix
        # ica.unmixing_matrix_ = np.linalg.pinv(mixing_matrix)
        
        # # ✅ 添加必要的属性，让ICLabel认为ICA已经拟合
        # ica.pca_explained_variance_ = np.ones(n_comp)
        # ica.pca_mean_ = np.zeros(len(chan_names))
        # ica.pca_components_ = np.eye(n_comp, len(chan_names))
    
        
        # # 直接使用ICLabel
        # labels = label_components(raw, ica, method='iclabel')
        # print("labels",labels)
        
        # # 获取分类结果
        # ic_probs = labels.get('y_pred_proba', None)
        # ic_labels = labels.get('y_pred', None)
        # print("ic_labels",ic_labels)
        # print("ic_probs",ic_probs)
        # if ic_labels is None and 'labels' in labels:
        #     ic_labels = labels['labels']
        
        # # 识别伪影
        # self.eog_indices = []
        # if ic_labels is not None:
        #     for i, label in enumerate(ic_labels):
        #         # 只要不是brain或other信号，就认为是伪影
        #         #if label not in ['brain', 'other']:
        #         #if label not in ['brain']:
        #         if label not in ['brain', 'other']:  # 保留 brain 和 other
        #             self.eog_indices.append(i)


        # # ✅ 强制去除指定IC（例如：IC 0）
        # # 如需可配置，可改为读取外部列表或GUI参数
        # # if 4 not in self.eog_indices:
        # #     self.eog_indices.append(4)

        # # if 5 not in self.eog_indices:
        # #     self.eog_indices.append(5)

        
        # self.eog_indices.append(1)


        
        # print(f"ICLabel识别到 {len(self.eog_indices)} 个伪影成分: {self.eog_indices}")
        
        # return ic_probs, ic_labels




    def classify_sources_directlyy(self, data,sources, mixing_matrix, chan_names, srate, threshold=0.8, n_comp=None):
        """直接对sources进行ICLabel分类，不依赖MNE ICA，并识别伪影"""
        from mne_icalabel import label_components
        from mne.preprocessing import ICA

        print("classify")
        print(data.shape)
        print(srate)
        
        # 创建Raw对象
        info = mne.create_info(chan_names, srate, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        raw.set_eeg_reference("average", projection=False)
        
        # 为 Emotiv EPOC 设备设置专门的 montage
        try:
            # 尝试使用 Emotiv EPOC 的专用 montage
            #self._setup_emotiv_epoc_montage(raw, chan_names)
            raw.set_montage(mne.channels.make_standard_montage("standard_1020"),on_missing='ignore',match_case=False)
            #raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
            #raw.set_montage(mne.channels.make_standard_montage("emotiv"))
        except ValueError:
            # 如果失败，创建 Emotiv EPOC 的自定义 montage
            #self._setup_emotiv_epoc_montage(raw, chan_names)
            raw.set_montage(mne.channels.make_standard_montage("standard_1020"),on_missing='ignore',match_case=False)
            #raw.set_montage(mne.channels.make_standard_montage("emotiv"))

        
        # 构建 ICA 容器，并"注入" A
        print("n_comp",n_comp)
        ica = ICA(n_components=n_comp, method='infomax')
        ica.n_components_ = n_comp
        ica.current_fit = 'raw'  # ✅ 关键：设置为 'raw' 而不是 'unfitted'
        ica.ch_names = chan_names
        ica._ica_names = [f'IC {k:03d}' for k in range(n_comp)]
        
        # 设置混合矩阵和分离矩阵
        ica.mixing_matrix_ = mixing_matrix
        ica.unmixing_matrix_ = np.linalg.pinv(mixing_matrix)
        
        # ✅ 添加必要的属性，让ICLabel认为ICA已经拟合
        ica.pca_explained_variance_ = np.ones(n_comp)
        ica.pca_mean_ = np.zeros(len(chan_names))
        ica.pca_components_ = np.eye(n_comp, len(chan_names))
    
        
        # 直接使用ICLabel
        labels = label_components(raw, ica, method='iclabel')
        print("labels",labels)
        
        # 获取分类结果
        ic_probs = labels.get('y_pred_proba', None)
        ic_labels = labels.get('y_pred', None)
        print("ic_labels",ic_labels)
        print("ic_probs",ic_probs)
        if ic_labels is None and 'labels' in labels:
            ic_labels = labels['labels']
        
        # 识别伪影
        self.eog_indices = []
        # if ic_labels is not None:
        #     for i, label in enumerate(ic_labels):
        #         # 只要不是brain或other信号，就认为是伪影
        #         #if label not in ['brain', 'other']:
        #         #if label not in ['brain']:
        #         if label not in ['brain', 'other']:  # 保留 brain 和 other
        #             self.eog_indices.append(i)


        # ✅ 强制去除指定IC（例如：IC 0）
        # 如需可配置，可改为读取外部列表或GUI参数
        # if 4 not in self.eog_indices:
        #     self.eog_indices.append(4)

        # if 5 not in self.eog_indices:
        #     self.eog_indices.append(5)

        
        self.eog_indices.append(1)


        
        print(f"ICLabel识别到 {len(self.eog_indices)} 个伪影成分: {self.eog_indices}")
        
        return ic_probs, ic_labels


    def classify_sources_directlyxx(self, data, sources, mixing_matrix, chan_names, srate,
                                threshold=0.8, n_comp=None, use_mne_builtin=True):
        """
        直接用你已有的 mixing_matrix(A) + data(X) 构造 ICA 容器，调用 ICLabel。
        data: 传感器级数据 X (n_chan x n_times), 单位 V
        sources: 可忽略（MNE会用 W@X 计算），若没有 X，可先 X = A @ sources
        mixing_matrix: A (n_chan x n_comp)
        """
        import mne
        import numpy as np
        from mne.preprocessing import ICA

        # 1) 构造 Raw
        info = mne.create_info(chan_names, srate, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        # 2) 设置 montage（优先你的自定义 Emotiv 坐标）
        try:
            #self._setup_emotiv_epoc_montage(raw, chan_names)  # 你自己的函数：写入坐标
            std = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(std, on_missing='warn')
        except Exception as e:
            print(f"[warn] Emotiv montage failed: {e} -> fallback to standard_1020")
            std = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(std, on_missing='warn')

        raw.set_eeg_reference("average", projection=False)

        # 3) 构造 ICA 容器（不fit，直接注入矩阵）
        A = np.asarray(mixing_matrix)
        n_chan = len(chan_names)
        if n_comp is None:
            n_comp = A.shape[1]
        assert A.shape == (n_chan, n_comp), f"A shape must be (n_chan, n_comp), got {A.shape}"

        ica = ICA(n_components=n_comp, method='infomax', random_state=97, max_iter='auto')
        ica.n_components_ = int(n_comp)
        ica.current_fit = 'raw'
        ica.ch_names = list(chan_names)
        ica._ica_names = [f'IC {k:03d}' for k in range(n_comp)]

        ica.mixing_matrix_ = A
        ica.unmixing_matrix_ = np.linalg.pinv(A)

        # 可不设这些“伪”PCA属性，避免形状不一致
        # ica.pca_explained_variance_ = np.ones(n_comp)
        # ica.pca_mean_ = np.zeros(n_chan)
        # ica.pca_components_ = np.eye(n_comp, n_chan)

        # 4) 调用 ICLabel
        if use_mne_builtin:
            # 推荐：使用 MNE 内置（mne>=1.3）
            from mne.preprocessing.iclabel import label_components
            labels, probs = label_components(raw, ica, method='iclabel')
        else:
            # 兼容老版本：mne-icalabel
            from mne_icalabel import label_components
            labels, probs = label_components(raw, ica, method='iclabel')

        print("labels:", labels[:10], " ...")
        print("probs shape:", probs.shape)

        # 5) 依据概率做伪影判定（可配合 threshold）
        classes = ['brain','muscle','eye','heart','line_noise','channel_noise','other']
        cid = {c:i for i,c in enumerate(classes)}
        # 强伪影示例：eye/muscle/line_noise 概率 > 阈值
        bad_mask = (probs[:, cid['eye']] > threshold) | \
                (probs[:, cid['muscle']] > threshold) | \
                (probs[:, cid['line_noise']] > threshold)

        self.eog_indices = np.where(bad_mask)[0].tolist()
        print(f"ICLabel识别到 {len(self.eog_indices)} 个伪影成分: {self.eog_indices}")

        return probs, labels



    def classify(self, data, chan_names, srate, montage='emotiv_epoc'):
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
        
        # 为 Emotiv EPOC 设备设置专门的 montage
        if montage == 'emotiv_epoc':
            try:
                raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
            except ValueError:
                self._setup_emotiv_epoc_montage(raw, chan_names)
        else:
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


    def classify_with_mne_ica(self, data, chan_names, srate, montage='emotiv_epoc'):
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
        
        # 为 Emotiv EPOC 设备设置专门的 montage
        if montage == 'emotiv_epoc':
            try:
                raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
            except ValueError:
                self._setup_emotiv_epoc_montage(raw, chan_names)
        else:
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
            #print("fft_vals",fft_vals.shape)
            #print("freqs",freqs.shape)
            low_freq_power = np.sum(fft_vals[(freqs >= 0.1) & (freqs <= 4)])#0.1-4hz的低频信号

            total_power = np.sum(fft_vals)

            ratio = low_freq_power / (total_power + 1e-10)



            if ratio > 0.35:  # 如果低频占比超过阈值，认为是 EOG
                self.eog_indices.append(i)
        #print(self.eog_indices)

        #print("sh:",self.eog_indices)

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

        #print("x")
        sources = self.ica.transform(new_data.T)
        sources[:, self.eog_indices] = 0  # Zero out EOG components
        #sources[:, self.eog_indices] = 0  # Zero out EOG components
        cleaned = self.ica.inverse_transform(sources)
        #print("y")
        return cleaned.T

    def update_buffer_old(self, new_chunk):
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
        #这一行就是为啥orica需要等一段时间，因为我需要等到足够数据后，才能
        #在stream_receiver.py,
        #if self.orica.update_buffer(chunk[self.channel_range, :]):
        #这句话判断为true

    def update_buffer(self, new_chunk):
        self.data_buffer = new_chunk
        return True