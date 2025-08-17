import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
import matplotlib.pyplot as plt


class ORICACalibration:
    def __init__(self, sample_rate=500, block_size=8):
        """
        ORICA校准类
        
        Args:
            sample_rate: 采样率 (Hz)
            block_size: 块大小
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # 校准结果
        self.calibration_results = {}
        
    def calibrate(self, calibration_data, calibration_window=None):
        """
        执行校准过程
        
        Args:
            calibration_data: 校准数据 (n_samples, n_channels)
            calibration_window: 时间窗口 [start_sec, end_sec]，默认使用全部数据
            
        Returns:
            dict: 校准结果
        """
        print("🔧 开始ORICA校准...")
        
        # 1. 数据预处理
        if calibration_window is not None:
            start_idx = int(calibration_window[0] * self.sample_rate)
            end_idx = int(calibration_window[1] * self.sample_rate)
            calibration_data = calibration_data[start_idx:end_idx, :]
        
        print(f"📊 校准数据形状: {calibration_data.shape}")
        
        # 2. 带通滤波 (0.5-45Hz)
        filtered_data = self._bandpass_filter(calibration_data)
        
        # 3. 重参考 (TP7, TP8)
        reref_data = self._rereference(filtered_data)
        
        # 4. 计算统计特征
        stats = self._compute_statistics(reref_data)
        
        # 5. 存储校准结果
        self.calibration_results = {
            'filtered_data': filtered_data,
            'reref_data': reref_data,
            'statistics': stats,
            'sample_rate': self.sample_rate,
            'n_channels': calibration_data.shape[1],
            'n_samples': calibration_data.shape[0],
            'duration_sec': calibration_data.shape[0] / self.sample_rate
        }
        
        print("✅ 校准完成!")
        return self.calibration_results
    
    def _bandpass_filter(self, data, low_freq=0.5, high_freq=45):
        """带通滤波"""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)
    
    def _rereference(self, data):
        """重参考到TP7, TP8"""
        # 假设TP7是第7个通道，TP8是第8个通道
        # 实际使用时需要根据通道名称确定
        tp7_idx = 6  # 0-based索引
        tp8_idx = 7
        ref_channels = data[:, [tp7_idx, tp8_idx]]
        ref_mean = np.mean(ref_channels, axis=1, keepdims=True)
        return data - ref_mean
    
    def _compute_statistics(self, data):
        """计算统计特征"""
        stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'kurtosis': kurtosis(data, axis=0),
            'correlation_matrix': np.corrcoef(data.T),
            'covariance_matrix': np.cov(data.T)
        }
        return stats
    
    def get_calibration_results(self):
        """获取校准结果"""
        return self.calibration_results
    
    def plot_calibration_summary(self):
        """绘制校准摘要"""
        if not self.calibration_results:
            print("❌ 没有校准结果，请先运行calibrate()")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始数据vs滤波后数据
        axes[0,0].plot(self.calibration_results['reref_data'][:1000, 0])
        axes[0,0].set_title('滤波后数据 (前1000样本)')
        axes[0,0].set_ylabel('振幅 (μV)')
        
        # 峰度分布
        kurt = self.calibration_results['statistics']['kurtosis']
        axes[0,1].hist(kurt, bins=20)
        axes[0,1].set_title('峰度分布')
        axes[0,1].set_xlabel('峰度值')
        
        # 相关性矩阵
        corr = self.calibration_results['statistics']['correlation_matrix']
        im = axes[1,0].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1,0].set_title('通道相关性矩阵')
        plt.colorbar(im, ax=axes[1,0])
        
        # 标准差分布
        std = self.calibration_results['statistics']['std']
        axes[1,1].bar(range(len(std)), std)
        axes[1,1].set_title('各通道标准差')
        axes[1,1].set_xlabel('通道')
        axes[1,1].set_ylabel('标准差')
        
        plt.tight_layout()
        plt.show()