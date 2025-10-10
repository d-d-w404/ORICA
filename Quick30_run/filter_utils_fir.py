import numpy as np
import scipy.signal as signal
from scipy.signal import firwin, remez, firls
from typing import Union, Tuple, Optional, Dict, Any
from mne.filter import filter_data
class FIRFilter:
    """
    Python实现MATLAB flt_fir.m的功能
    """
    
    def __init__(self, 
                 frequencies: Union[list, np.ndarray],
                 mode: str = 'bandpass',
                 filter_type: str = 'minimum-phase',
                 passband_ripple: float = -20,
                 stopband_ripple: float = -40,
                 design_rule: str = 'Frequency Sampling',
                 chunk_length: int = 50000,
                 normalize_amplitude: bool = False,
                 use_fft: bool = True):
        
        self.frequencies = np.array(frequencies)
        self.mode = mode
        self.filter_type = filter_type
        self.passband_ripple = 10**(passband_ripple/10) if passband_ripple < 0 else passband_ripple
        self.stopband_ripple = 10**(stopband_ripple/10) if stopband_ripple < 0 else stopband_ripple
        self.design_rule = design_rule
        self.chunk_length = chunk_length
        self.normalize_amplitude = normalize_amplitude
        self.use_fft = use_fft
        self.state = {}
        self.filter_coeffs = None
        
    def design_filter(self, srate: float) -> np.ndarray:
        """设计FIR滤波器系数"""
        
        # 转换频率规格
        if self.mode == 'bandpass':
            fspec = [self.frequencies, [0, 1, 0]]
        elif self.mode == 'highpass':
            fspec = [self.frequencies, [0, 1]]
        elif self.mode == 'lowpass':
            fspec = [self.frequencies, [1, 0]]
        elif self.mode == 'bandstop':
            fspec = [self.frequencies, [1, 0, 1]]
        else:
            raise ValueError(f"不支持的滤波模式: {self.mode}")
        
        # 零相位滤波器特殊处理
        if self.filter_type == 'zero-phase':
            fspec[1] = np.sqrt(fspec[1])
        
        # 设计滤波器
        if self.design_rule == 'Parks-McClellan':
            # 使用remez算法
            bands = np.concatenate([[0], fspec[0], [srate/2]])
            desired = np.repeat(fspec[1], 2)
            b = remez(101, bands, desired, fs=srate)
            
        elif self.design_rule == 'Window Method':
            # 使用firwin
            if self.mode == 'bandpass':
                b = firwin(101, fspec[0], pass_zero=False, fs=srate)
            elif self.mode == 'highpass':
                b = firwin(101, fspec[0], pass_zero=False, fs=srate)
            elif self.mode == 'lowpass':
                b = firwin(101, fspec[0], fs=srate)
            else:
                b = firwin(101, fspec[0], pass_zero=False, fs=srate)
                
        else:  # Frequency Sampling
            # 频率采样方法
            freqs = np.concatenate([[0], fspec[0]*2/srate, [1]])
            amps = np.repeat(fspec[1], 2)
            
            # 设计Kaiser窗
            pos = np.argmin(np.diff(freqs))
            wnd = self._design_kaiser(freqs[pos], freqs[pos+1], 
                                   -20*np.log10(self.stopband_ripple), 
                                   amps[-1] != 0)
            
            # 设计FIR滤波器
            b = self._design_fir(len(wnd)-1, freqs, amps, window=wnd)
        
        # 最小相位滤波器转换
        if self.filter_type == 'minimum-phase':
            b = self._minimum_phase_conversion(b)
        
        # 幅度归一化
        if self.normalize_amplitude:
            maxamp = np.max(np.abs(np.fft.fft(np.concatenate([b, np.zeros(1000)]))))
            b = b * np.max(fspec[1]) / maxamp * (1 + self.passband_ripple)
        
        self.filter_coeffs = b
        return b
    
    def _design_kaiser(self, f1: float, f2: float, 
                      stopband_attenuation: float, 
                      passband_gain: bool) -> np.ndarray:
        """设计Kaiser窗"""
        # 简化的Kaiser窗设计
        beta = 0.1102 * (stopband_attenuation - 8.7)
        N = int(np.ceil((stopband_attenuation - 7.95) / (2.285 * (f2 - f1))))
        if N % 2 == 0:
            N += 1
        return signal.windows.kaiser(N, beta)
    
    def _design_fir(self, N: int, freqs: np.ndarray, 
                   amps: np.ndarray, window: np.ndarray = None) -> np.ndarray:
        """频率采样方法设计FIR滤波器"""
        # 简化的频率采样实现
        if window is None:
            window = np.ones(N+1)
        
        # 创建理想频率响应
        ideal_response = np.interp(np.linspace(0, 1, N+1), freqs, amps)
        
        # 应用窗函数
        h = np.fft.ifft(ideal_response * np.fft.fft(window, N+1))
        return np.real(h)
    
    def _minimum_phase_conversion(self, b: np.ndarray) -> np.ndarray:
        """最小相位滤波器转换"""
        n = len(b)
        wnd = np.concatenate([
            [1], 
            2*np.ones((n+1)//2-1), 
            np.ones(1-n%2), 
            np.zeros((n+1)//2-1)
        ])
        
        # 使用cepstral方法
        log_magnitude = np.log(np.abs(np.fft.fft(b)) + self.stopband_ripple)
        cepstrum = np.fft.ifft(log_magnitude)
        cepstrum_windowed = wnd * np.real(cepstrum)
        minimum_phase = np.real(np.fft.ifft(np.exp(np.fft.fft(cepstrum_windowed))))
        
        return minimum_phase
    
    def apply_filter(self, signal_data: np.ndarray, srate: float) -> Tuple[np.ndarray, Dict]:
        """应用滤波器到信号"""
        
        # 设计滤波器（如果还没有）
        if self.filter_coeffs is None:
            self.design_filter(srate)
        
        b = self.filter_coeffs
        n = len(b)
        
        # 处理每个时间序列字段
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(-1, 1)
        
        filtered_data = np.zeros_like(signal_data)
        
        for ch in range(signal_data.shape[1]):
            X = signal_data[:, ch].copy()
            
            # 检查是否有状态
            state_key = f'channel_{ch}'
            if state_key not in self.state:
                # 添加镜像段减少启动瞬态
                X = np.concatenate([
                    np.repeat(2*X[0], n) - X[np.mod(np.arange(n, 0, -1)-1, len(X))],
                    X
                ])
                
                if self.filter_type == 'zero-phase':
                    # 零相位滤波
                    X = X[::-1]
                    X = np.concatenate([
                        np.repeat(2*X[0], n) - X[np.arange(n, 0, -1)],
                        X
                    ])
                    X = signal.lfilter(b, 1.0, X)
                    X = X[-(len(signal_data[:, ch])+n):-n][::-1]
                else:
                    # 分块处理
                    S = len(X)
                    numsplits = int(np.ceil(S / self.chunk_length))
                    
                    for i in range(numsplits):
                        start = int(i * S / numsplits)
                        end = min(S, int((i+1) * S / numsplits))
                        range_idx = slice(start, end)
                        
                        if i == 0:
                            filtered_chunk = signal.lfilter(b, 1.0, X[range_idx], zi=None)
                            X[range_idx] = filtered_chunk
                        else:
                            filtered_chunk = signal.lfilter(b, 1.0, X[range_idx], zi=self.state[state_key])
                            X[range_idx] = filtered_chunk
                    
                    # 移除镜像段
                    X = X[n:]
                
                self.state[state_key] = None  # 标记为已处理
            else:
                # 在线处理
                if self.filter_type == 'zero-phase':
                    raise ValueError("零相位滤波器不能用于在线处理")
                
                # 分块处理
                S = len(X)
                numsplits = int(np.ceil(S / self.chunk_length))
                
                for i in range(numsplits):
                    start = int(i * S / numsplits)
                    end = min(S, int((i+1) * S / numsplits))
                    range_idx = slice(start, end)
                    
                    X[range_idx], self.state[state_key] = signal.lfilter(
                        b, 1.0, X[range_idx], zi=self.state[state_key]
                    )
            
            filtered_data[:, ch] = X
        
        # 计算滤波器延迟
        filter_delay = 0
        if self.filter_type == 'linear-phase':
            filter_delay = (len(b) // 2 - 1) / srate
        elif self.filter_type == 'minimum-phase':
            maxidx = np.argmax(b)
            filter_delay = (maxidx - 1) / srate
        
        # 返回结果和状态
        result_info = {
            'filter_delay': filter_delay,
            'filter_coeffs': b,
            'state': self.state
        }
        
        return filtered_data, result_info

# 使用示例
def example_usage(signal_data, fs):
    """使用示例"""
    # 生成测试信号
    # fs = 250  # 采样率
    # t = np.linspace(0, 10, 10*fs)  # 10秒数据
    # signal_data = (np.sin(2*np.pi*10*t) +  # 10Hz信号
    #                np.sin(2*np.pi*50*t) +  # 50Hz信号
    #                0.5*np.random.randn(len(t)))  # 噪声

    #print(signal_data)
    
    # 创建滤波器
    fir_filter = FIRFilter(
        frequencies=[1, 2, 49, 50],  # 1-50Hz带通，避免0和超过fs/2
        mode='bandpass',
        filter_type='linear-phase',
        design_rule='Window Method'
    )
    
    # 应用滤波器
    filtered_data, info = fir_filter.apply_filter(signal_data, fs)
    
    print(f"滤波器延迟: {info['filter_delay']:.4f} 秒")
    print(f"滤波器系数长度: {len(info['filter_coeffs'])}")
    #print(filtered_data)
    
    return filtered_data, info

if __name__ == "__main__":
    #filtered_data, info = example_usage()


    print("---------------")
    import numpy as np
    from mne.filter import filter_data

    # 输入数据示例
    signal = np.random.randn(1000)  # 单通道，1000个样本点
    # 或者
    signal = np.random.randn(4, 1000)  # 4通道，1000个样本点

    srate = 250  # 采样率
    fmin = 7     # 低截止频率
    fmax = 30    # 高截止频率
    print(signal)

    # 滤波处理
    filtered = filter_data(
        data=signal,        # 输入：原始信号
        sfreq=srate,       # 输入：采样率
        l_freq=fmin,       # 输入：低截止频率
        h_freq=fmax,       # 输入：高截止频率
        method='fir',
        fir_design='firwin',
        phase='zero',
        filter_length='auto',
        l_trans_bandwidth='auto',
        h_trans_bandwidth='auto',
        verbose=False
    )

    # 输出数据
    print("输入形状:", signal.shape)
    print("输出形状:", filtered.shape)
    print("输入类型:", signal.dtype)
    print("输出类型:", filtered.dtype)

    print(signal)
    print(filtered)