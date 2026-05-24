"""
REST兼容的FIR滤波器 - 简单易用版本
输入数据，返回滤波结果
"""

import numpy as np
from scipy import signal


def rest_fir_filter(data, srate, cutoff, mode='bandpass', ftype='zero-phase', 
                    online=False, filter_state=None):
    """
    REST兼容的FIR滤波器
    
    参数:
        data: ndarray
            输入数据，shape可以是:
            - (n_samples,) : 单通道
            - (n_channels, n_samples) : 多通道
            
        srate: float
            采样率 (Hz)
            
        cutoff: list or float
            截止频率 (Hz)
            - 高通: [0.5] 或 [0.25, 0.5]
            - 低通: [40] 或 [40, 50]
            - 带通: [0.5, 40]
            
        mode: str
            'highpass', 'lowpass', 'bandpass'
            
        ftype: str
            'zero-phase'(离线) 或 'linear-phase'(在线)
            
        online: bool
            是否在线处理模式
            
        filter_state: dict
            滤波器状态（在线模式用）
            
    返回:
        filtered_data: ndarray
            滤波后的数据
            
        state: dict (仅online=True时)
            滤波器状态
    """
    
    # 确保data是2D
    data = np.atleast_2d(data)
    if data.shape[0] > data.shape[1]:
        was_transposed = True
        data = data.T
    else:
        was_transposed = False
    
    n_channels, n_samples = data.shape
    
    # 设计或获取滤波器
    if filter_state is None or 'b' not in filter_state:
        filter_coef = _design_rest_fir(srate, cutoff, mode)
        if filter_state is None:
            filter_state = {}
        filter_state['b'] = filter_coef
        filter_state['n_channels'] = n_channels
        filter_state['zi'] = None
    else:
        filter_coef = filter_state['b']
    
    # 应用滤波
    if ftype == 'zero-phase' and not online:
        filtered_data = _apply_zero_phase(data, filter_coef)
    else:
        filtered_data, filter_state['zi'] = _apply_linear_phase(
            data, filter_coef, filter_state.get('zi')
        )
    
    # 恢复原始shape
    if was_transposed:
        filtered_data = filtered_data.T
    if filtered_data.shape[0] == 1:
        filtered_data = filtered_data[0]
    
    if online:
        return filtered_data, filter_state
    else:
        return filtered_data


def _design_rest_fir(srate, cutoff, mode):
    """设计REST风格的FIR滤波器"""
    nyq = srate / 2
    stopripple = 0.001  # -60dB
    
    # 构建频率和幅度
    if mode == 'bandpass':
        if len(cutoff) == 2:
            low, high = cutoff
            trans = max(2.0, (high - low) * 0.1)
            freqs = [0, (low-trans)/nyq, low/nyq, high/nyq, (high+trans)/nyq, 1]
            # 修复重复频率问题
            freqs = [max(0, min(1, f)) for f in freqs]
            # 去重并保持顺序
            seen = set()
            freqs_unique = []
            amps_unique = []
            original_amps = [0, 0, 1, 1, 0, 0]  # 对应的幅度值
            for i, f in enumerate(freqs):
                if f not in seen:
                    freqs_unique.append(f)
                    amps_unique.append(original_amps[i])
                    seen.add(f)
            freqs = freqs_unique
            amps = amps_unique
        else:
            freqs = [0] + [f/nyq for f in cutoff] + [1]
            amps = [0, 0, 1, 1, 0, 0]
        
    elif mode == 'highpass':
        if isinstance(cutoff, (int, float)):
            cutoff = [cutoff * 0.5, cutoff]
        trans_start, trans_end = cutoff
        freqs = [0, trans_start/nyq, trans_end/nyq, 1]
        amps = [0, 0, 1, 1]
        
    elif mode == 'lowpass':
        if isinstance(cutoff, (int, float)):
            cutoff = [cutoff, cutoff * 1.5]
        trans_start, trans_end = cutoff
        freqs = [0, trans_start/nyq, trans_end/nyq, 1]
        amps = [1, 1, 0, 0]
    
    # 计算滤波器长度
    trans_widths = np.diff(freqs)
    min_trans = np.min(trans_widths[trans_widths > 0])
    atten = -20 * np.log10(stopripple)
    numtaps = int(np.round((atten - 7.95) / (2 * np.pi * 2.285 * min_trans))) + 1
    
    if numtaps % 2 == 0:
        numtaps += 1
    numtaps = max(15, min(numtaps, int(10 * srate)))
    
    # 设计滤波器
    # 计算 Kaiser 窗的 beta 参数
    beta = 0.1102 * (atten - 8.7) if atten > 50 else 0.5842 * (atten - 21)**0.4 + 0.07886 * (atten - 21)
    b = signal.firwin2(numtaps, freqs, amps, window=('kaiser', beta))
    
    # 归一化
    freq_response = np.fft.fft(b, 2048)
    max_gain = np.max(np.abs(freq_response))
    if max_gain > 0:
        b = b / max_gain
    
    return b


def _mirror_prepend(data, n_pad):
    """镜像数据预处理（REST的核心技巧）"""
    n_channels, n_samples = data.shape
    first_sample = data[:, 0:1]
    
    if n_samples >= n_pad:
        reversed_data = data[:, n_pad:0:-1]
    else:
        repeats = int(np.ceil(n_pad / n_samples))
        repeated = np.tile(data, repeats)
        reversed_data = repeated[:, n_pad:0:-1]
    
    mirror_data = 2 * first_sample - reversed_data
    padded_data = np.concatenate([mirror_data, data], axis=1)
    
    return padded_data


def _apply_zero_phase(data, b):
    """零相位滤波（双向）"""
    n_channels, n_samples = data.shape
    n_pad = len(b)
    filtered = np.zeros_like(data)
    
    for ch in range(n_channels):
        # 正向
        padded1 = _mirror_prepend(data[ch:ch+1, :], n_pad)
        forward = signal.lfilter(b, 1, padded1[0])
        
        # 反向
        forward_rev = forward[::-1]
        forward_rev_2d = forward_rev.reshape(1, -1)
        padded2 = _mirror_prepend(forward_rev_2d, n_pad)
        backward = signal.lfilter(b, 1, padded2[0])
        
        # 切掉padding
        result = backward[::-1]
        filtered[ch] = result[n_pad:n_pad+n_samples]
    
    return filtered


def _apply_linear_phase(data, b, zi):
    """线性相位滤波（单向，支持在线）"""
    n_channels, n_samples = data.shape
    n_pad = len(b)
    filtered = np.zeros_like(data)
    
    if zi is None:
        # 首次：镜像预处理
        padded = _mirror_prepend(data, n_pad)
        zi_init = signal.lfilter_zi(b, 1)
        zi_all = np.tile(zi_init.reshape(-1, 1), (1, n_channels))
        
        for ch in range(n_channels):
            result, zi_all[:, ch] = signal.lfilter(b, 1, padded[ch], zi=zi_all[:, ch])
            filtered[ch] = result[n_pad:]
        
        zi = zi_all
    else:
        # 在线：使用状态
        for ch in range(n_channels):
            filtered[ch], zi[:, ch] = signal.lfilter(b, 1, data[ch], zi=zi[:, ch])
    
    return filtered, zi


# 便捷函数
def highpass_filter(data, srate, cutoff=0.5, **kwargs):
    """高通滤波"""
    if isinstance(cutoff, (int, float)):
        cutoff = [cutoff * 0.5, cutoff]
    return rest_fir_filter(data, srate, cutoff, mode='highpass', **kwargs)


def lowpass_filter(data, srate, cutoff=40, **kwargs):
    """低通滤波"""
    if isinstance(cutoff, (int, float)):
        cutoff = [cutoff, cutoff * 1.25]
    return rest_fir_filter(data, srate, cutoff, mode='lowpass', **kwargs)


def bandpass_filter(data, srate, low=0.5, high=40, **kwargs):
    """带通滤波"""
    return rest_fir_filter(data, srate, [low, high], mode='bandpass', **kwargs)