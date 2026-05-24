"""
ASR处理脚本 - 对 .set 文件进行 ASR 处理
使用 meegkit.asr 对 EEG 数据进行伪影去除

使用方法：
    python receiver_meegkit_asr.py

参数调整：
    修改脚本中的参数配置部分即可
"""

import numpy as np
from pathlib import Path
import mne
from mne.filter import filter_data
from meegkit import asr
from datetime import datetime

from paths import CODE_DIR

# ============================================================================
# 参数配置区域（在这里调整所有参数）
# ============================================================================

# 输入文件路径
INPUT_FILE = CODE_DIR / "calibration/laporoscopic_1309_EEGmerged.set"

# 输出文件路径（如果为None，会自动生成）
OUTPUT_FILE = None  # 如果为None，会保存为 INPUT_FILE 同目录下的 meegkit_asr.npz

# ========== IIR 滤波参数 ==========
IIR_CUTOFF = (1, 50)  # 带通滤波的截止频率 (low, high) Hz
IIR_ORDER = 4  # Butterworth 滤波器阶数

# ========== 数据截取参数（可选）==========
# 如果只想处理部分数据，设置这些参数
USE_DATA_SUBSET = False  # 是否只处理部分数据
DATA_START_SECOND = 30  # 起始时间（秒）
DATA_DURATION_SECONDS = 60  # 处理时长（秒），如果为None，处理到文件末尾

# ========== ASR 参数 ==========
# 主要参数
ASR_CUTOFF = 5.0  # 截止阈值（标准差倍数），值越小越激进，值越大越保守
                  # 推荐范围：2.5（非常激进）到 10（非常保守）
                  # 默认：5（保守）

# 窗口参数
ASR_WIN_LEN = 0.5  # 窗口长度（秒），用于检测伪影的时间窗口
ASR_WIN_OVERLAP = 0.66  # 窗口重叠比例（0-1），默认0.66（66%）

# 质量控制参数
ASR_MIN_CLEAN_FRACTION = 0.25  # 最小干净窗口比例（0-1），默认0.25
ASR_MAX_DROPOUT_FRACTION = 0.1  # 最大丢失窗口比例（0-1），默认0.1
ASR_MAX_BAD_CHANS = 0.3  # 最大坏通道比例（0-1），默认0.3

# 计算方法参数
ASR_METHOD = 'euclid'  # 方法：'euclid'（欧几里得距离）或 'riemann'（黎曼距离，需要pyriemann）
ASR_ESTIMATOR = 'scm'  # 估计器：'scm'（样本协方差矩阵）
ASR_BLOCKSIZE = 100  # 块大小（样本数），用于分块计算，减少内存占用

# ============================================================================
# 处理函数
# ============================================================================

def apply_iir_filter(data: np.ndarray, srate: float, cutoff: tuple = (1, 50), order: int = 4):
    """
    应用IIR滤波（与receiver.py中的实现完全一致）
    
    Args:
        data: EEG数据 (channels, samples)
        srate: 采样率
        cutoff: 截止频率 (low, high)，默认 (1, 50) Hz
        order: 滤波器阶数，默认4
    
    Returns:
        filtered_data: 滤波后的数据 (channels, samples)
    """
    try:
        filtered_data = filter_data(
            data=data,
            sfreq=srate,
            l_freq=cutoff[0],      # low frequency cutoff
            h_freq=cutoff[1],      # high frequency cutoff
            method='iir',          # apply IIR filter
            iir_params={'order': order, 'ftype': 'butter'},  # Butterworth
            verbose=False
        )
        print(f"✅ IIR滤波完成: {cutoff[0]}-{cutoff[1]} Hz, {order}阶")
        return filtered_data
    except Exception as e:
        print(f"❌ IIR滤波失败: {e}")
        import traceback
        traceback.print_exc()
        return data


def load_set_file(file_path: Path):
    """
    加载 .set 文件
    
    Args:
        file_path: .set 文件路径
    
    Returns:
        data: EEG数据 (channels, samples)
        srate: 采样率
        channels: 通道名称列表
    """
    print(f"\n📂 加载文件: {file_path.name}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 使用 MNE 读取 .set 文件
    raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
    
    # 只选择 EEG 通道
    try:
        raw.pick_types(eeg=True, meg=False, eog=False, ecg=False, emg=False, 
                      stim=False, misc=False, ref_meg=False, fnirs=False, 
                      exclude='bads')
    except:
        # 如果 pick_types 失败，尝试手动选择
        eeg_ch_idx = [i for i, ch_type in enumerate(raw.get_channel_types()) 
                     if ch_type == 'eeg']
        if eeg_ch_idx:
            raw.pick_channels([raw.ch_names[i] for i in eeg_ch_idx])
    
    # 获取采样率和数据
    srate = int(raw.info['sfreq'])
    data = raw.get_data()  # MNE返回 (n_channels, n_samples) 格式
    channels = raw.ch_names
    
    # 验证数据格式
    if data.shape[0] > data.shape[1]:
        print(f"   ⚠️  警告: 检测到数据形状 {data.shape}，可能是 (samples, channels)")
        print(f"   通道数: {len(channels)}, 将转置为 (channels, samples)")
        data = data.T
        print(f"   转置后形状: {data.shape} (channels, samples)")
    
    print(f"   ✅ 文件加载成功")
    print(f"   通道数: {len(channels)}")
    print(f"   采样率: {srate} Hz")
    print(f"   数据形状: {data.shape} (channels, samples)")
    print(f"   总时长: {data.shape[1] / srate:.2f} 秒 ({data.shape[1] / srate / 60:.2f} 分钟)")
    
    return data, srate, channels


def process_data_with_asr(data, srate, channels):
    """
    使用ASR处理数据
    
    Args:
        data: EEG数据 (channels, samples)
        srate: 采样率
        channels: 通道名称列表
    
    Returns:
        cleaned_data: ASR处理后的数据 (channels, samples)
        asr_filter: ASR滤波器对象
    """
    print(f"\n{'='*80}")
    print(f"🔧 开始ASR处理")
    print(f"{'='*80}")
    
    # 打印ASR参数
    print(f"\n📋 ASR参数配置:")
    print(f"   cutoff: {ASR_CUTOFF} (标准差倍数)")
    print(f"   win_len: {ASR_WIN_LEN} 秒")
    print(f"   win_overlap: {ASR_WIN_OVERLAP*100:.0f}%")
    print(f"   min_clean_fraction: {ASR_MIN_CLEAN_FRACTION}")
    print(f"   max_dropout_fraction: {ASR_MAX_DROPOUT_FRACTION}")
    print(f"   max_bad_chans: {ASR_MAX_BAD_CHANS}")
    print(f"   method: {ASR_METHOD}")
    print(f"   estimator: {ASR_ESTIMATOR}")
    print(f"   blocksize: {ASR_BLOCKSIZE}")
    
    # 初始化ASR
    print(f"\n🔧 初始化ASR滤波器...")
    asr_filter = asr.ASR(
        sfreq=srate,
        cutoff=ASR_CUTOFF,
        win_len=ASR_WIN_LEN,
        win_overlap=ASR_WIN_OVERLAP,
        min_clean_fraction=ASR_MIN_CLEAN_FRACTION,
        max_dropout_fraction=ASR_MAX_DROPOUT_FRACTION,
        method=ASR_METHOD,
        estimator=ASR_ESTIMATOR,
        blocksize=ASR_BLOCKSIZE,
    )
    
    # 拟合校准数据（使用部分数据作为校准数据，避免过度清理）
    print(f"🔧 拟合ASR校准数据...")
    
    # 关键修复：使用前60秒作为校准数据，剩余数据用于处理
    # 这样可以避免"用全部数据校准，然后对同样数据处理"导致的过度清理
    calibration_duration = 60  # 使用前60秒作为校准数据
    calibration_samples = int(calibration_duration * srate)
    calibration_samples = min(calibration_samples, data.shape[1] // 2)  # 最多用一半数据校准
    
    calibration_data = data[:, :calibration_samples]
    processing_data = data[:, calibration_samples:]
    
    print(f"   校准数据: {calibration_data.shape[1]} 样本 ({calibration_data.shape[1]/srate:.2f} 秒)")
    print(f"   待处理数据: {processing_data.shape[1]} 样本 ({processing_data.shape[1]/srate:.2f} 秒)")
    
    clean_calibration, sample_mask = asr_filter.fit(calibration_data)
    
    print(f"✅ ASR校准完成！")
    print(f"   校准后干净数据: {clean_calibration.shape[1]} 样本 ({clean_calibration.shape[1]/srate:.2f} 秒)")
    print(f"   保留窗口比例: {np.sum(sample_mask) / len(sample_mask):.2%}")
    
    # 应用ASR处理（只处理剩余的数据）
    print(f"\n🔧 应用ASR处理...")
    print(f"   处理数据: {processing_data.shape[1]} 样本 ({processing_data.shape[1]/srate:.2f} 秒)")
    
    cleaned_processing = asr_filter.transform(processing_data)
    
    # 合并结果：校准数据（已清理）+ 处理后的数据
    cleaned_data = np.concatenate([clean_calibration, cleaned_processing], axis=1)
    
    # 计算处理前后的差异
    diff = np.mean(np.abs(data - cleaned_data))
    diff_per_channel = np.mean(np.abs(data - cleaned_data), axis=1)
    
    print(f"✅ ASR处理完成！")
    print(f"   处理前后平均差异: {diff:.6f}")
    print(f"   各通道平均差异: {np.round(diff_per_channel[:5], 6)}... (前5个通道)")
    
    return cleaned_data, asr_filter


def save_results(data, srate, channels, output_path: Path):
    """
    保存处理结果到 .npz 文件
    
    Args:
        data: 处理后的数据 (channels, samples)
        srate: 采样率
        channels: 通道名称列表
        output_path: 输出文件路径
    """
    print(f"\n💾 保存结果到: {output_path.name}")
    
    try:
        save_dict = {
            'data': data.astype(np.float32),  # 节省空间
            'cleaned_data': data.astype(np.float32),  # 兼容性
            'sampling_rate': srate,
            'srate': srate,  # 兼容性
            'channels': np.array(channels, dtype=object),
            'duration': data.shape[1] / srate,
            'total_samples': data.shape[1],
            'n_channels': data.shape[0],
            'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        np.savez(output_path, **save_dict)
        
        print(f"✅ 保存成功！")
        print(f"   文件: {output_path}")
        print(f"   数据形状: {data.shape}")
        print(f"   通道数: {data.shape[0]}")
        print(f"   数据长度: {data.shape[1]} 样本 ({data.shape[1]/srate:.2f} 秒)")
        print(f"   采样率: {srate} Hz")
        print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    主函数
    """
    print("="*80)
    print("ASR 处理脚本 - meegkit")
    print("="*80)
    
    # 检查输入文件
    if not INPUT_FILE.exists():
        print(f"❌ 输入文件不存在: {INPUT_FILE}")
        return
    
    # 确定输出文件路径
    if OUTPUT_FILE is None:
        output_path = INPUT_FILE.parent / "meegkit_asr.npz"
    else:
        output_path = Path(OUTPUT_FILE)
    
    print(f"\n📋 处理配置:")
    print(f"   输入文件: {INPUT_FILE}")
    print(f"   输出文件: {output_path}")
    
    try:
        # 1. 加载 .set 文件
        data, srate, channels = load_set_file(INPUT_FILE)
        
        # 2. 数据截取（可选）
        if USE_DATA_SUBSET:
            start_sample = int(DATA_START_SECOND * srate)
            if DATA_DURATION_SECONDS is None:
                end_sample = data.shape[1]
            else:
                end_sample = start_sample + int(DATA_DURATION_SECONDS * srate)
                end_sample = min(end_sample, data.shape[1])
            
            data = data[:, start_sample:end_sample]
            print(f"\n✂️  截取数据:")
            print(f"   起始时间: {DATA_START_SECOND} 秒")
            print(f"   截取长度: {(end_sample - start_sample) / srate:.2f} 秒")
            print(f"   截取后形状: {data.shape}")
        
        # 3. 应用 IIR 滤波
        print(f"\n{'='*80}")
        print(f"🔧 应用 IIR 滤波")
        print(f"{'='*80}")
        filtered_data = apply_iir_filter(data, srate, cutoff=IIR_CUTOFF, order=IIR_ORDER)
        
        # 4. ASR 处理
        cleaned_data, asr_filter = process_data_with_asr(filtered_data, srate, channels)
        
        # 5. 保存结果
        print(f"\n{'='*80}")
        print(f"💾 保存结果")
        print(f"{'='*80}")
        save_results(cleaned_data, srate, channels, output_path)
        
        print(f"\n{'='*80}")
        print(f"✅ 处理完成！")
        print(f"{'='*80}")
        print(f"\n📝 结果文件: {output_path}")
        print(f"   可以使用以下代码加载结果:")
        print(f"   import numpy as np")
        print(f"   data = np.load('{output_path}')")
        print(f"   cleaned_data = data['cleaned_data']")
        print(f"   channels = data['channels']")
        print(f"   srate = data['sampling_rate']")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

