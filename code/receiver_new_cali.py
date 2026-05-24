"""
ASR校准脚本
用于收集校准数据并生成ASR校准文件

使用方法：
1. 从LSL数据流收集校准数据
2. 或从已有文件加载校准数据
3. 使用meegkit.asr进行校准
4. 保存校准数据到.npz文件
"""

import numpy as np
import time
from pathlib import Path
from meegkit import asr
import scipy.io
from datetime import datetime
import mne
from mne.filter import filter_data

# 如果需要从LSL流收集数据，可以导入receiver
# from receiver_new import LSLStreamReceiver


def collect_calibration_data_from_stream(receiver, duration_seconds=40):
    """
    从LSL数据流收集校准数据
    
    Args:
        receiver: LSLStreamReceiver实例
        duration_seconds: 收集时长（秒），默认40秒
    
    Returns:
        calibration_data: 校准数据 (channels, samples)
    """
    print(f"📊 开始收集 {duration_seconds} 秒校准数据...")
    
    receiver.start()
    calibration_data_list = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_seconds:
            # 获取原始数据（IIR滤波后，ORICA之前）
            raw_chunk = receiver.get_raw_data()
            if raw_chunk is not None:
                # 只选择使用的通道
                chunk_selected = raw_chunk[receiver.chan_range, :]
                calibration_data_list.append(chunk_selected)
            
            time.sleep(0.1)  # 100ms间隔
        
        # 合并所有chunk
        if len(calibration_data_list) > 0:
            calibration_data = np.concatenate(calibration_data_list, axis=1)
            print(f"✅ 收集完成: {calibration_data.shape} (channels, samples)")
            return calibration_data
        else:
            print("❌ 未收集到数据")
            return None
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断收集")
        if len(calibration_data_list) > 0:
            calibration_data = np.concatenate(calibration_data_list, axis=1)
            print(f"✅ 已收集数据: {calibration_data.shape}")
            return calibration_data
        return None
    finally:
        receiver.stop()


def apply_iir_filter(data: np.ndarray, srate: float, cutoff: tuple = (1, 50)):
    """
    应用IIR滤波（与receiver_new.py中的实现完全一致）
    
    Args:
        data: EEG数据 (channels, samples)
        srate: 采样率
        cutoff: 截止频率 (low, high)，默认 (1, 50) Hz
    
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
            iir_params={'order': 4, 'ftype': 'butter'},  # 4th order Butterworth
            verbose=False
        )
        print(f"✅ IIR滤波完成: {cutoff[0]}-{cutoff[1]} Hz")
        return filtered_data
    except Exception as e:
        print(f"❌ IIR滤波失败: {e}")
        import traceback
        traceback.print_exc()
        return data


def load_calibration_data_from_file(file_path):
    """
    从文件加载校准数据
    支持格式：.mat, .npz, .npy, .set (EEGLAB格式)
    
    Args:
        file_path: 文件路径
    
    Returns:
        calibration_data: 校准数据 (channels, samples)
        srate: 采样率
        channels: 通道名称列表（如果有）
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return None, None, None
    
    print(f"📂 加载校准数据: {file_path.name}")
    
    calibration_data = None
    srate = None
    channels = None
    file_ext = file_path.suffix.lower()
    is_set_file = (file_ext == '.set')  # 标记是否为.set文件
    
    try:
        if file_ext == '.set':
            # 加载 EEGLAB .set 文件
            print("   使用MNE读取EEGLAB格式文件...")
            
            # 检查对应的 .fdt 文件是否存在
            fdt_file = file_path.with_suffix('.fdt')
            if not fdt_file.exists():
                print(f"   ⚠️  警告: 未找到对应的 .fdt 文件: {fdt_file.name}")
                print(f"   如果数据内嵌在 .set 文件中，将尝试继续...")
            
            # 使用 MNE 读取 .set 文件（会自动处理 .fdt 文件）
            raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
            
            # 只选择 EEG 通道（排除其他类型的通道）
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
            calibration_data = raw.get_data()  # MNE返回 (n_channels, n_samples) 格式
            channels = raw.ch_names
            
            # 验证数据格式：MNE返回的应该是 (channels, samples)
            # 如果第一个维度（通道数）小于第二个维度（样本数），说明格式正确
            # 如果第一个维度大于第二个维度，说明可能是 (samples, channels)，需要转置
            if calibration_data.shape[0] > calibration_data.shape[1]:
                # 如果第一个维度更大，说明是 (samples, channels)，需要转置
                print(f"   ⚠️  警告: 检测到数据形状 {calibration_data.shape}，可能是 (samples, channels)")
                print(f"   通道数: {len(channels)}, 将转置为 (channels, samples)")
                calibration_data = calibration_data.T
                print(f"   转置后形状: {calibration_data.shape} (channels, samples)")
            
            print(f"   ✅ .set文件加载成功")
            print(f"   通道数: {len(channels)}")
            print(f"   采样率: {srate} Hz")
            print(f"   最终数据形状: {calibration_data.shape} (channels, samples)")
            
        elif file_ext == '.mat':
            # 加载 MATLAB 文件
            mat_data = scipy.io.loadmat(str(file_path))
            
            # 提取数据
            if 'cleaned_data' in mat_data:
                eeg_struct = mat_data['cleaned_data'][0, 0]
                if 'data' in eeg_struct.dtype.names:
                    calibration_data = eeg_struct['data']
                if 'srate' in eeg_struct.dtype.names:
                    srate = int(eeg_struct['srate'][0, 0])
            elif 'data' in mat_data:
                calibration_data = mat_data['data']
            elif 'calibration_data' in mat_data:
                calibration_data = mat_data['calibration_data']
            
        elif file_ext == '.npz':
            # 加载 NPZ 文件
            npz_data = np.load(file_path, allow_pickle=True)
            
            # 尝试多种可能的键名
            for key in ['data', 'calibration_data', 'eeg_data']:
                if key in npz_data:
                    calibration_data = npz_data[key]
                    print(f"   使用键: {key}")
                    break
            
            if 'sampling_rate' in npz_data:
                srate = int(npz_data['sampling_rate'])
            elif 'srate' in npz_data:
                srate = int(npz_data['srate'])
            
            if 'channels' in npz_data:
                channels = npz_data['channels']
                if isinstance(channels, np.ndarray):
                    channels = channels.tolist()
                
        elif file_ext == '.npy':
            # 加载 NPY 文件
            calibration_data = np.load(file_path)
            
        else:
            print(f"❌ 不支持的文件格式: {file_ext}")
            return None, None, None
        
        if calibration_data is None:
            print(f"❌ 无法从文件中提取校准数据")
            return None, None, None
        
        # 转换为标准数组格式 (channels, samples)
        calibration_data = np.asarray(calibration_data, dtype=np.float64)
        
        # 确保数据格式为 (channels, samples)
        # 注意：对于 .set 文件，MNE已经返回 (channels, samples) 格式，不需要转置
        # 对于其他格式，如果第一个维度明显大于第二个维度（比如样本数远大于通道数），才需要转置
        if calibration_data.ndim == 2 and not is_set_file:
            # 只有当第一个维度远大于第二个维度时（比如 1000000 > 100），才可能是 (samples, channels)
            # 这里使用一个更保守的判断：如果第一个维度是第二个维度的10倍以上，才转置
            if calibration_data.shape[0] > calibration_data.shape[1] * 10:
                calibration_data = calibration_data.T
                print("   ⚠️ 检测到数据格式为 (samples, channels)，已自动转置")
        
        print(f"✅ 校准数据加载成功 - 形状: {calibration_data.shape} (channels, samples)")
        if srate:
            print(f"   采样率: {srate} Hz")
        if channels is not None:
            print(f"   通道数: {len(channels)}")
        
        return calibration_data, srate, channels
        
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def create_asr_calibration(calibration_data, srate, cutoff=5):
    """
    创建ASR校准对象
    
    Args:
        calibration_data: 校准数据 (channels, samples)
        srate: 采样率
        cutoff: ASR截止频率，默认5 Hz
    
    Returns:
        asr_filter: 拟合好的ASR滤波器对象
    """
    print(f"\n🔧 创建ASR校准...")
    print(f"   数据形状: {calibration_data.shape}")
    print(f"   采样率: {srate} Hz")
    print(f"   截止频率: {cutoff} Hz")
    
    try:
        # 初始化ASR
        asr_filter = asr.ASR(
            sfreq=srate,
            cutoff=cutoff,
        )
        
        # 拟合校准数据
        asr_filter.fit(calibration_data)
        
        print(f"✅ ASR校准完成！")
        print(f"   通道数: {calibration_data.shape[0]}")
        print(f"   数据长度: {calibration_data.shape[1]} 样本 ({calibration_data.shape[1]/srate:.2f} 秒)")
        
        return asr_filter
        
    except Exception as e:
        print(f"❌ ASR校准失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_calibration_file(calibration_data, srate, channels=None, output_path=None):
    """
    保存校准数据到文件
    
    Args:
        calibration_data: 校准数据 (channels, samples)
        srate: 采样率
        channels: 通道名称列表（可选）
        output_path: 输出文件路径（可选，默认自动生成，保存在calibration文件夹）
    
    Returns:
        output_path: 保存的文件路径
    """
    if output_path is None:
        # 自动生成文件名，保存在calibration文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = Path(__file__).parent
        calibration_dir = script_dir / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        output_path = calibration_dir / f"asr_calibration_{timestamp}.npz"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存校准数据到: {output_path.name}")
    
    try:
        save_dict = {
            'calibration_data': calibration_data.astype(np.float32),  # 节省空间
            'sampling_rate': srate,
            'srate': srate,  # 兼容性
            'duration': calibration_data.shape[1] / srate,
            'total_samples': calibration_data.shape[1],
            'n_channels': calibration_data.shape[0],
        }
        
        if channels is not None:
            save_dict['channels'] = np.array(channels, dtype=object)
        
        np.savez(output_path, **save_dict)
        
        print(f"✅ 校准数据保存成功！")
        print(f"   文件: {output_path}")
        print(f"   数据形状: {calibration_data.shape}")
        print(f"   通道数: {calibration_data.shape[0]}")
        print(f"   数据长度: {calibration_data.shape[1]} 样本 ({calibration_data.shape[1]/srate:.2f} 秒)")
        print(f"   采样率: {srate} Hz")
        print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    主函数：ASR校准流程
    """
    print("=" * 80)
    print("ASR 校准脚本")
    print("=" * 80)
    
    # ========== 配置参数 ==========
    # 方式1：从文件加载校准数据
    # 支持格式：.set (EEGLAB), .mat, .npz, .npy
    script_dir = Path(__file__).parent
    input_file = script_dir / "calibration" / "laparoscopic_1309_EEGmerged.set"  # 设置 .set 文件路径
    
    # 方式2：从LSL流收集数据（需要取消注释下面的代码）
    # from receiver_new import LSLStreamReceiver
    # receiver = LSLStreamReceiver()
    # collect_from_stream = True
    # collection_duration = 40  # 秒
    
    # IIR滤波参数（与receiver_new.py中的一致）
    iir_cutoff = (1, 50)  # Hz
    
    # ASR参数
    asr_cutoff = 5  # Hz
    srate = 500  # 如果从文件加载，会自动检测
    
    # 校准数据截取参数（重要！）
    # ASR推荐使用1-2分钟的干净数据，而不是整个文件
    calibration_duration_seconds = 60  # 截取的校准数据长度（秒），推荐60-120秒
    calibration_start_second = 30  # 从第几秒开始截取（跳过前30秒，可能有初始伪影）
    
    # 输出文件
    output_file = None  # 如果为None，会自动生成文件名
    
    # ========== 执行校准 ==========
    
    calibration_data = None
    channels = None
    is_set_file = False
    
    # 方式1：从文件加载
    if input_file:
        input_path = Path(input_file)
        is_set_file = input_path.suffix.lower() == '.set'
        
        calibration_data, file_srate, channels = load_calibration_data_from_file(input_file)
        if file_srate:
            srate = file_srate
        if calibration_data is None:
            print("❌ 无法加载校准数据，退出")
            return
        
        # 如果是 .set 文件，先截取校准数据，再应用 IIR 滤波
        if is_set_file:
            # 计算总时长
            total_duration = calibration_data.shape[1] / srate
            print(f"\n📊 原始数据信息：")
            print(f"   总时长: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
            print(f"   总样本数: {calibration_data.shape[1]}")
            
            # 截取校准数据（推荐1-2分钟）
            if total_duration > calibration_duration_seconds:
                start_sample = int(calibration_start_second * srate)
                end_sample = start_sample + int(calibration_duration_seconds * srate)
                
                # 确保不超出范围
                if end_sample > calibration_data.shape[1]:
                    end_sample = calibration_data.shape[1]
                    calibration_duration_seconds = (end_sample - start_sample) / srate
                
                calibration_data = calibration_data[:, start_sample:end_sample]
                print(f"\n✂️  截取校准数据：")
                print(f"   起始时间: {calibration_start_second} 秒")
                print(f"   截取长度: {calibration_duration_seconds:.2f} 秒")
                print(f"   截取后形状: {calibration_data.shape}")
            else:
                print(f"\n⚠️  数据总长度 ({total_duration:.2f} 秒) 小于推荐长度 ({calibration_duration_seconds} 秒)")
                print(f"   将使用全部数据")
            
            # 应用 IIR 滤波（1-50 Hz）
            print("\n" + "=" * 80)
            print("🔧 对校准数据应用 IIR 滤波 (1-50 Hz)...")
            print("=" * 80)
            calibration_data = apply_iir_filter(calibration_data, srate, cutoff=iir_cutoff)
            print(f"✅ IIR滤波完成，数据形状: {calibration_data.shape}")
    
    # 方式2：从LSL流收集（需要取消注释）
    # elif collect_from_stream:
    #     calibration_data = collect_calibration_data_from_stream(receiver, collection_duration)
    #     if calibration_data is None:
    #         print("❌ 无法收集校准数据，退出")
    #         return
    #     srate = receiver.srate
    #     channels = receiver.chan_labels
    
    if calibration_data is None:
        print("\n❌ 请配置输入文件或启用流收集")
        print("   修改 main() 函数中的 input_file 或 collect_from_stream 参数")
        return
    
    # 创建ASR校准
    print("\n" + "=" * 80)
    print("🔧 创建ASR校准...")
    print("=" * 80)
    asr_filter = create_asr_calibration(calibration_data, srate, cutoff=asr_cutoff)
    
    if asr_filter is None:
        print("❌ ASR校准失败，退出")
        return
    
    # 保存校准数据（保存的是IIR滤波后的数据，用于ASR校准）
    output_path = save_calibration_file(calibration_data, srate, channels, output_file)
    
    if output_path:
        print("\n" + "=" * 80)
        print("✅ ASR校准完成！")
        print("=" * 80)
        print(f"\n📝 使用方法：")
        print(f"   在 receiver_new.py 中调用：")
        print(f"   receiver.initialize_asr_from_npz('{output_path}')")
        print("=" * 80)
    else:
        print("\n❌ 保存失败")


if __name__ == '__main__':
    main()

