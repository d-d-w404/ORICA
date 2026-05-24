"""
LSL 流输出工具 - 支持 .npz 和 .set 文件
可以广播 .npz 文件（如 meegkit_asr.npz）和 .set 文件（EEGLAB格式）

使用方法：
    python aa_lsl_npz.py

修改 INPUT_FILE 变量即可切换不同的文件
"""

import time
import numpy as np
from pathlib import Path
import mne
from pylsl import StreamInfo, StreamOutlet

from paths import DATA_ROOT, INPUT_DATA_ROOT

_SCRIPT_DIR = Path(__file__).resolve().parent  # code/

# ============================================================================
# 配置参数（在这里修改文件路径）
# ============================================================================

# 输入文件路径（支持 .npz 和 .set 文件）
# 示例1：.npz 文件
#INPUT_FILE = _SCRIPT_DIR / "calibration/meegkit_iir1.npz"

# 示例2：.set 文件（取消注释使用）
#INPUT_FILE = _SCRIPT_DIR / "calibration/laparoscopic_1309_EEGmerged.set"
#INPUT_FILE = Path(r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_1309_EEGasr.set")



#这个是经过了IIR之后的数据
#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/IIR_filter2/laparoscopic_1309_EEGmerged.npz"


#这个是不经过IIR的原始数据
#1309
#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/npz_data/laparoscopic_1309_EEGmerged.npz"

#1307
#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/npz_data/laparoscopic_1307_EEGmerged.npz"

#1311
#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/npz_data/laparoscopic_1311_EEGmerged.npz"

#1295
#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/npz_data/laparoscopic_1295_EEGmerged.npz"

#1284
#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/npz_data/laparoscopic_1284_EEGmerged.npz"

#1271
#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/npz_data/laparoscopic_1271_EEGmerged.npz"

#003
#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/npz_data/laparoscopic_003_EEGmerged.npz"

#001
#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/npz_data/laparoscopic_001_EEGmerged.npz"


#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/set_data/laparoscopic_1309_EEGmerged.set"



#A01T
#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/npz_data/BNCI/npz/A01T.npz"

#INPUT_FILE = DATA_ROOT / "artifact_removal_verify/set_npz/npz_data/online_cali/Record_data/lsl_input_npz/online_calibration_1_45hz_20260505_013846_for_lsl.npz"


# LSL 广播：原始 EEG 流（未做 ASR 校准滤波）
INPUT_FILE = INPUT_DATA_ROOT / "npz/Shawn_shared/s28_resampled.npz"

# ASR 校准数据（IIR 1–50 Hz 后，通常用 2min 版做 calibration）
#INPUT_FILE = INPUT_DATA_ROOT / "asr_cali/Shawn_shared/2min/s28_resampled.npz"
#INPUT_FILE = INPUT_DATA_ROOT / "asr_cali_offline_notch/Shawn_shared/2min/s01_resampled.npz"

# 原始 npz（其他子目录）
#INPUT_FILE = INPUT_DATA_ROOT / "npz/Shawn_shared_raw/s05_061019m_new_ref.npz"

# 陷波后 npz（offline notch 60 Hz）
#INPUT_FILE = INPUT_DATA_ROOT / "npz_offline_notch/Shawn_shared/s01_resampled.npz"


# LSL 流配置
STREAM_NAME = "mybrain"  # 与 receiver.py 中的 stream_name 匹配
STREAM_TYPE = "EEG"
SOURCE_ID = "file2lsl_001"

# 传输配置
CHUNK_SIZE = 50  # 一次推送的样本数（可调）
WAIT_TIME = 2  # 等待 LSL 连接建立的时间（秒）

# ============================================================================
# 加载数据函数
# ============================================================================

def load_npz_file(npz_path: Path):
    """
    加载 .npz 文件
    
    Args:
        npz_path: .npz 文件路径
    
    Returns:
        data: EEG数据 (channels, samples)
        srate: 采样率
        channels: 通道名称列表
    """
    print(f"🔍 正在读取 .npz 文件: {npz_path.name}")
    
    if not npz_path.exists():
        raise FileNotFoundError(f"文件不存在: {npz_path}")
    
    try:
        data_dict = np.load(npz_path, allow_pickle=True)
        
        # 获取数据（优先使用 cleaned_data，然后是 data）
        if 'cleaned_data' in data_dict:
            data = data_dict['cleaned_data']
            print(f"   使用键: cleaned_data")
        elif 'data' in data_dict:
            data = data_dict['data']
            print(f"   使用键: data")
        else:
            print(f"❌ 未找到数据字段")
            print(f"   可用字段: {list(data_dict.keys())}")
            raise KeyError("未找到数据字段")
        
        # 确保数据格式为 (channels, samples)
        if data.ndim != 2:
            raise ValueError(f"数据维度错误: {data.ndim}，期望2维")
        
        # 如果第一个维度远大于第二个维度，可能是 (samples, channels)，需要转置
        if data.shape[0] > data.shape[1] * 10:
            print(f"   ⚠️  检测到数据形状 {data.shape}，可能是 (samples, channels)，将转置")
            data = data.T
            print(f"   转置后形状: {data.shape} (channels, samples)")
        
        # 获取采样率
        if 'sampling_rate' in data_dict:
            srate = int(data_dict['sampling_rate'])
        elif 'srate' in data_dict:
            srate = int(data_dict['srate'])
        elif 'fs' in data_dict:
            srate = int(data_dict['fs'])
        else:
            print(f"⚠️  未找到采样率，使用默认值 500 Hz")
            srate = 500
        
        # 获取通道名称
        if 'channels' in data_dict:
            channels = data_dict['channels']
            if isinstance(channels, np.ndarray):
                channels = channels.tolist()
        else:
            print(f"⚠️  未找到通道名称，使用默认通道名")
            channels = [f"Ch{i+1}" for i in range(data.shape[0])]
        
        print(f"✅ .npz文件加载成功")
        print(f"   数据形状: {data.shape} (channels, samples)")
        print(f"   采样率: {srate} Hz")
        print(f"   通道数: {len(channels)}")
        print(f"   总时长: {data.shape[1] / srate:.2f} 秒 ({data.shape[1] / srate / 60:.2f} 分钟)")
        
        return data, srate, channels
        
    except Exception as e:
        print(f"❌ 加载 .npz 文件失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def _eeglab_chanlocs_to_names(eeg, n_ch: int):
    """从 EEGLAB EEG.chanlocs 提取通道名；失败则用 Ch01..。"""
    names: list = []
    try:
        cl = eeg.chanlocs
        if cl is None:
            return [f"Ch{i + 1}" for i in range(n_ch)]
        arr = np.squeeze(np.asarray(cl, dtype=object))
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.size == 0:
            return [f"Ch{i + 1}" for i in range(n_ch)]
        n_loc = min(n_ch, int(arr.shape[0]))
        for i in range(n_loc):
            row = arr[i]
            lab = getattr(row, "labels", None)
            if lab is None and isinstance(row, np.void) and row.dtype.names and "labels" in row.dtype.names:
                lab = row["labels"]
            if isinstance(lab, (bytes, bytearray)):
                names.append(lab.decode("utf-8", errors="ignore").strip())
            elif lab is not None:
                names.append(str(np.asarray(lab).squeeze()).strip())
            else:
                names.append(f"Ch{i + 1}")
        while len(names) < n_ch:
            names.append(f"Ch{len(names) + 1}")
        return names[:n_ch]
    except Exception:
        return [f"Ch{i + 1}" for i in range(n_ch)]


def _load_set_via_scipy_eeglab_mat(set_path: Path):
    """
    MNE 对部分 .set 会因 chaninfo 缺少 nodatchans 抛 KeyError。
    对「标准 MAT 格式」EEGLAB .set，用 scipy 读 EEG 结构（EEG.data 默认按 EEGLAB 为 µV）。
    """
    import scipy.io as sio

    try:
        d = sio.loadmat(str(set_path), struct_as_record=False, squeeze_me=True)
    except NotImplementedError as e:
        raise RuntimeError(
            "该 .set 为 MATLAB v7.3 (HDF5)，scipy 无法直接读取。"
            "请在 EEGLAB 中 File → Save 另存为 v6 .set，或升级 MNE / 先导出 .npz。"
        ) from e

    if "EEG" not in d:
        raise ValueError("MAT 中无 EEG 变量，不是标准 EEGLAB .set")

    eeg = d["EEG"]
    if isinstance(eeg, np.ndarray) and eeg.dtype == object:
        eeg = eeg.item()

    data = np.asarray(eeg.data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"EEG.data 期望 2 维，得到 shape={data.shape}")

    srate = int(np.round(float(np.asarray(eeg.srate).squeeze())))
    nbchan = int(np.round(float(np.asarray(eeg.nbchan).squeeze())))

    # EEGLAB 连续数据一般为 (channels, samples)；若与 nbchan 不符则尝试转置
    if data.shape[0] != nbchan and data.shape[1] == nbchan:
        data = data.T
    n_ch = data.shape[0]
    channels = _eeglab_chanlocs_to_names(eeg, n_ch)
    # EEGLAB 默认存 µV；LSL 侧仍按 uV 推送，与 MNE 路径 raw.get_data()*1e6 一致
    data = np.ascontiguousarray(data.astype(np.float32))

    print(f"✅ scipy 回退读取成功（绕过 MNE chaninfo/nodatchans）")
    print(f"   数据形状: {data.shape} (channels, samples)，单位按 EEGLAB 默认 µV")
    print(f"   采样率: {srate} Hz")
    return data, srate, channels


def load_set_file(set_path: Path):
    """
    加载 .set 文件（EEGLAB格式）
    
    Args:
        set_path: .set 文件路径
    
    Returns:
        data: EEG数据 (channels, samples)
        srate: 采样率
        channels: 通道名称列表
    """
    print(f"🔍 正在读取 .set 文件: {set_path.name}")
    
    if not set_path.exists():
        raise FileNotFoundError(f"文件不存在: {set_path}")
    
    try:
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
        print(f"✅ 成功读取文件")
        print(f"📊 原始通道数: {len(raw.ch_names)}")
        print(f"📋 所有通道: {raw.ch_names}")
        
        # 只选择EEG通道
        raw = raw.pick_types(eeg=True)
        print(f"✅ 选择EEG通道后: {len(raw.ch_names)} 通道")
        print(f"📋 EEG通道: {raw.ch_names}")
        
        srate = int(raw.info["sfreq"])
        # 将数据从 V 转为 uV 发送
        data = (raw.get_data().astype(np.float32) * 1e6)  # shape: (n_chan, n_samp)，单位：uV
        channels = raw.ch_names
        
        print(f"✅ .set文件加载成功")
        print(f"   数据形状: {data.shape} (channels, samples)")
        print(f"   采样率: {srate} Hz")
        print(f"   通道数: {len(channels)}")
        print(f"   总时长: {data.shape[1] / srate:.2f} 秒 ({data.shape[1] / srate / 60:.2f} 分钟)")
        
        return data, srate, channels

    except KeyError as e:
        if e.args and e.args[0] == "nodatchans":
            print(
                "⚠️ MNE 读取失败: EEGLAB 结构 chaninfo 缺少 nodatchans（常见于旧版/非标准 .set）。"
                "尝试 scipy 直接读 MAT…"
            )
            data, srate, channels = _load_set_via_scipy_eeglab_mat(set_path)
            print(f"   总时长: {data.shape[1] / srate:.2f} 秒 ({data.shape[1] / srate / 60:.2f} 分钟)")
            return data, srate, channels
        raise
        
    except Exception as e:
        print(f"❌ 加载 .set 文件失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_file(file_path: Path):
    """
    根据文件扩展名自动选择加载方式
    
    Args:
        file_path: 文件路径
    
    Returns:
        data: EEG数据 (channels, samples)
        srate: 采样率
        channels: 通道名称列表
    """
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.npz':
        return load_npz_file(file_path)
    elif file_ext == '.set':
        return load_set_file(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}，只支持 .npz 和 .set 文件")


def create_lsl_stream(n_channels, srate, channels, stream_name, stream_type, source_id):
    """
    创建 LSL 流
    
    Args:
        n_channels: 通道数
        srate: 采样率
        channels: 通道名称列表
        stream_name: 流名称
        stream_type: 流类型
        source_id: 源ID
    
    Returns:
        outlet: LSL StreamOutlet 对象
    """
    print(f"\n🔧 创建LSL流: {n_channels} 通道, {srate} Hz")
    
    # 建立 EEG 流
    info = StreamInfo(
        name=stream_name,
        type=stream_type,
        channel_count=n_channels,
        nominal_srate=srate,
        channel_format="float32",
        source_id=source_id
    )
    
    # 写入通道标签
    chans = info.desc().append_child("channels")
    for ch in channels:
        c = chans.append_child("channel")
        c.append_child_value("label", ch)
        c.append_child_value("unit", "uV")  # 单位：微伏
        c.append_child_value("type", "EEG")
    
    outlet = StreamOutlet(info)
    
    return outlet


def stream_data(outlet, data, srate, chunk_size=CHUNK_SIZE):
    """
    流式传输数据
    
    Args:
        outlet: LSL StreamOutlet 对象
        data: EEG数据 (channels, samples)
        srate: 采样率
        chunk_size: 每次推送的样本数
    """
    n_chan, n_samp = data.shape
    
    print(f"\n🚀 开始流式传输数据...")
    print(f"   总样本数: {n_samp}")
    print(f"   采样率: {srate} Hz")
    print(f"   预计时长: {n_samp / srate:.2f} 秒")
    print(f"   Chunk大小: {chunk_size} 样本")
    
    # 数据统计信息
    print(f"\n📈 数据统计信息:")
    for i in range(min(5, n_chan)):  # 只显示前5个通道
        ch_data = data[i, :]
        print(f"   通道{i+1}: 范围 [{ch_data.min():8.3f}, {ch_data.max():8.3f}], "
              f"均值 {ch_data.mean():8.3f}, 标准差 {ch_data.std():8.3f}")
    if n_chan > 5:
        print(f"   ... (共 {n_chan} 个通道)")
    
    # 检查数据是否有变化
    data_variance = np.var(data, axis=1)
    print(f"\n🔍 数据变化检测:")
    flat_channels = sum(1 for v in data_variance if v < 1e-6)
    if flat_channels > 0:
        print(f"   ⚠️  {flat_channels} 个通道方差极小 - 可能显示为直线")
    else:
        print(f"   ✅ 所有通道方差正常")
    
    # 等待连接建立
    print(f"\n⏳ 等待 LSL 连接建立...")
    time.sleep(WAIT_TIME)
    
    t0 = time.time()
    i = 0
    
    try:
        while i < n_samp:
            j = min(i + chunk_size, n_samp)
            
            # 获取当前数据块
            chunk_data = data[:, i:j]  # shape: (channels, samples)
            
            # 检查数据块的有效性
            if chunk_data.shape[0] != n_chan:
                print(f"❌ 数据块通道数不匹配: 期望 {n_chan}, 实际 {chunk_data.shape[0]}")
                break
                
            if chunk_data.shape[1] == 0:
                print(f"❌ 数据块样本数为0")
                break
            
            # 转置数据：LSL 需要 (samples, channels) 格式
            chunk_transposed = chunk_data.T  # shape: (samples, channels)
            
            # 确保转置后的数据是 C 连续的
            chunk_transposed = np.ascontiguousarray(chunk_transposed, dtype=np.float32)
            
            # 推送数据
            outlet.push_chunk(chunk_transposed)
            
            # 按采样率控速（避免"瞬间喷完"）
            played_time = j / srate
            elapsed_time = time.time() - t0
            
            if played_time > elapsed_time:
                sleep_time = played_time - elapsed_time
                time.sleep(sleep_time)
            
            i = j
            
            # 显示进度
            if i % (chunk_size * 10) == 0 or i == n_samp:
                progress = (i / n_samp) * 100
                elapsed = time.time() - t0
                remaining_samples = n_samp - i
                remaining_time = remaining_samples / srate
                print(f"📈 进度: {progress:.1f}% ({i}/{n_samp}) | "
                      f"已用时: {elapsed:.1f}s | 剩余: {remaining_time:.1f}s")
                
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断传输")
    except Exception as e:
        print(f"❌ 传输错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("✅ 传输完成")
    
    # 统计信息
    elapsed_total = time.time() - t0
    print(f"\n📊 传输统计:")
    print(f"   总共传输: {i} 个样本")
    print(f"   实际用时: {elapsed_total:.2f} 秒")
    
    if elapsed_total > 0 and i > 0:
        print(f"   平均速度: {i / elapsed_total:.1f} 样本/秒")
        print(f"   速度比: {elapsed_total / (n_samp/srate):.2f}x")


def main():
    """
    主函数
    """
    print("=" * 80)
    print("📡 文件 → LSL 流输出工具（支持 .npz 和 .set 文件）")
    print("=" * 80)
    
    # 检查输入文件
    if not INPUT_FILE.exists():
        print(f"❌ 输入文件不存在: {INPUT_FILE}")
        print(f"   请修改脚本中的 INPUT_FILE 变量")
        return
    
    print(f"\n📋 配置信息:")
    print(f"   输入文件: {INPUT_FILE}")
    print(f"   文件类型: {INPUT_FILE.suffix}")
    print(f"   流名称: {STREAM_NAME}")
    print(f"   Chunk大小: {CHUNK_SIZE} 样本")
    
    try:
        # 加载数据
        data, srate, channels = load_file(INPUT_FILE)
        
        # 验证数据
        n_channels, n_samples = data.shape
        if n_channels == 0 or n_samples == 0:
            print("❌ 数据为空，无法传输")
            return
        
        if len(channels) != n_channels:
            print(f"⚠️ 警告: 通道名称数量 ({len(channels)}) 与数据通道数 ({n_channels}) 不匹配")
            print(f"   将使用默认通道名称")
            channels = [f"Ch{i+1}" for i in range(n_channels)]
        
        # 创建 LSL 流
        outlet = create_lsl_stream(n_channels, srate, channels, STREAM_NAME, STREAM_TYPE, SOURCE_ID)
        
        # 流式传输数据
        stream_data(outlet, data, srate, CHUNK_SIZE)
        
        print("\n" + "=" * 80)
        print("🎉 完成！")
        print("=" * 80)
        print(f"💡 提示: 你可以在 receiver.py 或其他 LSL 接收工具中查看这个流")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

