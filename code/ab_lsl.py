#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从保存的 npz 文件读取处理后数据并通过 LSL 输出

用于查看和验证保存的处理后数据（ORICA + ASR处理后的数据）
"""

import time
import numpy as np
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet

# ========== 配置 ==========
# 默认文件路径（可以根据需要修改）
from paths import OUTPUT_DATA_ROOT

DEFAULT_SAVE_DIR = OUTPUT_DATA_ROOT
DEFAULT_FILE = DEFAULT_SAVE_DIR / "eegnet_data_1311_after_con_orica.npz"

# LSL 流配置
STREAM_NAME = "mybrain"  # 与 receiver_new.py 中的 stream_name 匹配
STREAM_TYPE = "EEG"
SOURCE_ID = "npz2lsl_001"

# 传输配置
CHUNK_SIZE = 50  # 一次推送的样本数（可调）
WAIT_TIME = 2  # 等待 LSL 连接建立的时间（秒）


def load_npz_data(npz_path):
    """
    加载 npz 文件中的数据
    
    Returns:
        data: (channels, samples) 数据数组
        channels: 通道名称列表
        sampling_rate: 采样率
        metadata: 其他元数据
    """
    print(f"🔍 正在读取文件: {npz_path}")
    
    if not npz_path.exists():
        print(f"❌ 文件不存在: {npz_path}")
        return None, None, None, None
    
    try:
        data_dict = np.load(npz_path, allow_pickle=True)
        
        # 获取数据
        if "data" in data_dict:
            data = data_dict["data"]
        else:
            print(f"❌ 未找到 'data' 字段")
            print(f"   可用字段: {list(data_dict.keys())}")
            return None, None, None, None
        
        # 获取通道名称
        if "channels" in data_dict:
            channels = data_dict["channels"].tolist() if isinstance(data_dict["channels"], np.ndarray) else data_dict["channels"]
        else:
            print(f"⚠️ 未找到 'channels' 字段，使用默认通道名")
            channels = [f"Ch{i+1}" for i in range(data.shape[0])]
        
        # 获取采样率
        if "sampling_rate" in data_dict:
            sampling_rate = int(data_dict["sampling_rate"])
        elif "srate" in data_dict:
            sampling_rate = int(data_dict["srate"])
        elif "fs" in data_dict:
            sampling_rate = int(data_dict["fs"])
        else:
            print(f"⚠️ 未找到采样率，使用默认值 500 Hz")
            sampling_rate = 500
        
        # 获取其他元数据
        metadata = {}
        for key in ["channel_indices", "duration", "total_samples"]:
            if key in data_dict:
                metadata[key] = data_dict[key]
        
        print(f"✅ 成功读取文件")
        print(f"📊 数据形状: {data.shape} (通道数, 样本数)")
        print(f"📊 采样率: {sampling_rate} Hz")
        print(f"📊 通道数: {len(channels)}")
        print(f"📊 总样本数: {data.shape[1]}")
        print(f"📊 总时长: {data.shape[1]/sampling_rate:.2f} 秒")
        print(f"📋 通道名称: {channels}")
        
        # 数据统计信息
        print("\n📈 数据统计信息:")
        for i, ch_name in enumerate(channels):
            ch_data = data[i, :]
            print(f"  {ch_name:>6}: 范围 [{ch_data.min():8.3f}, {ch_data.max():8.3f}], "
                  f"均值 {ch_data.mean():8.3f}, 标准差 {ch_data.std():8.3f}")
        
        # 检查数据变化
        data_variance = np.var(data, axis=1)
        print(f"\n🔍 数据变化检测:")
        for i, ch_name in enumerate(channels):
            if data_variance[i] < 1e-6:
                print(f"  ⚠️  {ch_name}: 方差极小 ({data_variance[i]:.2e}) - 可能显示为直线")
            else:
                print(f"  ✅  {ch_name}: 方差正常 ({data_variance[i]:.2e})")
        
        return data, channels, sampling_rate, metadata
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def create_lsl_stream(n_channels, sampling_rate, channel_labels, stream_name=STREAM_NAME, stream_type=STREAM_TYPE, source_id=SOURCE_ID):
    """
    创建 LSL 输出流
    
    Returns:
        outlet: StreamOutlet 对象
    """
    print(f"\n🔧 创建 LSL 流...")
    print(f"   流名称: {stream_name}")
    print(f"   流类型: {stream_type}")
    print(f"   通道数: {n_channels}")
    print(f"   采样率: {sampling_rate} Hz")
    
    # 创建流信息
    info = StreamInfo(
        name=stream_name,
        type=stream_type,
        channel_count=n_channels,
        nominal_srate=sampling_rate,
        channel_format="float32",
        source_id=source_id
    )
    
    # 添加通道信息
    chans = info.desc().append_child("channels")
    for ch_label in channel_labels:
        c = chans.append_child("channel")
        c.append_child_value("label", str(ch_label))
        c.append_child_value("unit", "uV")  # 单位：微伏
        c.append_child_value("type", "EEG")
    
    # 创建输出流
    outlet = StreamOutlet(info)
    print(f"✅ LSL 流创建成功")
    
    return outlet


def stream_data(outlet, data, sampling_rate, chunk_size=CHUNK_SIZE):
    """
    按照采样率流式传输数据
    
    Args:
        outlet: StreamOutlet 对象
        data: (channels, samples) 数据数组
        sampling_rate: 采样率
        chunk_size: 每次推送的样本数
    """
    n_channels, n_samples = data.shape
    
    print(f"\n🚀 开始流式传输数据...")
    print(f"   总样本数: {n_samples}")
    print(f"   采样率: {sampling_rate} Hz")
    print(f"   预计时长: {n_samples/sampling_rate:.2f} 秒")
    print(f"   块大小: {chunk_size} 样本")
    
    # 等待连接建立
    print(f"⏳ 等待 {WAIT_TIME} 秒让 LSL 连接建立...")
    time.sleep(WAIT_TIME)
    
    t0 = time.time()
    i = 0
    
    try:
        while i < n_samples:
            j = min(i + chunk_size, n_samples)
            
            # 获取当前数据块
            chunk_data = data[:, i:j]  # shape: (channels, samples)
            
            # 检查数据块的有效性
            if chunk_data.shape[0] != n_channels:
                print(f"❌ 数据块通道数不匹配: 期望 {n_channels}, 实际 {chunk_data.shape[0]}")
                break
                
            if chunk_data.shape[1] == 0:
                print(f"❌ 数据块样本数为0")
                break
            
            # 转置数据：LSL 需要 (samples, channels) 格式
            chunk_transposed = chunk_data.T  # shape: (samples, channels)
            
            # 确保数据是 C 连续的
            chunk_transposed = np.ascontiguousarray(chunk_transposed, dtype=np.float32)
            
            # 推送数据
            outlet.push_chunk(chunk_transposed)
            
            # 按采样率控速（避免"瞬间喷完"）
            played_time = j / sampling_rate
            elapsed_time = time.time() - t0
            
            if played_time > elapsed_time:
                sleep_time = played_time - elapsed_time
                time.sleep(sleep_time)
            
            i = j
            
            # 显示进度
            if i % (chunk_size * 10) == 0 or i == n_samples:
                progress = (i / n_samples) * 100
                elapsed = time.time() - t0
                remaining_samples = n_samples - i
                remaining_time = remaining_samples / sampling_rate
                print(f"📈 进度: {progress:.1f}% ({i}/{n_samples}) | "
                      f"已用时: {elapsed:.1f}s | 剩余: {remaining_time:.1f}s")
                
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断传输")
    except Exception as e:
        print(f"❌ 传输错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("✅ 传输完成")
    
    elapsed_total = time.time() - t0
    print(f"\n🎯 传输统计:")
    print(f"   总共传输: {i} 个样本")
    print(f"   实际用时: {elapsed_total:.2f} 秒")
    print(f"   预计用时: {n_samples/sampling_rate:.2f} 秒")
    
    if elapsed_total > 0 and i > 0:
        print(f"   平均速度: {i / elapsed_total:.1f} 样本/秒")
        print(f"   速度比: {elapsed_total / (n_samples/sampling_rate):.2f}x")


def main():
    """主函数"""
    print("=" * 80)
    print("📡 NPZ 文件 → LSL 流输出工具")
    print("=" * 80)
    
    # 1. 选择文件
    npz_file = DEFAULT_FILE
    
    # 如果默认文件不存在，尝试查找其他文件
    if not npz_file.exists():
        print(f"⚠️ 默认文件不存在: {npz_file}")
        if DEFAULT_SAVE_DIR.exists():
            npz_files = list(DEFAULT_SAVE_DIR.glob("*.npz"))
            if npz_files:
                print(f"📂 找到 {len(npz_files)} 个 npz 文件:")
                for i, f in enumerate(npz_files):
                    print(f"   [{i}] {f.name}")
                # 使用最新的文件
                npz_file = max(npz_files, key=lambda p: p.stat().st_mtime)
                print(f"✅ 使用最新文件: {npz_file.name}")
            else:
                print(f"❌ 未找到任何 npz 文件")
                return
        else:
            print(f"❌ 保存目录不存在: {DEFAULT_SAVE_DIR}")
            return
    
    # 2. 加载数据
    data, channels, sampling_rate, metadata = load_npz_data(npz_file)
    
    if data is None:
        print("❌ 无法加载数据，退出")
        return
    
    # 验证数据
    n_channels, n_samples = data.shape
    if n_channels == 0 or n_samples == 0:
        print("❌ 数据为空，无法传输")
        return
    
    if len(channels) != n_channels:
        print(f"⚠️ 警告: 通道名称数量 ({len(channels)}) 与数据通道数 ({n_channels}) 不匹配")
        print(f"   将使用默认通道名称")
        channels = [f"Ch{i+1}" for i in range(n_channels)]
    
    # 3. 创建 LSL 流
    outlet = create_lsl_stream(n_channels, sampling_rate, channels, STREAM_NAME, STREAM_TYPE, SOURCE_ID)
    
    # 4. 流式传输数据
    stream_data(outlet, data, sampling_rate, CHUNK_SIZE)
    
    print("\n" + "=" * 80)
    print("🎉 完成！")
    print("=" * 80)
    print(f"💡 提示: 你可以在其他 LSL 接收工具中查看这个流")
    print(f"   流名称: {STREAM_NAME}")
    print(f"   流类型: {STREAM_TYPE}")


if __name__ == "__main__":
    main()

