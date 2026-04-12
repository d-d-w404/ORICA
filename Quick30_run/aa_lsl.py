# pip install mne pylsl
import time
import numpy as np
import mne
from pylsl import StreamInfo, StreamOutlet

#SET_FILE = r"D:\LaparoscopicTrainingEEGData\EEG data-20250520T175836Z-1-001\EEG data\laparoscopic_003_EEGmerged.set"  # ← 替换为你的 .set
#SET_FILE = r"D:\work\matlab_project\Lap_data2\00_rawdata\laparoscopic_1307_EEGmerged.set"


#raw data
#SET_FILE = r"D:\work\Python_Project\ORICA\Quick30_run\processed_data_saves_threshold\laparoscopic_1295_EEGmerged.set"


SET_FILE =r"D:\work\matlab_project\Lap_data2\00_rawdata\laparoscopic_1309_EEGmerged.set"

#SET_FILE = r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_1309_EEGasr.set"



# #001 asr
# SET_FILE = r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_001_EEGasr.set"

# #003 asr
# SET_FILE = r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_003_EEGasr.set"

# #1271 asr
# SET_FILE = r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_1271_EEGasr.set"

# #1284 asr
# SET_FILE = r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_1284_EEGasr.set"


# #1285 asr
# SET_FILE = r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_1295_EEGasr.set"
# #
#
# #1295 asr
# SET_FILE = r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_1295_EEGasr.set"

# #1307 asr
#SET_FILE = r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_1307_EEGasr.set"
#
# #1309 asr
# SET_FILE = r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_1309_EEGasr.set"

#1311 asr
#SET_FILE = r"D:\work\matlab_project\Lap_data2\05_processed_data\laparoscopic_1311_EEGasr.set"


# eye open
#SET_FILE=r"D:\work\matlab_project\REST_original\data\Quick30_Shawn_EyeOpen.set"

#SET_FILE =r"D:\work\matlab_project\REST\data\Quick30_Shawn_EyeOpen.set"
#SET_FILE = r"D:\LaparoscopicTrainingEEGData\Preprocessed Data-20250520T174502Z-1-001\Preprocessed Data\laparoscopic_001_EEGiclabel24Chans.set"
#SET_FILE = r"D:\LaparoscopicTrainingEEGData\EEG data-20250520T175836Z-1-001\EEG data\EEG data05_processed_data\laparoscopic_001_EEGasr.set"

#"D:\work\matlab_project\REST\data\Quick30_Shawn_EyeOpen.set"
#"D:\LaparoscopicTrainingEEGData\Preprocessed Data-20250520T174502Z-1-001\Preprocessed Data\laparoscopic_001_EEGiclabel24Chans.set"
#"D:\LaparoscopicTrainingEEGData\EEG data-20250520T175836Z-1-001\EEG data\EEG data05_processed_data\laparoscopic_001_EEGasr.set"
print(f"🔍 正在读取文件: {SET_FILE}")

# 读取数据
try:
    raw = mne.io.read_raw_eeglab(SET_FILE, preload=True)
    print(f"✅ 成功读取文件")
    print(f"📊 原始通道数: {len(raw.ch_names)}")
    print(f"📋 所有通道: {raw.ch_names}")
    
    # 只选择EEG通道
    raw = raw.pick_types(eeg=True)
    print(f"✅ 选择EEG通道后: {len(raw.ch_names)} 通道")
    print(f"📋 EEG通道: {raw.ch_names}")
    
except Exception as e:
    print(f"❌ 读取文件失败: {e}")
    exit(1)

sr = int(raw.info["sfreq"])
# 将数据从 V 转为 uV 发送
data = (raw.get_data().astype(np.float32) * 1e6)          # shape: (n_chan, n_samp)，单位：uV
n_chan, n_samp = data.shape

print(f"📊 最终数据: {n_chan} 通道, {n_samp} 样本, 采样率 {sr} Hz (单位：uV)")
print(f"📋 通道名称: {raw.ch_names}")

# 🔍 添加数据统计信息，诊断可视化问题
print("\n📈 数据统计信息:")
for i, ch_name in enumerate(raw.ch_names):
    ch_data = data[i, :]
    print(f"  {ch_name:>6}: 范围 [{ch_data.min():8.3f}, {ch_data.max():8.3f}], "
          f"均值 {ch_data.mean():8.3f}, 标准差 {ch_data.std():8.3f}")

# 检查数据是否有变化
data_variance = np.var(data, axis=1)
print(f"\n🔍 数据变化检测:")
for i, ch_name in enumerate(raw.ch_names):
    if data_variance[i] < 1e-6:
        print(f"  ⚠️  {ch_name}: 方差极小 ({data_variance[i]:.2e}) - 可能显示为直线")
    else:
        print(f"  ✅  {ch_name}: 方差正常 ({data_variance[i]:.2e})")

# 验证数据一致性
if len(raw.ch_names) != n_chan:
    print(f"❌ 通道数量不一致: ch_names={len(raw.ch_names)}, data.shape[0]={n_chan}")
    exit(1)

# 检查数据完整性
if n_chan == 0 or n_samp == 0:
    print("❌ 数据为空，无法传输")
    exit(1)

print(f"🔒 通道数量验证通过: {n_chan}")

# 建立 EEG 流（name/type/source_id 可自定义，注意唯一）
# 关键修复：确保 channel_count 与实际数据通道数一致
info = StreamInfo(name="mybrain", type="EEG",  # 改为与 stream_receiver.py 匹配的名称
                  channel_count=n_chan, nominal_srate=sr,  # 使用实际的通道数
                  channel_format="float32", source_id="py_set2lsl_001")

print(f"🔧 创建LSL流: {n_chan} 通道, {sr} Hz")

# 写入通道标签（可选）
chans = info.desc().append_child("channels")
for ch in raw.ch_names:
    c = chans.append_child("channel")
    c.append_child_value("label", ch)
    c.append_child_value("unit", "uV")  # 单位：微伏
    c.append_child_value("type", "EEG")

outlet = StreamOutlet(info)

# 等待连接建立
print("⏳ 等待 LSL 连接建立...")
time.sleep(2)

chunk_size = 50  # 一次推送样本数（可调）
t0 = time.time()
i = 0
print(f"🚀 开始流式传输 {SET_FILE} at {sr} Hz ...")

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
        
        # 关键修复：正确的数据转置和连续性处理
        # 1. 先转置数据
        chunk_transposed = chunk_data.T  # shape: (samples, channels)
        
        # 2. 确保转置后的数据是 C 连续的
        chunk_transposed = np.ascontiguousarray(chunk_transposed)
        
        # 推送数据
        outlet.push_chunk(chunk_transposed)
        
        # 按采样率控速（避免"瞬间喷完"）
        played_time = j / sr
        elapsed_time = time.time() - t0
        
        if played_time > elapsed_time:
            sleep_time = played_time - elapsed_time
            time.sleep(sleep_time)
        
        i = j
        
        # 显示进度
        if i % (chunk_size * 10) == 0:
            progress = (i / n_samp) * 100
            print(f"📈 进度: {progress:.1f}% ({i}/{n_samp})")
            
except KeyboardInterrupt:
    print("\n⏹️ 用户中断传输")
except Exception as e:
    print(f"❌ 传输错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("✅ 传输完成.")

print(f"🎯 总共传输了 {i} 个样本")
elapsed_total = time.time() - t0
print(f"⏱️ 实际用时: {elapsed_total:.2f} 秒")

# 避免除零错误
if elapsed_total > 0 and i > 0:
    print(f"📊 平均速度: {i / elapsed_total:.1f} 样本/秒")
else:
    print("📊 平均速度: 无法计算（传输时间或样本数为0）")
