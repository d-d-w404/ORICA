from pylsl import resolve_byprop, StreamInlet
import time

# Step 1: 找到 Quick-30 的 EEG 流
print("🔍 正在寻找 EEG 类型的 LSL 流...")
streams = resolve_byprop('type', 'EEG', timeout=5)
if not streams:
    raise RuntimeError("❌ 未找到 EEG 流")

# Step 2: 创建数据入口
inlet = StreamInlet(streams[0])
info = inlet.info()

# Step 3: 读取通道名称
ch_names = []
desc = info.desc().child("channels")
ch = desc.child("channel")
for _ in range(info.channel_count()):
    ch_names.append(ch.child_value("label"))
    ch = ch.next_sibling()

print(f"✅ 已连接到流：{info.name()} ({info.source_id()})")
print(f"通道数：{len(ch_names)}，采样率：{info.nominal_srate()} Hz")

# Step 4: 实时读取样本并配对输出
while True:
    sample, timestamp = inlet.pull_sample(timeout=1.0)
    if sample:
        print(f"[{timestamp:.3f}] ", end="")
        for i in range(min(8, len(sample))):  # 只显示前 8 个通道
            print(f"{ch_names[i]}: {sample[i]:.2f} µV", end=", ")
        print("...")
    time.sleep(0.01)
