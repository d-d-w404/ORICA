import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylsl import StreamInlet, resolve_byprop
from collections import deque

# 连接 LSL EEG 流
print("🔍 正在寻找 EEG 类型的 LSL 流...")
streams = resolve_byprop('type', 'EEG', timeout=5)
if not streams:
    raise RuntimeError("❌ 未找到 EEG 流")

inlet = StreamInlet(streams[0])
info = inlet.info()
sfreq = int(info.nominal_srate())
channel_count = info.channel_count()

# 读取通道标签
ch_names = []
desc = info.desc().child("channels")
ch = desc.child("channel")
for _ in range(channel_count):
    ch_names.append(ch.child_value("label"))
    ch = ch.next_sibling()

print(f"✅ 已连接到流：{info.name()}，通道数：{channel_count}，采样率：{sfreq} Hz")

# 设置平均参考函数
def apply_reference(sample_matrix, mode="average"):
    if mode == "average":
        avg = np.mean(sample_matrix, axis=0, keepdims=True)
        return sample_matrix - avg
    else:
        return sample_matrix

# 创建滚动缓冲区
buffer_secs = 5
buffer_size = buffer_secs * sfreq
data_buffer = [deque([0.0]*buffer_size, maxlen=buffer_size) for _ in range(min(8, channel_count))]
x = np.linspace(-buffer_secs, 0, buffer_size)

# 设置绘图
fig, ax = plt.subplots()
lines = []
for i in range(len(data_buffer)):
    line, = ax.plot(x, list(data_buffer[i]), label=ch_names[i])
    lines.append(line)

ax.set_ylim(-100, 100)
ax.set_xlim(-buffer_secs, 0)
ax.set_xlabel("时间 (s)")
ax.set_ylabel("电压 (µV)")
ax.set_title("🧠 实时参考后 EEG 波形 (前 8 通道)")
ax.legend(loc='upper right')

# 实时更新函数
sample_matrix = []

def update(frame):
    global sample_matrix
    sample, timestamp = inlet.pull_sample(timeout=0.0)
    if sample:
        sample_matrix.append(sample)
        if len(sample_matrix) >= 5:  # 每 5 个点做一次参考
            data_array = np.array(sample_matrix)
            ref_data = apply_reference(data_array, mode="average")
            for i in range(len(data_buffer)):
                data_buffer[i].append(ref_data[-1][i])
                lines[i].set_ydata(data_buffer[i])
            sample_matrix = []
    return lines

# 启动动画
ani = animation.FuncAnimation(fig, update, interval=20, blit=True)
plt.tight_layout()
plt.show()
