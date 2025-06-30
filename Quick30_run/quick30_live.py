import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylsl import StreamInlet, resolve_byprop

class StreamLSL:
    def __init__(self, type='EEG', timeout=5):
        print("🔍 Resolving EEG stream...")
        streams = resolve_byprop('type', type, timeout=timeout)
        if not streams:
            raise RuntimeError(f"❌ No stream found with type='{type}'")

        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.sfreq = int(info.nominal_srate())
        self.n_channels = info.channel_count()

        # 获取通道名称
        chs = info.desc().child('channels')
        ch = chs.child('channel')
        labels = []
        for i in range(self.n_channels):
            label = ch.child_value('label')
            labels.append(label if label else f"CH{i}")
            ch = ch.next_sibling()

        self.ch_names = labels
        print(f"✅ Connected to stream '{info.name()}' with {self.n_channels} channels at {self.sfreq} Hz")

    def pull_sample(self):
        sample, _ = self.inlet.pull_sample(timeout=0.1)
        return sample

def main():
    stream = StreamLSL()

    DISPLAY_SEC = 5
    REFRESH_HZ = 20
    CHANNEL_COUNT = 32
    SCALE_DIVISOR = 1e3
    sfreq = stream.sfreq
    n_samples = int(DISPLAY_SEC * sfreq)

    CHANNEL_COUNT = stream.n_channels

    print(stream.n_channels)

    CHANNEL_COUNT=20



    buffer = np.zeros((CHANNEL_COUNT, n_samples))

    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = np.arange(CHANNEL_COUNT) * 200
    lines = [ax.plot(buffer[i] + offsets[i])[0] for i in range(CHANNEL_COUNT)]

    ax.set_ylim(-300, offsets[-1] + 300)
    ax.set_xlim(0, n_samples)
    ax.set_yticks(offsets)
    ax.set_yticklabels([f"CH{i}" for i in range(CHANNEL_COUNT)])
    ax.set_title("📡 Quick30 Real-Time EEG (Rolling)")
    ax.set_xlabel("Samples")

    REFERENCE_NAME = "Cz"
    try:
        reference_idx = stream.ch_names.index(REFERENCE_NAME)
    except ValueError:
        reference_idx = 0
        print(f"⚠️ Reference channel '{REFERENCE_NAME}' not found, fallback to CH0")

    def update(frame):
        nonlocal buffer
        for _ in range(int(sfreq / REFRESH_HZ)):
            sample = stream.pull_sample()
            if sample:
                scaled = np.array(sample[:CHANNEL_COUNT]) / SCALE_DIVISOR
                ref_val = scaled[reference_idx]
                scaled = scaled - ref_val
                buffer = np.roll(buffer, -1, axis=1)
                buffer[:, -1] = scaled

        # 💡 计算并打印 RMS（参考后）
        for i in range(min(8, CHANNEL_COUNT)):
            rms_val = np.sqrt(np.mean(buffer[i] ** 2))
            ch_name = stream.ch_names[i] if i < len(stream.ch_names) else f"CH{i + 1}"
            print(f"RMS {ch_name}: {rms_val:.2f} µV", end=", ")
        print()

        for i, line in enumerate(lines):
            line.set_ydata(buffer[i] + offsets[i])
        return lines

    anim = animation.FuncAnimation(
        fig, update, interval=1000 / REFRESH_HZ,
        cache_frame_data=False
    )
    print("✅ 动画开始")
    plt.show()

if __name__ == "__main__":
    main()
