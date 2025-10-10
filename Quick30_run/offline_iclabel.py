# %% 依赖安装（如需）
# pip install mne

import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne_icalabel import label_components

# 禁用MNE的详细输出
mne.set_log_level('ERROR')

# ========= 用户区：改这里 =========
eeglab_set_path = r"D:\work\Python_Project\ORICA\temp_txt\Demo_EmotivEPOC_EyeOpen.set"           # <-- 改成你的 X.set 路径

# 如果你有Quick-30的自定义坐标，用 DigMontage；否则先用standard_1020近似
USE_CUSTOM_MONTAGE = False
custom_ch_pos = {
    # "Fp1": (-0.03, 0.09, 0.03),
    # "Fp2": ( 0.03, 0.09, 0.03),
    # ... 把你的28个通道都填上 (x,y,z) in meters
}
hp, lp = 1.0, 40.0         # 带通滤波
notch = 60.0               # 60Hz 电源工频；如在国内用 50.0
n_components = None        # None=自动取通道数；或手动设为28
bad_ic_thresh = dict(      # 自动剔除阈值（可按需调）
    muscle=0.70,
    eye=0.70,
    line_noise=0.70
)

# 读取 EEGLAB 数据（会自动读 .fdt）
raw = mne.io.read_raw_eeglab(eeglab_set_path, preload=True, verbose=False)

# 只保留 EEG 通道；标坏道请提前在 EEGLAB 里做或在 raw.info['bads'] 填
raw.pick_types(eeg=True, eog=False, ecg=False, stim=False, exclude="bads")

# 设置电极坐标
if USE_CUSTOM_MONTAGE and len(custom_ch_pos) > 0:
    mont = mne.channels.make_dig_montage(ch_pos=custom_ch_pos, coord_frame="head")
else:
    mont = mne.channels.make_standard_montage("standard_1020")

try:
    raw.set_montage(mont, on_missing="ignore")
    print(f"✅ 成功设置montage: {len(raw.ch_names)} 个通道")
except Exception as e:
    print(f"⚠️ 设置montage失败: {e}")
    # 创建简单的圆形montage作为备用
    n_chans = len(raw.ch_names)
    angles = np.linspace(0, 2*np.pi, n_chans, endpoint=False)
    ch_pos = {}
    for i, ch_name in enumerate(raw.ch_names):
        ch_pos[ch_name] = (np.cos(angles[i]) * 0.1, np.sin(angles[i]) * 0.1, 0.0)
    mont = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    raw.set_montage(mont)
    print(f"✅ 使用备用圆形montage")

# 参考与滤波
raw.set_eeg_reference("average", projection=False)
raw.notch_filter(notch, picks="eeg", verbose=False)
raw.filter(hp, lp, picks="eeg", verbose=False)

# （可选）重采样，加快 ICA
if raw.info["sfreq"] > 256:
    raw.resample(256, npad="auto")

# ICA
if n_components is None:
    n_components = len(mne.pick_types(raw.info, eeg=True, exclude="bads"))
ica = ICA(n_components=n_components, method="fastica", random_state=97, max_iter="auto")
ica.fit(raw, picks="eeg")

# ICLabel
labels, probs = label_components(raw, ica, method="iclabel")  # labels: ['brain','muscle',...]
class_names = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']

# 检查ICLabel是否成功
if isinstance(labels, str) or (isinstance(labels, list) and len(labels) == 0):
    print("⚠️ ICLabel分类失败，尝试替代方法...")
    # 使用简化的手动分类方法
    labels = []
    probs = []
    
    for i in range(n_components):
        # 获取成分数据
        comp_data = ica.get_components()[:, i]
        
        # 简单的启发式分类
        frontal_power = np.mean(np.abs(comp_data[:4]))  # 前额区域功率
        temporal_power = np.mean(np.abs(comp_data[-4:]))  # 颞部区域功率
        
        if frontal_power > 0.3:  # 前额功率高，可能是眼动
            labels.append('eye')
            probs.append([0.1, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0])  # eye概率高
        elif temporal_power > 0.3:  # 颞部功率高，可能是肌电
            labels.append('muscle')
            probs.append([0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0])  # muscle概率高
        else:  # 其他情况，标记为脑电
            labels.append('brain')
            probs.append([0.7, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1])  # brain概率高
    
    probs = np.array(probs)
    print(f"✅ 使用替代分类方法，识别出: {set(labels)}")
else:
    # 正常的ICLabel结果处理
    probs = np.array(probs)
    if probs.ndim == 0:
        print("⚠️ probs是标量，ICLabel失败")
        labels = ['other'] * n_components
        probs = np.ones((n_components, len(class_names))) / len(class_names)
    elif probs.ndim == 1:
        probs = probs.reshape(1, -1)

# 依据阈值自动选择伪影 IC
try:
    idx_muscle = np.where(probs[:, class_names.index('muscle')] > bad_ic_thresh['muscle'])[0]
    idx_eye    = np.where(probs[:, class_names.index('eye')]    > bad_ic_thresh['eye'])[0]
    idx_line   = np.where(probs[:, class_names.index('line_noise')] > bad_ic_thresh['line_noise'])[0]
    bad_idx = np.unique(np.concatenate([idx_muscle, idx_eye, idx_line]))
    ica.exclude = list(map(int, bad_idx))
except Exception as e:
    print(f"⚠️ 自动选择伪影成分失败: {e}")
    ica.exclude = []

# 输出ICLabel结果
print("=== ICLabel 分类结果 ===")
print(f"检测到的伪影成分: {ica.exclude}")
print("\n各成分详细分类:")
for k in range(len(labels)):
    top3 = np.argsort(probs[k])[::-1][:3]
    status = "❌ 伪影" if k in ica.exclude else "✅ 保留"
    print(f"IC {k:02d} {status}: {[(class_names[i], f'{probs[k,i]:.3f}') for i in top3]}")

# 统计各类成分数量
label_counts = {}
for label in labels:
    label_counts[label] = label_counts.get(label, 0) + 1
print(f"\n成分分布: {label_counts}")

# 创建并显示topomap
print("\n正在生成topomap图...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(min(n_components, 12)):  # 最多显示12个成分
    ax = axes[i]
    
    # 绘制topomap
    try:
        im, _ = mne.viz.plot_topomap(
            ica.get_components()[:, i], 
            raw.info, 
            axes=ax, 
            show=False,
            cmap='RdBu_r'
        )
        
        # 设置标题，包含分类结果
        label = labels[i]
        prob = probs[i, class_names.index(label)]
        status = "❌" if i in ica.exclude else "✅"
        ax.set_title(f'IC {i} {status}\n{label} ({prob:.3f})', fontsize=10)
        
    except Exception as e:
        ax.set_title(f'IC {i} - 绘制失败', fontsize=10)
        ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)

# 隐藏多余的子图
for i in range(min(n_components, 12), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('ICLabel 分类结果 - Topomap', fontsize=16)
plt.tight_layout()
plt.show()

print("✅ 分析完成！")
