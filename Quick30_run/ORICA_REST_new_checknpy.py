import numpy as np

# 加载并查看
data = np.load('ORICA_results/SIM_STAT_16ch_3min_sphere_20250822_024147.npy')
print(f"形状: {data.shape}")
print(f"数据类型: {data.dtype}")
print(f"数据范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
print(f"前几行:\n{data[:3, :3]}")