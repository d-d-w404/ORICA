#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间集中度计算详解
"""

import numpy as np
import matplotlib.pyplot as plt

def explain_spatial_concentration():
    """详细解释空间集中度计算"""
    print("=== 空间集中度计算详解 ===\n")
    
    # 模拟解混矩阵W (3个IC, 5个通道)
    print("📊 解混矩阵W示例 (3个IC, 5个通道):")
    W = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],  # IC1: 集中在通道1
        [0.1, 0.2, 0.3, 0.2, 0.2],     # IC2: 相对均匀
        [0.02, 0.03, 0.05, 0.1, 0.8]   # IC3: 集中在通道5
    ])
    
    print("W矩阵:")
    print(W)
    print(f"形状: {W.shape}")
    
    # 计算绝对值
    W_abs = np.abs(W)
    print(f"\n📐 W的绝对值:")
    print(W_abs)
    
    # 计算每个IC的最大权重
    max_weights = np.max(W_abs, axis=1)
    print(f"\n🔝 每个IC的最大权重:")
    print(f"IC1: {max_weights[0]:.3f}")
    print(f"IC2: {max_weights[1]:.3f}")
    print(f"IC3: {max_weights[2]:.3f}")
    
    # 计算每个IC的平均权重
    mean_weights = np.mean(W_abs, axis=1)
    print(f"\n📊 每个IC的平均权重:")
    print(f"IC1: {mean_weights[0]:.3f}")
    print(f"IC2: {mean_weights[1]:.3f}")
    print(f"IC3: {mean_weights[2]:.3f}")
    
    # 计算集中度比率
    concentration_ratios = max_weights / mean_weights
    print(f"\n⚖️ 集中度比率 (最大权重/平均权重):")
    print(f"IC1: {concentration_ratios[0]:.3f}")
    print(f"IC2: {concentration_ratios[1]:.3f}")
    print(f"IC3: {concentration_ratios[2]:.3f}")
    
    # 计算总体空间集中度
    spatial_concentration = np.mean(concentration_ratios)
    print(f"\n🎯 总体空间集中度: {spatial_concentration:.3f}")
    
    return W, spatial_concentration

def compare_different_distributions():
    """比较不同空间分布的集中度"""
    print("\n=== 不同空间分布对比 ===\n")
    
    # 1. 高度集中分布
    W_concentrated = np.array([
        [0.95, 0.02, 0.01, 0.01, 0.01],  # 几乎只在通道1
        [0.01, 0.95, 0.02, 0.01, 0.01],  # 几乎只在通道2
        [0.01, 0.01, 0.95, 0.02, 0.01]   # 几乎只在通道3
    ])
    
    # 2. 中等集中分布
    W_moderate = np.array([
        [0.6, 0.2, 0.1, 0.05, 0.05],     # 主要在通道1
        [0.1, 0.5, 0.2, 0.1, 0.1],       # 主要在通道2
        [0.05, 0.1, 0.6, 0.15, 0.1]      # 主要在通道3
    ])
    
    # 3. 均匀分布
    W_uniform = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],       # 完全均匀
        [0.2, 0.2, 0.2, 0.2, 0.2],       # 完全均匀
        [0.2, 0.2, 0.2, 0.2, 0.2]        # 完全均匀
    ])
    
    distributions = {
        '高度集中': W_concentrated,
        '中等集中': W_moderate,
        '均匀分布': W_uniform
    }
    
    results = {}
    for name, W in distributions.items():
        W_abs = np.abs(W)
        max_weights = np.max(W_abs, axis=1)
        mean_weights = np.mean(W_abs, axis=1)
        concentration_ratios = max_weights / mean_weights
        spatial_concentration = np.mean(concentration_ratios)
        
        results[name] = spatial_concentration
        
        print(f"📊 {name}:")
        print(f"  解混矩阵:")
        print(f"  {W}")
        print(f"  空间集中度: {spatial_concentration:.3f}")
        print()
    
    return results

def visualize_spatial_concentration():
    """可视化空间集中度"""
    print("\n=== 空间集中度可视化 ===\n")
    
    # 生成不同集中度的W矩阵
    n_components = 3
    n_channels = 8
    
    # 1. 高度集中
    W_high = np.random.rand(n_components, n_channels) * 0.1
    for i in range(n_components):
        W_high[i, i] = 0.9  # 对角线元素很大
    
    # 2. 中等集中
    W_medium = np.random.rand(n_components, n_channels) * 0.3
    for i in range(n_components):
        W_medium[i, i] = 0.5  # 对角线元素中等
    
    # 3. 低集中度
    W_low = np.random.rand(n_components, n_channels)
    W_low = W_low / np.sum(W_low, axis=1, keepdims=True)  # 归一化
    
    # 计算集中度
    def compute_concentration(W):
        W_abs = np.abs(W)
        max_weights = np.max(W_abs, axis=1)
        mean_weights = np.mean(W_abs, axis=1)
        concentration_ratios = max_weights / mean_weights
        return np.mean(concentration_ratios)
    
    conc_high = compute_concentration(W_high)
    conc_medium = compute_concentration(W_medium)
    conc_low = compute_concentration(W_low)
    
    # 绘制热力图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    matrices = [W_high, W_medium, W_low]
    titles = [f'高集中度\n({conc_high:.2f})', 
              f'中等集中度\n({conc_medium:.2f})', 
              f'低集中度\n({conc_low:.2f})']
    
    for i, (W, title) in enumerate(zip(matrices, titles)):
        im = axes[i].imshow(np.abs(W), cmap='viridis', aspect='auto')
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel('通道')
        axes[i].set_ylabel('IC')
        axes[i].set_xticks(range(n_channels))
        axes[i].set_yticks(range(n_components))
        
        # 添加数值标注
        for j in range(n_components):
            for k in range(n_channels):
                text = axes[i].text(k, j, f'{np.abs(W[j, k]):.2f}',
                                   ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('spatial_concentration_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 集中度对比:")
    print(f"  高集中度: {conc_high:.3f}")
    print(f"  中等集中度: {conc_medium:.3f}")
    print(f"  低集中度: {conc_low:.3f}")

def interpret_concentration_values():
    """解释集中度数值的含义"""
    print("\n=== 集中度数值解释 ===\n")
    
    print("🎯 空间集中度的数值含义:")
    print()
    print("1.0 - 1.5: 极低集中度")
    print("  • IC在所有通道上几乎均匀分布")
    print("  • 每个通道的贡献基本相同")
    print("  • 理想情况，但可能表示分离效果不佳")
    print()
    print("1.5 - 2.5: 低集中度")
    print("  • IC分布相对均匀")
    print("  • 有轻微的空间集中")
    print("  • 通常表示良好的分离效果")
    print()
    print("2.5 - 4.0: 中等集中度")
    print("  • IC在少数几个通道上集中")
    print("  • 存在明显的空间模式")
    print("  • 需要关注是否过度集中")
    print()
    print("4.0 - 8.0: 高集中度")
    print("  • IC高度集中在1-2个通道")
    print("  • 可能存在伪影或噪声")
    print("  • 需要检查数据质量和算法参数")
    print()
    print("> 8.0: 极高集中度")
    print("  • IC几乎只来自单个通道")
    print("  • 很可能存在数值问题或算法故障")
    print("  • 需要立即检查和修复")
    print()
    
    print("⚠️ 注意事项:")
    print("• 集中度过低可能表示分离不充分")
    print("• 集中度过高可能表示伪影或噪声")
    print("• 理想范围通常在1.5-3.0之间")
    print("• 需要结合其他指标综合判断")

def practical_recommendations():
    """实际应用建议"""
    print("\n=== 实际应用建议 ===\n")
    
    print("🔧 当空间集中度过高时:")
    print("1. 检查学习率是否合适")
    print("2. 调整正交化频率")
    print("3. 使用增强初始化")
    print("4. 添加空间多样性约束")
    print("5. 检查数据预处理")
    print()
    
    print("🔧 当空间集中度过低时:")
    print("1. 增加学习率")
    print("2. 减少正交化频率")
    print("3. 检查数据质量")
    print("4. 调整非线性函数")
    print("5. 增加训练时间")
    print()
    
    print("📊 监控建议:")
    print("• 定期检查空间集中度变化")
    print("• 结合熵和互信息指标")
    print("• 观察收敛趋势")
    print("• 记录最佳参数组合")

if __name__ == "__main__":
    # 详细解释计算过程
    W, conc = explain_spatial_concentration()
    
    # 比较不同分布
    results = compare_different_distributions()
    
    # 可视化
    visualize_spatial_concentration()
    
    # 解释数值含义
    interpret_concentration_values()
    
    # 实际建议
    practical_recommendations()
    
    print("\n✅ 空间集中度分析完成！") 