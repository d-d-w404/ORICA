#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORICA 源信号可视化工具
接收 ORICA 源信号，进行 ICLabel 分类并显示 topomap
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import scipy.io
import argparse
import os
from pathlib import Path

class ORICAVisualizer:
    def __init__(self, n_channels=14, srate=128):
        """
        初始化 ORICA 可视化器
        
        Args:
            n_channels: 通道数
            srate: 采样率
        """
        self.n_channels = n_channels
        self.srate = srate
        self.chan_labels = [f'Ch{i+1:02d}' for i in range(n_channels)]
        
        # 创建标准 10-20 电极位置
        self.setup_montage()
        
    def setup_montage(self):
        """设置电极位置"""
        try:
            # 尝试使用标准 10-20 系统
            self.montage = mne.channels.make_standard_montage("standard_1020")
            # 如果通道数不匹配，使用自定义位置
            if len(self.montage.ch_names) != self.n_channels:
                self.create_custom_montage()
        except:
            self.create_custom_montage()
    
    def create_custom_montage(self):
        """创建自定义电极位置（圆形排列）"""
        print(f"创建 {self.n_channels} 通道的自定义电极位置...")
        # 在单位圆上均匀分布电极
        angles = np.linspace(0, 2*np.pi, self.n_channels, endpoint=False)
        positions = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(self.n_channels)])
        
        # 创建自定义 montage
        self.montage = mne.channels.make_dig_montage(
            ch_pos=dict(zip(self.chan_labels, positions)),
            coord_frame='head'
        )
    
    def load_orica_data(self, sources_file, mixing_file=None, icaweights_file=None, icasphere_file=None):
        """
        加载 ORICA 数据
        
        Args:
            sources_file: 源信号文件 (.npy, .mat, .npz)
            mixing_file: 混合矩阵文件 (可选)
            icaweights_file: ICA 权重文件 (可选)
            icasphere_file: ICA 球化文件 (可选)
        """
        print(f"加载源信号: {sources_file}")
        
        # 加载源信号
        if sources_file.endswith('.npy'):
            self.sources = np.load(sources_file)
        elif sources_file.endswith('.mat'):
            data = scipy.io.loadmat(sources_file)
            # 尝试不同的键名
            for key in ['sources', 'icaact', 'data', 'X']:
                if key in data:
                    self.sources = data[key]
                    break
            else:
                raise ValueError("未找到源信号数据")
        elif sources_file.endswith('.npz'):
            data = np.load(sources_file)
            self.sources = data['sources']
        else:
            raise ValueError("不支持的文件格式")
        
        print(f"源信号形状: {self.sources.shape}")
        
        # 确保源信号是 (samples, components) 格式
        if self.sources.shape[0] < self.sources.shape[1]:
            self.sources = self.sources.T
        print(f"调整后源信号形状: {self.sources.shape}")
        
        # 加载混合矩阵
        self.mixing_matrix = None
        if mixing_file:
            self.load_mixing_matrix(mixing_file)
        elif icaweights_file and icasphere_file:
            self.load_ica_matrices(icaweights_file, icasphere_file)
        else:
            print("⚠️ 未提供混合矩阵，将使用随机矩阵进行演示")
            self.create_demo_mixing_matrix()
    
    def load_mixing_matrix(self, mixing_file):
        """加载混合矩阵"""
        print(f"加载混合矩阵: {mixing_file}")
        if mixing_file.endswith('.npy'):
            self.mixing_matrix = np.load(mixing_file)
        elif mixing_file.endswith('.mat'):
            data = scipy.io.loadmat(mixing_file)
            self.mixing_matrix = data['A'] if 'A' in data else data['mixing_matrix']
        elif mixing_file.endswith('.npz'):
            data = np.load(mixing_file)
            self.mixing_matrix = data['A'] if 'A' in data else data['mixing_matrix']
        
        print(f"混合矩阵形状: {self.mixing_matrix.shape}")
    
    def load_ica_matrices(self, icaweights_file, icasphere_file):
        """从 ICA 权重和球化矩阵计算混合矩阵"""
        print(f"加载 ICA 矩阵: {icaweights_file}, {icasphere_file}")
        
        # 加载权重矩阵
        if icaweights_file.endswith('.npy'):
            icaweights = np.load(icaweights_file)
        elif icaweights_file.endswith('.mat'):
            data = scipy.io.loadmat(icaweights_file)
            icaweights = data['icaweights']
        elif icaweights_file.endswith('.npz'):
            data = np.load(icaweights_file)
            icaweights = data['icaweights']
        
        # 加载球化矩阵
        if icasphere_file.endswith('.npy'):
            icasphere = np.load(icasphere_file)
        elif icasphere_file.endswith('.mat'):
            data = scipy.io.loadmat(icasphere_file)
            icasphere = data['icasphere']
        elif icasphere_file.endswith('.npz'):
            data = np.load(icasphere_file)
            icasphere = data['icasphere']
        
        # 计算混合矩阵 A = pinv(W)
        self.mixing_matrix = np.linalg.pinv(icaweights)
        print(f"混合矩阵形状: {self.mixing_matrix.shape}")
    
    def create_demo_mixing_matrix(self):
        """创建演示用混合矩阵"""
        n_comp = self.sources.shape[1]
        # 创建随机正交矩阵
        A = np.random.randn(self.n_channels, n_comp)
        U, _, _ = np.linalg.svd(A, full_matrices=False)
        self.mixing_matrix = U
        print(f"创建演示混合矩阵: {self.mixing_matrix.shape}")
    
    def classify_sources_directly(self, data, sources, mixing_matrix, chan_names, srate, threshold=0.8, n_comp=None):
        """
        直接对源信号进行 ICLabel 分类
        
        Args:
            data: 原始通道数据 (channels, samples)
            sources: 源信号 (components, samples)
            mixing_matrix: 混合矩阵 (channels, components)
            chan_names: 通道名称列表
            srate: 采样率
            threshold: 分类阈值
            n_comp: 成分数量
            
        Returns:
            ic_probs: 分类概率
            ic_labels: 分类标签
        """
        try:
            print("开始 ICLabel 分类...")
            
            # 创建 MNE Raw 对象
            info = mne.create_info(chan_names, srate, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            raw.set_montage(self.montage)
            
            # 创建临时 ICA 对象
            ica = ICA(n_components=n_comp, method='infomax')
            ica.n_components_ = n_comp
            ica.current_fit = 'raw'
            ica.ch_names = chan_names
            ica._ica_names = [f'IC {k:03d}' for k in range(n_comp)]
            
            # 设置混合矩阵
            ica.mixing_matrix_ = mixing_matrix
            ica.unmixing_matrix_ = np.linalg.pinv(mixing_matrix)
            ica.pca_explained_variance_ = np.ones(n_comp)
            ica.pca_mean_ = np.zeros(len(chan_names))
            ica.pca_components_ = np.eye(n_comp, len(chan_names))
            
            # 运行 ICLabel
            labels = label_components(raw, ica, method='iclabel')
            
            ic_probs = labels.get('y_pred_proba', None)
            ic_labels = labels.get('y_pred', None)
            if ic_labels is None and 'labels' in labels:
                ic_labels = labels['labels']
            
            print(f"ICLabel 分类完成: {len(ic_labels) if ic_labels is not None else 0} 个成分")
            return ic_probs, ic_labels
            
        except Exception as e:
            print(f"ICLabel 分类失败: {e}")
            return None, None
    
    def create_topomap(self, ic_probs=None, ic_labels=None, save_path=None):
        """
        创建 topomap 可视化
        
        Args:
            ic_probs: 分类概率
            ic_labels: 分类标签
            save_path: 保存路径
        """
        print("创建 topomap 可视化...")
        
        n_comp = self.sources.shape[1]
        n_cols = min(4, n_comp)
        n_rows = (n_comp + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_comp == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 创建 MNE info 对象
        info = mne.create_info(self.chan_labels, self.srate, ch_types='eeg')
        info.set_montage(self.montage)
        
        for i in range(n_comp):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # 计算该成分的 topomap
            component_data = self.mixing_matrix[:, i]  # 第 i 个成分的混合系数
            
            # 绘制 topomap
            im, _ = mne.viz.plot_topomap(
                component_data, 
                info, 
                axes=ax, 
                show=False,
                cmap='RdBu_r',
                contours=6
            )
            
            # 设置标题
            if ic_labels is not None and i < len(ic_labels):
                label = ic_labels[i]
                if ic_probs is not None and i < ic_probs.shape[0]:
                    prob = ic_probs[i, np.argmax(ic_probs[i])]
                    title = f'IC {i}: {label} ({prob:.2f})'
                else:
                    title = f'IC {i}: {label}'
            else:
                title = f'IC {i}'
            
            ax.set_title(title, fontsize=10)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 隐藏多余的子图
        for i in range(n_comp, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Topomap 已保存到: {save_path}")
        
        plt.show()
    
    def run_analysis(self, sources_file, mixing_file=None, icaweights_file=None, icasphere_file=None, save_path=None):
        """
        运行完整分析流程
        
        Args:
            sources_file: 源信号文件
            mixing_file: 混合矩阵文件
            icaweights_file: ICA 权重文件
            icasphere_file: ICA 球化文件
            save_path: 保存路径
        """
        print("=" * 50)
        print("ORICA 源信号可视化分析")
        print("=" * 50)
        
        # 加载数据
        self.load_orica_data(sources_file, mixing_file, icaweights_file, icasphere_file)
        
        # 准备数据用于 ICLabel
        data = self.mixing_matrix @ self.sources.T  # 重构原始通道数据
        sources = self.sources.T  # (components, samples)
        
        # 运行 ICLabel 分类
        ic_probs, ic_labels = self.classify_sources_directly(
            data=data,
            sources=sources,
            mixing_matrix=self.mixing_matrix,
            chan_names=self.chan_labels,
            srate=self.srate,
            n_comp=self.sources.shape[1]
        )
        
        # 打印分类结果
        if ic_labels is not None:
            print("\n分类结果:")
            for i, label in enumerate(ic_labels):
                if ic_probs is not None and i < ic_probs.shape[0]:
                    prob = ic_probs[i, np.argmax(ic_probs[i])]
                    print(f"  IC {i:2d}: {label:10s} (概率: {prob:.3f})")
                else:
                    print(f"  IC {i:2d}: {label:10s}")
        
        # 创建 topomap
        self.create_topomap(ic_probs, ic_labels, save_path)
        
        print("分析完成!")

def main():
    parser = argparse.ArgumentParser(description='ORICA 源信号可视化工具')
    parser.add_argument('sources_file', help='源信号文件路径')
    parser.add_argument('--mixing', help='混合矩阵文件路径')
    parser.add_argument('--icaweights', help='ICA 权重文件路径')
    parser.add_argument('--icasphere', help='ICA 球化文件路径')
    parser.add_argument('--channels', type=int, default=14, help='通道数 (默认: 14)')
    parser.add_argument('--srate', type=int, default=128, help='采样率 (默认: 128)')
    parser.add_argument('--save', help='保存 topomap 图片路径')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = ORICAVisualizer(n_channels=args.channels, srate=args.srate)
    
    # 运行分析
    visualizer.run_analysis(
        sources_file=args.sources_file,
        mixing_file=args.mixing,
        icaweights_file=args.icaweights,
        icasphere_file=args.icasphere,
        save_path=args.save
    )

if __name__ == "__main__":
    # 示例用法
    if len(os.sys.argv) == 1:
        print("ORICA 源信号可视化工具")
        print("用法示例:")
        print("  python orica_visualizer.py sources.npy --mixing A.npy")
        print("  python orica_visualizer.py sources.mat --icaweights W.mat --icasphere S.mat")
        print("  python orica_visualizer.py sources.npz --save topomap.png")
    else:
        main()
