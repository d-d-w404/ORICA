#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时 ORICA 源信号可视化工具 - 专为 Emotiv EPOC 优化
从 LSL 流接收 ORICA 源信号，实时进行 ICLabel 分类并显示 topomap

支持设备：
- Emotiv EPOC (14通道, 128Hz) - 主要支持
- 其他 EEG 设备 (自动适配)

使用方法：
python realtime_orica_visualizer.py --stream your_lsl_stream --test
python realtime_orica_visualizer.py --stream your_lsl_stream --mixing mixing_matrix.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pylsl import resolve_streams, StreamInlet
import threading
import time
import queue
from collections import deque
import argparse

class RealtimeORICAVisualizer:
    def __init__(self, n_channels=14, srate=128, stream_name='mybrain', 
                 update_interval=2.0, buffer_duration=10.0, test_mode=False):
        """
        初始化实时 ORICA 可视化器
        
        Args:
            n_channels: 通道数
            srate: 采样率
            stream_name: LSL 流名称
            update_interval: 更新间隔（秒）
            buffer_duration: 缓冲区持续时间（秒）
        """
        self.n_channels = n_channels
        self.srate = srate
        self.stream_name = stream_name
        self.update_interval = update_interval
        self.buffer_duration = buffer_duration
        self.buffer_size = int(srate * buffer_duration)
        self.test_mode = test_mode
        
        # 通道标签 - 为 Emotiv EPOC 优化
        if n_channels == 14:
            # Emotiv EPOC 的标准通道名称
            self.chan_labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                               'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        else:
            # 其他设备使用通用标签
            self.chan_labels = [f'Ch{i+1:02d}' for i in range(n_channels)]
        
        # 数据缓冲区
        self.sources_buffer = deque(maxlen=self.buffer_size)
        self.mixing_matrix = None
        self.ic_probs = None
        self.ic_labels = None
        
        # LSL 相关
        self.inlet = None
        self.is_running = False
        self.data_queue = queue.Queue()
        
        # 设置电极位置
        self.setup_montage()
        
        # 设置图形界面
        self.setup_plot()
        
        
    def setup_montage(self):
        """设置电极位置 - 专门为 Emotiv EPOC 优化"""
        try:
            # 首先尝试使用标准 10-20 系统
            self.montage = mne.channels.make_standard_montage("emotiv")
            if len(self.montage.ch_names) != self.n_channels:
                self.create_emotiv_epoc_montage()
        except:
            print("fish")
            self.create_emotiv_epoc_montage()
    
    def create_emotiv_epoc_montage(self):
        """为 Emotiv EPOC 设备创建专门的电极位置"""
        print(f"为 Emotiv EPOC 设备创建 {self.n_channels} 通道的电极位置...")
        
        # Emotiv EPOC 的标准电极位置（基于实际设备布局）
        emotiv_positions = {
            'AF3': [0.0, 0.5, 0.0],      # 前额
            'F7': [-0.3, 0.3, 0.0],      # 左前额
            'F3': [-0.2, 0.4, 0.0],      # 左前额
            'FC5': [-0.4, 0.2, 0.0],     # 左前中央
            'T7': [-0.5, 0.0, 0.0],      # 左颞
            'P7': [-0.4, -0.2, 0.0],     # 左后颞
            'O1': [-0.2, -0.4, 0.0],     # 左枕
            'O2': [0.2, -0.4, 0.0],      # 右枕
            'P8': [0.4, -0.2, 0.0],      # 右后颞
            'T8': [0.5, 0.0, 0.0],       # 右颞
            'FC6': [0.4, 0.2, 0.0],      # 右前中央
            'F4': [0.2, 0.4, 0.0],       # 右前额
            'F8': [0.3, 0.3, 0.0],       # 右前额
            'AF4': [0.0, 0.5, 0.0]       # 前额
        }
        
        if self.n_channels == 14:
            # 使用 Emotiv EPOC 的标准位置
            if all(ch in emotiv_positions for ch in self.chan_labels):
                ch_pos = {ch: emotiv_positions[ch] for ch in self.chan_labels}
                print("✅ 使用 Emotiv EPOC 标准电极位置")
            else:
                # 如果通道名称不匹配，使用圆形排列
                ch_pos = self._create_circular_positions()
                print("✅ 使用 Emotiv EPOC 圆形电极排列")
        else:
            # 其他通道数使用圆形排列
            ch_pos = self._create_circular_positions()
            print(f"✅ 为 {self.n_channels} 通道创建圆形电极排列")
        
        self.montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            coord_frame='head'
        )
    
    def _create_circular_positions(self):
        """创建圆形排列的电极位置"""
        angles = np.linspace(0, 2*np.pi, self.n_channels, endpoint=False)
        positions = np.column_stack([
            np.cos(angles) * 0.4,  # x坐标
            np.sin(angles) * 0.4,  # y坐标
            np.zeros(self.n_channels)  # z坐标
        ])
        return dict(zip(self.chan_labels, positions))
    
    def create_custom_montage(self):
        """创建自定义电极位置（圆形排列） - 保持向后兼容"""
        print(f"创建 {self.n_channels} 通道的自定义电极位置...")
        ch_pos = self._create_circular_positions()
        
        self.montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            coord_frame='head'
        )
    
    def setup_plot(self):
        """设置图形界面"""
        self.fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 4, figure=self.fig)
        
        # 状态信息
        self.status_ax = self.fig.add_subplot(gs[0, :])
        self.status_ax.set_title("实时 ORICA 源信号分析", fontsize=16, fontweight='bold')
        self.status_ax.axis('off')
        
        # Topomap 区域 - 确保显示8个
        self.topomap_axes = []
        for i in range(8):  # 显示8个成分
            row = 1 + i // 4
            col = i % 4
            ax = self.fig.add_subplot(gs[row, col])
            self.topomap_axes.append(ax)
            # 确保每个轴都可见
            ax.set_visible(True)
        
        # 时间序列显示
        self.timeseries_ax = self.fig.add_subplot(gs[2, :])
        self.timeseries_ax.set_title("源信号时间序列", fontsize=12)
        self.timeseries_ax.set_xlabel("时间 (s)")
        self.timeseries_ax.set_ylabel("幅度")
        
        plt.tight_layout()
        plt.ion()  # 交互模式
    
    def connect_lsl_stream(self):
        """连接 LSL 流"""
        print(f"正在寻找 LSL 流: {self.stream_name}")
        
        streams = resolve_streams()
        target_stream = None
        
        for stream in streams:
            if self.stream_name in stream.name():
                target_stream = stream
                break
        
        if target_stream is None:
            raise RuntimeError(f"未找到名为 '{self.stream_name}' 的 LSL 流")
        
        # 自动检测通道数和采样率
        detected_channels = target_stream.channel_count()
        detected_srate = int(target_stream.nominal_srate())
        
        print(f"找到流: {target_stream.name()}")
        print(f"检测到通道数: {detected_channels}, 采样率: {detected_srate}")
        
        # 如果检测到的通道数与初始化时不同，更新配置
        if detected_channels != self.n_channels:
            print(f"⚠️ 通道数不匹配: 初始化={self.n_channels}, 检测到={detected_channels}")
            self.n_channels = detected_channels
            # 重新生成通道标签
            if self.n_channels == 14:
                self.chan_labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                                   'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
            else:
                self.chan_labels = [f'Ch{i+1:02d}' for i in range(self.n_channels)]
            # 重新设置 montage
            self.setup_montage()
            print("✅ 已更新通道配置")
        
        if detected_srate != self.srate:
            print(f"⚠️ 采样率不匹配: 初始化={self.srate}, 检测到={detected_srate}")
            self.srate = detected_srate
            self.buffer_size = int(self.srate * self.buffer_duration)
            print("✅ 已更新采样率配置")
        
        self.inlet = StreamInlet(target_stream)
        self.is_running = True
        
        # 启动数据接收线程
        self.data_thread = threading.Thread(target=self.data_receiver, daemon=True)
        self.data_thread.start()
        
        print("LSL 流连接成功!")
    
    def reconnect_stream(self):
        """重连 LSL 流"""
        print("尝试重连 LSL 流...")
        try:
            if self.inlet:
                self.inlet.close_stream()
            
            streams = resolve_streams()
            target_stream = None
            for stream in streams:
                if self.stream_name in stream.name():
                    target_stream = stream
                    break
            
            if target_stream:
                self.inlet = StreamInlet(target_stream)
                print("LSL 流重连成功!")
            else:
                print("未找到目标 LSL 流")
        except Exception as e:
            print(f"重连失败: {e}")
    
    def data_receiver(self):
        """数据接收线程"""
        while self.is_running:
            try:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.1, max_samples=100)
                if timestamps:
                    # 将数据放入队列
                    self.data_queue.put((samples, timestamps))
                else:
                    # 没有数据时短暂等待
                    time.sleep(0.01)
            except Exception as e:
                print(f"数据接收错误: {e}")
                # 尝试重连
                try:
                    self.reconnect_stream()
                except:
                    pass
                time.sleep(1.0)
    
    def load_mixing_matrix(self, mixing_file=None, icaweights_file=None, icasphere_file=None):
        """加载混合矩阵"""
        if mixing_file:
            print(f"加载混合矩阵: {mixing_file}")
            if mixing_file.endswith('.npy'):
                self.mixing_matrix = np.load(mixing_file)
            elif mixing_file.endswith('.mat'):
                import scipy.io
                data = scipy.io.loadmat(mixing_file)
                self.mixing_matrix = data['A'] if 'A' in data else data['mixing_matrix']
            elif mixing_file.endswith('.npz'):
                data = np.load(mixing_file)
                self.mixing_matrix = data['A'] if 'A' in data else data['mixing_matrix']
        
        elif icaweights_file and icasphere_file:
            print(f"从 ICA 矩阵计算混合矩阵: {icaweights_file}, {icasphere_file}")
            if icaweights_file.endswith('.npy'):
                icaweights = np.load(icaweights_file)
            elif icaweights_file.endswith('.mat'):
                import scipy.io
                data = scipy.io.loadmat(icaweights_file)
                icaweights = data['icaweights']
            elif icaweights_file.endswith('.npz'):
                data = np.load(icaweights_file)
                icaweights = data['icaweights']
            
            if icasphere_file.endswith('.npy'):
                icasphere = np.load(icasphere_file)
            elif icasphere_file.endswith('.mat'):
                import scipy.io
                data = scipy.io.loadmat(icasphere_file)
                icasphere = data['icasphere']
            elif icasphere_file.endswith('.npz'):
                data = np.load(icasphere_file)
                icasphere = data['icasphere']
            
            self.mixing_matrix = np.linalg.pinv(icaweights)
        
        else:
            print("⚠️ 未提供混合矩阵，将使用随机矩阵")
            self.create_demo_mixing_matrix()
        
        print(f"混合矩阵形状: {self.mixing_matrix.shape}")
    
    def create_demo_mixing_matrix(self):
        """创建演示用混合矩阵"""
        n_comp = self.n_channels
        
        # 创建更有意义的空间模式
        A = np.zeros((self.n_channels, n_comp))
        
        # 为每个成分创建不同的空间模式
        for i in range(n_comp):
            # 创建不同的空间分布模式
            if i < n_comp // 3:
                # 前1/3：前额模式
                A[:n_comp//3, i] = np.random.randn(n_comp//3) * 0.8
            elif i < 2 * n_comp // 3:
                # 中1/3：中央模式
                A[n_comp//3:2*n_comp//3, i] = np.random.randn(2*n_comp//3 - n_comp//3) * 0.8
            else:
                # 后1/3：后部模式
                A[2*n_comp//3:, i] = np.random.randn(n_comp - 2*n_comp//3) * 0.8
            
            # 添加一些随机性
            A[:, i] += np.random.randn(self.n_channels) * 0.2
            
            # 归一化
            A[:, i] = A[:, i] / np.linalg.norm(A[:, i])
        
        self.mixing_matrix = A
        print(f"创建演示混合矩阵: {self.mixing_matrix.shape}")
    
    def classify_sources_directly(self, data, sources, mixing_matrix, chan_names, srate, n_comp=None):
        """直接对源信号进行 ICLabel 分类"""
        try:
            if mixing_matrix is None:
                return None, None
            
            # 创建 MNE Raw 对象
            info = mne.create_info(chan_names, srate, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            
            # 为 Emotiv EPOC 设备设置专门的 montage
            try:
                raw.set_montage(self.montage)
            except ValueError as e:
                print(f"⚠️ Montage 设置失败: {e}")
                # 如果失败，尝试使用标准 10-20 系统
                try:
                    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
                except:
                    print("❌ 无法设置任何 montage，使用默认位置")
            
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
            
            return ic_probs, ic_labels
            
        except Exception as e:
            print(f"ICLabel 分类失败: {e}")
            return None, None
    
    def update_plot(self, frame):
        """更新图形显示"""
        # 处理队列中的数据
        data_processed = False
        while not self.data_queue.empty():
            try:
                samples, timestamps = self.data_queue.get_nowait()
                if samples and len(samples) > 0:
                    # 转换数据格式
                    data = np.array(samples).T  # (channels, samples)
                    
                    # 检查数据形状
                    if data.shape[0] == self.n_channels:
                        sources = data.T  # (samples, components)
                        self.sources_buffer.extend(sources)
                        data_processed = True
                    else:
                        print(f"数据形状不匹配: 期望 {self.n_channels}, 得到 {data.shape[0]}")
                        continue
            except queue.Empty:
                break
            except Exception as e:
                print(f"数据处理错误: {e}")
                break
        
        # 如果没有处理到数据，显示等待状态
        if not data_processed and len(self.sources_buffer) < self.srate:
            self.status_ax.clear()
            self.status_ax.set_title("实时 ORICA 源信号分析", fontsize=16, fontweight='bold')
            self.status_ax.text(0.5, 0.5, f"等待数据... ({len(self.sources_buffer)}/{self.srate})", 
                              ha='center', va='center', fontsize=14, transform=self.status_ax.transAxes)
            self.status_ax.axis('off')
            return
        
        # 检查是否有足够的数据
        if len(self.sources_buffer) < self.srate:  # 至少1秒数据
            return
        
        # 获取最近的数据
        recent_sources = np.array(list(self.sources_buffer)[-self.srate:])  # 最近1秒
        recent_sources = recent_sources.T  # (components, samples)
        
        # 调试信息：显示源信号的变化
        if len(recent_sources) > 0:
            source_std = np.std(recent_sources, axis=1)
            source_mean = np.mean(np.abs(recent_sources), axis=1)
            print(f"源信号变化 - 均值: {source_mean[:3]}, 标准差: {source_std[:3]}")
        
        # 运行 ICLabel 分类
        if self.mixing_matrix is not None:
            data = self.mixing_matrix @ recent_sources  # 重构原始通道数据
            try:
                ic_probs, ic_labels = self.classify_sources_directly(
                    data=data,
                    sources=recent_sources,
                    mixing_matrix=self.mixing_matrix,
                    chan_names=self.chan_labels,
                    srate=self.srate,
                    n_comp=recent_sources.shape[0]
                )
                
                if ic_probs is not None and ic_labels is not None:
                    self.ic_probs = ic_probs
                    self.ic_labels = ic_labels
                    print(f"ICLabel 分类成功: {len(ic_labels)} 个成分")
                else:
                    print("ICLabel 分类失败，使用默认标签")
            except Exception as e:
                print(f"ICLabel 分类异常: {e}")
                # 即使 ICLabel 失败，也继续显示 topomap
        
        # 更新状态信息
        self.status_ax.clear()
        self.status_ax.set_title("实时 ORICA 源信号分析", fontsize=16, fontweight='bold')
        
        import time
        current_time = time.strftime("%H:%M:%S")
        status_text = f"时间: {current_time}\n"
        status_text += f"缓冲区: {len(self.sources_buffer)}/{self.buffer_size} 样本\n"
        status_text += f"采样率: {self.srate} Hz\n"
        status_text += f"混合矩阵: {'已加载' if self.mixing_matrix is not None else '未加载'}\n"
        
        if self.ic_labels is not None:
            status_text += f"ICLabel 状态: 已分类 {len(self.ic_labels)} 个成分\n"
            # 统计各类别数量
            unique_labels, counts = np.unique(self.ic_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                status_text += f"  {label}: {count} 个\n"
        else:
            status_text += "ICLabel 状态: 未分类\n"
        
        self.status_ax.text(0.05, 0.5, status_text, fontsize=12, verticalalignment='center',
                           transform=self.status_ax.transAxes)
        self.status_ax.axis('off')
        
        # 更新 topomap - 传入最新的源信号数据
        self.update_topomaps(recent_sources)
        
        # 更新时间序列
        self.update_timeseries(recent_sources)
        
        # 强制刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update_topomaps(self, recent_sources=None):
        """更新 topomap 显示"""
        # print(f"调试: mixing_matrix={self.mixing_matrix is not None}, ic_labels={self.ic_labels is not None}")
        
        if self.mixing_matrix is None:
            print("警告: 混合矩阵为空，无法绘制 topomap")
            # 显示提示信息
            for i, ax in enumerate(self.topomap_axes):
                ax.clear()
                ax.text(0.5, 0.5, f'IC {i}\n需要混合矩阵', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(f'IC {i}', fontsize=10)
            return
        
        # 如果没有传入源信号数据，使用缓冲区中的数据
        if recent_sources is None:
            if len(self.sources_buffer) < self.srate:
                print("警告: 缓冲区数据不足，无法绘制 topomap")
                return
            recent_sources = np.array(list(self.sources_buffer)[-self.srate:]).T
        
        if self.ic_labels is None:
            print("警告: ICLabel 结果为空，使用默认标签")
            # 使用默认标签继续绘制
            ic_labels = [f'Unknown' for _ in range(min(8, self.mixing_matrix.shape[1]))]
        else:
            ic_labels = self.ic_labels
        
        n_comp = min(8, self.mixing_matrix.shape[1])  # 最多显示8个成分
        print(f"准备绘制 {n_comp} 个 topomap，混合矩阵形状: {self.mixing_matrix.shape}")
        
        for i in range(8):  # 总是显示8个
            ax = self.topomap_axes[i]
            ax.clear()
            ax.set_visible(True)
            
            if i < n_comp:
                # 计算该成分的 topomap - 使用混合矩阵和源信号的组合
                # 混合矩阵的列表示该成分在原始通道上的投影权重
                component_weights = self.mixing_matrix[:, i]
                
                # 计算该成分的当前激活强度（源信号的平均幅度）
                if i < recent_sources.shape[0]:
                    component_activation = np.mean(np.abs(recent_sources[i, :]))
                    # 将激活强度应用到权重上，使 topomap 反映当前活动
                    component_data = component_weights * component_activation
                else:
                    component_data = component_weights
                
                # 创建 MNE info 对象
                info = mne.create_info(self.chan_labels, self.srate, ch_types='eeg')
                info.set_montage(self.montage)
                
                # 绘制 topomap
                try:
                    print(f"绘制 IC {i}: 权重形状 {component_weights.shape}, 激活强度 {component_activation:.3f}, 数据范围 [{component_data.min():.3f}, {component_data.max():.3f}]")
                    
                    # 动态调整颜色范围
                    data_max = np.max(np.abs(component_data))
                    vlim = (-data_max, data_max) if data_max > 0 else (-1, 1)
                    
                    # 使用简化的 topomap 绘制
                    im, _ = mne.viz.plot_topomap(
                        component_data, 
                        info, 
                        axes=ax, 
                        show=False,
                        cmap='RdBu_r',
                        contours=6,
                        vlim=vlim  # 动态范围
                    )
                    
                    # 设置标题 - 包含激活强度信息
                    if i < len(ic_labels):
                        label = ic_labels[i]
                        if self.ic_probs is not None and i < self.ic_probs.shape[0]:
                            prob = self.ic_probs[i, np.argmax(self.ic_probs[i])]
                            title = f'IC {i}: {label}\n激活: {component_activation:.3f}\n置信: {prob:.2f}'
                        else:
                            title = f'IC {i}: {label}\n激活: {component_activation:.3f}'
                    else:
                        title = f'IC {i}\n激活: {component_activation:.3f}'
                    
                    ax.set_title(title, fontsize=9)
                    print(f"IC {i} topomap 绘制成功")
                    
                except Exception as e:
                    print(f"IC {i} topomap 绘制失败: {e}")
                    # 尝试简单的热力图绘制
                    try:
                        self.draw_simple_topomap(ax, component_data, i, ic_labels)
                    except Exception as e2:
                        print(f"简单 topomap 也失败: {e2}")
                        ax.text(0.5, 0.5, f'IC {i}\nError\n{e}', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=8)
                        ax.set_title(f'IC {i}', fontsize=10)
            else:
                # 如果成分不足8个，显示空白
                ax.text(0.5, 0.5, f'IC {i}\nNo Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, alpha=0.5)
                ax.set_title(f'IC {i}', fontsize=10)
                ax.set_visible(True)
    
    def draw_simple_topomap(self, ax, component_data, i, ic_labels):
        """绘制简单的 topomap（备用方法）"""
        # 创建简单的圆形热力图
        n_points = 50
        x = np.linspace(-1, 1, n_points)
        y = np.linspace(-1, 1, n_points)
        X, Y = np.meshgrid(x, y)
        
        # 计算距离中心的距离
        R = np.sqrt(X**2 + Y**2)
        
        # 将电极位置映射到圆形上
        angles = np.linspace(0, 2*np.pi, len(component_data), endpoint=False)
        electrode_x = np.cos(angles)
        electrode_y = np.sin(angles)
        
        # 插值到网格
        from scipy.interpolate import griddata
        points = np.column_stack([electrode_x, electrode_y])
        Z = griddata(points, component_data, (X, Y), method='cubic', fill_value=0)
        
        # 在圆形外设为 NaN
        Z[R > 1] = np.nan
        
        # 绘制热力图
        im = ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # 绘制电极位置
        ax.scatter(electrode_x, electrode_y, c=component_data, s=100, 
                  cmap='RdBu_r', vmin=-1, vmax=1, edgecolors='black', linewidth=1)
        
        # 设置标题
        if i < len(ic_labels):
            label = ic_labels[i]
            title = f'IC {i}: {label}'
        else:
            title = f'IC {i}'
        
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        print(f"IC {i} 简单 topomap 绘制成功")
    
    def update_timeseries(self, sources):
        """更新时间序列显示"""
        self.timeseries_ax.clear()
        self.timeseries_ax.set_title("源信号时间序列", fontsize=12)
        self.timeseries_ax.set_xlabel("时间 (s)")
        self.timeseries_ax.set_ylabel("幅度")
        
        # 显示前4个成分的时间序列
        n_show = min(4, sources.shape[0])
        time_axis = np.arange(sources.shape[1]) / self.srate
        
        for i in range(n_show):
            label = f'IC {i}'
            if self.ic_labels is not None and i < len(self.ic_labels):
                label += f' ({self.ic_labels[i]})'
            
            self.timeseries_ax.plot(time_axis, sources[i], label=label, alpha=0.7)
        
        self.timeseries_ax.legend()
        self.timeseries_ax.grid(True, alpha=0.3)
    
    def run(self):
        """运行实时可视化"""
        self.ani = None
        try:
            if self.test_mode:
                print("运行在测试模式...")
                self.run_test_mode()
            else:
                # 连接 LSL 流
                self.connect_lsl_stream()
            
            # 启动动画
            self.ani = animation.FuncAnimation(
                self.fig, self.update_plot, 
                interval=int(self.update_interval * 1000), 
                blit=False,
                cache_frame_data=False,
                save_count=1000
            )
            
            # 保持动画对象引用
            plt.show(block=True)
            
        except KeyboardInterrupt:
            print("\n用户中断，正在退出...")
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            self.is_running = False
            if self.inlet:
                self.inlet.close_stream()
            if self.ani:
                self.ani.event_source.stop()
    
    def run_test_mode(self):
        """运行测试模式（生成模拟数据）"""
        self.is_running = True
        
        # 如果没有混合矩阵，创建一个测试用的
        if self.mixing_matrix is None:
            print("测试模式：创建测试用混合矩阵")
            self.create_demo_mixing_matrix()
        
        # 生成模拟数据
        def generate_test_data():
            t = 0
            while self.is_running:
                # 生成变化的源信号（加入时间变化）
                test_sources = np.zeros((100, self.n_channels))
                for i in range(self.n_channels):
                    # 每个成分有不同的频率和幅度
                    freq = 0.5 + i * 0.1  # 不同频率
                    amplitude = 1.0 + 0.5 * np.sin(t * 0.01)  # 幅度变化
                    phase = i * np.pi / 4  # 不同相位
                    
                    time_vec = np.arange(100) / self.srate
                    test_sources[:, i] = amplitude * np.sin(2 * np.pi * freq * time_vec + phase)
                
                # 添加噪声
                test_sources += 0.1 * np.random.randn(100, self.n_channels)
                
                self.data_queue.put((test_sources.T, [time.time()] * 100))  # (channels, samples)
                t += 1
                time.sleep(0.1)
        
        # 启动测试数据生成线程
        self.test_thread = threading.Thread(target=generate_test_data, daemon=True)
        self.test_thread.start()
        print("测试模式：生成模拟 ORICA 源信号数据")

def main():
    parser = argparse.ArgumentParser(description='实时 ORICA 源信号可视化工具 - 专为 Emotiv EPOC 优化')
    parser.add_argument('--stream', default='mybrain', help='LSL 流名称 (默认: mybrain)')
    parser.add_argument('--channels', type=int, default=14, help='通道数 (默认: 14, Emotiv EPOC 标准)')
    parser.add_argument('--srate', type=int, default=128, help='采样率 (默认: 128 Hz, Emotiv EPOC 标准)')
    parser.add_argument('--interval', type=float, default=2.0, help='更新间隔秒数 (默认: 2.0)')
    parser.add_argument('--mixing', help='混合矩阵文件路径')
    parser.add_argument('--icaweights', help='ICA 权重文件路径')
    parser.add_argument('--icasphere', help='ICA 球化文件路径')
    parser.add_argument('--test', action='store_true', help='运行测试模式（使用模拟数据）')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = RealtimeORICAVisualizer(
        n_channels=args.channels,
        srate=args.srate,
        stream_name=args.stream,
        update_interval=args.interval,
        test_mode=args.test
    )
    
    # 加载混合矩阵
    visualizer.load_mixing_matrix(
        mixing_file=args.mixing,
        icaweights_file=args.icaweights,
        icasphere_file=args.icasphere
    )
    
    # 运行实时可视化
    visualizer.run()

if __name__ == "__main__":
    main()
