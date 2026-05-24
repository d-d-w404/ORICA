#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的LSL数据接收程序
接收名为 mybrain 的LSL流数据并显示基本信息
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylsl import resolve_streams, StreamInlet
import threading
import time

class SimpleLSLReceiver:
    def __init__(self, stream_name='mybrain'):
        """
        简单的LSL数据接收器
        
        Args:
            stream_name: LSL流名称
        """
        self.stream_name = stream_name
        self.inlet = None
        self.is_running = False
        
        # 数据存储
        self.latest_data = None
        self.data_count = 0
        
        # 创建图形界面
        self.setup_plot()
    
    def setup_plot(self):
        """设置图形界面"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle('LSL数据接收状态', fontsize=16, fontweight='bold')
        self.ax.axis('off')
    
    def connect_lsl_stream(self):
        """连接LSL流"""
        print(f"正在寻找LSL流: {self.stream_name}")
        
        streams = resolve_streams()
        target_stream = None
        
        print(f"找到{len(streams)}个LSL流:")
        for stream in streams:
            print(f"  - {stream.name()}")
            if self.stream_name in stream.name():
                target_stream = stream
        
        if target_stream is None:
            raise RuntimeError(f"未找到名为 '{self.stream_name}' 的LSL流")
        
        # 获取流信息
        n_channels = target_stream.channel_count()
        srate = int(target_stream.nominal_srate())
        
        print(f"\n✅ 找到目标流: {target_stream.name()}")
        print(f"   通道数: {n_channels}")
        print(f"   采样率: {srate} Hz")
        
        self.n_channels = n_channels
        self.srate = srate
        
        # 创建inlet
        self.inlet = StreamInlet(target_stream)
        self.is_running = True
        
        # 启动数据接收线程
        self.data_thread = threading.Thread(target=self.data_receiver, daemon=True)
        self.data_thread.start()
        
        print("✅ LSL流连接成功！开始接收数据...")
    
    def data_receiver(self):
        """数据接收线程"""
        while self.is_running:
            try:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.1, max_samples=100)
                if timestamps:
                    # 转换为numpy数组
                    data = np.array(samples).T  # (channels, samples)
                    self.latest_data = data
                    self.data_count += data.shape[1]
                    
                    # 打印接收信息
                    print(f"接收数据: shape={data.shape}, 总计{self.data_count}个样本")
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"数据接收错误: {e}")
                time.sleep(1.0)
    
    def update_plot(self, frame):
        """更新图形显示"""
        self.ax.clear()
        self.ax.axis('off')
        
        if self.latest_data is None:
            # 显示等待消息
            info_text = f"等待数据...\n\n"
            info_text += f"LSL流名称: {self.stream_name}\n"
            info_text += f"状态: 已连接，等待数据"
        else:
            # 显示数据信息
            info_text = f"LSL数据接收状态\n\n"
            info_text += f"流名称: {self.stream_name}\n"
            info_text += f"通道数: {self.n_channels}\n"
            info_text += f"采样率: {self.srate} Hz\n"
            info_text += f"最新数据形状: {self.latest_data.shape}\n"
            info_text += f"总接收样本数: {self.data_count}\n\n"
            
            # 显示最新数据的统计信息
            info_text += f"数据统计:\n"
            info_text += f"  均值: {np.mean(self.latest_data):.4f}\n"
            info_text += f"  标准差: {np.std(self.latest_data):.4f}\n"
            info_text += f"  最小值: {np.min(self.latest_data):.4f}\n"
            info_text += f"  最大值: {np.max(self.latest_data):.4f}\n\n"
            
            info_text += f"前3个通道的前5个样本:\n"
            info_text += f"{self.latest_data[:3, :5]}"
        
        self.ax.text(0.1, 0.5, info_text, 
                    transform=self.ax.transAxes,
                    fontsize=12, verticalalignment='center',
                    family='monospace')
    
    def run(self):
        """运行程序"""
        try:
            # 连接LSL流
            self.connect_lsl_stream()
            
            # 启动动画
            self.ani = animation.FuncAnimation(
                self.fig, self.update_plot,
                interval=1000,  # 每秒更新一次
                blit=False
            )
            
            print("\n图形界面已启动，按Ctrl+C退出...")
            plt.show(block=True)
            
        except KeyboardInterrupt:
            print("\n用户中断，退出...")
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            self.is_running = False
            if self.inlet:
                self.inlet.close_stream()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='简单的LSL数据接收程序')
    parser.add_argument('--stream', default='mybrain', help='LSL流名称（默认: mybrain）')
    
    args = parser.parse_args()
    
    # 创建接收器并运行
    receiver = SimpleLSLReceiver(stream_name=args.stream)
    receiver.run()
