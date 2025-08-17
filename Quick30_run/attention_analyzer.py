#!/usr/bin/env python3
"""
注意力分析器 - 使用数据接口模式
"""

import threading
import time
import numpy as np
from scipy.signal import welch
from scipy.special import expit

class AttentionAnalyzer:
    """注意力分析器 - 使用数据接口模式"""
    
    def __init__(self, receiver, gui=None, update_interval=1.0):
        """
        初始化注意力分析器
        
        Args:
            receiver: LSLStreamReceiver实例
            gui: GUI实例，用于更新显示
            update_interval: 更新间隔（秒）
        """
        self.receiver = receiver
        self.gui = gui
        self.update_interval = update_interval
        self.is_running = False
        self.analysis_thread = None
        
        # 注意力评分历史
        self.attention_history = []
        self.max_history = 10
        
        # 注意力通道配置
        self.attention_channels = ['F7', 'F8', 'T7', 'T8', 'Fpz']
        #self.attention_channels=['AF7', 'Fpz', 'F7', 'Fz', 'T7', 'FC6', 'Fp1', 'F4', 'C4', 'CP6', 'Cz', 'CP5', 'O2', 'O1', 'P3', 'P4', 'P7', 'P8', 'Pz', 'PO7', 'T8', 'C3', 'Fp2', 'F3', 'F8', 'FC5', 'AF8']
        
    def start(self):
        """启动分析线程"""
        if self.is_running:
            print("⚠️ 注意力分析器已在运行")
            return
            
        self.is_running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        print("✅ 注意力分析器已启动")
        
    def stop(self):
        """停止分析线程"""
        self.is_running = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
        print("🛑 注意力分析器已停止")
        
    def _analysis_loop(self):
        """分析循环 - 在独立线程中运行"""
        while self.is_running:
            try:
                # 检查数据是否可用
                if not self.receiver.is_data_available():
                    time.sleep(0.1)
                    continue
                
                # 获取处理后数据
                processed_data = self.receiver.get_buffer_data('processed')
                if processed_data is None:
                    time.sleep(0.1)
                    continue
                

                # 获取通道信息
                channel_info = self.receiver.get_channel_info()
                srate = channel_info['sampling_rate']

                # 计算注意力评分
                attention_score = self._calculate_attention_score(processed_data, channel_info, srate)
                
                # 更新历史数据
                self._update_history(attention_score)

                # 更新GUI（如果有）
                if self.gui:
                    self.gui.update_attention_circle(attention_score)
                

                # 打印结果（可选）
                #print(f"🎯 注意力评分: {attention_score:.3f}")

                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ 注意力分析错误: {e}")
                time.sleep(0.1)
                
    def _calculate_attention_score(self, data, channel_info, srate):
        """计算注意力评分"""
        try:
            # 选择注意力相关通道
            selected_indices = []
            for channel_name in self.attention_channels:
                if channel_name in channel_info['labels']:
                    idx = channel_info['labels'].index(channel_name)
                    selected_indices.append(idx)
            
            if len(selected_indices) < 5:
                #print(f"⚠️ 注意力通道不足，需要5个，当前{len(selected_indices)}个")
                return 0.5
                
            # 提取注意力通道数据
            attention_data = data[selected_indices, :]
            # 数据长度保护，当我使用buffer之后没有这个问题了。如果使用的是processor的数据，长度是不确定的，有可能出问题。但是buffer的大小固定
            min_length = 100  # 或 srate
            if attention_data.shape[1] < min_length:
                print("⚠️ 数据长度不足，跳过本次分析")
                return 0.5
            
            # 重新参考（减去Fpz）
            ref_channel = attention_data[4, :]  # Fpz是第5个通道
            attention_data = attention_data[:4, :] - ref_channel
            
            # 计算Hjorth参数
            d1 = np.diff(attention_data, axis=1)
            d2 = np.diff(d1, axis=1)
            activity = np.mean(np.var(attention_data, axis=1))
            complexity = np.mean(np.sqrt(np.var(d2, axis=1) / np.var(d1, axis=1)))
            
            # 计算频段功率
            nperseg = min(srate, attention_data.shape[1])
            freqs, psd = welch(attention_data, fs=srate, nperseg=nperseg, axis=1)
            # print("psd:", psd.shape)  #psd: (4, 251)
            # print("freq:", freqs.shape)   #freq: (251,)
            
            
            def band_power(fmin, fmax):
                idx = (freqs >= fmin) & (freqs <= fmax)
                if not np.any(idx):
                    return 0.0
                return np.mean(psd[:, idx])
            
            alpha = band_power(8, 13)
            theta = band_power(4, 8)
            beta = band_power(13, 30)
            gamma = band_power(30, 45)
            
            epsilon = 1e-6
            engagement = np.nan_to_num((beta + epsilon) / (alpha + theta + epsilon))

            # 只允许有效engagement进入history
            # if not np.isnan(engagement) and not np.isinf(engagement):
            #     self.attention_history.append(engagement)
            #     if len(self.attention_history) > self.max_history:
            #         self.attention_history.pop(0)
            # else:
            #     print("⚠️ engagement is NaN/Inf, not added to history")
            # # 归一化时只用有效分数
            # valid_history = [x for x in self.attention_history if not np.isnan(x) and not np.isinf(x)]
            # if not valid_history:
            #     print("⚠️ valid_history为空，返回默认分数")
            #     return 0.5
            engagement_avg = engagement


            # E_min = min(valid_history)
            # E_max = max(valid_history)
            # range_ = max(E_max - E_min, epsilon)
            # normalized = (engagement_avg - E_min) / range_
            #print(f"engagement_avg={engagement_avg:.4f} ")


            E_min_fixed = 0.03  # 最小engagement值（根据你的实际数据调整）
            E_max_fixed = 0.15  # 最大engagement值（根据你的实际数据调整）
            normalized = (engagement_avg - E_min_fixed) / (E_max_fixed - E_min_fixed)
            normalized = np.clip(normalized, 0.0, 1.0)

            #print(f"engagement_avg={engagement_avg:.4f}, E_min_fixed={E_min_fixed}, E_max_fixed={E_max_fixed}")


            #print("normalized", normalized)
            if np.isnan(normalized) or np.isinf(normalized):
                normalized = 0.5
            return normalized
        except Exception as e:
            print(f"❌ 注意力评分计算错误: {e}")
            return 0.5
            
    def _update_history(self, attention_score):
        """更新注意力评分历史"""
        self.attention_history.append(attention_score)
        if len(self.attention_history) > self.max_history:
            self.attention_history.pop(0)
            
    def get_latest_attention_score(self):
        """获取最新的注意力评分"""
        return self.attention_history[-1] if self.attention_history else 0.5
        
    def get_attention_history(self):
        """获取注意力评分历史"""
        return self.attention_history.copy()
        
    def get_average_attention_score(self, window_size=10):
        """获取平均注意力评分"""
        if len(self.attention_history) >= window_size:
            return np.mean(self.attention_history[-window_size:])
        else:
            return np.mean(self.attention_history) if self.attention_history else 0.5 