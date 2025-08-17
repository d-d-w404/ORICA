#!/usr/bin/env python3
"""
Bandpower分析器 - 使用数据接口模式
"""

import threading
import time
import numpy as np
from scipy.signal import welch

class BandpowerAnalyzer:
    """频段功率分析器 - 使用数据接口模式"""
    
    def __init__(self, receiver, update_interval=1.0, gui=None):
        """
        初始化频段功率分析器
        
        Args:
            receiver: LSLStreamReceiver实例
            update_interval: 更新间隔（秒）
            gui: GUI实例，用于更新显示
        """
        self.receiver = receiver
        self.update_interval = update_interval
        self.gui = gui
        self.is_running = False
        self.analysis_thread = None
        
        # 频段定义
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # 历史数据
        self.bandpower_history = {band: [] for band in self.bands.keys()}
        self.max_history = 100
        
    def start(self):
        """启动分析线程"""
        if self.is_running:
            print("⚠️ Bandpower分析器已在运行")
            return
            
        self.is_running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        print("✅ Bandpower分析器已启动")
        
    def stop(self):
        """停止分析线程"""
        self.is_running = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
        print("🛑 Bandpower分析器已停止")
        
    def _analysis_loop(self):
        """分析循环 - 在独立线程中运行"""
        while self.is_running:
            try:
                # 检查数据是否可用
                if not self.receiver.is_data_available():
                    time.sleep(0.1)
                    continue
                
                # 获取处理后数据
                processed_data = self.receiver.get_processed_data()
                if processed_data is None:
                    time.sleep(0.1)
                    continue
                    
                # 获取通道信息
                channel_info = self.receiver.get_channel_info()
                srate = channel_info['sampling_rate']
                
                # 计算频段功率
                bandpower_results = self._calculate_bandpower(processed_data, srate)
                
                # 更新历史数据
                self._update_history(bandpower_results)
                
                # 更新GUI（如果有）
                if self.gui and hasattr(self.gui, 'bandpower_plot'):
                    self.gui.bandpower_plot.update_bandpower(bandpower_results)
                
                # 打印结果（可选）
                self._print_results(bandpower_results)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Bandpower分析错误: {e}")
                time.sleep(0.1)
                
    def _calculate_bandpower(self, data, srate):
        """计算频段功率"""
        try:
            nperseg = min(srate, data.shape[1])
            # 使用Welch方法计算功率谱密度
            freqs, psd = welch(data, fs=srate, nperseg=nperseg, axis=1)
            
            bandpower_results = {}
            
            for band_name, (fmin, fmax) in self.bands.items():
                idx = (freqs >= fmin) & (freqs <= fmax)
                if np.any(idx):
                    # 计算该频段的平均功率
                    band_power = np.mean(psd[:, idx])
                    bandpower_results[band_name] = float(band_power)
                else:
                    bandpower_results[band_name] = 0.0
                    
            return bandpower_results
            
        except Exception as e:
            print(f"❌ 频段功率计算错误: {e}")
            return {band: 0.0 for band in self.bands.keys()}
            
    def _update_history(self, bandpower_results):
        """更新历史数据"""
        for band_name, power in bandpower_results.items():
            self.bandpower_history[band_name].append(power)
            
            # 限制历史数据长度
            if len(self.bandpower_history[band_name]) > self.max_history:
                self.bandpower_history[band_name].pop(0)
                
    def _print_results(self, bandpower_results):
        """打印分析结果"""
        # print("📊 频段功率分析:")
        # for band_name, power in bandpower_results.items():
        #     print(f"  {band_name}: {power:.4f}")
        pass
            
    def get_latest_bandpower(self):
        """获取最新的频段功率数据"""
        return {band: values[-1] if values else 0.0 
                for band, values in self.bandpower_history.items()}
                
    def get_bandpower_history(self, band_name=None):
        """获取频段功率历史数据"""
        if band_name:
            return self.bandpower_history.get(band_name, [])
        else:
            return self.bandpower_history.copy()
            
    def get_average_bandpower(self, window_size=10):
        """获取平均频段功率"""
        results = {}
        for band_name, history in self.bandpower_history.items():
            if len(history) >= window_size:
                avg_power = np.mean(history[-window_size:])
                results[band_name] = float(avg_power)
            else:
                results[band_name] = 0.0
        return results 