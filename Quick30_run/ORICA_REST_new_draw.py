#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORICA_REST_new 数据可视化程序
读取ORICA算法输出的源信号数据，绘制交互式图像
支持左右拖动、缩放等操作
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import scipy.io

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ORICADataVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("ORICA_REST_new 多文件数据可视化器")
        self.root.geometry("1400x900")
        
        # 数据存储
        self.sources1 = None  # 第一个文件的数据
        self.sources2 = None  # 第二个文件的数据
        self.data_loaded1 = False
        self.data_loaded2 = False
        self.channel_names1 = []
        self.channel_names2 = []
        self.sample_rate = 1000
        
        # 绘图相关
        self.fig = None
        self.ax = None
        self.canvas = None
        self.line_objects = []  # 存储线条对象
        
        # 显示设置
        self.window_size_var = tk.IntVar(value=100)
        self.linewidth_var = tk.IntVar(value=1)
        self.color_theme_var = tk.StringVar(value="default")
        self.show_file1_var = tk.BooleanVar(value=True)
        self.show_file2_var = tk.BooleanVar(value=True)
        
        # 源信号开关控制
        self.source_switches1 = {}  # 文件1的源信号开关
        self.source_switches2 = {}  # 文件2的源信号开关
        
        # 当前显示的样本范围
        self.current_start_sample = 0


        self.gain_var = tk.DoubleVar(value=1.0)  # 垂直增益，默认 1.0

        
        # 创建界面
        self.create_widgets()
        self.create_plot()
        
    def create_widgets(self):
        """创建界面控件"""
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 文件1控制
        file1_frame = tk.LabelFrame(left_frame, text="文件1", font=('Arial', 10, 'bold'))
        file1_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(file1_frame, text="chose file1", command=lambda: self.load_data(1),
                 bg='lightgreen', font=('Arial', 9)).pack(fill=tk.X, padx=5, pady=2)

        
        self.file1_info_label = tk.Label(file1_frame, text="unload file", font=('Arial', 8))
        self.file1_info_label.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Checkbutton(file1_frame, text="chose file1", variable=self.show_file1_var,
                      command=self.redraw_plot, font=('Arial', 8)).pack(anchor=tk.W, padx=5)
        
        # 文件2控制
        file2_frame = tk.LabelFrame(left_frame, text="file2", font=('Arial', 10, 'bold'))
        file2_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(file2_frame, text="chose file2", command=lambda: self.load_data(2),
                 bg='lightblue', font=('Arial', 9)).pack(fill=tk.X, padx=5, pady=2)
        

        
        self.file2_info_label = tk.Label(file2_frame, text="unload file", font=('Arial', 8))
        self.file2_info_label.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Checkbutton(file2_frame, text="show file2", variable=self.show_file2_var,
                      command=self.redraw_plot, font=('Arial', 8)).pack(anchor=tk.W, padx=5)
        
        # 绘图控制
        plot_frame = tk.LabelFrame(left_frame, text="plot control", font=('Arial', 10, 'bold'))
        plot_frame.pack(fill=tk.X, pady=5)
        
        # 样本窗口大小
        tk.Label(plot_frame, text="sample window size:", font=('Arial', 8)).pack(anchor=tk.W, padx=5)
        window_size_scale = tk.Scale(plot_frame, from_=50, to=500, orient=tk.HORIZONTAL, 
                                   variable=self.window_size_var, command=self.update_window_size)
        window_size_scale.pack(fill=tk.X, padx=5)
        
        # 线宽控制
        tk.Label(plot_frame, text="line width:", font=('Arial', 8)).pack(anchor=tk.W, padx=5)
        linewidth_scale = tk.Scale(plot_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
                                 variable=self.linewidth_var, command=self.update_linewidth)
        linewidth_scale.pack(fill=tk.X, padx=5)
        
        # 颜色主题
        tk.Label(plot_frame, text="color theme:", font=('Arial', 8)).pack(anchor=tk.W, padx=5)
        color_combo = ttk.Combobox(plot_frame, textvariable=self.color_theme_var, 
                                  values=["default", "rainbow", "grey", "compare"], state="readonly")
        color_combo.pack(fill=tk.X, padx=5, pady=2)
        color_combo.bind('<<ComboboxSelected>>', self.update_colors)

        tk.Label(plot_frame, text="ylim range:", font=('Arial', 8)).pack(anchor=tk.W, padx=5)
        self.gain_scale = tk.Scale(
            plot_frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self.gain_var, command=lambda v: self.redraw_plot()
        )
        self.gain_scale.pack(fill=tk.X, padx=5)
        
        # 源信号开关控制
        source_control_frame = tk.LabelFrame(left_frame, text="Source signal switch control", font=('Arial', 10, 'bold'))
        source_control_frame.pack(fill=tk.X, pady=5)
        
        # 文件1源信号开关
        self.file1_source_frame = tk.Frame(source_control_frame)
        self.file1_source_frame.pack(fill=tk.X, pady=2)
        tk.Label(self.file1_source_frame, text="file1:", font=('Arial', 8, 'bold'), fg='darkgreen').pack(side=tk.LEFT)
        
        # 文件2源信号开关
        self.file2_source_frame = tk.Frame(source_control_frame)
        self.file2_source_frame.pack(fill=tk.X, pady=2)
        tk.Label(self.file2_source_frame, text="file2:", font=('Arial', 8, 'bold'), fg='darkblue').pack(side=tk.LEFT)
        
        # 统计信息
        stats_frame = tk.LabelFrame(left_frame, text="information", font=('Arial', 10, 'bold'))
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=35, font=('Arial', 8))
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        # 右侧绘图区域
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 绘图控制按钮
        control_frame = tk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(control_frame, text="保存图像", command=self.save_plot, 
                 bg='orange', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        # 滑动条框架
        self.scroll_frame = tk.Frame(right_frame)
        self.scroll_frame.pack(fill=tk.X, pady=5)
        
        # 绘图区域
        self.plot_frame = tk.Frame(right_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
    def create_plot(self):
        """创建绘图区域"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 设置图像标题和标签
        self.ax.set_title("ORICA  source signal visualization", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("smaple points", fontsize=12)
        self.ax.set_ylabel("source signals", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # 启用交互功能
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
    def load_data(self, file_index):
        """加载数据文件"""
        # 默认指向temp_txt目录
        initial_dir = r"D:\work\Python_Project\ORICA\temp_txt"
        if not os.path.exists(initial_dir):
            initial_dir = "."
            
        file_path = filedialog.askopenfilename(
            title=f"选择ORICA输出文件 (文件{file_index})",
            initialdir=initial_dir,
            filetypes=[
                ("文本文件", "*.txt"),
                ("NumPy文件", "*.npy"),
                ("MATLAB文件", "*.mat"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            if file_path.endswith('.npy'):
                self.load_npy_file(file_path, file_index)
            elif file_path.endswith('.txt'):
                self.load_txt_file(file_path, file_index)
            elif file_path.endswith('.mat'):
                self.load_mat_file(file_path, file_index)
            else:
                messagebox.showerror("错误", "不支持的文件格式")
                return
                
            # 更新数据加载状态
            if file_index == 1:
                self.data_loaded1 = True
            elif file_index == 2:
                self.data_loaded2 = True
                
            self.update_file_info(file_path, file_index)
            self.update_channel_list()
            self.update_statistics()
            
            # 创建源信号开关
            self.create_source_switches(file_index)
            
            # 如果还没有滑动条，创建滑动条
            if not hasattr(self, 'scrollbar'):
                self.add_scrollbar()
                
            self.redraw_plot()
            
            print(f"✅ 文件{file_index}加载成功")
            
        except Exception as e:
            print(f"❌ 文件{file_index}加载失败: {e}")
            messagebox.showerror("加载错误", f"无法加载文件: {str(e)}")
    

    def load_npy_file(self, file_path, file_index):
        """加载.npy文件"""
        if file_index == 1:
            self.sources1 = np.load(file_path)
            if self.sources1.ndim == 1:
                self.sources1 = self.sources1.reshape(1, -1)
            elif self.sources1.ndim > 2:
                self.sources1 = self.sources1.reshape(self.sources1.shape[0], -1)
            
            # 创建默认通道名称
            self.channel_names1 = [f"source{i+1:02d}" for i in range(self.sources1.shape[0])]
            self.sample_rate = 1000  # 默认采样率
        elif file_index == 2:
            self.sources2 = np.load(file_path)
            if self.sources2.ndim == 1:
                self.sources2 = self.sources2.reshape(1, -1)
            elif self.sources2.ndim > 2:
                self.sources2 = self.sources2.reshape(self.sources2.shape[0], -1)
            
            # 创建默认通道名称
            self.channel_names2 = [f"s{i+1:02d}" for i in range(self.sources2.shape[0])]
            self.sample_rate = 1000  # 默认采样率
            
    def load_txt_file(self, file_path, file_index):
        """加载.txt文件 - 简化版本，支持44.txt格式"""
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                raise ValueError("文件为空")
            
            # 解析第一行的元信息
            header_line = lines[0].strip()
            if header_line.startswith('# rows='):
                # 解析元信息：例如 "# rows=16 cols=23041 class=double"
                parts = header_line.split()
                rows = int(parts[1].split('=')[1])
                cols = int(parts[2].split('=')[1])
                print(f"📊 文件信息: {rows} 行 x {cols} 列")
            else:
                raise ValueError("文件格式不支持，需要以 '# rows=' 开头的元信息行")
            
            # 从第二行开始读取数据
            data_lines = []
            for i, line in enumerate(lines[1:], 1):
                line = line.strip()
                if line and not line.startswith('#'):  # 跳过注释行
                    try:
                        # 分割行数据（制表符分隔）
                        parts = line.split('\t')
                        if parts:
                            # 转换为浮点数
                            row_data = [float(part.strip()) for part in parts if part.strip()]
                            if row_data:
                                data_lines.append(row_data)
                    except ValueError as e:
                        print(f"⚠️ 跳过第{i+1}行（无法解析）: {e}")
                        continue
            
            if not data_lines:
                raise ValueError("未找到有效数据")
            
            # 检查数据行数是否匹配
            if len(data_lines) != rows:
                print(f"⚠️ 警告: 期望 {rows} 行数据，实际读取到 {len(data_lines)} 行")
            
            # 转换为numpy数组
            data_array = np.array(data_lines)
            print(f"✅ 成功加载数据: {data_array.shape}")
            
            if file_index == 1:
                self.sources1 = data_array
                self.channel_names1 = [f"s{i+1:02d}" for i in range(self.sources1.shape[0])]
                self.sample_rate = 1000
            elif file_index == 2:
                self.sources2 = data_array
                self.channel_names2 = [f"s{i+1:02d}" for i in range(self.sources2.shape[0])]
                self.sample_rate = 1000
                
        except Exception as e:
            print(f"❌ 加载txt文件失败: {e}")
            raise
            
    def load_mat_file(self, file_path, file_index):
        """加载.mat文件"""
        try:
            mat_data = scipy.io.loadmat(file_path)
            
            if file_index == 1:
                if 'sources' in mat_data:
                    self.sources1 = mat_data['sources']
                else:
                    # 查找第一个数值数组
                    for key in mat_data.keys():
                        if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                            if mat_data[key].ndim == 2:
                                self.sources1 = mat_data[key]
                                break
                                
                if self.sources1 is None:
                    raise ValueError("未找到有效的数值数据")
                    
                # 获取通道名称
                if 'channel_names' in mat_data:
                    self.channel_names1 = [str(name) for name in mat_data['channel_names'].flatten()]
                else:
                    self.channel_names1 = [f"s{i+1:02d}" for i in range(self.sources1.shape[0])]
                    
                # 获取采样率
                if 'sample_rate' in mat_data:
                    self.sample_rate = int(mat_data['sample_rate'].flatten()[0])
                else:
                    self.sample_rate = 1000
            elif file_index == 2:
                if 'sources' in mat_data:
                    self.sources2 = mat_data['sources']
                else:
                    # 查找第一个数值数组
                    for key in mat_data.keys():
                        if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                            if mat_data[key].ndim == 2:
                                self.sources2 = mat_data[key]
                                break
                                
                if self.sources2 is None:
                    raise ValueError("未找到有效的数值数据")
                    
                # 获取通道名称
                if 'channel_names' in mat_data:
                    self.channel_names2 = [str(name) for name in mat_data['channel_names'].flatten()]
                else:
                    self.channel_names2 = [f"s{i+1:02d}" for i in range(self.sources2.shape[0])]
                    
                # 获取采样率
                if 'sample_rate' in mat_data:
                    self.sample_rate = int(mat_data['sample_rate'].flatten()[0])
                else:
                    self.sample_rate = 1000
                    
        except Exception as e:
            messagebox.showerror("错误", f"无法加载.mat文件: {str(e)}")
            
    def update_file_info(self, file_path, file_index):
        """更新文件信息显示"""
        if file_index == 1:
            filename = os.path.basename(file_path)
            if self.sources1 is not None:
                info = f"已加载: {filename} | 形状: {self.sources1.shape[0]}×{self.sources1.shape[1]}"
                self.file1_info_label.config(text=info, fg='green')
            else:
                self.file1_info_label.config(text="加载失败", fg='red')
        elif file_index == 2:
            filename = os.path.basename(file_path)
            if self.sources2 is not None:
                info = f"已加载: {filename} | 形状: {self.sources2.shape[0]}×{self.sources2.shape[1]}"
                self.file2_info_label.config(text=info, fg='green')
            else:
                self.file2_info_label.config(text="加载失败", fg='red')
            
    def update_channel_list(self):
        """更新通道列表"""
        # 这个方法现在不需要了，因为我们已经有了源信号开关
        pass
        
    def update_statistics(self):
        """更新统计信息"""
        if self.sources1 is None and self.sources2 is None:
            return
            
        stats_info = f"""数据统计信息:
        
文件1信息:
- 通道数: {self.sources1.shape[0] if self.sources1 is not None else 0}
- 样本数: {self.sources1.shape[1] if self.sources1 is not None else 0}
- 采样率: {self.sample_rate if self.sources1 is not None else 0} Hz
- 时长: {self.sources1.shape[1]/self.sample_rate if self.sources1 is not None else 0:.2f} 秒

文件2信息:
- 通道数: {self.sources2.shape[0] if self.sources2 is not None else 0}
- 样本数: {self.sources2.shape[1] if self.sources2 is not None else 0}
- 采样率: {self.sample_rate if self.sources2 is not None else 0} Hz
- 时长: {self.sources2.shape[1]/self.sample_rate if self.sources2 is not None else 0:.2f} 秒

数值统计:
"""
        
        # 文件1统计
        if self.sources1 is not None:
            stats_info += f"""
文件1统计:
- 全局均值: {np.mean(self.sources1):.6f}
- 全局标准差: {np.std(self.sources1):.6f}
- 全局最小值: {np.min(self.sources1):.6f}
- 全局最大值: {np.max(self.sources1):.6f}

各通道统计:
"""
            for i in range(min(10, self.sources1.shape[0])):  # 只显示前10个通道
                channel_data = self.sources1[i, :]
                stats_info += f"s{i+1:2d}: 均值={np.mean(channel_data):.4f}, 标准差={np.std(channel_data):.4f}\n"
                
            if self.sources1.shape[0] > 10:
                stats_info += f"... 还有 {self.sources1.shape[0] - 10} 个通道\n"
                
        # 文件2统计
        if self.sources2 is not None:
            stats_info += f"""
文件2统计:
- 全局均值: {np.mean(self.sources2):.6f}
- 全局标准差: {np.std(self.sources2):.6f}
- 全局最小值: {np.min(self.sources2):.6f}
- 全局最大值: {np.max(self.sources2):.6f}

各通道统计:
"""
            for i in range(min(10, self.sources2.shape[0])):  # 只显示前10个通道
                channel_data = self.sources2[i, :]
                stats_info += f"s{i+1:2d}: 均值={np.mean(channel_data):.4f}, 标准差={np.std(channel_data):.4f}\n"
                
            if self.sources2.shape[0] > 10:
                stats_info += f"... 还有 {self.sources2.shape[0] - 10} 个通道\n"
                
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_info)
        
    def create_source_switches(self, file_index):
        """创建源信号开关控制"""
        if file_index == 1 and self.sources1 is not None:
            # 清除旧的开关
            for widget in self.file1_source_frame.winfo_children():
                if isinstance(widget, tk.Checkbutton):
                    widget.destroy()
            
            self.source_switches1 = {}
            for i in range(self.sources1.shape[0]):
                source_name = self.channel_names1[i]
                var = tk.BooleanVar(value=True)
                self.source_switches1[source_name] = var
                
                # 创建复选框
                cb = tk.Checkbutton(self.file1_source_frame, text=f"{source_name}", 
                                  variable=var, command=self.redraw_plot,
                                  font=('Arial', 7), fg='darkgreen')
                cb.pack(side=tk.LEFT, padx=2)
                
        elif file_index == 2 and self.sources2 is not None:
            # 清除旧的开关
            for widget in self.file2_source_frame.winfo_children():
                if isinstance(widget, tk.Checkbutton):
                    widget.destroy()
            
            self.source_switches2 = {}
            for i in range(self.sources2.shape[0]):
                source_name = self.channel_names2[i]
                var = tk.BooleanVar(value=True)
                self.source_switches2[source_name] = var
                
                # 创建复选框
                cb = tk.Checkbutton(self.file2_source_frame, text=f"{source_name}", 
                                  variable=var, command=self.redraw_plot,
                                  font=('Arial', 7), fg='darkblue')
                cb.pack(side=tk.LEFT, padx=2)
        
        # 重新绘制图像
        self.redraw_plot()
        
    def redraw_plot(self):
        """重新绘制图像"""
        if not self.data_loaded1 and not self.data_loaded2:
            return
            
        self.ax.clear()
        
        # 获取当前窗口大小和起始位置
        window_size = self.window_size_var.get()
        current_start = getattr(self, 'current_start_sample', 0)
        
        # 确定数据范围
        max_samples = 0
        if self.sources1 is not None:
            max_samples = max(max_samples, self.sources1.shape[1])
        if self.sources2 is not None:
            max_samples = max(max_samples, self.sources2.shape[1])
            
        current_end = min(current_start + window_size, max_samples)
        
        # 清空之前的线条对象
        self.line_objects = []
        
        # 收集所有要显示的源信号，按源编号分组
        sources_by_number = {}  # 按源编号分组
        
        # 添加文件1的源信号
        if self.show_file1_var.get() and self.sources1 is not None:
            for i, source_name in enumerate(self.channel_names1):
                if source_name in self.source_switches1 and self.source_switches1[source_name].get():
                    # 提取名称中的连续数字，避免如"源10"被错误解析
                    digits = ''.join(ch for ch in source_name if ch.isdigit())
                    if not digits:
                        continue
                    source_number = int(digits)
                    if source_number not in sources_by_number:
                        sources_by_number[source_number] = []
                    sources_by_number[source_number].append({
                        'name': source_name,
                        'data': self.sources1[i, current_start:current_end],
                        'file': 1,
                        'index': i
                    })
        
        # 添加文件2的源信号
        if self.show_file2_var.get() and self.sources2 is not None:
            for i, source_name in enumerate(self.channel_names2):
                if source_name in self.source_switches2 and self.source_switches2[source_name].get():
                    # 提取名称中的连续数字，避免如"源10"被错误解析
                    digits = ''.join(ch for ch in source_name if ch.isdigit())
                    if not digits:
                        continue
                    source_number = int(digits)
                    if source_number not in sources_by_number:
                        sources_by_number[source_number] = []
                    sources_by_number[source_number].append({
                        'name': source_name,
                        'data': self.sources2[i, current_start:current_end],
                        'file': 2,
                        'index': i
                    })
        
        if not sources_by_number:
            return
            
        # 按源编号排序
        sorted_source_numbers = sorted(sources_by_number.keys())
        
        # 添加调试信息
        print(f"调试信息: 源编号列表: {sorted_source_numbers}")
        print(f"调试信息: 源数量: {len(sorted_source_numbers)}")
        
        # 计算Y轴位置 - 直接使用源编号作为Y轴位置，确保一直向上绘制
        # 不再使用索引位置，而是直接使用源编号
        y_positions = np.array(sorted_source_numbers)
        
        # 动态计算垂直间距：源越多，间距越大
        if len(sorted_source_numbers) <= 10:
            spacing = 4.0  # 10个源以内使用4.0间距
        elif len(sorted_source_numbers) <= 20:
            spacing = 6.0  # 10-20个源使用6.0间距
        else:
            spacing = 8.0  # 20个源以上使用8.0间距
            
        print(f"调试信息: 使用的垂直间距: {spacing}")
        
        # 获取颜色主题 - 使用固定的颜色数量，确保每个源编号都有固定颜色
        max_source_number = max(sorted_source_numbers) if sorted_source_numbers else 0
        colors = self.get_color_theme(max_source_number + 1)  # +1 因为源编号从1开始
        
        for i, source_number in enumerate(sorted_source_numbers):
            sources_for_this_number = sources_by_number[source_number]
            
            # 为每个源编号添加垂直偏移，避免重叠
            # 直接使用源编号乘以间距，确保一直向上绘制
            offset = source_number * spacing
            
            # 添加调试信息
            print(f"调试信息: 源{source_number:02d} - 位置索引: {i}, 源编号: {source_number}, 偏移量: {offset}")
            
            # 绘制该源编号下的所有文件数据
            for j, source_info in enumerate(sources_for_this_number):
                #y_data = source_info['data'] + offset

                gain = self.gain_var.get()
                y_data = gain * source_info['data'] + offset

                
                # 根据文件选择线型和颜色
                if source_info['file'] == 1:
                    # 文件1：实线，深色
                    line_style = '-'
                    base_color = colors[source_number - 1]  # 使用源编号-1作为颜色索引，确保固定颜色
                    # 创建深色版本
                    color = tuple(max(0, min(1, c * 0.8)) for c in base_color[:3]) + (base_color[3],)
                    alpha = 0.9
                else:
                    # 文件2：虚线，浅色
                    line_style = '--'
                    base_color = colors[source_number - 1]  # 使用源编号-1作为颜色索引，确保固定颜色
                    # 创建浅色版本
                    color = tuple(max(0, min(1, c * 0.6 + 0.4)) for c in base_color[:3]) + (base_color[3],)
                    alpha = 0.7
                
                # 绘制线图
                line, = self.ax.plot(y_data, 
                                   label=f"{source_info['name']} (文件{source_info['file']})",
                                   color=color,
                                   linestyle=line_style,
                                   linewidth=self.linewidth_var.get(),
                                   alpha=alpha)
                self.line_objects.append(line)
            
            # 添加源标签（只添加一次，因为相同编号的源在同一位置）
            source_name = f"s{source_number:02d}"
            # 将标签放在Y轴左侧，确保在可见区域内
            label_x = -window_size * 0.01  # 减少左侧偏移
            self.ax.text(label_x, offset, source_name, 
                       fontsize=10, ha='right', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[source_number - 1], alpha=0.7))
        
        # 设置Y轴刻度为源信号名称
        self.ax.set_yticks(y_positions * spacing)
        self.ax.set_yticklabels([f"s{num:02d}" for num in sorted_source_numbers])
        
        # 设置X轴范围为当前窗口
        self.ax.set_xlim(0, window_size)
        
        # # 设置Y轴范围，确保所有源都可见
        # y_min = -1.0  # 添加一些底部边距
        # y_max = max(sorted_source_numbers) * spacing + 3.0  # 使用最大源编号计算顶部边距
        # self.ax.set_ylim(y_min, y_max)


        # 计算当前窗口的最大幅度，配合增益给一点边距
        max_amp = 0.0
        for grp in sources_by_number.values():
            for s in grp:
                if len(s['data']) > 0:
                    m = float(np.max(np.abs(s['data'])))
                    if m > max_amp:
                        max_amp = m

        pad = 2.0  # 顶部/底部留白
        y_min = -pad
        y_max = max(sorted_source_numbers) * spacing + pad + gain * max_amp
        self.ax.set_ylim(y_min, y_max)


        
        # 添加调试信息
        print(f"调试信息: Y轴范围: {y_min} 到 {y_max}")
        print(f"调试信息: 最大源编号: {max(sorted_source_numbers)}")
        
        # 设置标题
        self.ax.set_title(f"ORICA source signal visualization (samples {current_start}-{current_end})", fontsize=12, fontweight='bold')
        
        # 设置标签
        self.ax.set_xlabel("sample points", fontsize=10)
        self.ax.set_ylabel("source signals", fontsize=10)
        
        # 添加网格
        self.ax.grid(True, alpha=0.3)
        
        # 更新画布
        self.canvas.draw()
        
        # 更新滚动信息
        if hasattr(self, 'scroll_info_label'):
            self.scroll_info_label.config(text=f"sample {current_start}-{current_end}")
            
    def add_scrollbar(self):
        """添加水平滚动条"""
        # 创建滚动条框架
        scroll_frame = tk.Frame(self.scroll_frame)
        scroll_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # 创建滚动条
        window_size = self.window_size_var.get()
        max_start = 0
        if self.sources1 is not None:
            max_start = max(max_start, self.sources1.shape[1] - window_size)
        if self.sources2 is not None:
            max_start = max(max_start, self.sources2.shape[1] - window_size)
        
        self.scrollbar = tk.Scale(scroll_frame, 
                                from_=0, 
                                to=max_start,
                                orient=tk.HORIZONTAL,
                                command=self.on_scroll,
                                resolution=1)  # 精确到1个样本
        
        # 添加标签
        tk.Label(scroll_frame, text="Swipe to see samples:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 添加显示信息标签
        self.scroll_info_label = tk.Label(scroll_frame, text=f"showing: sample 0-{window_size}", font=('Arial', 9))
        self.scroll_info_label.pack(side=tk.RIGHT, padx=5)
        
        # 初始化当前起始样本位置
        self.current_start_sample = 0
        
    def on_scroll(self, value):
        """滚动条事件处理"""
        if not self.data_loaded1 and not self.data_loaded2:
            return
            
        try:
            start_sample = int(float(value))
            window_size = self.window_size_var.get()
            
            # 确定数据范围
            max_samples = 0
            if self.sources1 is not None:
                max_samples = max(max_samples, self.sources1.shape[1])
            if self.sources2 is not None:
                max_samples = max(max_samples, self.sources2.shape[1])
                
            end_sample = min(start_sample + window_size, max_samples)
            
            # 更新当前起始样本位置
            self.current_start_sample = start_sample
            
            # 重新绘制
            self.redraw_plot()
            
        except Exception as e:
            print(f"滚动条更新失败: {e}")
            
    def get_color_theme(self, num_colors):
        """获取颜色主题"""
        theme = self.color_theme_var.get()
        
        if theme == "default":
            return plt.cm.Set1(np.linspace(0, 1, num_colors))
        elif theme == "rainbow":
            return plt.cm.rainbow(np.linspace(0, 1, num_colors))
        elif theme == "grey":
            return plt.cm.Greys(np.linspace(0.2, 0.8, num_colors))
        elif theme == "compare":
            return plt.cm.tab10(np.linspace(0, 1, num_colors))
        else:
            return plt.cm.Set1(np.linspace(0, 1, num_colors))
            
    def update_linewidth(self, value):
        """更新线宽"""
        if self.data_loaded1 or self.data_loaded2:
            self.redraw_plot()
            
    def update_colors(self, event=None):
        """更新颜色主题"""
        if self.data_loaded1 or self.data_loaded2:
            self.redraw_plot()
            
    def update_window_size(self, value):
        """更新样本窗口大小"""
        if self.data_loaded1 or self.data_loaded2:
            new_window_size = int(value)
            
            # 更新滚动条范围
            if hasattr(self, 'scrollbar'):
                max_start = 0
                if self.sources1 is not None:
                    max_start = max(max_start, self.sources1.shape[1] - new_window_size)
                if self.sources2 is not None:
                    max_start = max(max_start, self.sources2.shape[1] - new_window_size)
                    
                self.scrollbar.config(to=max_start)
                
                # 如果当前滚动位置超出新范围，重置为0
                if self.scrollbar.get() > max_start:
                    self.scrollbar.set(0)
                    self.current_start_sample = 0
                    
            # 重新绘制
            self.redraw_plot()
            
    def save_plot(self):
        """保存图像"""
        if not self.data_loaded1 and not self.data_loaded2:
            messagebox.showwarning("警告", "请先加载数据")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="保存图像",
            defaultextension=".png",
            filetypes=[
                ("PNG图像", "*.png"),
                ("JPEG图像", "*.jpg"),
                ("PDF文档", "*.pdf"),
                ("SVG图像", "*.svg")
            ]
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("成功", f"图像已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
                
    def on_motion(self, event):
        """鼠标移动事件"""
        if event.inaxes == self.ax:
            # 显示坐标信息
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # 可以在这里添加坐标显示逻辑
                pass
                
    def on_click(self, event):
        """鼠标点击事件"""
        if event.inaxes == self.ax:
            # 可以在这里添加点击逻辑
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ORICADataVisualizer(root)
    root.mainloop()
