#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORICA评估结果可视化脚本
用于分析和绘制ORICA分离质量的评估结果
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ORICAEvaluationPlotter:
    """ORICA评估结果可视化器"""
    
    def __init__(self, results_dir="./Results"):
        self.results_dir = results_dir
        self.evaluation_files = []
        self.evaluation_data = []
        
    def load_evaluation_files(self, pattern="orica_evaluation.json"):
        """加载评估结果文件"""
        search_pattern = os.path.join(self.results_dir, pattern)
        self.evaluation_files = glob.glob(search_pattern)
        self.evaluation_files.sort()  # 按文件名排序
        
        print(f"📁 找到 {len(self.evaluation_files)} 个评估文件:")
        for f in self.evaluation_files:
            print(f"  - {os.path.basename(f)}")
        
        # 加载数据
        self.evaluation_data = []
        for file_path in self.evaluation_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['file_path'] = file_path
                    data['filename'] = os.path.basename(file_path)
                    self.evaluation_data.append(data)
            except Exception as e:
                print(f"❌ 加载文件失败 {file_path}: {e}")
        
        print(f"✅ 成功加载 {len(self.evaluation_data)} 个评估结果")
        
    def plot_temporal_evolution(self, save_plot=True):
        """绘制评估指标随时间的变化"""
        if not self.evaluation_data:
            print("❌ 没有评估数据，请先加载文件")
            return
        
        # 提取时间序列数据
        timestamps = []
        kurtosis_values = []
        mi_values = []
        
        for data in self.evaluation_data:
            try:
                # 解析时间戳
                if 'timestamp' in data:
                    dt_str = data['timestamp']
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    timestamps.append(dt)
                else:
                    timestamps.append(datetime.now())
                
                # 提取指标
                kurtosis_values.append(data['kurtosis'])
                mi_values.append(data['mutual_info'])
                
            except Exception as e:
                print(f"⚠️ 处理数据时出错: {e}")
                continue
        
        if len(timestamps) < 2:
            print("❌ 数据点不足，无法绘制时间序列")
            return
        
        # 定义"良好区间"的阈值
        # 峭度：一般认为 > 3.0 表示非高斯性良好
        kurtosis_good_threshold = 3.0
        # 互信息：一般认为 < 0.05 表示独立性良好
        mi_good_threshold = 0.05
        
        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('ORICA评估指标时间演化', fontsize=16, fontweight='bold')
        
        # 峭度均值变化
        axes[0].plot(timestamps, kurtosis_values, 'o-', color='blue', linewidth=2, markersize=6)
        
        # 添加"良好区间"阴影区域
        axes[0].axhspan(kurtosis_good_threshold, max(kurtosis_values) + 0.5, 
                        alpha=0.2, color='green', label=f'良好区间 (>{kurtosis_good_threshold})')
        axes[0].axhline(y=kurtosis_good_threshold, color='green', linestyle='--', 
                       alpha=0.7, linewidth=1)
        
        axes[0].set_ylabel('峭度均值', fontsize=12)
        axes[0].set_title('峭度均值变化趋势 (越高越好)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend()
        
        # 互信息均值变化
        axes[1].plot(timestamps, mi_values, 's-', color='red', linewidth=2, markersize=6)
        
        # 添加"良好区间"阴影区域
        axes[1].axhspan(0, mi_good_threshold, alpha=0.2, color='green', 
                        label=f'良好区间 (<{mi_good_threshold})')
        axes[1].axhline(y=mi_good_threshold, color='green', linestyle='--', 
                       alpha=0.7, linewidth=1)
        
        axes[1].set_ylabel('互信息均值', fontsize=12)
        axes[1].set_xlabel('时间', fontsize=12)
        axes[1].set_title('互信息均值变化趋势 (越低越好)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"./Results/orica_temporal_evolution_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"💾 时间演化图已保存: {plot_filename}")
        
        plt.show()
        
    def plot_component_analysis(self, file_index=0, save_plot=True):
        """绘制单个评估文件的详细组件分析"""
        if not self.evaluation_data or file_index >= len(self.evaluation_data):
            print("❌ 无效的文件索引")
            return
        
        data = self.evaluation_data[file_index]
        filename = data['filename']
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'ORICA组件分析 - {filename}', fontsize=16, fontweight='bold')
        
        # 1. 峭度值
        kurt_val = data['kurtosis']
        axes[0].bar(['峭度均值'], [kurt_val], color='skyblue', alpha=0.7)
        axes[0].set_ylabel('峭度值')
        axes[0].set_title('峭度均值 (越高越非高斯)')
        axes[0].grid(True, alpha=0.3)
        
        # 添加数值标签
        axes[0].text(0, kurt_val + 0.1, f'{kurt_val:.3f}', 
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 2. 互信息值
        mi_val = data['mutual_info']
        axes[1].bar(['互信息均值'], [mi_val], color='lightcoral', alpha=0.7)
        axes[1].set_ylabel('互信息值')
        axes[1].set_title('互信息均值 (越低越独立)')
        axes[1].grid(True, alpha=0.3)
        
        # 添加数值标签
        axes[1].text(0, mi_val + 0.001, f'{mi_val:.3f}', 
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"./Results/orica_component_analysis_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"💾 组件分析图已保存: {plot_filename}")
        
        plt.show()
        
    def plot_comparison_summary(self, save_plot=True):
        """绘制多个评估结果的对比总结"""
        if len(self.evaluation_data) < 2:
            print("❌ 需要至少2个评估结果才能进行对比")
            return
        
        # 提取对比数据
        filenames = [data['filename'] for data in self.evaluation_data]
        kurtosis_values = [data['kurtosis'] for data in self.evaluation_data]
        mi_values = [data['mutual_info'] for data in self.evaluation_data]
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('ORICA评估结果对比分析', fontsize=16, fontweight='bold')
        
        # 1. 峭度均值对比
        x_pos = np.arange(len(filenames))
        bars1 = axes[0].bar(x_pos, kurtosis_values, color='skyblue', alpha=0.7)
        axes[0].set_xlabel('评估文件')
        axes[0].set_ylabel('峭度均值')
        axes[0].set_title('峭度均值对比 (越高越好)')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f.split('_')[2][:8] for f in filenames], rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, kurtosis_values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # 2. 互信息均值对比
        bars2 = axes[1].bar(x_pos, mi_values, color='lightcoral', alpha=0.7)
        axes[1].set_xlabel('评估文件')
        axes[1].set_ylabel('互信息均值')
        axes[1].set_title('互信息均值对比 (越低越好)')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f.split('_')[2][:8] for f in filenames], rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars2, mi_values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"./Results/orica_comparison_summary_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"💾 对比总结图已保存: {plot_filename}")
        
        plt.show()
        
    def generate_summary_report(self, save_report=True):
        """生成评估结果总结报告"""
        if not self.evaluation_data:
            print("❌ 没有评估数据，请先加载文件")
            return
        
        # 生成报告
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ORICA评估结果总结报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"评估文件数量: {len(self.evaluation_data)}")
        report_lines.append("")
        
        # 总体统计
        all_kurtosis = [data['kurtosis'] for data in self.evaluation_data]
        all_mi = [data['mutual_info'] for data in self.evaluation_data]
        
        report_lines.append("📊 总体统计:")
        report_lines.append(f"  峭度均值范围: {min(all_kurtosis):.3f} - {max(all_kurtosis):.3f}")
        report_lines.append(f"  峭度均值: {np.mean(all_kurtosis):.3f} ± {np.std(all_kurtosis):.3f}")
        report_lines.append(f"  互信息均值范围: {min(all_mi):.3f} - {max(all_mi):.3f}")
        report_lines.append(f"  互信息均值: {np.mean(all_mi):.3f} ± {np.std(all_mi):.3f}")
        report_lines.append("")
        
        # 各文件详细结果
        report_lines.append("📁 各文件详细结果:")
        for i, data in enumerate(self.evaluation_data):
            report_lines.append(f"  文件 {i+1}: {data['filename']}")
            report_lines.append(f"    峭度均值: {data['kurtosis']:.3f}")
            report_lines.append(f"    互信息均值: {data['mutual_info']:.3f}")
            report_lines.append(f"    时间: {data['timestamp']}")
            report_lines.append("")
        
        # 最佳和最差结果
        best_idx = np.argmax(all_kurtosis)  # 峭度越高越好
        worst_idx = np.argmin(all_kurtosis)
        
        report_lines.append("🏆 最佳结果:")
        report_lines.append(f"  文件: {self.evaluation_data[best_idx]['filename']}")
        report_lines.append(f"  峭度均值: {all_kurtosis[best_idx]:.3f}")
        report_lines.append(f"  互信息均值: {all_mi[best_idx]:.3f}")
        report_lines.append("")
        
        report_lines.append("⚠️ 最差结果:")
        report_lines.append(f"  文件: {self.evaluation_data[worst_idx]['filename']}")
        report_lines.append(f"  峭度均值: {all_kurtosis[worst_idx]:.3f}")
        report_lines.append(f"  互信息均值: {all_mi[worst_idx]:.3f}")
        report_lines.append("")
        
        # 建议
        report_lines.append("💡 分析建议:")
        if np.std(all_kurtosis) < 0.5:
            report_lines.append("  - 峭度值稳定，ORICA算法表现一致")
        else:
            report_lines.append("  - 峭度值波动较大，建议检查数据质量或算法参数")
        
        if np.mean(all_kurtosis) > 3.0:
            report_lines.append("  - 峭度较高，非高斯性良好，分离效果较好")
        else:
            report_lines.append("  - 峭度较低，建议检查数据预处理或增加数据量")
        
        if np.mean(all_mi) < 0.05:
            report_lines.append("  - 互信息较低，独立性良好")
        else:
            report_lines.append("  - 互信息较高，独立性有待改善")
        
        report_lines.append("=" * 80)
        
        # 打印报告
        for line in report_lines:
            print(line)
        
        # 保存报告
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"./Results/orica_summary_report_{timestamp}.txt"
            
            try:
                os.makedirs(os.path.dirname(report_filename), exist_ok=True)
                with open(report_filename, 'w', encoding='utf-8') as f:
                    for line in report_lines:
                        f.write(line + '\n')
                print(f"\n💾 总结报告已保存: {report_filename}")
            except Exception as e:
                print(f"❌ 保存报告失败: {e}")

def main():
    """主函数"""
    print("🎨 ORICA评估结果可视化工具")
    print("=" * 50)
    
    # 创建可视化器
    plotter = ORICAEvaluationPlotter()
    
    # 加载评估文件
    plotter.load_evaluation_files()
    
    if not plotter.evaluation_data:
        print("❌ 没有找到评估文件，请先运行ORICA评估")
        return
    
    # 生成所有可视化
    print("\n📈 生成时间演化图...")
    plotter.plot_temporal_evolution()
    
    print("\n🔍 生成组件分析图...")
    plotter.plot_component_analysis(file_index=0)  # 分析第一个文件
    
    if len(plotter.evaluation_data) > 1:
        print("\n📊 生成对比总结图...")
        plotter.plot_comparison_summary()
    
    print("\n📋 生成总结报告...")
    plotter.generate_summary_report()
    
    print("\n✅ 所有可视化完成！")

if __name__ == "__main__":
    main()
