#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测结果绘图脚本
用于分析和可视化在线学习的预测结果
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def load_prediction_results(filepath):
    """加载预测结果JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded: {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def plot_accuracy_from_json(data, title="Prediction Accuracy Over Time", save_plot=True, show_plot=True):
    """从JSON数据绘制准确率图表"""
    if not data or 'statistics' not in data:
        print("❌ Invalid data format")
        return
    
    stats = data['statistics']
    accuracy_history = stats.get('accuracy_history', [])
    prediction_history = stats.get('prediction_history', [])
    
    if not accuracy_history:
        print("❌ No accuracy history data found")
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: 准确率趋势
    ax1.plot(accuracy_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title(f'{title} - Accuracy Trend', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Prediction Number', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 添加平均线
    if len(accuracy_history) > 1:
        avg_accuracy = np.mean(accuracy_history)
        ax1.axhline(y=avg_accuracy, color='r', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_accuracy:.3f}')
        ax1.legend()
    
    # 子图2: 预测vs实际标签
    if prediction_history:
        predictions = [p['prediction'] for p in prediction_history]
        actual_labels = [p['true_label'] for p in prediction_history]
        correct_predictions = [p['is_correct'] for p in prediction_history]
        
        x = range(len(predictions))
        colors = ['green' if correct else 'red' for correct in correct_predictions]
        
        ax2.scatter(x, predictions, c=colors, alpha=0.7, s=50, label='Predictions')
        ax2.scatter(x, actual_labels, c='blue', alpha=0.5, s=30, marker='s', label='Actual')
        ax2.set_title('Predictions vs Actual Labels', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Prediction Number', fontsize=12)
        ax2.set_ylabel('Label (7=Left, 8=Right)', fontsize=12)
        ax2.set_ylim(6.5, 8.5)
        ax2.set_yticks([7, 8])
        ax2.set_yticklabels(['Left (7)', 'Right (8)'])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    
    # 保存图表
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"accuracy_plot_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {plot_filename}")
    
    # 显示图表
    if show_plot:
        plt.show()
    
    return fig

def print_summary(data):
    """打印预测结果摘要"""
    if not data or 'statistics' not in data:
        print("❌ Invalid data format")
        return
    
    stats = data['statistics']
    model_info = data.get('model_info', {})
    
    print("\n" + "="*60)
    print("📊 PREDICTION RESULTS SUMMARY")
    print("="*60)
    
    # 基本信息
    print(f"📅 Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"🧠 Model Type: {model_info.get('model_type', 'N/A')}")
    print(f"📈 Feature Type: {model_info.get('feature_type', 'N/A')}")
    print(f"🔢 Feature Dimension: {model_info.get('feature_dim', 'N/A')}")
    
    # 统计信息
    print(f"\n📊 Statistics:")
    print(f"   Total Predictions: {stats.get('total_predictions', 0)}")
    print(f"   Correct Predictions: {stats.get('correct_predictions', 0)}")
    print(f"   Final Accuracy: {stats.get('accuracy', 0):.3f}")
    
    # 准确率历史
    accuracy_history = stats.get('accuracy_history', [])
    if accuracy_history:
        print(f"   Accuracy Range: {min(accuracy_history):.3f} - {max(accuracy_history):.3f}")
        print(f"   Average Accuracy: {np.mean(accuracy_history):.3f}")
    
    # 预测历史
    prediction_history = stats.get('prediction_history', [])
    if prediction_history:
        print(f"\n🎯 Recent Predictions (last 5):")
        for i, pred in enumerate(prediction_history[-5:]):
            status = "✅" if pred['is_correct'] else "❌"
            print(f"   {i+1}. Target: {pred['true_label']}, Prediction: {pred['prediction']}, {status}")
    
    print("="*60)

def main():
    """主函数"""
    print("📊 Prediction Results Plotter")
    print("="*40)
    
    # 要分析的文件列表
    files_to_analyze = [
        "./Results/prediction_results_20250811_042312.json"
    ]
    
    # 绘图选项
    save_plot = True
    show_plot = True
    comparison_mode = False
    
    # 分析每个文件
    for filepath in files_to_analyze:
        print(f"\n📁 Analyzing: {filepath}")
        
        # 加载数据
        data = load_prediction_results(filepath)
        if data is None:
            continue
        
        # 打印摘要
        print_summary(data)
        
        # 绘制图表
        title = f"Prediction Results - {os.path.basename(filepath)}"
        plot_accuracy_from_json(data, title=title, save_plot=save_plot, show_plot=show_plot)
    
    print(f"\n✅ Analysis completed for {len(files_to_analyze)} file(s)")

if __name__ == "__main__":
    main()
