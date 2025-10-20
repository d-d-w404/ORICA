#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ORICA连续保存功能的脚本
"""

import os
import json
import csv
from datetime import datetime
import numpy as np

def test_continuous_saving():
    """测试连续保存功能"""
    print("🧪 测试ORICA连续保存功能")
    
    # 确保Results目录存在
    os.makedirs('./Results', exist_ok=True)
    
    # 测试数据
    test_data = [
        {
            'timestamp': '2025-01-27T10:00:00',
            'kurtosis_mean': 3.5,
            'mi_mean': 0.03,
            'n_components': 25,
            'n_samples': 5000
        },
        {
            'timestamp': '2025-01-27T10:01:00',
            'kurtosis_mean': 3.8,
            'mi_mean': 0.02,
            'n_components': 25,
            'n_samples': 5000
        },
        {
            'timestamp': '2025-01-27T10:02:00',
            'kurtosis_mean': 4.1,
            'mi_mean': 0.01,
            'n_components': 25,
            'n_samples': 5000
        }
    ]
    
    # 测试JSON追加
    json_filename = './Results/orica_evaluation_continuous.json'
    print(f"\n📝 测试JSON追加到: {json_filename}")
    
    for i, data in enumerate(test_data):
        print(f"  添加数据 {i+1}: kurtosis={data['kurtosis_mean']:.1f}, MI={data['mi_mean']:.2f}")
        
        # 模拟追加过程
        if not os.path.exists(json_filename):
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump([data], f, ensure_ascii=False, indent=2)
        else:
            try:
                with open(json_filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
            
            if isinstance(existing_data, list):
                existing_data.append(data)
            else:
                existing_data = [existing_data, data]
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    # 测试CSV追加
    csv_filename = './Results/orica_evaluation_continuous.csv'
    print(f"\n📝 测试CSV追加到: {csv_filename}")
    
    for i, data in enumerate(test_data):
        print(f"  添加数据 {i+1}: kurtosis={data['kurtosis_mean']:.1f}, MI={data['mi_mean']:.2f}")
        
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                writer.writeheader()
                writer.writerow(data)
        else:
            with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                writer.writerow(data)
    
    # 验证结果
    print(f"\n✅ 测试完成！")
    print(f"📁 JSON文件: {json_filename}")
    print(f"📁 CSV文件: {csv_filename}")
    
    # 显示JSON内容
    if os.path.exists(json_filename):
        with open(json_filename, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        print(f"\n📊 JSON文件包含 {len(loaded_data)} 条记录")
        for i, item in enumerate(loaded_data):
            print(f"  记录 {i+1}: {item['timestamp']} - kurtosis={item['kurtosis_mean']:.1f}, MI={item['mi_mean']:.2f}")
    
    # 显示CSV内容
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"\n📊 CSV文件包含 {len(lines)-1} 条记录（不包括表头）")

if __name__ == "__main__":
    test_continuous_saving()


















































































