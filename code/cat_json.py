#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON文件查看器
类似cat命令，用于查看JSON文件内容
"""

import json
import sys
import os

def cat_json(filepath):
    """查看JSON文件内容"""
    try:
        # 检查文件是否存在
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        
        # 读取文件内容
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析JSON
        try:
            data = json.loads(content)
            # 格式化输出
            formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
            print(formatted_json)
            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            print("📄 Raw file content:")
            print(content)
            return False
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("Usage: python cat_json.py <json_file>")
        print("Example: python cat_json.py ./Results/prediction_results_20250728_170618.json")
        return
    
    filepath = sys.argv[1]
    print(f"📁 Viewing: {filepath}")
    print("="*50)
    
    success = cat_json(filepath)
    
    if success:
        print("\n✅ JSON file displayed successfully")
    else:
        print("\n❌ Failed to display JSON file")

if __name__ == "__main__":
    main() 