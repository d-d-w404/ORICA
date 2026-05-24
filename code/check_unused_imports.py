#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 Quick30_run 文件夹中所有文件的未使用导入
从 main_gui.py 开始追踪
"""

import ast
import os
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict

def get_imports_from_file(file_path: Path) -> Tuple[Set[str], Set[str]]:
    """解析文件，返回 (导入的模块名集合, 导入的符号名集合)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        imported_modules = set()  # 导入的模块（如 'numpy', 'pandas'）
        imported_symbols = set()  # 导入的符号（如 'np', 'pd', 'LSLStreamReceiver'）
        
        for node in ast.walk(tree):
            # import module
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
                    if alias.asname:
                        imported_symbols.add(alias.asname)
                    else:
                        imported_symbols.add(alias.name.split('.')[0])
            
            # from module import symbol
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)
                    for alias in node.names:
                        if alias.asname:
                            imported_symbols.add(alias.asname)
                        else:
                            imported_symbols.add(alias.name)
        
        return imported_modules, imported_symbols
    except Exception as e:
        print(f"⚠️ 解析 {file_path} 失败: {e}")
        return set(), set()

def get_used_symbols_from_file(file_path: Path) -> Set[str]:
    """获取文件中实际使用的符号名"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        used_symbols = set()
        
        for node in ast.walk(tree):
            # 变量名、函数名、类名等
            if isinstance(node, ast.Name):
                used_symbols.add(node.id)
            # 属性访问 (如 obj.attr)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_symbols.add(node.value.id)
                used_symbols.add(node.attr)
        
        return used_symbols
    except Exception as e:
        print(f"⚠️ 解析 {file_path} 失败: {e}")
        return set()

def find_local_files(import_name: str, base_dir: Path) -> List[Path]:
    """查找本地文件（.py）"""
    # 直接文件名
    direct_path = base_dir / f"{import_name}.py"
    if direct_path.exists():
        return [direct_path]
    
    # 子目录中的文件
    for py_file in base_dir.rglob(f"{import_name}.py"):
        return [py_file]
    
    return []

def analyze_file(file_path: Path, base_dir: Path, analyzed: Set[Path] = None) -> Dict:
    """分析单个文件的导入使用情况"""
    if analyzed is None:
        analyzed = set()
    
    if file_path in analyzed:
        return {}
    
    analyzed.add(file_path)
    
    print(f"\n📄 分析: {file_path.name}")
    
    imported_modules, imported_symbols = get_imports_from_file(file_path)
    used_symbols = get_used_symbols_from_file(file_path)
    
    # 检查未使用的导入
    unused_imports = []
    
    for symbol in imported_symbols:
        # 检查是否在代码中使用
        if symbol not in used_symbols:
            # 特殊处理：某些导入可能通过其他方式使用（如装饰器、类型注解等）
            # 这里简化处理，只检查直接使用
            unused_imports.append(symbol)
    
    result = {
        'file': str(file_path),
        'imported_modules': list(imported_modules),
        'imported_symbols': list(imported_symbols),
        'used_symbols_count': len(used_symbols),
        'unused_imports': unused_imports,
    }
    
    return result

def main():
    base_dir = Path(__file__).parent
    main_file = base_dir / "main_gui.py"
    
    if not main_file.exists():
        print(f"❌ 未找到 {main_file}")
        return
    
    print("=" * 60)
    print("🔍 检查 Quick30_run 文件夹中的未使用导入")
    print("=" * 60)
    
    # 获取所有 Python 文件
    py_files = list(base_dir.glob("*.py"))
    py_files = [f for f in py_files if f.name != "check_unused_imports.py"]
    
    print(f"\n找到 {len(py_files)} 个 Python 文件")
    
    all_results = []
    
    for py_file in sorted(py_files):
        result = analyze_file(py_file, base_dir)
        if result:
            all_results.append(result)
    
    # 汇总未使用的导入
    print("\n" + "=" * 60)
    print("📊 未使用的导入汇总")
    print("=" * 60)
    
    total_unused = 0
    for result in all_results:
        if result['unused_imports']:
            print(f"\n📄 {Path(result['file']).name}:")
            for imp in result['unused_imports']:
                print(f"   - {imp}")
                total_unused += 1
    
    if total_unused == 0:
        print("\n✅ 没有发现未使用的导入！")
    else:
        print(f"\n📈 总共发现 {total_unused} 个可能未使用的导入")
        print("\n⚠️  注意：某些导入可能通过反射、装饰器等方式间接使用，")
        print("   请手动检查确认后再删除。")

if __name__ == "__main__":
    main()


