#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将标签添加到只有EEG数据的npz文件中
读取：processed_data_*.npz (只有 'data' 键)
标签来源：stress_ratings_processed.mat
输出：包含 'X' 和 'y' 的新npz文件（用于EEGNet训练）
"""

import numpy as np
from pathlib import Path
import scipy.io as sio
import re

# ========== 配置 ==========
# 使用绝对路径或相对于脚本所在目录的路径
import os
BASE_DIR = Path(__file__).parent.parent  # 项目根目录
NPZ_FILE = BASE_DIR / 'Quick30' / 'train_data' / 'eegnet_data_1307_after_con_orica.npz'  # 输入的npz文件（只有EEG数据）
MAT_FILE = BASE_DIR / 'Quick30' / 'train_data' / 'stress_ratings_processed.mat'  # 标签文件
OUTPUT_FILE = BASE_DIR / 'Quick30' / 'train_data' / 'eegnet_data_1307_with_labels.npz'  # 输出的npz文件

WINDOW_SEC = 2.0  # 窗口长度（秒）
STEP_SEC = 1.0    # 步长（秒，50%重叠）

# 标签策略：'fixed_two_bins' 表示1-5为class 0，5-10为class 1
LABEL_STRATEGY = 'fixed_two_bins'
# ==========================

def _deref(obj):
    """去除多余的维度/单元素包装"""
    try:
        if isinstance(obj, np.ndarray):
            obj = np.squeeze(obj)
            if obj.dtype == object and obj.size == 1:
                return obj.item()
        return obj
    except Exception:
        return obj

def _get_field(obj, key):
    """从对象中获取字段：支持 dict、dtype.names、属性等"""
    try:
        # dict
        if isinstance(obj, dict) and key in obj:
            val = _deref(obj[key])
            if callable(val):
                return None
            return val
        # numpy 结构体
        if isinstance(obj, np.ndarray) and obj.dtype.names and key in obj.dtype.names:
            val = _deref(obj[key])
            if callable(val):
                return None
            return val
        # 属性
        if hasattr(obj, key):
            val = _deref(getattr(obj, key))
            if callable(val):
                return None
            return val
        # 单元素数组展开后再尝试
        if isinstance(obj, np.ndarray) and obj.size == 1:
            return _get_field(obj.item(), key)
    except Exception:
        return None
    return None

def load_labels_from_mat(mat_path: Path, subject_field: str):
    """从stress_ratings_processed.mat中读取指定subject的标签"""
    try:
        mat = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    except Exception as e:
        print(f"❌ 无法读取 {mat_path.name}: {e}")
        return None

    # 优先解析 stress.subXXXX.all
    stress = mat.get('stress', None)
    if stress is not None:
        sub = _get_field(stress, subject_field)
        if sub is None and isinstance(stress, np.ndarray) and stress.size == 1:
            sub = _get_field(stress.item(), subject_field)
        if sub is None:
            return None
        else:
            ratings = _get_field(sub, 'all')
            if ratings is None:
                ratings = _get_field(sub, 'labels')
            time_sec = None
            for k in ['time', 'time_sec', 't']:
                time_sec = _get_field(sub, k)
                if time_sec is not None:
                    break
            if ratings is not None:
                ratings = np.squeeze(np.array(ratings)).astype(float)
                if time_sec is not None:
                    time_sec = np.squeeze(np.array(time_sec)).astype(float)
                return {'values': ratings, 'time_sec': time_sec}
    return None

def infer_subject_from_npz_filename(npz_path: Path):
    """从npz文件名中推断subject ID"""
    # 尝试从文件名中提取数字
    m = re.search(r'(\d{3,4})', npz_path.stem)
    if m:
        return f'sub{m.group(1)}'
    return None

def main():
    npz_path = Path(NPZ_FILE)
    mat_path = Path(MAT_FILE)
    out_path = Path(OUTPUT_FILE)
    
    # 检查文件存在
    if not npz_path.exists():
        print(f"❌ 未找到npz文件: {npz_path}")
        return
    if not mat_path.exists():
        print(f"❌ 未找到标签文件: {mat_path}")
        return
    
    print("=" * 60)
    print("📂 读取EEG数据...")
    # 读取npz文件
    npz_data = np.load(npz_path)
    
    # 检查npz文件中的键
    print(f"   npz文件中的键: {list(npz_data.keys())}")
    
    # 尝试从不同键名读取数据
    if 'data' in npz_data:
        eeg_data = npz_data['data']  # (channels, samples)
        print(f"   ✅ 从'data'键读取数据: shape={eeg_data.shape}")
    elif 'X' in npz_data:
        eeg_data = npz_data['X']
        print(f"   ✅ 从'X'键读取数据: shape={eeg_data.shape}")
        # 如果是(N, C, T)格式，需要转换为连续数据
        if len(eeg_data.shape) == 3:
            print("   ⚠️ 数据已经是窗口格式，需要转换为连续格式")
            # 这里假设可以重组，但更可能是需要用户提供原始连续数据
            print("   ❌ 错误：数据已经是窗口格式，无法添加标签")
            return
    else:
        print(f"   ❌ 未找到'data'或'X'键，可用键: {list(npz_data.keys())}")
        return
    
    # 获取采样率
    if 'sampling_rate' in npz_data:
        fs = int(npz_data['sampling_rate'])
    elif 'srate' in npz_data:
        fs = int(npz_data['srate'])
    else:
        # 默认采样率
        fs = 500
        print(f"   ⚠️ 未找到采样率信息，使用默认值: {fs} Hz")
    
    print(f"   采样率: {fs} Hz")
    print(f"   EEG数据形状: {eeg_data.shape} (channels, samples)")
    print(f"   总时长: {eeg_data.shape[1]/fs:.2f} 秒")
    
    # 推断subject ID
    subject_field = infer_subject_from_npz_filename(npz_path)
    if not subject_field:
        print("   ⚠️ 无法从文件名推断subject ID，尝试使用默认值...")
        # 尝试从文件名中查找
        if '1307' in npz_path.stem:
            subject_field = 'sub1307'
        else:
            print("   ❌ 请手动指定subject字段")
            subject_field = input("   请输入subject字段（如sub1307）: ").strip()
    
    print(f"\n📂 读取标签数据 (subject: {subject_field})...")
    labels_info = load_labels_from_mat(mat_path, subject_field)
    
    if labels_info is None:
        print("   ❌ 无法加载标签数据")
        return
    
    ratings = labels_info['values']
    time_sec = labels_info.get('time_sec', None)
    
    print(f"   ✅ 标签加载成功")
    print(f"   标签数量: {len(ratings)}")
    print(f"   评分范围: {np.nanmin(ratings):.2f} ~ {np.nanmax(ratings):.2f}")
    print(f"   是否有时间轴: {'是' if time_sec is not None else '否'}")
    
    # 将标签对齐到EEG时间轴
    print("\n🔄 对齐标签到EEG时间轴...")
    eeg_total_sec = eeg_data.shape[1] / fs
    eeg_time = np.arange(0.0, eeg_total_sec, 1.0 / fs)
    if len(eeg_time) > eeg_data.shape[1]:
        eeg_time = eeg_time[:eeg_data.shape[1]]
    
    if time_sec is None:
        # 如果没有时间轴，假设均匀分布
        if len(ratings) == len(eeg_time):
            ratings_aligned = ratings.astype(float)
        else:
            src_time = np.linspace(0.0, eeg_total_sec, num=len(ratings), endpoint=False)
            ratings_aligned = np.interp(eeg_time, src_time, ratings.astype(float))
    else:
        # 有时间轴，使用插值
        print(f"   EEG时间范围: 0s ~ {eeg_total_sec:.2f}s")
        print(f"   标签时间范围: {time_sec.min():.2f}s ~ {time_sec.max():.2f}s")
        ratings_aligned = np.interp(eeg_time, time_sec.astype(float), ratings.astype(float))
    
    print(f"   ✅ 标签对齐完成，对齐后长度: {len(ratings_aligned)}")
    
    # 分割数据成窗口并分配标签
    print("\n🔄 分割数据成窗口...")
    win_len = int(WINDOW_SEC * fs)
    step_len = int(STEP_SEC * fs)
    
    X_list, y_list = [], []
    
    for s in range(0, eeg_data.shape[1] - win_len + 1, step_len):
        # 提取窗口数据
        clip = eeg_data[:, s:s+win_len]  # (channels, window_samples)
        X_list.append(clip.astype(np.float32))
        
        # 计算窗口内的平均评分作为标签
        win_mean = float(np.mean(ratings_aligned[s:s+win_len]))
        
        # 根据策略分配类别
        if LABEL_STRATEGY == 'fixed_two_bins':
            if win_mean <= 5.0:
                win_lbl = 0  # 1-5
            else:
                win_lbl = 1  # 5-10
        else:
            # 其他策略可以在这里添加
            win_lbl = int(np.rint(win_mean))
        
        y_list.append(win_lbl)
    
    if len(X_list) == 0:
        print("   ❌ 未切出任何样本")
        return
    
    # 转换为数组
    X = np.stack(X_list, axis=0)  # (N, C, T)
    y = np.array(y_list, dtype=np.int64)
    
    print(f"   ✅ 窗口分割完成")
    print(f"   X形状: {X.shape} (N, C, T) = ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")
    print(f"   y形状: {y.shape}")
    
    # 统计标签分布
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\n📊 标签分布:")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"   class {lbl}: {cnt} 个样本 ({cnt/len(y)*100:.1f}%)")
    
    # 保存文件
    print(f"\n💾 保存文件: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y)
    
    print("✅ 完成！")
    print("=" * 60)
    print(f"输出文件: {out_path}")
    print(f"X形状: {X.shape}")
    print(f"y形状: {y.shape}")
    print("=" * 60)

if __name__ == '__main__':
    main()

