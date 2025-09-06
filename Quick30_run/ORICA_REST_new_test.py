#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORICA_REST_new 测试脚本
读取 .set 文件，应用ORICA算法，保存结果
"""

import numpy as np
import scipy.io as sio
import os
import sys
from datetime import datetime
import argparse

# 添加当前目录到Python路径，以便导入ORICA_REST_new
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ORICA_REST_new import ORICAZ
except ImportError:
    print("❌ 无法导入 ORICA_REST_new，请确保文件在同一目录下")
    sys.exit(1)

def load_set_file(set_file_path):
    """
    读取 .set 文件
    
    Args:
        set_file_path: .set 文件路径
        
    Returns:
        data: numpy数组，形状 (samples, channels)
        channel_names: 通道名称列表
        sample_rate: 采样率
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(set_file_path):
            raise FileNotFoundError(f"文件不存在: {set_file_path}")
        
        # 读取 .set 文件
        print(f"📁 正在读取文件: {set_file_path}")
        
        # 使用 scipy.io 读取 .set 文件
        # 注意：.set 文件是EEGLAB格式，可能需要特殊处理
        # 这里我们尝试直接读取，如果失败则提供替代方案
        
        try:
            # 尝试直接读取
            mat_data = sio.loadmat(set_file_path)
            print(f"✅ 成功读取 .set 文件")
            
            # 查找数据字段
            data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
            print(f"📊 文件包含的字段: {data_keys}")
            
            # 通常EEG数据存储在 'EEG' 字段中，需要正确提取data子字段
            if 'EEG' in mat_data:
                eeg_data = mat_data['EEG']
                print(f"📊 EEG结构体字段: {eeg_data.dtype.names if hasattr(eeg_data, 'dtype') and hasattr(eeg_data.dtype, 'names') else '无字段信息'}")
                
                if isinstance(eeg_data, np.ndarray) and eeg_data.dtype.names:
                    # EEG是结构体数组，需要提取data字段
                    if 'data' in eeg_data.dtype.names:
                        data = eeg_data['data'][0, 0]  # 提取第一个元素的data字段
                        print(f"✅ 从EEG结构体中提取data字段，形状: {data.shape}")
                    else:
                        raise ValueError("EEG结构体中未找到'data'字段")
                elif isinstance(eeg_data, np.ndarray):
                    data = eeg_data
                else:
                    raise ValueError("无法解析EEG数据结构")
            elif 'data' in mat_data:
                data = mat_data['data']
            else:
                # 如果没有找到标准字段，使用第一个数值数组
                numeric_keys = [key for key in data_keys if isinstance(mat_data[key], np.ndarray) and mat_data[key].dtype.kind in 'fc']
                if numeric_keys:
                    data = mat_data[numeric_keys[0]]
                    print(f"⚠️ 使用字段 '{numeric_keys[0]}' 作为数据源")
                else:
                    raise ValueError("未找到有效的数值数据字段")
            
            # 确保数据是二维的
            print(f"🔍 原始数据形状: {data.shape}, 数据类型: {data.dtype}")
            
            if data.ndim == 3:
                # 如果是3D数据 (channels, samples, epochs)，取第一个epoch
                data = data[:, :, 0].T  # 转置为 (samples, channels)
                print(f"⚠️ 3D数据，使用第一个epoch，形状: {data.shape}")
            elif data.ndim == 2:
                # 如果是2D数据，检查是否需要转置
                if data.shape[0] < data.shape[1]:
                    # 如果第一个维度小于第二个维度，可能是 (channels, samples)
                    data = data.T  # 转置为 (samples, channels)
                    print(f"⚠️ 数据已转置，形状: {data.shape}")
            else:
                raise ValueError(f"不支持的数据维度: {data.ndim}")
            
            # 检查数据是否为数值类型
            if not np.issubdtype(data.dtype, np.number):
                print(f"⚠️ 数据不是数值类型，尝试转换...")
                try:
                    data = data.astype(np.float64)
                    print(f"✅ 数据已转换为float64类型")
                except Exception as e:
                    print(f"❌ 数据转换失败: {e}")
                    raise ValueError(f"无法将数据转换为数值类型: {data.dtype}")
            
            # 获取通道信息
            channel_names = []
            if 'EEG' in mat_data and hasattr(eeg_data, 'dtype') and hasattr(eeg_data.dtype, 'names'):
                if 'chanlocs' in eeg_data.dtype.names:
                    chanlocs = eeg_data['chanlocs'][0, 0]
                    if hasattr(chanlocs, 'dtype') and hasattr(chanlocs.dtype, 'names'):
                        if 'labels' in chanlocs.dtype.names:
                            try:
                                labels = chanlocs['labels'][0, 0]
                                channel_names = [str(label[0]) for label in labels]
                                print(f"✅ 从EEG结构体提取通道名称: {len(channel_names)} 个通道")
                            except Exception as e:
                                print(f"⚠️ 通道名称提取失败: {e}")
                        else:
                            print(f"⚠️ chanlocs中未找到'labels'字段")
                    else:
                        print(f"⚠️ chanlocs不是结构体")
                else:
                    print(f"⚠️ EEG结构体中未找到'chanlocs'字段")
            
            # 如果没有通道信息，创建默认通道名
            if not channel_names:
                # 从数据形状推断通道数
                n_channels = data.shape[1] if data.ndim >= 2 else 1
                channel_names = [f'Ch{i+1:02d}' for i in range(n_channels)]
                print(f"⚠️ 使用默认通道名称: {len(channel_names)} 个通道")
            
            # 获取采样率
            sample_rate = 1000  # 默认采样率
            if 'EEG' in mat_data and hasattr(eeg_data, 'dtype') and hasattr(eeg_data.dtype, 'names'):
                if 'srate' in eeg_data.dtype.names:
                    try:
                        sample_rate = int(eeg_data['srate'][0, 0])
                        print(f"✅ 从EEG结构体提取采样率: {sample_rate} Hz")
                    except Exception as e:
                        print(f"⚠️ 采样率提取失败: {e}, 使用默认值: {sample_rate} Hz")
                else:
                    print(f"⚠️ EEG结构体中未找到'srate'字段，使用默认值: {sample_rate} Hz")
            else:
                print(f"⚠️ 使用默认采样率: {sample_rate} Hz")
            
            print(f"✅ 数据加载成功:")
            print(f"   - 数据形状: {data.shape}")
            print(f"   - 通道数: {data.shape[1]}")
            print(f"   - 样本数: {data.shape[0]}")
            print(f"   - 采样率: {sample_rate} Hz")
            print(f"   - 通道名称: {channel_names[:5]}{'...' if len(channel_names) > 5 else ''}")
            
            return data, channel_names, sample_rate
            
        except Exception as e:
            print(f"⚠️ 直接读取失败: {e}")
            print("💡 尝试使用替代方法...")
            
            # 替代方案：如果 .set 文件读取失败，尝试读取对应的 .fdt 文件
            fdt_file_path = set_file_path.replace('.set', '.fdt')
            if os.path.exists(fdt_file_path):
                print(f"📁 尝试读取对应的 .fdt 文件: {fdt_file_path}")
                # 读取 .fdt 文件（EEGLAB的二进制数据文件）
                try:
                    # 获取通道数和采样点数信息
                    nbchan = 16  # 默认值
                    pnts = 1000  # 默认值
                    
                    if 'nbchan' in eeg_data.dtype.names:
                        try:
                            nbchan_val = eeg_data['nbchan'][0, 0]
                            if hasattr(nbchan_val, 'item'):
                                nbchan = int(nbchan_val.item())
                            else:
                                nbchan = int(nbchan_val)
                        except Exception as e:
                            print(f"⚠️ 通道数提取失败: {e}, 使用默认值: {nbchan}")
                    
                    if 'pnts' in eeg_data.dtype.names:
                        try:
                            pnts_val = eeg_data['pnts'][0, 0]
                            if hasattr(pnts_val, 'item'):
                                pnts = int(pnts_val.item())
                            else:
                                pnts = int(pnts_val)
                        except Exception as e:
                            print(f"⚠️ 采样点数提取失败: {e}, 使用默认值: {pnts}")
                    print(f"📊 从EEG结构体获取信息: 通道数={nbchan}, 采样点数={pnts}")
                    
                    # 获取通道名称和采样率信息
                    channel_names = []
                    sample_rate = 1000  # 默认采样率
                    
                    # 尝试从EEG结构体提取通道名称
                    if 'chanlocs' in eeg_data.dtype.names:
                        try:
                            chanlocs = eeg_data['chanlocs'][0, 0]
                            if hasattr(chanlocs, 'dtype') and hasattr(chanlocs.dtype, 'names'):
                                if 'labels' in chanlocs.dtype.names:
                                    labels = chanlocs['labels'][0, 0]
                                    channel_names = [str(label[0]) for label in labels]
                                    print(f"✅ 从EEG结构体提取通道名称: {len(channel_names)} 个通道")
                        except Exception as e:
                            print(f"⚠️ 通道名称提取失败: {e}")
                    
                    # 如果没有通道信息，创建默认通道名
                    if not channel_names:
                        channel_names = [f'Ch{i+1:02d}' for i in range(nbchan)]
                        print(f"⚠️ 使用默认通道名称: {len(channel_names)} 个通道")
                    
                    # 尝试从EEG结构体提取采样率
                    if 'srate' in eeg_data.dtype.names:
                        try:
                            sample_rate = int(eeg_data['srate'][0, 0])
                            print(f"✅ 从EEG结构体提取采样率: {sample_rate} Hz")
                        except Exception as e:
                            print(f"⚠️ 采样率提取失败: {e}, 使用默认值: {sample_rate} Hz")
                    else:
                        print(f"⚠️ 使用默认采样率: {sample_rate} Hz")
                    
                    # 读取 .fdt 文件
                    with open(fdt_file_path, 'rb') as f:
                        # 读取二进制数据
                        raw_data = np.fromfile(f, dtype=np.float32)
                    
                    # 重塑数据为 (channels, samples) 然后转置为 (samples, channels)
                    if len(raw_data) == nbchan * pnts:
                        data = raw_data.reshape(nbchan, pnts).T  # 转置为 (samples, channels)
                        print(f"✅ 成功读取 .fdt 文件，数据形状: {data.shape}")
                    else:
                        # 如果数据长度不匹配，尝试自动推断
                        print(f"⚠️ 数据长度不匹配，尝试自动推断...")
                        if len(raw_data) % nbchan == 0:
                            inferred_pnts = len(raw_data) // nbchan
                            data = raw_data.reshape(nbchan, inferred_pnts).T
                            print(f"✅ 推断采样点数: {inferred_pnts}, 数据形状: {data.shape}")
                        else:
                            raise ValueError(f"无法推断数据维度: 总长度={len(raw_data)}, 通道数={nbchan}")
                    
                    return data, channel_names, sample_rate
                    
                except Exception as e:
                    print(f"❌ .fdt 文件读取失败: {e}")
                    raise ValueError(f"无法读取 .fdt 文件: {e}")
            else:
                raise ValueError(f"无法读取 .set 文件，且未找到对应的 .fdt 文件")
                
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        raise

def apply_orica(data, n_components=None, block_size=8, num_passes=1, use_rls_whitening=True):
    """
    应用ORICA算法
    
    Args:
        data: 输入数据 (samples, channels)
        n_components: 独立成分数量，如果为None则使用通道数
        block_size: 块大小
        num_passes: 训练遍数
        use_rls_whitening: 是否使用RLS白化
        
    Returns:
        sources: ORICA分离后的源信号 (components, samples)
        orica_model: 训练好的ORICA模型
    """
    try:
        n_samples, n_channels = data.shape
        
        # 设置独立成分数量
        if n_components is None:
            n_components = n_channels
        
        print(f"🔧 配置ORICA参数:")
        print(f"   - 独立成分数: {n_components}")
        print(f"   - 块大小: {block_size}")
        print(f"   - 训练遍数: {num_passes}")
        print(f"   - RLS白化: {'是' if use_rls_whitening else '否'}")
        
        # 创建ORICA模型
        orica_model = ORICAZ(
            n_components=n_components,
            block_size_ica=block_size,
            use_rls_whitening=use_rls_whitening,
            verbose=True
        )
        
        # 使用 fit_block_stream 进行训练
        print(f"🚀 开始ORICA训练...")
        sources = orica_model.fit_block_stream(data, block_size=block_size, num_passes=num_passes)
        
        print(f"✅ ORICA训练完成:")
        print(f"   - 源信号形状: {sources.shape}")
        print(f"   - 解混矩阵形状: {orica_model.get_W().shape}")
        print(f"   - 白化矩阵形状: {orica_model.get_whitening_matrix().shape}")
        
        return sources, orica_model
        
    except Exception as e:
        print(f"❌ ORICA训练失败: {e}")
        raise

def save_results(sources, channel_names, sample_rate, output_dir, base_filename):
    """
    保存结果到文件
    
    Args:
        sources: ORICA分离后的源信号 (components, samples)
        channel_names: 通道名称列表
        output_dir: 输出目录
        base_filename: 基础文件名（不含扩展名）
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存源信号为 .mat 文件
        mat_filename = f"{base_filename}_orica_sources_{timestamp}.mat"
        mat_path = os.path.join(output_dir, mat_filename)
        
        # 准备保存的数据
        save_data = {
            'sources': sources,
            'channel_names': channel_names,
            'sample_rate': sample_rate,
            'timestamp': timestamp,
            'algorithm': 'ORICA_REST_new',
            'source_count': sources.shape[0],
            'sample_count': sources.shape[1]
        }
        
        sio.savemat(mat_path, save_data)
        print(f"💾 源信号已保存到: {mat_path}")
        
        # 保存源信号为 .txt 文件（便于查看）
        txt_filename = f"{base_filename}_orica_sources_{timestamp}.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"ORICA算法结果 - {timestamp}\n")
            f.write(f"算法: ORICA_REST_new\n")
            f.write(f"源信号数量: {sources.shape[0]}\n")
            f.write(f"样本数量: {sources.shape[1]}\n")
            f.write(f"采样率: {sample_rate} Hz\n")
            f.write(f"通道名称: {', '.join(channel_names)}\n")
            f.write("\n" + "="*50 + "\n\n")
            
            # 保存源信号数据（完整样本）
            f.write(f"源信号数据 (完整{sources.shape[1]}个样本):\n")
            f.write("源\\样本\t" + "\t".join([f"{i+1:3d}" for i in range(sources.shape[1])]) + "\n")
            
            for i in range(sources.shape[0]):
                # 确保数据是数值类型，避免字符串格式化错误
                row_data = []
                for j in range(sources.shape[1]):
                    try:
                        value = float(sources[i, j])
                        row_data.append(f"{value:8.4f}")
                    except (ValueError, TypeError):
                        row_data.append(f"{str(sources[i, j]):>8}")
                f.write(f"源{i+1:2d}\t" + "\t".join(row_data) + "\n")
        
        print(f"💾 文本结果已保存到: {txt_path}")
        
        # 保存源信号为 .npy 文件（numpy格式）
        npy_filename = f"{base_filename}_orica_sources_{timestamp}.npy"
        npy_path = os.path.join(output_dir, npy_filename)
        
        np.save(npy_path, sources)
        print(f"💾 NumPy格式结果已保存到: {npy_path}")
        
        return mat_path, txt_path, npy_path
        
    except Exception as e:
        print(f"❌ 结果保存失败: {e}")
        raise

def save_whitening_results(X_whitened, sphere, channel_names, sample_rate, output_dir, base_filename):
    """
    保存白化结果（白化后的数据与白化矩阵/球化矩阵）
    
    Args:
        X_whitened: 白化后的数据，形状 (samples, channels)
        sphere: 白化矩阵（icasphere），形状 (channels, channels)
        channel_names: 通道名称列表
        sample_rate: 采样率
        output_dir: 输出目录
        base_filename: 基础文件名（不含扩展名）
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存为 .mat
        mat_filename = f"{base_filename}_whitening_{timestamp}.mat"
        mat_path = os.path.join(output_dir, mat_filename)
        sio.savemat(mat_path, {
            'whitened': X_whitened,
            'sphere': sphere,
            'channel_names': channel_names,
            'sample_rate': sample_rate,
            'timestamp': timestamp
        })
        print(f"💾 白化结果已保存到: {mat_path}")

        # 保存为 .txt 文件
        txt_w_filename = f"{base_filename}_whitened_{timestamp}.txt"
        txt_w_path = os.path.join(output_dir, txt_w_filename)
        
        # 保存白化后的数据为txt
        with open(txt_w_path, 'w', encoding='utf-8') as f:
            # 写入通道名称作为第一行
            f.write("通道名称: " + " ".join(channel_names) + "\n")
            f.write(f"采样率: {sample_rate}\n")
            f.write("白化后数据:\n")
            # 写入数据，每行一个通道（源1、源2...），每列一个样本
            # 注意：X_whitened的形状是 (samples, channels)，需要转置为 (channels, samples)
            X_whitened_T = X_whitened.T  # 转置为 (channels, samples)
            for i in range(X_whitened_T.shape[0]):
                row_str = f"源{i+1:2d}\t" + "\t".join([f"{val:.6f}" for val in X_whitened_T[i, :]])
                f.write(row_str + "\n")
        
        print(f"💾 白化数据(TXT)已保存到: {txt_w_path}")


        txt_s_filename = f"{base_filename}_sphere_{timestamp}.txt"
        txt_s_path = os.path.join(output_dir, txt_s_filename)
        
        # 保存白化矩阵为txt
        with open(txt_s_path, 'w') as f:
            f.write(f"白化矩阵 (形状: {sphere.shape})\n")
            f.write("通道名称: " + " ".join(channel_names) + "\n")
            f.write("矩阵数据:\n")
            # 写入矩阵，每行一个通道
            for i in range(sphere.shape[0]):
                row_str = " ".join([f"{val:.6f}" for val in sphere[i, :]])
                f.write(row_str + "\n")
        
        print(f"💾 白化矩阵(TXT)已保存到: {txt_s_path}")

        return mat_path, txt_w_path, txt_s_path
    except Exception as e:
        print(f"❌ 白化结果保存失败: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ORICA_REST_new 测试脚本')
    parser.add_argument('input_file', help='输入的 .set 文件路径')
    parser.add_argument('-o', '--output_dir', default='./ORICA_results', help='输出目录 (默认: ./ORICA_results)')
    parser.add_argument('-c', '--components', type=int, help='独立成分数量 (默认: 使用通道数)')
    parser.add_argument('-b', '--block_size', type=int, default=8, help='块大小 (默认: 8)')
    parser.add_argument('-p', '--passes', type=int, default=1, help='训练遍数 (默认: 1)')
    parser.add_argument('-r', '--rls', action='store_true', help='使用RLS白化')
    
    args = parser.parse_args()
    
    try:
        print("="*60)
        print("🚀 ORICA_REST_new 测试脚本")
        print("="*60)
        
        # 1. 读取输入文件
        print("\n📂 步骤1: 读取输入文件")
        data, channel_names, sample_rate = load_set_file(args.input_file)
        
        # 2. 应用ORICA算法
        print("\n🔬 步骤2: 应用ORICA算法")
        sources, orica_model = apply_orica(
            data=data,
            n_components=args.components,
            block_size=args.block_size,
            num_passes=args.passes,
            use_rls_whitening=args.rls
        )

        # 2.1 计算并保存白化结果（与 transform 相同逻辑）
        try:
            X = data.copy()
            if getattr(orica_model, 'mean', None) is not None and orica_model.mean.shape[0] == X.shape[1]:
                X = X - orica_model.mean
            sphere = orica_model.get_whitening_matrix()
            X_whitened = X @ sphere.T
            base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
            save_whitening_results(X_whitened, sphere, channel_names, sample_rate, args.output_dir, base_filename)
        except Exception as e:
            print(f"⚠️ 白化结果保存失败(非致命): {e}")
        
        # 3. 保存结果
        print("\n💾 步骤3: 保存结果")
        base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
        mat_path, txt_path, npy_path = save_results(
            sources, channel_names, sample_rate, args.output_dir, base_filename
        )
        
        print("\n" + "="*60)
        print("🎉 处理完成！")
        print("="*60)
        print(f"📁 输出目录: {args.output_dir}")
        print(f"📊 源信号文件: {os.path.basename(mat_path)}")
        print(f"📝 文本结果: {os.path.basename(txt_path)}")
        print(f"🔢 NumPy格式: {os.path.basename(npy_path)}")
        
        # 4. 显示结果摘要
        print(f"\n📈 结果摘要:")
        print(f"   - 输入数据: {data.shape[0]} 样本 × {data.shape[1]} 通道")
        print(f"   - 输出源信号: {sources.shape[0]} 成分 × {sources.shape[1]} 样本")
        print(f"   - 数据范围: [{np.min(sources):.4f}, {np.max(sources):.4f}]")
        print(f"   - 标准差: {np.std(sources):.4f}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        print("💡 请检查输入文件格式和参数设置")
        sys.exit(1)

if __name__ == "__main__":
    # 直接设置参数，无需交互
    try:
        # 预设参数
        input_file = r"D:\work\matlab_project\orica-master\orica-master\SIM_STAT_16ch_3min.set"
        output_dir = "./ORICA_results"
        n_components = 16  # 设置为16，与您的通道数匹配
        block_size = 8
        num_passes = 1
        use_rls = False
        
        print("="*60)
        print("🚀 ORICA_REST_new 测试脚本")
        print("="*60)
        
        print(f"📁 使用预设文件路径: {input_file}")
        print(f"📋 参数设置:")
        print(f"   - 输出目录: {output_dir}")
        print(f"   - 独立成分数: {n_components}")
        print(f"   - 块大小: {block_size}")
        print(f"   - 训练遍数: {num_passes}")
        print(f"   - RLS白化: {'是' if use_rls else '否'}")
        
        # 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"❌ 预设文件不存在: {input_file}")
            print("💡 请检查文件路径是否正确")
            sys.exit(1)
        
        # 1. 读取输入文件
        print("\n📂 步骤1: 读取输入文件")
        data, channel_names, sample_rate = load_set_file(input_file)
        
        # 2. 应用ORICA算法
        print("\n🔬 步骤2: 应用ORICA算法")
        sources, orica_model = apply_orica(
            data=data,
            n_components=n_components,
            block_size=block_size,
            num_passes=num_passes,
            use_rls_whitening=use_rls
        )

        # 2.1 计算并保存白化结果（与 transform 相同逻辑）
        try:
            X = data.copy()
            if getattr(orica_model, 'mean', None) is not None and orica_model.mean.shape[0] == X.shape[1]:
                X = X - orica_model.mean
            sphere = orica_model.get_whitening_matrix()
            #X_whitened = X @ sphere.T
            X_whitened = sphere @ X.T
            print("xxxxxxxx")
            print(X_whitened)
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
            save_whitening_results(X_whitened, sphere, channel_names, sample_rate, output_dir, base_filename)
        except Exception as e:
            print(f"⚠️ 白化结果保存失败(非致命): {e}")
        
        # 3. 保存结果
        print("\n💾 步骤3: 保存结果")
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        mat_path, txt_path, npy_path = save_results(
            sources, channel_names, sample_rate, output_dir, base_filename
        )
        
        print("\n" + "="*60)
        print("🎉 处理完成！")
        print("="*60)
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 源信号文件: {os.path.basename(mat_path)}")
        print(f"📝 文本结果: {os.path.basename(txt_path)}")
        print(f"🔢 NumPy格式: {os.path.basename(npy_path)}")
        
        # 4. 显示结果摘要
        print(f"\n📈 结果摘要:")
        print(f"   - 输入数据: {data.shape[0]} 样本 × {data.shape[1]} 通道")
        print(f"   - 输出源信号: {sources.shape[0]} 成分 × {sources.shape[1]} 样本")
        print(f"   - 数据范围: [{np.min(sources):.4f}, {np.max(sources):.4f}]")
        print(f"   - 标准差: {np.std(sources):.4f}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        print("💡 请检查输入文件格式和参数设置")
        sys.exit(1)
