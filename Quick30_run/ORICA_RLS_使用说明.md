# ORICA RLS白化和双曲正切函数使用说明

## 概述

本文档介绍如何在ORICA中实现RLS（Recursive Least Squares）白化和双曲正切非线性函数。RLS白化是一种在线自适应白化方法，更适合实时处理EEG数据。

## 主要特性

### 1. RLS白化
- **在线自适应**: 能够实时适应数据分布的变化
- **遗忘因子**: 控制历史数据的影响程度
- **数值稳定性**: 使用协方差矩阵逆的递归更新

### 2. 双曲正切非线性函数
- **替代高斯函数**: 提供不同的非线性特性
- **有界输出**: 输出范围在[-1, 1]之间
- **平滑导数**: 便于梯度计算

## 参数说明

### 初始化参数

```python
ORICA(
    n_components=4,           # 独立成分数量
    learning_rate=0.001,      # 学习率
    ortho_every=10,          # 正交化频率
    use_rls_whitening=True,   # 是否使用RLS白化
    forgetting_factor=0.995,  # RLS遗忘因子 (0 < λ < 1)
    nonlinearity='gaussian'   # 非线性函数类型 ('gaussian', 'tanh')
)
```

### 关键参数详解

#### `use_rls_whitening`
- `True`: 使用RLS白化（推荐用于在线处理）
- `False`: 使用传统批量白化

#### `forgetting_factor`
- 范围: 0 < λ < 1
- 推荐值: 0.99 - 0.999
- 较小值: 更快适应变化，但可能不稳定
- 较大值: 更稳定，但适应变化较慢

#### `nonlinearity`
- `'gaussian'`: 高斯非线性函数
- `'tanh'`: 双曲正切非线性函数

## 使用示例

### 基本使用

```python
from ORICA import ORICA
import numpy as np

# 创建ORICA实例
orica = ORICA(
    n_components=4,
    use_rls_whitening=True,
    forgetting_factor=0.995,
    nonlinearity='tanh'
)

# 初始化（需要初始数据）
X_init = your_initial_data  # shape: (n_channels, n_samples)
orica.initialize(X_init)

# 在线处理
for i in range(n_samples):
    x_t = your_data[:, i]  # 单个时间点数据
    result = orica.partial_fit(x_t)
    # result 是分离后的独立成分
```

### 动态切换非线性函数

```python
# 创建实例
orica = ORICA(n_components=4, nonlinearity='gaussian')

# 运行时切换非线性函数
orica.set_nonlinearity('tanh')

# 运行时调整遗忘因子
orica.set_forgetting_factor(0.999)
```

### 批量处理

```python
# 批量变换数据
X_new = your_new_data  # shape: (n_channels, n_samples)
Y = orica.transform(X_new)  # 分离后的独立成分
```

## RLS白化原理

### 算法步骤

1. **初始化**: 协方差矩阵逆 C = I
2. **递归更新**: 对每个新样本 x_t
   - 计算增益向量: k = C * x_t / (λ + x_t^T * C * x_t)
   - 更新协方差矩阵逆: C = (C - k * x_t^T * C) / λ
   - 更新白化矩阵: W = sqrt(C)

### 数学公式

```
k_t = C_{t-1} * x_t / (λ + x_t^T * C_{t-1} * x_t)
C_t = (C_{t-1} - k_t * x_t^T * C_{t-1}) / λ
W_t = sqrt(C_t)
```

## 双曲正切函数

### 函数定义

```python
def tanh_nonlinearity(y):
    g_y = np.tanh(y)
    g_prime = 1 - np.tanh(y)**2
    return g_y, g_prime
```

### 特性

- **有界性**: 输出范围 [-1, 1]
- **单调性**: 单调递增函数
- **对称性**: 奇函数，tanh(-x) = -tanh(x)

## 性能对比

### RLS vs 传统白化

| 特性 | RLS白化 | 传统白化 |
|------|---------|----------|
| 计算复杂度 | O(n²) | O(n³) |
| 内存需求 | 低 | 高 |
| 在线处理 | 支持 | 不支持 |
| 适应性 | 高 | 低 |
| 稳定性 | 中等 | 高 |

### 双曲正切 vs 高斯非线性

| 特性 | 双曲正切 | 高斯 |
|------|----------|------|
| 输出范围 | [-1, 1] | 无界 |
| 计算复杂度 | 低 | 中等 |
| 梯度消失 | 较少 | 较多 |
| 适用场景 | 有界信号 | 无界信号 |

## 最佳实践

### 1. 参数选择

```python
# 实时EEG处理推荐参数
orica = ORICA(
    n_components=your_channel_count,
    use_rls_whitening=True,
    forgetting_factor=0.995,  # 平衡稳定性和适应性
    nonlinearity='tanh',      # 适合EEG信号
    learning_rate=0.001,
    ortho_every=10
)
```

### 2. 初始化

```python
# 使用足够的数据进行初始化
init_samples = min(100, your_data.shape[1] // 10)
X_init = your_data[:, :init_samples]
orica.initialize(X_init)
```

### 3. 监控性能

```python
# 定期评估分离效果
if i % 100 == 0:
    kurtosis = orica.evaluate_separation(recent_results)
    print(f"峰度: {kurtosis}")
```

## 故障排除

### 常见问题

1. **数值不稳定**
   - 降低遗忘因子
   - 增加正则化

2. **收敛缓慢**
   - 增加学习率
   - 调整遗忘因子

3. **分离效果差**
   - 尝试不同的非线性函数
   - 检查数据预处理

### 调试技巧

```python
# 检查白化矩阵条件数
condition_number = np.linalg.cond(orica.get_whitening_matrix())
print(f"白化矩阵条件数: {condition_number}")

# 检查解混矩阵正交性
W = orica.get_W()
orthogonality_error = np.linalg.norm(W @ W.T - np.eye(W.shape[0]))
print(f"正交性误差: {orthogonality_error}")
```

## 测试脚本

运行测试脚本验证功能：

```bash
cd Quick30_run
python test_rls_orica.py
```

测试脚本将：
- 比较RLS和传统白化效果
- 比较双曲正切和高斯非线性函数
- 测试参数敏感性
- 生成可视化结果

## 总结

RLS白化和双曲正切函数的引入使ORICA更适合实时EEG信号处理：

1. **RLS白化**提供了在线自适应能力
2. **双曲正切函数**提供了有界的非线性特性
3. **参数可调**允许根据具体应用优化性能

这些改进使ORICA在实时脑机接口应用中具有更好的性能和稳定性。 