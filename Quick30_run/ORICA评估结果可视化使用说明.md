# ORICA评估结果可视化使用说明

## 📋 功能概述

这个工具可以帮助你分析和可视化ORICA（在线递归独立成分分析）的评估结果，包括峭度分析、互信息分析和综合分离质量评估。

## 🚀 使用方法

### 1. 自动保存评估结果

当你运行ORICA处理时，评估结果会自动保存到 `./Results/` 目录下，文件格式为：
- **JSON格式**: `orica_evaluation_YYYYMMDD_HHMMSS.json` (完整数据)
- **CSV格式**: `orica_evaluation_YYYYMMDD_HHMMSS.csv` (表格数据，方便画图)

### 2. 运行可视化脚本

```bash
cd Quick30_run
python plot_orica_evaluation.py
```

## 📊 生成的图表类型

### 1. 时间演化图 (`orica_temporal_evolution_*.png`)
- **峭度均值变化趋势**: 显示峭度随时间的变化（越高越好）
- **互信息均值变化趋势**: 显示互信息随时间的变化（越低越好）
- **分离质量变化趋势**: 显示综合分离质量指标的变化（越高越好）

### 2. 组件分析图 (`orica_component_analysis_*.png`)
- **各IC峭度分布**: 柱状图显示每个独立成分的峭度值
- **峭度分布直方图**: 峭度值的分布情况
- **互信息矩阵热力图**: 显示各IC之间的相关性（越红越相关）
- **评估指标雷达图**: 综合评估指标的雷达图显示

### 3. 对比总结图 (`orica_comparison_summary_*.png`)
- **峭度均值对比**: 多个评估结果的峭度对比
- **互信息均值对比**: 多个评估结果的互信息对比
- **分离质量对比**: 多个评估结果的分离质量对比
- **综合评分雷达图**: 多个文件的综合评分对比

## 📁 输出文件说明

### JSON文件结构
```json
{
  "data_info": {
    "n_components": 25,
    "n_samples": 5000,
    "data_shape": [25, 5000],
    "timestamp": 1703123456.789,
    "datetime": "2023-12-21 15:30:56"
  },
  "kurtosis_analysis": {
    "mean": 2.456,
    "std": 1.234,
    "min": 0.123,
    "max": 5.678,
    "median": 2.345,
    "values": [2.1, 2.3, 2.5, ...]
  },
  "mutual_info_analysis": {
    "mean": 0.045,
    "std": 0.023,
    "min": 0.001,
    "max": 0.123,
    "matrix": [[0, 0.01, 0.02, ...], ...]
  },
  "evaluation_summary": {
    "separation_quality": 1.234,
    "kurtosis_score": 2.456,
    "independence_score": 0.956,
    "overall_rating": "good"
  }
}
```

### CSV文件结构
CSV文件采用键值对格式，便于导入到Excel或其他数据分析工具：
```csv
Category,Metric,Value
Data Info,n_components,25
Data Info,n_samples,5000
Kurtosis,mean,2.456
Kurtosis,std,1.234
Mutual Info,mean,0.045
Quality,separation_quality,1.234
IC_Kurtosis,IC_1,2.1
IC_Kurtosis,IC_2,2.3
...
```

## 🎯 评估指标解释

### 峭度 (Kurtosis)
- **含义**: 衡量信号的非高斯性
- **越高越好**: 峭度越高，信号越非高斯，分离效果越好
- **参考值**: 
  - > 3.0: 优秀
  - 2.0-3.0: 良好
  - 1.0-2.0: 一般
  - < 1.0: 较差

### 互信息 (Mutual Information)
- **含义**: 衡量各独立成分之间的相关性
- **越低越好**: 互信息越低，成分越独立
- **参考值**:
  - < 0.05: 优秀
  - 0.05-0.1: 良好
  - 0.1-0.2: 一般
  - > 0.2: 较差

### 分离质量指标
- **计算公式**: `峭度均值 / (1 + 互信息均值)`
- **参考值**:
  - > 2.0: excellent (优秀)
  - 1.0-2.0: good (良好)
  - 0.5-1.0: fair (一般)
  - < 0.5: poor (较差)

## 🔧 自定义使用

### 修改保存路径
在 `orica_processor.py` 中修改：
```python
filename = f"./Results/orica_evaluation_{timestamp}.json"  # 修改路径
```

### 修改评估参数
在 `evaluate_orica_sources` 方法中调整：
```python
def evaluate_orica_sources(self, sources, n_bins=10, save_to_file=True, filename=None):
    # n_bins: 互信息计算的离散化bin数
    # save_to_file: 是否保存到文件
    # filename: 自定义文件名
```

### 添加新的评估指标
在 `evaluate_orica_sources` 方法中添加新的计算：
```python
# 例如：添加偏度分析
skew_vals = skew(sources, axis=1)
skew_mean = np.mean(np.abs(skew_vals))

# 添加到保存数据中
save_data['skewness_analysis'] = {
    'mean': float(skew_mean),
    'values': skew_vals.tolist()
}
```

## 📈 画图建议

### 使用Python画图
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('orica_evaluation_20231221_153056.csv')

# 筛选峭度数据
kurtosis_data = df[df['Category'] == 'Kurtosis']
plt.figure(figsize=(10, 6))
plt.bar(kurtosis_data['Metric'], kurtosis_data['Value'])
plt.title('ORICA峭度分析')
plt.ylabel('峭度值')
plt.xticks(rotation=45)
plt.show()
```

### 使用Excel画图
1. 打开CSV文件
2. 选择数据 → 插入 → 图表
3. 选择合适的图表类型（柱状图、折线图、雷达图等）

## 🐛 常见问题

### Q: 没有生成评估文件？
A: 检查 `orica_processor.py` 中的 `evaluate_orica_sources` 是否被调用，以及 `save_to_file` 参数是否为 `True`

### Q: 图表显示中文乱码？
A: 确保系统安装了中文字体，或在代码中修改字体设置

### Q: 文件保存失败？
A: 检查 `./Results/` 目录是否存在，以及是否有写入权限

### Q: 如何比较不同时间的评估结果？
A: 运行 `plot_orica_evaluation.py` 脚本，它会自动加载所有评估文件并生成对比图表

## 📞 技术支持

如果遇到问题，请检查：
1. 文件路径是否正确
2. 依赖库是否安装完整（matplotlib, seaborn, pandas等）
3. 评估数据是否完整
4. 系统权限是否足够

## 🔄 更新日志

- **v1.0**: 基础评估结果保存和可视化功能
- 支持JSON和CSV两种格式
- 提供多种图表类型
- 自动生成总结报告


















































































