# validation_code

实验数据仍在上一级 `output_data/`（profile 子目录、npz、分析输出）。

## 目录

| 目录 | 说明 |
|------|------|
| **`core/`** | 日常使用的 4 个主脚本 |
| **`trash/`** | 历史/备用脚本（保留子目录结构，以后可能再用） |
| `_paths.py` / `_bootstrap.py` | 共用路径（`OUTPUT_DATA_ROOT` / `THRESHOLD_ROOT` = 数据根） |

## core 脚本

| 文件 | 用途 |
|------|------|
| `artifact_removal_analysis_ica.py` | 每窗各自 fit ICA + ICLabel |
| `artifact_removal_analysis_ica_same_save.py` | IIR 上 fit 一次 ICA + 分窗 ICLabel + 导出 ic_sources |
| `ica_source_energy_analysis_correctly.py` | ICA 源能量 / MS / exclude 段分析 |
| `aggregate_exclude_bad_pct_multisubject_average.py` | 跨被试 exclude 类占比汇总 |

## 路径（脚本内已自动解析）

```python
# 任意 core/ 或 trash/ 深度均可
from _bootstrap import bootstrap_paths
THRESHOLD_ROOT, DATA_ROOT, CODE_DIR, REPO_ROOT, ARTIFACT_VERIFY_ROOT = bootstrap_paths()
```

- **npz / 输出**：`THRESHOLD_ROOT / "<profile>" / ...`
- **code 模块**：`CODE_DIR`（receiver、orica_processor）

## 运行示例

```bash
cd .../validation_code/core
python artifact_removal_analysis_ica_same_save.py --window-sec 30
python ica_source_energy_analysis_correctly.py
```
