# validation_code

Experiment data still lives in the parent `output_data/` directory (profile subfolders, npz files, analysis outputs).

## Layout

| Directory | Description |
|-----------|-------------|
| **`core/`** | Four main scripts used day to day |
| **`trash/`** | Legacy / backup scripts (subdir layout preserved for possible reuse) |
| `_paths.py` / `_bootstrap.py` | Shared path helpers (`OUTPUT_DATA_ROOT` / `THRESHOLD_ROOT` = data root) |

## core scripts

| File | Purpose |
|------|---------|
| `artifact_removal_analysis_ica.py` | Fit ICA + ICLabel per sliding window |
| `artifact_removal_analysis_ica_same_save.py` | Single ICA fit on IIR data + windowed ICLabel + export `ic_sources` |
| `ica_source_energy_analysis_correctly.py` | ICA source energy / MS / exclude-segment analysis |
| `aggregate_exclude_bad_pct_multisubject_average.py` | Cross-subject summary of exclude-segment class percentages |

## Paths (resolved automatically in scripts)

```python
# Works from any depth under core/ or trash/
from _bootstrap import bootstrap_paths
THRESHOLD_ROOT, DATA_ROOT, CODE_DIR, REPO_ROOT, ARTIFACT_VERIFY_ROOT = bootstrap_paths()
```

- **npz / outputs**: `THRESHOLD_ROOT / "<profile>" / ...`
- **code modules**: `CODE_DIR` (receiver, orica_processor)

## Run examples

```bash
cd .../validation_code/core
python artifact_removal_analysis_ica_same_save.py --window-sec 30
python ica_source_energy_analysis_correctly.py
```
