# ORICA

Online EEG processing pipeline: simulates a real-time stream over LSL, runs IIR filtering → ASR → ORICA in a GUI, saves each stage to `data/output_data/`, and validates results offline.

## Repository layout

```
ORICA/
├── code/                    # Main scripts (LSL broadcast, GUI, receiver, processing)
├── data/
│   ├── Input_data/          # Test input data (see below)
│   └── output_data/         # Online experiment outputs + validation scripts
│       └── validation_code/
│           └── core/        # Offline validation scripts
├── .venv/                   # Local Python virtual environment (not tracked)
└── trash/                   # Archived legacy files (not tracked)
```

Path constants are defined in `code/paths.py`: `INPUT_DATA_ROOT`, `OUTPUT_DATA_ROOT`, etc.

---

## 1. Input_data: where test data comes from and how it is prepared

All test data used for LSL broadcasting lives under `data/Input_data/`. The preparation pipeline is:

```
set/  ──convert_set_to_npz.py──►  npz/
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            asr_cali/         npz_offline_notch/   (or broadcast set/ directly)
         filter_driving_      apply_notch60_to_npz.py
         sets_iir_1_50.py              │
                    │                 ▼
                    │         asr_cali_offline_notch/
                    │         filter_driving_sets_iir_1_50.py
                    └──────────────────────────────────────►  fed to aa_lsl_npz
```

### 1.1 `set/` — raw EEGLAB data

- Format: `.set` + matching `.fdt`
- Subdirectory layout matches subjects (e.g. `set/Shawn_shared/s28_resampled.set`)
- Can be broadcast directly by `aa_lsl_npz.py`; content is equivalent to npz, only the file format differs

### 1.2 `npz/` — converted from set

```bash
cd data/Input_data
python convert_set_to_npz.py              # fill in missing npz only
python convert_set_to_npz.py --check-only # compare only, no conversion
python convert_set_to_npz.py --force      # regenerate all
```

- Directory layout **mirrors** `set/` (`set/Shawn_shared/foo.set` → `npz/Shawn_shared/foo.npz`)
- npz keys: `data` (channels × samples, µV), `sampling_rate`, `channels`
- **npz is recommended for daily broadcasting** (faster to load)

### 1.3 `asr_cali/` — online ASR calibration data

Generated from `npz/` with 1–50 Hz IIR filtering:

```bash
cd data/Input_data/asr_cali
python filter_driving_sets_iir_1_50.py
```

Each source npz yields two calibration files:

| Subdirectory | Description |
|--------------|-------------|
| `asr_cali/<subdir>/full/` | Full recording after IIR 1–50 Hz |
| `asr_cali/<subdir>/2min/` | First 2 minutes only, after IIR 1–50 Hz |

Online ASR initialization typically uses the **`2min/`** version (see `asr_calib_npz` in `run_two_instances_Driving.py`).

### 1.4 `npz_offline_notch/` — offline 60 Hz notch

Offline zero-phase 60 Hz notch applied to `npz/`:

```bash
cd data/Input_data/npz_offline_notch
python apply_notch60_to_npz.py
```

Layout mirrors `npz/` (e.g. `npz_offline_notch/Shawn_shared/s01_resampled.npz`).

### 1.5 `asr_cali_offline_notch/` — ASR calibration after offline notch

Same workflow as `asr_cali/`, but the input source is `npz_offline_notch/`:

```bash
cd data/Input_data/asr_cali_offline_notch
python filter_driving_sets_iir_1_50.py
```

Also contains `full/` and `2min/` subdirectories. For the online pipeline with offline-notch data, use `run_two_instances_Driving_offline_notch.py`.

### Quick reference: broadcast vs calibration

| Purpose | Typical path |
|---------|--------------|
| LSL broadcast (simulated live EEG) | `npz/Shawn_shared/s28_resampled.npz` or the matching `.set` |
| Online ASR calibration | `asr_cali/Shawn_shared/2min/s28_resampled.npz` |
| Broadcast with offline notch | `npz_offline_notch/Shawn_shared/s01_resampled.npz` |
| ASR calibration with offline notch | `asr_cali_offline_notch/Shawn_shared/2min/s01_resampled.npz` |

---

## 2. LSL broadcast: `code/aa_lsl_npz.py`

Streams a local `.npz` or `.set` file over LSL as **`mybrain`**, paced at the real sampling rate, for consumption by `receiver.py`.

### Configuration

Edit `INPUT_FILE` at the top of the script:

```python
# Broadcast raw npz (most common)
INPUT_FILE = INPUT_DATA_ROOT / "npz/Shawn_shared/s28_resampled.npz"

# Or broadcast .set
# INPUT_FILE = INPUT_DATA_ROOT / "set/Shawn_shared/s28_resampled.set"
```

Other key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STREAM_NAME` | `"mybrain"` | Must match `stream_name` in `receiver.py` |
| `CHUNK_SIZE` | `50` | Samples pushed per chunk |
| `WAIT_TIME` | `2` | Seconds to wait after creating the stream for a receiver to connect |

### Run

```bash
cd code
python aa_lsl_npz.py
```

The script paces output to the sampling rate until the file is fully streamed. `.npz` and `.set` differ only in file format; broadcasting behavior is the same.

---

## 3. Online pipeline: `code/run_two_instances_Driving.py`

Launches one or more GUI instances that receive the same LSL stream, apply different parameters, and write to separate output directories.

### Parameters to edit

```python
# Subject ID — must match the filename in Input_data
subject_id = "s28_resampled"

# Output file prefix (first 3 chars of subject_id, e.g. "bs2")
file_tag = f"b{subject_id[:3]}"

# ASR calibration npz (2min version)
asr_calib_npz = INPUT_DATA_ROOT / "asr_cali/Shawn_shared/2min" / f"{subject_id}.npz"

# Experiment list (add/remove rows to run multiple configs in parallel)
raw_experiments = [
    {
        "method": "4",
        "save_dir": "output_data/xxSN_Driveasrpy20_2min_70",
        "asr_cutoff": "20",
        "icalabel_threshold": "0.7",
    },
]
```

These are passed to `main_gui.py` / `receiver.py` via environment variables:

| Environment variable | Description |
|---------------------|-------------|
| `EEG_GUI_INSTANCE` | GUI instance ID (window title and position) |
| `IIR_FILTER_METHOD` | Processing mode (Driving experiments use `"4"`: IIR + notch + ASR + ORICA) |
| `EEG_SAVE_DIR` | Output directory, relative to `data/output_data/` |
| `EEG_SAVE_FILE_TAG` | Save file prefix, e.g. `bs2` |
| `EEG_ASR_CALIB_NPZ` | Path to ASR calibration npz |
| `EEG_ASR_CUTOFF` | ASR cutoff (standard-deviation multiplier) |
| `EEG_ASR_BACKEND` | `"asrpy"` or `"meegkit"` |
| `EEG_ICALABEL_THRESHOLD` | ICLabel artifact rejection threshold |

### Run

```bash
cd code
python run_two_instances_Driving.py
```

This opens separate console windows running `main_gui.py`. For the offline-notch pipeline, use `run_two_instances_Driving_offline_notch.py` (calibration paths point to `asr_cali_offline_notch/`).

---

## 4. End-to-end workflow: Input_data → output_data

```
① Prepare Input_data (one-time, or when data is updated)
   set → npz → asr_cali (and optional notch branch)

② Start the receiver
   python run_two_instances_Driving.py
   Click "Start Stream" in the GUI
   → receiver listens for LSL stream "mybrain" (up to 60 s timeout)

③ Start the broadcaster (separate terminal)
   Set INPUT_FILE in aa_lsl_npz.py to match the subject
   python aa_lsl_npz.py
   → data is pushed at real-time rate

④ Wait for save
   After aa_lsl_npz finishes, the receiver detects end-of-stream,
   waits 5 s by default, then writes four files to output_data:
   {file_tag}eeg_raw1.npz
   {file_tag}eeg_iir1.npz
   {file_tag}eeg_asr1.npz
   {file_tag}eeg_orica1.npz

⑤ Offline validation (see next section)
```

**Recommended order**: run `run_two_instances_Driving.py` and click **Start Stream** first, then run `aa_lsl_npz.py` in another terminal. Start Stream does **not** launch the broadcaster automatically.

### How the receiver processes data (method=4)

1. Receive LSL chunks (raw)
2. Online IIR bandpass 1–50 Hz + 60 Hz notch
3. Initialize ASR from `EEG_ASR_CALIB_NPZ`, then apply online ASR artifact removal
4. ORICA + ICLabel artifact component rejection
5. Accumulate and save all four stages

Example output:

```
data/output_data/xxSN_Driveasrpy20_2min_70/
├── bs2eeg_raw1.npz
├── bs2eeg_iir1.npz
├── bs2eeg_asr1.npz
└── bs2eeg_orica1.npz
```

---

## 5. Validation: offline analysis of output_data

Scripts live in `data/output_data/validation_code/core/`. The two main scripts are used together:

### 5.1 Per-session analysis — `ica_source_energy_analysis_correctly.py`

For each subject / experiment profile, fits ICA on **full-length aligned data** and compares IC source mean-square (MS) energy across IIR / ASR / ORICA, plus percent reduction relative to IIR.

**Edit the config block at the top, then run:**

```python
PROFILES: List[str] = ["SN_Driveasrpy20_2min_70"]   # experiment folder under output_data
DATASET_IDS: List[str] = ["s28"]                     # subject ID (without the b prefix)

# Bad segments to exclude (seconds); ICA still uses full length, mask applied only for stats
EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0, 120), ...]

WINDOW_SEC = 10.0
```

```bash
cd data/output_data/validation_code/core
python ica_source_energy_analysis_correctly.py
```

**Outputs** (under `output_data/<profile>/ica_source_analysis/b<dataset>/.../`):

- Per-IC IIR/ASR/ORICA source MS and pct comparison plots
- `ic_source_ms_and_pct_exclude_bad_segments_per_ic.csv` (read by the aggregation script)
- Windowed time curves, power-weighted summaries, etc.

### 5.2 Cross-subject aggregation — `aggregate_exclude_bad_pct_multisubject_average.py`

Reads the per-subject CSVs from the step above, aggregates by ICLabel class within each subject, then computes cross-subject mean ± SD.

**Edit config:**

```python
EXPERIMENT: str = "SN_Driveasrpy20_2min_70"          # must match PROFILES
SEGMENT_PREFIX: str = "1segment_exclude_window_10_remove"  # match per-subject output subdir prefix
WITHIN_SUBJECT_IC_AGGREGATION = "median"              # nanmean or nanmedian for ICs of the same class
USE_ALL_CLASSES = True                                # whether x-axis includes all ICLabel classes
```

```bash
cd data/output_data/validation_code/core
python aggregate_exclude_bad_pct_multisubject_average.py
```

**Outputs** (under `output_data/<EXPERIMENT>/<aggregation_folder>/`):

- Cross-subject class-average pct plots (ASR vs IIR, ORICA vs IIR)
- Summary tables such as `cross_subject_mean_of_class_medians.csv`
- Per-subject subplots and `run_meta.txt` with run parameters

### Recommended order

```
run_two_instances_Driving.py  →  output_data/*.npz
        ↓
ica_source_energy_analysis_correctly.py  (configure EXCLUDE segments per subject)
        ↓
aggregate_exclude_bad_pct_multisubject_average.py  (aggregate all subjects)
```

---

## 6. Other scripts (brief)

| Script | Purpose |
|--------|---------|
| `run_two_instances.py` | Multi-instance launcher for other datasets (e.g. laparoscopic) |
| `run_two_instances_BNCI.py` | BNCI dataset |
| `run_two_instances_Driving_offline_notch.py` | Driving pipeline with offline-notch calibration |
| `main_gui.py` | GUI main window (usually launched by run_two_instances) |
| `receiver.py` | LSL receiver, online IIR/ASR/ORICA, data saving |
| `data/Input_data/list_npz_channels.py` | Inspect channel names in npz files |

---

## Environment

Use the project-local `.venv`:

```bash
# Windows
.venv\Scripts\activate
pip install numpy scipy mne pylsl PyQt5 matplotlib meegkit
```

Key dependencies: `numpy`, `scipy`, `mne`, `pylsl`, `PyQt5`, `matplotlib`, `meegkit` (plus ASR-related packages).
