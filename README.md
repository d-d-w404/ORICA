# ORICA Online EEG Pipeline

This repository implements **real-time EEG** acquisition over **LSL (Lab Streaming Layer)** in **`Quick30_run/`**, with **IIR filtering**, optional **ASR (Artifact Subspace Reconstruction)**, **online ORICA**, and **ICLabel**-based artifact component detection. A **PyQt5** GUI and multi-stage saving are included. The pipeline can run on a **live amplifier LSL stream** or on an **LSL stream replayed from disk**.

---

## Component overview

| Module | Role |
|--------|------|
| `main_gui.py` | Main UI: start/stop stream, filter bands, ASR toggle, channel selection, bandpower and attention tabs |
| `receiver.py` | `LSLStreamReceiver`: LSL inlet, channel exclusion, IIR/ASR/ORICA pipeline, buffers, saving |
| `orica_processor.py` | ORICA + ICLabel wrapper (uses `ORICA_final_no_print_quick30`, etc.) |
| `aa_lsl_npz.py` | Streams `.npz` / `.set` as an LSL outlet (hardware-free replay) |
| `run_two_instances.py` / `run_two_instances_BNCI.py` | Launch several `main_gui.py` processes on the same source; parameters and save dirs via env vars |

---

## Requirements

- **Python 3** (use the same interpreter as your scientific stack)
- Typical dependencies (from imports across the project; pin versions as needed):  
  `numpy`, `scipy`, `pandas`, `matplotlib`, `PyQt5`, `pylsl`, `mne`, `mne-icalabel`, `meegkit`, and optionally **`asrpy`** when `EEG_ASR_BACKEND=asrpy`

There is no checked-in `requirements.txt`; install with `pip` in your target environment, or derive a full list from the imports at the top of `Quick30_run/main_gui.py` and `receiver.py`.

---

## Quick start (live LSL hardware)

1. Confirm your amp or recording software publishes an LSL stream and note the **stream name** (`StreamInfo.name()`).
2. In `Quick30_run/receiver.py`, set **`stream_name`** in `LSLStreamReceiver.__init__` to match the publisher (examples in code include `'mybrain'` or device-specific names such as `CGX Quick-32r ...`).
3. From `Quick30_run`, start the GUI:

   ```bash
   cd Quick30_run
   python main_gui.py
   ```

4. Set filter cutoffs, enable ASR if needed, then click **Start Stream**.

---

## File replay as LSL (no hardware)

1. Prepare an `.npz` with `data` or `cleaned_data` shaped **`(n_channels, n_samples)`**, plus one of `sampling_rate` / `srate` / `fs`; `channels` is optional.
2. In `aa_lsl_npz.py`, set **`INPUT_FILE`** and make **`STREAM_NAME`** match **`stream_name`** in `receiver.py` (e.g. both `mybrain`).
3. Run `python aa_lsl_npz.py`, then `python main_gui.py`.

**Online calibration npz → replay-ready format**: use `Quick30_run/Record_data/calibration_npz_to_lsl_input.py` to turn `calibration_npz` files (`calibration_data`) into `*_for_lsl.npz` with `data`, `sampling_rate`, and `channels` for `aa_lsl_npz.py`.

---

## Multi-instance experiments (environment variables)

`run_two_instances.py` (laparoscopic subject IDs) and `run_two_instances_BNCI.py` (BNCI-style runs) spawn multiple `main_gui.py` processes via **`subprocess`** and inject variables read in `receiver.py`:

| Variable | Purpose |
|----------|---------|
| `EEG_GUI_INSTANCE` | Instance id for window title and staggered placement |
| `IIR_FILTER_METHOD` | IIR/processing branch (e.g. `'4'`; see branches in `receiver.py`) |
| `EEG_SAVE_DIR` | Relative or absolute directory for saved processed data |
| `EEG_SAVE_FILE_TAG` | Filename tag (e.g. `b11`) |
| `EEG_ASR_CALIB_NPZ` | Path to ASR calibration `.npz` |
| `EEG_ASR_CUTOFF` | ASR cutoff parameter |
| `EEG_ASR_BACKEND` | `meegkit` (default) or `asrpy` |
| `EEG_ICALABEL_THRESHOLD` | ICLabel threshold (e.g. `0.7`) |

Edit `subject_id`, `asr_calib_npz`, and the `raw_experiments` list in the chosen `run_two_instances*.py` to compare ASR/ICLabel settings in parallel.

---

## ASR calibration

- Example capture script: `Quick30_run/artifact_removal_verify/set_npz/npz_data/online_cali/collect_lsl_calibration_1_45hz.py` (records a segment from LSL into `calibration_npz`).
- Keep the **`exclude`** channel list in `receiver.py` consistent with acquisition and `ChannelManager` logic. **Full stream channel count** (`info.channel_count()`) and **`chan_range`** (EEG subset after exclusions) must stay aligned: visualization buffers follow the full LSL layout, while ASR/ORICA often run on the reduced set—see the current `receiver.py` slice/write-back pattern.

---

## Offline utilities

- **BNCI `.mat` → `.npz`**: `artifact_removal_verify/set_npz/npz_data/BNCI/mat/convert_mat_to_npz.py`
- **Post-hoc analysis / multi-subject aggregation**: scripts under `Quick30_run/processed_data_saves_threshold/` (e.g. `aggregate_exclude_bad_pct_multisubject_average.py`, `artifact_removal_analysis_windows_check1.py`)

---

## Troubleshooting

- **No LSL stream**: Ensure `resolve_byprop('name', ...)` matches the publisher exactly; you can test with `aa_lsl_npz.py` and a fixed `STREAM_NAME`.
- **ASR / shape mismatches**: Calibration npz channel count must match online **`chan_range`**; ASR should be fitted and applied on the excluded-channel subset and written back into the full `chunk` (see `receiver.py`).
- **ICLabel and channel names**: Standard 10–20 names work best with default montages; exotic labels may affect montage matching and label quality.

---

## License and citations

If you distribute this work, add a `LICENSE` file and cite ORICA, MNE, ICLabel, and other third-party tools as appropriate.
