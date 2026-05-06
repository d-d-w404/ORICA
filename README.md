# ORICA Online EEG Pipeline

This project runs **real-time ORICA + ICLabel** on EEG delivered over **LSL**. The description below is the **main story**: what happens after you start from **`main_gui.py`**, **which files participate**, and **why** the pipeline is structured that way.

---

## 1. End-to-end flow (starting at `main_gui.py`)

### 1.1 GUI construction

- **`main_gui.py`** builds `EEGGUI`, which owns:
  - **`LSLStreamReceiver`** from **`receiver.py`** (all streaming, filtering, ASR, ORICA, saving).
  - **`LSLStreamVisualizer`** from **`viewer.py`** (matplotlib traces; reads the receiver’s rolling buffers).
  - Optional analytics: **`BandpowerAnalyzer`**, **`AttentionAnalyzer`**, **`RealTimeRegressor`**, **`ICAComponentWindow`** (ICA maps / sources when you open them), etc.

Nothing touches LSL yet; the receiver is created with default `stream_name` / `stream_type` from **`receiver.py`**.

### 1.2 User clicks **Start Stream**

`start_stream()` does roughly:

1. **`update_filter_params()`** — copies cutoff text fields into **`receiver.cutoff`** and the ASR checkbox into **`receiver.use_asr`**.
2. **`receiver.start()`** — opens the LSL stream and starts the background worker (see §2).
3. **`viewer.start()`** — begins refreshing the plot from the receiver’s buffers.
4. Starts **bandpower** and **attention** analyzer threads that periodically read the receiver.

So: **all heavy lifting is in `receiver.py`**; the GUI is orchestration + display.

### 1.3 Background thread: `_data_update_loop`

Inside **`LSLStreamReceiver.start()`** (`receiver.py`):

1. **`find_and_open_stream()`** — resolves the LSL inlet, reads **`StreamInfo`**, sets **`srate`**, builds **`ChannelManager`**, applies **`exclude`** lists to get **`chan_range`** / **`chan_labels`**, allocates **`buffer` / `raw_buffer` / `asr_buffer`**, and calls **`reinitialize_orica()`**.
2. A **daemon thread** runs **`_data_update_loop`**, which repeatedly calls **`pull_and_update_buffer()`** with a short sleep (`update_interval`).

Every loop iteration is one “tick” of your online pipeline.

---

## 2. Per-chunk pipeline inside `pull_and_update_buffer()`

### 2.1 Pull and subset channels

- **`pylsl`** **`pull_chunk`** returns variable-length blocks; they are turned into **`(n_channels, n_samples)`**.
- The code then keeps only **`chan_range`** rows (EEG subset after dropping triggers, ACC, etc.), so **`chunk`** is already “reduced montage” for most of the signal path.

### 2.2 Branching: `IIR_FILTER_METHOD` (environment)

- Read from **`os.environ['IIR_FILTER_METHOD']`** (default `'1'`). Typical experiment setups use **`'4'`** (see **`run_two_instances.py`** / **`run_two_instances_BNCI.py`**).
- **`EEG_ICALABEL_THRESHOLD`** can override **`receiver.icalabel_threshold`** for ICLabel.

**Example: method `'4'`** (common for your experiments):

1. **Raw snapshot** → **`self.raw_chunk`**
2. **Online IIR** — **`apply_online_iir_filter`** (coefficients / state live on the receiver; implementation uses **`filter_utils`** / SciPy-style IIR).
3. **ASR** (if enabled and calibrated):
   - First time: **`initialize_asr_from_npz_1`** loads **`EEG_ASR_CALIB_NPZ`** and **`EEG_ASR_CUTOFF`** from the environment (required in this branch).
   - **`meegkit`** or **`asrpy`** is selected via **`EEG_ASR_BACKEND`**.
   - Chunks are accumulated briefly so ASR sees enough samples, then trimmed back to the current frame width for sync with **`n_in`**.
4. **ORICA + ICLabel** — **`process_orica(chunk)`** (§3).
5. Rolling buffers (**`raw_buffer`**, **`asr_buffer`**, **`buffer`**) are updated so **`viewer.py`** and **`get_buffer_data()`** see aligned “before / after” traces.
6. Optional **multi-stage npz saving** (raw / IIR / ASR / ORICA lists) when **`save_processed_data`** is on — paths/tags from **`EEG_SAVE_DIR`**, **`EEG_SAVE_FILE_TAG`**, etc.

Other method values (`'1'`, `'41'`, …) are alternate IIR/ASR/ORICA orderings or visualization semantics; the **same ORICA entry point** is **`process_orica`**.

**Design idea:** one class (**`LSLStreamReceiver`**) owns stream I/O, filter state, ASR state, ORICA wrapper, and visualization buffers so the GUI stays thin and all real-time constraints stay in one place.

---

## 3. ORICA path: `receiver.py` → `orica_processor.py` → core ORICA

### 3.1 `process_orica` (`receiver.py`)

- Copies the incoming **`chunk`**.
- If **`self.orica`** exists, passes **`chunk[self.chan_range, :]`** into **`ORICAProcessor.update_buffer`** (currently each tick replaces the internal **`data_buffer`** with that chunk).
- When **`fit(...)`** runs, it uses **`chan_range`**, **`chan_labels`**, **`srate`**, and the **ICLabel threshold**.
- **`transform`** projects to sources, **zeros artifact ICs** (from ICLabel), and inverse-transforms; result is written back into **`cleaned_chunk[self.chan_range, :]`**.
- **`latest_sources`**, **`latest_ic_probs`**, **`latest_ic_labels`**, **`latest_eog_indices`** are updated for the GUI (**`ICAComponentWindow`**, debugging).

### 3.2 `ORICAProcessor` (`orica_processor.py`)

- Wraps **`ORICA_final_new`** from **`ORICA_final_no_print_quick30.py`** (initialize / incremental **`fit`** on `(samples × channels)` as required by that implementation).
- Builds an **MNE `Raw`-like** view for **`mne_icalabel.label_components`** inside **`use_icalabel_online`**.
- **ICLabel** marks components as artifacts (non-brain classes above **threshold**); indices are stored in **`eog_indices`** (name is historical; not only EOG).
- **`transform`** uses **`self.ica.transform` / `inverse_transform`** with those indices zeroed.

**Design idea:** separate **numeric ORICA** (your **`ORICA_final_*`**) from **labeling** (ICLabel) and from **LSL I/O** (`receiver`), so you can change the ICA backend or labeling without rewriting the inlet.

### 3.3 Re-initialization

- **`reinitialize_orica()`** is called after **`find_and_open_stream`** and when the user changes channels via **`ChannelSelectorDialog`** → **`set_channel_range_and_labels`** — new **`ORICAProcessor(n_components=len(chan_range), srate=...)`** so component count matches data.

---

## 4. File roles (quick map)

| File | Role |
|------|------|
| **`main_gui.py`** | PyQt app: start/stop, filter UI, ASR toggle, launches viewer & analyzers |
| **`receiver.py`** | LSL inlet, channel exclusion, IIR, ASR, **`process_orica`**, buffers, saving |
| **`filter_utils.py`** (and related) | Filter design / online IIR helpers used by the receiver |
| **`orica_processor.py`** | ORICA + ICLabel + transform API |
| **`ORICA_final_no_print_quick30.py`** | Concrete ORICA implementation (`ORICA_final_new`) |
| **`viewer.py`** | Matplotlib rolling plot from **`get_buffer_data`** |
| **`ica_component_window.py`**, **`topomap_visualizer.py`**, … | Optional visualization of ICs / topomaps |
| **`aa_lsl_npz.py`** | Replay `.npz`/`.set` as LSL (no amp) — must match **`stream_name`** in **`receiver.py`** |
| **`run_two_instances*.py`** | Spawn multiple **`main_gui.py`** with different **`IIR_FILTER_METHOD`**, ASR npz, save dirs |

---

## 5. Minimal setup reference

- **Python 3**; typical stack: `numpy`, `scipy`, `matplotlib`, `PyQt5`, `pylsl`, `mne`, `mne-icalabel`, `meegkit`, optional `asrpy`.
- **Live hardware:** set **`stream_name`** in **`LSLStreamReceiver.__init__`** to your outlet name.
- **Replay:** run **`aa_lsl_npz.py`** first, then **`main_gui.py`**; align **`STREAM_NAME`** with **`stream_name`**.
- **Batch experiments:** see **`EEG_GUI_INSTANCE`**, **`IIR_FILTER_METHOD`**, **`EEG_SAVE_DIR`**, **`EEG_SAVE_FILE_TAG`**, **`EEG_ASR_CALIB_NPZ`**, **`EEG_ASR_CUTOFF`**, **`EEG_ASR_BACKEND`**, **`EEG_ICALABEL_THRESHOLD`** in **`run_two_instances.py`** / **`receiver.py`**.

---

## 6. License and citations

Add a `LICENSE` and cite ORICA, MNE, ICLabel, and other tools if you redistribute this work.
