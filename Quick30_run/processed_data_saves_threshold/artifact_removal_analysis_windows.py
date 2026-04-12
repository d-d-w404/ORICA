"""
按固定时长窗口切分各阶段 NPZ，对每个窗口做与 pipeline 相同的单段指标（analyze_stage），
画出 4 行子图：RMS / Power / Peak / Variance。

本脚本**完全独立**，不 import artifact_removal_analysis_pipeline。

配置：
- _STAGE_DEFS：默认 npz 文件名；可注释某行改基准顺序（与 pipeline 相同思路）
- WINDOWS_ANALYSIS_TMIN_MIN / WINDOWS_ANALYSIS_TMAX_MIN：全局裁剪（分钟）；全为 None 时看下面秒级或全长
- USE_GLOBAL_TIME_WINDOW + GLOBAL_ANALYSIS_TMIN_S / GLOBAL_ANALYSIS_TMAX_S：秒级全局裁剪
- WINDOW_DURATION_S：每个分析子窗长度（秒）

依赖：numpy, scipy, matplotlib
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# 与 pipeline 中 analyze_stage 一致的 epoch 长度（仅用于内部 epoch_rms 列表）
EPOCH_DURATION_S = 2.0

# ========== 窗口长度（秒）；命令行 --window-sec 可覆盖 ==========
WINDOW_DURATION_S = 60.0

# ========== 全局裁剪 1：分钟（相对各 npz 时间轴 0）；None 表示该端不截 ==========
WINDOWS_ANALYSIS_TMIN_MIN: Optional[float] = None
WINDOWS_ANALYSIS_TMAX_MIN: Optional[float] = None

# ========== 全局裁剪 2：仅当上面两个都是 None 时生效 ==========
# False = 不裁剪（全长）；True = 用 GLOBAL_ANALYSIS_*（秒）
USE_GLOBAL_TIME_WINDOW = False
GLOBAL_ANALYSIS_TMIN_S: Optional[float] = None
GLOBAL_ANALYSIS_TMAX_S: Optional[float] = None

USE_ONLY_COMPLETE_WINDOWS = True

# 键 -> (图例标签, 默认文件名，相对本脚本目录)
_STAGE_DEFS: Dict[str, Tuple[str, str]] = {
    "raw": ("Raw", "b95eeg_raw1.npz"),
    "iir": ("IIR", "b95eeg_iir1.npz"),
    "asr": ("ASR", "b95eeg_asr1.npz"),
    "orica": ("ORICA", "b95eeg_orica1.npz"),
}

_DEFAULT_ORDER_KEYS: List[str] = [k for k in ("raw", "iir", "asr", "orica") if k in _STAGE_DEFS]
_DEFAULT_ORDER_STR = ",".join(_DEFAULT_ORDER_KEYS)


def get_global_crop_seconds() -> Tuple[Optional[float], Optional[float], str]:
    if WINDOWS_ANALYSIS_TMIN_MIN is not None or WINDOWS_ANALYSIS_TMAX_MIN is not None:
        tmin = (
            float(WINDOWS_ANALYSIS_TMIN_MIN) * 60.0
            if WINDOWS_ANALYSIS_TMIN_MIN is not None
            else None
        )
        tmax = (
            float(WINDOWS_ANALYSIS_TMAX_MIN) * 60.0
            if WINDOWS_ANALYSIS_TMAX_MIN is not None
            else None
        )
        return tmin, tmax, "minutes (this script)"
    if USE_GLOBAL_TIME_WINDOW:
        return GLOBAL_ANALYSIS_TMIN_S, GLOBAL_ANALYSIS_TMAX_S, "seconds (this script)"
    return None, None, "full"


def _parse_order(order_str: str) -> List[str]:
    allowed = frozenset(_STAGE_DEFS.keys())
    seen: set[str] = set()
    out: List[str] = []
    for part in order_str.split(","):
        k = part.strip().lower()
        if not k:
            continue
        if k not in allowed:
            raise ValueError(
                f"Unknown stage {part!r} in --order. Allowed: {', '.join(sorted(allowed))}"
            )
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    if not out:
        out = list(_DEFAULT_ORDER_KEYS)
    return out


def _build_stage_list(
    script_dir: Path,
    order_keys: List[str],
    explicit_paths: Dict[str, Path],
) -> List[Tuple[str, Path]]:
    stages: List[Tuple[str, Path]] = []
    for k in order_keys:
        label, default_name = _STAGE_DEFS[k]
        path = explicit_paths.get(k, script_dir / default_name)
        stages.append((label, path))
    return stages


def load_npz_file(npz_path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "data" not in data:
            print(f"[ERROR] {npz_path.name}: missing 'data'")
            return None
        eeg_data = np.asarray(data["data"], dtype=np.float64)
        if eeg_data.ndim != 2:
            print(f"[ERROR] {npz_path.name}: data must be 2D, got {eeg_data.shape}")
            return None
        if eeg_data.shape[1] <= 256 and eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T
        sr = data.get("sampling_rate", data.get("srate", 500))
        if isinstance(sr, np.ndarray):
            sr = float(sr.item())
        return {
            "data": eeg_data,
            "sampling_rate": float(sr),
            "file_name": npz_path.name,
            "file_path": npz_path,
        }
    except Exception as e:
        print(f"[ERROR] load {npz_path.name}: {e}")
        return None


def apply_time_window(
    info: Dict[str, Any],
    tmin_s: Optional[float],
    tmax_s: Optional[float],
) -> Optional[Dict[str, Any]]:
    if tmin_s is None and tmax_s is None:
        return info

    data = info["data"]
    fs = float(info["sampling_rate"])
    n = data.shape[1]
    duration = n / fs

    t0 = 0.0 if tmin_s is None else float(tmin_s)
    t1 = duration if tmax_s is None else float(tmax_s)
    t0 = max(0.0, min(t0, duration))
    t1 = max(0.0, min(t1, duration))

    if t0 >= t1:
        print(
            f"[ERROR] empty time window (duration={duration:.4f}s, tmin={t0}, tmax={t1})"
        )
        return None

    i0 = max(0, int(round(t0 * fs)))
    i1 = min(n, int(round(t1 * fs)))
    if i1 <= i0:
        print(f"[ERROR] empty sample range i0={i0}, i1={i1}, n={n}")
        return None

    out = dict(info)
    out["data"] = np.ascontiguousarray(data[:, i0:i1])
    out["time_window_s"] = (t0, t1)
    out["time_window_samples"] = (i0, i1)
    return out


def calculate_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def _welch_band_power(x: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    nperseg = min(2048, max(256, len(x)))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def calculate_power_total(x: np.ndarray, fs: float) -> float:
    return float(np.mean(x**2))


def calculate_snr(x: np.ndarray, fs: float) -> float:
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(2048, max(256, len(x))))
    signal_mask = (freqs >= 1) & (freqs <= 40)
    signal_power = np.trapz(psd[signal_mask], freqs[signal_mask])
    line_mask = (freqs >= 48) & (freqs <= 52)
    line_power = np.trapz(psd[line_mask], freqs[line_mask])
    nyq = fs / 2
    hf = 0.0
    if nyq > 100:
        hf_mask = (freqs >= 100) & (freqs <= min(200, nyq * 0.9))
        if np.any(hf_mask):
            hf = np.trapz(psd[hf_mask], freqs[hf_mask])
    noise_power = line_power + hf
    if noise_power < signal_power * 0.01:
        noise_power = signal_power * 0.01
    if noise_power <= 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def analyze_stage(file_info: Dict[str, Any]) -> Dict[str, Any]:
    data = file_info["data"]
    fs = file_info["sampling_rate"]
    name = file_info["file_name"]
    n_ch = data.shape[0]

    rms_list: List[float] = []
    p_tot: List[float] = []
    p_low: List[float] = []
    p_high: List[float] = []
    p_line: List[float] = []
    snr_list: List[float] = []
    peak_list: List[float] = []
    var_list: List[float] = []

    for ch in range(n_ch):
        x = data[ch, :]
        rms_list.append(calculate_rms(x))
        p_tot.append(calculate_power_total(x, fs))
        p_low.append(_welch_band_power(x, fs, (0.5, 4)))
        p_high.append(_welch_band_power(x, fs, (20, 100)))
        p_line.append(_welch_band_power(x, fs, (48, 52)))
        s = calculate_snr(x, fs)
        if np.isfinite(s):
            snr_list.append(s)
        peak_list.append(float(np.max(np.abs(x))))
        var_list.append(float(np.var(x)))

    epoch_samples = int(EPOCH_DURATION_S * fs)
    n_epochs = data.shape[1] // epoch_samples
    epoch_rms_list: List[float] = []
    for i in range(n_epochs):
        s0, s1 = i * epoch_samples, (i + 1) * epoch_samples
        block = data[:, s0:s1]
        epoch_rms_list.append(
            float(np.mean([calculate_rms(block[ch, :]) for ch in range(n_ch)]))
        )

    out: Dict[str, Any] = {
        "file_name": name,
        "stage": file_info.get("stage_label", name),
        "n_channels": n_ch,
        "n_samples": data.shape[1],
        "duration_s": data.shape[1] / fs,
        "fs": fs,
        "rms_mean": float(np.mean(rms_list)),
        "rms_std": float(np.std(rms_list)),
        "power_mean": float(np.mean(p_tot)),
        "low_freq_power_mean": float(np.mean(p_low)),
        "high_freq_power_mean": float(np.mean(p_high)),
        "line_noise_power_mean": float(np.mean(p_line)),
        "peak_mean": float(np.mean(peak_list)),
        "peak_std": float(np.std(peak_list)),
        "variance_mean": float(np.mean(var_list)),
        "variance_std": float(np.std(var_list)),
        "epoch_rms_list": epoch_rms_list,
        "n_epochs": n_epochs,
    }
    if snr_list:
        out["snr_mean"] = float(np.mean(snr_list))
        out["snr_std"] = float(np.std(snr_list))
    else:
        out["snr_mean"] = None
        out["snr_std"] = None
    if file_info.get("time_window_s") is not None:
        out["time_window_s"] = file_info["time_window_s"]
    return out


def _min_length_across(infos: List[Dict[str, Any]]) -> int:
    return min(int(x["data"].shape[1]) for x in infos)


def _analyze_windows_for_stage(
    info: Dict[str, Any],
    label: str,
    win_samples: int,
    n_win: int,
    t0_global_s: float,
    fs: float,
) -> List[Dict[str, Any]]:
    data = info["data"]
    out: List[Dict[str, Any]] = []
    for w in range(n_win):
        s0 = w * win_samples
        s1 = (w + 1) * win_samples
        block = data[:, s0:s1]
        tw_start = t0_global_s + s0 / fs
        tw_end = t0_global_s + s1 / fs
        sub: Dict[str, Any] = {
            "data": block,
            "sampling_rate": fs,
            "file_name": info["file_name"],
            "file_path": info.get("file_path"),
            "stage_label": label,
            "time_window_s": (tw_start, tw_end),
        }
        out.append(analyze_stage(sub))
    return out


def plot_windowed_metrics(
    stage_labels: List[str],
    per_stage_windows: List[List[Dict[str, Any]]],
    baseline_stem: str,
    baseline_stage: str,
    window_sec: float,
    output_dir: Path,
    t0_global_s: float,
) -> Path:
    n_stages = len(stage_labels)
    n_win = len(per_stage_windows[0]) if per_stage_windows else 0
    if n_win == 0:
        raise ValueError("没有可用窗口")

    metrics = [
        ("rms_mean", "RMS (mean over channels)"),
        ("power_mean", "Power mean(x^2)"),
        ("peak_mean", "Peak mean(|x|_max)"),
        ("variance_mean", "Variance"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(max(10, n_win * 0.35), 14), sharex=False)
    fig.suptitle(
        f"Windowed overall metrics\n"
        f"Baseline: {baseline_stage} / {baseline_stem}.npz | "
        f"window = {window_sec:.3g}s | n_windows = {n_win}",
        fontsize=14,
        fontweight="bold",
    )

    bar_colors = plt.cm.Set3(np.linspace(0, 1, max(n_stages - 1, 1)))
    x = np.arange(n_win)
    width = 0.8 / max(n_stages, 1)

    def xtick_labels() -> List[str]:
        labs = []
        for w in range(n_win):
            c0 = t0_global_s + (w + 0.5) * window_sec
            labs.append(f"W{w}\n{c0:.1f}s")
        return labs

    for ax, (key, title) in zip(axes, metrics):
        for si, lab in enumerate(stage_labels):
            vals = [per_stage_windows[si][w][key] for w in range(n_win)]
            offset = (si - n_stages / 2 + 0.5) * width
            c = "#95a5a6" if si == 0 else bar_colors[min(si - 1, len(bar_colors) - 1)]
            ax.bar(
                x + offset,
                vals,
                width,
                label=lab,
                alpha=0.85 if si > 0 else 0.65,
                color=c,
                edgecolor="black",
            )
        ax.set_ylabel("Value (log)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", ncol=min(n_stages, 4))

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xtick_labels(), fontsize=7)
    axes[-1].set_xlabel("Window (center time in recording)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_dir.mkdir(parents=True, exist_ok=True)
    wtag = str(window_sec).replace(".", "p")
    plot_path = output_dir / f"pipeline_windows__{baseline_stem}__win{wtag}s.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="按时间窗口切分各阶段 NPZ，画 RMS/Power/Peak/Variance（独立脚本）"
    )
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--iir", type=Path, default=None)
    parser.add_argument("--asr", type=Path, default=None)
    parser.add_argument("--orica", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--order", type=str, default=_DEFAULT_ORDER_STR)
    parser.add_argument(
        "--window-sec",
        type=float,
        default=None,
        help=f"子窗长度（秒），默认 WINDOW_DURATION_S={WINDOW_DURATION_S}",
    )
    args = parser.parse_args()

    window_sec = float(args.window_sec) if args.window_sec is not None else float(WINDOW_DURATION_S)
    if window_sec <= 0:
        print("[ERROR] window-sec must be > 0")
        return

    explicit_paths: Dict[str, Path] = {}
    if args.raw:
        explicit_paths["raw"] = args.raw
    if args.iir:
        explicit_paths["iir"] = args.iir
    if args.asr:
        explicit_paths["asr"] = args.asr
    if args.orica:
        explicit_paths["orica"] = args.orica

    try:
        order_keys = _parse_order(args.order)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    stages = _build_stage_list(script_dir, order_keys, explicit_paths)

    infos: List[Dict[str, Any]] = []
    stage_labels: List[str] = []
    for label, path in stages:
        path = path.resolve()
        if not path.is_file():
            print(f"[WARN] skip (missing file): {label} -> {path}")
            continue
        info = load_npz_file(path)
        if info is None:
            continue
        tmin, tmax, _crop_mode = get_global_crop_seconds()
        info = apply_time_window(info, tmin, tmax)
        if info is None:
            print(f"[WARN] skip {label}: invalid global time window")
            continue
        infos.append(info)
        stage_labels.append(label)

    if not infos:
        print("没有可分析的文件。")
        return

    _tm, _tx, crop_mode = get_global_crop_seconds()
    print(f"Global crop: {crop_mode}")
    if WINDOWS_ANALYSIS_TMIN_MIN is not None or WINDOWS_ANALYSIS_TMAX_MIN is not None:
        print(
            f"  WINDOWS_ANALYSIS_TMIN_MIN={WINDOWS_ANALYSIS_TMIN_MIN}, "
            f"WINDOWS_ANALYSIS_TMAX_MIN={WINDOWS_ANALYSIS_TMAX_MIN} (minutes)"
        )
    elif USE_GLOBAL_TIME_WINDOW:
        print(
            f"  GLOBAL_ANALYSIS_TMIN_S={GLOBAL_ANALYSIS_TMIN_S}, "
            f"GLOBAL_ANALYSIS_TMAX_S={GLOBAL_ANALYSIS_TMAX_S} (seconds)"
        )

    n_min = _min_length_across(infos)
    fs = float(infos[0]["sampling_rate"])
    if any(float(x["sampling_rate"]) != fs for x in infos):
        print("[WARN] sampling_rate differs across files; using first file's fs")

    win_samples = max(int(round(window_sec * fs)), 1)
    if USE_ONLY_COMPLETE_WINDOWS:
        n_win = n_min // win_samples
    else:
        n_win = int(np.ceil(n_min / win_samples))

    if n_win < 1:
        print(
            f"[ERROR] no complete window: n_samples_min={n_min}, win_samples={win_samples}"
        )
        return

    tw = infos[0].get("time_window_s")
    t0_global_s = float(tw[0]) if tw else 0.0

    per_stage_windows: List[List[Dict[str, Any]]] = []
    for info, lab in zip(infos, stage_labels):
        data = info["data"]
        if data.shape[1] > n_win * win_samples:
            data = data[:, : n_win * win_samples]
        info_adj = dict(info)
        info_adj["data"] = data
        per_stage_windows.append(
            _analyze_windows_for_stage(
                info_adj, lab, win_samples, n_win, t0_global_s, fs
            )
        )

    baseline_stem = Path(str(infos[0]["file_name"])).stem
    baseline_stage = stage_labels[0]

    out_dir = (args.out_dir or (script_dir / "analysis_results_window")).resolve()
    plot_path = plot_windowed_metrics(
        stage_labels,
        per_stage_windows,
        baseline_stem,
        baseline_stage,
        window_sec,
        out_dir,
        t0_global_s,
    )
    print(f"[OK] Windowed figure saved: {plot_path}")
    print(f"Baseline (first in --order): {baseline_stage} | windows: {n_win} x {window_sec}s")


if __name__ == "__main__":
    main()
