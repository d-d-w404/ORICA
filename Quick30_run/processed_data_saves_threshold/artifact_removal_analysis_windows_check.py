"""
交互查看 Raw / IIR / ASR / ORICA 四阶段数据（与 artifact_removal_analysis_windows.py 相同的 npz、裁剪规则）。

- 同一坐标系叠加最多四条曲线（颜色：Raw 灰虚线、IIR 红虚线、ASR 绿实线、ORICA 蓝实线）。
- 勾选框：只显示选中的阶段。
- 滑块：通道索引；时间起点（在可见窗宽内沿时间轴平移查看整段波形）。

运行（在脚本所在目录或任意目录）:
  python artifact_removal_analysis_windows_check.py
  python artifact_removal_analysis_windows_check.py --view-sec 8

依赖：numpy, matplotlib；会 import 同目录 artifact_removal_analysis_windows（仅复用加载逻辑）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, Slider

# 与 windows 脚本一致：从同目录导入
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import artifact_removal_analysis_windows as aw  # noqa: E402

# 固定阶段顺序与样式（显示名与 windows 的 stage 标签一致）
_STAGE_STYLE: List[Tuple[str, str, str, str]] = [
    ("Raw", "gray", "--", "raw"),
    ("IIR", "red", "--", "iir"),
    ("ASR", "green", "-", "asr"),
    ("ORICA", "blue", "-", "orica"),
]

# 默认屏幕上一次显示多长的波形（秒）
VIEW_DURATION_S = 10.0


def load_stages_data(
    script_dir: Path,
    order_keys: Optional[List[str]] = None,
    explicit_paths: Optional[Dict[str, Path]] = None,
) -> Tuple[float, float, Dict[str, np.ndarray], List[str]]:
    """
    按 windows 的规则加载并裁剪；对齐到最短长度。
    返回 fs, t0_global_s, {显示名: (n_ch, n_samp)}, 实际加载的显示名列表（顺序固定 Raw→ORICA 中存在的）。
    """
    explicit_paths = explicit_paths or {}
    if order_keys is None:
        order_keys = ["raw", "iir", "asr", "orica"]
        order_keys = [k for k in order_keys if k in aw._STAGE_DEFS]

    stages = aw._build_stage_list(script_dir, order_keys, explicit_paths)
    loaded: List[Tuple[str, np.ndarray, float, str]] = []
    t0_global = 0.0
    got_t0 = False

    for label, path in stages:
        path = path.resolve()
        if not path.is_file():
            print(f"[WARN] missing: {label} -> {path}")
            continue
        info = aw.load_npz_file(path)
        if info is None:
            continue
        tmin, tmax, _ = aw.get_global_crop_seconds()
        info = aw.apply_time_window(info, tmin, tmax)
        if info is None:
            print(f"[WARN] skip {label}: invalid crop")
            continue
        fs = float(info["sampling_rate"])
        if not got_t0 and info.get("time_window_s") is not None:
            t0_global = float(info["time_window_s"][0])
            got_t0 = True
        loaded.append((label, info["data"], fs, info["file_name"]))

    if not loaded:
        raise RuntimeError("No npz loaded.")

    fs0 = loaded[0][2]
    if any(abs(x[2] - fs0) > 1e-6 for x in loaded):
        print("[WARN] sampling_rate differs; using first file's fs for time axis")

    n_min = min(x[1].shape[1] for x in loaded)

    data_by_label: Dict[str, np.ndarray] = {}
    labels_order: List[str] = []
    for label, data, _fs, fname in loaded:
        data_by_label[label] = np.ascontiguousarray(data[:, :n_min])
        labels_order.append(label)
        print(f"  loaded {label}: {fname} shape={data_by_label[label].shape}")

    return fs0, t0_global, data_by_label, labels_order


def run_interactive(
    view_duration_s: float,
    script_dir: Path,
    order_keys: Optional[List[str]],
    explicit_paths: Dict[str, Path],
) -> None:
    fs, t0, data_by_label, loaded_labels = load_stages_data(
        script_dir, order_keys=order_keys, explicit_paths=explicit_paths
    )

    # 固定四条在图例中的顺序，仅对已加载的显示勾选与曲线
    present: List[Tuple[str, str, str, str]] = []
    for disp, color, ls, key in _STAGE_STYLE:
        if disp in data_by_label:
            present.append((disp, color, ls, key))

    if not present:
        raise RuntimeError("No stage data to plot")

    n_ch = next(iter(data_by_label.values())).shape[0]
    n_samp = next(iter(data_by_label.values())).shape[1]
    total_dur = n_samp / fs
    view_w = min(float(view_duration_s), total_dur)
    if view_w <= 0:
        raise ValueError("view duration must be > 0")

    t_start_max = max(0.0, total_dur - view_w)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.12, bottom=0.28, right=0.98, top=0.92)

    lines: Dict[str, plt.Line2D] = {}
    for disp, color, ls, _key in present:
        (ln,) = ax.plot(
            [],
            [],
            color=color,
            linestyle=ls,
            linewidth=0.9,
            label=disp,
            alpha=0.9,
        )
        lines[disp] = ln

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Stages overlay | {n_ch} ch, {total_dur:.1f}s @ {fs:.0f} Hz | crop t0={t0:.3f}s"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    def slice_for_time_start(t_rel_start: float) -> Tuple[np.ndarray, slice]:
        """t_rel_start: 相对裁剪起点 0..total_dur-view_w"""
        t_rel_start = float(np.clip(t_rel_start, 0.0, t_start_max))
        i0 = int(round(t_rel_start * fs))
        i1 = int(round((t_rel_start + view_w) * fs))
        i1 = min(i1, n_samp)
        if i1 <= i0:
            i1 = min(i0 + 1, n_samp)
        t_abs = t0 + (np.arange(i0, i1, dtype=np.float64) / fs)
        return t_abs, slice(i0, i1)

    def update_plot(channel_idx: int, t_rel_start: float) -> None:
        ch = int(np.clip(channel_idx, 0, n_ch - 1))
        t_abs, sl = slice_for_time_start(t_rel_start)
        status = check.get_status() if len(check_labels) else []
        ys: List[np.ndarray] = []
        for i, (disp, _, _, _) in enumerate(present):
            y = data_by_label[disp][ch, sl]
            lines[disp].set_data(t_abs, y)
            if i < len(status) and status[i]:
                ys.append(y)
        ax.set_xlim(t_abs[0], t_abs[-1])
        if ys:
            y_cat = np.concatenate(ys)
            y_min, y_max = float(np.min(y_cat)), float(np.max(y_cat))
            pad = 0.05 * (y_max - y_min + 1e-12)
            ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_ylabel(f"Amplitude (ch {ch})")
        fig.canvas.draw_idle()

    # --- 滑块：时间（相对裁剪段内的起点）---
    ax_time = fig.add_axes((0.12, 0.12, 0.76, 0.03))
    slider_time = Slider(
        ax_time,
        "Time start (s in segment)",
        0.0,
        t_start_max,
        valinit=0.0,
        valstep=max(total_dur / 2000, view_w / 200),
    )

    # --- 滑块：通道 ---
    ax_ch = fig.add_axes((0.12, 0.07, 0.76, 0.03))
    slider_ch = Slider(
        ax_ch,
        "Channel",
        0,
        max(0, n_ch - 1),
        valinit=0,
        valstep=1,
    )

    # --- 勾选：阶段显示 ---
    rax = fig.add_axes((0.02, 0.45, 0.08, 0.18))
    rax.set_title("Show", fontsize=8)
    check_labels = [p[0] for p in present]
    check = CheckButtons(rax, check_labels, actives=[True] * len(check_labels))

    def on_slider(_val: float) -> None:
        t_rel = slider_time.val
        ch = int(round(slider_ch.val))
        update_plot(ch, t_rel)

    def on_check(_label: str) -> None:
        status = check.get_status()
        for i, lab in enumerate(check_labels):
            lines[lab].set_visible(status[i])
        update_plot(int(round(slider_ch.val)), slider_time.val)

    slider_time.on_changed(on_slider)
    slider_ch.on_changed(on_slider)
    check.on_clicked(on_check)

    update_plot(0, 0.0)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="四阶段 EEG 交互波形查看（滑块+勾选）")
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--iir", type=Path, default=None)
    parser.add_argument("--asr", type=Path, default=None)
    parser.add_argument("--orica", type=Path, default=None)
    parser.add_argument(
        "--order",
        type=str,
        default=None,
        help="逗号分隔，默认 raw,iir,asr,orica（仅含 _STAGE_DEFS 里有的）",
    )
    parser.add_argument(
        "--view-sec",
        type=float,
        default=VIEW_DURATION_S,
        help="横向一次显示的时长（秒）",
    )
    args = parser.parse_args()

    script_dir = _SCRIPT_DIR
    explicit: Dict[str, Path] = {}
    if args.raw:
        explicit["raw"] = args.raw
    if args.iir:
        explicit["iir"] = args.iir
    if args.asr:
        explicit["asr"] = args.asr
    if args.orica:
        explicit["orica"] = args.orica

    order_keys = None
    if args.order:
        order_keys = aw._parse_order(args.order)

    run_interactive(args.view_sec, script_dir, order_keys, explicit)


if __name__ == "__main__":
    main()
