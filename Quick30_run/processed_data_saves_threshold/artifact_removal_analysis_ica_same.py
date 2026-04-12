"""
按时间窗切分 NPZ；在**完整对齐后的 IIR 数据上只拟合一次 ICA**，得到固定解混矩阵，
再对每个时间窗的 IIR / ASR / ORICA 数据块用**同一** `ica` 做 ICLabel（便于对比同一空间滤波
下各阶段在局部时段的分类差异）。

若命令行 --order 含 raw，则 Raw 仍按「每窗独立拟合 ICA」处理（与上述全段 IIR 模型无关）。

默认文件见 _STAGE_DEFS。

ICLabel 7 类顺序（与 mne-icalabel 概率矩阵列顺序一致）:
  brain, muscle, eye, heart, line_noise, channel_noise, other
mne-icalabel 返回的字符串标签为 muscle artifact / eye blink / heart beat 等，
脚本内会规范成与上表一致的简短类名再统计（否则会全部被误判为 other）。

依赖: numpy, matplotlib, scipy, mne, mne-icalabel
  pip install mne mne-icalabel

窗长建议: ICA 需要足够样本；默认 30s，可用 --window-sec 调整，不宜低于约 15–20s（高采样率可适当缩短）。

输出默认写入本脚本同目录下的 ica_analysis_result/（可用 --out-dir 修改）；
IIR/ASR/ORICA 结果图/CSV 前缀为 ica_iclabel_sameica__，热力图为 ica_iclabel_proba_heatmap_sameica__；
另输出 IC 源功率分组柱图（每窗：总能量、artifact、other、brain）ica_iclabel_power_sameica__ 等；
多阶段时 Y 轴取「各图柱峰值中的最小值」统一量程，超出部分柱内标真实值并画锯齿折断提示。
若本次同时含 IIR+ASR+ORICA，另存四行合成功率图 ica_iclabel_power_grid_sameica__（每行一类功率，每窗三柱对比三阶段；跨度大时 Y 为 asinh 或 symlog）。
另对整段对齐数据（不分窗）做 ICLabel，输出 ica_iclabel_fullrecording_sameica__*.png（功率四行+IC 计数堆叠）。
raw 仍为 ica_iclabel_windows__ / ica_iclabel_proba_heatmap__。
文件名均含 __win{秒}s（如 __win120p0s），与 --window-sec 一致。
"""


from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# ICLabel 类别（顺序需与 y_pred_proba 列一致）
# ---------------------------------------------------------------------------
ICLABEL_CLASSES: Tuple[str, ...] = (
    "brain",
    "muscle",
    "eye",
    "heart",
    "line_noise",
    "channel_noise",
    "other",
)
ARTIFACT_CLASSES = frozenset(
    {"muscle", "eye", "heart", "line_noise", "channel_noise"}
)

# 与 IIR 共用「该窗在 IIR 上拟合的 ICA」的阶段（raw 除外）
_STAGES_SHARED_IIR_ICA = frozenset({"iir", "asr", "orica"})

# mne-icalabel ICLABEL_NUMERICAL_TO_STRING 经 _normalize_label 后的别名 -> 本脚本 ICLABEL_CLASSES 键
_ICLABEL_NORMALIZED_ALIASES: Dict[str, str] = {
    "muscle_artifact": "muscle",
    "eye_blink": "eye",
    "heart_beat": "heart",
}

_STAGE_DEFS: Dict[str, Tuple[str, str]] = {
    "raw": ("Raw", "b01eeg_raw1.npz"),
    "iir": ("IIR", "b01eeg_iir1.npz"),
    "asr": ("ASR", "b01eeg_asr1.npz"),
    "orica": ("ORICA", "b01eeg_orica1.npz"),
}

# _STAGE_DEFS: Dict[str, Tuple[str, str]] = {
#     "raw": ("Raw", "b07eeg_offline_raw1.npz"),
#     "iir": ("IIR", "b07eeg_offline_iir1.npz"),
#     "asr": ("ASR", "b07eeg_offline_asr1.npz"),
#     "orica": ("ORICA", "b07eeg_offline_ica1.npz"),
    
# }


_DEFAULT_WINDOW_SEC = 60.0*2
_DEFAULT_OUT_SUBDIR = "ica_analysis_result_01_2min_same"
_MIN_SAMPLES_FACTOR = 20  # 至少约 20 * n_ch 个样本再尝试 ICA（经验下限，偏保守）

# ICA 算法：fastica 可用 fit_params 的 fun；infomax 不支持 fun（会报 unexpected keyword 'fun'）
_ICA_METHOD: str = "infomax"


@dataclass
class WindowICLabelResult:
    win_idx: int
    t0_s: float
    t1_s: float
    n_components: int
    counts: Dict[str, int] = field(default_factory=dict)
    n_brain: int = 0
    n_other: int = 0
    n_artifact: int = 0
    artifact_subtype_counts: Dict[str, int] = field(default_factory=dict)
    mean_conf_predicted: float = float("nan")
    mean_conf_artifact_ics: float = float("nan")
    mean_sum_artifact_prob: float = float("nan")
    # 该窗内「预测为 brain / other / 任一伪影类」的 IC 上，对应当类概率的平均（无该类 IC 时为 nan）
    avg_conf_brain: float = float("nan")
    avg_conf_other: float = float("nan")
    avg_conf_artifact: float = float("nan")
    proba_mean_per_class: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ICLABEL_CLASSES))
    )
    # 按预测类汇总的 IC 源功率：各 IC 取时间序列均方 mean(x^2)，再对 brain / other / artifact 求和
    power_sum_brain: float = float("nan")
    power_sum_other: float = float("nan")
    power_sum_artifact: float = float("nan")
    power_sum_total: float = float("nan")
    error: Optional[str] = None


def load_npz_file(npz_path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "data" not in data:
            print(f"[ERROR] {npz_path.name}: missing 'data'")
            return None
        eeg = np.asarray(data["data"], dtype=np.float64)
        if eeg.ndim != 2:
            print(f"[ERROR] {npz_path.name}: data must be 2D")
            return None
        if eeg.shape[1] <= 256 and eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T
        sr = data.get("sampling_rate", data.get("srate", 500))
        if isinstance(sr, np.ndarray):
            sr = float(sr.item())
        ch_names: List[str]
        if "channels" in data:
            ch_names = [_decode_ch(x) for x in np.asarray(data["channels"]).ravel()]
        else:
            ch_names = [f"EEG{i+1:03d}" for i in range(eeg.shape[0])]
        return {
            "data": eeg,
            "sampling_rate": float(sr),
            "ch_names": ch_names,
            "file_name": npz_path.name,
            "file_path": npz_path,
        }
    except Exception as e:
        print(f"[ERROR] load {npz_path}: {e}")
        return None


def _decode_ch(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _window_suffix_for_fname(window_sec: float) -> str:
    """拼在文件名末尾（扩展名前），如 win60s、win30p5s（小数点用 p）。"""
    if abs(window_sec - round(window_sec)) < 1e-6:
        return f"win{int(round(window_sec))}s"
    return f"win{str(window_sec).replace('.', 'p')}s"


def _parse_order(order_str: str) -> List[str]:
    allowed = frozenset(_STAGE_DEFS.keys())
    out: List[str] = []
    seen: set[str] = set()
    for part in order_str.split(","):
        k = part.strip().lower()
        if not k or k not in allowed or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out or list(_STAGE_DEFS.keys())


def _build_stage_list(
    script_dir: Path,
    order_keys: List[str],
    explicit: Dict[str, Path],
) -> List[Tuple[str, Path]]:
    stages: List[Tuple[str, Path]] = []
    for k in order_keys:
        label, default_name = _STAGE_DEFS[k]
        stages.append((label, explicit.get(k, script_dir / default_name)))
    return stages


def _normalize_label(s: Any) -> str:
    t = s.decode("utf-8", errors="ignore") if isinstance(s, bytes) else str(s)
    return t.strip().lower().replace(" ", "_")


def _canonical_icalabel_label(normalized: str) -> str:
    """把 API 标签统一到 ICLABEL_CLASSES 使用的键（避免 muscle_artifact 等被当成 unknown → other）。"""
    return _ICLABEL_NORMALIZED_ALIASES.get(normalized, normalized)


def _extract_proba_from_labels_out(
    labels_out: Any, n_ic: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从 label_components 返回值中取出概率。
    返回 (P_2d, p_1d)：
    - P_2d: (n_ic, n_col)，列尽量与 ICLABEL_CLASSES 前 n_col 列对齐；
    - p_1d: (n_ic,) 仅当 API 只给「预测类」边际概率时使用（无完整多类矩阵时）。
    """
    raw: Any = None
    if hasattr(labels_out, "y_pred_proba"):
        raw = labels_out.y_pred_proba
    elif isinstance(labels_out, dict):
        raw = labels_out.get("y_pred_proba")
        if raw is None:
            raw = labels_out.get("y_pred_proba_full")
    if raw is None:
        return None, None
    p = np.asarray(raw, dtype=np.float64)
    if p.ndim == 1:
        if p.size == n_ic:
            return None, p
        return None, None
    if p.ndim != 2:
        return None, None
    # 常见误排：(n_classes, n_ic) → 转置
    if p.shape[1] == n_ic and p.shape[0] != n_ic:
        p = p.T
    if p.shape[0] != n_ic:
        return None, None
    return p, None


def _extract_pred_labels(labels_out: Any, n_ic: int) -> List[str]:
    lab = None
    if hasattr(labels_out, "labels"):
        lab = labels_out.labels
    elif isinstance(labels_out, dict):
        lab = labels_out.get("labels")
    if lab is None:
        return ["other"] * n_ic
    arr = np.asarray(lab).ravel()
    return [_normalize_label(arr[i]) for i in range(min(n_ic, len(arr)))] + [
        "other"
    ] * max(0, n_ic - len(arr))


def _raw_from_array(
    data_ch_by_time: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    apply_iir: bool,
) -> Any:
    import mne

    n_ch, _n_samp = data_ch_by_time.shape
    ch_list = list(ch_names)
    if n_ch != len(ch_list):
        ch_list = [f"EEG{i+1:03d}" for i in range(n_ch)]

    x = np.asarray(data_ch_by_time, dtype=np.float64)
    if apply_iir:
        x = mne.filter.filter_data(
            x, sfreq, l_freq=1.0, h_freq=50.0, verbose=False
        )

    info = mne.create_info(ch_list, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(x, info)
    try:
        raw.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass
    return raw


def _ic_per_component_mean_square_power(ica: Any, raw: Any) -> Optional[np.ndarray]:
    """各 IC 源信号在时间上的 mean(x^2)，形状 (n_ic,)。"""
    try:
        sources = ica.get_sources(raw)
        src = sources.get_data()
    except Exception:
        return None
    src = np.asarray(src, dtype=np.float64)
    if src.ndim != 2 or src.shape[0] < 1:
        return None
    return np.mean(src * src, axis=1)


def _power_attr_from_result(r: WindowICLabelResult, attr: str) -> float:
    if r.error is not None:
        return float("nan")
    v = getattr(r, attr, float("nan"))
    return float(v) if np.isfinite(v) else float("nan")


_POWER_GRID_ROWS: Tuple[Tuple[str, str], ...] = (
    ("总能量 Σ（全部 IC）", "power_sum_total"),
    ("artifact", "power_sum_artifact"),
    ("other", "power_sum_other"),
    ("brain", "power_sum_brain"),
)

# 一行内 max/min(正值) 超过该倍数时用非线性 Y（asinh 或 symlog 回退：小值近线性、大值压缩）
_POWER_GRID_COMPRESS_RATIO = 12.0


def _collect_positive_powers_for_row(
    rows_res: Tuple[Sequence[WindowICLabelResult], ...],
    n_w: int,
    attr: str,
) -> Tuple[List[float], float]:
    """本行所有窗、三阶段中出现的正值功率列表与全局最大值。"""
    vals: List[float] = []
    row_max = 0.0
    for res_seq in rows_res:
        for i in range(n_w):
            v = _power_attr_from_result(res_seq[i], attr)
            if np.isfinite(v) and v > 0:
                fv = float(v)
                vals.append(fv)
                if fv > row_max:
                    row_max = fv
    return vals, row_max


def _choose_compress_linear_width(pos_vals: Sequence[float], row_max: float) -> float:
    """asinh.linear_width / symlog.linthresh：约小于此值的区间在轴上更「线性」，大值侧压缩。"""
    if not pos_vals or row_max <= 0:
        return max(row_max * 0.05, 1e-15)
    arr = np.asarray(pos_vals, dtype=np.float64)
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    cand = max(p25, p50 * 0.15, row_max / 200.0)
    return float(np.clip(cand, row_max / 400.0, row_max / 6.0))


def _power_grid_row_outlier_focus_cap(
    pos_vals: Sequence[float], row_max: float
) -> Optional[float]:
    """
    当极少数柱（如第 0 窗）远大于其余柱时，返回用于**线性 Y 轴**的上限：
    超出部分裁顶并标注真实值，让后面各窗的柱子仍能拉开对比。
    """
    if row_max <= 0:
        return None
    arr = np.asarray(
        [float(x) for x in pos_vals if np.isfinite(x) and float(x) > 0],
        dtype=np.float64,
    )
    if arr.size < 4:
        return None
    arr_s = np.sort(arr)
    second = float(arr_s[-2])
    p90 = float(np.percentile(arr, 90))
    if p90 <= 0 or second <= 0:
        return None
    # 最大柱远高于「次大柱」与高分位：典型单窗离群
    if row_max < second * 12.0 and row_max < p90 * 25.0:
        return None

    # 从大到小剥掉与「余下样本」明显脱节的尖峰（同一窗可能 1～3 根柱都极大）
    body = arr_s.astype(np.float64, copy=True)
    while body.size >= 6:
        cur = float(body[-1])
        rest = body[:-1]
        p90_r = float(np.percentile(rest, 90))
        sec_r = float(rest[-1])
        if cur > max(sec_r * 8.0, p90_r * 6.0, 1e-30):
            body = rest
            continue
        break

    bulk_max = float(body[-1])
    if bulk_max <= 0:
        return None
    median_b = float(np.percentile(body, 50))
    p94_b = float(np.percentile(body, 94))
    p95_b = float(np.percentile(body, 95))
    # 仅用「主体」估上限：避免 p50*10 等把 Y 顶到次大仍看不清
    cap = max(
        bulk_max * 1.07,
        p95_b * 1.1,
        p94_b * 1.14,
        median_b * 3.0,
    )
    cap = min(cap, bulk_max * 2.0)
    cap = max(cap, bulk_max * 1.02, median_b * 1.12, 1e-20)

    if cap >= row_max * 0.9:
        return None
    if row_max < cap * 1.8:
        return None
    return float(cap)


def plot_ic_power_grid_iir_asr_orica(
    results_iir: Sequence[WindowICLabelResult],
    results_asr: Sequence[WindowICLabelResult],
    results_orica: Sequence[WindowICLabelResult],
    out_path: Path,
    combo_tag: str,
    win_suffix: str,
    ica_fitted_note: Optional[str] = None,
) -> None:
    """
    四行子图：总能量、artifact、other、brain；每行每窗三根柱 IIR / ASR / ORICA。

    若某行出现「极少数极大柱」拉高坐标：自动改用**线性 Y + 裁顶离群柱**（柱内标真实值），
    其余情况仍可用 asinh/symlog 压缩跨度。
    """
    n_w = len(results_iir)
    if n_w == 0 or len(results_asr) != n_w or len(results_orica) != n_w:
        print("[WARN] IIR/ASR/ORICA 窗数不一致，跳过功率合成图。")
        return

    rows_res = (results_iir, results_asr, results_orica)
    stage_names_colors = (
        ("IIR", "#1f77b4"),
        ("ASR", "#ff7f0e"),
        ("ORICA", "#9467bd"),
    )
    n_st = 3
    xs = np.arange(n_w, dtype=float)
    bar_w = min(0.22, 0.65 / max(n_st, 1))
    gap = bar_w * 0.1
    step = bar_w + gap
    centers = (np.arange(n_st) - (n_st - 1) / 2.0) * step

    bad_idx = [
        i
        for i in range(n_w)
        if any(rows_res[s][i].error is not None for s in range(3))
    ]

    fig_h = 3.05 * len(_POWER_GRID_ROWS) + 1.35
    fig_w = max(12.0, n_w * 0.52)
    fig, axes = plt.subplots(
        len(_POWER_GRID_ROWS),
        1,
        figsize=(fig_w, fig_h),
        sharex=True,
    )
    axes_arr = np.atleast_1d(axes)

    note_line = f"{ica_fitted_note}\n" if ica_fitted_note else ""
    fig.suptitle(
        f"IIR / ASR / ORICA — IC 源能量对比（{combo_tag}）  [{win_suffix}]\n"
        f"{note_line}"
        f"四行：总能量、artifact、other、brain；每窗三柱为 IIR、ASR、ORICA。"
        f"若有极端离群柱，该行自动线性 Y 并裁顶标注，便于看清其余窗。",
        fontsize=10,
    )

    from matplotlib.patches import Patch

    leg_handles = [
        Patch(
            facecolor=stage_names_colors[i][1],
            edgecolor="white",
            label=stage_names_colors[i][0],
        )
        for i in range(n_st)
    ]

    for row_i, (row_title, attr) in enumerate(_POWER_GRID_ROWS):
        ax = axes_arr[row_i]
        pos_vals, row_max = _collect_positive_powers_for_row(rows_res, n_w, attr)
        focus_cap = _power_grid_row_outlier_focus_cap(pos_vals, row_max)

        row_min_pos = min(pos_vals) if pos_vals else 0.0
        spread = (
            row_max / max(row_min_pos, row_max * 1e-9)
            if row_max > 0
            else 1.0
        )
        use_compress = bool(
            focus_cap is None and row_max > 0 and spread >= _POWER_GRID_COMPRESS_RATIO
        )
        linear_w = _choose_compress_linear_width(pos_vals, row_max) if use_compress else 1.0

        for si, (_st_name, color) in enumerate(stage_names_colors):
            res_seq = rows_res[si]
            for i in range(n_w):
                r = res_seq[i]
                if r.error is not None:
                    continue
                x_pos = xs[i] + centers[si]
                v = _power_attr_from_result(r, attr)
                if not np.isfinite(v) or v < 0:
                    v = 0.0
                fv = float(v)
                if focus_cap is not None:
                    h = min(fv, focus_cap)
                    clipped = fv > focus_cap * (1.0 + 1e-9)
                else:
                    h = fv
                    clipped = False
                ax.bar(
                    x_pos,
                    h,
                    width=bar_w,
                    color=color,
                    edgecolor="white",
                    linewidth=0.35,
                    hatch="///" if clipped else None,
                )
                if clipped and h > 0 and focus_cap is not None:
                    txt = f"{fv:.0f}" if fv >= 100 else f"{fv:.2g}"
                    ty = min(h * 0.5, focus_cap * 0.48)
                    tcol = "white" if color != "#7f7f7f" else "black"
                    ax.text(
                        x_pos,
                        ty,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=5,
                        rotation=90,
                        color=tcol,
                        fontweight="bold",
                        clip_on=True,
                    )

        scale_lbl = ""
        if focus_cap is not None:
            scale_lbl = f"linear, cap≈{focus_cap:.3g} (outliers clipped)"
        elif use_compress:
            scale_lbl = f"asinh/symlog, w≈{linear_w:.3g}"
        ax.set_ylabel(
            f"{row_title}\n(Y: {scale_lbl})" if scale_lbl else row_title,
            fontsize=8,
        )
        ax.grid(True, axis="y", alpha=0.3, which="both")

        if row_max <= 0:
            ax.set_ylim(0.0, 1.0)
            ax.set_yscale("linear")
        elif focus_cap is not None:
            ax.set_yscale("linear")
            ax.set_ylim(0.0, float(focus_cap) * 1.08)
            _draw_yaxis_scale_break_zigzag(ax, float(focus_cap))
        elif use_compress:
            try:
                ax.set_yscale("asinh", linear_width=float(linear_w))
            except (ValueError, TypeError):
                ax.set_yscale(
                    "symlog",
                    linthresh=max(float(linear_w), 1e-15),
                    linscale=0.4,
                    base=10,
                )
            ax.set_ylim(0.0, row_max * 1.12)
        else:
            ax.set_yscale("linear")
            ax.set_ylim(0.0, max(row_max * 1.08, 1e-12))

        for i in bad_idx:
            ax.axvline(i, color="k", alpha=0.12, lw=1.0)

    axes_arr[-1].set_xlabel("Window index")
    for ax in axes_arr:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x)) for x in xs])

    fig.legend(
        handles=leg_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        ncol=3,
        fontsize=8,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved power grid figure: {out_path}")


def _maybe_plot_ic_power_grid(
    stage_runs: Sequence[Tuple[str, str, Path, List[WindowICLabelResult]]],
    out_dir: Path,
    win_suffix: str,
) -> None:
    """若 stage_runs 中同时有 iir、asr、orica，则输出一张四行合成功率图。"""
    got: Dict[str, Tuple[str, Path, List[WindowICLabelResult]]] = {}
    for sk, label, path, results in stage_runs:
        if sk in ("iir", "asr", "orica"):
            got[sk] = (label, path, results)
    needed = ("iir", "asr", "orica")
    if not all(k in got for k in needed):
        if any(k in got for k in needed):
            print("[INFO] 未同时包含 IIR+ASR+ORICA，跳过四行功率合成图。")
        return
    _lb_i, path_i, ri = got["iir"]
    _lb_a, path_a, ra = got["asr"]
    _lb_o, path_o, ro = got["orica"]
    combo_stem = f"{path_i.stem}__{path_a.stem}__{path_o.stem}"
    combo_tag = f"{path_i.stem} | {path_a.stem} | {path_o.stem}"
    plot_ic_power_grid_iir_asr_orica(
        ri,
        ra,
        ro,
        out_dir / f"ica_iclabel_power_grid_sameica__{combo_stem}__{win_suffix}.png",
        combo_tag=combo_tag,
        win_suffix=win_suffix,
        ica_fitted_note=_SHARED_ICA_NOTE,
    )


_STAGE_BAR_COLOR: Dict[str, str] = {
    "iir": "#1f77b4",
    "asr": "#ff7f0e",
    "orica": "#9467bd",
}


def plot_full_recording_ic_analysis_figure(
    full_res: Dict[str, WindowICLabelResult],
    stage_keys: Sequence[str],
    stage_display_names: Sequence[str],
    out_path: Path,
    combo_tag: str,
    duration_s: float,
    ica_fitted_note: Optional[str] = None,
) -> None:
    """
    整段数据（不分窗）：上四行与功率网格相同指标，每行各阶段一根柱；
    最下一行为各阶段 IC 计数堆叠（brain / other / artifact）。
    """
    if not full_res or not stage_keys:
        return
    n_st = len(stage_keys)
    if len(stage_display_names) != n_st:
        raise ValueError("stage_display_names 与 stage_keys 长度须一致")

    xs = np.arange(n_st, dtype=float)
    bar_w = min(0.55, 2.4 / max(n_st, 1))
    colors = [_STAGE_BAR_COLOR.get(k, "#555555") for k in stage_keys]

    fig, axes = plt.subplots(5, 1, figsize=(max(7.5, n_st * 2.2), 13.0))
    note_line = f"{ica_fitted_note}\n" if ica_fitted_note else ""
    fig.suptitle(
        f"整段数据 ICLabel（不分窗，0–{duration_s:.1f}s）— {combo_tag}\n"
        f"{note_line}"
        f"上行四格：IC 源功率；末行：各类 IC 个数堆叠",
        fontsize=10,
    )

    from matplotlib.patches import Patch

    leg_st = [
        Patch(facecolor=colors[i], edgecolor="white", label=stage_display_names[i])
        for i in range(n_st)
    ]

    for row_i, (row_title, attr) in enumerate(_POWER_GRID_ROWS):
        ax = axes[row_i]
        vals: List[float] = []
        pos_vals: List[float] = []
        for k in stage_keys:
            r = full_res[k]
            if r.error is not None:
                vals.append(0.0)
                continue
            v = getattr(r, attr, float("nan"))
            if np.isfinite(v) and v >= 0:
                fv = float(v)
                vals.append(fv)
                if fv > 0:
                    pos_vals.append(fv)
            else:
                vals.append(0.0)

        row_max = max(vals) if vals else 0.0
        row_min_pos = min(pos_vals) if pos_vals else 0.0
        spread = (
            row_max / max(row_min_pos, row_max * 1e-9) if row_max > 0 else 1.0
        )
        use_compress = bool(row_max > 0 and spread >= _POWER_GRID_COMPRESS_RATIO)
        linear_w = (
            _choose_compress_linear_width(pos_vals, row_max) if use_compress else 1.0
        )

        for ki, v in enumerate(vals):
            ax.bar(
                xs[ki],
                v,
                width=bar_w,
                color=colors[ki],
                edgecolor="white",
                linewidth=0.4,
            )

        scale_lbl = ""
        if use_compress:
            scale_lbl = f"asinh/symlog, w≈{linear_w:.3g}"
        ax.set_ylabel(
            f"{row_title}\n(Y: {scale_lbl})" if scale_lbl else row_title,
            fontsize=8,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(stage_display_names)
        ax.grid(True, axis="y", alpha=0.3, which="both")

        if row_max <= 0:
            ax.set_ylim(0.0, 1.0)
            ax.set_yscale("linear")
        elif use_compress:
            try:
                ax.set_yscale("asinh", linear_width=float(linear_w))
            except (ValueError, TypeError):
                ax.set_yscale(
                    "symlog",
                    linthresh=max(float(linear_w), 1e-15),
                    linscale=0.4,
                    base=10,
                )
            ax.set_ylim(0.0, row_max * 1.12)
        else:
            ax.set_yscale("linear")
            ax.set_ylim(0.0, max(row_max * 1.08, 1e-12))

    ax5 = axes[4]
    brain_v = np.zeros(n_st)
    other_v = np.zeros(n_st)
    art_v = np.zeros(n_st)
    for ki, k in enumerate(stage_keys):
        r = full_res[k]
        if r.error is not None:
            continue
        brain_v[ki] = float(r.n_brain)
        other_v[ki] = float(r.n_other)
        art_v[ki] = float(r.n_artifact)

    ax5.bar(
        xs,
        brain_v,
        width=bar_w,
        label="brain",
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.4,
    )
    ax5.bar(
        xs,
        other_v,
        width=bar_w,
        bottom=brain_v,
        label="other",
        color="#7f7f7f",
        edgecolor="white",
        linewidth=0.4,
    )
    ax5.bar(
        xs,
        art_v,
        width=bar_w,
        bottom=brain_v + other_v,
        label="artifact",
        color="#d62728",
        edgecolor="white",
        linewidth=0.4,
    )
    ax5.set_ylabel("# ICs（预测类）", fontsize=9)
    ax5.set_xticks(xs)
    ax5.set_xticklabels(stage_display_names)
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, axis="y", alpha=0.3)
    ax5.set_title("整段：各阶段 IC 个数（按类堆叠）")

    fig.legend(
        handles=leg_st,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        ncol=min(3, n_st),
        fontsize=8,
        title="阶段",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved full-recording figure: {out_path}")


def _maybe_plot_full_recording_figure(
    full_res: Optional[Dict[str, WindowICLabelResult]],
    loaded: Dict[str, Tuple[str, Path, Dict[str, Any]]],
    out_dir: Path,
) -> None:
    if not full_res:
        return
    preferred = ("iir", "asr", "orica")
    sks = [k for k in preferred if k in full_res]
    if not sks:
        sks = [k for k in full_res.keys()]
    if not sks:
        return
    names = [loaded[k][0] for k in sks if k in loaded]
    paths = [loaded[k][1] for k in sks if k in loaded]
    if len(names) != len(sks):
        names = list(sks)
    combo_stem = "__".join(p.stem for p in paths) if paths else "_".join(sks)
    combo_tag = " | ".join(p.stem for p in paths) if paths else " | ".join(sks)
    t1 = float(full_res[sks[0]].t1_s)
    plot_full_recording_ic_analysis_figure(
        full_res,
        sks,
        names,
        out_dir / f"ica_iclabel_fullrecording_sameica__{combo_stem}.png",
        combo_tag=combo_tag,
        duration_s=t1,
        ica_fitted_note=_SHARED_ICA_NOTE,
    )


def _power_plot_peak_per_stage(results: Sequence[WindowICLabelResult]) -> float:
    """单张功率柱图中，所有窗、四柱里出现的最大能量值（用于多阶段统一 Y 轴）。"""
    m = 0.0
    for r in results:
        if r.error is not None:
            continue
        for attr in (
            "power_sum_total",
            "power_sum_artifact",
            "power_sum_other",
            "power_sum_brain",
        ):
            v = getattr(r, attr, float("nan"))
            if np.isfinite(v) and float(v) > m:
                m = float(v)
    return m


def _draw_yaxis_scale_break_zigzag(ax: Any, y_mid: float) -> None:
    """在 y=y_mid 靠近左缘画横向小锯齿，表示量程上沿/有数据被裁剪。"""
    xmin, xmax = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x = xmax - xmin
    span_y = y1 - y0
    if span_x <= 0 or span_y <= 0 or not np.isfinite(y_mid):
        return
    x1 = xmin + span_x * 0.01
    d = span_x * 0.005
    amp = max(span_y * 0.018, abs(y_mid) * 0.025 if y_mid != 0 else span_y * 0.018)
    pts_x = [x1, x1 + d, x1 + 2 * d, x1 + 3 * d, x1 + 4 * d]
    pts_y = [y_mid, y_mid + amp, y_mid - amp, y_mid + amp, y_mid]
    ax.plot(pts_x, pts_y, "k-", lw=1.15, clip_on=False, zorder=50)


def _fit_ica_on_raw(
    raw: Any,
    random_state: int,
    ica_max_iter: int,
) -> Tuple[Any, Optional[str]]:
    from mne.preprocessing import ICA

    n_comp = len(raw.ch_names)
    if _ICA_METHOD == "fastica":
        ica_fit_params: Dict[str, Any] = dict(fun="logcosh")
    elif _ICA_METHOD == "infomax":
        ica_fit_params = dict(extended=True)
    else:
        ica_fit_params = {}
    try:
        ica = ICA(
            n_components=n_comp,
            method=_ICA_METHOD,
            random_state=random_state,
            max_iter=ica_max_iter,
            fit_params=ica_fit_params,
        )
        ica.fit(raw, verbose=False, reject_by_annotation=False)
    except Exception as e:
        return None, f"ICA fit failed: {e}"
    return ica, None


def _pack_iclabel_result_from_outputs(
    ica: Any,
    labels_out: Any,
    n_comp: int,
    raw: Optional[Any] = None,
) -> WindowICLabelResult:
    pred_labels = [
        _canonical_icalabel_label(lbl)
        for lbl in _extract_pred_labels(labels_out, n_comp)
    ]
    P, p1d = _extract_proba_from_labels_out(labels_out, n_comp)
    ls = getattr(ica, "labels_scores_", None)
    if ls is not None:
        ls_arr = np.asarray(ls, dtype=np.float64)
        n_cls = len(ICLABEL_CLASSES)
        if ls_arr.ndim == 2 and ls_arr.shape[0] == n_comp and ls_arr.shape[1] == n_cls:
            P = ls_arr
            p1d = None

    counts: Dict[str, int] = {c: 0 for c in ICLABEL_CLASSES}
    for i in range(n_comp):
        lbl = pred_labels[i] if i < len(pred_labels) else "other"
        if lbl not in counts:
            lbl = "other"
        counts[lbl] += 1

    n_brain = counts.get("brain", 0)
    n_other = counts.get("other", 0)
    n_artifact = sum(counts[c] for c in ARTIFACT_CLASSES)

    subtype = {c: counts[c] for c in ARTIFACT_CLASSES}

    conf_pred: List[float] = []
    conf_art: List[float] = []
    sum_art_prob: List[float] = []
    brain_confs: List[float] = []
    other_confs: List[float] = []
    artifact_confs: List[float] = []

    class_to_idx = {c: j for j, c in enumerate(ICLABEL_CLASSES)}
    n_cls = len(ICLABEL_CLASSES)

    def _row_full_7(i: int) -> Optional[np.ndarray]:
        if P is None or i >= P.shape[0]:
            return None
        ncol = min(int(P.shape[1]), n_cls)
        row = np.zeros(n_cls, dtype=np.float64)
        row[:ncol] = np.clip(P[i, :ncol], 0.0, 1.0)
        return row

    def _one_ic_prob(i: int, lbl: str) -> float:
        row7 = _row_full_7(i)
        if row7 is not None:
            j = class_to_idx.get(lbl, -1)
            if 0 <= j < n_cls:
                return float(row7[j])
            return float(np.max(row7))
        if p1d is not None and i < p1d.size:
            return float(np.clip(p1d.flat[i], 0.0, 1.0))
        return float("nan")

    for i in range(n_comp):
        lbl = pred_labels[i] if i < len(pred_labels) else "other"
        if lbl not in counts:
            lbl = "other"
        cp = _one_ic_prob(i, lbl)
        conf_pred.append(cp)
        row7 = _row_full_7(i)
        if row7 is not None:
            p_art = float(np.sum([row7[class_to_idx[c]] for c in ARTIFACT_CLASSES]))
            sum_art_prob.append(p_art)
        else:
            sum_art_prob.append(float("nan"))
        if lbl in ARTIFACT_CLASSES:
            conf_art.append(cp if np.isfinite(cp) else float("nan"))
        if np.isfinite(cp):
            if lbl == "brain":
                brain_confs.append(cp)
            elif lbl == "other":
                other_confs.append(cp)
            elif lbl in ARTIFACT_CLASSES:
                artifact_confs.append(cp)

    if P is not None and P.ndim == 2:
        ncol = min(int(P.shape[1]), n_cls)
        padded = np.zeros((n_comp, n_cls), dtype=np.float64)
        padded[:, :ncol] = np.clip(P[:, :ncol], 0.0, 1.0)
        proba_mean = np.mean(padded, axis=0)
    else:
        proba_mean = np.zeros(n_cls, dtype=np.float64)

    def _nanmean(a: List[float]) -> float:
        v = np.asarray(a, dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if v.size else float("nan")

    p_brain = p_other = p_art = float("nan")
    p_total = float("nan")
    if raw is not None:
        per_ic_pow = _ic_per_component_mean_square_power(ica, raw)
        if per_ic_pow is not None and per_ic_pow.size >= n_comp:
            per_ic_pow = per_ic_pow[:n_comp].astype(np.float64, copy=False)
            p_total = float(np.sum(per_ic_pow))
            sb = so = sa = 0.0
            for i in range(n_comp):
                lbl = pred_labels[i] if i < len(pred_labels) else "other"
                if lbl not in counts:
                    lbl = "other"
                pv = float(per_ic_pow[i])
                if lbl == "brain":
                    sb += pv
                elif lbl in ARTIFACT_CLASSES:
                    sa += pv
                else:
                    so += pv
            p_brain, p_other, p_art = sb, so, sa

    return WindowICLabelResult(
        win_idx=-1,
        t0_s=0.0,
        t1_s=0.0,
        n_components=n_comp,
        counts=dict(counts),
        n_brain=n_brain,
        n_other=n_other,
        n_artifact=n_artifact,
        artifact_subtype_counts=subtype,
        mean_conf_predicted=_nanmean(conf_pred),
        mean_conf_artifact_ics=_nanmean(conf_art),
        mean_sum_artifact_prob=_nanmean(sum_art_prob),
        avg_conf_brain=_nanmean(brain_confs),
        avg_conf_other=_nanmean(other_confs),
        avg_conf_artifact=_nanmean(artifact_confs),
        proba_mean_per_class=np.asarray(proba_mean, dtype=np.float64),
        power_sum_brain=p_brain,
        power_sum_other=p_other,
        power_sum_artifact=p_art,
        power_sum_total=p_total,
    )


def iclabel_on_raw_with_fitted_ica(
    raw: Any,
    ica: Any,
) -> Tuple[Optional[WindowICLabelResult], Optional[str]]:
    from mne_icalabel import label_components

    n_comp = int(ica.n_components_)
    try:
        labels_out = label_components(raw, ica, method="iclabel")
    except Exception as e:
        return None, f"ICLabel failed: {e}"
    return _pack_iclabel_result_from_outputs(ica, labels_out, n_comp, raw=raw), None


def run_ica_iclabel_window(
    data_ch_by_time: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    random_state: int,
    apply_iir: bool,
    ica_max_iter: int,
) -> Tuple[Optional[WindowICLabelResult], Optional[str]]:
    raw = _raw_from_array(data_ch_by_time, sfreq, ch_names, apply_iir)
    ica, err = _fit_ica_on_raw(raw, random_state, ica_max_iter)
    if err:
        return None, err
    return iclabel_on_raw_with_fitted_ica(raw, ica)


def analyze_file_windows(
    info: Dict[str, Any],
    window_sec: float,
    min_samples: int,
    apply_iir: bool,
    random_state: int,
    ica_max_iter: int,
    max_windows: int,
) -> List[WindowICLabelResult]:
    data = info["data"]
    sfreq = float(info["sampling_rate"])
    ch_names = info["ch_names"]
    n_ch, n_tot = data.shape
    win_samp = int(round(window_sec * sfreq))
    if win_samp < min_samples:
        print(
            f"[WARN] window_sec={window_sec}s -> {win_samp} samples < min_samples={min_samples}, skip file."
        )
        return []

    results: List[WindowICLabelResult] = []
    n_win = n_tot // win_samp
    if max_windows > 0:
        n_win = min(n_win, max_windows)

    for w in range(n_win):
        i0 = w * win_samp
        i1 = i0 + win_samp
        chunk = data[:, i0:i1]
        t0 = i0 / sfreq
        t1 = i1 / sfreq
        r, err = run_ica_iclabel_window(
            chunk, sfreq, ch_names, random_state + w, apply_iir, ica_max_iter
        )
        if r is None:
            emsg = err or "unknown"
            print(f"  [window {w}] failed: {emsg}")
            results.append(
                WindowICLabelResult(
                    win_idx=w,
                    t0_s=t0,
                    t1_s=t1,
                    n_components=n_ch,
                    error=emsg,
                )
            )
            continue
        r.win_idx = w
        r.t0_s = t0
        r.t1_s = t1
        results.append(r)
    return results


def _align_stage_infos_crop(
    infos_by_key: Dict[str, Dict[str, Any]],
    keys_needed: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], float, List[str], int]:
    """将各阶段 `data` 裁到相同时间长度；通道数、采样率须一致。"""
    ks = [k for k in keys_needed if k in infos_by_key]
    if not ks:
        raise ValueError("align: no infos")
    n_ch_ref: Optional[int] = None
    sfreq: Optional[float] = None
    ch_names_ref: Optional[List[str]] = None
    lengths: List[int] = []
    for k in ks:
        inf = infos_by_key[k]
        d = np.asarray(inf["data"], dtype=np.float64)
        lengths.append(d.shape[1])
        if n_ch_ref is None:
            n_ch_ref = d.shape[0]
            sfreq = float(inf["sampling_rate"])
            ch_names_ref = list(inf["ch_names"])
        else:
            if d.shape[0] != n_ch_ref:
                raise ValueError(
                    f"align: channel count mismatch for stage {k}: "
                    f"{d.shape[0]} vs {n_ch_ref}"
                )
            if abs(float(inf["sampling_rate"]) - float(sfreq)) > 1e-3:
                raise ValueError(f"align: sampling_rate mismatch for stage {k}")
    assert n_ch_ref is not None and sfreq is not None and ch_names_ref is not None
    n_min = min(lengths)
    out: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        inf = dict(infos_by_key[k])
        inf["data"] = np.asarray(inf["data"], dtype=np.float64)[:, :n_min].copy()
        out[k] = inf
    return out, sfreq, ch_names_ref, n_min


def analyze_windows_shared_iir_ica(
    infos_by_key: Dict[str, Dict[str, Any]],
    output_stage_keys: Sequence[str],
    window_sec: float,
    min_samples: int,
    apply_iir: bool,
    random_state: int,
    ica_max_iter: int,
    max_windows: int,
) -> Dict[str, List[WindowICLabelResult]]:
    """
    在**完整 IIR** 上 `fit` **一次** ICA，再对每个时间窗、每个 output_stage_keys 阶段
    的数据块用**同一** `ica` 调用 ICLabel。`infos_by_key` 须含 `iir` 及输出阶段键。
    """
    keys_align = list(frozenset(output_stage_keys) | frozenset({"iir"}))
    aligned, sfreq, ch_names, n_tot = _align_stage_infos_crop(infos_by_key, keys_align)
    n_ch = len(ch_names)

    win_samp = int(round(window_sec * sfreq))
    if win_samp < min_samples:
        print(
            f"[WARN] window_sec={window_sec}s -> {win_samp} samples < min_samples={min_samples}, skip shared ICA run."
        )
        return {k: [] for k in output_stage_keys}

    if n_tot < min_samples:
        print(
            f"[WARN] IIR 总长度 {n_tot} 样本 < min_samples={min_samples}，无法在整段 IIR 上拟合 ICA。"
        )
        return {k: [] for k in output_stage_keys}

    full_iir = aligned["iir"]["data"][:, :n_tot]
    raw_iir_full = _raw_from_array(full_iir, sfreq, ch_names, apply_iir)
    ica, err_fit = _fit_ica_on_raw(raw_iir_full, random_state, ica_max_iter)
    if ica is None:
        emsg = err_fit or "ICA fit failed"
        print(f"[ERROR] 整段 IIR 上 ICA 拟合失败: {emsg}")
        return {k: [] for k in output_stage_keys}

    print(
        f"[INFO] ICA 已在完整 IIR 上拟合（{n_tot} 样本 ≈ {n_tot / sfreq:.1f}s），"
        f"各时间窗复用该模型对 ASR/ORICA 等做 ICLabel。"
    )

    n_win = n_tot // win_samp
    if max_windows > 0:
        n_win = min(n_win, max_windows)

    results: Dict[str, List[WindowICLabelResult]] = {k: [] for k in output_stage_keys}

    for w in range(n_win):
        i0 = w * win_samp
        i1 = i0 + win_samp
        t0 = i0 / sfreq
        t1 = i1 / sfreq

        for sk in output_stage_keys:
            chunk_sk = aligned[sk]["data"][:, i0:i1]
            raw_sk = _raw_from_array(chunk_sk, sfreq, ch_names, apply_iir)
            r, err_l = iclabel_on_raw_with_fitted_ica(raw_sk, ica)
            if r is None:
                results[sk].append(
                    WindowICLabelResult(
                        win_idx=w,
                        t0_s=t0,
                        t1_s=t1,
                        n_components=n_ch,
                        error=err_l or "ICLabel failed",
                    )
                )
            else:
                r.win_idx = w
                r.t0_s = t0
                r.t1_s = t1
                results[sk].append(r)
    return results


def analyze_full_recording_shared_iir_ica(
    infos_by_key: Dict[str, Dict[str, Any]],
    output_stage_keys: Sequence[str],
    min_samples: int,
    apply_iir: bool,
    random_state: int,
    ica_max_iter: int,
) -> Dict[str, WindowICLabelResult]:
    """
    不分时间窗：在完整 IIR 上 fit 一次 ICA，对对齐后的各阶段**整段**数据做 ICLabel + 功率等统计。
    与分窗分析独立；仅要求整段样本数 >= min_samples（不要求满足 window_sec）。
    """
    keys_align = list(frozenset(output_stage_keys) | frozenset({"iir"}))
    aligned, sfreq, ch_names, n_tot = _align_stage_infos_crop(infos_by_key, keys_align)
    n_ch = len(ch_names)

    if n_tot < min_samples:
        print(
            f"[WARN] 整段分析跳过：IIR 总样本 {n_tot} < min_samples={min_samples}。"
        )
        return {}

    full_iir = aligned["iir"]["data"][:, :n_tot]
    raw_iir_full = _raw_from_array(full_iir, sfreq, ch_names, apply_iir)
    ica, err_fit = _fit_ica_on_raw(raw_iir_full, random_state, ica_max_iter)
    if ica is None:
        print(f"[ERROR] 整段分析：IIR 上 ICA 拟合失败: {err_fit}")
        return {}

    t1 = float(n_tot / sfreq)
    out: Dict[str, WindowICLabelResult] = {}
    for sk in output_stage_keys:
        chunk = aligned[sk]["data"][:, :n_tot]
        raw_sk = _raw_from_array(chunk, sfreq, ch_names, apply_iir)
        r, err_l = iclabel_on_raw_with_fitted_ica(raw_sk, ica)
        if r is None:
            out[sk] = WindowICLabelResult(
                win_idx=-1,
                t0_s=0.0,
                t1_s=t1,
                n_components=n_ch,
                error=err_l or "ICLabel failed",
            )
        else:
            r.win_idx = -1
            r.t0_s = 0.0
            r.t1_s = t1
            out[sk] = r

    print(
        f"[INFO] 整段数据 ICLabel 完成（0–{t1:.1f}s，{n_tot} 样本），"
        f"阶段: {list(out.keys())}"
    )
    return out


def compute_global_label_stats(
    results: Sequence[WindowICLabelResult],
) -> Dict[str, float]:
    """
    对所有「成功」窗口内的 IC 预测计数求和，得到 brain / other / artifact 的总数与占比（%）。
    占比分母为三者之和；每窗 n_brain+n_other+n_artifact 应等于该窗通道数（IC 数）。
    """
    ok = [r for r in results if r.error is None]
    nb = int(sum(r.n_brain for r in ok))
    no = int(sum(r.n_other for r in ok))
    na = int(sum(r.n_artifact for r in ok))
    n = nb + no + na
    if n == 0:
        return {
            "n_windows_ok": float(len(ok)),
            "total_ic": 0.0,
            "n_brain": 0.0,
            "n_other": 0.0,
            "n_artifact": 0.0,
            "pct_brain": 0.0,
            "pct_other": 0.0,
            "pct_artifact": 0.0,
        }
    return {
        "n_windows_ok": float(len(ok)),
        "total_ic": float(n),
        "n_brain": float(nb),
        "n_other": float(no),
        "n_artifact": float(na),
        "pct_brain": 100.0 * nb / n,
        "pct_other": 100.0 * no / n,
        "pct_artifact": 100.0 * na / n,
    }


def plot_stage_results(
    stage_label: str,
    file_stem: str,
    results: Sequence[WindowICLabelResult],
    out_path: Path,
    artifact_subtype_ymax: Optional[float] = None,
    ica_fitted_note: Optional[str] = None,
) -> None:
    ok = [r for r in results if r.error is None]
    bad = [r for r in results if r.error is not None]
    if not ok and not bad:
        print(f"[WARN] no windows for {stage_label}, skip figure.")
        return

    n_w = len(results)
    xs = np.arange(n_w)
    gst = compute_global_label_stats(results)
    summary_line = (
        f"全局占比（成功窗合计 {int(gst['n_windows_ok'])} 窗，"
        f"共 {int(gst['total_ic'])} 次 IC 预测）: "
        f"brain {gst['pct_brain']:.2f}%  |  other {gst['pct_other']:.2f}%  |  "
        f"artifact {gst['pct_artifact']:.2f}%"
    )

    fig, axes = plt.subplots(3, 1, figsize=(max(10.0, n_w * 0.35), 10.5), sharex=True)
    note_line = f"{ica_fitted_note}\n" if ica_fitted_note else ""
    fig.suptitle(
        f"{stage_label} ({file_stem}) — ICA + ICLabel per window\n"
        f"{note_line}"
        f"failed windows: {len(bad)} / {n_w}\n"
        f"{summary_line}",
        fontsize=10,
    )

    # --- 1) 堆叠计数：brain / other / artifact ---
    ax0 = axes[0]
    b = np.array([r.n_brain for r in results])
    o = np.array([r.n_other for r in results])
    a = np.array([r.n_artifact for r in results])
    ax0.bar(xs, b, label="brain", color="#2ca02c")
    ax0.bar(xs, o, bottom=b, label="other", color="#7f7f7f")
    ax0.bar(xs, a, bottom=b + o, label="artifact (all)", color="#d62728")
    ax0.set_ylabel("# ICs")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.set_title("Per-window IC counts (predicted class)")

    # --- 2) 伪影子类堆叠 ---
    ax1 = axes[1]
    sub_names = [c for c in ICLABEL_CLASSES if c in ARTIFACT_CLASSES]
    colors = ["#ff9896", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    bottom = np.zeros(n_w)
    for si, name in enumerate(sub_names):
        col = np.array(
            [r.artifact_subtype_counts.get(name, 0) for r in results], dtype=float
        )
        ax1.bar(xs, col, bottom=bottom, label=name, color=colors[si % len(colors)])
        bottom += col
    if artifact_subtype_ymax is not None and artifact_subtype_ymax > 0:
        ax1.set_ylim(0.0, artifact_subtype_ymax)
    ax1.set_ylabel("# ICs")
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title("Artifact subtypes (muscle / eye / heart / line_noise / channel_noise)")

    # --- 3) 每窗三类 IC 的平均预测置信度（分组柱：brain / other / artifact）---
    ax2 = axes[2]

    def _bar_h(r: WindowICLabelResult, attr: str) -> float:
        if r.error is not None:
            return 0.0
        v = getattr(r, attr, float("nan"))
        return float(v) if np.isfinite(v) else 0.0

    b_conf = np.array([_bar_h(r, "avg_conf_brain") for r in results])
    o_conf = np.array([_bar_h(r, "avg_conf_other") for r in results])
    a_conf = np.array([_bar_h(r, "avg_conf_artifact") for r in results])
    w = 0.22
    ax2.bar(xs - w, b_conf, width=w, label="brain (该类 IC 平均置信度)", color="#2ca02c")
    ax2.bar(xs, o_conf, width=w, label="other (该类 IC 平均置信度)", color="#7f7f7f")
    ax2.bar(xs + w, a_conf, width=w, label="artifact (该类 IC 平均置信度)", color="#d62728")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("Window index")
    ax2.set_ylabel("平均预测类概率")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_title(
        "每窗：预测为 brain / other / 伪影 的 IC 上，对应当类概率的平均（无该类 IC 时柱高为 0）"
    )

    for i, r in enumerate(results):
        if r.error:
            for ax in axes:
                ax.axvline(i, color="k", alpha=0.15, lw=0.8)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure: {out_path}")


def plot_ic_class_power_by_window(
    stage_label: str,
    file_stem: str,
    results: Sequence[WindowICLabelResult],
    out_path: Path,
    ica_fitted_note: Optional[str] = None,
    y_power_cap: Optional[float] = None,
) -> None:
    """
    每个时间窗画 4 根并列柱：总能量（全部 IC）、artifact、other、brain。
    能量定义：各 IC 源时间序列 mean(x²)，同类 IC 相加；总能量为三类之和（= 全部 IC）。

    y_power_cap：多阶段对比时统一的 Y 轴上限（通常取「各阶段峰值中的最小值」）。
    超出部分柱顶裁到该上限，柱内标注真实数值，并在 y=y_power_cap 处画锯齿折断线。
    """
    if not results:
        print(f"[WARN] no windows for {stage_label}, skip power figure.")
        return

    n_w = len(results)
    xs = np.arange(n_w, dtype=float)
    totals: List[float] = []
    arts: List[float] = []
    others: List[float] = []
    brains: List[float] = []
    bad_idx: List[int] = []

    for i, r in enumerate(results):
        if r.error is not None:
            bad_idx.append(i)
            totals.append(0.0)
            arts.append(0.0)
            others.append(0.0)
            brains.append(0.0)
            continue
        t = r.power_sum_total
        if not np.isfinite(t) or t < 0:
            bad_idx.append(i)
            totals.append(0.0)
            arts.append(0.0)
            others.append(0.0)
            brains.append(0.0)
            continue
        totals.append(float(t))
        arts.append(
            float(r.power_sum_artifact)
            if np.isfinite(r.power_sum_artifact)
            else 0.0
        )
        others.append(
            float(r.power_sum_other) if np.isfinite(r.power_sum_other) else 0.0
        )
        brains.append(
            float(r.power_sum_brain) if np.isfinite(r.power_sum_brain) else 0.0
        )

    fig_w = max(12.0, n_w * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))
    note_line = f"{ica_fitted_note}\n" if ica_fitted_note else ""
    fig.suptitle(
        f"{stage_label} ({file_stem}) — 每窗 IC 源能量（柱状）\n"
        f"{note_line}"
        f"各柱：总能量 | artifact | other | brain（单位：mean(x²) 在 IC 与时间上的汇总）",
        fontsize=10,
    )

    # 顺序：总能量、artifact、other、brain
    series: List[Tuple[str, List[float], str]] = [
        ("总能量 Σ", totals, "#1f77b4"),
        ("artifact", arts, "#d62728"),
        ("other", others, "#7f7f7f"),
        ("brain", brains, "#2ca02c"),
    ]
    n_bar = len(series)
    w = min(0.18, 0.75 / (n_bar + 1))
    step = w + w * 0.12
    centers = (np.arange(n_bar) - (n_bar - 1) / 2.0) * step

    cap = y_power_cap
    if cap is not None and (not np.isfinite(cap) or cap <= 0):
        cap = None

    stage_max = max(
        (max(vals) if vals else 0.0)
        for _, vals, _ in series
    )
    any_clipped = bool(cap is not None and stage_max > cap * (1.0 + 1e-9))

    for k, (lbl, vals, color) in enumerate(series):
        for i, v in enumerate(vals):
            if not np.isfinite(v) or v < 0:
                v = 0.0
            x_pos = xs[i] + centers[k]
            h = float(min(v, cap)) if cap is not None else float(v)
            clipped = cap is not None and v > cap * (1.0 + 1e-9)
            ax.bar(
                x_pos,
                h,
                width=w,
                label=lbl if i == 0 else "_nolegend_",
                color=color,
                edgecolor="white",
                linewidth=0.4,
                hatch="///" if clipped else None,
            )
            if clipped and h > 0:
                txt = f"{v:.0f}" if v >= 100 else f"{v:.1f}"
                tx = min(h * 0.5, cap * 0.45) if cap is not None else h * 0.5
                tcol = "white" if color not in ("#7f7f7f",) else "black"
                ax.text(
                    x_pos,
                    tx,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=5,
                    rotation=90,
                    color=tcol,
                    fontweight="bold",
                    clip_on=True,
                )

    ax.set_xlabel("Window index")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])
    cap_note = (
        f"（统一 Y 上限={cap:.4g}，各阶段峰值取最小；超出柱内数字为真实值）"
        if cap is not None
        else ""
    )
    ax.set_ylabel("能量（Σ mean(x²)）")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"每窗四柱：总能量 | artifact | other | brain{cap_note}")

    if cap is not None:
        ax.set_ylim(0.0, float(cap))
        if any_clipped:
            _draw_yaxis_scale_break_zigzag(ax, float(cap))
    else:
        ymax = max(stage_max * 1.06, 1e-12)
        ax.set_ylim(0.0, ymax)

    for i in bad_idx:
        ax.axvline(i, color="k", alpha=0.12, lw=1.0)

    plt.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved power figure: {out_path}")


def save_results_csv(
    stage_label: str,
    file_stem: str,
    results: Sequence[WindowICLabelResult],
    out_path: Path,
) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_cols = [c for c in ICLABEL_CLASSES if c in ARTIFACT_CLASSES]
    header = (
        ["window", "t0_s", "t1_s", "error", "n_ic", "n_brain", "n_other", "n_artifact"]
        + [f"n_{c}" for c in sub_cols]
        + [
            "mean_conf_pred",
            "mean_conf_artifact_ics",
            "mean_sum_P_artifact",
            "avg_conf_brain",
            "avg_conf_other",
            "avg_conf_artifact",
            "power_sum_brain",
            "power_sum_other",
            "power_sum_artifact",
            "power_sum_total",
        ]
        + [f"mean_prob_{c}" for c in ICLABEL_CLASSES]
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# stage={stage_label}", f"file={file_stem}"])
        w.writerow(header)
        for r in results:
            row = [
                r.win_idx,
                f"{r.t0_s:.4f}",
                f"{r.t1_s:.4f}",
                r.error or "",
                r.n_components,
                r.n_brain,
                r.n_other,
                r.n_artifact,
            ]
            row += [r.artifact_subtype_counts.get(c, 0) for c in sub_cols]
            row += [
                f"{r.mean_conf_predicted:.6f}"
                if np.isfinite(r.mean_conf_predicted)
                else "",
                f"{r.mean_conf_artifact_ics:.6f}"
                if np.isfinite(r.mean_conf_artifact_ics)
                else "",
                f"{r.mean_sum_artifact_prob:.6f}"
                if np.isfinite(r.mean_sum_artifact_prob)
                else "",
                f"{r.avg_conf_brain:.6f}" if np.isfinite(r.avg_conf_brain) else "",
                f"{r.avg_conf_other:.6f}" if np.isfinite(r.avg_conf_other) else "",
                f"{r.avg_conf_artifact:.6f}"
                if np.isfinite(r.avg_conf_artifact)
                else "",
                f"{r.power_sum_brain:.8e}" if np.isfinite(r.power_sum_brain) else "",
                f"{r.power_sum_other:.8e}" if np.isfinite(r.power_sum_other) else "",
                f"{r.power_sum_artifact:.8e}"
                if np.isfinite(r.power_sum_artifact)
                else "",
                f"{r.power_sum_total:.8e}" if np.isfinite(r.power_sum_total) else "",
            ]
            for j, _ in enumerate(ICLABEL_CLASSES):
                v = r.proba_mean_per_class[j] if j < len(r.proba_mean_per_class) else 0.0
                row.append(f"{float(v):.6f}")
            w.writerow(row)

        gst = compute_global_label_stats(results)
        w.writerow([])
        w.writerow(["# GLOBAL_SUMMARY", "仅统计 error 为空的窗口"])
        w.writerow(["metric", "value"])
        w.writerow(["successful_windows", int(gst["n_windows_ok"])])
        w.writerow(["total_ic_predictions", int(gst["total_ic"])])
        w.writerow(["n_brain", int(gst["n_brain"])])
        w.writerow(["n_other", int(gst["n_other"])])
        w.writerow(["n_artifact", int(gst["n_artifact"])])
        w.writerow(["pct_brain_percent", f"{gst['pct_brain']:.6f}"])
        w.writerow(["pct_other_percent", f"{gst['pct_other']:.6f}"])
        w.writerow(["pct_artifact_percent", f"{gst['pct_artifact']:.6f}"])
    print(f"  saved CSV: {out_path}")


def plot_proba_heatmap(
    stage_label: str,
    file_stem: str,
    results: Sequence[WindowICLabelResult],
    out_path: Path,
    ica_fitted_note: Optional[str] = None,
) -> None:
    ok = [r for r in results if r.error is None]
    if not ok:
        return
    mat = np.stack([r.proba_mean_per_class for r in results], axis=0)
    gst = compute_global_label_stats(results)
    title_extra = (
        f"全局占比: brain {gst['pct_brain']:.2f}% | other {gst['pct_other']:.2f}% | "
        f"artifact {gst['pct_artifact']:.2f}% "
        f"（{int(gst['n_windows_ok'])} 窗，{int(gst['total_ic'])} IC）"
    )
    note = f"{ica_fitted_note}\n" if ica_fitted_note else ""
    fig, ax = plt.subplots(figsize=(max(8.0, len(results) * 0.25), 5.0))
    im = ax.imshow(mat.T, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(ICLABEL_CLASSES)))
    ax.set_yticklabels(ICLABEL_CLASSES, fontsize=8)
    ax.set_xlabel("Window index")
    ax.set_title(
        f"{stage_label} — mean ICLabel class probability per IC (per window)\n"
        f"{note}{title_extra}",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="mean prob")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved heatmap: {out_path}")


_SHARED_ICA_NOTE = (
    "ICA：在完整 IIR 上仅 fit 一次；本图各窗对应当阶段数据复用该解混矩阵做 ICLabel。"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "分窗 ICLabel：在完整 IIR 上拟合一次 ICA，各窗对 IIR/ASR/ORICA 复用该模型；"
            "若含 raw，则 raw 仍为每窗独立拟合 ICA。"
        ),
    )
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--iir", type=Path, default=None)
    parser.add_argument("--asr", type=Path, default=None)
    parser.add_argument("--orica", type=Path, default=None)
    parser.add_argument(
        "--order",
        type=str,
        default="iir,asr,orica",
        help="逗号分隔阶段顺序（raw 若存在则单独每窗拟合 ICA）",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=_DEFAULT_WINDOW_SEC,
        help="每窗时长（秒），建议 >= 20–30",
    )
    parser.add_argument(
        "--iir-before-ica",
        action="store_true",
        help="ICA 前对每窗做 1–50 Hz 带通（Raw 可开；已滤波阶段慎用以避免双重滤波）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ica-max-iter", type=int, default=500)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="每文件最多处理窗数，0 表示不限制",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录，默认 本脚本目录/ica_analysis_result",
    )
    args = parser.parse_args()

    script_dir = _SCRIPT_DIR
    out_dir = args.out_dir or (script_dir / _DEFAULT_OUT_SUBDIR)
    order_keys = _parse_order(args.order)
    explicit: Dict[str, Path] = {}
    if args.raw:
        explicit["raw"] = args.raw
    if args.iir:
        explicit["iir"] = args.iir
    if args.asr:
        explicit["asr"] = args.asr
    if args.orica:
        explicit["orica"] = args.orica

    try:
        import mne  # noqa: F401
        import mne_icalabel  # noqa: F401
    except ImportError as e:
        print("请先安装: pip install mne mne-icalabel")
        raise SystemExit(1) from e

    stages = _build_stage_list(script_dir, order_keys, explicit)
    win_suffix = _window_suffix_for_fname(float(args.window_sec))

    loaded: Dict[str, Tuple[str, Path, Dict[str, Any]]] = {}
    for sk, (label, path) in zip(order_keys, stages):
        path = path.resolve()
        if not path.is_file():
            print(f"[SKIP] missing {label}: {path}")
            continue
        inf = load_npz_file(path)
        if inf is None:
            continue
        loaded[sk] = (label, path, inf)

    shared_in_order = [k for k in order_keys if k in _STAGES_SHARED_IIR_ICA]
    if shared_in_order and "iir" not in loaded:
        iir_path = (explicit.get("iir") or (script_dir / _STAGE_DEFS["iir"][1])).resolve()
        if iir_path.is_file():
            inf_iir = load_npz_file(iir_path)
            if inf_iir is not None:
                lbl, _ = _STAGE_DEFS["iir"]
                loaded["iir"] = (lbl, iir_path, inf_iir)
                print(
                    f"[INFO] 为共用 ICA 已加载 IIR（未在 --order 中也可）: {iir_path.name}"
                )
        else:
            print(
                f"[WARN] 需要 IIR 以拟合共用 ICA，但文件不存在: {iir_path}"
            )

    infos_shared: Dict[str, Dict[str, Any]] = {}
    for k in frozenset(shared_in_order) | frozenset({"iir"}):
        if k in loaded:
            infos_shared[k] = loaded[k][2]

    shared_res: Optional[Dict[str, List[WindowICLabelResult]]] = None
    full_recording_results: Optional[Dict[str, WindowICLabelResult]] = None
    if shared_in_order:
        if "iir" not in infos_shared:
            print("[ERROR] 无法运行 IIR 共用 ICA：缺少 IIR 数据。")
        else:
            out_keys = [k for k in order_keys if k in _STAGES_SHARED_IIR_ICA and k in loaded]
            missing_align = [k for k in out_keys if k not in infos_shared]
            if missing_align:
                print(
                    f"[WARN] 共用 ICA 跳过缺失阶段: {missing_align}"
                )
            out_keys = [k for k in out_keys if k in infos_shared]
            if out_keys:
                iir_inf = infos_shared["iir"]
                n_ch = int(iir_inf["data"].shape[0])
                min_samples = max(
                    int(_MIN_SAMPLES_FACTOR * n_ch),
                    int(10 * float(iir_inf["sampling_rate"])),
                )
                print(
                    f"\n=== 共用 ICA（完整 IIR 上 fit 一次）| 输出阶段: {out_keys} | "
                    f"IIR shape={iir_inf['data'].shape} | "
                    f"min_samples={min_samples} (~{min_samples/float(iir_inf['sampling_rate']):.1f}s) ==="
                )
                try:
                    shared_res = analyze_windows_shared_iir_ica(
                        infos_shared,
                        out_keys,
                        window_sec=args.window_sec,
                        min_samples=min_samples,
                        apply_iir=args.iir_before_ica,
                        random_state=args.seed,
                        ica_max_iter=args.ica_max_iter,
                        max_windows=args.max_windows,
                    )
                except ValueError as e:
                    print(f"[ERROR] 共用 ICA 对齐失败: {e}")
                    shared_res = None
                if shared_res:
                    for k in out_keys:
                        gst = compute_global_label_stats(shared_res[k])
                        print(
                            f"  [{loaded[k][0]}] 全局占比: "
                            f"brain {gst['pct_brain']:.2f}% | other {gst['pct_other']:.2f}% | "
                            f"artifact {gst['pct_artifact']:.2f}% "
                            f"（{int(gst['total_ic'])} 次预测，{int(gst['n_windows_ok'])} 成功窗）"
                        )

                try:
                    full_recording_results = analyze_full_recording_shared_iir_ica(
                        infos_shared,
                        out_keys,
                        min_samples,
                        args.iir_before_ica,
                        args.seed,
                        args.ica_max_iter,
                    )
                except ValueError as e:
                    print(f"[WARN] 整段 ICLabel 对齐失败: {e}")
                    full_recording_results = None
                if full_recording_results:
                    for k in out_keys:
                        rr = full_recording_results.get(k)
                        if rr is None:
                            continue
                        if rr.error:
                            print(f"  [整段 {loaded[k][0]}] 失败: {rr.error}")
                        else:
                            n_ic = rr.n_brain + rr.n_other + rr.n_artifact
                            print(
                                f"  [整段 {loaded[k][0]}] IC 计数: brain {rr.n_brain}, "
                                f"other {rr.n_other}, artifact {rr.n_artifact} "
                                f"（合计 {n_ic} IC）"
                            )

    stage_runs: List[Tuple[str, str, Path, List[WindowICLabelResult]]] = []
    global_artifact_subtype_peak = 0.0

    for sk in order_keys:
        if sk == "raw" and sk in loaded:
            label, path, inf = loaded[sk]
            n_ch = inf["data"].shape[0]
            min_samples = max(
                int(_MIN_SAMPLES_FACTOR * n_ch), int(10 * inf["sampling_rate"])
            )
            print(
                f"\n=== {label}（每窗独立 ICA）| {path.name} | shape={inf['data'].shape} | "
                f"min_samples={min_samples} (~{min_samples/inf['sampling_rate']:.1f}s) ==="
            )
            results = analyze_file_windows(
                inf,
                window_sec=args.window_sec,
                min_samples=min_samples,
                apply_iir=args.iir_before_ica,
                random_state=args.seed,
                ica_max_iter=args.ica_max_iter,
                max_windows=args.max_windows,
            )
            gst = compute_global_label_stats(results)
            print(
                f"  全局占比（成功窗内 IC 预测合计）: "
                f"brain {gst['pct_brain']:.2f}%  |  other {gst['pct_other']:.2f}%  |  "
                f"artifact {gst['pct_artifact']:.2f}%  "
                f"（共 {int(gst['total_ic'])} 次预测，{int(gst['n_windows_ok'])} 个成功窗）"
            )
            local_peak = 0.0
            for r in results:
                if r.error is not None:
                    continue
                local_peak = max(
                    local_peak,
                    float(
                        sum(r.artifact_subtype_counts.get(c, 0) for c in ARTIFACT_CLASSES)
                    ),
                )
            global_artifact_subtype_peak = max(global_artifact_subtype_peak, local_peak)
            stage_runs.append((sk, label, path, results))

        elif sk in _STAGES_SHARED_IIR_ICA and sk in loaded and shared_res is not None:
            label, path, _ = loaded[sk]
            results = shared_res.get(sk, [])
            if not results:
                print(f"[SKIP] {label}: 共用 ICA 无结果（可能被跳过或失败）")
                continue
            local_peak = 0.0
            for r in results:
                if r.error is not None:
                    continue
                local_peak = max(
                    local_peak,
                    float(
                        sum(r.artifact_subtype_counts.get(c, 0) for c in ARTIFACT_CLASSES)
                    ),
                )
            global_artifact_subtype_peak = max(global_artifact_subtype_peak, local_peak)
            stage_runs.append((sk, label, path, results))

    artifact_subtype_ymax = max(1.0, float(np.ceil(global_artifact_subtype_peak)))

    power_peaks: List[float] = []
    for _sk, _label, _path, _results in stage_runs:
        pk = _power_plot_peak_per_stage(_results)
        if pk > 0:
            power_peaks.append(pk)
    unified_power_cap: Optional[float] = (
        float(min(power_peaks)) if power_peaks else None
    )
    if unified_power_cap is not None:
        tag = (
            "多阶段：各图峰值中的最小值"
            if len(power_peaks) > 1
            else "单阶段：本图柱峰值"
        )
        print(
            f"[INFO] 功率图 Y 轴上限={unified_power_cap:.6g}（{tag}；"
            f"超出时柱顶裁切+柱内数字+斜线填充+锯齿线）"
        )

    for sk, label, path, results in stage_runs:
        stem = path.stem
        use_shared = sk in _STAGES_SHARED_IIR_ICA
        prefix = "ica_iclabel_sameica__" if use_shared else "ica_iclabel_windows__"
        note = _SHARED_ICA_NOTE if use_shared else None
        plot_stage_results(
            label,
            stem,
            results,
            out_dir / f"{prefix}{stem}__{win_suffix}.png",
            artifact_subtype_ymax=artifact_subtype_ymax,
            ica_fitted_note=note,
        )
        pow_fname = (
            f"ica_iclabel_power_sameica__{stem}__{win_suffix}.png"
            if use_shared
            else f"ica_iclabel_power__{stem}__{win_suffix}.png"
        )
        plot_ic_class_power_by_window(
            label,
            stem,
            results,
            out_dir / pow_fname,
            ica_fitted_note=note,
            y_power_cap=unified_power_cap,
        )
        hm_fname = (
            f"ica_iclabel_proba_heatmap_sameica__{stem}__{win_suffix}.png"
            if use_shared
            else f"ica_iclabel_proba_heatmap__{stem}__{win_suffix}.png"
        )
        plot_proba_heatmap(
            label,
            stem,
            results,
            out_dir / hm_fname,
            ica_fitted_note=note,
        )
        save_results_csv(
            label,
            stem,
            results,
            out_dir / f"{prefix}{stem}__{win_suffix}.csv",
        )

    _maybe_plot_ic_power_grid(stage_runs, out_dir, win_suffix)
    _maybe_plot_full_recording_figure(full_recording_results, loaded, out_dir)


if __name__ == "__main__":
    main()
