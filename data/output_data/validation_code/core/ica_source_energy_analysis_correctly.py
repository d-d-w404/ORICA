"""
独立脚本：始终在全长对齐数据上拟合 ICA（不做 ICA 前 crop）。
解混矩阵由 IIR 通道数据上的 ICA 确定后，对 IIR/ASR/ORICA 三阶段求源信号，
再计算源 MS 与相对 IIR 的 %。

可选 EXCLUDE_TIME_RANGES_S：仅在「解混之后」对源时间维做 mask，排除指定时间段后再算 MS
（坏段不参与能量统计；ICA/ICLabel 仍用全长）。
分窗类均值曲线：x 轴为录音时间（秒）；另存与 EXCLUDE 重叠的窗置 NaN + 阴影标注的 `*_exclude_bad_segments_timeline.png`。

额外输出：每个 IC 三根柱 —— IIR / ASR / ORICA 的源均方（MS）。

在「等权 IC」类汇总行之外，增加按 IIR 源 MS 加权的类平均下降（pp），并写入同一张 per-IC 对比图的标题区；
另存 `reduction_summary_power_weighted_*.csv`。

输出：{script_dir}/{profile}/ica_source_analysis/b{dataset}/{子目录}/

子目录名里 EXCLUDE 段：用各区间**起点**（秒，取 int）下划线拼接，例如
``..._window_10_remove_0_122_445_794_983_1149``，不再写入每段终点。

通道筛选（ICA 前）：可设 INCLUDE_CHANNEL_NAMES（仅保留）与/或 EXCLUDE_CHANNEL_NAMES（剔除），
名称与 npz 中 channels 一致、不区分大小写。被剔除的通道名会写入输出子目录名（过长时短名+哈希）。

运行：python ica_source_energy_analysis_correctly.py
"""

from __future__ import annotations



# --- validation_code 路径（数据在 output_data 根目录）---
import sys
from pathlib import Path as _Path

_vc = _Path(__file__).resolve().parent
for _p in (_vc, *_vc.parents):
    if (_p / "_paths.py").is_file() and (_p / "_bootstrap.py").is_file():
        _vc = _p
        break
else:
    raise RuntimeError(f"validation_code root not found above {_Path(__file__).resolve()}")
if str(_vc) not in sys.path:
    sys.path.insert(0, str(_vc))
from _bootstrap import bootstrap_paths
THRESHOLD_ROOT, DATA_ROOT, CODE_DIR, REPO_ROOT, ARTIFACT_VERIFY_ROOT = bootstrap_paths()
_SCRIPT_DIR = THRESHOLD_ROOT
import csv
import hashlib
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


PROFILES: List[str] = ["SN_Driveasrpy20_2min_70"]
DATASET_IDS: List[str] = ["s28"]
#DATASET_IDS: List[str] = ["07","09","11","71","84","95"]
CONTINUE_ON_ERROR = True

WINDOW_SEC = 10.0

#1asr20_2min_70
# 解混之后排除的时间段（秒，原始时间轴）。空列表 = 不排除。
# 例：[(0.0, 120.0)] 表示前 120s 不参与源 MS 统计（ICA 仍用全长）。
# #07
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,32), (122,133), (445,447), (794,795), (983,1039), (1149,1185)]
# # #09
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,120), (141,160), (749,853), (1027,1056), (1545,1554)]
# # #11
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,128), (432,566), (735,763), (1095,1110), (1513,1568)]
# # #71
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,120), (122,170), (430,574), (796,885), (1362,1423)]
# # #84
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,242), (593,740), (1194,1221)]
# #95
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,249), (396,417), (726,746), (1071,1080), (1253,1331)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = []

#1asrpy20_2min_70
#07 Cz
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,39), (122,131), (445,518), (794,824), (983,988), (1149,1185)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,131)]
#09 O1
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,111), (141,192), (749,751), (1027,1028), (1545,1573)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,111)]
#11
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,184), (432,494), (735,753), (1095,1110), (1513,1536)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,184)]
#71
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,8), (122,150), (430,450), (796,808), (1362,1379)]
# EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,150), (430,450), (796,808), (1362,1379)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,150)]

#84 M1 M2
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,251), (593,613), (1194,1224)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,251)]
#95
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,162), (396,436), (726,781), (1071,1094), (1253,1254)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,162)]



#1asrpy100_2min_70
#07 Cz
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,32), (122,133), (445,447), (794,795), (983,1005), (1149,1150)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,133), (445,447), (794,795), (983,1005), (1149,1150)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,133)]
#09 O1
# EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,69), (141,159), (749,751), (1027,1040), (1545,1622)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,159)]

# #11
# EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,188), (432,473), (735,764), (1095,1128), (1513,1533)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,188)]

# #71
# #EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,72), (122,131), (430,492), (796,801), (1362,1392)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,131)]
# #84 M1 M2
# EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,81), (593,639), (1194,1262)]
#EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,81)]
# #95
# EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,92), (123,206), (396,439), (726,791), (1071,1082), (1253,1273)]
EXCLUDE_TIME_RANGES_S: List[Tuple[float, float]] = [(0,120)]

# 通道筛选（三阶段对齐之后、ICA 之前）。名称与 npz「channels」一致，不区分大小写。
# INCLUDE_CHANNEL_NAMES 非空：仅保留这些通道（在数据中的出现顺序）。
# EXCLUDE_CHANNEL_NAMES：从当前候选集合中再剔除。
INCLUDE_CHANNEL_NAMES: Optional[List[str]] = None
EXCLUDE_CHANNEL_NAMES: List[str] = []

EXPERIMENT_TAG = "1segment_exclude"
ICA_SEED = 42
ICA_MAX_ITER = 500
ICA_METHOD = "infomax"
IIR_BEFORE_ICA = False

EEG_NPZ_FILENAME_SUFFIX: str = ""

MIN_MS_IIR_REL_FLOOR = 1e-7
MIN_MS_ASR_REL_FLOOR = 1e-7
MAX_PCT_VS_IIR_FOR_STATS: Optional[float] = 150.0

PER_IC_PLOT_YMAX_PCT: Optional[float] = 118.0
PER_IC_ANNOTATE_OVERFLOW: bool = True
PER_IC_ANNOTATE_DROPS: bool = False

# 每 IC 三根柱（IIR/ASR/ORICA MS）纵轴：True=log10(MS)，False=线性 MS
USE_LOG_Y_FOR_MS_TRIBAR: bool = True
# 与上项配合：True 时对纵轴加常数 shift=max(0, floor−min(log10 MS))，使刻度主要在正数区间（等价 log10(MS×10^shift)）
MS_TRIBAR_LOG_SHIFT_TO_POSITIVE: bool = True
MS_TRIBAR_LOG_Y_FLOOR: float = 0.2
IC_TOPOMAP_NCOLS: int = 8

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
_ICLABEL_NORMALIZED_ALIASES: Dict[str, str] = {
    "muscle_artifact": "muscle",
    "eye_blink": "eye",
    "heart_beat": "heart",
}


def _effective_exclude_ranges(
    ranges: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in ranges:
        if float(b) > float(a):
            out.append((float(a), float(b)))
    return out


def _ch_norm(s: Any) -> str:
    return str(s).strip().lower()


def _sanitize_ch_for_path_token(name: str, *, max_len: int = 28) -> str:
    s = str(name).strip()
    for c in '<>:"/\\|?* \t\n\r':
        s = s.replace(c, "_")
    if not s:
        s = "ch"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _filename_token_for_removed_channels(removed: Sequence[str]) -> str:
    """写入输出子目录名；过长时用短前缀 + 数量 + md5，避免 Windows 路径过长。"""
    if not removed:
        return ""
    safe = [_sanitize_ch_for_path_token(str(nm)) for nm in removed]
    body = "+".join(safe)
    max_total = 96
    if len(body) <= max_total:
        return f"dropped_{body}"
    h = hashlib.md5("||".join(str(x) for x in removed).encode("utf-8")).hexdigest()[:10]
    head = "+".join(safe[:8])
    if len(head) > 56:
        head = head[:56]
    return f"dropped_{head}_n{len(removed)}_{h}"


def apply_channel_include_exclude(
    xi: np.ndarray,
    xa: np.ndarray,
    xo: np.ndarray,
    ch_names: Sequence[str],
    *,
    include_only: Optional[Sequence[str]],
    exclude: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    按 INCLUDE / EXCLUDE 裁剪通道维。

    返回:
        xi2, xa2, xo2, ch_kept, removed_ordered（相对原始 ch_names 顺序）,
        missing_include（INCLUDE 中有但数据中未出现的规范化名，用于告警）
    """
    n = len(ch_names)
    if xi.shape[0] != n or xa.shape[0] != n or xo.shape[0] != n:
        raise ValueError(
            f"通道维与 ch_names 不一致: xi{xi.shape} xa{xa.shape} xo{xo.shape} n_ch={n}"
        )

    all_idx = list(range(n))
    inc_set: Optional[set] = None
    if include_only:
        inc_set = {_ch_norm(x) for x in include_only if str(x).strip()}
    if inc_set:
        keep_idx = [i for i in all_idx if _ch_norm(ch_names[i]) in inc_set]
        found_norm = {_ch_norm(ch_names[i]) for i in keep_idx}
        missing_include = sorted(inc_set - found_norm)
    else:
        keep_idx = all_idx
        missing_include = []

    exc_set = {_ch_norm(x) for x in exclude if str(x).strip()}
    if exc_set:
        keep_idx = [i for i in keep_idx if _ch_norm(ch_names[i]) not in exc_set]

    kset = set(keep_idx)
    removed_ordered = [str(ch_names[i]) for i in all_idx if i not in kset]

    if len(keep_idx) < 2:
        raise RuntimeError(
            f"通道筛选后仅剩 {len(keep_idx)} 路，ICA 至少需要 2 路。"
            f" removed（前 30 个）={removed_ordered[:30]}"
        )

    k = np.asarray(keep_idx, dtype=np.int64)
    ch_kept = [str(ch_names[i]) for i in keep_idx]
    return (
        np.asarray(xi[k, :], dtype=np.float64).copy(),
        np.asarray(xa[k, :], dtype=np.float64).copy(),
        np.asarray(xo[k, :], dtype=np.float64).copy(),
        ch_kept,
        removed_ordered,
        missing_include,
    )


def _output_subdir_name(*, channel_selection_slug: str = "") -> str:
    w = int(round(WINDOW_SEC)) if WINDOW_SEC > 0 else 0
    eff = _effective_exclude_ranges(EXCLUDE_TIME_RANGES_S)
    if not eff:
        base = f"{EXPERIMENT_TAG}__window_{w}_remove_none"
    else:
        parts = [EXPERIMENT_TAG, f"window_{w}_remove"]
        for a, _b in eff:
            parts.append(str(int(a)))
        base = "_".join(parts)
    if channel_selection_slug:
        return f"{base}__{channel_selection_slug}"
    return base


def _sample_mask_exclude(
    n_tot: int, sfreq: float, ranges: Sequence[Tuple[float, float]]
) -> np.ndarray:
    """True = 参与能量统计；False = EXCLUDE 区间内样本。"""
    mask = np.ones(n_tot, dtype=bool)
    for a, b in ranges:
        if b <= a:
            continue
        i0 = int(np.floor(a * sfreq))
        i1 = int(np.ceil(b * sfreq))
        i0 = max(0, min(n_tot, i0))
        i1 = max(0, min(n_tot, i1))
        if i1 > i0:
            mask[i0:i1] = False
    return mask


def _savefig_longpath_safe(fig: Any, path: Path, **kwargs: Any) -> None:
    """
    兼容 Windows 长路径的 savefig。
    优先按普通 Path 保存；若触发 FileNotFoundError，再用 `\\\\?\\` 前缀重试。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, **kwargs)
        return
    except FileNotFoundError:
        if os.name != "nt":
            raise
    p = str(path.resolve())
    if p.startswith("\\\\?\\"):
        long_p = p
    elif p.startswith("\\\\"):
        long_p = "\\\\?\\UNC\\" + p[2:]
    else:
        long_p = "\\\\?\\" + p
    fig.savefig(long_p, **kwargs)


def window_overlaps_exclude(
    t0: float, t1: float, ranges: Sequence[Tuple[float, float]]
) -> bool:
    for a, b in ranges:
        if b <= a:
            continue
        if not (t1 <= a or t0 >= b):
            return True
    return False


def _per_window_start_seconds(n_win: int, win_samp: int, sfreq: float) -> np.ndarray:
    """与分窗循环一致：第 wi 窗起点 = wi * win_samp / sfreq（秒）。"""
    return np.arange(n_win, dtype=np.float64) * (win_samp / sfreq)


def _per_window_excluded_mask(
    n_win: int, win_samp: int, sfreq: float, eff_exclude: Sequence[Tuple[float, float]]
) -> np.ndarray:
    if not eff_exclude:
        return np.zeros(n_win, dtype=bool)
    out = np.zeros(n_win, dtype=bool)
    for wi in range(n_win):
        i0 = wi * win_samp
        i1 = i0 + win_samp
        t0, t1 = i0 / sfreq, i1 / sfreq
        if window_overlaps_exclude(t0, t1, eff_exclude):
            out[wi] = True
    return out


def _draw_exclude_time_spans_on_axis(
    ax: Any, eff_exclude: Sequence[Tuple[float, float]], *, zorder: float = 0.5
) -> None:
    """x 为录音时间（秒）时，标出 EXCLUDE 区间（不重排时间轴）。"""
    for a, b in eff_exclude:
        lo, hi = float(a), float(b)
        if hi > lo:
            ax.axvspan(
                lo,
                hi,
                facecolor="0.45",
                alpha=0.22,
                hatch="///",
                edgecolor="0.35",
                linewidth=0.0,
                zorder=zorder,
            )


def _decode_ch(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def load_eeg_npz(npz_path: Path) -> Optional[Dict[str, Any]]:
    try:
        z = np.load(npz_path, allow_pickle=True)
        if "data" not in z:
            print(f"[ERROR] {npz_path.name}: missing 'data'")
            return None
        eeg = np.asarray(z["data"], dtype=np.float64)
        if eeg.ndim != 2:
            return None
        if eeg.shape[1] <= 256 and eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T
        sr = z.get("sampling_rate", z.get("srate", 500))
        if isinstance(sr, np.ndarray):
            sr = float(sr.item())
        if "channels" in z:
            ch_names = [_decode_ch(x) for x in np.asarray(z["channels"]).ravel()]
        else:
            ch_names = [f"EEG{i+1:03d}" for i in range(eeg.shape[0])]
        return {"data": eeg, "sampling_rate": float(sr), "ch_names": ch_names}
    except Exception as e:
        print(f"[ERROR] load {npz_path}: {e}")
        return None


def align_three_stages(
    eeg_iir: Dict[str, Any],
    eeg_asr: Dict[str, Any],
    eeg_orica: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[str], int]:
    sfreq = float(eeg_iir["sampling_rate"])
    for name, d in ("asr", eeg_asr), ("orica", eeg_orica):
        if abs(float(d["sampling_rate"]) - sfreq) > 1e-3:
            print(
                f"[WARN] {name} 采样率 {d['sampling_rate']} != IIR {sfreq}，仍按最小长度对齐。"
            )
    n_ch = min(
        eeg_iir["data"].shape[0],
        eeg_asr["data"].shape[0],
        eeg_orica["data"].shape[0],
    )
    n_tot = min(
        eeg_iir["data"].shape[1],
        eeg_asr["data"].shape[1],
        eeg_orica["data"].shape[1],
    )
    ch_iir = list(eeg_iir["ch_names"])[:n_ch]
    ch_asr = list(eeg_asr["ch_names"])[:n_ch]
    ch_ori = list(eeg_orica["ch_names"])[:n_ch]
    if ch_iir != ch_asr or ch_iir != ch_ori:
        print("[WARN] 三阶段通道名不完全一致，以 IIR 前 n_ch 个通道名为准。")
    xi = np.asarray(eeg_iir["data"], dtype=np.float64)[:n_ch, :n_tot].copy()
    xa = np.asarray(eeg_asr["data"], dtype=np.float64)[:n_ch, :n_tot].copy()
    xo = np.asarray(eeg_orica["data"], dtype=np.float64)[:n_ch, :n_tot].copy()
    return xi, xa, xo, sfreq, ch_iir, n_tot


def _normalize_label(s: Any) -> str:
    t = s.decode("utf-8", errors="ignore") if isinstance(s, bytes) else str(s)
    return t.strip().lower().replace(" ", "_")


def _canonical_icalabel_label(normalized: str) -> str:
    return _ICLABEL_NORMALIZED_ALIASES.get(normalized, normalized)


def _extract_pred_labels(labels_out: Any, n_ic: int) -> List[str]:
    lab = None
    if hasattr(labels_out, "labels"):
        lab = labels_out.labels
    elif isinstance(labels_out, dict):
        lab = labels_out.get("labels")
    if lab is None:
        return ["other"] * n_ic
    arr = np.asarray(lab).ravel()
    raw = [_normalize_label(arr[i]) for i in range(min(n_ic, len(arr)))]
    raw += ["other"] * max(0, n_ic - len(arr))
    return [_canonical_icalabel_label(x) for x in raw]


def _extract_proba_from_labels_out(
    labels_out: Any, n_ic: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
        return (None, p) if p.size == n_ic else (None, None)
    if p.ndim != 2:
        return None, None
    if p.shape[1] == n_ic and p.shape[0] != n_ic:
        p = p.T
    if p.shape[0] != n_ic:
        return None, None
    return p, None


def predicted_class_prob(
    labels_out: Any, pred_labels: Sequence[str], n_ic: int
) -> np.ndarray:
    P, p1d = _extract_proba_from_labels_out(labels_out, n_ic)
    n_cls = len(ICLABEL_CLASSES)
    class_to_idx = {c: j for j, c in enumerate(ICLABEL_CLASSES)}
    out = np.ones(n_ic, dtype=np.float64)
    if P is not None and P.ndim == 2:
        ncol = min(int(P.shape[1]), n_cls)
        pad = np.zeros((n_ic, n_cls), dtype=np.float64)
        pad[:, :ncol] = np.clip(P[:, :ncol], 0.0, 1.0)
        for i in range(n_ic):
            lbl = pred_labels[i] if i < len(pred_labels) else "other"
            j = class_to_idx.get(lbl, class_to_idx["other"])
            out[i] = float(pad[i, j])
        return np.clip(out, 0.0, 1.0)
    if p1d is not None and p1d.size >= n_ic:
        return np.clip(np.asarray(p1d[:n_ic], dtype=np.float64), 0.0, 1.0)
    return out


def _raw_from_array(
    data_ch_by_time: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    apply_iir: bool,
) -> Any:
    import mne

    n_ch, _ = data_ch_by_time.shape
    ch_list_raw = (
        list(ch_names) if len(ch_names) == n_ch else [f"EEG{i+1:03d}" for i in range(n_ch)]
    )
    # Normalize channel spellings (e.g. FP1/FZ/CZ -> Fp1/Fz/Cz) so standard_1020
    # can provide positions for ICLabel topography features.
    try:
        std = mne.channels.make_standard_montage("standard_1020")
        lookup = {str(name).upper().replace(" ", ""): str(name) for name in std.ch_names}
        ch_list = []
        n_changed = 0
        for ch in ch_list_raw:
            src = str(ch).strip()
            dst = lookup.get(src.upper().replace(" ", ""), src)
            if dst != src:
                n_changed += 1
            ch_list.append(dst)
        if n_changed > 0:
            print(
                f"[INFO] 通道名标准化: {n_changed}/{len(ch_list)} "
                "（如 FP1->Fp1, FZ->Fz）"
            )
    except Exception:
        ch_list = ch_list_raw
    x = np.asarray(data_ch_by_time, dtype=np.float64)
    if apply_iir:
        x = mne.filter.filter_data(x, sfreq, l_freq=1.0, h_freq=50.0, verbose=False)
    info = mne.create_info(ch_list, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(x, info)
    try:
        raw.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass
    return raw


def _fit_ica_on_raw(raw: Any, random_state: int, ica_max_iter: int) -> Tuple[Any, Optional[str]]:
    from mne.preprocessing import ICA

    # 与 ica_source_energy_analysis_correct.py / test.py 口径一致：
    # infomax + extended，且 n_components = n_ch - 1
    n_ch = len(raw.ch_names)
    #n_comp = max(1, min(n_ch - 1, n_ch))
    n_comp = len(raw.ch_names)
    try:
        ica = ICA(
            n_components=n_comp,
            method="infomax",
            random_state=random_state,
            max_iter=ica_max_iter,
            fit_params={"extended": True},
        )
        ica.fit(raw, verbose=False, reject_by_annotation=False)
    except Exception as e:
        return None, f"ICA fit failed: {e}"
    return ica, None


def mean_square_per_ic(mat: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is None:
        return np.mean(mat * mat, axis=1).astype(np.float64)
    mct = float(np.sum(mask))
    if mct <= 0.0:
        return np.full(mat.shape[0], np.nan, dtype=np.float64)
    x = mat[:, mask]
    return np.mean(x * x, axis=1).astype(np.float64)


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    eps = 1e-20
    d = np.where(np.isfinite(den), den, np.nan)
    n = np.where(np.isfinite(num), num, np.nan)
    out = np.full_like(n, np.nan, dtype=np.float64)
    ok = np.isfinite(n) & np.isfinite(d) & (np.abs(d) > eps)
    out[ok] = n[ok] / d[ok]
    return out


def pct_vs_iir(ms_stage: np.ndarray, ms_iir: np.ndarray) -> np.ndarray:
    return safe_ratio(ms_stage, ms_iir) * 100.0


def reliable_ms_iir_mask(ms_iir: np.ndarray, rel_floor: float) -> np.ndarray:
    ms_iir = np.asarray(ms_iir, dtype=np.float64).ravel()
    max_iir = float(np.nanmax(ms_iir))
    if not np.isfinite(max_iir) or max_iir <= 0:
        return np.zeros(len(ms_iir), dtype=bool)
    floor = max(max_iir * float(rel_floor), 1e-30)
    return ms_iir >= floor


def mask_pct_by_ms_iir_floor(
    pct: np.ndarray, ms_iir: np.ndarray, rel_floor: float
) -> np.ndarray:
    ok = reliable_ms_iir_mask(ms_iir, rel_floor)
    p = np.asarray(pct, dtype=np.float64)
    return np.where(ok, p, np.nan)


def reliable_ms_den_mask(ms: np.ndarray, rel_floor: float) -> np.ndarray:
    m = np.asarray(ms, dtype=np.float64).ravel()
    mx = float(np.nanmax(m))
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros(len(m), dtype=bool)
    floor = max(mx * float(rel_floor), 1e-30)
    return m >= floor


def pct_orica_vs_asr(ms_orica: np.ndarray, ms_asr: np.ndarray, rel_floor: float) -> np.ndarray:
    ok = reliable_ms_den_mask(ms_asr, rel_floor)
    r = safe_ratio(ms_orica, ms_asr) * 100.0
    return np.where(ok, r, np.nan)


def compute_reduction_summary_rows(
    pred_labels: Sequence[str],
    pct_asr: np.ndarray,
    pct_orica: np.ndarray,
    ms_asr: np.ndarray,
    ms_ori: np.ndarray,
    *,
    ok_for_stats: Optional[np.ndarray] = None,
) -> List[List[Any]]:
    pl = [str(pred_labels[i]) for i in range(len(pred_labels))]
    pct_a = np.asarray(pct_asr, dtype=np.float64)
    pct_o = np.asarray(pct_orica, dtype=np.float64)
    p_oa = pct_orica_vs_asr(ms_ori, ms_asr, MIN_MS_ASR_REL_FLOOR)
    if ok_for_stats is None:
        ok = np.ones(len(pl), dtype=bool)
    else:
        ok = np.asarray(ok_for_stats, dtype=bool)
        if ok.shape[0] != len(pl):
            raise ValueError("ok_for_stats 长度与 pred_labels 不一致")

    drop_asr_iir = 100.0 - pct_a
    drop_ori_iir = 100.0 - pct_o
    drop_ori_asr = 100.0 - p_oa

    def _msd_pick(x: np.ndarray, pick: List[int]) -> Tuple[float, float]:
        v = np.asarray([x[i] for i in pick], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (float("nan"), float("nan"))
        return (float(np.mean(v)), float(np.std(v, ddof=0)))

    def _stats(idx: List[int]) -> Tuple[int, float, float, float, float, float, float]:
        if not idx:
            return (0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        n = len(idx)
        m1, s1 = _msd_pick(drop_asr_iir, idx)
        m2, s2 = _msd_pick(drop_ori_iir, idx)
        idx_oa = [i for i in idx if np.isfinite(p_oa[i])]
        m3, s3 = _msd_pick(drop_ori_asr, idx_oa)
        return (n, m1, s1, m2, s2, m3, s3)

    rows: List[List[Any]] = []
    group_order: List[Tuple[str, Callable[[str], bool]]] = [
        ("brain", lambda lb: lb == "brain"),
        ("eye", lambda lb: lb == "eye"),
        ("muscle", lambda lb: lb == "muscle"),
        ("heart", lambda lb: lb == "heart"),
        ("line_noise", lambda lb: lb == "line_noise"),
        ("channel_noise", lambda lb: lb == "channel_noise"),
        ("other", lambda lb: lb == "other"),
        ("artifact_all", lambda lb: lb in ARTIFACT_CLASSES),
        ("all_ic", lambda lb: True),
    ]
    for name, pred in group_order:
        idx = [i for i, lb in enumerate(pl) if pred(lb) and ok[i]]
        n, m1, s1, m2, s2, m3, s3 = _stats(idx)
        rows.append([name, n, m1, s1, m2, s2, m3, s3])
    return rows


def save_reduction_summary_csv(path: Path, rows: List[List[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "# Units: percentage points (pp). mean_drop = 100 - relative_pct_vs_baseline.",
            ]
        )
        w.writerow(
            [
                "group",
                "n_ic",
                "mean_drop_asr_vs_iir",
                "std_drop_asr_vs_iir",
                "mean_drop_orica_vs_iir",
                "std_drop_orica_vs_iir",
                "mean_drop_orica_vs_asr",
                "std_drop_orica_vs_asr",
            ]
        )
        for r in rows:
            w.writerow(r)


def reduction_summary_for_plot_caption(rows: List[List[Any]]) -> str:
    d = {str(r[0]): r for r in rows}

    def fmt3(r: List[Any]):
        n = int(r[1])
        if n <= 0:
            return None

        def f(x: Any) -> str:
            try:
                v = float(x)
            except (TypeError, ValueError):
                return "—"
            return f"{v:.1f}" if np.isfinite(v) else "—"

        return (n, f(r[2]), f(r[4]), f(r[6]))

    lines: List[str] = []
    if "artifact_all" in d:
        t = fmt3(d["artifact_all"])
        if t:
            n, a, o, oa = t
            lines.append(
                f"All artifacts (n={n}): mean drop ASR/IIR {a} pp, ORICA/IIR {o} pp, ORICA/ASR {oa} pp"
            )
    if "brain" in d:
        t = fmt3(d["brain"])
        if t:
            n, a, o, oa = t
            lines.append(
                f"brain (n={n}): mean drop ASR/IIR {a} pp, ORICA/IIR {o} pp, ORICA/ASR {oa} pp"
            )
    parts: List[str] = []
    for key in ("eye", "muscle", "heart", "line_noise", "channel_noise"):
        if key not in d:
            continue
        t = fmt3(d[key])
        if t:
            n, a, o, oa = t
            parts.append(f"{key} (n={n}): {a} / {o} / {oa}")
    if parts:
        lines.append(
            "Per-artifact class means (pp; ASR/IIR | ORICA/IIR | ORICA/ASR): "
            + " | ".join(parts)
        )
    return "\n".join(lines)


def compute_power_weighted_reduction_rows(
    pred_labels: Sequence[str],
    pct_asr: np.ndarray,
    pct_orica: np.ndarray,
    ms_iir: np.ndarray,
    ms_asr: np.ndarray,
    ms_ori: np.ndarray,
    *,
    ok_for_stats: Optional[np.ndarray] = None,
) -> List[List[Any]]:
    """按 IIR 源 MS 加权：类内 sum(w_i * drop_i) / sum(w_i)，w_i = ms_iir[i]。"""
    pl = [str(pred_labels[i]) for i in range(len(pred_labels))]
    pct_a = np.asarray(pct_asr, dtype=np.float64)
    pct_o = np.asarray(pct_orica, dtype=np.float64)
    ms_i = np.asarray(ms_iir, dtype=np.float64)
    ms_a = np.asarray(ms_asr, dtype=np.float64)
    ms_o = np.asarray(ms_ori, dtype=np.float64)
    p_oa = pct_orica_vs_asr(ms_o, ms_a, MIN_MS_ASR_REL_FLOOR)
    if ok_for_stats is None:
        ok = np.ones(len(pl), dtype=bool)
    else:
        ok = np.asarray(ok_for_stats, dtype=bool)
        if ok.shape[0] != len(pl):
            raise ValueError("ok_for_stats 长度与 pred_labels 不一致")

    drop_ai = 100.0 - pct_a
    drop_oi = 100.0 - pct_o
    drop_oa = 100.0 - p_oa
    eps_w = 1e-30

    def _wmean_for_indices(
        pick: List[int], drop: np.ndarray
    ) -> Tuple[float, float, int]:
        wsum = 0.0
        acc = 0.0
        n_used = 0
        for i in pick:
            if not ok[i]:
                continue
            w = float(ms_i[i])
            if not np.isfinite(w) or w <= eps_w:
                continue
            d = float(drop[i])
            if not np.isfinite(d):
                continue
            wsum += w
            acc += w * d
            n_used += 1
        if wsum <= eps_w or n_used == 0:
            return float("nan"), float("nan"), 0
        return acc / wsum, wsum, n_used

    group_order: List[Tuple[str, Callable[[str], bool]]] = [
        ("brain", lambda lb: lb == "brain"),
        ("eye", lambda lb: lb == "eye"),
        ("muscle", lambda lb: lb == "muscle"),
        ("heart", lambda lb: lb == "heart"),
        ("line_noise", lambda lb: lb == "line_noise"),
        ("channel_noise", lambda lb: lb == "channel_noise"),
        ("other", lambda lb: lb == "other"),
        ("artifact_all", lambda lb: lb in ARTIFACT_CLASSES),
        ("all_ic", lambda lb: True),
    ]
    rows: List[List[Any]] = []
    for name, pred in group_order:
        idx = [i for i, lb in enumerate(pl) if pred(lb) and ok[i]]
        n_group = len(idx)
        wai, sw_ai, n_w_ai = _wmean_for_indices(idx, drop_ai)
        woi, sw_oi, n_w_oi = _wmean_for_indices(idx, drop_oi)
        idx_oa = [
            i
            for i in idx
            if ok[i]
            and np.isfinite(drop_oa[i])
            and np.isfinite(ms_i[i])
            and float(ms_i[i]) > eps_w
        ]
        woa, sw_oa, n_w_oa = _wmean_for_indices(idx_oa, drop_oa)
        rows.append(
            [
                name,
                n_group,
                n_w_ai,
                sw_ai,
                wai,
                woi,
                woa,
                n_w_oa,
                sw_oa,
            ]
        )
    return rows


def save_power_weighted_reduction_csv(path: Path, rows: List[List[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "# Energy-weighted mean drop (pp): sum(ms_iir_i * drop_i) / sum(ms_iir_i); "
                "drop = 100 - pct_vs_baseline; weights = IIR source MS per IC.",
            ]
        )
        w.writerow(
            [
                "group",
                "n_ic_in_group",
                "n_ic_used_weight_asr_iir",
                "sum_ms_iir_weight_asr_iir",
                "wmean_drop_asr_vs_iir",
                "wmean_drop_orica_vs_iir",
                "wmean_drop_orica_vs_asr",
                "n_ic_used_weight_orica_asr",
                "sum_ms_iir_weight_orica_asr",
            ]
        )
        for r in rows:
            w.writerow(r)


def _fmt_pp(v: float) -> str:
    return f"{v:.1f}" if np.isfinite(v) else "—"


def reduction_summary_power_weighted_caption(rows: List[List[Any]]) -> str:
    """从 compute_power_weighted_reduction_rows 结果生成标题附加行（英文）。"""
    d = {str(r[0]): r for r in rows}
    lines: List[str] = []
    if "artifact_all" in d:
        r = d["artifact_all"]
        n_g = int(r[1])
        if n_g > 0:
            lines.append(
                "Energy-wtd (IIR MS): "
                f"All artifacts (n={n_g}): ASR/IIR {_fmt_pp(float(r[4]))} pp, "
                f"ORICA/IIR {_fmt_pp(float(r[5]))} pp, ORICA/ASR {_fmt_pp(float(r[6]))} pp"
            )
    if "brain" in d:
        r = d["brain"]
        n_g = int(r[1])
        if n_g > 0:
            lines.append(
                "Energy-wtd (IIR MS): "
                f"brain (n={n_g}): ASR/IIR {_fmt_pp(float(r[4]))} pp, "
                f"ORICA/IIR {_fmt_pp(float(r[5]))} pp, ORICA/ASR {_fmt_pp(float(r[6]))} pp"
            )
    parts: List[str] = []
    for key in ("eye", "muscle", "heart", "line_noise", "channel_noise"):
        if key not in d:
            continue
        r = d[key]
        n_g = int(r[1])
        if n_g <= 0:
            continue
        parts.append(
            f"{key} (n={n_g}): {_fmt_pp(float(r[4]))} / {_fmt_pp(float(r[5]))} / {_fmt_pp(float(r[6]))}"
        )
    if parts:
        lines.append(
            "Energy-wtd per-artifact class (pp; ASR/IIR | ORICA/IIR | ORICA/ASR): "
            + " | ".join(parts)
        )
    return "\n".join(lines)


def plot_caption_equal_and_power_weighted(
    equal_rows: List[List[Any]],
    power_rows: List[List[Any]],
) -> str:
    a = reduction_summary_for_plot_caption(equal_rows)
    b = reduction_summary_power_weighted_caption(power_rows)
    if a and b:
        return a + "\n" + b
    return a or b


def _setup_matplotlib_font() -> None:
    import matplotlib.pyplot as plt

    # English-first figure text; CJK fonts as fallback for rare mixed labels
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Arial",
        "Helvetica",
        "Segoe UI",
        "Microsoft YaHei",
        "SimHei",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_per_ic_dual_bars_capped(
    ax: Any,
    x_ic: np.ndarray,
    pct_asr_ord: np.ndarray,
    pct_orica_ord: np.ndarray,
    ymax: float,
    annotate_overflow: bool,
    annotate_drops: bool,
) -> None:
    bw_ic = 0.35
    ya = np.asarray(pct_asr_ord, dtype=np.float64)
    yo = np.asarray(pct_orica_ord, dtype=np.float64)
    ya_d = np.where(np.isfinite(ya), np.clip(ya, 0.0, ymax), 0.0)
    yo_d = np.where(np.isfinite(yo), np.clip(yo, 0.0, ymax), 0.0)
    ax.bar(
        x_ic - bw_ic / 2,
        ya_d,
        width=bw_ic,
        label="ASR vs IIR (%)",
        color="#ff7f0e",
    )
    ax.bar(
        x_ic + bw_ic / 2,
        yo_d,
        width=bw_ic,
        label="ORICA vs IIR (%)",
        color="#9467bd",
    )
    ax.axhline(100.0, color="k", ls="--", lw=1.0, alpha=0.7)
    if annotate_overflow:
        dy = max(ymax * 0.018, 1.0)
        for j, xi in enumerate(x_ic):
            if np.isfinite(ya[j]) and ya[j] > ymax:
                ax.text(
                    float(xi) - bw_ic / 2,
                    ymax + dy,
                    f"{ya[j]:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    color="#cc5500",
                )
            if np.isfinite(yo[j]) and yo[j] > ymax:
                ax.text(
                    float(xi) + bw_ic / 2,
                    ymax + dy,
                    f"{yo[j]:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    color="#663399",
                )
    if annotate_drops:
        ymin = -max(0.22 * ymax, 18.0)
        for j, xi in enumerate(x_ic):
            bits: List[str] = []
            if np.isfinite(ya[j]):
                bits.append(f"A↓{100.0 - ya[j]:.0f}")
            else:
                bits.append("A:—")
            if np.isfinite(yo[j]):
                bits.append(f"O↓{100.0 - yo[j]:.0f}")
            else:
                bits.append("O:—")
            if np.isfinite(ya[j]) and np.isfinite(yo[j]):
                bits.append(f"Δ{ya[j] - yo[j]:+.0f}")
            txt = " ".join(bits)
            ax.text(
                float(xi),
                ymin * 0.55,
                txt,
                ha="center",
                va="top",
                fontsize=4.7,
                color="#333333",
                clip_on=False,
            )
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0.0, ymax)


def plot_per_ic_triple_ms_bars(
    ax: Any,
    x_ic: np.ndarray,
    ms_iir_ord: np.ndarray,
    ms_asr_ord: np.ndarray,
    ms_ori_ord: np.ndarray,
    *,
    use_log: bool,
) -> None:
    """每 IC 三根柱：IIR / ASR / ORICA 源均方。"""
    mi = np.asarray(ms_iir_ord, dtype=np.float64)
    ma = np.asarray(ms_asr_ord, dtype=np.float64)
    mo = np.asarray(ms_ori_ord, dtype=np.float64)
    eps = 1e-30
    if use_log:
        li = np.log10(np.maximum(mi, eps))
        la = np.log10(np.maximum(ma, eps))
        lo = np.log10(np.maximum(mo, eps))
        shift = 0.0
        if MS_TRIBAR_LOG_SHIFT_TO_POSITIVE:
            finite = np.concatenate([li.ravel(), la.ravel(), lo.ravel()])
            finite = finite[np.isfinite(finite)]
            if finite.size:
                vmin = float(np.min(finite))
                shift = max(0.0, float(MS_TRIBAR_LOG_Y_FLOOR) - vmin)
        plot_mi = li + shift
        plot_ma = la + shift
        plot_mo = lo + shift
        if shift > 1e-9:
            ylab = f"log10(mean square) + {shift:.2f}"
        else:
            ylab = "log10(mean square)"
    else:
        plot_mi, plot_ma, plot_mo = mi, ma, mo
        ylab = "Mean square (per IC source)"
    w = 0.22
    ax.bar(x_ic - w, plot_mi, width=w, label="IIR MS", color="#17becf")
    ax.bar(x_ic, plot_ma, width=w, label="ASR MS", color="#ff7f0e")
    ax.bar(x_ic + w, plot_mo, width=w, label="ORICA MS", color="#9467bd")
    ax.set_ylabel(ylab)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def ok_ic_for_pct_stats_cap(
    pct_asr: np.ndarray,
    pct_orica: np.ndarray,
    max_pct: Optional[float],
) -> np.ndarray:
    n = len(np.asarray(pct_asr))
    if max_pct is None:
        return np.ones(n, dtype=bool)
    pa = np.asarray(pct_asr, dtype=np.float64)
    po = np.asarray(pct_orica, dtype=np.float64)
    bad = np.zeros(n, dtype=bool)
    bad |= np.isfinite(pa) & (pa > float(max_pct))
    bad |= np.isfinite(po) & (po > float(max_pct))
    return ~bad


def count_class_in_stats(
    labels: Sequence[str], cls: str, ok: np.ndarray
) -> int:
    ok = np.asarray(ok, dtype=bool)
    return int(
        sum(1 for i, x in enumerate(labels) if i < len(ok) and x == cls and ok[i])
    )


def mean_for_class(values: np.ndarray, labels: Sequence[str], cls: str) -> float:
    idx = [i for i, x in enumerate(labels) if x == cls]
    if not idx:
        return float("nan")
    arr = np.asarray([values[i] for i in idx], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def count_for_class(labels: Sequence[str], cls: str) -> int:
    return int(sum(1 for x in labels if x == cls))


def _safe_text(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if np.isfinite(v) else "nan"


def _ic_group(lbl: str) -> str:
    if lbl in ARTIFACT_CLASSES:
        return "artifact"
    if lbl == "brain":
        return "brain"
    return "other"


def _ordered_ic_indices(labels: Sequence[str]) -> np.ndarray:
    order_rank = {"artifact": 0, "other": 1, "brain": 2}
    idx = list(range(len(labels)))
    idx.sort(key=lambda i: (order_rank.get(_ic_group(labels[i]), 1), i))
    return np.asarray(idx, dtype=int)


def _welch_log_psd_1d(sig: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (freq_hz, 10*log10(PSD))，Welch 功率谱密度。"""
    from scipy.signal import welch

    x = np.asarray(sig, dtype=np.float64).ravel()
    n = int(x.size)
    if n < 32:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    nperseg = int(min(max(256, n // 8), n, int(float(sfreq) * 8)))
    nperseg = max(64, nperseg)
    nover = max(0, nperseg // 2)
    f, pxx = welch(
        x,
        fs=float(sfreq),
        nperseg=nperseg,
        noverlap=nover,
        window="hann",
        scaling="density",
        detrend="linear",
    )
    log_psd = 10.0 * np.log10(np.maximum(np.asarray(pxx, dtype=np.float64), 1e-30))
    return np.asarray(f, dtype=np.float64), log_psd


def save_ic_topomap_mosaics_three_stages(
    ica_iir: Any,
    ica_asr: Any,
    ica_orica: Any,
    raw_iir: Any,
    raw_asr: Any,
    raw_orica: Any,
    out_dir: Path,
    *,
    sfreq: float,
) -> None:
    """按阶段导出 IC topomap + 源 PSD 拼图（风格对齐 correct.py）。"""
    import matplotlib.pyplot as plt
    from mne_icalabel import label_components
    from mne.viz import plot_topomap

    def _one(ica_x: Any, raw_x: Any, stage_key: str, stage_label: str) -> None:
        comps = np.asarray(ica_x.get_components(), dtype=np.float64)
        if comps.ndim != 2:
            print(f"[WARN] skip {stage_key}: bad components shape {comps.shape}")
            return
        src = np.asarray(ica_x.get_sources(raw_x).get_data(), dtype=np.float64)
        n_ic = int(src.shape[0])
        if int(comps.shape[1]) != n_ic:
            print(
                f"[WARN] skip {stage_key}: components n_ic={comps.shape[1]} != sources n_ic={n_ic}"
            )
            return
        labels_out = label_components(raw_x, ica_x, method="iclabel")
        pred_labels = _extract_pred_labels(labels_out, n_ic)
        pred_prob = predicted_class_prob(labels_out, pred_labels, n_ic)
        ic_order = _ordered_ic_indices(pred_labels)
        ncols = max(1, int(IC_TOPOMAP_NCOLS))
        nrows = int(math.ceil(n_ic / float(ncols)))

        fig = plt.figure(figsize=(ncols * 2.35, nrows * 3.55))
        outer = fig.add_gridspec(nrows, ncols, hspace=0.55, wspace=0.28)
        for r in range(nrows):
            for c in range(ncols):
                k = r * ncols + c
                if k >= n_ic:
                    ax_blank = fig.add_subplot(outer[r, c])
                    ax_blank.axis("off")
                    continue
                ic_i = int(ic_order[k])
                inner = outer[r, c].subgridspec(
                    3, 1, height_ratios=[2.0, 1.0, 1.0], hspace=0.12
                )
                ax_t = fig.add_subplot(inner[0, 0])
                ax_p1 = fig.add_subplot(inner[1, 0])
                ax_p2 = fig.add_subplot(inner[2, 0])

                try:
                    plot_topomap(
                        comps[:, ic_i],
                        raw_x.info,
                        axes=ax_t,
                        show=False,
                        sensors=True,
                        outlines="head",
                    )
                except Exception as e:
                    ax_t.text(0.5, 0.5, f"IC{ic_i}\n{e}", ha="center", va="center", fontsize=6)
                lbl = pred_labels[ic_i] if ic_i < len(pred_labels) else "other"
                ptxt = _safe_text(float(pred_prob[ic_i]), 2) if ic_i < len(pred_prob) else "nan"
                ax_t.set_title(f"IC{ic_i}\n{lbl}\np={ptxt}", fontsize=7)

                f_hz, log_psd = _welch_log_psd_1d(src[ic_i, :], sfreq)
                if f_hz.size > 0:
                    m1 = (f_hz >= 3.0) & (f_hz <= 40.0)
                    m2 = (f_hz >= 3.0) & (f_hz <= 80.0)
                    ax_p1.plot(f_hz[m1], log_psd[m1], color="#c62828", lw=0.85)
                    ax_p1.set_xlim(3.0, 40.0)
                    ax_p1.grid(True, alpha=0.28)
                    ax_p1.tick_params(axis="both", labelsize=5)
                    ax_p1.set_ylabel("10·log10 PSD", fontsize=5)
                    ax_p2.plot(f_hz[m2], log_psd[m2], color="#c62828", lw=0.85)
                    ax_p2.set_xlim(3.0, 80.0)
                    ax_p2.grid(True, alpha=0.28)
                    ax_p2.tick_params(axis="both", labelsize=5)
                    ax_p2.set_xlabel("Hz", fontsize=6)
                else:
                    ax_p1.text(0.5, 0.5, "PSD n/a", ha="center", va="center", fontsize=6)
                    ax_p1.axis("off")
                    ax_p2.axis("off")

        fig.suptitle(f"IC topomap + source PSD ({stage_label})", fontsize=10)
        plt.tight_layout(rect=[0, 0.01, 1, 0.96])
        outp = out_dir / f"ic_topomaps_mosaic__{stage_key}.png"
        fig.savefig(outp, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {outp}")

    _one(ica_iir, raw_iir, "iir", "IIR")
    _one(ica_asr, raw_asr, "asr", "ASR")
    _one(ica_orica, raw_orica, "orica", "ORICA")


def write_analysis_meta(
    path: Path,
    *,
    legacy_ref_dir: Path,
    output_run_dir: Path,
    n_ic: int,
    n_tot: int,
    sfreq: float,
    eeg_npz_filename_suffix: str,
    eff_exclude: List[Tuple[float, float]],
    n_keep_for_ms: int,
    channels_kept: Optional[Sequence[str]] = None,
    channels_removed: Optional[Sequence[str]] = None,
    include_requested: Optional[Sequence[str]] = None,
    exclude_requested: Optional[Sequence[str]] = None,
) -> None:
    ex_txt = eff_exclude if eff_exclude else "无"
    lines = [
        "ica_source_energy_analysis_correctly.py",
        "",
        "全长数据上 IIR 拟合 ICA，固定解混；IIR/ASR/ORICA 同索引 IC。不做 ICA 前 crop。",
        "类汇总「能量加权」：按各 IC 的 IIR 源 MS 对下降百分比 (pp) 加权平均；见 reduction_summary_power_weighted_*.csv。",
        f"EXCLUDE 区间（秒，仅用于解混后算源 MS 时按时间 mask；ICA/ICLabel 用全长）: {ex_txt}",
        f"解混后参与 MS 的采样数: {n_keep_for_ms} / {n_tot}",
        "",
        f"输入 npz 文件名后缀: {eeg_npz_filename_suffix!r}",
        f"参考路径（传统 same_save 树，可不存在）: {legacy_ref_dir}",
        f"本次输出目录: {output_run_dir}",
        f"n_ic={n_ic}, n_samples={n_tot}, sfreq={sfreq}",
        f"WINDOW_SEC: {WINDOW_SEC}",
        f"USE_LOG_Y_FOR_MS_TRIBAR: {USE_LOG_Y_FOR_MS_TRIBAR}",
        f"MS_TRIBAR_LOG_SHIFT_TO_POSITIVE: {MS_TRIBAR_LOG_SHIFT_TO_POSITIVE}",
        f"MS_TRIBAR_LOG_Y_FLOOR: {MS_TRIBAR_LOG_Y_FLOOR}",
        f"MIN_MS_IIR_REL_FLOOR: {MIN_MS_IIR_REL_FLOOR}",
        f"MAX_PCT_VS_IIR_FOR_STATS: {MAX_PCT_VS_IIR_FOR_STATS}",
    ]
    if include_requested is not None and len(list(include_requested)) > 0:
        lines.append(f"INCLUDE_CHANNEL_NAMES（请求）: {list(include_requested)}")
    if exclude_requested is not None and len(list(exclude_requested)) > 0:
        lines.append(f"EXCLUDE_CHANNEL_NAMES（请求）: {list(exclude_requested)}")
    if channels_removed is not None and len(list(channels_removed)) > 0:
        lines.append(f"实际未参与 ICA 的通道（相对对齐后全表）: {list(channels_removed)}")
    if channels_kept is not None:
        lines.append(f"参与 ICA 的通道（顺序）: {list(channels_kept)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def run_one(profile: str, dataset_id: str) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from mne_icalabel import label_components
    except ImportError as e:
        print("需要: pip install matplotlib mne mne-icalabel")
        raise SystemExit(1) from e

    _setup_matplotlib_font()

    ds = dataset_id
    suf = EEG_NPZ_FILENAME_SUFFIX
    eeg_npz_paths: Dict[str, Path] = {
        "iir": _SCRIPT_DIR / f"{profile}/b{ds}eeg_iir1{suf}.npz",
        "asr": _SCRIPT_DIR / f"{profile}/b{ds}eeg_asr1{suf}.npz",
        "orica": _SCRIPT_DIR / f"{profile}/b{ds}eeg_orica1{suf}.npz",
    }
    print(f"[INFO] 输入 EEG npz 后缀 suf={suf!r} → 例: {eeg_npz_paths['orica'].name}")
    legacy_result_dir = _SCRIPT_DIR / f"{profile}/ica_analysis_result_{ds}_2min_same_save"
    out_parent = (
        _SCRIPT_DIR / profile / "ica_source_analysis" / f"b{ds}"
    ).resolve()
    out_parent.mkdir(parents=True, exist_ok=True)

    loaded: Dict[str, Optional[Dict[str, Any]]] = {}
    for k in ("iir", "asr", "orica"):
        p = eeg_npz_paths[k]
        if not p.is_file():
            raise FileNotFoundError(f"缺少文件: {p}")
        loaded[k] = load_eeg_npz(p)
        if loaded[k] is None:
            raise RuntimeError(f"加载失败: {p}")

    xi, xa, xo, sfreq, ch_names, n_tot = align_three_stages(
        loaded["iir"],  # type: ignore[arg-type]
        loaded["asr"],  # type: ignore[arg-type]
        loaded["orica"],  # type: ignore[arg-type]
    )

    inc_req = INCLUDE_CHANNEL_NAMES or []
    exc_req = EXCLUDE_CHANNEL_NAMES or []
    all_norm = {_ch_norm(c) for c in ch_names}
    orphan_exc = sorted(
        {_ch_norm(x) for x in exc_req if str(x).strip()} - all_norm
    )
    if orphan_exc:
        print(
            f"[WARN] EXCLUDE_CHANNEL_NAMES 中有 {len(orphan_exc)} 个在数据中不存在: {orphan_exc[:15]}"
        )

    ch_slug = ""
    removed_for_meta: List[str] = []
    if inc_req or exc_req:
        xi, xa, xo, ch_names, removed_for_meta, miss_inc = apply_channel_include_exclude(
            xi,
            xa,
            xo,
            ch_names,
            include_only=inc_req if inc_req else None,
            exclude=exc_req,
        )
        if miss_inc:
            print(
                f"[WARN] INCLUDE_CHANNEL_NAMES 中有 {len(miss_inc)} 个在数据中未找到: {miss_inc}"
            )
        tok = _filename_token_for_removed_channels(removed_for_meta)
        if tok:
            ch_slug = tok
        print(
            f"[INFO] 通道筛选: ICA 使用 {len(ch_names)} 路；未参与: {len(removed_for_meta)} 路"
        )
        if removed_for_meta:
            print(f"       剔除: {removed_for_meta}")

    eff_exclude = _effective_exclude_ranges(EXCLUDE_TIME_RANGES_S)
    mask_keep = (
        _sample_mask_exclude(n_tot, sfreq, eff_exclude)
        if eff_exclude
        else np.ones(n_tot, dtype=bool)
    )
    n_keep_ms = int(np.sum(mask_keep))
    if eff_exclude:
        print(
            f"[INFO] EXCLUDE（仅解混后源能量）: {eff_exclude}；保留采样 {n_keep_ms}/{n_tot}"
        )

    out_dir = out_parent / _output_subdir_name(channel_selection_slug=ch_slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 输出: {out_dir}")

    raw_iir = _raw_from_array(xi, sfreq, ch_names, IIR_BEFORE_ICA)
    raw_asr = _raw_from_array(xa, sfreq, ch_names, IIR_BEFORE_ICA)
    raw_orica = _raw_from_array(xo, sfreq, ch_names, IIR_BEFORE_ICA)

    # 只在 IIR 上拟合 ICA；ASR/ORICA 统一使用这同一套解混矩阵
    ica_iir, err = _fit_ica_on_raw(raw_iir, ICA_SEED, ICA_MAX_ITER)
    if ica_iir is None:
        raise RuntimeError(err or "ICA failed")

    # 统计口径沿用 IIR 标签轴；源信号也统一来自同一套 IIR-ICA 投影
    labels_out = label_components(raw_iir, ica_iir, method="iclabel")
    s1 = ica_iir.get_sources(raw_iir).get_data()
    s2 = ica_iir.get_sources(raw_asr).get_data()
    s3 = ica_iir.get_sources(raw_orica).get_data()
    n_ic = int(s1.shape[0])
    pred_labels = _extract_pred_labels(labels_out, n_ic)
    pred_prob = predicted_class_prob(labels_out, pred_labels, n_ic)

    try:
        save_ic_topomap_mosaics_three_stages(
            ica_iir,
            ica_iir,
            ica_iir,
            raw_iir,
            raw_asr,
            raw_orica,
            out_dir,
            sfreq=float(sfreq),
        )
    except Exception as e:
        print(f"[WARN] IC topomap mosaic export failed: {e}")

    write_analysis_meta(
        out_dir / "analysis_meta.txt",
        legacy_ref_dir=legacy_result_dir,
        output_run_dir=out_dir,
        n_ic=n_ic,
        n_tot=n_tot,
        sfreq=sfreq,
        eeg_npz_filename_suffix=suf,
        eff_exclude=eff_exclude,
        n_keep_for_ms=n_keep_ms,
        channels_kept=ch_names,
        channels_removed=removed_for_meta if (inc_req or exc_req) else None,
        include_requested=inc_req if inc_req else None,
        exclude_requested=exc_req if exc_req else None,
    )

    ms_iir = mean_square_per_ic(s1, None)
    ms_asr = mean_square_per_ic(s2, None)
    ms_ori = mean_square_per_ic(s3, None)
    pct_a_raw = pct_vs_iir(ms_asr, ms_iir)
    pct_o_raw = pct_vs_iir(ms_ori, ms_iir)
    iir_ok = reliable_ms_iir_mask(ms_iir, MIN_MS_IIR_REL_FLOOR)
    pct_a = mask_pct_by_ms_iir_floor(pct_a_raw, ms_iir, MIN_MS_IIR_REL_FLOOR)
    pct_o = mask_pct_by_ms_iir_floor(pct_o_raw, ms_iir, MIN_MS_IIR_REL_FLOOR)
    n_bad_floor = int(np.sum(~iir_ok))
    if n_bad_floor:
        print(
            f"[WARN] {n_bad_floor} 个 IC 的 ms_iir 低于 max(ms_iir)×{MIN_MS_IIR_REL_FLOOR}，"
            f"相对 % 已置空；原始比值仍见 pct_*_raw 列。"
        )

    ok_pct_stats = ok_ic_for_pct_stats_cap(pct_a, pct_o, MAX_PCT_VS_IIR_FOR_STATS)
    n_skip_pct = int(np.sum(~ok_pct_stats))
    if MAX_PCT_VS_IIR_FOR_STATS is not None and n_skip_pct > 0:
        print(
            f"[INFO] {n_skip_pct} 个 IC 因 ASR 或 ORICA 相对 IIR %% > "
            f"{MAX_PCT_VS_IIR_FOR_STATS} 不参与类均值 / reduction / 部分图"
        )

    pct_a_stat = np.asarray(pct_a, dtype=np.float64).copy()
    pct_o_stat = np.asarray(pct_o, dtype=np.float64).copy()
    pct_a_stat[~ok_pct_stats] = np.nan
    pct_o_stat[~ok_pct_stats] = np.nan

    full_csv = out_dir / "ic_source_ms_and_pct_fullrecording_per_ic.csv"
    with full_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# ms_* 源均方；pct_* 相对 IIR；included_in_pct_stats 见 MAX_PCT_VS_IIR_FOR_STATS。"])
        w.writerow(
            [
                "ic_idx",
                "label_iir",
                "label_prob",
                "ms_iir",
                "ms_asr",
                "ms_orica",
                "iir_ok_for_ratio",
                "pct_asr_vs_iir",
                "pct_orica_vs_iir",
                "pct_asr_vs_iir_raw",
                "pct_orica_vs_iir_raw",
                "included_in_pct_stats",
            ]
        )
        for i in range(n_ic):
            w.writerow(
                [
                    i,
                    pred_labels[i] if i < len(pred_labels) else "other",
                    float(pred_prob[i]) if i < len(pred_prob) else "",
                    float(ms_iir[i]),
                    float(ms_asr[i]),
                    float(ms_ori[i]),
                    int(iir_ok[i]),
                    float(pct_a[i]) if np.isfinite(pct_a[i]) else "",
                    float(pct_o[i]) if np.isfinite(pct_o[i]) else "",
                    float(pct_a_raw[i]) if np.isfinite(pct_a_raw[i]) else "",
                    float(pct_o_raw[i]) if np.isfinite(pct_o_raw[i]) else "",
                    int(ok_pct_stats[i]),
                ]
            )
    print(f"[OK] {full_csv}")

    cls_rows: List[Tuple[str, int, int, float, float]] = []
    for cls in ICLABEL_CLASSES:
        cls_rows.append(
            (
                cls,
                count_for_class(pred_labels, cls),
                count_class_in_stats(pred_labels, cls, ok_pct_stats),
                mean_for_class(pct_a_stat, pred_labels, cls),
                mean_for_class(pct_o_stat, pred_labels, cls),
            )
        )
    sum_csv = out_dir / "ic_energy_pct_vs_iir_fullrecording_by_class.csv"
    with sum_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "class",
                "n_ic_labeled",
                "n_ic_in_stats",
                "mean_pct_asr",
                "mean_pct_orica",
            ]
        )
        for r in cls_rows:
            w.writerow(list(r))
    print(f"[OK] {sum_csv}")

    red_rows = compute_reduction_summary_rows(
        pred_labels,
        pct_a,
        pct_o,
        ms_asr,
        ms_ori,
        ok_for_stats=ok_pct_stats,
    )
    red_path = out_dir / "reduction_summary_by_icalabel_fullrecording.csv"
    save_reduction_summary_csv(red_path, red_rows)
    print(f"[OK] {red_path}")
    red_rows_pw = compute_power_weighted_reduction_rows(
        pred_labels,
        pct_a,
        pct_o,
        ms_iir,
        ms_asr,
        ms_ori,
        ok_for_stats=ok_pct_stats,
    )
    red_pw_path = out_dir / "reduction_summary_power_weighted_fullrecording.csv"
    save_power_weighted_reduction_csv(red_pw_path, red_rows_pw)
    print(f"[OK] {red_pw_path}")

    cls_names = [r[0] for r in cls_rows]
    arr_a = np.array([r[3] for r in cls_rows], dtype=np.float64)
    arr_o = np.array([r[4] for r in cls_rows], dtype=np.float64)
    xs = np.arange(len(cls_names))
    bw = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(xs - bw / 2, arr_a, width=bw, label="ASR vs IIR (%)", color="#ff7f0e")
    ax.bar(xs + bw / 2, arr_o, width=bw, label="ORICA vs IIR (%)", color="#9467bd")
    ax.axhline(100.0, color="k", ls="--", lw=1.0, alpha=0.65, label="IIR baseline (=100%)")
    ax.set_xticks(xs)
    ax.set_xticklabels(cls_names, rotation=22)
    ax.set_ylabel("Mean MS ratio × 100 (per class)")
    ax.set_title("Per-class mean energy vs IIR baseline (ICLabel on IIR-only ICA)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p_cls = out_dir / "energy_pct_fullrecording_by_class.png"
    fig.savefig(p_cls, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p_cls}")

    ic_order = _ordered_ic_indices(pred_labels)
    x_ic = np.arange(n_ic)
    labels_ic = [
        f"IC{i}\n{pred_labels[i]}\np={_safe_text(float(pred_prob[i]), 2)}"
        for i in ic_order
    ]

    fig_ms, ax_ms = plt.subplots(figsize=(max(12.0, n_ic * 0.5), 6.0))
    plot_per_ic_triple_ms_bars(
        ax_ms,
        x_ic,
        ms_iir[ic_order],
        ms_asr[ic_order],
        ms_ori[ic_order],
        use_log=USE_LOG_Y_FOR_MS_TRIBAR,
    )
    ax_ms.set_xticks(x_ic)
    ax_ms.set_xticklabels(labels_ic, fontsize=7)
    ax_ms.set_xlabel("IC (ordered)")
    ttl_ms = "Per-IC source mean square: IIR / ASR / ORICA"
    if USE_LOG_Y_FOR_MS_TRIBAR:
        ttl_ms += " (log10 scale"
        if MS_TRIBAR_LOG_SHIFT_TO_POSITIVE:
            ttl_ms += "; y shifted to mostly positive"
        ttl_ms += ")"
    ax_ms.set_title(ttl_ms)
    plt.tight_layout()
    p_ms = out_dir / "source_ms_fullrecording_per_ic_ordered.png"
    fig_ms.savefig(p_ms, dpi=160, bbox_inches="tight")
    plt.close(fig_ms)
    print(f"[OK] {p_ms}")

    cap = plot_caption_equal_and_power_weighted(red_rows, red_rows_pw)
    fig_h = 7.2 if cap else 6.0
    fig, ax = plt.subplots(figsize=(max(12.0, n_ic * 0.48), fig_h))
    ymax = float(PER_IC_PLOT_YMAX_PCT) if PER_IC_PLOT_YMAX_PCT is not None else 125.0
    plot_per_ic_dual_bars_capped(
        ax,
        x_ic,
        pct_a_stat[ic_order],
        pct_o_stat[ic_order],
        ymax,
        PER_IC_ANNOTATE_OVERFLOW,
        PER_IC_ANNOTATE_DROPS,
    )
    ax.set_xticks(x_ic)
    ax.set_xticklabels(labels_ic, fontsize=7)
    ax.set_ylabel("% of IIR MS (same IC index)")
    ttl_main = "Per-IC vs IIR (=100%); Y capped"
    if n_bad_floor:
        ttl_main += f" | low IIR-MS omitted: {n_bad_floor} IC"
    if MAX_PCT_VS_IIR_FOR_STATS is not None and n_skip_pct > 0:
        ttl_main += f" | pct>{MAX_PCT_VS_IIR_FOR_STATS}% omitted: {n_skip_pct} IC"
    if cap:
        ax.set_title(f"{ttl_main}\n{cap}", fontsize=9, linespacing=1.35)
    else:
        ax.set_title(ttl_main)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    p_ic = out_dir / "energy_pct_fullrecording_per_ic_ordered.png"
    fig.savefig(p_ic, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p_ic}")

    if WINDOW_SEC > 0:
        win_samp = int(round(WINDOW_SEC * sfreq))
        if win_samp <= 0:
            print("[WARN] WINDOW_SEC 过小，跳过分窗输出")
        else:
            n_win = n_tot // win_samp
            win_csv = out_dir / "ic_energy_pct_vs_iir_by_class_per_window.csv"
            per_ic_win_a = np.full((n_ic, n_win), np.nan, dtype=np.float64)
            per_ic_win_o = np.full((n_ic, n_win), np.nan, dtype=np.float64)
            with win_csv.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "window",
                        "t0_s",
                        "t1_s",
                        "class",
                        "n_ic",
                        "mean_pct_asr",
                        "mean_pct_orica",
                        "excluded_overlap",
                    ]
                )
                for wi in range(n_win):
                    i0 = wi * win_samp
                    i1 = i0 + win_samp
                    t0, t1 = i0 / sfreq, i1 / sfreq
                    excl = (
                        window_overlaps_exclude(t0, t1, eff_exclude)
                        if eff_exclude
                        else False
                    )
                    pw_i = np.mean(s1[:, i0:i1] ** 2, axis=1)
                    pw_a = np.mean(s2[:, i0:i1] ** 2, axis=1)
                    pw_o = np.mean(s3[:, i0:i1] ** 2, axis=1)
                    rw_a = mask_pct_by_ms_iir_floor(
                        pct_vs_iir(pw_a, pw_i), pw_i, MIN_MS_IIR_REL_FLOOR
                    )
                    rw_o = mask_pct_by_ms_iir_floor(
                        pct_vs_iir(pw_o, pw_i), pw_i, MIN_MS_IIR_REL_FLOOR
                    )
                    rw_as = np.asarray(rw_a, dtype=np.float64).copy()
                    rw_os = np.asarray(rw_o, dtype=np.float64).copy()
                    rw_as[~ok_pct_stats] = np.nan
                    rw_os[~ok_pct_stats] = np.nan
                    per_ic_win_a[:, wi] = rw_a
                    per_ic_win_o[:, wi] = rw_o
                    for cls in ICLABEL_CLASSES:
                        w.writerow(
                            [
                                wi,
                                f"{t0:.4f}",
                                f"{t1:.4f}",
                                cls,
                                count_for_class(pred_labels, cls),
                                mean_for_class(rw_as, pred_labels, cls),
                                mean_for_class(rw_os, pred_labels, cls),
                                int(excl),
                            ]
                        )
            print(f"[OK] {win_csv}")

            if MAX_PCT_VS_IIR_FOR_STATS is not None:
                per_ic_win_a[~ok_pct_stats, :] = np.nan
                per_ic_win_o[~ok_pct_stats, :] = np.nan

            data_a: Dict[str, List[float]] = {c: [] for c in ICLABEL_CLASSES}
            data_o: Dict[str, List[float]] = {c: [] for c in ICLABEL_CLASSES}
            for wi in range(n_win):
                i0 = wi * win_samp
                i1 = i0 + win_samp
                pw_i = np.mean(s1[:, i0:i1] ** 2, axis=1)
                pw_a = np.mean(s2[:, i0:i1] ** 2, axis=1)
                pw_o = np.mean(s3[:, i0:i1] ** 2, axis=1)
                rw_a = mask_pct_by_ms_iir_floor(
                    pct_vs_iir(pw_a, pw_i), pw_i, MIN_MS_IIR_REL_FLOOR
                )
                rw_o = mask_pct_by_ms_iir_floor(
                    pct_vs_iir(pw_o, pw_i), pw_i, MIN_MS_IIR_REL_FLOOR
                )
                rw_as = np.asarray(rw_a, dtype=np.float64).copy()
                rw_os = np.asarray(rw_o, dtype=np.float64).copy()
                rw_as[~ok_pct_stats] = np.nan
                rw_os[~ok_pct_stats] = np.nan
                for cls in ICLABEL_CLASSES:
                    data_a[cls].append(mean_for_class(rw_as, pred_labels, cls))
                    data_o[cls].append(mean_for_class(rw_os, pred_labels, cls))

            t_win_s = _per_window_start_seconds(n_win, win_samp, sfreq)
            win_excluded = _per_window_excluded_mask(n_win, win_samp, sfreq, eff_exclude)
            win_dur_s = win_samp / sfreq
            x_right = float(n_tot / sfreq)

            for stage_key, title in (
                ("asr", "ASR vs IIR (%)"),
                ("orica", "ORICA vs IIR (%)"),
            ):
                series = data_a if stage_key == "asr" else data_o
                fig, ax = plt.subplots(figsize=(11.2, 5.0))
                for cls in ICLABEL_CLASSES:
                    ys = np.asarray(series[cls], dtype=np.float64)
                    ax.plot(
                        t_win_s,
                        ys,
                        marker="o",
                        ms=3,
                        lw=1.2,
                        alpha=0.9,
                        label=cls,
                        zorder=2,
                    )
                ax.axhline(100.0, color="k", ls="--", lw=1.0, alpha=0.55, zorder=1)
                ax.set_xlim(0.0, x_right)
                ax.set_xlabel("Time (s) — window start; bin width ≈ {:.3g} s".format(win_dur_s))
                ax.set_ylabel("% of IIR MS")
                ax.set_title(
                    f"Per-window class mean ({title}); label fixed by IIR ICLabel | "
                    f"x = recording time (full timeline)"
                )
                ax.grid(True, alpha=0.25)
                ax.legend(ncol=4, fontsize=8)
                plt.tight_layout()
                pw_png = out_dir / f"energy_pct_per_window_by_class__{stage_key}.png"
                fig.savefig(pw_png, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"[OK] {pw_png}")

                if eff_exclude:
                    fig_e, ax_e = plt.subplots(figsize=(11.2, 5.0))
                    _draw_exclude_time_spans_on_axis(ax_e, eff_exclude, zorder=0.5)
                    for cls in ICLABEL_CLASSES:
                        ys = np.asarray(series[cls], dtype=np.float64).copy()
                        ys[win_excluded] = np.nan
                        ax_e.plot(
                            t_win_s,
                            ys,
                            marker="o",
                            ms=3,
                            lw=1.2,
                            alpha=0.9,
                            label=cls,
                            zorder=2,
                        )
                    ax_e.axhline(100.0, color="k", ls="--", lw=1.0, alpha=0.55, zorder=1)
                    ax_e.set_xlim(0.0, x_right)
                    ax_e.set_xlabel(
                        "Time (s) — window start; bin width ≈ {:.3g} s".format(win_dur_s)
                    )
                    ax_e.set_ylabel("% of IIR MS")
                    ax_e.set_title(
                        f"Per-window class mean ({title}) | masked windows + shaded EXCLUDE | "
                        f"label fixed by IIR ICLabel"
                    )
                    ax_e.grid(True, alpha=0.25)
                    ax_e.legend(ncol=4, fontsize=8, loc="upper right")
                    ax_e.text(
                        0.01,
                        0.02,
                        "Shaded: EXCLUDE_TIME_RANGES_S on original time axis\n"
                        "Curves: NaN where window overlaps EXCLUDE (axis not compressed)",
                        transform=ax_e.transAxes,
                        fontsize=7,
                        verticalalignment="bottom",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.75),
                    )
                    plt.tight_layout()
                    pw_e = (
                        out_dir
                        / f"energy_pct_per_window_by_class__{stage_key}__exclude_bad_segments_timeline.png"
                    )
                    _savefig_longpath_safe(fig_e, pw_e, dpi=150, bbox_inches="tight")
                    plt.close(fig_e)
                    print(f"[OK] {pw_e}")

            hm_norm = mcolors.Normalize(vmin=0.0, vmax=150.0, clip=False)
            try:
                hm_cmap = plt.colormap("viridis").copy()
            except AttributeError:
                hm_cmap = plt.cm.get_cmap("viridis").copy()  # type: ignore[attr-defined]
            hm_cmap.set_over("#fff200")
            for arr, nm, ttl in (
                (per_ic_win_a, "asr", "ASR vs IIR (%)"),
                (per_ic_win_o, "orica", "ORICA vs IIR (%)"),
            ):
                fig, ax = plt.subplots(
                    figsize=(max(10.0, n_win * 0.45), max(6.0, n_ic * 0.26))
                )
                arr_show = np.ma.masked_invalid(arr[ic_order, :])
                im = ax.imshow(
                    arr_show,
                    aspect="auto",
                    interpolation="nearest",
                    cmap=hm_cmap,
                    norm=hm_norm,
                )
                ax.set_xlabel("Window index")
                ax.set_ylabel("IC (ordered)")
                ylabels = [
                    f"IC{i} ({pred_labels[i]}, p={_safe_text(float(pred_prob[i]), 2)})"
                    for i in ic_order
                ]
                ax.set_yticks(np.arange(n_ic))
                ax.set_yticklabels(ylabels, fontsize=7)
                ax.set_title(f"Per-IC per-window energy % ({ttl}); IIR baseline = 100%")
                cbar = fig.colorbar(im, ax=ax, extend="max")
                cbar.set_label("% of IIR MS (0–150 main; >150 highlighted)")
                plt.tight_layout()
                ph = out_dir / f"energy_pct_per_window_per_ic_heatmap__{nm}.png"
                fig.savefig(ph, dpi=160, bbox_inches="tight")
                plt.close(fig)
                print(f"[OK] {ph}")

    if eff_exclude:
        if not np.any(mask_keep):
            print("[WARN] EXCLUDE 后无剩余采样，跳过 exclude_bad_segments 版 CSV/图")
        else:
            ms_i_e = mean_square_per_ic(s1, mask_keep)
            ms_a_e = mean_square_per_ic(s2, mask_keep)
            ms_o_e = mean_square_per_ic(s3, mask_keep)
            pct_ae_raw = pct_vs_iir(ms_a_e, ms_i_e)
            pct_oe_raw = pct_vs_iir(ms_o_e, ms_i_e)
            ok_e = reliable_ms_iir_mask(ms_i_e, MIN_MS_IIR_REL_FLOOR)
            pct_ae = mask_pct_by_ms_iir_floor(
                pct_ae_raw, ms_i_e, MIN_MS_IIR_REL_FLOOR
            )
            pct_oe = mask_pct_by_ms_iir_floor(
                pct_oe_raw, ms_i_e, MIN_MS_IIR_REL_FLOOR
            )
            ok_pct_stats_ex = ok_ic_for_pct_stats_cap(
                pct_ae, pct_oe, MAX_PCT_VS_IIR_FOR_STATS
            )
            pct_ae_stat = np.asarray(pct_ae, dtype=np.float64).copy()
            pct_oe_stat = np.asarray(pct_oe, dtype=np.float64).copy()
            pct_ae_stat[~ok_pct_stats_ex] = np.nan
            pct_oe_stat[~ok_pct_stats_ex] = np.nan
            ex_csv = (
                out_dir / "ic_source_ms_and_pct_exclude_bad_segments_per_ic.csv"
            )
            with ex_csv.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "# EXCLUDE 时间 mask 后源 MS；列与 ic_source_ms_and_pct_fullrecording_per_ic 相同。",
                    ]
                )
                w.writerow(
                    [
                        "ic_idx",
                        "label_iir",
                        "label_prob",
                        "ms_iir",
                        "ms_asr",
                        "ms_orica",
                        "iir_ok_for_ratio",
                        "pct_asr_vs_iir",
                        "pct_orica_vs_iir",
                        "pct_asr_vs_iir_raw",
                        "pct_orica_vs_iir_raw",
                        "included_in_pct_stats",
                    ]
                )
                for i in range(n_ic):
                    w.writerow(
                        [
                            i,
                            pred_labels[i] if i < len(pred_labels) else "other",
                            float(pred_prob[i]) if i < len(pred_prob) else "",
                            float(ms_i_e[i]),
                            float(ms_a_e[i]),
                            float(ms_o_e[i]),
                            int(ok_e[i]),
                            float(pct_ae[i]) if np.isfinite(pct_ae[i]) else "",
                            float(pct_oe[i]) if np.isfinite(pct_oe[i]) else "",
                            float(pct_ae_raw[i]) if np.isfinite(pct_ae_raw[i]) else "",
                            float(pct_oe_raw[i]) if np.isfinite(pct_oe_raw[i]) else "",
                            int(ok_pct_stats_ex[i]),
                        ]
                    )
            print(f"[OK] {ex_csv}")

            red_rows_e = compute_reduction_summary_rows(
                pred_labels,
                pct_ae,
                pct_oe,
                ms_a_e,
                ms_o_e,
                ok_for_stats=ok_pct_stats_ex,
            )
            red_path_e = out_dir / "reduction_summary_by_icalabel_exclude_bad_segments.csv"
            save_reduction_summary_csv(red_path_e, red_rows_e)
            print(f"[OK] {red_path_e}")
            red_rows_e_pw = compute_power_weighted_reduction_rows(
                pred_labels,
                pct_ae,
                pct_oe,
                ms_i_e,
                ms_a_e,
                ms_o_e,
                ok_for_stats=ok_pct_stats_ex,
            )
            red_pw_e_path = (
                out_dir / "reduction_summary_power_weighted_exclude_bad_segments.csv"
            )
            save_power_weighted_reduction_csv(red_pw_e_path, red_rows_e_pw)
            print(f"[OK] {red_pw_e_path}")

            cap_e = plot_caption_equal_and_power_weighted(red_rows_e, red_rows_e_pw)
            fig_he = 7.2 if cap_e else 6.0
            fig_mse, ax_mse = plt.subplots(
                figsize=(max(12.0, n_ic * 0.5), 6.0)
            )
            plot_per_ic_triple_ms_bars(
                ax_mse,
                x_ic,
                ms_i_e[ic_order],
                ms_a_e[ic_order],
                ms_o_e[ic_order],
                use_log=USE_LOG_Y_FOR_MS_TRIBAR,
            )
            ax_mse.set_xticks(x_ic)
            ax_mse.set_xticklabels(labels_ic, fontsize=7)
            ax_mse.set_xlabel("IC (ordered)")
            ttl_mse = "Per-IC MS (EXCLUDE masked): IIR / ASR / ORICA"
            if USE_LOG_Y_FOR_MS_TRIBAR:
                ttl_mse += " (log10"
                if MS_TRIBAR_LOG_SHIFT_TO_POSITIVE:
                    ttl_mse += "; y shifted to mostly positive"
                ttl_mse += ")"
            ax_mse.set_title(ttl_mse)
            plt.tight_layout()
            p_mse = out_dir / "source_ms_fullrecording_per_ic_ordered_exclude_bad_segments.png"
            fig_mse.savefig(p_mse, dpi=160, bbox_inches="tight")
            plt.close(fig_mse)
            print(f"[OK] {p_mse}")

            fig_he2 = 7.2 if cap_e else 6.0
            fig_ex, ax_ex = plt.subplots(figsize=(max(12.0, n_ic * 0.48), fig_he2))
            ymax_e = (
                float(PER_IC_PLOT_YMAX_PCT)
                if PER_IC_PLOT_YMAX_PCT is not None
                else 125.0
            )
            plot_per_ic_dual_bars_capped(
                ax_ex,
                x_ic,
                pct_ae_stat[ic_order],
                pct_oe_stat[ic_order],
                ymax_e,
                PER_IC_ANNOTATE_OVERFLOW,
                PER_IC_ANNOTATE_DROPS,
            )
            ax_ex.set_xticks(x_ic)
            ax_ex.set_xticklabels(labels_ic, fontsize=7)
            ax_ex.set_ylabel("% of IIR MS (same IC index)")
            ttl_ex = "EXCLUDE masked vs IIR=100%; Y capped"
            if cap_e:
                ax_ex.set_title(f"{ttl_ex}\n{cap_e}", fontsize=9, linespacing=1.35)
            else:
                ax_ex.set_title(ttl_ex)
            ax_ex.grid(True, axis="y", alpha=0.25)
            ax_ex.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            p_ex = out_dir / "energy_pct_fullrecording_per_ic_ordered_exclude_bad_segments.png"
            fig_ex.savefig(p_ex, dpi=160, bbox_inches="tight")
            plt.close(fig_ex)
            print(f"[OK] {p_ex}")

    return out_dir


def main() -> None:
    ok: List[Tuple[str, str, str]] = []
    failed: List[Tuple[str, str, str]] = []

    for profile in PROFILES:
        for ds in DATASET_IDS:
            tag = f"{profile}/b{ds}"
            print(f"\n{'='*60}\n[RUN] {tag}\n{'='*60}")
            try:
                out = run_one(profile, ds)
                ok.append((profile, ds, str(out)))
            except SystemExit as e:
                msg = f"exit {e.code}"
                print(f"[FAIL] {tag}: {msg}")
                failed.append((profile, ds, msg))
                if not CONTINUE_ON_ERROR:
                    sys.exit(e.code if isinstance(e.code, int) else 1)
            except Exception as e:
                print(f"[FAIL] {tag}: {e}")
                traceback.print_exc()
                failed.append((profile, ds, str(e)))
                if not CONTINUE_ON_ERROR:
                    raise

    print("\n" + "=" * 60 + "\n摘要\n" + "=" * 60)
    print(f"成功: {len(ok)}")
    for p, d, path in ok:
        print(f"  OK  {p}  b{d}  -> {path}")
    if failed:
        print(f"失败: {len(failed)}")
        for p, d, err in failed:
            print(f"  FAIL {p}  b{d}  {err}")


if __name__ == "__main__":
    main()
