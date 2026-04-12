"""
对多个阶段 NPZ（Raw / IIR / ASR / ORICA）按时间窗切分，每窗独立做 MNE ICA + ICLabel，
统计每窗 brain / other / 各类伪影成分数量及置信度，并生成汇总图。

默认读取同目录下 beeg_raw1.npz, beeg_iir1.npz, beeg_asr1.npz, beeg_orica1.npz。

ICLabel 7 类顺序（与 mne-icalabel 概率矩阵列顺序一致）:
  brain, muscle, eye, heart, line_noise, channel_noise, other
mne-icalabel 返回的字符串标签为 muscle artifact / eye blink / heart beat 等，
脚本内会规范成与上表一致的简短类名再统计（否则会全部被误判为 other）。

依赖: numpy, matplotlib, scipy, mne, mne-icalabel
  pip install mne mne-icalabel

窗长建议: ICA 需要足够样本；默认 30s，可用 --window-sec 调整，不宜低于约 15–20s（高采样率可适当缩短）。

输出默认写入本脚本同目录下的 ica_analysis_result/（可用 --out-dir 修改）；
图/表文件名在扩展名前追加 __win{秒}s（如 __win60s），与 --window-sec 一致。
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

# mne-icalabel ICLABEL_NUMERICAL_TO_STRING 经 _normalize_label 后的别名 -> 本脚本 ICLABEL_CLASSES 键
_ICLABEL_NORMALIZED_ALIASES: Dict[str, str] = {
    "muscle_artifact": "muscle",
    "eye_blink": "eye",
    "heart_beat": "heart",
}

_STAGE_DEFS: Dict[str, Tuple[str, str]] = {
    "raw": ("Raw", "b03eeg_raw1.npz"),
    "iir": ("IIR", "b03eeg_iir1.npz"),
    "asr": ("ASR", "b03eeg_asr1.npz"),
    "orica": ("ORICA", "b03eeg_orica1.npz"),
}

# _STAGE_DEFS: Dict[str, Tuple[str, str]] = {
#     "raw": ("Raw", "b07eeg_offline_raw1.npz"),
#     "iir": ("IIR", "b07eeg_offline_iir1.npz"),
#     "asr": ("ASR", "b07eeg_offline_asr1.npz"),
#     "orica": ("ORICA", "b07eeg_offline_ica1.npz"),
    
# }


_DEFAULT_WINDOW_SEC = 60.0*2
_DEFAULT_OUT_SUBDIR = "ica_analysis_result_03_2min"
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


def run_ica_iclabel_window(
    data_ch_by_time: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    random_state: int,
    apply_iir: bool,
    ica_max_iter: int,
) -> Tuple[Optional[WindowICLabelResult], Optional[str]]:
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    n_ch, n_samp = data_ch_by_time.shape
    if n_ch != len(ch_names):
        ch_names = [f"EEG{i+1:03d}" for i in range(n_ch)]

    x = np.asarray(data_ch_by_time, dtype=np.float64)
    if apply_iir:
        x = mne.filter.filter_data(
            x, sfreq, l_freq=1.0, h_freq=50.0, verbose=False
        )

    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    raw = mne.io.RawArray(x, info)
    try:
        raw.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass

    n_comp = n_ch
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

    try:
        labels_out = label_components(raw, ica, method="iclabel")
    except Exception as e:
        return None, f"ICLabel failed: {e}"

    pred_labels = [
        _canonical_icalabel_label(lbl)
        for lbl in _extract_pred_labels(labels_out, n_comp)
    ]
    P, p1d = _extract_proba_from_labels_out(labels_out, n_comp)
    # label_components 返回的 y_pred_proba 常为「获胜类」一维；完整 (n_ic, 7) 在 ica.labels_scores_
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

    res = WindowICLabelResult(
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
    )
    return res, None


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
    fig.suptitle(
        f"{stage_label} ({file_stem}) — ICA + ICLabel per window\n"
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
    fig, ax = plt.subplots(figsize=(max(8.0, len(results) * 0.25), 5.0))
    im = ax.imshow(mat.T, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(ICLABEL_CLASSES)))
    ax.set_yticklabels(ICLABEL_CLASSES, fontsize=8)
    ax.set_xlabel("Window index")
    ax.set_title(
        f"{stage_label} — mean ICLabel class probability per IC (per window)\n{title_extra}",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="mean prob")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved heatmap: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分窗 ICA + ICLabel，统计 brain/other/伪影并出图",
    )
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--iir", type=Path, default=None)
    parser.add_argument("--asr", type=Path, default=None)
    parser.add_argument("--orica", type=Path, default=None)
    parser.add_argument(
        "--order",
        type=str,
        default="raw,iir,asr,orica",
        help="逗号分隔阶段顺序",
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

    stage_runs: List[Tuple[str, Path, List[WindowICLabelResult]]] = []
    global_artifact_subtype_peak = 0.0

    for label, path in stages:
        path = path.resolve()
        if not path.is_file():
            print(f"[SKIP] missing {label}: {path}")
            continue
        inf = load_npz_file(path)
        if inf is None:
            continue
        n_ch = inf["data"].shape[0]
        min_samples = max(int(_MIN_SAMPLES_FACTOR * n_ch), int(10 * inf["sampling_rate"]))
        print(
            f"\n=== {label} | {path.name} | shape={inf['data'].shape} | "
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
            # 第二子图是伪影子类「堆叠柱」，应按每窗堆叠总高度取最大值
            local_peak = max(
                local_peak,
                float(sum(r.artifact_subtype_counts.get(c, 0) for c in ARTIFACT_CLASSES)),
            )
        global_artifact_subtype_peak = max(global_artifact_subtype_peak, local_peak)
        stage_runs.append((label, path, results))

    # 若全为 0，则给一个最小显示范围，避免不同图自动缩放难以对比
    artifact_subtype_ymax = max(1.0, float(np.ceil(global_artifact_subtype_peak)))

    for label, path, results in stage_runs:
        stem = path.stem
        plot_stage_results(
            label,
            stem,
            results,
            out_dir / f"ica_iclabel_windows__{stem}__{win_suffix}.png",
            artifact_subtype_ymax=artifact_subtype_ymax,
        )
        plot_proba_heatmap(
            label,
            stem,
            results,
            out_dir / f"ica_iclabel_proba_heatmap__{stem}__{win_suffix}.png",
        )
        save_results_csv(
            label,
            stem,
            results,
            out_dir / f"ica_iclabel_windows__{stem}__{win_suffix}.csv",
        )


if __name__ == "__main__":
    main()
