"""
跨被试汇总：读取各 subject 下 `ic_source_ms_and_pct_exclude_bad_segments_per_ic.csv`。

1. **单被试**：按 ICLabel（label_iir）分组；同类多 IC 的 `pct_asr_vs_iir` / `pct_orica_vs_iir` 用
   **`WITHIN_SUBJECT_IC_AGGREGATION`** 选 **nanmean** 或 **nanmedian**（每类一个数对）。
   可选：`label_prob` 门控见 `MIN_LABEL_PROB_BY_CLASS`。
2. **跨被试**：对每个类，在各 subject 的上述「类内聚合值」上再 **mean**，并报告 **std(ddof=1)**。

`WITHIN_SUBJECT_IC_AGGREGATION="median"` 时，输出目录前半段由 `MULTISUBJECT_OUTPUT_BASENAME` 经
**首次** `mean`→`median` 替换得到（便于与 mean 模式区分）；不含子串 `mean` 时目录名与常量一致。

运行: python aggregate_exclude_bad_pct_multisubject_average.py

默认横轴为全部 ICLabel 标准类；若只要子集，设 USE_ALL_CLASSES = False 并改 CATEGORIES。
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
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon
    from scipy.stats import ttest_rel
except ImportError:  # pragma: no cover
    wilcoxon = None  # type: ignore
    ttest_rel = None  # type: ignore

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 与 ica_source_energy_analysis_correctly 中 ICLabel 归一化一致
_LABEL_ALIASES: Dict[str, str] = {
    "muscle_artifact": "muscle",
    "eye_blink": "eye",
    "heart_beat": "heart",
}

CSV_NAME = "ic_source_ms_and_pct_exclude_bad_segments_per_ic.csv"

# 与 ica_source_energy_analysis_correctly.ICLABEL_CLASSES 顺序一致（含 heart / line_noise / channel_noise）
ALL_KNOWN_CLASSES: Tuple[str, ...] = (
    "brain",
    "muscle",
    "eye",
    "heart",
    "line_noise",
    "channel_noise",
    "other",
)

# 仅当 USE_ALL_CLASSES 为 False 时作为横轴类别（可改成任意子集）
DEFAULT_CATEGORIES: Tuple[str, ...] = ALL_KNOWN_CLASSES

# ---------------------------------------------------------------------------
# 运行配置（只改这里，无需命令行参数）
# ---------------------------------------------------------------------------
EXPERIMENT: str = "Lapaasrpy20_2min_70"
#EXPERIMENT: str = "SN_Driveasrpy20_2min_70"
#SEGMENT_PREFIX: str = "segment_exclude_window_10_remove"
#SEGMENT_PREFIX: str = "para_basic_window_10_remove"
#SEGMENT_PREFIX: str = "para_basic_window_120_remove"
SEGMENT_PREFIX: str = "1segment_exclude_window_10_remove"
# 汇总输出文件夹名前半段：先按 WITHIN_SUBJECT_IC_AGGREGATION 做 mean→median 替换（仅 median 模式、且仅首次），再消毒；最后为 `{该名}__{SEGMENT_PREFIX 消毒后}/`。
MULTISUBJECT_OUTPUT_BASENAME: str = "multisubject_exclude_bad_mean_by_class_prob_0_full"
# 单被试内同类多 IC：`mean` -> nanmean；`median` -> nanmedian。跨被试仍为各 subject 该值的 mean±SD。
WITHIN_SUBJECT_IC_AGGREGATION: Literal["mean", "median"] = "median"
# True：横轴为上面全部 ICLabel 类；False：只用下面 CATEGORIES 子集
USE_ALL_CLASSES: bool = True
CATEGORIES: Tuple[str, ...] = DEFAULT_CATEGORIES

# 各类别最小 ICLabel 置信度：仅当 ic CSV 中 label_prob **严格大于**该值时，该 IC 才参与该类 nanmean/nanmedian。
# 未出现的类别视为 0 → 不按概率过滤。缺省/非法 label_prob 在阈值>0 时会被排除。
MIN_LABEL_PROB_BY_CLASS: Dict[str, float] = {
    # "brain": 0.7,
    # "eye": 0.7,
    # "heart": 0.7,
    # "muscle": 0.7,
    # "other": 0.7,
    # "line_noise": 0.7,
    # "channel_noise": 0.7,
}


def _norm_label(s: str) -> str:
    t = str(s).strip().lower()
    return _LABEL_ALIASES.get(t, t)


def _sanitize_tag(prefix: str) -> str:
    return re.sub(r"[^\w\-]+", "_", prefix.strip())[:80]


def _effective_multisubject_output_basename(
    raw_basename: str,
    within_subject_agg: Literal["mean", "median"],
) -> str:
    """median 模式且名称中含 `mean` 时，将首次出现的 `mean` 替换为 `median`（便于输出目录区分）。"""
    b = raw_basename.strip()
    if within_subject_agg == "median" and "mean" in b:
        return b.replace("mean", "median", 1)
    return b


def _find_segment_dir(subject_dir: Path, segment_prefix: str) -> Optional[Path]:
    cands = sorted(
        [p for p in subject_dir.glob(f"{segment_prefix}*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not cands:
        return None
    if len(cands) > 1:
        print(
            f"[WARN] {subject_dir.name}: 匹配到 {len(cands)} 个目录，使用按名字排序的第一个:\n"
            f"       {cands[0].name}"
        )
    return cands[0]


def _read_exclude_bad_csv(path: Path) -> List[Dict[str, Any]]:
    """
    源 CSV 首行常为「# ...」注释；若直接 DictReader，会把注释行当表头，
    导致 label_iir 等键全错、读不到任何数据。此处先跳过所有 # 行再读表头。
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return []
            if line.lstrip("\ufeff").strip().startswith("#"):
                continue
            f.seek(pos)
            break
        reader = csv.DictReader(f)
        for row in reader:
            if not row or row.get("ic_idx") is None:
                continue
            if str(row.get("ic_idx", "")).startswith("#"):
                continue
            rows.append(row)
    return rows


def _parse_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _parse_int01(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _min_label_prob_threshold(lab: str, min_by_class: Dict[str, float]) -> float:
    v = min_by_class.get(lab)
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def per_subject_class_means(
    rows: Sequence[Dict[str, Any]],
    categories: Sequence[str],
    min_label_prob_by_class: Optional[Dict[str, float]] = None,
    within_subject_agg: Literal["mean", "median"] = "mean",
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, int]]:
    """
    单被试内：同一 ICLabel 类下多个 IC 的 pct 取 **nanmean** 或 **nanmedian**（每类一个 ASR%、一个 ORICA%）。
    若 min_label_prob_by_class 中该类阈值 >0，则仅保留 label_prob > 阈值的行。
    返回 (每类聚合值, 每类参与聚合的 IC 个数)。
    """
    prob_floor: Dict[str, float] = dict(min_label_prob_by_class or {})
    by_class: Dict[str, List[Tuple[float, float]]] = {c: [] for c in categories}
    for row in rows:
        lab = _norm_label(row.get("label_iir", "") or "")
        if lab not in by_class:
            continue
        inc = _parse_int01(row.get("included_in_pct_stats"))
        if inc is not None and inc == 0:
            continue
        pa = _parse_float(row.get("pct_asr_vs_iir"))
        po = _parse_float(row.get("pct_orica_vs_iir"))
        if pa is None or po is None:
            continue
        p_need = _min_label_prob_threshold(lab, prob_floor)
        if p_need > 0.0:
            p_lab = _parse_float(row.get("label_prob"))
            if p_lab is None or p_lab <= p_need:
                continue
        by_class[lab].append((pa, po))

    avg: Dict[str, Tuple[float, float]] = {}
    counts: Dict[str, int] = {}
    for c in categories:
        vals = by_class.get(c) or []
        counts[c] = len(vals)
        if not vals:
            avg[c] = (float("nan"), float("nan"))
            continue
        a = np.asarray([v[0] for v in vals], dtype=np.float64)
        o = np.asarray([v[1] for v in vals], dtype=np.float64)
        if within_subject_agg == "median":
            avg[c] = (float(np.nanmedian(a)), float(np.nanmedian(o)))
        else:
            avg[c] = (float(np.nanmean(a)), float(np.nanmean(o)))
    return avg, counts


def collect_experiment(
    experiment_dir: Path,
    segment_prefix: str,
    categories: Sequence[str],
    min_label_prob_by_class: Optional[Dict[str, float]] = None,
    within_subject_agg: Literal["mean", "median"] = "mean",
) -> Tuple[
    List[str],
    Dict[str, Dict[str, Tuple[float, float]]],
    Dict[str, Dict[str, int]],
    List[str],
]:
    """
    Returns:
      subjects_sorted,
      subj_class_means[subject][class] -> (mean_asr, mean_orica),
      n_ic_used[subject][class] -> 参与类内 mean/median 的 IC 数,
      warnings (missing csv etc.)
    """
    ica_root = experiment_dir / "ica_source_analysis"
    if not ica_root.is_dir():
        raise FileNotFoundError(f"未找到目录: {ica_root}")

    warnings: List[str] = []
    subj_class_means: Dict[str, Dict[str, Tuple[float, float]]] = {}
    n_ic_used: Dict[str, Dict[str, int]] = {}

    subj_dirs = sorted([p for p in ica_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for sd in subj_dirs:
        seg = _find_segment_dir(sd, segment_prefix)
        if seg is None:
            warnings.append(f"{sd.name}: 无匹配 '{segment_prefix}*' 的子文件夹")
            continue
        csv_p = seg / CSV_NAME
        if not csv_p.is_file():
            warnings.append(f"{sd.name}: 缺少 {csv_p.name}（{seg.name}）")
            continue
        rows = _read_exclude_bad_csv(csv_p)
        if not rows:
            warnings.append(f"{sd.name}: {csv_p} 解析后无数据行（检查文件或 # 注释行）")
        m, cnt = per_subject_class_means(
            rows,
            categories,
            min_label_prob_by_class=min_label_prob_by_class,
            within_subject_agg=within_subject_agg,
        )
        subj_class_means[sd.name] = m
        n_ic_used[sd.name] = cnt

    return sorted(subj_class_means.keys()), subj_class_means, n_ic_used, warnings


def summarize_across_subjects(
    subjects: Sequence[str],
    subj_class_means: Dict[str, Dict[str, Tuple[float, float]]],
    categories: Sequence[str],
    n_ic_used: Optional[Dict[str, Dict[str, int]]] = None,
    within_subject_agg: Literal["mean", "median"] = "mean",
) -> Tuple[
    Dict[str, Tuple[int, float, float, float, float]],
    List[Dict[str, Any]],
]:
    """
    对每个 class：在各 subject「类内 mean 或 median」上再算跨被试 mean、std（ddof=1），并统计有效 n。
    """
    summary: Dict[str, Tuple[int, float, float, float, float]] = {}
    long_rows: List[Dict[str, Any]] = []
    for subj in subjects:
        m = subj_class_means[subj]
        for c in categories:
            a, o = m.get(c, (float("nan"), float("nan")))
            nic = 0
            if n_ic_used and subj in n_ic_used:
                nic = int(n_ic_used[subj].get(c, 0))
            if within_subject_agg == "median":
                long_rows.append(
                    {
                        "subject": subj,
                        "class": c,
                        "n_ic_used_for_median": nic,
                        "median_pct_asr_vs_iir": a,
                        "median_pct_orica_vs_iir": o,
                    }
                )
            else:
                long_rows.append(
                    {
                        "subject": subj,
                        "class": c,
                        "n_ic_used_for_mean": nic,
                        "mean_pct_asr_vs_iir": a,
                        "mean_pct_orica_vs_iir": o,
                    }
                )

    for c in categories:
        asrs: List[float] = []
        oris: List[float] = []
        for subj in subjects:
            a, o = subj_class_means[subj].get(c, (float("nan"), float("nan")))
            if np.isfinite(a) and np.isfinite(o):
                asrs.append(a)
                oris.append(o)
        ma = float(np.mean(asrs)) if asrs else float("nan")
        sa = float(np.std(asrs, ddof=1)) if len(asrs) > 1 else (0.0 if asrs else float("nan"))
        mo = float(np.mean(oris)) if oris else float("nan")
        sd_o = float(np.std(oris, ddof=1)) if len(oris) > 1 else (0.0 if oris else float("nan"))
        summary[c] = (len(asrs), ma, sa, mo, sd_o)
    return summary, long_rows


def _stars_p(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "†"
    return "ns"


def paired_pvalues_asr_vs_orica(
    subjects: Sequence[str],
    subj_class_means: Dict[str, Dict[str, Tuple[float, float]]],
    cls: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (p_ttest, p_wilcoxon) for paired ASR vs ORICA in one class.
    Both are two-sided; when unavailable/invalid return None.
    """
    xs: List[float] = []
    ys: List[float] = []
    for subj in subjects:
        a, o = subj_class_means[subj].get(cls, (float("nan"), float("nan")))
        if np.isfinite(a) and np.isfinite(o):
            xs.append(a)
            ys.append(o)
    if len(xs) < 3:
        return None, None
    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    if np.allclose(xs_arr, ys_arr):
        return 1.0, 1.0
    p_t: Optional[float] = None
    p_w: Optional[float] = None
    if ttest_rel is not None:
        try:
            _t, p = ttest_rel(xs_arr, ys_arr, alternative="two-sided", nan_policy="omit")
            if p is not None and np.isfinite(p):
                p_t = float(p)
        except Exception:
            p_t = None
    if wilcoxon is not None:
        try:
            stat, p = wilcoxon(xs_arr, ys_arr, alternative="two-sided", zero_method="wilcox")
            p_w = float(p)
        except ValueError:
            p_w = None
    # 无 scipy 时：给 t-test 提供一个粗略正态近似（仅作参考）
    if p_t is None:
        # 配对差近似正态的简单双侧 t（仅作参考）
        d = xs_arr - ys_arr
        n = len(d)
        md = float(np.mean(d))
        sd = float(np.std(d, ddof=1))
        if sd > 1e-12 and n >= 2:
            t = md / (sd / np.sqrt(n))
            from math import erfc, sqrt

            z = abs(t)  # 误用 t 当 z，小样本不准；无 scipy 时仅作粗标星
            p_two = erfc(z / sqrt(2.0))
            p_t = float(min(max(p_two, 1e-15), 1.0))
    return p_t, p_w


def _p_label(prefix: str, p: Optional[float]) -> str:
    star = _stars_p(p)
    if not star:
        return ""
    return f"{prefix}:{star}"


def _join_test_labels(p_t: Optional[float], p_w: Optional[float]) -> str:
    parts: List[str] = []
    lt = _p_label("t", p_t)
    lw = _p_label("w", p_w)
    if lt:
        parts.append(lt)
    if lw:
        parts.append(lw)
    return " | ".join(parts)


def plot_grouped(
    categories: Sequence[str],
    subjects: Sequence[str],
    subj_class_means: Dict[str, Dict[str, Tuple[float, float]]],
    summary: Dict[str, Tuple[int, float, float, float, float]],
    out_png: Path,
    title: str,
    within_subject_agg: Literal["mean", "median"] = "mean",
) -> None:
    x = np.arange(len(categories), dtype=np.float64)
    w = 0.34
    means_asr = []
    stds_asr = []
    means_ori = []
    stds_ori = []
    for c in categories:
        _, ma, sa, mo, so = summary[c]
        means_asr.append(ma)
        stds_asr.append(0.0 if (not np.isfinite(sa)) else sa)
        means_ori.append(mo)
        stds_ori.append(0.0 if (not np.isfinite(so)) else so)

    mas = np.asarray(means_asr, dtype=np.float64)
    mos = np.asarray(means_ori, dtype=np.float64)
    sas = np.asarray(stds_asr, dtype=np.float64)
    sos = np.asarray(stds_ori, dtype=np.float64)
    mas_plot = np.where(np.isfinite(mas), mas, 0.0)
    mos_plot = np.where(np.isfinite(mos), mos, 0.0)
    err_asr = np.where(np.isfinite(mas) & np.isfinite(sas), sas, 0.0)
    err_ori = np.where(np.isfinite(mos) & np.isfinite(sos), sos, 0.0)

    fig, ax = plt.subplots(figsize=(max(8.0, len(categories) * 1.85), 5.6))
    c_asr = "#e6c200"
    c_ori = "#c0392b"
    ax.bar(
        x - w / 2,
        mas_plot,
        width=w,
        yerr=err_asr,
        label="Power(ASR) / Power(IIR)  [%]",
        color=c_asr,
        edgecolor="#333",
        linewidth=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        zorder=2,
    )
    ax.bar(
        x + w / 2,
        mos_plot,
        width=w,
        yerr=err_ori,
        label="Power(ASR+ORICA) / Power(IIR)  [%]",
        color=c_ori,
        edgecolor="#333",
        linewidth=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        zorder=2,
    )

    n_sub = len(subjects)
    for si, subj in enumerate(subjects):
        off = (si - (n_sub - 1) / 2.0) * min(0.018, 0.5 / max(n_sub, 1))
        for ci, c in enumerate(categories):
            a, o = subj_class_means[subj].get(c, (float("nan"), float("nan")))
            if np.isfinite(a):
                ax.scatter(
                    ci - w / 2 + off,
                    a,
                    s=28,
                    marker="o",
                    facecolors="none",
                    edgecolors="#5d4e37",
                    linewidths=0.9,
                    alpha=0.85,
                    zorder=4,
                )
            if np.isfinite(o):
                ax.scatter(
                    ci + w / 2 + off,
                    o,
                    s=28,
                    marker="s",
                    facecolors="none",
                    edgecolors="#7b241c",
                    linewidths=0.9,
                    alpha=0.85,
                    zorder=4,
                )

    ax.axhline(100.0, color="k", linestyle="--", linewidth=1.0, alpha=0.55, label="IIR baseline (100%)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in categories])
    ax.set_ylabel("Power relative to IIR-cleaned data (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right", fontsize=9)

    stack = np.concatenate([mas + sas, mos + sos])
    hi = float(np.nanmax(stack)) if np.any(np.isfinite(stack)) else 100.0
    ymax = max(105.0, hi * 1.12)
    ax.set_ylim(0.0, ymax)

    y_bracket = ymax * 0.94
    for ci, c in enumerate(categories):
        p_t, p_w = paired_pvalues_asr_vs_orica(subjects, subj_class_means, c)
        mark = _join_test_labels(p_t, p_w)
        if mark:
            x0, x1 = ci - w / 2, ci + w / 2
            h = y_bracket - ci * 0.01 * ymax * 0.02
            ax.plot([x0, x0, x1, x1], [h - 1.5, h, h, h - 1.5], color="#333", lw=0.9, clip_on=False)
            ax.text(
                ci,
                h + 0.02 * ymax,
                mark,
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    wline = (
        "Per subject: class-wise mean of IC pct; error bars = SD across subjects of those per-subject class means.\n"
        if within_subject_agg == "mean"
        else "Per subject: class-wise median of IC pct; error bars = SD across subjects of those per-subject class medians.\n"
    )
    fig.text(
        0.01,
        0.02,
        wline
        + "Hollow markers = each subject (○ ASR/IIR, □ ORICA/IIR). "
        + "Bracket labels: paired tests, t=paired t-test, w=Wilcoxon (two-sided).\n"
        + "ICs may be gated by label_prob > MIN_LABEL_PROB_BY_CLASS[class] (see run_meta.txt).",
        fontsize=7.5,
        color="#444",
        va="bottom",
    )
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    if not np.any(np.isfinite(mas)) and not np.any(np.isfinite(mos)):
        print(
            f"[WARN] 图 {out_png.name}：所有类别均无有效数据（常为 CSV 表头被注释行占用，已修复读取逻辑后请重跑）。",
            file=sys.stderr,
        )


def plot_single_subject_bars(
    categories: Sequence[str],
    med: Dict[str, Tuple[float, float]],
    out_png: Path,
    title: str,
) -> None:
    """单个 subject：各类别类内聚合值（mean 或 median，由 main 标题区分）的分组柱（无跨被试误差条）。"""
    x = np.arange(len(categories), dtype=np.float64)
    w = 0.34
    mas = np.asarray([med.get(c, (np.nan, np.nan))[0] for c in categories], dtype=np.float64)
    mos = np.asarray([med.get(c, (np.nan, np.nan))[1] for c in categories], dtype=np.float64)
    mas_plot = np.where(np.isfinite(mas), mas, 0.0)
    mos_plot = np.where(np.isfinite(mos), mos, 0.0)
    fig, ax = plt.subplots(figsize=(max(8.0, len(categories) * 1.55), 4.6))
    c_asr = "#e6c200"
    c_ori = "#c0392b"
    ax.bar(x - w / 2, mas_plot, width=w, label="Power(ASR)/Power(IIR) [%]", color=c_asr, edgecolor="#333", linewidth=0.6, zorder=2)
    ax.bar(x + w / 2, mos_plot, width=w, label="Power(ASR+ORICA)/Power(IIR) [%]", color=c_ori, edgecolor="#333", linewidth=0.6, zorder=2)
    ax.axhline(100.0, color="k", linestyle="--", linewidth=1.0, alpha=0.55, label="IIR baseline (100%)")
    ax.set_xticks(x)
    rot = 22 if len(categories) <= 4 else 38
    ax.set_xticklabels([c.replace("_", " ") for c in categories], rotation=rot, ha="right")
    ax.set_ylabel("Power relative to IIR (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right", fontsize=8)
    stack = np.concatenate([mas, mos])
    hi = float(np.nanmax(stack)) if np.any(np.isfinite(stack)) else 100.0
    ymax = max(105.0, hi * 1.15)
    ax.set_ylim(0.0, ymax)
    for ci, _c in enumerate(categories):
        if np.isfinite(mas[ci]):
            ax.text(ci - w / 2, min(float(mas[ci]) + ymax * 0.02, ymax * 0.98), f"{mas[ci]:.1f}", ha="center", va="bottom", fontsize=7)
        if np.isfinite(mos[ci]):
            ax.text(ci + w / 2, min(float(mos[ci]) + ymax * 0.02, ymax * 0.98), f"{mos[ci]:.1f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_subjects_grid(
    categories: Sequence[str],
    subjects: Sequence[str],
    subj_class_means: Dict[str, Dict[str, Tuple[float, float]]],
    out_png: Path,
    suptitle: str,
) -> None:
    n = len(subjects)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.2), squeeze=False)
    w = 0.34
    x = np.arange(len(categories), dtype=np.float64)
    c_asr = "#e6c200"
    c_ori = "#c0392b"
    for i, subj in enumerate(subjects):
        ri, ci = divmod(i, ncols)
        ax = axes[ri][ci]
        med = subj_class_means[subj]
        mas = np.asarray([med.get(c, (np.nan, np.nan))[0] for c in categories], dtype=np.float64)
        mos = np.asarray([med.get(c, (np.nan, np.nan))[1] for c in categories], dtype=np.float64)
        mas_plot = np.where(np.isfinite(mas), mas, 0.0)
        mos_plot = np.where(np.isfinite(mos), mos, 0.0)
        ax.bar(x - w / 2, mas_plot, width=w, color=c_asr, edgecolor="#333", linewidth=0.5, label="ASR/IIR")
        ax.bar(x + w / 2, mos_plot, width=w, color=c_ori, edgecolor="#333", linewidth=0.5, label="ORICA/IIR")
        ax.axhline(100.0, color="k", linestyle="--", linewidth=0.8, alpha=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=7)
        ax.set_title(subj, fontsize=10, fontweight="bold")
        ax.set_ylabel("% vs IIR", fontsize=8)
        ax.grid(True, axis="y", alpha=0.22)
        if i == 0:
            ax.legend(loc="upper right", fontsize=6)
        stack = np.concatenate([mas, mos])
        hi = float(np.nanmax(stack)) if np.any(np.isfinite(stack)) else 100.0
        ax.set_ylim(0.0, max(105.0, hi * 1.12))
    for j in range(len(subjects), nrows * ncols):
        ri, ci = divmod(j, ncols)
        axes[ri][ci].set_visible(False)
    fig.suptitle(suptitle, fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    path: Path,
    summary: Dict[str, Tuple[int, float, float, float, float]],
    within_subject_agg: Literal["mean", "median"] = "mean",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if within_subject_agg == "median":
            w.writerow(
                [
                    "class",
                    "n_subjects_valid_paired",
                    "mean_of_median_pct_asr_vs_iir",
                    "std_across_subjects_asr",
                    "mean_of_median_pct_orica_vs_iir",
                    "std_across_subjects_orica",
                ]
            )
        else:
            w.writerow(
                [
                    "class",
                    "n_subjects_valid_paired",
                    "mean_of_mean_pct_asr_vs_iir",
                    "std_across_subjects_asr",
                    "mean_of_mean_pct_orica_vs_iir",
                    "std_across_subjects_orica",
                ]
            )
        for c, tup in summary.items():
            w.writerow([c, tup[0], tup[1], tup[2], tup[3], tup[4]])


def write_long_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    experiment_dir = (THRESHOLD_ROOT / EXPERIMENT).resolve()
    if not experiment_dir.is_dir():
        print(f"[ERR] 实验目录不存在: {experiment_dir}", file=sys.stderr)
        return 1

    if WITHIN_SUBJECT_IC_AGGREGATION not in ("mean", "median"):
        print(
            "[ERR] WITHIN_SUBJECT_IC_AGGREGATION 必须为 'mean' 或 'median'",
            file=sys.stderr,
        )
        return 1

    if USE_ALL_CLASSES:
        categories = list(ALL_KNOWN_CLASSES)
    else:
        categories = list(CATEGORIES)

    tag = _sanitize_tag(SEGMENT_PREFIX)
    w_agg = WITHIN_SUBJECT_IC_AGGREGATION
    eff_basename = _effective_multisubject_output_basename(
        MULTISUBJECT_OUTPUT_BASENAME, w_agg
    )
    if w_agg == "median" and "mean" not in MULTISUBJECT_OUTPUT_BASENAME.strip():
        print(
            "[WARN] median 模式但 MULTISUBJECT_OUTPUT_BASENAME 不含子串 'mean'，"
            "输出目录名未自动替换；若与 mean 运行重名请自行改 BASENAME。",
            file=sys.stderr,
        )

    subjects, subj_class_means, n_ic_used, warns = collect_experiment(
        experiment_dir,
        SEGMENT_PREFIX,
        categories,
        min_label_prob_by_class=MIN_LABEL_PROB_BY_CLASS,
        within_subject_agg=w_agg,
    )
    for w in warns:
        print(f"[WARN] {w}")

    if not subjects:
        print("[ERR] 未找到任何有效 subject CSV。", file=sys.stderr)
        return 1

    summary, long_rows = summarize_across_subjects(
        subjects,
        subj_class_means,
        categories,
        n_ic_used=n_ic_used,
        within_subject_agg=w_agg,
    )
    out_leaf = f"{_sanitize_tag(eff_basename)}__{tag}"
    out_base = experiment_dir / out_leaf
    out_base.mkdir(parents=True, exist_ok=True)
    per_sub_dir = out_base / "per_subject_plots"
    per_sub_dir.mkdir(parents=True, exist_ok=True)

    if w_agg == "median":
        long_path = out_base / "per_subject_class_medians.csv"
        sum_path = out_base / "cross_subject_mean_of_class_medians.csv"
        png_name = "power_pct_vs_iir_by_class_grouped_median_by_class.png"
        subj_png_suffix = "_median_pct_by_class.png"
        grid_png_name = "all_subjects_median_pct_by_class_grid.png"
    else:
        long_path = out_base / "per_subject_class_means.csv"
        sum_path = out_base / "cross_subject_mean_of_class_means.csv"
        png_name = "power_pct_vs_iir_by_class_grouped_mean_by_class.png"
        subj_png_suffix = "_mean_pct_by_class.png"
        grid_png_name = "all_subjects_mean_pct_by_class_grid.png"

    meta_path = out_base / "run_meta.txt"

    write_long_csv(long_path, long_rows)
    write_summary_csv(sum_path, summary, within_subject_agg=w_agg)

    with meta_path.open("w", encoding="utf-8") as f:
        if w_agg == "median":
            f.write(
                "aggregation=nanmedian_within_subject_per_ICLabel_class_then_mean_across_subjects\n"
            )
        else:
            f.write(
                "aggregation=nanmean_within_subject_per_ICLabel_class_then_mean_across_subjects\n"
            )
        f.write(
            "label_prob_gate=include_row_only_if_label_prob>MIN_LABEL_PROB_BY_CLASS[label_iir] "
            "(per-class threshold; missing class => 0 => no gate)\n"
        )
        for cls in sorted(set(categories) | set(MIN_LABEL_PROB_BY_CLASS.keys())):
            t = _min_label_prob_threshold(cls, MIN_LABEL_PROB_BY_CLASS)
            if t > 0.0:
                f.write(f"MIN_LABEL_PROB_BY_CLASS[{cls}]={t}\n")
        if not any(_min_label_prob_threshold(c, MIN_LABEL_PROB_BY_CLASS) > 0.0 for c in categories):
            f.write("MIN_LABEL_PROB_BY_CLASS=(all zero or empty; no label_prob filtering)\n")
        f.write(f"WITHIN_SUBJECT_IC_AGGREGATION={w_agg!r}\n")
        f.write(f"MULTISUBJECT_OUTPUT_BASENAME={MULTISUBJECT_OUTPUT_BASENAME!r}\n")
        f.write(f"effective_output_basename={eff_basename!r}\n")
        f.write(f"experiment_dir={experiment_dir}\n")
        f.write(f"out_dir_leaf={out_leaf}\n")
        f.write(f"segment_prefix={SEGMENT_PREFIX!r}\n")
        f.write(f"n_subjects={len(subjects)}\n")
        f.write(f"subjects={', '.join(subjects)}\n")
        f.write(f"categories={', '.join(categories)}\n")
        f.write(f"csv={CSV_NAME}\n")
        f.write(f"per_subject_plots={per_sub_dir}\n")

    agg_word = "mean" if w_agg == "mean" else "median"
    title = (
        f"{EXPERIMENT}: class-wise {agg_word} IC % vs IIR (EXCLUDE bad), "
        f"then mean±SD across subjects\nprefix={SEGMENT_PREFIX}"
    )
    png_path = out_base / png_name
    plot_grouped(
        categories,
        subjects,
        subj_class_means,
        summary,
        png_path,
        title,
        within_subject_agg=w_agg,
    )

    for subj in subjects:
        p_sub = per_sub_dir / f"{subj}{subj_png_suffix}"
        plot_single_subject_bars(
            categories,
            subj_class_means[subj],
            p_sub,
            f"{subj} (EXCLUDE bad): class-wise {agg_word} % vs IIR",
        )

    grid_png = per_sub_dir / grid_png_name
    plot_all_subjects_grid(
        categories,
        subjects,
        subj_class_means,
        grid_png,
        f"{EXPERIMENT}: each subject, {agg_word} % vs IIR by class",
    )

    print(f"[OK] subjects={len(subjects)} -> {out_base}")
    print(f"     {long_path.name}, {sum_path.name}, {png_path.name}")
    print(f"     {per_sub_dir.name}/ (one PNG per subject + {grid_png.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
