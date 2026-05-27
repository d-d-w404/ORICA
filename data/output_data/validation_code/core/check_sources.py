"""check_sources.py

Interactive viewer for saved ORICA IC sources (*eeg_ic_sources1.npz).

This script expects the npz to contain per-chunk ICLabel outputs:
  - sources: (n_ic, n_samples_total)
  - chunk_start_samples: (n_chunks,)
  - chunk_sizes: (n_chunks,)
  - ic_labels_by_chunk: (n_chunks, n_ic)
  - ic_prob_top1_by_chunk: (n_chunks, n_ic)
  - optional: ic_probs_full_by_chunk: (n_chunks, n_ic, 7)

Features:
  - Slider to browse t from 0s to end.
  - Plot a fixed window (WINDOW_SEC).
  - Each chunk segment is colored by the label at that chunk.
  - Hover shows label and probabilities of the hovered chunk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# --- validation_code paths ---
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

THRESHOLD_ROOT, _, _, _, _ = bootstrap_paths()

# ---------------------------------------------------------------------------
# User config (edit here)
# ---------------------------------------------------------------------------
NPZ_PATH: Optional[_Path] = THRESHOLD_ROOT / "xLapaasrpy20_2min_70" / "b71eeg_ic_sources1.npz"
MAX_COMPONENTS: Optional[int] = None  # None = plot all ICs
WINDOW_SEC: float = 10.0
TIME_START_S: float = 0.0

ICLABEL_CLASS_NAMES: Tuple[str, ...] = (
    "brain",
    "muscle",
    "eye",
    "heart",
    "line_noise",
    "channel_noise",
    "other",
)

CLASS_COLORS: Dict[str, str] = {
    "brain": "#1f77b4",
    "muscle": "#d62728",
    "eye": "#ff7f0e",
    "heart": "#e377c2",
    "line_noise": "#9467bd",
    "channel_noise": "#8c564b",
    "other": "#7f7f7f",
}

_LABEL_ALIASES = {
    "muscle_artifact": "muscle",
    "eye_blink": "eye",
    "heart_beat": "heart",
}


def _norm_label(s: Any) -> str:
    if isinstance(s, bytes):
        t = s.decode("utf-8", errors="ignore")
    else:
        t = str(s)
    t = t.strip().lower().replace(" ", "_")
    return _LABEL_ALIASES.get(t, t)


def _load_npz(path: _Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"IC sources file not found: {path}")
    z = np.load(path, allow_pickle=True)
    sources = z["sources"] if "sources" in z.files else z["data"]
    sources = np.asarray(sources, dtype=np.float64)
    if sources.ndim != 2:
        raise ValueError(f"Expected 2-D sources, got shape {sources.shape}")

    srate = float(np.asarray(z["sampling_rate"]).squeeze()) if "sampling_rate" in z.files else 500.0

    n_ic = sources.shape[0]
    ic_labels = None
    if "ic_labels" in z.files:
        ic_labels = [_norm_label(x) for x in np.asarray(z["ic_labels"]).ravel()[:n_ic]]
    else:
        ic_labels = ["other"] * n_ic

    # Per-chunk label/prob (preferred)
    if "chunk_start_samples" not in z.files or "chunk_sizes" not in z.files:
        raise KeyError(
            "Missing chunk_start_samples/chunk_sizes in npz. "
            "Please regenerate ic_sources with updated receiver.py."
        )
    chunk_start_samples = np.asarray(z["chunk_start_samples"], dtype=np.int64).ravel()
    chunk_sizes = np.asarray(z["chunk_sizes"], dtype=np.int64).ravel()
    if chunk_start_samples.size != chunk_sizes.size:
        raise ValueError("chunk_start_samples and chunk_sizes length mismatch")
    n_chunks = int(chunk_sizes.size)

    if "ic_labels_by_chunk" in z.files:
        ic_labels_by_chunk = np.asarray(z["ic_labels_by_chunk"], dtype=object)
        if ic_labels_by_chunk.shape[0] != n_chunks:
            raise ValueError("ic_labels_by_chunk first dim must equal n_chunks")
    else:
        # fallback: use last-snapshot labels for all chunks
        ic_labels_by_chunk = np.asarray([ic_labels] * n_chunks, dtype=object)

    ic_prob_top1_by_chunk = None
    if "ic_prob_top1_by_chunk" in z.files:
        ic_prob_top1_by_chunk = np.asarray(z["ic_prob_top1_by_chunk"], dtype=np.float64)
        if ic_prob_top1_by_chunk.shape[0] != n_chunks:
            ic_prob_top1_by_chunk = None

    ic_probs_full_by_chunk = None
    if "ic_probs_full_by_chunk" in z.files:
        ic_probs_full_by_chunk = np.asarray(z["ic_probs_full_by_chunk"], dtype=np.float64)
        if ic_probs_full_by_chunk.shape[0] != n_chunks:
            ic_probs_full_by_chunk = None

    class_names = list(ICLABEL_CLASS_NAMES)
    if "ic_label_classes" in z.files:
        class_names = [str(x) for x in np.asarray(z["ic_label_classes"]).ravel()]

    artifact_idx = []
    if "artifact_indices" in z.files:
        artifact_idx = list(np.asarray(z["artifact_indices"]).astype(int).ravel())

    return {
        "path": path,
        "sources": sources,
        "srate": srate,
        "ic_labels": ic_labels,
        "chunk_start_samples": chunk_start_samples,
        "chunk_sizes": chunk_sizes,
        "ic_labels_by_chunk": ic_labels_by_chunk,
        "ic_prob_top1_by_chunk": ic_prob_top1_by_chunk,
        "ic_probs_full_by_chunk": ic_probs_full_by_chunk,
        "class_names": class_names,
        "artifact_indices": artifact_idx,
    }


def _prob_hover_text(
    ic_idx: int,
    label: str,
    prob_top1: Optional[float],
    probs_full_row: Optional[np.ndarray],
    class_names: Sequence[str],
) -> str:
    lines = [f"IC {ic_idx}", f"label: {label}"]
    if prob_top1 is not None and np.isfinite(prob_top1):
        lines.append(f"top-1 prob: {prob_top1:.3f}")
    if probs_full_row is not None:
        row = probs_full_row
        n = min(len(row), len(class_names))
        parts = []
        for j in range(n):
            if np.isfinite(row[j]):
                parts.append((class_names[j], float(row[j])))
        parts.sort(key=lambda x: -x[1])
        lines.append("probs:")
        for name, p in parts[:7]:
            lines.append(f"  {name}: {p:.3f}")
    return "\n".join(lines)


def _attach_hover(
    fig: plt.Figure,
    ax_list: List[plt.Axes],
    state: dict,
) -> None:
    annot = fig.text(
        0.02,
        0.98,
        "",
        transform=fig.transFigure,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.92),
        visible=False,
    )

    def _on_move(event):
        if event.inaxes is None or event.xdata is None:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        try:
            ax_idx = ax_list.index(event.inaxes)
        except ValueError:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return

        # hovered time -> sample -> chunk index
        t = float(event.xdata)
        srate = float(state["srate"])
        sample = int(round(t * srate))
        starts = state["chunk_start_samples"]
        sizes = state["chunk_sizes"]
        j = int(np.searchsorted(starts, sample, side="right") - 1)
        if j < 0:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        j = min(j, len(starts) - 1)
        if sample >= int(starts[j] + sizes[j]):
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return

        ic_idx = int(state["ic_indices"][ax_idx])
        label = str(state["ic_labels_by_chunk"][j][ic_idx])

        prob_top1 = None
        if state.get("ic_prob_top1_by_chunk") is not None:
            try:
                prob_top1 = float(state["ic_prob_top1_by_chunk"][j][ic_idx])
            except Exception:
                prob_top1 = None

        probs_full_row = None
        if state.get("ic_probs_full_by_chunk") is not None:
            try:
                probs_full_row = state["ic_probs_full_by_chunk"][j][ic_idx]
            except Exception:
                probs_full_row = None

        y = state["sources"][ic_idx]
        i_near = int(np.clip(sample, 0, y.size - 1))

        annot.set_text(
            _prob_hover_text(
                ic_idx,
                label,
                prob_top1,
                probs_full_row,
                state.get("class_names", ICLABEL_CLASS_NAMES),
            )
            + f"\n\nchunk={j}  t={t:.3f}s  value={float(y[i_near]):.4g}"
        )
        annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)


def plot_ic_sources(bundle: dict, max_components: Optional[int] = None, t0: float = 0.0) -> None:
    sources = bundle["sources"]
    srate = float(bundle["srate"])
    n_ic, n_samp = sources.shape
    total_sec = n_samp / srate

    n_plot = n_ic if max_components is None else min(n_ic, int(max_components))
    ic_indices = list(range(n_plot))

    fig_h = max(4.5, 1.45 * n_plot)
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, fig_h), sharex=True)
    if n_plot == 1:
        axes = [axes]

    # Leave space for slider
    plt.subplots_adjust(bottom=0.10)
    slider_ax = fig.add_axes([0.10, 0.03, 0.80, 0.03])
    s_max = max(0.0, total_sec - WINDOW_SEC)
    slider = Slider(slider_ax, "t (s)", 0.0, s_max, valinit=max(0.0, min(t0, s_max)))

    starts = np.asarray(bundle["chunk_start_samples"], dtype=np.int64)
    sizes = np.asarray(bundle["chunk_sizes"], dtype=np.int64)
    labels_by_chunk = np.asarray(bundle["ic_labels_by_chunk"], dtype=object)
    prob_by_chunk = bundle.get("ic_prob_top1_by_chunk")
    probs_full_by_chunk = bundle.get("ic_probs_full_by_chunk")
    class_names = bundle.get("class_names", ICLABEL_CLASS_NAMES)

    state = {
        "srate": srate,
        "sources": sources,
        "chunk_start_samples": starts,
        "chunk_sizes": sizes,
        "ic_labels_by_chunk": labels_by_chunk,
        "ic_prob_top1_by_chunk": prob_by_chunk,
        "ic_probs_full_by_chunk": probs_full_by_chunk,
        "class_names": class_names,
        "ic_indices": ic_indices,
    }
    _attach_hover(fig, list(axes), state)

    def _draw_window(t_start: float) -> None:
        i0 = int(round(t_start * srate))
        i1 = int(min(n_samp, round((t_start + WINDOW_SEC) * srate)))
        if i1 <= i0:
            return

        j0 = int(np.searchsorted(starts, i0, side="right") - 1)
        j0 = max(j0, 0)
        j1 = int(np.searchsorted(starts, i1, side="left"))
        j1 = min(j1 + 1, len(starts))

        for k, ic in enumerate(ic_indices):
            ax = axes[k]
            ax.cla()
            ax.grid(True, alpha=0.25)

            for j in range(j0, j1):
                cs = int(starts[j])
                ce = int(starts[j] + sizes[j])
                seg0 = max(i0, cs)
                seg1 = min(i1, ce)
                if seg1 <= seg0:
                    continue
                lbl = str(labels_by_chunk[j][ic])
                col = CLASS_COLORS.get(lbl, CLASS_COLORS["other"])
                y = sources[ic, seg0:seg1]
                x = np.arange(seg0, seg1, dtype=np.float64) / srate
                ax.plot(x, y, color=col, linewidth=0.7, alpha=0.95)

            mid_sample = (i0 + i1) // 2
            j_mid = int(np.searchsorted(starts, mid_sample, side="right") - 1)
            j_mid = max(0, min(j_mid, len(starts) - 1))
            lbl_mid = str(labels_by_chunk[j_mid][ic])
            col_mid = CLASS_COLORS.get(lbl_mid, CLASS_COLORS["other"])
            p_mid = None
            if prob_by_chunk is not None:
                try:
                    p_mid = float(prob_by_chunk[j_mid][ic])
                except Exception:
                    p_mid = None
            p_txt = f" p={p_mid:.2f}" if p_mid is not None and np.isfinite(p_mid) else ""
            ax.set_ylabel(f"IC{ic}", fontsize=8)
            ax.set_title(f"IC {ic}: {lbl_mid}{p_txt}", fontsize=9, color=col_mid)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"IC sources (per-chunk ICLabel) — {bundle['path'].name}\n"
            f"fs={srate:.1f} Hz  window=[{i0/srate:.2f}, {i1/srate:.2f}] s",
            fontsize=11,
            y=0.995,
        )
        fig.canvas.draw_idle()

    slider.on_changed(lambda v: _draw_window(float(v)))
    _draw_window(float(slider.val))
    plt.show()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot IC sources from eeg_ic_sources1.npz")
    p.add_argument(
        "--npz",
        type=_Path,
        default=NPZ_PATH,
        help="Path to *eeg_ic_sources1.npz (default: NPZ_PATH in script)",
    )
    p.add_argument("--max-ic", type=int, default=MAX_COMPONENTS)
    p.add_argument("--t-start", type=float, default=TIME_START_S)
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.npz is None:
        raise SystemExit(
            "Set NPZ_PATH in check_sources.py or pass --npz path/to/bXXeeg_ic_sources1.npz"
        )
    bundle = _load_npz(_Path(args.npz))
    print(f"Loaded: {bundle['path']}")
    print(f"  sources shape: {bundle['sources'].shape} (n_ic, n_samples)")
    print(f"  labels: {bundle['ic_labels']}")
    global WINDOW_SEC
    WINDOW_SEC = float(args.window_sec)
    plot_ic_sources(bundle, max_components=args.max_ic, t0=args.t_start)


if __name__ == "__main__":
    main()
