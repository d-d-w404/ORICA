"""
从 Input_data/npz_offline_notch/ 生成 ASR 校准数据，目录结构镜像 npz_offline_notch/：

  npz_offline_notch/Shawn_shared/foo.npz
    → asr_cali_offline_notch/Shawn_shared/full/foo.npz
    → asr_cali_offline_notch/Shawn_shared/2min/foo.npz

默认只补缺失文件；已有则跳过。

用法:
  python filter_driving_sets_iir_1_50.py
  python filter_driving_sets_iir_1_50.py --check-only
  python filter_driving_sets_iir_1_50.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR.parent / "npz_offline_notch"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR

L_FREQ = 1.0
H_FREQ = 50.0
FIRST_MINUTES = 2.0
FULL_SUBDIR = "full"
HEAD_SUBDIR = "2min"


class BuildStats(Dict[str, int]):
    pass


def _pick_srate(z: np.lib.npyio.NpzFile, fallback: float) -> float:
    for key in ("sampling_rate", "srate", "fs"):
        if key in z:
            return float(np.asarray(z[key]).squeeze())
    return float(fallback)


def _collect_input_files(root: Path, pattern: str) -> List[Tuple[Path, Path]]:
    """返回 (npz 路径, 相对 input root 的子目录)。"""
    pairs: List[Tuple[Path, Path]] = []
    if not root.is_dir():
        return pairs
    root = root.resolve()
    if any(root.glob(pattern)):
        for p in sorted(root.glob(pattern)):
            pairs.append((p, Path(".")))
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(sub.glob(pattern)):
            pairs.append((p, Path(sub.name)))
    return pairs


def _mirror_out_dir(out_root: Path, rel_sub: Path, kind: str) -> Path:
    if rel_sub == Path("."):
        return out_root / kind
    return out_root / rel_sub / kind


def load_npz_eeg(npz_path: Path) -> mne.io.BaseRaw:
    with np.load(npz_path, allow_pickle=True) as z:
        if "cleaned_data" in z:
            data = np.asarray(z["cleaned_data"], dtype=np.float64)
            key_used = "cleaned_data"
        elif "data" in z:
            data = np.asarray(z["data"], dtype=np.float64)
            key_used = "data"
        else:
            raise KeyError(f"{npz_path.name} 无 cleaned_data/data，键: {list(z.files)}")

        if data.ndim != 2:
            raise ValueError(f"{npz_path.name}: 期望 2D，得到 {data.shape}")

        if data.shape[0] > data.shape[1] * 10:
            data = np.ascontiguousarray(data.T)
            print(f"   [INFO] 已转置为 (channels, samples): {data.shape}")

        srate = _pick_srate(z, 500.0)

        if "channels" in z:
            ch = z["channels"]
            if isinstance(ch, np.ndarray):
                channels = [str(c) for c in ch.tolist()]
            else:
                channels = [str(c) for c in list(ch)]
        else:
            channels = [f"Ch{i + 1}" for i in range(data.shape[0])]

        if len(channels) != data.shape[0]:
            channels = [f"Ch{i + 1}" for i in range(data.shape[0])]

        print(f"   [LOAD] 键={key_used} shape={data.shape} fs={srate} Hz")

    data_v = np.ascontiguousarray(data.astype(np.float64) * 1e-6)
    info = mne.create_info(ch_names=list(channels), sfreq=srate, ch_types="eeg")
    return mne.io.RawArray(data_v, info, verbose=False)


def apply_iir_bandpass(raw: mne.io.BaseRaw) -> None:
    raw.filter(
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        picks="eeg",
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )


def raw_to_uv_npz_payload(raw: mne.io.BaseRaw, note: str) -> Dict[str, Any]:
    data_uv = (raw.get_data().astype(np.float32) * 1e6)
    fs = float(raw.info["sfreq"])
    ch = np.asarray(raw.ch_names, dtype=object)
    return {
        "data": np.ascontiguousarray(data_uv),
        "sampling_rate": np.asarray(fs),
        "channels": ch,
        "note": np.asarray(note, dtype=object),
    }


def save_npz(path: Path, payload: Dict[str, Any], extra_keys: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    if extra_keys:
        out.update(extra_keys)
    if "calibration_data" not in out and "data" in out:
        out["calibration_data"] = out["data"]
    np.savez_compressed(path, **out)


def process_one(
    src: Path,
    rel_sub: Path,
    out_root: Path,
    input_root: Path,
    first_minutes: float,
    *,
    skip_existing: bool,
    check_only: bool,
    stats: BuildStats,
) -> None:
    out_full = _mirror_out_dir(out_root, rel_sub, FULL_SUBDIR) / src.name
    out_head = _mirror_out_dir(out_root, rel_sub, HEAD_SUBDIR) / src.name
    rel_label = out_full.relative_to(out_root)

    full_exists = out_full.is_file()
    head_exists = out_head.is_file()
    if full_exists and head_exists and skip_existing and not check_only:
        print(f"→ {src.name}  [SKIP]  {rel_label.parent}")
        stats["skipped"] += 2
        return

    if check_only:
        for out in (out_full, out_head):
            exists = out.is_file()
            tag = "[OK]" if exists else "[MISSING]"
            print(f"→ {src.name}  {tag}  {out.relative_to(out_root)}")
            if exists:
                stats["skipped"] += 1
            else:
                stats["missing"] += 1
        return

    print(f"→ {src.name}  [{rel_sub}]")
    raw = load_npz_eeg(src)
    apply_iir_bandpass(raw)

    full_payload = raw_to_uv_npz_payload(
        raw,
        f"IIR {L_FREQ}-{H_FREQ} Hz Butterworth order=4, microvolts",
    )
    source_rel = str(src.relative_to(input_root.resolve()))
    if not full_exists or not skip_existing:
        save_npz(
            out_full,
            full_payload,
            {"source_npz": np.asarray(source_rel, dtype=object)},
        )
        print(f"   [OK] full: {out_full.relative_to(out_root)}  {full_payload['data'].shape}")
        stats["converted"] += 1
    else:
        print(f"   [SKIP] full: {out_full.relative_to(out_root)}")
        stats["skipped"] += 1

    fs = float(full_payload["sampling_rate"].squeeze())
    n_keep = min(
        full_payload["data"].shape[1],
        int(round(first_minutes * 60.0 * fs)),
    )
    head_payload = {k: v for k, v in full_payload.items()}
    head_payload["data"] = np.ascontiguousarray(full_payload["data"][:, :n_keep])
    head_payload["calibration_data"] = head_payload["data"]
    head_payload["note"] = np.asarray(
        f"IIR {L_FREQ}-{H_FREQ} Hz, first {first_minutes:g} min only, microvolts",
        dtype=object,
    )
    if not head_exists or not skip_existing:
        save_npz(
            out_head,
            head_payload,
            {
                "source_npz": np.asarray(source_rel, dtype=object),
                "trim_minutes": np.asarray(float(first_minutes)),
            },
        )
        print(f"   [OK] 2min: {out_head.relative_to(out_root)}  {head_payload['data'].shape}")
        stats["converted"] += 1
    else:
        print(f"   [SKIP] 2min: {out_head.relative_to(out_root)}")
        stats["skipped"] += 1


def _print_summary(stats: BuildStats, check_only: bool) -> None:
    action_key = "待生成" if check_only else "已生成"
    action_val = stats["missing"] if check_only else stats["converted"]
    print(
        f"[SUMMARY] 输入={stats['total']}  "
        f"缺失={stats['missing']}  "
        f"已存在={stats['skipped']}  "
        f"{action_key}={action_val}  "
        f"失败={stats['errors']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="npz_offline_notch/ → IIR 1–50 Hz → asr_cali_offline_notch/{subdir}/full|2min/"
    )
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--first-minutes", type=float, default=FIRST_MINUTES)
    ap.add_argument("--pattern", type=str, default="*.npz")
    ap.add_argument("--force", action="store_true", help="覆盖已有 asr_cali_offline_notch 文件")
    ap.add_argument("--check-only", action="store_true", help="只对比，不生成")
    args = ap.parse_args()

    in_root = args.input_dir.resolve()
    out_root = args.output_dir.resolve()
    skip_existing = not args.force
    check_only = args.check_only

    if not in_root.is_dir():
        print(f"[ERR] 输入目录不存在: {in_root}", file=sys.stderr)
        return 1

    items = _collect_input_files(in_root, args.pattern)
    if not items:
        print(f"[ERR] {in_root} 下无匹配 {args.pattern!r}", file=sys.stderr)
        return 1

    stats: BuildStats = {
        "total": len(items),
        "missing": 0,
        "converted": 0,
        "skipped": 0,
        "errors": 0,
    }

    print(f"[INFO] input : {in_root}")
    print(f"[INFO] output: {out_root}")
    print(f"[INFO] IIR   : {L_FREQ}–{H_FREQ} Hz")
    print(f"[INFO] files : {len(items)}")

    for src, rel_sub in items:
        try:
            process_one(
                src,
                rel_sub,
                out_root,
                in_root,
                float(args.first_minutes),
                skip_existing=skip_existing,
                check_only=check_only,
                stats=stats,
            )
        except Exception as e:
            print(f"   [ERR] {src.name}: {e}", file=sys.stderr)
            stats["errors"] += 1

    _print_summary(stats, check_only)
    if stats["errors"] > 0:
        return 1
    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
