"""
读取 Input_data/npz/ 下全部 .npz，对 EEG 做 **60 Hz 工频陷波**（离线、零相位），
目录结构与 npz/ 镜像对应：

  npz/Shawn_shared/foo.npz  → npz_offline_notch/Shawn_shared/foo.npz

仅当输出 npz 不存在时才转换（与 convert_set_to_npz.py 相同逻辑）。

用法:
  python apply_notch60_to_npz.py
  python apply_notch60_to_npz.py --check-only
  python apply_notch60_to_npz.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import mne
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NPZ_ROOT = SCRIPT_DIR.parent / "npz"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR

NOTCH_FREQ_HZ = 60.0


class NotchStats(Dict[str, int]):
    pass


def _pick_srate(z: np.lib.npyio.NpzFile, fallback: float) -> float:
    for key in ("sampling_rate", "srate", "fs"):
        if key in z:
            return float(np.asarray(z[key]).squeeze())
    return float(fallback)


def _mirror_out_dir(in_dir: Path) -> Path:
    in_dir = in_dir.resolve()
    npz_root = DEFAULT_NPZ_ROOT.resolve()
    if in_dir == npz_root:
        return DEFAULT_OUTPUT_ROOT.resolve()
    try:
        rel = in_dir.relative_to(npz_root)
    except ValueError:
        return DEFAULT_OUTPUT_ROOT.resolve()
    return (DEFAULT_OUTPUT_ROOT / rel).resolve()


def _default_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if not DEFAULT_NPZ_ROOT.is_dir():
        return pairs
    if any(DEFAULT_NPZ_ROOT.glob("*.npz")):
        pairs.append((DEFAULT_NPZ_ROOT, DEFAULT_OUTPUT_ROOT))
    for sub in sorted(p for p in DEFAULT_NPZ_ROOT.iterdir() if p.is_dir()):
        pairs.append((sub, DEFAULT_OUTPUT_ROOT / sub.name))
    return pairs


def load_npz_eeg(npz_path: Path) -> mne.io.BaseRaw:
    if not npz_path.exists():
        raise FileNotFoundError(str(npz_path))

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
            print(
                f"   [WARN] channels 长度 {len(channels)} != {data.shape[0]}，改用 Ch01.."
            )
            channels = [f"Ch{i + 1}" for i in range(data.shape[0])]

        print(f"   [LOAD] 键={key_used} shape={data.shape} fs={srate} Hz")

    data_v = np.ascontiguousarray(data.astype(np.float64) * 1e-6)
    info = mne.create_info(ch_names=list(channels), sfreq=srate, ch_types="eeg")
    raw = mne.io.RawArray(data_v, info, verbose=False)
    return raw


def apply_notch(raw: mne.io.BaseRaw, freq_hz: float) -> None:
    sfreq = float(raw.info["sfreq"])
    nyq = 0.5 * sfreq
    if freq_hz >= nyq - 0.5:
        raise ValueError(f"陷波 {freq_hz} Hz 必须 < 奈奎斯特 {nyq:.1f} Hz")
    raw.notch_filter(freqs=freq_hz, picks="eeg", verbose=False)


def raw_to_uv_npz_payload(raw: mne.io.BaseRaw, freq_hz: float) -> dict:
    data_uv = (raw.get_data().astype(np.float32) * 1e6)
    fs = float(raw.info["sfreq"])
    ch = np.asarray(raw.ch_names, dtype=object)
    return {
        "data": np.ascontiguousarray(data_uv),
        "sampling_rate": np.asarray(fs),
        "channels": ch,
        "note": np.asarray(
            f"Notch {freq_hz:g} Hz (MNE raw.notch_filter), microvolts, offline zero-phase",
            dtype=object,
        ),
    }


def save_npz(path: Path, payload: dict, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    if extra:
        out.update(extra)
    if "calibration_data" not in out and "data" in out:
        out["calibration_data"] = out["data"]
    np.savez_compressed(path, **out)


def convert_dir(
    in_dir: Path,
    out_dir: Path,
    pattern: str,
    freq_hz: float,
    *,
    skip_existing: bool = True,
    check_only: bool = False,
) -> NotchStats:
    stats: NotchStats = {
        "total": 0,
        "missing": 0,
        "converted": 0,
        "skipped": 0,
        "errors": 0,
    }

    files = sorted(in_dir.glob(pattern))
    if not files:
        print(f"[WARN] {in_dir} 下无匹配 {pattern!r}，跳过")
        return stats

    print(f"[INFO] input : {in_dir}")
    print(f"[INFO] output: {out_dir}")
    print(f"[INFO] .npz  : {len(files)}")

    stats["total"] = len(files)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in files:
        out = out_dir / src.name
        exists = out.is_file()
        if exists and skip_existing and not check_only:
            print(f"→ {src.name}  [SKIP] 已有 {out.relative_to(DEFAULT_OUTPUT_ROOT)}")
            stats["skipped"] += 1
            continue
        if not exists:
            stats["missing"] += 1
            tag = "[MISSING]"
        elif check_only:
            tag = "[OK]"
        else:
            tag = "[FORCE]"

        if check_only:
            print(f"→ {src.name}  {tag}  {out.relative_to(DEFAULT_OUTPUT_ROOT)}")
            if exists:
                stats["skipped"] += 1
            continue

        try:
            print(f"→ {src.name}  {tag}")
            raw = load_npz_eeg(src)
            apply_notch(raw, freq_hz)
            payload = raw_to_uv_npz_payload(raw, freq_hz)
            save_npz(
                out,
                payload,
                {"source_npz": np.asarray(str(src.relative_to(DEFAULT_NPZ_ROOT)), dtype=object)},
            )
            print(f"   [OK] {out.relative_to(DEFAULT_OUTPUT_ROOT)}  shape={payload['data'].shape}")
            stats["converted"] += 1
        except Exception as e:
            print(f"   [ERR] {src.name}: {e}", file=sys.stderr)
            stats["errors"] += 1

    return stats


def _merge_stats(a: NotchStats, b: NotchStats) -> NotchStats:
    return {k: a[k] + b[k] for k in a}


def _print_summary(stats: NotchStats, check_only: bool) -> None:
    action_key = "待转换" if check_only else "已转换"
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
        description="Input_data/npz/ → 离线工频陷波 → npz_offline_notch/（镜像子目录）"
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="含 .npz 的目录；默认扫描 npz/ 下全部子目录及根目录",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出根目录；与 --input-dir 成对使用，或留空以自动镜像到 npz_offline_notch/",
    )
    ap.add_argument(
        "--freq",
        type=float,
        default=NOTCH_FREQ_HZ,
        help="陷波中心频率 Hz（默认 60）",
    )
    ap.add_argument("--pattern", type=str, default="*.npz")
    ap.add_argument(
        "--force",
        action="store_true",
        help="即使输出 npz 已存在也重新生成（默认只补缺失）",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="只对比 npz 与 npz_offline_notch，不执行转换",
    )
    args = ap.parse_args()

    skip_existing = not args.force
    check_only = args.check_only
    freq = float(args.freq)

    def run_pair(in_dir: Path, out_dir: Path) -> NotchStats:
        return convert_dir(
            in_dir,
            out_dir,
            args.pattern,
            freq,
            skip_existing=skip_existing,
            check_only=check_only,
        )

    if args.input_dir is not None:
        in_dir = args.input_dir.resolve()
        if args.output_dir is not None:
            out_dir = args.output_dir.resolve()
        else:
            out_dir = _mirror_out_dir(in_dir)
        if not in_dir.is_dir():
            print(f"[ERR] 输入目录不存在: {in_dir}", file=sys.stderr)
            return 1
        stats = run_pair(in_dir, out_dir)
        _print_summary(stats, check_only)
        if stats["total"] == 0:
            return 1
        if stats["errors"] > 0:
            return 1
    else:
        pairs = _default_pairs()
        if not pairs:
            print(f"[ERR] 未找到 {DEFAULT_NPZ_ROOT} 或其子目录", file=sys.stderr)
            return 1
        stats: NotchStats = {
            "total": 0,
            "missing": 0,
            "converted": 0,
            "skipped": 0,
            "errors": 0,
        }
        for in_dir, out_dir in pairs:
            stats = _merge_stats(stats, run_pair(in_dir, out_dir))
        _print_summary(stats, check_only)
        if stats["total"] == 0:
            return 1
        if stats["errors"] > 0:
            return 1

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
