"""
将 Input_data 下 EEGLAB .set 转为 .npz，目录结构与 set/ 镜像对应。

默认扫描 set/ 下全部内容：
  - set/<任意子目录>/*.set  ↔ npz/<同名子目录>/*.npz
  - set/*.set（根目录）     ↔ npz/*.npz

仅当 npz 中缺少对应文件时才转换（.set 读取时会自动关联同名 .fdt）。
npz 键：data (channels, samples)，单位 µV；sampling_rate；channels

用法:
  python convert_set_to_npz.py                  # 只补缺失的 npz
  python convert_set_to_npz.py --check-only     # 只对比，不转换
  python convert_set_to_npz.py --force          # 全部重新生成
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SET_ROOT = SCRIPT_DIR / "set"
DEFAULT_NPZ_ROOT = SCRIPT_DIR / "npz"


class ConvertStats(Dict[str, int]):
    pass


def _eeglab_chanlocs_to_names(eeg, n_ch: int) -> List[str]:
    names: list = []
    try:
        cl = eeg.chanlocs
        if cl is None:
            return [f"Ch{i + 1}" for i in range(n_ch)]
        arr = np.squeeze(np.asarray(cl, dtype=object))
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.size == 0:
            return [f"Ch{i + 1}" for i in range(n_ch)]
        n_loc = min(n_ch, int(arr.shape[0]))
        for i in range(n_loc):
            row = arr[i]
            lab = getattr(row, "labels", None)
            if lab is None and isinstance(row, np.void) and row.dtype.names and "labels" in row.dtype.names:
                lab = row["labels"]
            if isinstance(lab, (bytes, bytearray)):
                names.append(lab.decode("utf-8", errors="ignore").strip())
            elif lab is not None:
                names.append(str(np.asarray(lab).squeeze()).strip())
            else:
                names.append(f"Ch{i + 1}")
        while len(names) < n_ch:
            names.append(f"Ch{len(names) + 1}")
        return names[:n_ch]
    except Exception:
        return [f"Ch{i + 1}" for i in range(n_ch)]


def _load_set_scipy(set_path: Path) -> Tuple[np.ndarray, float, List[str]]:
    import scipy.io as sio

    try:
        d = sio.loadmat(str(set_path), struct_as_record=False, squeeze_me=True)
    except NotImplementedError as e:
        raise RuntimeError(
            "该 .set 为 MATLAB v7.3 (HDF5)，请在 EEGLAB 另存为 v6 .set。"
        ) from e

    if "EEG" not in d:
        raise ValueError("MAT 中无 EEG 变量")

    eeg = d["EEG"]
    if isinstance(eeg, np.ndarray) and eeg.dtype == object:
        eeg = eeg.item()

    data = np.asarray(eeg.data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"EEG.data 期望 2 维，得到 shape={data.shape}")

    srate = float(np.asarray(eeg.srate).squeeze())
    nbchan = int(np.round(float(np.asarray(eeg.nbchan).squeeze())))

    if data.shape[0] != nbchan and data.shape[1] == nbchan:
        data = data.T
    n_ch = data.shape[0]
    ch_names = _eeglab_chanlocs_to_names(eeg, n_ch)
    # EEGLAB 连续数据通常为 µV
    data_uv = np.ascontiguousarray(data.astype(np.float32))
    return data_uv, srate, ch_names


def load_set_to_uv(set_path: Path) -> Tuple[np.ndarray, float, List[str]]:
    try:
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    except KeyError as e:
        if e.args and e.args[0] == "nodatchans":
            print(f"   [INFO] scipy 回退: {set_path.name}")
            return _load_set_scipy(set_path)
        raise

    raw.pick_types(eeg=True, exclude=[])
    srate = float(raw.info["sfreq"])
    data_uv = (raw.get_data().astype(np.float32) * 1e6)
    ch_names = list(raw.ch_names)
    return data_uv, srate, ch_names


def ensure_ch_samples(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"期望 2D，得到 {data.shape}")
    if data.shape[0] > data.shape[1] * 10:
        return np.ascontiguousarray(data.T)
    return np.ascontiguousarray(data)


def save_npz(
    out_path: Path,
    data_uv: np.ndarray,
    srate: float,
    channels: List[str],
    source_name: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        data=data_uv.astype(np.float32),
        sampling_rate=np.asarray(float(srate)),
        channels=np.asarray(channels, dtype=object),
        source_set=np.asarray(source_name, dtype=object),
    )


def _mirror_out_dir(in_dir: Path) -> Path:
    """set/ 内任意路径 → npz/ 下镜像路径。"""
    in_dir = in_dir.resolve()
    set_root = DEFAULT_SET_ROOT.resolve()
    if in_dir == set_root:
        return DEFAULT_NPZ_ROOT.resolve()
    try:
        rel = in_dir.relative_to(set_root)
    except ValueError:
        return DEFAULT_NPZ_ROOT.resolve()
    return (DEFAULT_NPZ_ROOT / rel).resolve()


def _default_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if not DEFAULT_SET_ROOT.is_dir():
        return pairs

    if any(DEFAULT_SET_ROOT.glob("*.set")) or any(DEFAULT_SET_ROOT.glob("*.fdt")):
        pairs.append((DEFAULT_SET_ROOT, DEFAULT_NPZ_ROOT))

    for sub in sorted(p for p in DEFAULT_SET_ROOT.iterdir() if p.is_dir()):
        pairs.append((sub, DEFAULT_NPZ_ROOT / sub.name))
    return pairs


def _orphan_fdt_files(in_dir: Path) -> list[Path]:
    set_stems = {p.stem for p in in_dir.glob("*.set")}
    return sorted(
        p for p in in_dir.glob("*.fdt") if p.stem not in set_stems
    )


def convert_dir(
    in_dir: Path,
    out_dir: Path,
    pattern: str,
    *,
    skip_existing: bool = True,
    check_only: bool = False,
) -> ConvertStats:
    stats: ConvertStats = {
        "total_sets": 0,
        "missing": 0,
        "converted": 0,
        "skipped": 0,
        "errors": 0,
        "orphan_fdt": 0,
    }

    files = sorted(in_dir.glob(pattern))
    orphans = _orphan_fdt_files(in_dir)
    stats["orphan_fdt"] = len(orphans)

    if not files and not orphans:
        print(f"[WARN] {in_dir} 下无匹配 {pattern!r}，跳过")
        return stats

    print(f"[INFO] input : {in_dir}")
    print(f"[INFO] output: {out_dir}")
    print(f"[INFO] .set  : {len(files)}")
    if orphans:
        print(f"[WARN] 孤立 .fdt（无同名 .set，无法单独转换）: {len(orphans)}")
        for p in orphans:
            print(f"        {p.name}")

    stats["total_sets"] = len(files)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in files:
        out = out_dir / f"{src.stem}.npz"
        exists = out.is_file()
        if exists and skip_existing and not check_only:
            print(f"→ {src.name}  [SKIP] 已有 {out.name}")
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
            print(f"→ {src.name}  {tag}  {out.name}")
            if exists:
                stats["skipped"] += 1
            continue

        try:
            print(f"→ {src.name}  {tag}")
            data_uv, srate, ch = load_set_to_uv(src)
            data_uv = ensure_ch_samples(data_uv)
            if data_uv.shape[0] != len(ch):
                print(f"   [WARN] 通道名数 {len(ch)} 与数据行数 {data_uv.shape[0]} 不一致，改用 Ch01..")
                ch = [f"Ch{i + 1}" for i in range(data_uv.shape[0])]
            save_npz(out, data_uv, srate, ch, src.name)
            print(f"   [OK] {out.name}  shape={data_uv.shape}  {srate} Hz")
            stats["converted"] += 1
        except Exception as e:
            print(f"   [ERR] {src.name}: {e}", file=sys.stderr)
            stats["errors"] += 1

    return stats


def _merge_stats(a: ConvertStats, b: ConvertStats) -> ConvertStats:
    return {k: a[k] + b[k] for k in a}


def _print_summary(stats: ConvertStats, check_only: bool) -> None:
    action_key = "待转换" if check_only else "已转换"
    action_val = stats["missing"] if check_only else stats["converted"]
    print(
        f"[SUMMARY] .set={stats['total_sets']}  "
        f"缺失={stats['missing']}  "
        f"已存在={stats['skipped']}  "
        f"{action_key}={action_val}  "
        f"失败={stats['errors']}  "
        f"孤立.fdt={stats['orphan_fdt']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Driving .set → npz/（镜像 set/ 全部子目录）")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="含 .set 的目录；默认扫描 set/ 下全部子目录及根目录",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出 .npz 目录；与 --input-dir 成对使用，或留空以自动镜像到 npz/",
    )
    ap.add_argument("--pattern", type=str, default="*.set")
    ap.add_argument(
        "--force",
        action="store_true",
        help="即使 npz 已存在也重新生成（默认只补缺失）",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="只对比 set 与 npz，不执行转换",
    )
    args = ap.parse_args()

    skip_existing = not args.force
    check_only = args.check_only

    def run_pair(in_dir: Path, out_dir: Path) -> ConvertStats:
        return convert_dir(
            in_dir,
            out_dir,
            args.pattern,
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
        if stats["total_sets"] == 0 and stats["orphan_fdt"] == 0:
            return 1
        if stats["errors"] > 0:
            return 1
    else:
        pairs = _default_pairs()
        if not pairs:
            print(f"[ERR] 未找到 {DEFAULT_SET_ROOT} 或其子目录", file=sys.stderr)
            return 1
        stats: ConvertStats = {
            "total_sets": 0,
            "missing": 0,
            "converted": 0,
            "skipped": 0,
            "errors": 0,
            "orphan_fdt": 0,
        }
        for in_dir, out_dir in pairs:
            stats = _merge_stats(stats, run_pair(in_dir, out_dir))
        _print_summary(stats, check_only)
        if stats["total_sets"] == 0 and stats["orphan_fdt"] == 0:
            return 1
        if stats["errors"] > 0:
            return 1

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
