"""
列出本目录（或指定目录）下所有 .npz 的通道信息。

与 Driving 管线 npz 约定一致：优先读 channels；若无则按 data/cleaned_data 第一维推断 Ch1..

用法:
  python list_npz_channels.py
  python list_npz_channels.py --dir npz/Shawn_shared
  python list_npz_channels.py --pattern "s01*.npz"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NPZ_DIR = SCRIPT_DIR / "npz" / "Shawn_shared"


def _channels_from_npz(z: np.lib.npyio.NpzFile, n_ch: int) -> List[str]:
    if "channels" not in z:
        return [f"Ch{i + 1}" for i in range(n_ch)]
    ch = z["channels"]
    if isinstance(ch, np.ndarray):
        names = [str(c) for c in ch.tolist()]
    else:
        names = [str(c) for c in list(ch)]
    if len(names) != n_ch:
        print(
            f"      [WARN] channels 长度 {len(names)} != 数据行数 {n_ch}，按 Ch1.. 展示",
            file=sys.stderr,
        )
        return [f"Ch{i + 1}" for i in range(n_ch)]
    return names


def inspect_one(path: Path) -> None:
    print(f"\n{'=' * 60}\n文件: {path.name}\n{'=' * 60}")
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.files)
        print(f"  NPZ 键: {keys}")

        if "cleaned_data" in z:
            arr = np.asarray(z["cleaned_data"])
            data_key = "cleaned_data"
        elif "data" in z:
            arr = np.asarray(z["data"])
            data_key = "data"
        else:
            print("  [SKIP] 无 cleaned_data / data，无法推断通道维")
            return

        if arr.ndim != 2:
            print(f"  [SKIP] {data_key} 非 2D: shape={arr.shape}")
            return

        data = arr
        if data.shape[0] > data.shape[1] * 10:
            data = np.asarray(data).T
            print(
                f"  [INFO] 假定原排列为 (samples, channels)，已转置为 (channels, samples) = {data.shape}"
            )

        n_ch, n_samp = int(data.shape[0]), int(data.shape[1])
        names = _channels_from_npz(z, n_ch)

        for rate_key in ("sampling_rate", "srate", "fs"):
            if rate_key in z:
                fs = float(np.asarray(z[rate_key]).squeeze())
                print(f"  采样率 ({rate_key}): {fs} Hz")
                break
        else:
            print("  采样率: (未找到 sampling_rate / srate / fs)")

        print(f"  数据键: {data_key}  shape=(channels, samples)=({n_ch}, {n_samp})")
        print(f"  通道数: {n_ch}")
        print("  通道列表 (索引 -> 名称):")
        for i, name in enumerate(names):
            print(f"    {i:4d}  {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="列出目录内 npz 的 EEG 通道名")
    ap.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_NPZ_DIR,
        help="含 .npz 的目录（默认: npz/Shawn_shared）",
    )
    ap.add_argument("--pattern", type=str, default="*.npz")
    args = ap.parse_args()

    d = args.dir.resolve()
    if not d.is_dir():
        print(f"[ERR] 目录不存在: {d}", file=sys.stderr)
        return 1

    files = sorted(d.glob(args.pattern))
    if not files:
        print(f"[ERR] {d} 下无匹配 {args.pattern!r}", file=sys.stderr)
        return 1

    print(f"[INFO] 扫描: {d}\n[INFO] 匹配 {len(files)} 个文件")
    for p in files:
        try:
            inspect_one(p)
        except Exception as e:
            print(f"\n[ERR] {p.name}: {e}", file=sys.stderr)

    print(f"\n[DONE] 共处理 {len(files)} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
