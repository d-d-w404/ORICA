"""各子目录脚本 import：把 validation_code 加入 sys.path 并导出路径常量。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

from _paths import (
    ARTIFACT_VERIFY_ROOT,
    CODE_DIR,
    DATA_ROOT,
    INPUT_DATA_ROOT,
    OUTPUT_DATA_ROOT,
    REPO_ROOT,
    THRESHOLD_ROOT,
    VALIDATION_CODE_DIR,
)


def bootstrap_paths() -> Tuple[Path, Path, Path, Path, Path]:
    vc = VALIDATION_CODE_DIR
    if str(vc) not in sys.path:
        sys.path.insert(0, str(vc))
    code = CODE_DIR
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    return OUTPUT_DATA_ROOT, DATA_ROOT, CODE_DIR, REPO_ROOT, ARTIFACT_VERIFY_ROOT
