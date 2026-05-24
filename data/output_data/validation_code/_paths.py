"""validation 脚本共用路径（数据在 output_data 根目录）。"""
from pathlib import Path

VALIDATION_CODE_DIR = Path(__file__).resolve().parent
# 实验数据根：profile 子目录、npz、分析输出
OUTPUT_DATA_ROOT = VALIDATION_CODE_DIR.parent
# 旧名别名
THRESHOLD_ROOT = OUTPUT_DATA_ROOT

DATA_ROOT = OUTPUT_DATA_ROOT.parent
REPO_ROOT = DATA_ROOT.parent
CODE_DIR = REPO_ROOT / "code"
TRASH_ROOT = REPO_ROOT / "trash"
TEMP_TXT_ROOT = TRASH_ROOT / "temp_txt"
ARTIFACT_VERIFY_ROOT = DATA_ROOT / "artifact_removal_verify"
INPUT_DATA_ROOT = DATA_ROOT / "Input_data"
