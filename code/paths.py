"""仓库布局：ORICA/code（脚本）、ORICA/data（数据）。"""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DATA_ROOT = REPO_ROOT / "data"

# 兼容旧名
QUICK30_RUN = CODE_DIR

TRASH_ROOT = REPO_ROOT / "trash"
TEMP_TXT_ROOT = TRASH_ROOT / "temp_txt"

OUTPUT_DATA_ROOT = DATA_ROOT / "output_data"
# 旧常量名，指向同一目录
PROCESSED_DATA_ROOT = OUTPUT_DATA_ROOT

ARTIFACT_VERIFY_ROOT = DATA_ROOT / "artifact_removal_verify"
INPUT_DATA_ROOT = DATA_ROOT / "Input_data"

DEFAULT_ASR_MAT = TEMP_TXT_ROOT / "cleaned_data_quick30.mat"
DEFAULT_ASR_NPZ_CALIB = (
    ARTIFACT_VERIFY_ROOT
    / "IIR_filter2/cali/laporoscopic_1309_EEGmerged_845_1025.npz"
)
DEFAULT_ASR_NPZ_CALIB_LEGACY = (
    CODE_DIR / "calibration/asr_calibration_20260104_231043.npz"
)
DEFAULT_DEMO_SET = TEMP_TXT_ROOT / "Demo_EmotivEPOC_EyeOpen.set"


def resolve_save_dir(save_dir_env: str) -> Path:
    """解析 EEG_SAVE_DIR：支持 output_data/...、data/... 及旧名 processed_data_saves_threshold/...。"""
    s = save_dir_env.replace("\\", "/").strip("/")
    legacy_prefix = "processed_data_saves_threshold/"
    new_prefix = "output_data/"
    if s.startswith("data/"):
        return REPO_ROOT / s
    if s.startswith(new_prefix):
        return OUTPUT_DATA_ROOT / s[len(new_prefix) :]
    if s.startswith(legacy_prefix):
        return OUTPUT_DATA_ROOT / s[len(legacy_prefix) :]
    if s in ("processed_data_saves_threshold", "output_data"):
        return OUTPUT_DATA_ROOT
    return OUTPUT_DATA_ROOT / s
