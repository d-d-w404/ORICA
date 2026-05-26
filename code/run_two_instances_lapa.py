"""
同时启动多个 main_gui.py 实例（同一数据源，不同参数/保存目录）
"""
import subprocess
import sys
import time
import os
from pathlib import Path

from paths import INPUT_DATA_ROOT

def main():
    script_dir = Path(__file__).parent
    main_gui_path = script_dir / "main_gui.py"
    
    print("=" * 60)
    print("启动多个 main_gui.py 实例")
    print("=" * 60)
    print()

    # 只改这一项：须与 npz/lap/、asr_cali/lap/2min/ 下文件名一致（不含 .npz）
    #subject_id = "lapa_001"
    #subject_id = "lapa_003"
    subject_id = "lapa_1271"
    #subject_id = "lapa_1284"
    #subject_id = "lapa_1295"
    #subject_id = "lapa_1307"
    #subject_id = "lapa_1309" 
    #subject_id = "lapa_1311"

    # 从 subject_id 取出编号（如 lapa_1309 -> 1309），再取后两位 -> b09（与旧腹腔镜 1309 相同）
    _num = subject_id.split("_")[-1]
    file_tag = f"b{_num[-2:]}"

    asr_calib_npz = str(
        INPUT_DATA_ROOT
        / "asr_cali/Lapa/2min"
        / f"{subject_id}.npz"
    )

    # 只保留实验参数，避免重复写 file_tag / asr_calib_npz
    raw_experiments = [
        #{"method": "4", "save_dir": "output_data/1asr5_2min_70", "asr_cutoff": "5", "icalabel_threshold": "0.7"},
        {"method": "4", "save_dir": "output_data/Lapaasrpy20_2min_70", "asr_cutoff": "20", "icalabel_threshold": "0.7"},
        #{"method": "4", "save_dir": "output_data/Lapaasrpy100_2min_70", "asr_cutoff": "100", "icalabel_threshold": "0.7"},
        #{"method": "4", "save_dir": "output_data/1asr20_2min_90", "asr_cutoff": "20", "icalabel_threshold": "0.9"},
        #{"method": "4", "save_dir": "output_data/1asr20_2min_50", "asr_cutoff": "20", "icalabel_threshold": "0.5"},
    ]
    instance_configs = []
    for i, exp in enumerate(raw_experiments, start=1):
        cfg = dict(exp)
        cfg["instance"] = str(i)
        cfg["file_tag"] = file_tag
        cfg["asr_calib_npz"] = asr_calib_npz
        instance_configs.append(cfg)

    processes = []
    for cfg in instance_configs:
        print(
            f"启动实例 {cfg['instance']} | method={cfg['method']} "
            f"| save_dir={cfg['save_dir']} | file_tag={cfg['file_tag']} "
            f"| asr_cutoff={cfg['asr_cutoff']} | icalabel_th={cfg['icalabel_threshold']}"
        )
        env = os.environ.copy()
        env["EEG_GUI_INSTANCE"] = cfg["instance"]
        env["IIR_FILTER_METHOD"] = cfg["method"]
        env["EEG_SAVE_DIR"] = cfg["save_dir"]
        env["EEG_SAVE_FILE_TAG"] = cfg["file_tag"]
        env["EEG_ASR_CALIB_NPZ"] = cfg["asr_calib_npz"]
        env["EEG_ASR_CUTOFF"] = cfg["asr_cutoff"]
        # 可选：在线 ASR 改用 asrpy（默认 meegkit），见 receiver.py
        env["EEG_ASR_BACKEND"] = "asrpy"
        env["EEG_ICALABEL_THRESHOLD"] = cfg["icalabel_threshold"]
        p = subprocess.Popen(
            [sys.executable, str(main_gui_path)],
            env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0,
        )
        processes.append(p)
        time.sleep(1)

    print()
    print("=" * 60)
    print(f"[OK] 已启动 {len(processes)} 个实例！")
    print("=" * 60)
    print()
    print("提示：各实例会接收同一 LSL 数据流，但使用各自参数并写入各自目录。")
    print("LSL 广播请在 aa_lsl_npz.py 中设置：INPUT_DATA_ROOT / \"npz/lap\" / f\"{subject_id}.npz\"")
    print("如需新增实例，直接在 instance_configs 里加一行。")
    print()
    print("按 Ctrl+C 退出此脚本（不会关闭GUI窗口）")
    
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\n\n脚本已退出，GUI窗口仍在运行")

if __name__ == "__main__":
    main()
