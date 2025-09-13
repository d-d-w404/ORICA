# matlab_python_compare.py
import numpy as np
import os


# MATLAB_FILE = r"D:\work\matlab_project\orica-master\orica-master\2.txt"
# PYTHON_FILE = r"D:\work\matlab_project\orica-master\orica-master\2x.txt"
# PYTHON_FILE = r"D:\work\Python_Project\ORICA\temp_txt\44_11.txt"
# MATLAB_FILE= r"D:\work\Python_Project\ORICA\temp_txt\44_9.txt"




file_name = "44_9"  # 改这里就行，例如 "1" / "26" / "2025-08-30"
MATLAB_DIR = r"D:\work\matlab_project\orica-master\orica-master"
PYTHON_DIR = r"D:\work\Python_Project\ORICA\temp_txt"

MATLAB_FILE = os.path.join(MATLAB_DIR, f"{file_name}.txt")
PYTHON_FILE = os.path.join(PYTHON_DIR, f"{file_name}.txt")


def _split_row(line: str):
    parts = [p for p in line.strip().replace(",", " ").split() if p]
    return parts

def parse_matrix_txt(path):
    """解析 save_matrix_txt 输出的文件"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # 找到 hex 段落起始行
    hex_header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("# ieee754 hex"):
            hex_header_idx = i
            break
    if hex_header_idx is None:
        raise ValueError(f"File missing IEEE754 hex header: {path}")

    # 解析十进制数据
    dec_lines = [ln for ln in lines[:hex_header_idx] if not ln.strip().startswith("#")]
    dec_rows = []
    for ln in dec_lines:
        parts = _split_row(ln)
        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue
        if row:
            dec_rows.append(row)
    dec = np.array(dec_rows, dtype=np.float64)

    # 解析十六进制数据
    hex_rows = []
    for ln in lines[hex_header_idx+1:]:
        if not ln.strip() or ln.strip().startswith("#"):
            continue
        parts = _split_row(ln)
        if parts:
            hex_rows.append([p.lower() for p in parts])
    hx = np.array(hex_rows, dtype=object)

    return dec, hx

def compare_decimal_old(a: np.ndarray, b: np.ndarray, atol=1e-15, rtol=1e-12):
    if a.shape != b.shape:
        return False, f"shape mismatch: {a.shape} vs {b.shape}"
    equal = np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    if equal:
        return True, None
    else:
        diff = np.abs(a - b)
        max_abs = np.nanmax(diff)
        return False, f"values differ, max abs diff = {max_abs:.3e}"


def compare_decimal(a: np.ndarray, b: np.ndarray):
    """零容差：逐元素完全相等（含 NaN 对 NaN 视为相等）"""
    if a.shape != b.shape:
        return False, f"shape mismatch: {a.shape} vs {b.shape}"

    # 严格逐元素：把 NaN/NaN 当作相等
    eq = (a == b) | (np.isnan(a) & np.isnan(b))
    if np.all(eq):
        return True, None

    # 不相等时，给出一些定位信息
    mism = np.where(~eq)
    r0, c0 = int(mism[0][0]), int(mism[1][0])
    a0, b0 = a[r0, c0], b[r0, c0]

    # 只对有限值给出差值，避免 NaN/Inf 的减法
    if np.isfinite(a0) and np.isfinite(b0):
        info = f"first mismatch at ({r0},{c0}): {a0:.17g} vs {b0:.17g} (abs diff={abs(a0-b0):.3e})"
    else:
        info = f"first mismatch at ({r0},{c0}): {a0} vs {b0}"

    return False, f"{info}; total mismatches={len(mism[0])}"


def compare_hex(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        return False, f"shape mismatch: {a.shape} vs {b.shape}"
    eq = (a == b)
    if np.all(eq):
        return True, None
    else:
        mism = np.where(~eq)
        r, c = mism[0][0], mism[1][0]
        return False, f"hex differ at ({r},{c}): {a[r,c]} vs {b[r,c]}"

def main():
    dec_m, hex_m = parse_matrix_txt(MATLAB_FILE)
    dec_p, hex_p = parse_matrix_txt(PYTHON_FILE)

    print("=== Decimal Compare ===")
    dec_equal, dec_info = compare_decimal(dec_m, dec_p)
    print("Equal:", dec_equal)
    if dec_info:
        print("Info:", dec_info)

    print("\n=== IEEE754 Hex Compare ===")
    hex_equal, hex_info = compare_hex(hex_m, hex_p)
    print("Equal:", hex_equal)
    if hex_info:
        print("Info:", hex_info)

    print("\n=== Summary ===")
    if dec_equal and hex_equal:
        print("✅ 两个文件完全一致（数值和IEEE754位模式）。")
    elif dec_equal and not hex_equal:
        print("⚠️ 数值一致，但IEEE754位模式不同（可能是保存方式差异）。")
    elif not dec_equal and hex_equal:
        print("⚠️ IEEE754一致，但数值比较不一致（可能是文本解析问题）。")
    else:
        print("❌ 数值和IEEE754都不同。")

if __name__ == "__main__":
    main()
