# exact_dump_from_fdt.py
import os
import numpy as np
from scipy.io import loadmat
import struct

set_path   = r"D:\work\matlab_project\orica-master\orica-master\SIM_STAT_16ch_3min.set"
out_path   = r"D:\work\Python_Project\ORICA\temp_txt\x.txt"

# 读 .set 的元信息（nbchan, pnts, data 路径）
S = loadmat(set_path, squeeze_me=True, struct_as_record=False)
EEG = S['EEG']

def get_attr(obj, name):
    return getattr(obj, name) if hasattr(obj, name) else obj[name]

nbchan = int(get_attr(EEG, 'nbchan'))
pnts   = int(get_attr(EEG, 'pnts'))
data_f = get_attr(EEG, 'data')   # 当使用 .fdt 外部存储时，这里是相对路径字符串

# 解析 .fdt 路径
if isinstance(data_f, (str, bytes, np.str_)):
    fdt_path = data_f if os.path.isabs(data_f) else os.path.join(os.path.dirname(set_path), data_f)
    # 直接读原始 float32 小端
    raw = np.fromfile(fdt_path, dtype='<f4', count=nbchan*pnts)
    X = raw.reshape((nbchan, pnts), order='F')   # 与 MATLAB fread 按列读入对齐
else:
    # 罕见：数据内嵌在 .set
    X = np.asarray(data_f, dtype=np.float32, order='F')

X64 = X.astype(np.float64, copy=False)  # float32 → float64（精确映射）

# 保存：十进制 + IEEE754（逐元素按行）
with open(out_path, "w", encoding="utf-8") as f:
    R, C = 16, 3
    f.write(f"# rows={R} cols={C} dtype=float64\n")
    for i in range(R):
        f.write("\t".join(f"{X64[i,j]:.17g}" for j in range(C)) + "\n")
    f.write("# IEEE754 hex (big-endian, 64-bit)\n")
    for i in range(R):
        hexrow = [struct.pack(">d", float(X64[i,j])).hex() for j in range(C)]
        f.write("\t".join(hexrow) + "\n")
print("done:", out_path)
