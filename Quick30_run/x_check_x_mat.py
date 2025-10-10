
import numpy as np
import scipy.io








print("initialize")
data = np.load(r"D:\work\Python_Project\ORICA\temp_txt\orica_mats_20250915_040207.npz")
W = data["icaweights"]
whitening_matrix = data["icasphere"]
print(data)
print("self.W",W)
print("self.whitening_matrix",whitening_matrix)
print("initialize done")




print("initialize")
data = scipy.io.loadmat(r"D:\work\Python_Project\ORICA\temp_txt\cleaned_data_20251001_163725.mat")

# # 先查看文件中有哪些主要的键
# print("=== .mat 文件中的主要键 ===")
# for key in data.keys():
#     if not key.startswith('__'):  # 过滤掉MATLAB的内部变量
#         value = data[key]
#         print(f"键: {key}")
#         print(f"  类型: {type(value)}")
#         if hasattr(value, 'shape'):
#             print(f"  形状: {value.shape}")
#         if hasattr(value, 'dtype'):
#             print(f"  数据类型: {value.dtype}")
#         print()

# 访问cleaned_data结构中的具体字段
cleaned_data = data['cleaned_data']
print("=== cleaned_data 结构中的字段 ===")

# 获取所有字段名
field_names = cleaned_data.dtype.names
print(f"字段名: {field_names}")

# 尝试访问icaweights和icasphere
try:
    icaweights = cleaned_data[0, 0]['icaweights']
    print(f"\nicaweights 类型: {type(icaweights)}")
    print(f"icaweights 形状: {icaweights.shape}")
    print(f"icaweights 内容: {icaweights}")
    
    icasphere = cleaned_data[0, 0]['icasphere']
    print(f"\nicasphere 类型: {type(icasphere)}")
    print(f"icasphere 形状: {icasphere.shape}")
    print(f"icasphere 内容: {icasphere}")
    
except Exception as e:
    print(f"访问字段时出错: {e}")

print("initialize done")
