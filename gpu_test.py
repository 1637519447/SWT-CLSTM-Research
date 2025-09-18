# 检查GPU是否可用
import os

import torch

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA设备数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("当前CUDA版本:", torch.version.cuda)
    print("GPU型号:", torch.cuda.get_device_name(0))
else:
    print("未检测到GPU，将使用CPU进行训练")
    print("CUDA路径检查:", os.environ.get('CUDA_PATH', '未设置'))
    # 尝试加载CUDA库，看是否有错误
    try:
        import ctypes
        ctypes.CDLL("nvcuda.dll")
        print("CUDA库加载成功")
    except Exception as e:
        print("CUDA库加载失败:", str(e))