import numpy as np
import os

# 文件路径
file_path = 'h:\\work\\Pre_data\\Mem\\Arima_Alibaba_30s_test.npy'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误：文件 {file_path} 不存在")
    exit(1)

try:
    # 加载 npy 文件
    data = np.load(file_path)
    
    # 打印原始数据的形状
    print(f"原始数据形状: {data.shape}")
    print(f"原始数据前5个元素: {data[:5]}")
    
    # 删除前71个数据
    if len(data) <= 71:
        print(f"警告：数据长度({len(data)})小于或等于71，无法删除前71个元素")
        exit(1)
    
    new_data = data[71:]
    
    # 打印新数据的形状
    print(f"删除后数据形状: {new_data.shape}")
    print(f"删除后数据前5个元素: {new_data[:5]}")
    
    # 保存修改后的数据到原文件
    np.save(file_path, new_data)
    
    print(f"成功删除前71个数据并保存到 {file_path}")
    
except Exception as e:
    print(f"处理文件时出错: {e}")