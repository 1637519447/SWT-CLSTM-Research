import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pywt
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.figsize': (12, 6)
})

def load_csv_data(file_path):
    """加载CSV数据"""
    data = pd.read_csv(file_path, header=None)
    return data.iloc[:, 0].values

def load_npy_data(file_path):
    """加载NPY数据"""
    return np.load(file_path)

def sg_filter(data, window_length=51, polyorder=3):
    """Savitzky-Golay滤波"""
    # 确保window_length为奇数且小于数据长度
    if window_length >= len(data):
        window_length = len(data) - 1 if len(data) % 2 == 0 else len(data) - 2
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < polyorder + 1:
        window_length = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    
    return savgol_filter(data, window_length, polyorder)

def swt_decompose(data, wavelet='db4', levels=3):
    """静态小波变换分解，返回高频和低频分量"""
    # 确保数据长度为2的幂次
    original_length = len(data)
    power_of_2 = 2 ** int(np.ceil(np.log2(original_length)))
    
    # 填充数据到2的幂次长度
    if original_length < power_of_2:
        padded_data = np.pad(data, (0, power_of_2 - original_length), mode='edge')
    else:
        padded_data = data[:power_of_2]
    
    # 进行SWT分解
    coeffs = pywt.swt(padded_data, wavelet, level=levels)
    
    # 获取低频分量（近似系数）
    low_freq = coeffs[-1][0][:original_length]
    
    # 重构高频分量（细节系数的和）
    high_freq = np.zeros(original_length)
    for i in range(levels):
        detail_coeff = coeffs[i][1][:original_length]
        high_freq += detail_coeff
    
    return high_freq, low_freq

def plot_individual_figures():
    """分别绘制每张图"""
    # 确保输出目录存在
    output_dir = 'h:\\work\\images'
    os.makedirs(output_dir, exist_ok=True)
    
    # 统一的线条格式设置
    line_color = '#1f4e79'  # 深蓝色
    line_width = 1.2
    
    # 加载原始数据
    print("加载原始CSV数据...")
    csv_data = load_csv_data('h:\\work\\Alibaba_cpu_util_aggregated_30s.csv')
    
    # 加载真实值数据
    print("加载NPY真实值数据...")
    npy_data = load_npy_data('h:\\work\\Pre_data\\CPU\\Alibaba_30s_test_ground_truth.npy')
    
    # 应用SG滤波
    print("应用SG滤波...")
    sg_filtered = sg_filter(csv_data)
    
    # 应用SWT分解
    print("应用SWT分解...")
    high_freq, low_freq = swt_decompose(sg_filtered)
    
    # 绘制原始数据图
    print("绘制原始数据图...")
    plt.figure(figsize=(12, 6))
    plt.plot(csv_data, linewidth=line_width, color=line_color)
    plt.xlabel('时间步长', fontsize=12)
    plt.ylabel('CPU利用率', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alibaba_original_data.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'alibaba_original_data.pdf'), bbox_inches='tight')
    plt.close()
    
    # 绘制真实值数据图
    print("绘制真实值数据图...")
    plt.figure(figsize=(12, 6))
    plt.plot(npy_data, linewidth=line_width, color=line_color)
    plt.xlabel('时间步长', fontsize=12)
    plt.ylabel('CPU利用率', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alibaba_ground_truth.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'alibaba_ground_truth.pdf'), bbox_inches='tight')
    plt.close()
    
    # 绘制高频分量图
    print("绘制高频分量图...")
    plt.figure(figsize=(12, 6))
    plt.plot(high_freq, linewidth=line_width, color=line_color)
    plt.xlabel('时间步长', fontsize=12)
    plt.ylabel('高频分量幅值', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alibaba_high_freq.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'alibaba_high_freq.pdf'), bbox_inches='tight')
    plt.close()
    
    # 绘制低频分量图
    print("绘制低频分量图...")
    plt.figure(figsize=(12, 6))
    plt.plot(low_freq, linewidth=line_width, color=line_color)
    plt.xlabel('时间步长', fontsize=12)
    plt.ylabel('低频分量幅值', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alibaba_low_freq.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'alibaba_low_freq.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"图片已保存到 {output_dir} 目录:")
    print("- 原始数据: alibaba_original_data.png 和 alibaba_original_data.pdf")
    print("- 真实值数据: alibaba_ground_truth.png 和 alibaba_ground_truth.pdf")
    print("- 高频分量: alibaba_high_freq.png 和 alibaba_high_freq.pdf")
    print("- 低频分量: alibaba_low_freq.png 和 alibaba_low_freq.pdf")

if __name__ == "__main__":
    try:
        plot_individual_figures()
        print("绘图完成！")
    except Exception as e:
        print(f"绘图过程中出现错误: {e}")