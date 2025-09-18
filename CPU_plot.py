import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib参数，使图形符合IEEE标准并确保英文显示
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.figsize': (16, 6),
    'figure.autolayout': True,
    'text.usetex': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'grid.linewidth': 0.5
})

# 创建保存图像的目录
os.makedirs('h:\\work\\images', exist_ok=True)

# 定义新的数据文件路径
model_files = {
    'Google': {
        'arima': 'h:\\work\\Pre_data\\CPU\\Arima_Google_5m_test.npy',
        'lstm': 'h:\\work\\Pre_data\\CPU\\Lstm_Google_5m_test.npy',
        'swd': 'h:\\work\\Pre_data\\CPU\\SWT_CLSTM_Google_5m_test.npy',
        'tfc': 'h:\\work\\Pre_data\\CPU\\Tfc_Google_5m_test.npy',
        'timemixerplusplus': 'h:\\work\\Pre_data\\CPU\\TimeMixerPlusPlus_Google_5m_test.npy',
        'patchtst': 'h:\\work\\Pre_data\\CPU\\PatchTST_Google_5m_test.npy',
        'ground_truth': 'h:\\work\\Pre_data\\CPU\\Google_5m_test_ground_truth.npy'
    },
    'Alibaba': {
        'arima': 'h:\\work\\Pre_data\\CPU\\Arima_Alibaba_30s_test.npy',
        'lstm': 'h:\\work\\Pre_data\\CPU\\Lstm_Alibaba_30s_test.npy',
        'swd': 'h:\\work\\Pre_data\\CPU\\SWT_CLSTM_Alibaba_30s_test.npy',
        'tfc': 'h:\\work\\Pre_data\\CPU\\Tfc_Alibaba_30s_test.npy',
        'timemixerplusplus': 'h:\\work\\Pre_data\\CPU\\TimeMixerPlusPlus_Alibaba_30s_test.npy',
        'patchtst': 'h:\\work\\Pre_data\\CPU\\PatchTST_Alibaba_30s_test.npy',
        'ground_truth': 'h:\\work\\Pre_data\\CPU\\Alibaba_30s_test_ground_truth.npy'
    }
}

# 定义颜色和线型 - 使用更明显的颜色差异和更粗的线条
model_styles = {
    'ground_truth': {'color': '#000000', 'linestyle': '-', 'linewidth': 2.0, 'label': 'Ground Truth'},  # 黑色，加粗
    'arima': {'color': '#FF0000', 'linestyle': '--', 'linewidth': 1.5, 'label': 'ARIMA'},  # 鲜红色，加粗
    'lstm': {'color': '#0000FF', 'linestyle': '-.', 'linewidth': 1.5, 'label': 'LSTM'},  # 蓝色，加粗
    'swd': {'color': '#00AA00', 'linestyle': ':', 'linewidth': 1.8, 'label': 'SWT-CLSTM'},  # 绿色，点线加粗
    'tfc': {'color': '#FF8C00', 'linestyle': '--', 'linewidth': 1.5, 'label': 'TFC'},  # 深橙色，加粗
    'timemixerplusplus': {'color': '#8B008B', 'linestyle': '-.', 'linewidth': 1.5, 'label': 'TimeMixer++'},  # 深紫色，加粗
    'patchtst': {'color': '#FF1493', 'linestyle': '-', 'linewidth': 1.5, 'label': 'PatchTST'}  # 深粉色，加粗
}

def load_data(file_path):
    """加载NPY文件数据，处理可能的异常"""
    try:
        data = np.load(file_path)
        # 确保数据是一维的
        if data.ndim > 1:
            data = data.flatten()
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])

def calculate_metrics(y_true, y_pred):
    """计算RMSE, MAE和R²评估指标"""
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 计算MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 计算R²
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return rmse, mae, r2

def plot_model_comparison():
    """创建一行两列的图表，分别显示Google和Alibaba数据集的各模型预测结果"""
    # 创建图形和子图布局
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    datasets = ['Google', 'Alibaba']
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        print(f"\n处理 {dataset} 数据集...")
        
        # 加载真实值数据
        truth_path = model_files[dataset]['ground_truth']
        truth_data = load_data(truth_path)
        
        if len(truth_data) == 0:
            print(f"    警告: 无法加载 {dataset} 真实值数据，跳过")
            continue
        
        x = np.arange(len(truth_data))
        
        # 首先绘制真实值
        ax.plot(x, truth_data, 
               color=model_styles['ground_truth']['color'],
               linestyle=model_styles['ground_truth']['linestyle'],
               linewidth=model_styles['ground_truth']['linewidth'],
               label=model_styles['ground_truth']['label'])
        
        # 绘制各个模型的预测结果
        model_names = ['arima', 'lstm', 'swd', 'tfc', 'timemixerplusplus', 'patchtst']
        
        for model_name in model_names:
            model_path = model_files[dataset][model_name]
            model_data = load_data(model_path)
            
            if len(model_data) == 0:
                print(f"    警告: 无法加载 {dataset} {model_name} 数据，跳过")
                continue
            
            # 确保数据长度一致
            min_len = min(len(model_data), len(truth_data))
            model_data_plot = model_data[:min_len]
            x_plot = x[:min_len]
            
            # 绘制模型预测值
            ax.plot(x_plot, model_data_plot, 
                   color=model_styles[model_name]['color'],
                   linestyle=model_styles[model_name]['linestyle'],
                   linewidth=model_styles[model_name]['linewidth'],
                   label=model_styles[model_name]['label'])
            
            # 计算并打印评估指标
            truth_subset = truth_data[:min_len]
            rmse, mae, r2 = calculate_metrics(truth_subset, model_data_plot)
            print(f"    {model_styles[model_name]['label']}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # 设置标题和标签
        time_interval = '5 Minutes' if dataset == 'Google' else '30 Seconds'
        ax.set_title(f'{dataset} Dataset ({time_interval})', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('CPU Usage')
        
        # 添加图例
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('h:\\work\\images\\CPU_model_comparison_two_datasets.png', bbox_inches='tight', dpi=300)
    plt.savefig('h:\\work\\images\\CPU_model_comparison_two_datasets.pdf', bbox_inches='tight')
    plt.savefig('h:\\work\\images\\CPU_model_comparison_two_datasets.svg', bbox_inches='tight', format='svg')
    
    print("\n图表已保存到 h:\\work\\images\\CPU_model_comparison_two_datasets.png、.pdf 和 .svg")
    
    # plt.show()

def load_and_preprocess_data(file_path):
    """保留原有函数以兼容旧代码"""
    data = np.loadtxt(file_path, delimiter=' ')
    
    # 打印原始数据长度
    print(f"原始数据 {os.path.basename(file_path)} 长度: {len(data)}")
    
    # 去除0值
    data = data[data != 0]
    
    # 打印去除0值后的数据长度
    print(f"处理后数据 {os.path.basename(file_path)} 长度: {len(data)}")
    
    # 不再使用滤波器平滑数据
    return data, data  # 返回相同的数据作为原始数据和"平滑"数据

def format_time_axis_old(ax, interval, num_points):
    """保留原有函数以兼容旧代码"""
    if interval == '30s':
        # 30秒数据改为每2小时显示一个刻度
        hours = np.arange(0, num_points * 30 / 3600, 2)  # 从每1小时改为每2小时一个刻度
        ax.set_xticks(np.linspace(0, num_points, len(hours)))
        ax.set_xticklabels([f"{int(h)}" for h in hours])
        ax.set_xlabel('Time (Hours)')
    elif interval == '5m':
        # 5分钟数据改为每2天显示一个刻度
        days = np.arange(0, num_points * 5 / (60 * 24), 2)  # 每2天一个刻度
        ax.set_xticks(np.linspace(0, num_points, len(days)))
        ax.set_xticklabels([f"{int(d)}" for d in days])
        ax.set_xlabel('Time (Days)')
    elif interval == '1h':
        # 1小时数据保持显示天
        days = np.arange(0, num_points / 24, 2)  # 每2天一个刻度
        ax.set_xticks(np.linspace(0, num_points, len(days)))
        ax.set_xticklabels([f"{int(d)}" for d in days])
        ax.set_xlabel('Time (Days)')

def plot_data():
    """保留原有函数以兼容旧代码"""
    # 原有代码保持不变
    pass

def plot_data_with_statistics():
    """保留原有函数以兼容旧代码"""
    # 原有代码保持不变
    pass

if __name__ == "__main__":
    # 绘制新的模型比较图表
    plot_model_comparison()
    
    # 注释掉原有的绘图函数调用
    # plot_data()
    # plot_data_with_statistics()