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
    'font.size': 20,  # 从18增加到20
    'axes.labelsize': 22,  # 从20增加到22
    'axes.titlesize': 22,  # 从20增加到22
    'xtick.labelsize': 20,  # 从18增加到20
    'ytick.labelsize': 20,  # 从18增加到20
    'legend.fontsize': 14,  # 从12增加到14
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.figsize': (24, 8),  # 保持图形大小不变
    'figure.autolayout': True,
    'text.usetex': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'grid.linewidth': 0.5
})

# 创建保存图像的目录
os.makedirs('h:\\work\\images', exist_ok=True)

# 定义CPU数据文件路径
cpu_model_files = {
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

# 定义Memory数据文件路径
mem_model_files = {
    'Google': {
        'arima': 'h:\\work\\Pre_data\\Mem\\Arima_Google_5m_test.npy',
        'lstm': 'h:\\work\\Pre_data\\Mem\\Lstm_Google_5m_test.npy',
        'swd': 'h:\\work\\Pre_data\\Mem\\SWT_CLSTM_Google_5m_test.npy',
        'tfc': 'h:\\work\\Pre_data\\Mem\\Tfc_Google_5m_test.npy',
        'timemixerplusplus': 'h:\\work\\Pre_data\\Mem\\TimeMixerPlusPlus_Google_5m_test.npy',
        'patchtst': 'h:\\work\\Pre_data\\Mem\\PatchTST_Google_5m_test.npy',
        'ground_truth': 'h:\\work\\Pre_data\\Mem\\Google_5m_test_ground_truth.npy'
    },
    'Alibaba': {
        'arima': 'h:\\work\\Pre_data\\Mem\\Arima_Alibaba_30s_test.npy',
        'lstm': 'h:\\work\\Pre_data\\Mem\\Lstm_Alibaba_30s_test.npy',
        'swd': 'h:\\work\\Pre_data\\Mem\\SWT_CLSTM_Alibaba_30s_test.npy',
        'tfc': 'h:\\work\\Pre_data\\Mem\\Tfc_Alibaba_30s_test.npy',
        'timemixerplusplus': 'h:\\work\\Pre_data\\Mem\\TimeMixerPlusPlus_Alibaba_30s_test.npy',
        'patchtst': 'h:\\work\\Pre_data\\Mem\\PatchTST_Alibaba_30s_test.npy',
        'ground_truth': 'h:\\work\\Pre_data\\Mem\\Alibaba_30s_test_ground_truth.npy'
    }
}

# 定义颜色和线型 - 使用实线，线条变细
model_styles = {
    'ground_truth': {'color': '#000000', 'linestyle': '-', 'linewidth': 1.0, 'label': 'Ground Truth'},  # 黑色，实线
    'arima': {'color': '#FF0000', 'linestyle': '-', 'linewidth': 0.8, 'label': 'ARIMA'},  # 鲜红色，实线
    'lstm': {'color': '#0000FF', 'linestyle': '-', 'linewidth': 0.8, 'label': 'LSTM'},  # 蓝色，实线
    'swd': {'color': '#00AA00', 'linestyle': '-', 'linewidth': 0.8, 'label': 'SWT-CLSTM'},  # 绿色，实线
    'tfc': {'color': '#FF8C00', 'linestyle': '-', 'linewidth': 0.8, 'label': 'TFC'},  # 深橙色，实线
    'timemixerplusplus': {'color': '#8B008B', 'linestyle': '-', 'linewidth': 0.8, 'label': 'TimeMixer++'},  # 深紫色，实线
    'patchtst': {'color': '#FF1493', 'linestyle': '-', 'linewidth': 0.8, 'label': 'PatchTST'}  # 深粉色，实线
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

def plot_combined_comparison():
    """创建两行两列的图表，分别显示CPU Google、CPU Alibaba、Memory Google、Memory Alibaba数据集的各模型预测结果"""
    # 创建图形和子图布局 - 两行两列，减少高度
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))  # 从12减少到10
    
    # 定义子图配置 - 更新X轴标签
    # 定义子图配置 - 去掉原时间间隔信息
    subplot_configs = [
        {'data_files': cpu_model_files['Google'], 'ylabel': 'CPU Usage (%)', 'xlabel': 'Prediction Length (Hours)', 'subplot_label': '(a)', 'position': (0, 0)},
        {'data_files': cpu_model_files['Alibaba'], 'ylabel': 'CPU Usage (%)', 'xlabel': 'Prediction Length (Minutes)', 'subplot_label': '(b)', 'position': (0, 1)},
        {'data_files': mem_model_files['Google'], 'ylabel': 'Memory Usage (%)', 'xlabel': 'Prediction Length (Hours)', 'subplot_label': '(c)', 'position': (1, 0)},
        {'data_files': mem_model_files['Alibaba'], 'ylabel': 'Memory Usage (%)', 'xlabel': 'Prediction Length (Minutes)', 'subplot_label': '(d)', 'position': (1, 1)}
    ]
    
    for idx, config in enumerate(subplot_configs):
        row, col = config['position']
        ax = axes[row, col]
        data_files = config['data_files']
        
        print(f"\n处理第{idx+1}个子图: {config['subplot_label']}...")
        
        # 加载真实值数据
        truth_path = data_files['ground_truth']
        truth_data = load_data(truth_path)
        
        if len(truth_data) == 0:
            print(f"    警告: 无法加载真实值数据，跳过")
            continue
        
        # 恢复原有的时间步显示
        x = np.arange(len(truth_data))
        
        # 根据数据集类型转换X轴时间单位
        if 'Google' in str(data_files).replace('\\', '/'):
            # Google数据集：5分钟间隔，转换为小时
            x_converted = x * 5 / 60  # 转换为小时
            x_unit = 'Hour'
            time_info = '(5min intervals)'
        else:
            # Alibaba数据集：30秒间隔，转换为分钟
            x_converted = x * 30 / 60  # 转换为分钟
            x_unit = 'Minute'
            time_info = '(30s intervals)'
        
        # 首先绘制真实值
        ax.plot(x_converted, truth_data, 
               color=model_styles['ground_truth']['color'],
               linestyle=model_styles['ground_truth']['linestyle'],
               linewidth=model_styles['ground_truth']['linewidth'],
               label=model_styles['ground_truth']['label'])
        
        # 绘制各个模型的预测结果
        model_names = ['arima', 'lstm', 'swd', 'tfc', 'timemixerplusplus', 'patchtst']
        
        for model_name in model_names:
            model_path = data_files[model_name]
            model_data = load_data(model_path)
            
            if len(model_data) == 0:
                print(f"    警告: 无法加载 {model_name} 数据，跳过")
                continue
            
            # 确保数据长度一致
            min_len = min(len(model_data), len(truth_data))
            model_data_plot = model_data[:min_len]
            x_plot = x_converted[:min_len]
            
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
        
        # 设置Y轴和X轴标签，使用转换后的时间单位，减少X轴标签的labelpad
        ax.set_ylabel(config['ylabel'], fontsize=22)  # 从20增加到22
        ax.set_xlabel(f'Prediction Length ({x_unit})', fontsize=22, labelpad=8)  # 减少labelpad从15到12
        
        # 设置X轴刻度，使用转换后的时间值
        x_max = x_converted[-1] if len(x_converted) > 0 else 1
        
        # 根据时间范围设置合适的刻度间隔
        if x_unit == 'Hour':
            # 对于小时单位，设置合理的刻度间隔
            if x_max > 10:
                tick_interval = 2  # 每2小时一个刻度
            elif x_max > 5:
                tick_interval = 1  # 每1小时一个刻度
            else:
                tick_interval = 0.5  # 每30分钟一个刻度
        else:
            # 对于分钟单位（第2、4子图），增大刻度间隔以减少数值显示
            if x_max > 120:
                tick_interval = 60  # 每60分钟一个刻度（减少显示）
            elif x_max > 60:
                tick_interval = 30  # 每30分钟一个刻度（减少显示）
            else:
                tick_interval = 20  # 每20分钟一个刻度（减少显示）
        
        # 生成刻度位置
        x_ticks = np.arange(0, x_max + tick_interval, tick_interval)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='x', which='major', length=6, width=1)
        ax.tick_params(axis='y', which='major', length=6, width=1)
        
        # 设置次刻度
        ax.tick_params(axis='x', which='minor', length=3, width=0.5)
        ax.tick_params(axis='y', which='minor', length=3, width=0.5)
        
        # 设置次刻度位置
        from matplotlib.ticker import MultipleLocator, AutoMinorLocator
        if tick_interval >= 1:  # 只在刻度间隔较大时才添加次刻度
            ax.xaxis.set_minor_locator(MultipleLocator(tick_interval / 2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        # 在子图下方添加标签(a)、(b)、(c)、(d)，增加与X轴标签的距离
        ax.text(0.5, -0.3, config['subplot_label'], transform=ax.transAxes,
                ha='center', va='top', fontsize=18, fontweight='bold')  # 调整位置从-0.15到-0.25
        
        # 添加图例 - 每个子图都有图例
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=14)  # 从12增加到14
        
        # 添加网格，减少次网格线的显示以避免杂乱
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
        # 只在刻度间隔较大时才显示次网格线
        if tick_interval > 20:
            ax.grid(True, which='minor', linestyle=':', alpha=0.3, linewidth=0.2)  # 减少次网格线的透明度和线宽
    
    # 调整子图间距，适应两行两列布局，增加底部空间以容纳子图标签
    plt.subplots_adjust(left=0.08, bottom=0.20, right=0.95, top=0.95, wspace=0.25, hspace=0.30)  # 增加bottom从0.15到0.20，减少hspace从0.35到0.30
    
    # 保存图表
    plt.savefig('h:\\work\\images\\Combined_CPU_Mem_model_comparison.png', bbox_inches='tight', dpi=300)
    plt.savefig('h:\\work\\images\\Combined_CPU_Mem_model_comparison.pdf', bbox_inches='tight')
    plt.savefig('h:\\work\\images\\Combined_CPU_Mem_model_comparison.svg', bbox_inches='tight', format='svg')
    
    print("\n合并图表已保存到 h:\\work\\images\\Combined_CPU_Mem_model_comparison.png、.pdf 和 .svg")
    
    # plt.show()

if __name__ == "__main__":
    # 绘制合并的模型比较图表
    plot_combined_comparison()