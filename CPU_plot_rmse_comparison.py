import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

# 设置英文字体支持
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用Arial字体显示英文
plt.rcParams['axes.unicode_minus'] = True  # 正常显示负号
plt.rcParams['font.size'] = 12  # 设置字体大小
plt.rcParams['figure.figsize'] = (12, 5)  # 调整为一行两列的布局
plt.rcParams['figure.dpi'] = 300  # 设置DPI
plt.rcParams['savefig.dpi'] = 300  # 设置保存图片的DPI

# 定义数据目录
data_dir = "h:\\work\\Compared_data\\CPU\\"

# 定义两个数据集的数据文件
datasets = {
    'Google': {
        'files': {
            'Arima': 'Arima_Google_5m_test_step_rmse.npy',
            'Lstm': 'Lstm_Google_5m_test_step_rmse.npy',
            'SWT_CLSTM': 'SWT_CLSTM_Google_5m_test_step_rmse.npy',
            'Tfc': 'Tfc_Google_5m_test_step_rmse.npy',
            'TimeMixerPlusPlus': 'TimeMixerPlusPlus_Google_5m_test_step_rmse.npy',
            'PatchTST': 'PatchTST_Google_5m_test_step_rmse.npy'
        },
        'title': 'Google Dataset (5-Minute Scale)',
        'xlabel': 'Prediction Steps (5min/step)'
    },
    'Alibaba': {
        'files': {
            'Arima': 'Arima_Alibaba_30s_test_step_rmse.npy',
            'Lstm': 'Lstm_Alibaba_30s_test_step_rmse.npy',
            'SWT_CLSTM': 'SWT_CLSTM_Alibaba_30s_test_step_rmse.npy',
            'Tfc': 'Tfc_Alibaba_30s_test_step_rmse.npy',
            'TimeMixerPlusPlus': 'TimeMixerPlusPlus_Alibaba_30s_test_step_rmse.npy',
            'PatchTST': 'PatchTST_Alibaba_30s_test_step_rmse.npy'
        },
        'title': 'Alibaba Dataset (30-Second Scale)',
        'xlabel': 'Prediction Steps (30s/step)'
    }
}

# 定义模型名称和颜色 - 使用更明显的颜色差异
models = {
    'Arima': {'label': 'ARIMA', 'color': '#FF0000', 'linestyle': '-'},  # 鲜红色
    'Lstm': {'label': 'LSTM', 'color': '#0000FF', 'linestyle': '--'},  # 蓝色
    'SWT_CLSTM': {'label': 'SWT-CLSTM', 'color': '#00AA00', 'linestyle': '-.'},  # 绿色
    'Tfc': {'label': 'TFC', 'color': '#FF8C00', 'linestyle': ':'},  # 深橙色
    'TimeMixerPlusPlus': {'label': 'TimeMixer++', 'color': '#660066', 'linestyle': '-'},  # 深紫色
    'PatchTST': {'label': 'PatchTST', 'color': '#FF1493', 'linestyle': '-'}  # 深粉色
}

# 创建图形和子图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.3, left=0.08, right=0.95, bottom=0.15, top=0.9)

# 遍历每个数据集
for i, (dataset_name, dataset_info) in enumerate(datasets.items()):
    ax = axes[i]
    
    ax.set_title(dataset_info['title'], fontsize=12, fontweight='bold')
    ax.set_xlabel(dataset_info['xlabel'], fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # 存储当前数据集的所有RMSE数据，用于设置Y轴范围
    all_rmse_data = []
    max_steps = 0
    
    # 遍历每个模型的数据文件
    for model_name, file_name in dataset_info['files'].items():
        try:
            data_path = os.path.join(data_dir, file_name)
            rmse_data = np.load(data_path)
            
            # 记录最大步数
            max_steps = max(max_steps, len(rmse_data))
            
            # 收集所有RMSE数据用于Y轴范围设置
            all_rmse_data.extend(rmse_data)
            
            # 绘制RMSE曲线
            x_values = np.arange(1, len(rmse_data) + 1)
            ax.plot(
                x_values,
                rmse_data,
                label=models[model_name]['label'],
                color=models[model_name]['color'],
                linestyle=models[model_name]['linestyle'],
                linewidth=1.5,
                alpha=0.8
            )
            
        except Exception as e:
            print(f"Failed to load file {file_name}: {e}")
    
    # 设置x轴范围
    if max_steps > 0:
        ax.set_xlim(1, max_steps)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    
    # 设置y轴范围
    if all_rmse_data:
        y_min = max(0, np.min(all_rmse_data) * 0.95)
        y_max = np.max(all_rmse_data) * 1.05
        ax.set_ylim(y_min, y_max)
    
    # 设置刻度
    ax.tick_params(direction='in', labelsize=10)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.9)

# 保存图片
output_path = "h:\\work\\images\\CPU_rmse_comparison_plot.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')

print(f"Image saved to: {output_path}")
print(f"PDF image saved to: {output_path.replace('.png', '.pdf')}")
print(f"SVG image saved to: {output_path.replace('.png', '.svg')}")

# 显示图像
plt.show()