import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

# 设置英文字体支持
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用Arial字体显示英文
plt.rcParams['axes.unicode_minus'] = True  # 正常显示负号
plt.rcParams['font.size'] = 22  # 全局字体大小，从18增加到22
plt.rcParams['axes.labelsize'] = 26  # 坐标轴标签字体大小，从20增加到26
plt.rcParams['xtick.labelsize'] = 22  # X轴刻度标签字体大小，从18增加到22
plt.rcParams['ytick.labelsize'] = 22  # Y轴刻度标签字体大小，从18增加到22
plt.rcParams['legend.fontsize'] = 20  # 图例字体大小，从16增加到20
plt.rcParams['figure.figsize'] = (24, 8)  # 保持图形大小不变
plt.rcParams['figure.dpi'] = 300  # 设置DPI
plt.rcParams['savefig.dpi'] = 300  # 设置保存图片的DPI

# 定义数据目录
cpu_data_dir = "h:\\work\\Compared_data\\CPU\\"
mem_data_dir = "h:\\work\\Compared_data\\Mem\\"

# 定义四个子图的数据配置
subplots_config = [
    {
        'data_dir': cpu_data_dir,
        'files': {
            'Arima': 'Arima_Google_5m_test_step_rmse.npy',
            'Lstm': 'Lstm_Google_5m_test_step_rmse.npy',
            'SWT_CLSTM': 'SWT_CLSTM_Google_5m_test_step_rmse.npy',
            'Tfc': 'Tfc_Google_5m_test_step_rmse.npy',
            'TimeMixerPlusPlus': 'TimeMixerPlusPlus_Google_5m_test_step_rmse.npy',
            'PatchTST': 'PatchTST_Google_5m_test_step_rmse.npy'
        },
        'title': '(a)',
        'xlabel': 'Prediction Steps (5min/step)',
        'subplot_label': '(a)'
    },
    {
        'data_dir': cpu_data_dir,
        'files': {
            'Arima': 'Arima_Alibaba_30s_test_step_rmse.npy',
            'Lstm': 'Lstm_Alibaba_30s_test_step_rmse.npy',
            'SWT_CLSTM': 'SWT_CLSTM_Alibaba_30s_test_step_rmse.npy',
            'Tfc': 'Tfc_Alibaba_30s_test_step_rmse.npy',
            'TimeMixerPlusPlus': 'TimeMixerPlusPlus_Alibaba_30s_test_step_rmse.npy',
            'PatchTST': 'PatchTST_Alibaba_30s_test_step_rmse.npy'
        },
        'title': '(b)',
        'xlabel': 'Prediction Steps (30s/step)',
        'subplot_label': '(b)'
    },
    {
        'data_dir': mem_data_dir,
        'files': {
            'Arima': 'Arima_Google_5m_test_step_rmse.npy',
            'Lstm': 'Lstm_5m_test_step_rmse.npy',
            'SWT_CLSTM': 'SWT_CLSTM_Google_5m_test_step_rmse.npy',
            'Tfc': 'Tfc_Google_5m_test_step_rmse.npy',
            'TimeMixerPlusPlus': 'TimeMixerPlusPlus_Google_5m_test_step_rmse.npy',
            'PatchTST': 'PatchTST_Google_5m_test_step_rmse.npy'
        },
        'title': '(c)',
        'xlabel': 'Prediction Steps (5min/step)',
        'subplot_label': '(c)'
    },
    {
        'data_dir': mem_data_dir,
        'files': {
            'Arima': 'Arima_Alibaba_30s_test_step_rmse.npy',
            'Lstm': 'Lstm_30s_test_step_rmse.npy',
            'SWT_CLSTM': 'SWT_CLSTM_Alibaba_30s_test_step_rmse.npy',
            'Tfc': 'Tfc_Alibaba_30s_test_step_rmse.npy',
            'TimeMixerPlusPlus': 'TimeMixerPlusPlus_Alibaba_30s_test_step_rmse.npy',
            'PatchTST': 'PatchTST_Alibaba_30s_test_step_rmse.npy'
        },
        'title': '(d)',
        'xlabel': 'Prediction Steps (30s/step)',
        'subplot_label': '(d)'
    }
]

# 定义模型名称和颜色 - 使用更深的颜色，全部采用实线
models = {
    'Arima': {'label': 'ARIMA', 'color': '#CC0000', 'linestyle': '-'},  # 深红色
    'Lstm': {'label': 'LSTM', 'color': '#0066CC', 'linestyle': '-'},  # 深蓝色
    'SWT_CLSTM': {'label': 'SWT-CLSTM', 'color': '#006600', 'linestyle': '-'},  # 深绿色
    'Tfc': {'label': 'TFC', 'color': '#CC6600', 'linestyle': '-'},  # 深橙色
    'TimeMixerPlusPlus': {'label': 'TimeMixer++', 'color': '#660066', 'linestyle': '-'},  # 深紫色
    'PatchTST': {'label': 'PatchTST', 'color': '#FF1493', 'linestyle': '-'}  # 深粉色
}

# 创建图形和子图
# 创建图形和子图
fig, axes = plt.subplots(1, 4, figsize=(24, 8))  # 增加子图大小
fig.subplots_adjust(wspace=0.3, left=0.05, right=0.98, bottom=0.25, top=0.9)  # 增加子图间距wspace从0.25到0.3

# 遍历每个子图
for i, subplot_config in enumerate(subplots_config):
    ax = axes[i]

    
    # 每个子图都设置Y轴标签
    ax.set_ylabel('RMSE', fontsize=28)  # 从24增加到28
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # 存储当前数据集的所有RMSE数据，用于设置Y轴范围
    all_rmse_data = []
    max_steps = 0
    
    # 遍历每个模型的数据文件
    for model_name, file_name in subplot_config['files'].items():
        try:
            data_path = os.path.join(subplot_config['data_dir'], file_name)
            rmse_data = np.load(data_path)
            
            # 记录最大步数
            max_steps = max(max_steps, len(rmse_data))
            all_rmse_data.extend(rmse_data)
            
            # 绘制RMSE曲线 - 全部使用实线
            x_values = np.arange(1, len(rmse_data) + 1)
            ax.plot(
                x_values,
                rmse_data,
                label=models[model_name]['label'],
                color=models[model_name]['color'],
                linestyle=models[model_name]['linestyle'],
                linewidth=1,
                alpha=0.9
            )
            
        except Exception as e:
            print(f"Failed to load file {file_name}: {e}")
    
    # 设置x轴范围
    if max_steps > 0:
        ax.set_xlim(1, max_steps)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    
    # 设置y轴范围
    if all_rmse_data:
        y_min = max(0, np.min(all_rmse_data) * 0.95)
        y_max = np.max(all_rmse_data) * 1.05
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    
    # 设置刻度朝内
    ax.tick_params(direction='in', labelsize=22)  # 从18增加到22
    
    # 添加边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    
    # 每个子图都添加图例 - 调整位置和大小
    ax.legend(loc='upper right', fontsize=20, frameon=True, framealpha=0.9,   # 从16增加到20
              handlelength=2.0, handletextpad=0.6, columnspacing=1.0,        # 增加图例内部间距
              bbox_to_anchor=(1, 1), borderpad=0.5)                    # 精确控制图例位置
    
    # 将标题和x轴标签放在下方
    ax.set_xlabel(f"{subplot_config['xlabel']}\n\n{subplot_config['title']}", fontsize=28, labelpad=10)  # 从24增加到28

# 保存图片
output_path = "h:\\work\\images\\Combined_CPU_Mem_rmse_comparison.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')

print(f"Image saved to: {output_path}")
print(f"PDF image saved to: {output_path.replace('.png', '.pdf')}")
print(f"SVG image saved to: {output_path.replace('.png', '.svg')}")

# 显示图像
# plt.show()