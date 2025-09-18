import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import warnings
import matplotlib.font_manager as fm

# 忽略警告
warnings.filterwarnings('ignore')

# 检测系统中可用的字体
font_list = fm.findSystemFonts()
chinese_fonts = []

# 常见的中文字体
common_chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']

# 查找系统中可用的中文字体
for font in font_list:
    try:
        font_name = fm.FontProperties(fname=font).get_name()
        if any(chinese_font in font_name for chinese_font in common_chinese_fonts):
            chinese_fonts.append(font_name)
    except:
        pass

# 设置matplotlib参数，使图形符合IEEE标准并确保英文显示
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,  # 从10增加到14
    'axes.labelsize': 16,  # 从11增加到16
    'axes.titlesize': 18,  # 从12增加到18
    'xtick.labelsize': 14,  # 从9增加到14
    'ytick.labelsize': 14,  # 从9增加到14
    'legend.fontsize': 12,  # 从8增加到12
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.figsize': (8, 6),
    'figure.autolayout': True,
    'text.usetex': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'grid.linewidth': 0.5
})

# 创建保存图像的目录
os.makedirs('h:\\work\\images', exist_ok=True)

# 定义RMSE数据文件和对应的模型名称
RMSE_files = {
    'ARIMA': 'h:\\work\\Compared_data\\CPU\\Arima_Google_5m_test_step_rmse.npy',
    'LSTM': 'h:\\work\\Compared_data\\CPU\\Lstm_Google_5m_test_step_rmse.npy',
    'SWT-CLSTM': 'h:\\work\\Compared_data\\CPU\\SWT_CLSTM_Google_5m_test_step_rmse.npy',
    'TFC': 'h:\\work\\Compared_data\\CPU\\Tfc_Google_5m_test_step_rmse.npy',
    'TimeMixerPlusPlus': 'h:\\work\\Compared_data\\CPU\\TimeMixerPlusPlus_Google_5m_test_step_rmse.npy',
    'PatchTST': 'h:\\work\\Compared_data\\CPU\\PatchTST_Google_5m_test_step_rmse.npy'
}

# 定义其他数据集的RMSE文件配置
RMSE_files_alibaba_cpu = {
    'ARIMA': 'h:\\work\\Compared_data\\CPU\\Arima_Alibaba_30s_test_step_rmse.npy',
    'LSTM': 'h:\\work\\Compared_data\\CPU\\Lstm_Alibaba_30s_test_step_rmse.npy',
    'SWT-CLSTM': 'h:\\work\\Compared_data\\CPU\\SWT_CLSTM_Alibaba_30s_test_step_rmse.npy',
    'TFC': 'h:\\work\\Compared_data\\CPU\\Tfc_Alibaba_30s_test_step_rmse.npy',
    'TimeMixerPlusPlus': 'h:\\work\\Compared_data\\CPU\\TimeMixerPlusPlus_Alibaba_30s_test_step_rmse.npy',
    'PatchTST': 'h:\\work\\Compared_data\\CPU\\PatchTST_Alibaba_30s_test_step_rmse.npy'
}

RMSE_files_google_mem = {
    'ARIMA': 'h:\\work\\Compared_data\\Mem\\Arima_Google_5m_test_step_rmse.npy',
    'LSTM': 'h:\\work\\Compared_data\\Mem\\Lstm_Google_5m_test_step_rmse.npy',
    'SWT-CLSTM': 'h:\\work\\Compared_data\\Mem\\SWT_CLSTM_Google_5m_test_step_rmse.npy',
    'TFC': 'h:\\work\\Compared_data\\Mem\\Tfc_Google_5m_test_step_rmse.npy',
    'TimeMixerPlusPlus': 'h:\\work\\Compared_data\\Mem\\TimeMixerPlusPlus_Google_5m_test_step_rmse.npy',
    'PatchTST': 'h:\\work\\Compared_data\\Mem\\PatchTST_Google_5m_test_step_rmse.npy'
}

RMSE_files_alibaba_mem = {
    'ARIMA': 'h:\\work\\Compared_data\\Mem\\Arima_Alibaba_30s_test_step_rmse.npy',
    'LSTM': 'h:\\work\\Compared_data\\Mem\\Lstm_Alibaba_30s_test_step_rmse.npy',
    'SWT-CLSTM': 'h:\\work\\Compared_data\\Mem\\SWT_CLSTM_Alibaba_30s_test_step_rmse.npy',
    'TFC': 'h:\\work\\Compared_data\\Mem\\Tfc_Alibaba_30s_test_step_rmse.npy',
    'TimeMixerPlusPlus': 'h:\\work\\Compared_data\\Mem\\TimeMixerPlusPlus_Alibaba_30s_test_step_rmse.npy',
    'PatchTST': 'h:\\work\\Compared_data\\Mem\\PatchTST_Alibaba_30s_test_step_rmse.npy'
}

# 定义模型样式 - 使用点线样式
model_styles = {
    'ARIMA': {'color': '#3d85c6', 'linestyle': '-.', 'linewidth': 1.2, 'marker': 'o', 'markersize': 4, 'markevery': 5},
    'LSTM': {'color': '#a64d79', 'linestyle': '-.', 'linewidth': 1.2, 'marker': 'D', 'markersize': 4, 'markevery': 5},
    'SWT-CLSTM': {'color': '#f6b26b', 'linestyle': '-.', 'linewidth': 1.2, 'marker': 'x', 'markersize': 4, 'markevery': 5},
    'TFC': {'color': '#6aa84f', 'linestyle': '-.', 'linewidth': 1.2, 'marker': '^', 'markersize': 4, 'markevery': 5},
    'TimeMixerPlusPlus': {'color': '#8e7cc3', 'linestyle': '-.', 'linewidth': 1.2, 'marker': '+', 'markersize': 4, 'markevery': 5},
    'PatchTST': {'color': '#FF1493', 'linestyle': '-.', 'linewidth': 1.2, 'marker': 's', 'markersize': 4, 'markevery': 5}
}

def load_data(file_path):
    """加载NPY文件数据，处理可能的异常，直接使用RMSE值"""
    try:
        data = np.load(file_path)
        # 确保数据是一维的
        if data.ndim > 1:
            data = data.flatten()
        # 直接使用RMSE值，不再平方转换为MSE
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])

def calculate_cdf(data):
    """计算数据的累积分布函数，并通过插值增加数据点使曲线更平滑"""
    # 排序数据
    sorted_data = np.sort(data)
    
    # 添加起点(0,0)
    sorted_data = np.insert(sorted_data, 0, 0)
    
    # 计算原始累积概率
    y_orig = np.arange(0, len(sorted_data)) / (len(sorted_data) - 1)
    
    # 使用插值增加数据点数量，使曲线更平滑
    # 创建更密集的x轴点
    num_points = len(sorted_data) * 10  # 增加10倍的数据点
    x_interp = np.linspace(sorted_data[0], sorted_data[-1], num_points)
    
    # 使用线性插值计算对应的y值
    y_interp = np.interp(x_interp, sorted_data, y_orig)
    
    return x_interp, y_interp

# 设置matplotlib参数，恢复到修改前的样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,  # 从10增加到14
    'axes.labelsize': 16,  # 从11增加到16
    'axes.titlesize': 18,  # 从12增加到18
    'xtick.labelsize': 14,  # 从9增加到14
    'ytick.labelsize': 14,  # 从9增加到14
    'legend.fontsize': 12,  # 从8增加到12
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.figsize': (8, 6),
    'figure.autolayout': True,
    'text.usetex': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'grid.linewidth': 0.5
})

def plot_RMSE_cdf():
    """绘制不同模型RMSE值的CDF图（使用对数横坐标）"""
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # 存储所有RMSE值以确定x轴范围
    all_RMSE_values = []
    model_data = {}  # 存储每个模型的数据，用于后续绘制组合图
    
    # 处理每个模型
    for model_name, file_path in RMSE_files.items():
        print(f"处理 {model_name} 模型的RMSE数据...")
        
        # 加载RMSE数据
        RMSE_data = load_data(file_path)
        
        if len(RMSE_data) == 0:
            print(f"  警告: 无法加载 {model_name} 的RMSE数据，跳过")
            continue
        
        # 存储模型数据
        model_data[model_name] = RMSE_data
        
        # 收集所有RMSE值
        all_RMSE_values.extend(RMSE_data)
        
        # 计算CDF，使用插值增加数据点
        x_values, y_values = calculate_cdf(RMSE_data)
        
        # 绘制CDF曲线 - 使用原始样式
        ax.plot(x_values, y_values, 
                color=model_styles[model_name]['color'],
                linestyle=model_styles[model_name]['linestyle'],
                linewidth=model_styles[model_name]['linewidth'],
                marker=model_styles[model_name]['marker'],
                markersize=model_styles[model_name]['markersize'],
                markevery=model_styles[model_name]['markevery'],
                label=model_name)
    
    # 设置坐标轴标签
    ax.set_xlabel('RMSE (log scale)', fontsize=16)  # 从11增加到16
    ax.set_ylabel('Cumulative Distribution Function (CDF)', fontsize=16)  # 从11增加到16
    
    # 设置x轴范围，确保所有数据都能显示，并从最小非零值开始
    min_RMSE = min([x for x in all_RMSE_values if x > 0]) * 0.9
    max_RMSE = max(all_RMSE_values) * 1.1
    
    # 设置对数刻度
    ax.set_xscale('log')
    ax.set_xlim(min_RMSE, max_RMSE)
    
    # 增加X轴刻度点个数 - 设置更多的主刻度和次刻度
    from matplotlib.ticker import LogLocator, LogFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))  # 增加主刻度数量
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))  # 增加次刻度
    ax.xaxis.set_major_formatter(LogFormatter())
    
    # 增加网格密度 - 添加主网格和次网格
    ax.grid(True, which='major', linestyle='-', alpha=0.6, linewidth=0.8)  # 主网格线
    ax.grid(True, which='minor', linestyle=':', alpha=0.4, linewidth=0.5)  # 次网格线
    
    # 设置y轴范围
    ax.set_ylim(0, 1.05)
    
    # 增加Y轴网格密度
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))  # 每0.1一个主刻度
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))  # 每0.05一个次刻度
    
    # 格式化y轴为百分比
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    # 添加图例 - 移至右上角
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # 添加边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    
    # 保存图表
    plt.savefig('h:\\work\\images\\Cumulative Distribution of RMSE Values.png', bbox_inches='tight', dpi=300)
    plt.savefig('h:\\work\\images\\Cumulative Distribution of RMSE Values.pdf', bbox_inches='tight')
    
    # 明确指定SVG格式并确保正确保存
    plt.savefig('h:\\work\\images\\Cumulative Distribution of RMSE Values.svg', format='svg', bbox_inches='tight')
    
    print("\nLog scale CDF chart has been saved to h:\\work\\images\\Cumulative Distribution of RMSE Values.png, .pdf and .svg")
    
    # 返回收集的模型数据
    return model_data

def plot_RMSE_combined_chart(model_data):
    """绘制不同模型MSE值的散点、箱线、小提琴组合图"""
    # 检查是否有数据
    if not model_data:
        print("没有可用的模型数据进行绘图")
        return
    
    # 创建图形 - 保持相同的整体尺寸
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # 准备数据
    positions = []
    labels = []
    data_list = []
    colors = []
    
    # 获取模型名称和对应的数据
    for i, (model_name, data) in enumerate(model_data.items()):
        positions.append(i)
        labels.append(model_name)
        data_list.append(data)
        # 使用浅色版本的模型颜色，但比原来更深一些
        base_color = model_styles[model_name]['color']
        # 将十六进制颜色转换为RGB，然后调整亮度
        r, g, b = int(base_color[1:3], 16)/255, int(base_color[3:5], 16)/255, int(base_color[5:7], 16)/255
        # 创建更深的颜色版本 (减少亮度增加的幅度)
        light_color = (r*0.8+0.2, g*0.8+0.2, b*0.8+0.2, 0.8)  # 减少亮度增加幅度并增加不透明度
        colors.append(light_color)
    
    # 增加图表的有效显示区域 - 减小边距
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1)
    
    # 绘制散点图 - 仅显示左半部分
    for i, data in enumerate(data_list):
        # 创建抖动效果，但限制在左侧
        jitter = np.random.normal(-0.15, 0.05, size=len(data))
        ax.scatter(np.full_like(data, positions[i]) + jitter, data, 
                  color=model_styles[labels[i]]['color'], alpha=0.4, s=20)
    
    # 绘制箱线图 - 居中显示
    box_parts = ax.boxplot(data_list, positions=positions, patch_artist=True,
                          widths=0.3, showfliers=False, medianprops={'color': 'black', 'linewidth': 1.5})
    
    # 自定义箱线图的颜色和样式
    for i, box in enumerate(box_parts['boxes']):
        box.set(facecolor='white', edgecolor=model_styles[labels[i]]['color'], linewidth=1.5, alpha=0.8)
    
    # 修复：使用自定义方法绘制右侧小提琴图
    from scipy import stats
    for i, data in enumerate(data_list):
        # 计算核密度估计
        kde_xs = np.linspace(min(data), max(data), 100)
        kde = stats.gaussian_kde(data)
        kde_ys = kde(kde_xs)
        
        # 归一化密度值
        kde_ys = kde_ys / kde_ys.max() * 0.4  # 控制小提琴图的宽度
        
        # 构建右侧小提琴图的坐标点
        x_coords = []
        y_coords = []
        
        # 从上到下添加左侧边界点（箱线图中心线）
        for j in range(len(kde_xs)):
            x_coords.append(positions[i])
            y_coords.append(kde_xs[j])
        
        # 从下到上添加右侧曲线点
        for j in range(len(kde_xs)-1, -1, -1):
            x_coords.append(positions[i] + kde_ys[j])
            y_coords.append(kde_xs[j])
        
        # 绘制右侧小提琴图
        ax.fill(x_coords, y_coords, alpha=0.7, color=colors[i], edgecolor='none')
    
    # 设置坐标轴 - 增加标签字体大小
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=14)  # 从11增加到14
    ax.set_ylabel('RMSE Values', fontsize=16, fontweight='bold')  # 从12增加到16
    
    # 计算并标注每个模型的均值和中位数 - 仅在上方标注均值
    for i, data in enumerate(data_list):
        mean_val = np.mean(data)
        
        # 在箱线图上方添加均值标注
        ax.text(positions[i], np.percentile(data, 98), 
                f"Mean: {mean_val:.6f}", ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='dimgrey')  # 从9增加到12
    
    # 增加网格密度 - 添加主网格和次网格
    ax.yaxis.grid(True, which='major', linestyle='-', alpha=0.6, linewidth=0.8)  # 主网格线
    ax.yaxis.grid(True, which='minor', linestyle=':', alpha=0.4, linewidth=0.5)  # 次网格线
    ax.set_axisbelow(True)  # 网格线置于图形元素下方
    
    # 增加Y轴刻度密度
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    major_tick_interval = y_range / 10  # 10个主刻度
    minor_tick_interval = major_tick_interval / 5  # 每个主刻度间5个次刻度
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(major_tick_interval))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_interval))
    
    # 将图例移到图内正中间上方
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=1.8, label='Median'),
        plt.Rectangle((0, 0), 1, 1, fc='white', ec='black', alpha=0.9, linewidth=1.2, label='IQR (25%-75%)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='dimgrey', markersize=8, alpha=0.6, label='Data Points (Left)'),
        plt.Polygon([(0, 0), (0, 0), (0, 0)], fc='dimgrey', alpha=0.8, label='Distribution (Right)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
              ncol=2, frameon=True, framealpha=0.95, fontsize=12)  # 移到图内正中间上方
    
    # 美化图表
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    
    # 保存图表
    plt.savefig('h:\\work\\images\\Distribution of RMSE Values Across Models.png', bbox_inches='tight', dpi=300)
    plt.savefig('h:\\work\\images\\Distribution of RMSE Values Across Models.pdf', bbox_inches='tight')
    
    # 明确指定SVG格式并确保正确保存
    plt.savefig('h:\\work\\images\\Distribution of RMSE Values Across Models.svg', format='svg', bbox_inches='tight')
    
    print("\nCombined chart has been saved to h:\\work\\images\\Distribution of RMSE Values Across Models.png, .pdf and .svg")

# 修改主函数，添加新图表的绘制
if __name__ == "__main__":
    # 绘制RMSE的CDF图
    model_data = plot_RMSE_cdf()
    
    # 绘制RMSE的散点、箱线、小提琴组合图
    plot_RMSE_combined_chart(model_data)
    
    print("所有图表已生成完成！")