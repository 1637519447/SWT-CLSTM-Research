import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.max_open_warning'] = 50

def load_model_data(data_type, dataset, model_name):
    """加载模型预测数据和RMSE数据"""
    try:
        # 加载预测数据
        pred_file = f"h:\\work\\Pre_data\\{data_type}\\{model_name}_{dataset}_test.npy"
        pred_data = np.load(pred_file)
        
        # 加载RMSE数据 - 处理Mem目录下LSTM文件的特殊命名
        if data_type == 'Mem' and model_name == 'Lstm':
            if dataset == 'Alibaba_30s':
                rmse_file = f"h:\\work\\Compared_data\\{data_type}\\{model_name}_30s_test_step_rmse.npy"
            elif dataset == 'Google_5m':
                rmse_file = f"h:\\work\\Compared_data\\{data_type}\\{model_name}_5m_test_step_rmse.npy"
            else:
                rmse_file = f"h:\\work\\Compared_data\\{data_type}\\{model_name}_{dataset}_test_step_rmse.npy"
        else:
            rmse_file = f"h:\\work\\Compared_data\\{data_type}\\{model_name}_{dataset}_test_step_rmse.npy"
        
        rmse_data = np.load(rmse_file)
        
        return pred_data, rmse_data
    except FileNotFoundError as e:
        print(f"警告: 无法找到 {model_name} 在 {dataset} 数据集的文件: {e}")
        return None, None

def load_ground_truth(data_type, dataset):
    """加载真实值数据"""
    try:
        gt_file = f"h:\\work\\Pre_data\\{data_type}\\{dataset}_test_ground_truth.npy"
        return np.load(gt_file)
    except FileNotFoundError:
        print(f"警告: 无法找到 {dataset} 数据集的真实值文件")
        return None

def calculate_mse_errors(predictions, ground_truth):
    """计算每个预测点的MSE误差"""
    if predictions is None or ground_truth is None:
        return None
    
    # 确保数据长度一致
    min_len = min(len(predictions), len(ground_truth))
    pred_trimmed = predictions[:min_len]
    gt_trimmed = ground_truth[:min_len]
    
    # 计算每个点的平方误差
    mse_errors = (pred_trimmed.flatten() - gt_trimmed.flatten()) ** 2
    return mse_errors

def perform_t_test(errors1, errors2, model1_name, model2_name):
    """执行配对t检验"""
    if errors1 is None or errors2 is None:
        return None
    
    # 确保数据长度一致
    min_len = min(len(errors1), len(errors2))
    errors1_trimmed = errors1[:min_len]
    errors2_trimmed = errors2[:min_len]
    
    # 执行配对t检验
    t_stat, p_value = stats.ttest_rel(errors1_trimmed, errors2_trimmed)
    
    # 计算效应大小 (Cohen's d)
    diff = errors1_trimmed - errors2_trimmed
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'model1_mean_error': np.mean(errors1_trimmed),
        'model2_mean_error': np.mean(errors2_trimmed),
        'model1_name': model1_name,
        'model2_name': model2_name
    }

def perform_wilcoxon_test(errors1, errors2, model1_name, model2_name):
    """执行Wilcoxon符号秩检验（非参数检验）"""
    if errors1 is None or errors2 is None:
        return None
    
    # 确保数据长度一致
    min_len = min(len(errors1), len(errors2))
    errors1_trimmed = errors1[:min_len]
    errors2_trimmed = errors2[:min_len]
    
    # 执行Wilcoxon符号秩检验
    try:
        w_stat, p_value = stats.wilcoxon(errors1_trimmed, errors2_trimmed)
        return {
            'w_statistic': w_stat,
            'p_value': p_value,
            'model1_median_error': np.median(errors1_trimmed),
            'model2_median_error': np.median(errors2_trimmed),
            'model1_name': model1_name,
            'model2_name': model2_name
        }
    except ValueError:
        return None

def main():
    """主函数"""
    # 定义模型和数据集
    models = ['SWT_CLSTM', 'Lstm', 'Arima', 'Tfc', 'PatchTST', 'TimeMixerPlusPlus']
    datasets = ['Alibaba_30s', 'Google_5m']
    data_types = ['CPU', 'Mem']
    
    # 存储所有结果
    all_t_test_results = []
    all_wilcoxon_results = []
    
    print("开始统计显著性检验分析...")
    print("="*60)
    
    for data_type in data_types:
        print(f"\n处理 {data_type} 数据:")
        print("-"*40)
        
        for dataset in datasets:
            print(f"\n数据集: {dataset}")
            
            # 加载真实值
            ground_truth = load_ground_truth(data_type, dataset)
            if ground_truth is None:
                continue
            
            # 加载所有模型数据并计算误差
            model_errors = {}
            model_rmse = {}
            
            for model in models:
                pred_data, rmse_data = load_model_data(data_type, dataset, model)
                if pred_data is not None:
                    errors = calculate_mse_errors(pred_data, ground_truth)
                    if errors is not None:
                        model_errors[model] = errors
                        model_rmse[model] = rmse_data
            
            # 以SWT_CLSTM为基准，与其他模型进行比较
            if 'SWT_CLSTM' not in model_errors:
                print(f"  警告: SWT_CLSTM 数据不可用于 {dataset}")
                continue
            
            swtclstm_errors = model_errors['SWT_CLSTM']
            
            print(f"  SWT_CLSTM vs 其他模型的统计检验结果:")
            
            for model in models:
                if model == 'SWT_CLSTM' or model not in model_errors:
                    continue
                
                # 执行t检验
                t_result = perform_t_test(model_errors[model], swtclstm_errors, model, 'SWT_CLSTM')
                if t_result:
                    t_result['data_type'] = data_type
                    t_result['dataset'] = dataset
                    all_t_test_results.append(t_result)
                    
                    # 执行Wilcoxon检验
                    w_result = perform_wilcoxon_test(model_errors[model], swtclstm_errors, model, 'SWT_CLSTM')
                    if w_result:
                        w_result['data_type'] = data_type
                        w_result['dataset'] = dataset
                        all_wilcoxon_results.append(w_result)
                    
                    # 打印结果
                    significance = "***" if t_result['p_value'] < 0.001 else "**" if t_result['p_value'] < 0.01 else "*" if t_result['p_value'] < 0.05 else "ns"
                    print(f"    {model:15} vs SWT_CLSTM: t={t_result['t_statistic']:6.3f}, p={t_result['p_value']:8.6f} {significance}")
                    print(f"                     Cohen's d={t_result['cohens_d']:6.3f}, 平均误差比: {t_result['model1_mean_error']/t_result['model2_mean_error']:6.3f}")
    
    # 创建结果汇总表
    print("\n" + "="*80)
    print("统计显著性检验汇总结果")
    print("="*80)
    
    # 转换为DataFrame便于分析
    t_test_df = pd.DataFrame(all_t_test_results)
    wilcoxon_df = pd.DataFrame(all_wilcoxon_results)
    
    if not t_test_df.empty:
        # 按p值排序
        t_test_df_sorted = t_test_df.sort_values('p_value')
        
        print("\nT检验结果 (按p值排序):")
        print("-"*80)
        for _, row in t_test_df_sorted.iterrows():
            significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
            improvement = (row['model1_mean_error'] - row['model2_mean_error']) / row['model2_mean_error'] * 100
            print(f"{row['data_type']:3} {row['dataset']:12} {row['model1_name']:15} vs SWT_CLSTM: "
                  f"p={row['p_value']:8.6f} {significance:3} (改进: {improvement:+6.2f}%)")
    
    # 创建可视化图表
    create_significance_plots(t_test_df, wilcoxon_df)
    
    # 保存结果到文件
    if not t_test_df.empty:
        t_test_df.to_csv('h:\\work\\statistical_t_test_results.csv', index=False, encoding='utf-8-sig')
        print(f"\nT检验结果已保存到: h:\\work\\statistical_t_test_results.csv")
    
    if not wilcoxon_df.empty:
        wilcoxon_df.to_csv('h:\\work\\statistical_wilcoxon_results.csv', index=False, encoding='utf-8-sig')
        print(f"Wilcoxon检验结果已保存到: h:\\work\\statistical_wilcoxon_results.csv")
    
    print("\n统计显著性检验分析完成！")

def create_significance_plots(t_test_df, wilcoxon_df):
    """创建统计显著性检验的可视化图表"""
    if t_test_df.empty:
        return
    
    # 创建图表目录
    import os
    os.makedirs('h:\\work\\images', exist_ok=True)
    
    # 1. p值热力图
    plt.figure(figsize=(14, 10))
    
    # 准备热力图数据
    pivot_data = t_test_df.pivot_table(
        values='p_value', 
        index=['data_type', 'dataset'], 
        columns='model1_name', 
        fill_value=1.0
    )
    
    # 创建显著性标记
    annot_data = pivot_data.copy()
    for i in range(annot_data.shape[0]):
        for j in range(annot_data.shape[1]):
            p_val = annot_data.iloc[i, j]
            if p_val < 0.001:
                annot_data.iloc[i, j] = f"{p_val:.1e}***"
            elif p_val < 0.01:
                annot_data.iloc[i, j] = f"{p_val:.3f}**"
            elif p_val < 0.05:
                annot_data.iloc[i, j] = f"{p_val:.3f}*"
            else:
                annot_data.iloc[i, j] = f"{p_val:.3f}"
    
    # 绘制热力图
    sns.heatmap(pivot_data, annot=annot_data, fmt='', cmap='RdYlBu_r', 
                center=0.05, vmin=0, vmax=0.1, cbar_kws={'label': 'p值'})
    plt.title('SWT_CLSTM vs 其他模型的统计显著性检验 (p值热力图)\n***p<0.001, **p<0.01, *p<0.05', 
              fontsize=14, pad=20)
    plt.xlabel('对比模型', fontsize=12)
    plt.ylabel('数据类型_数据集', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    # plt.savefig('h:\\work\\images\\statistical_significance_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.savefig('h:\\work\\images\\statistical_significance_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. 效应大小图
    plt.figure(figsize=(12, 8))
    
    # 按效应大小排序
    t_test_sorted = t_test_df.sort_values('cohens_d', ascending=True)
    
    # 创建条形图
    colors = ['red' if p < 0.05 else 'gray' for p in t_test_sorted['p_value']]
    bars = plt.barh(range(len(t_test_sorted)), t_test_sorted['cohens_d'], color=colors, alpha=0.7)
    
    # 添加标签
    labels = [f"{row['data_type']}_{row['dataset']}_{row['model1_name']}" for _, row in t_test_sorted.iterrows()]
    plt.yticks(range(len(t_test_sorted)), labels)
    plt.xlabel('Cohen\'s d (效应大小)', fontsize=12)
    plt.title('SWT_CLSTM vs 其他模型的效应大小\n红色表示p<0.05（显著差异）', fontsize=14, pad=20)
    
    # 添加垂直线表示效应大小阈值
    plt.axvline(x=0.2, color='green', linestyle='--', alpha=0.5, label='小效应 (0.2)')
    plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='中等效应 (0.5)')
    plt.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='大效应 (0.8)')
    plt.legend()
    
    plt.tight_layout()
    # plt.savefig('h:\\work\\images\\effect_size_comparison.png', dpi=300, bbox_inches='tight')
    # plt.savefig('h:\\work\\images\\effect_size_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. 性能改进百分比图
    plt.figure(figsize=(12, 8))
    
    # 计算性能改进百分比
    improvement_pct = ((t_test_df['model1_mean_error'] - t_test_df['model2_mean_error']) / 
                      t_test_df['model2_mean_error'] * 100)
    
    # 创建数据框
    improvement_df = t_test_df.copy()
    improvement_df['improvement_pct'] = improvement_pct
    improvement_df = improvement_df.sort_values('improvement_pct', ascending=True)
    
    # 创建条形图
    colors = ['green' if p < 0.05 and imp > 0 else 'red' if p < 0.05 and imp < 0 else 'gray' 
              for p, imp in zip(improvement_df['p_value'], improvement_df['improvement_pct'])]
    
    bars = plt.barh(range(len(improvement_df)), improvement_df['improvement_pct'], color=colors, alpha=0.7)
    
    # 添加标签
    labels = [f"{row['data_type']}_{row['dataset']}_{row['model1_name']}" for _, row in improvement_df.iterrows()]
    plt.yticks(range(len(improvement_df)), labels)
    plt.xlabel('性能改进百分比 (%)', fontsize=12)
    plt.title('SWT_CLSTM相对于其他模型的性能改进\n绿色：显著改进，红色：显著退化，灰色：无显著差异', fontsize=14, pad=20)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    # plt.savefig('h:\\work\\images\\performance_improvement.png', dpi=300, bbox_inches='tight')
    # plt.savefig('h:\\work\\images\\performance_improvement.pdf', bbox_inches='tight')
    plt.close()
    
    # print("\n可视化图表已保存到 h:\\work\\images\\ 目录:")
    # print("  - statistical_significance_heatmap.png/pdf: p值热力图")
    # print("  - effect_size_comparison.png/pdf: 效应大小比较")
    # print("  - performance_improvement.png/pdf: 性能改进百分比")

if __name__ == "__main__":
    main()