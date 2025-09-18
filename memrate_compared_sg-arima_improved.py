# ARIMA模型预测内存使用率
# 含滤波，ARIMA
# 改进版：支持多个时间间隔数据集的训练与评估
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import os
import time
import json
import math
from joblib import dump, load
from tqdm import tqdm
import warnings
import torch
from torch import nn
import torch.optim as optim

# 忽略所有警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置matplotlib参数，解决中文显示问题
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rcParams['figure.max_open_warning'] = 50  # 避免图形过多警告
# 确保使用Agg后端，避免中文显示问题
matplotlib.use('Agg')

# 设置ARIMA模型参数
p, d, q = 7, 1, 2

# 定义要处理的数据集列表
datasets = [
    {
        'name': 'Alibaba_30s',
        'file': 'Alibaba_mem_util_aggregated_30s.csv',
        'results_dir': 'h:\\work\\alibaba_mem_arima_results_30s\\'
    },
    {
        'name': 'Google_5m',
        'file': 'Google_mem_util_aggregated_5m.csv',
        'results_dir': 'h:\\work\\google_mem_arima_results_5m\\'
    }
]

# 创建统一的结果目录
unified_results_dir = 'h:\\work\\Mem_arima_unified_results\\'
os.makedirs(unified_results_dir, exist_ok=True)

# 创建结果目录
for dataset in datasets:
    os.makedirs(dataset['results_dir'], exist_ok=True)


# 计算不同预测步长的RMSE
def calculate_step_rmse(y_true, y_pred, max_steps=30):
    """
    计算不同预测步长的RMSE

    参数:
    y_true: 真实值
    y_pred: 预测值
    max_steps: 最大预测步长

    返回:
    step_rmse: 不同步长的RMSE列表
    """
    step_rmse = []

    for step in range(1, max_steps + 1):
        if step < len(y_true) and step < len(y_pred):
            # 确定比较的数据长度
            compare_length = min(len(y_true) - step, len(y_pred) - step)
            if compare_length <= 0:
                break

            # 从step开始比较
            y_true_segment = y_true[step:step + compare_length]
            y_pred_segment = y_pred[step:step + compare_length]

            # 将数据转换为二维数组，如果它们是一维的
            if y_true_segment.ndim == 1:
                y_true_segment = y_true_segment.reshape(-1, 1)
            if y_pred_segment.ndim == 1:
                y_pred_segment = y_pred_segment.reshape(-1, 1)

            # 计算MSE和RMSE
            mse = mean_squared_error(y_true_segment, y_pred_segment)
            rmse = math.sqrt(mse)
            step_rmse.append(rmse)
        else:
            break

    return step_rmse


# 计算ARIMA模型复杂度的函数
def calculate_arima_complexity(p, d, q, n_samples):
    """
    估计ARIMA模型的计算复杂度
    
    参数:
    p: AR阶数
    d: 差分阶数
    q: MA阶数
    n_samples: 样本数量
    
    返回:
    macs: 乘加运算次数估计
    params: 模型参数数量
    """
    # 参数数量: AR参数 + MA参数 + 常数项
    params = p + q + 1
    
    # 估计每次预测的乘加运算次数
    # AR部分: p个参数乘以p个历史值
    ar_macs = p * p
    # MA部分: q个参数乘以q个历史误差
    ma_macs = q * q
    # 差分部分: 每个差分级别需要的操作
    diff_macs = d * 2
    
    # 每个样本的总MACs
    per_sample_macs = ar_macs + ma_macs + diff_macs + params
    
    # 训练过程中的总MACs (考虑迭代优化)
    # 假设平均需要100次迭代来拟合模型
    iterations = 100
    total_macs = per_sample_macs * n_samples * iterations
    
    return total_macs, params


# 主函数：处理每个数据集
def process_dataset(dataset_info):
    dataset_name = dataset_info['name']
    file_path = dataset_info['file']
    results_dir = dataset_info['results_dir']
    
    print(f"\n{'='*50}")
    print(f"处理数据集: {dataset_name} (文件: {file_path})")
    print(f"{'='*50}\n")
    
    # 加载数据
    try:
        data = np.loadtxt(file_path, delimiter=' ')
        data = data[data != 0]  # 去除0元素
    except:
        print(f"无法加载文件，使用随机数据进行测试")
        # 生成测试数据
        np.random.seed(42)
        data = np.random.rand(1000) * 100
    
    # 使用Savitzky-Golay滤波器去噪
    window_length = min(11, len(data) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    
    if window_length < 3:
        smoothed_data = data.copy()
        print("数据长度过短，跳过滤波")
    else:
        smoothed_data = savgol_filter(data, window_length=window_length, polyorder=min(2, window_length-1))
    
    # 确保数据长度为2的幂次方，这对某些算法很重要
    # 计算最接近的2的幂次方
    power = int(np.ceil(np.log2(len(smoothed_data))))
    padded_length = 2**power
    
    # 如果数据长度不是2的幂次方，进行填充
    if len(smoothed_data) != padded_length:
        # 使用边界值填充，保持数据特性
        pad_width = padded_length - len(smoothed_data)
        smoothed_data = np.pad(smoothed_data, (0, pad_width), mode='symmetric')
        print(f"数据长度已填充至 {padded_length} (2^{power})")
    else:
        print(f"数据长度已经是2的幂次方: {padded_length}")
    
    # 分割数据为训练集和测试集
    train_size = int(len(smoothed_data) * 0.8)
    train, test = smoothed_data[:train_size], smoothed_data[train_size:]
    
    # 从训练集中删除前70个数据点
    if len(train) > 70:
        train = train[70:]
        print(f"已从训练集中删除前70个数据点")
    else:
        print(f"警告: 训练集长度不足70，无法删除前70个数据点")
    
    print(f"训练集长度: {len(train)}")
    print(f"测试集长度: {len(test)}")
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    scaled_test = scaler.transform(test.reshape(-1, 1)).flatten()
    
    # 保存scaler
    dump(scaler, os.path.join(results_dir, 'scaler.joblib'))
    
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"当前CUDA版本: {torch.version.cuda}")
        print(f"当前PyTorch版本: {torch.__version__}")
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到GPU，将使用CPU进行训练")
    
    # 记录训练开始时间
    train_start_time = time.time()
    
    # 训练 ARIMA 模型
    model = ARIMA(scaled_train, order=(p, d, q))
    model_fit = model.fit()
    
    # 保存训练好的模型
    model_save_path = os.path.join(results_dir, 'arima_model.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_fit, f)
    print(f"模型已保存到: {model_save_path}")
    
    # 计算模型复杂度
    macs, params = calculate_arima_complexity(p, d, q, len(scaled_train))
    print(f"模型复杂度估计:")
    print(f"  - MACs (乘加运算次数): {macs:,}")
    print(f"  - Parameters (参数数量): {params}")
    
    # 计算训练时间（毫秒）
    train_time_ms = (time.time() - train_start_time) * 1000
    
    # 记录预测开始时间
    predict_start_time = time.time()
    
    # 使用多步预测方法，每次预测2步
    test_predict = []
    history = list(scaled_train)  # 使用训练集的历史数据
    
    # 使用tqdm显示预测进度
    pbar = tqdm(total=len(scaled_test), desc="多步预测测试集(每次2步)")
    
    # 记录每一步的预测时间
    step_times = []
    
    # 逐步预测
    i = 0
    while i < len(scaled_test):
        # 记录预测开始时间
        step_start_time = time.time()
        
        # 确定当前批次的预测步数
        remaining = len(scaled_test) - i
        current_steps = min(2, remaining)  # 每次预测2步或剩余的步数
        
        # 使用历史数据训练模型
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit()
        
        # 预测多个时间点
        forecast = model_fit.forecast(steps=current_steps)
        
        # 计算预测时间并记录
        step_time = (time.time() - step_start_time) * 1000  # 转换为毫秒
        step_times.append(step_time)
        
        # 逐步添加预测值和实际值到历史数据
        for j in range(current_steps):
            if i + j < len(scaled_test):
                test_predict.append(forecast[j])
                # 更新历史数据 - 添加实际值
                history.append(scaled_test[i + j])
                # 更新进度
                pbar.update(1)
        
        # 移动到下一批次
        i += current_steps
    
    pbar.close()
    
    # 计算总预测时间（毫秒）
    prediction_time_ms = (time.time() - predict_start_time) * 1000
    
    # 计算平均单步预测时间（毫秒）
    per_sample_time_ms = np.mean(step_times) if step_times else 0
    
    # 计算单步预测时间的标准差
    per_sample_time_std_ms = np.std(step_times) if step_times else 0
    per_sample_time_ms = prediction_time_ms / len(scaled_test)
    
    # 反向转换预测值
    test_predict = scaler.inverse_transform(np.array(test_predict).reshape(-1, 1))
    y_test_original = scaler.inverse_transform(scaled_test.reshape(-1, 1))
    
    # 从第71个预测值开始计算评估指标
    start_idx = 70  # 从第71个值开始（索引从0开始）
    
    # 确保有足够的数据
    if len(test_predict) > start_idx:
        print(f"从第{start_idx+1}个预测值开始计算评估指标")
        test_predict_eval = test_predict[start_idx:]
        y_test_original_eval = y_test_original[start_idx:]
    else:
        print(f"警告: 预测值数量不足{start_idx+1}个，使用全部数据计算评估指标")
        test_predict_eval = test_predict
        y_test_original_eval = y_test_original
    
    # 计算各种评估指标
    mse = mean_squared_error(y_test_original_eval, test_predict_eval)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_original_eval, test_predict_eval)
    r2 = r2_score(y_test_original_eval, test_predict_eval)
    
    # 计算MAPE (平均绝对百分比误差)
    epsilon = 1e-10  # 避免除以零
    mape = np.mean(np.abs((y_test_original_eval - test_predict_eval) / (y_test_original_eval + epsilon))) * 100

    # 计算不同预测步长的RMSE - 也从第71个值开始
    max_steps = 30  # 默认最大预测步长
    if dataset_name == 'Alibaba_30s':
        max_steps = 90  # 30秒级别，最大90步
    elif dataset_name == 'Google_5m':
        max_steps = 60  # 5分钟级别，最大60步
    elif dataset_name == '1h':
        max_steps = 30  # 1小时级别，最大30步

    step_rmse = calculate_step_rmse(y_test_original_eval, test_predict_eval, max_steps)

    # 保存不同步长的RMSE
    np.save(os.path.join(results_dir, 'test_step_rmse.npy'), np.array(step_rmse))

    # 同时保存到统一目录
    np.save(os.path.join(unified_results_dir, f'Arima_{dataset_name}_test.npy'), test_predict)
    np.save(os.path.join(unified_results_dir, f'Arima_test_ground_truth_{dataset_name}.npy'), y_test_original)
    np.save(os.path.join(unified_results_dir, f'Arima_{dataset_name}_test_step_rmse.npy'), np.array(step_rmse))

    # 计算Log RMSE (避免负值或零值)
    y_test_log = np.log1p(np.maximum(y_test_original, 0))
    y_pred_log = np.log1p(np.maximum(test_predict, 0))
    log_rmse = math.sqrt(mean_squared_error(y_test_log, y_pred_log))

    # 保存评估指标
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'log_rmse': float(log_rmse),
        'mape': float(mape),  # 添加MAPE指标
        'prediction_time_ms': float(prediction_time_ms),
        'per_sample_time_ms': float(per_sample_time_ms),
        'per_sample_time_std_ms': float(per_sample_time_std_ms),  # 添加单步预测时间的标准差
        'train_time_ms': float(train_time_ms),
        'step_rmse': [float(x) for x in step_rmse],  # 添加步长RMSE
        'macs': float(macs),  # 添加MACs信息
        'params': float(params)  # 添加参数数量信息
    }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # 保存预测结果
    np.save(os.path.join(results_dir, 'predictions.npy'), test_predict)
    np.save(os.path.join(results_dir, 'ground_truth.npy'), y_test_original)

    # 保存模型参数
    model_params = {
        'p': p,
        'd': d,
        'q': q,
        'order': f"({p},{d},{q})",
        'multi_step': 2,  # 修改为每次预测2步
        'macs': float(macs),  # 添加MACs信息
        'params': float(params)  # 添加参数数量信息
    }

    with open(os.path.join(results_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)

    # 合并所有结果
    results = {**metrics, **model_params, 'train_time_ms': train_time_ms}

    # 保存所有结果的汇总
    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n数据集 {dataset_name} 处理完成")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
    print(f"训练时间: {train_time_ms:.2f} 毫秒")
    print(f"预测时间: {prediction_time_ms:.2f} 毫秒 (每样本: {per_sample_time_ms:.2f} ± {per_sample_time_std_ms:.2f} 毫秒)")
    print(f"模型复杂度: MACs={macs:,}, 参数数量={params}")
    
    return results


# 主程序
if __name__ == "__main__":
    # 处理所有数据集
    all_datasets_results = {}

    # 创建模型复杂度汇总表
    complexity_summary = {
        'model_name': 'ARIMA',
        'datasets': {}
    }

    for dataset in datasets:
        print(f"\n开始处理数据集: {dataset['name']}")
        results = process_dataset(dataset)
        all_datasets_results[dataset['name']] = results
        
        # 添加到复杂度汇总
        complexity_summary['datasets'][dataset['name']] = {
            'macs': results['macs'],
            'params': results['params']
        }
    
    # 保存模型复杂度汇总
    with open('h:\\work\\mem_arima_model_complexity.json', 'w') as f:
        json.dump(complexity_summary, f, indent=4)
    
    # 同时保存到统一目录
    with open(os.path.join(unified_results_dir, 'arima_model_complexity.json'), 'w') as f:
        json.dump(complexity_summary, f, indent=4)

    print("\n所有数据集处理完成！")
    print(f"结果已保存到各自的目录中，汇总结果保存在: {unified_results_dir}")
    print(f"模型复杂度汇总保存在: h:\\work\\mem_arima_model_complexity.json")