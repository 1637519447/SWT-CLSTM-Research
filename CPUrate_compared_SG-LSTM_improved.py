# 去除小波分解与CNN（纯LSTM）
# 含滤波，LSTM，无小波分解与cnn
# 使用PyTorch实现
# 改进版：支持多个时间间隔数据集的训练与评估

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time
import json
import math
from joblib import dump, load
from tqdm import tqdm
import warnings
from thop import profile, clever_format  # 添加thop库用于计算MACs和参数量

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

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"当前CUDA版本: {torch.version.cuda}")
    print(f"当前PyTorch版本: {torch.__version__}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到GPU，将使用CPU进行训练")

# 定义要处理的数据集列表
datasets = [
    {
        'name': 'Alibaba_30s',
        'file': 'Alibaba_cpu_util_aggregated_30s.csv',
        'results_dir': 'h:\\work\\CPU_lstm_alibaba_results_30s\\'
    },
    {
        'name': 'Google_5m',
        'file': 'Google_cpu_util_aggregated_5m.csv',
        'results_dir': 'h:\\work\\CPU_lstm_google_results_5m\\'
    }
]

# 创建统一的结果目录
unified_results_dir = 'h:\\work\\CPU_lstm_unified_results\\'
os.makedirs(unified_results_dir, exist_ok=True)

# 创建结果目录
for dataset in datasets:
    os.makedirs(dataset['results_dir'], exist_ok=True)

# 准备LSTM模型训练数据
def create_dataset(dataset, look_back=1, for_prediction=False):
    X, Y = [], []
    if for_prediction:
        # 如果是用于预测，我们需要保留完整的输入序列
        for i in range(len(dataset) - look_back + 1):
            a = dataset[i:(i + look_back)]
            X.append(a)
            if i < len(dataset) - look_back:
                Y.append(dataset[i + look_back])
            else:
                # 在最后一个片段，Y可以是None或者最后一个值
                Y.append(None)
    else:
        # 如果不是用于预测，按照原来的逻辑
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back)]
            X.append(a)
            Y.append(dataset[i + look_back])

    # 返回X和Y，但在预测模式下只返回X
    if for_prediction:
        return np.array(X), None
    else:
        return np.array(X), np.array(Y)

# 创建LSTM模型 - PyTorch版本
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim3, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_dim3, hidden_dim4, batch_first=True)
        self.lstm5 = nn.LSTM(hidden_dim4, hidden_dim5, batch_first=True)  # 第五层LSTM
        self.fc = nn.Linear(hidden_dim5, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm1_out, _ = self.lstm1(x)
        # lstm1_out shape: (batch_size, seq_len, hidden_dim1)
        lstm2_out, _ = self.lstm2(lstm1_out)
        # lstm2_out shape: (batch_size, seq_len, hidden_dim2)
        lstm3_out, _ = self.lstm3(lstm2_out)
        # lstm3_out shape: (batch_size, seq_len, hidden_dim3)
        lstm4_out, _ = self.lstm4(lstm3_out)
        # lstm4_out shape: (batch_size, seq_len, hidden_dim4)
        lstm5_out, _ = self.lstm5(lstm4_out)  # 通过第五层LSTM
        # lstm5_out shape: (batch_size, seq_len, hidden_dim5)
        out = self.fc(lstm5_out[:, -1, :])
        # out shape: (batch_size, output_dim)
        return out

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
            y_true_segment = y_true[step:step+compare_length]
            y_pred_segment = y_pred[step:step+compare_length]
            
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
    
    print(f"训练集长度: {len(train)}")
    print(f"测试集长度: {len(test)}")
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    scaled_test = scaler.transform(test.reshape(-1, 1)).flatten()
    
    # 保存scaler
    dump(scaler, os.path.join(results_dir, 'scaler.joblib'))
    
    # 设置LSTM参数
    look_back = 70  # 时间步长
    epochs = 50
    batch_size = 16
    
    # 创建数据集
    X_train, y_train = create_dataset(scaled_train, look_back)
    X_test, y_test = create_dataset(scaled_test, look_back)
    
    # 转换为PyTorch张量并调整维度
    # 原来的X形状是 [samples, look_back]，需要改为 [samples, look_back, 1]
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2)  # 添加特征维度
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(2)  # 添加特征维度
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 分割训练集为训练和验证
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 在process_dataset函数中修改模型初始化部分
    input_dim = 1  # 输入特征维度
    hidden_dim1 = 200  # 第一个LSTM层的隐藏单元数
    hidden_dim2 = 160  # 第二个LSTM层的隐藏单元数
    hidden_dim3 = 130  # 第三个LSTM层的隐藏单元数
    hidden_dim4 = 100  # 第四个LSTM层的隐藏单元数
    hidden_dim5 = 70   # 第五个LSTM层的隐藏单元数 (新增)
    model = LSTMModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=0.00005)
    
    # 训练模型
    train_losses = []
    val_losses = []
    
    # 记录训练开始时间
    train_start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        # 添加进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_X, batch_y in train_pbar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).view(-1, 1)
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # 更新进度条显示当前损失
            train_pbar.set_postfix({'loss': loss.item()})
        
        # 验证
        model.eval()
        val_loss = 0
        # 添加验证集进度条
        val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]')
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).view(-1, 1)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                # 更新进度条显示当前损失
                val_pbar.set_postfix({'loss': loss.item()})
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # 计算训练时间（毫秒）
    train_time_ms = (time.time() - train_start_time) * 1000
    
    # 计算模型复杂度 (MACs和参数量)
    sample_input = torch.randn(1, look_back, 1).to(device)
    macs, params = profile(model, inputs=(sample_input,), verbose=False)
    
    # 格式化为更易读的形式
    macs_str, params_str = clever_format([macs, params], "%.3f")
    
    # 保存模型
    model_path = os.path.join(results_dir, f'lstm_model_cpu_{dataset_name}.pt')
    torch.save(model.state_dict(), model_path)
    
    # 保存训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {dataset_name}')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.close()
    
    # 记录预测开始时间
    predict_start_time = time.time()
    
    # 预测
    model.eval()
    with torch.no_grad():
        train_predict = model(X_train_tensor.to(device)).cpu().numpy()
        test_predict = model(X_test_tensor.to(device)).cpu().numpy()
    
    # 计算预测时间（毫秒）
    prediction_time_ms = (time.time() - predict_start_time) * 1000
    per_sample_time_ms = prediction_time_ms / (len(X_train_tensor) + len(X_test_tensor))
    
    # 反向转换预测值
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 计算各种评估指标
    train_mse = mean_squared_error(y_train_original, train_predict)
    train_rmse = math.sqrt(train_mse)
    train_mae = mean_absolute_error(train_predict, y_train_original)
    train_r2 = r2_score(y_train_original, train_predict)
    
    # 计算训练集MAPE (平均绝对百分比误差)
    epsilon = 1e-10  # 避免除以零
    train_mape = np.mean(np.abs((y_train_original - train_predict) / (y_train_original + epsilon))) * 100
    
    test_mse = mean_squared_error(y_test_original, test_predict)
    test_rmse = math.sqrt(test_mse)
    test_mae = mean_absolute_error(test_predict, y_test_original)
    test_r2 = r2_score(y_test_original, test_predict)
    
    # 计算测试集MAPE
    test_mape = np.mean(np.abs((y_test_original - test_predict) / (y_test_original + epsilon))) * 100
    
    # 计算不同预测步长的RMSE
    # 根据数据集名称动态设置max_steps
    max_steps = 90 if '30s' in dataset_name else 60 if '5m' in dataset_name else 30
        
    train_step_rmse = calculate_step_rmse(y_train_original, train_predict, max_steps)
    test_step_rmse = calculate_step_rmse(y_test_original, test_predict, max_steps)
    
    # 保存不同步长的RMSE
    np.save(os.path.join(results_dir, 'train_step_rmse.npy'), np.array(train_step_rmse))
    np.save(os.path.join(results_dir, 'test_step_rmse.npy'), np.array(test_step_rmse))
    
    # 同时保存到统一目录
    np.save(os.path.join(unified_results_dir, f'Lstm_{dataset_name}_test.npy'), test_predict)
    np.save(os.path.join(unified_results_dir, f'Lstm_test_ground_truth_{dataset_name}.npy'), y_test_original)
    np.save(os.path.join(unified_results_dir, f'Lstm_{dataset_name}_test_step_rmse.npy'), np.array(test_step_rmse))
    
    # 绘制不同预测步长的RMSE图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(test_step_rmse) + 1), test_step_rmse, marker='o', linestyle='-', linewidth=2, markersize=4)
    
    # 设置图表标题和标签
    if dataset_name == '30s':
        plt.xlabel('预测步长 (秒)')
        plt.title('LSTM 30秒级预测的RMSE随预测步长的变化')
    elif dataset_name == '5m':
        plt.xlabel('预测步长 (分钟)')
        plt.title('LSTM 5分钟级预测的RMSE随预测步长的变化')
    elif dataset_name == '1h':
        plt.xlabel('预测步长 (小时)')
        plt.title('LSTM 1小时级预测的RMSE随预测步长的变化')
    
    plt.ylabel('RMSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    rmse_plot_path = os.path.join(results_dir, 'step_rmse_plot.png')
    plt.savefig(rmse_plot_path)
    
    # 同时保存到统一目录
    unified_rmse_plot_path = os.path.join(unified_results_dir, f'Lstm_{dataset_name}_step_rmse_plot.png')
    plt.savefig(unified_rmse_plot_path)
    plt.close()
    
    # 计算Log RMSE (避免负值或零值)
    y_train_log = np.log1p(np.maximum(y_train_original, 0))
    train_pred_log = np.log1p(np.maximum(train_predict, 0))
    train_log_rmse = math.sqrt(mean_squared_error(y_train_log, train_pred_log))
    
    y_test_log = np.log1p(np.maximum(y_test_original, 0))
    test_pred_log = np.log1p(np.maximum(test_predict, 0))
    test_log_rmse = math.sqrt(mean_squared_error(y_test_log, test_pred_log))
    
    # 保存评估指标
    metrics = {
        'train': {
            'mse': float(train_mse),
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'r2': float(train_r2),
            'log_rmse': float(train_log_rmse),
            'mape': float(train_mape),  # 添加MAPE指标
            'step_rmse': [float(x) for x in train_step_rmse]
        },
        'test': {
            'mse': float(test_mse),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2': float(test_r2),
            'log_rmse': float(test_log_rmse),
            'mape': float(test_mape),  # 添加MAPE指标
            'step_rmse': [float(x) for x in test_step_rmse]
        },
        'time': {
            'train_time_ms': float(train_time_ms),
            'prediction_time_ms': float(prediction_time_ms),
            'per_sample_time_ms': float(per_sample_time_ms)
        },
        # 添加模型复杂度指标
        'complexity': {
            'macs': float(macs),
            'macs_readable': macs_str,
            'params': float(params),
            'params_readable': params_str
        }
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    
    # 保存预测结果
    np.save(os.path.join(results_dir, 'train_predictions.npy'), train_predict)
    np.save(os.path.join(results_dir, 'test_predictions.npy'), test_predict)
    np.save(os.path.join(results_dir, 'train_ground_truth.npy'), y_train_original)
    np.save(os.path.join(results_dir, 'test_ground_truth.npy'), y_test_original)
    
    # 绘制预测结果图
    plt.figure(figsize=(16, 8))
    
    # 绘制训练集预测与真实值对比
    plt.subplot(121)
    plt.plot(y_train_original, label='Ground Truth (Train)')
    plt.plot(train_predict, label='Predictions (Train)', linestyle='--')
    plt.legend()
    plt.title(f'Train Set Predictions vs Ground Truth - {dataset_name}')
    
    # 绘制测试集预测与真实值对比
    plt.subplot(122)
    plt.plot(y_test_original, label='Ground Truth (Test)')
    plt.plot(test_predict, label='Predictions (Test)', linestyle='--')
    plt.legend()
    plt.title(f'Test Set Predictions vs Ground Truth - {dataset_name}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'predictions_plot.png'))
    plt.close()
    
    # 打印结果摘要
    print(f"\n数据集 {dataset_name} 处理完成")
    print(f"训练集 - MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}, MAPE: {train_mape:.2f}%")
    print(f"测试集 - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}, MAPE: {test_mape:.2f}%")
    print(f"训练时间: {train_time_ms:.2f} 毫秒")
    print(f"预测时间: {prediction_time_ms:.2f} 毫秒 (每样本: {per_sample_time_ms:.2f} 毫秒)")
    
    return metrics

# 主程序
if __name__ == "__main__":
    # 确保已安装thop库
    try:
        from thop import profile, clever_format
    except ImportError:
        print("正在安装thop库...")
        os.system('pip install thop')
        from thop import profile, clever_format
        
    # 处理所有数据集
    all_datasets_results = {}
    all_complexity_info = {}

    for dataset in datasets:
        print(f"\n开始处理数据集: {dataset['name']}")
        results = process_dataset(dataset)
        all_datasets_results[dataset['name']] = results
        
        # 从结果中提取复杂度信息
        try:
            with open(os.path.join(dataset['results_dir'], 'model_complexity_info.json'), 'r', encoding='utf-8') as f:
                complexity_info = json.load(f)
                all_complexity_info[dataset['name']] = complexity_info
        except:
            print(f"无法加载 {dataset['name']} 的复杂度信息")

    # 保存所有数据集的汇总结果
    with open('h:\\work\\SG-LSTM_all_datasets_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_datasets_results, f, indent=4)

    with open(os.path.join(unified_results_dir, 'lstm_all_datasets_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_datasets_results, f, indent=4)
        
    # 保存所有数据集的复杂度信息汇总
    with open('h:\\work\\SG-LSTM_all_complexity_info.json', 'w', encoding='utf-8') as f:
        json.dump(all_complexity_info, f, indent=4)
        
    with open(os.path.join(unified_results_dir, 'lstm_all_complexity_info.json'), 'w', encoding='utf-8') as f:
        json.dump(all_complexity_info, f, indent=4)

    print("\n所有数据集处理完成！")
    print(f"结果已保存到各自的目录中，汇总结果保存在: {unified_results_dir}")
    print(f"模型复杂度信息已保存到: {unified_results_dir}/lstm_all_complexity_info.json")