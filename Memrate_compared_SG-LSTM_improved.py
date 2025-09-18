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
from tqdm import tqdm
import os
import time
import json
import math
from joblib import dump, load
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
        'file': 'Alibaba_mem_util_aggregated_30s.csv',
        'results_dir': 'h:\\work\\alibaba_mem_lstm_results_30s\\'
    },
    {
        'name': 'Google_5m',
        'file': 'Google_mem_util_aggregated_5m.csv',
        'results_dir': 'h:\\work\\google_mem_lstm_results_5m\\'
    }
]

# 创建统一的结果目录
unified_results_dir = 'h:\\work\\Mem_lstm_unified_results\\'
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

# 设置LSTM参数
look_back = 70
epochs = 50
batch_size = 16

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

# 训练模型函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, results_dir):
    model.to(device)
    train_losses = []
    val_losses = []
    
    # 记录训练时间
    train_start_time = time.time()
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
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
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 验证
        model.eval()
        val_loss = 0
        # 添加验证集进度条
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]')
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).view(-1, 1)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                # 更新进度条显示当前损失
                val_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 记录每个epoch的时间（毫秒）
        epoch_end_time = time.time()
        epoch_time_ms = (epoch_end_time - epoch_start_time) * 1000
        epoch_times.append(epoch_time_ms)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time_ms:.2f} ms')
    
    # 计算总训练时间（毫秒）
    total_train_time_ms = (time.time() - train_start_time) * 1000
    avg_epoch_time_ms = sum(epoch_times) / len(epoch_times)
    
    # 保存训练时间信息
    time_info = {
        'total_train_time_ms': total_train_time_ms,
        'avg_epoch_time_ms': avg_epoch_time_ms,
        'epoch_times_ms': epoch_times
    }
    
    with open(os.path.join(results_dir, 'train_time.json'), 'w') as f:
        json.dump(time_info, f, indent=4)
    
    return model, time_info, train_losses, val_losses

# 评估模型性能并记录预测时间
def evaluate_model(model, X_test, y_test, scaler, results_dir):
    model.eval()
    
    # 转换为PyTorch张量
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # 记录预测时间
    start_time = time.time()
    
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
    
    # 计算预测时间（毫秒）
    prediction_time_ms = (time.time() - start_time) * 1000
    per_sample_time_ms = prediction_time_ms / len(X_test)
    
    # 反归一化预测结果
    y_pred = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 计算各种评估指标
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    
    # 计算MAPE (平均绝对百分比误差)
    epsilon = 1e-10  # 避免除以零
    mape = np.mean(np.abs((y_test_original - y_pred) / (y_test_original + epsilon))) * 100
    
    # 计算不同预测步长的RMSE
    dataset_name = os.path.basename(os.path.dirname(results_dir))
    max_steps = 30  # 默认最大预测步长
    if '30s' in dataset_name:
        max_steps = 90  # 30秒级别，最大90步
    elif '5m' in dataset_name:
        max_steps = 60  # 5分钟级别，最大60步
    elif '1h' in dataset_name:
        max_steps = 30  # 1小时级别，最大30步
        
    step_rmse = calculate_step_rmse(y_test_original, y_pred, max_steps)
    
    # 保存不同步长的RMSE
    np.save(os.path.join(results_dir, 'test_step_rmse.npy'), np.array(step_rmse))
    
    # 同时保存到统一目录
    dataset_name = dataset_name.split('_')[-1]  # 提取30s/5m/1h
    np.save(os.path.join(unified_results_dir, f'Lstm_{dataset_name}_test.npy'), y_pred)
    np.save(os.path.join(unified_results_dir, f'Lstm_test_ground_truth_{dataset_name}.npy'), y_test_original)
    np.save(os.path.join(unified_results_dir, f'Lstm_{dataset_name}_test_step_rmse.npy'), np.array(step_rmse))
    
    # 计算Log RMSE (避免负值或零值)
    y_test_log = np.log1p(np.maximum(y_test_original, 0))
    y_pred_log = np.log1p(np.maximum(y_pred, 0))
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
        'step_rmse': [float(x) for x in step_rmse]
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 保存预测结果
    np.save(os.path.join(results_dir, 'predictions.npy'), y_pred)
    np.save(os.path.join(results_dir, 'ground_truth.npy'), y_test_original)
    
    return metrics, y_pred, y_test_original

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
    
    # 确保数据长度为2的幂次方，与SWD-CLSTM保持一致
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
    scaled_train = scaler.fit_transform(train.reshape(-1, 1))
    scaled_test = scaler.transform(test.reshape(-1, 1))
    
    # 保存归一化器，用于后续反归一化
    dump(scaler, os.path.join(results_dir, 'scaler.joblib'))
    
    # 创建时间序列数据集
    X_train, y_train = create_dataset(scaled_train, look_back)
    X_test, y_test = create_dataset(scaled_test, look_back)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
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
    
    # 初始化模型
    # 初始化模型
    input_dim = 1  # 输入特征维度
    hidden_dim1 = 200  # 第一个LSTM层的隐藏单元数
    hidden_dim2 = 160  # 第二个LSTM层的隐藏单元数
    hidden_dim3 = 130  # 第三个LSTM层的隐藏单元数
    hidden_dim4 = 100  # 第四个LSTM层的隐藏单元数
    hidden_dim5 = 70   # 第五个LSTM层的隐藏单元数
    model = LSTMModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, output_dim=1).to(device)
    
    # 计算模型复杂度
    sample_input = torch.randn(1, look_back, 1).to(device)  # 注意LSTM需要3维输入
    macs, params = calculate_model_complexity(model, sample_input)
    print(f"模型复杂度 - MACs: {macs}, Parameters: {params}")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=0.00005)
    
    # 训练模型
    model, train_time_info, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        epochs, device, results_dir
    )
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(results_dir, 'lstm_model.pt'))
    
    # 评估模型
    metrics, y_pred, y_test_original = evaluate_model(model, X_test, y_test, scaler, results_dir)
    
    # 合并训练时间和评估指标，并添加模型复杂度信息
    results = {
        **train_time_info, 
        **metrics,
        'model_complexity': {
            'macs': macs,
            'parameters': params
        }
    }
    
    # 保存所有结果的汇总
    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def calculate_model_complexity(model, input_tensor):
    """计算模型的MACs和Parameters"""
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor,))
        macs, params = clever_format([macs, params], "%.3f")
    return macs, params

# 主程序
if __name__ == "__main__":
    # 处理所有数据集
    all_datasets_results = {}
    
    for dataset in datasets:
        print(f"\n开始处理数据集: {dataset['name']}")
        results = process_dataset(dataset)
        all_datasets_results[dataset['name']] = results
    
    # 保存所有数据集的汇总结果
    with open('h:\\work\\lstm_all_datasets_results.json', 'w') as f:
        json.dump(all_datasets_results, f, indent=4)
    
    # 同时保存到统一目录
    with open(os.path.join(unified_results_dir, 'lstm_all_datasets_results.json'), 'w') as f:
        json.dump(all_datasets_results, f, indent=4)
    
    print("\n所有数据集处理完成！")
    print(f"结果已保存到各自的目录中，汇总结果保存在: {unified_results_dir}")

