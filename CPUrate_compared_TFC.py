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
import torch.nn.functional as F
from thop import profile, clever_format

# 忽略所有警告
warnings.filterwarnings("ignore")

# 设置matplotlib参数
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.max_open_warning'] = 50
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
        'results_dir': 'h:\\work\\alibaba_cpu_tfc_results_30s\\'
    },
    # {
    #     'name': 'Google_5m',
    #     'file': 'Google_cpu_util_aggregated_5m.csv',
    #     'results_dir': 'h:\\work\\google_cpu_tfc_results_5m\\'
    # }
]

# 创建统一的结果目录
unified_results_dir = 'h:\\work\\CPU_tfc_unified_results\\'
os.makedirs(unified_results_dir, exist_ok=True)

# 创建结果目录
for dataset in datasets:
    os.makedirs(dataset['results_dir'], exist_ok=True)

def create_dataset(data, look_back=70):
    """创建时间序列数据集"""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def calculate_step_rmse(y_true, y_pred, max_steps=30):
    """计算不同预测步长的RMSE"""
    step_rmse = []
    
    for step in range(1, max_steps + 1):
        if step < len(y_true) and step < len(y_pred):
            compare_length = min(len(y_true) - step, len(y_pred) - step)
            if compare_length <= 0:
                break
                
            y_true_segment = y_true[step:step+compare_length]
            y_pred_segment = y_pred[step:step+compare_length]
            
            # 将数据转换为二维数组，如果它们是一维的
            if y_true_segment.ndim == 1:
                y_true_segment = y_true_segment.reshape(-1, 1)
            if y_pred_segment.ndim == 1:
                y_pred_segment = y_pred_segment.reshape(-1, 1)
            
            rmse = np.sqrt(mean_squared_error(y_true_segment, y_pred_segment))
            step_rmse.append(rmse)
        else:
            break
    
    return step_rmse

class TimeEncoder(nn.Module):
    """时域编码器"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TimeEncoder, self).__init__()
        # 修改为3层Conv1d
        # self.conv1 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=9, padding=4, dilation=1)
        # self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=7, padding=3, dilation=1)
        # self.conv3 = nn.Conv1d(hidden_dim // 2, output_dim, kernel_size=5, padding=2, dilation=1)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=9, padding=4, dilation=1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=9, padding=4, dilation=1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, output_dim, kernel_size=9, padding=4, dilation=1)

        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 残差连接
        self.residual_conv = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.mid_residual_conv1 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        x = x.unsqueeze(1)
        
        # 保存输入用于残差连接
        residual = self.residual_conv(x)
        
        # 第一层Conv1d
        x1 = self.swish(self.bn1(self.conv1(x)))
        x1 = self.dropout(x1)
        
        # 第二层Conv1d
        x2 = self.gelu(self.bn2(self.conv2(x1)))
        x2 = x2 + self.mid_residual_conv1(x1)  # 残差连接
        x2 = self.dropout(x2)
        
        # 第三层Conv1d
        x3 = self.relu(self.bn3(self.conv3(x2)))
        
        # 添加主残差连接
        x3 = x3 + residual
        x3 = self.dropout(x3)
        
        # Global average pooling
        x3 = self.pool(x3)
        x3 = x3.squeeze(-1)
        
        return x3

class FrequencyEncoder(nn.Module):
    """频域编码器"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FrequencyEncoder, self).__init__()
        # 修改为3层Conv1d
        # self.conv1 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=9, padding=4, dilation=1)
        # self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=7, padding=3, dilation=1)
        # self.conv3 = nn.Conv1d(hidden_dim // 2, output_dim, kernel_size=5, padding=2, dilation=1)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=9, padding=4, dilation=1)
        self.conv2 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=9, padding=4, dilation=1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, output_dim, kernel_size=9, padding=4, dilation=1)

        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 残差连接
        self.residual_conv = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.mid_residual_conv1 = nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=1)
        
    def forward(self, x):
        # Convert to frequency domain using FFT
        x_fft = torch.fft.fft(x, dim=-1)
        x_mag = torch.abs(x_fft)
        
        # x_mag shape: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        x_mag = x_mag.unsqueeze(1)
        
        # 保存输入用于残差连接
        residual = self.residual_conv(x_mag)
        
        # 第一层Conv1d
        x1 = self.swish(self.bn1(self.conv1(x_mag)))
        x1 = self.dropout(x1)
        
        # 第二层Conv1d
        x2 = self.gelu(self.bn2(self.conv2(x1)))
        x2 = x2 + self.mid_residual_conv1(x1)  # 残差连接
        x2 = self.dropout(x2)
        
        # 第三层Conv1d
        x3 = self.relu(self.bn3(self.conv3(x2)))
        
        # 添加主残差连接
        x3 = x3 + residual
        x3 = self.dropout(x3)
        
        # Global average pooling
        x3 = self.pool(x3)
        x3 = x3.squeeze(-1)
        
        return x3

class ProjectionHead(nn.Module):
    """投影头"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)  # 增加dropout
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
    def forward(self, x):
        x = self.gelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class TFCModel(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=128, projection_dim=32, output_dim=1):  # 增加hidden_dim到384，projection_dim到64
        super(TFCModel, self).__init__()
        
        # 时域和频域编码器
        self.time_encoder = TimeEncoder(1, hidden_dim, hidden_dim)
        self.freq_encoder = FrequencyEncoder(1, hidden_dim, hidden_dim)
        
        # 投影头
        self.time_projection = ProjectionHead(hidden_dim, hidden_dim, projection_dim)
        self.freq_projection = ProjectionHead(hidden_dim, hidden_dim, projection_dim)
        
        # 简化的预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),  # 从0.4降低到0.25
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),   # 从0.3降低到0.2
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),   # 从0.2降低到0.1
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # 残差连接
        self.residual_predictor = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x, return_projections=False):
        # 时域和频域编码
        time_features = self.time_encoder(x)
        freq_features = self.freq_encoder(x)
        
        if return_projections:
            # 返回投影用于对比学习
            time_proj = self.time_projection(time_features)
            freq_proj = self.freq_projection(freq_features)
            return time_proj, freq_proj
        
        # 特征融合
        combined_features = torch.cat([time_features, freq_features], dim=1)
        
        # 主预测路径
        output = self.predictor(combined_features)
        
        # 残差连接
        residual = self.residual_predictor(combined_features)
        output = output + residual
        
        return output

def contrastive_loss(time_proj, freq_proj, temperature=0.05):  # 从0.2降低到0.1，增强对比学习效果
    """计算对比损失"""
    batch_size = time_proj.shape[0]
    
    # 归一化
    time_proj = F.normalize(time_proj, dim=1)
    freq_proj = F.normalize(freq_proj, dim=1)
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(time_proj, freq_proj.T) / temperature
    
    # 正样本标签（对角线为1）
    labels = torch.arange(batch_size).to(time_proj.device)
    
    # 计算交叉熵损失
    loss_t2f = F.cross_entropy(similarity_matrix, labels)
    loss_f2t = F.cross_entropy(similarity_matrix.T, labels)
    
    return (loss_t2f + loss_f2t) / 2

def calculate_model_complexity(model, input_tensor):
    """计算模型的MACs和Parameters"""
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor,))
        macs, params = clever_format([macs, params], "%.3f")
    return macs, params

# 在process_dataset函数中修改训练参数
def process_dataset(dataset_info):
    """处理单个数据集"""
    dataset_name = dataset_info['name']
    file_path = dataset_info['file']
    results_dir = dataset_info['results_dir']
    
    look_back = 70
    
    print(f"\n{'='*50}")
    print(f"处理数据集: {dataset_name} (文件: {file_path})")
    print(f"{'='*50}\n")
    
    # 加载数据
    try:
        data = np.loadtxt(file_path, delimiter=' ')
        data = data[data != 0]
    except:
        print(f"无法加载文件，使用随机数据进行测试")
        np.random.seed(42)
        data = np.random.rand(1000) * 100
    
    print(f"原始数据长度: {len(data)}")
    
    # 应用Savitzky-Golay滤波
    window_length = min(11, len(data) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    
    if window_length < 3:
        smoothed_data = data.copy()
        print("数据长度过短，跳过滤波")
    else:
        smoothed_data = savgol_filter(data, window_length=window_length, polyorder=min(2, window_length-1))
    print(f"滤波后数据长度: {len(smoothed_data)}")
    
    # 确保数据长度为2的幂次方
    power = int(np.ceil(np.log2(len(smoothed_data))))
    padded_length = 2**power
    
    if len(smoothed_data) != padded_length:
        pad_width = padded_length - len(smoothed_data)
        smoothed_data = np.pad(smoothed_data, (0, pad_width), mode='symmetric')
        print(f"数据长度已填充至 {padded_length} (2^{power})")
    else:
        print(f"数据长度已经是2的幂次方: {padded_length}")
    
    # 分割数据
    train_size = int(len(smoothed_data) * 0.8)
    train, test = smoothed_data[:train_size], smoothed_data[train_size:]
    
    print(f"训练集长度: {len(train)}")
    print(f"测试集长度: {len(test)}")
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train.reshape(-1, 1))
    scaled_test = scaler.transform(test.reshape(-1, 1))
    
    # 创建时间序列数据集
    X_train, y_train = create_dataset(scaled_train, look_back)
    X_test, y_test = create_dataset(scaled_test, look_back)
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 创建模型
    # 创建TFC模型
    model = TFCModel(hidden_dim=384).to(device)  # 增加hidden_dim到384
    
    # 计算模型复杂度
    sample_input = torch.randn(1, look_back).to(device)
    macs, params = calculate_model_complexity(model, sample_input)
    print(f"模型复杂度 - MACs: {macs}, Parameters: {params}")
    
    # 训练模型
    print("开始训练...")
    
    # 训练参数
    num_epochs = 50  # 增加训练轮数
    best_loss = float('inf')
    patience = 15    # 增加patience
    patience_counter = 0
    warmup_epochs = 5  # 增加warmup epochs
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.003, betas=(0.9, 0.999))  # 提高学习率，适度降低权重衰减
    
    # 调度器
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00005)  # 调整调度器参数
    
    # 记录列表
    train_losses = []
    val_losses = []
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_contrastive_loss = 0.0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(batch_X)
            
            # 预测损失
            pred_loss = criterion(predictions.squeeze(), batch_y)
            
            # 对比学习损失
            time_proj, freq_proj = model(batch_X, return_projections=True)
            contrast_loss = contrastive_loss(time_proj, freq_proj)
            
            # 动态对比学习权重
            contrast_weight = max(0.005, 0.05 * (1 - epoch / num_epochs))  # 增加对比学习权重
            total_loss = pred_loss + contrast_weight * contrast_loss
            
            # 减少L1正则化强度
            l1_lambda = 0.0001  # 从0.001降低到0.0001
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            total_loss = total_loss + l1_lambda * l1_norm
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += pred_loss.item()
            epoch_contrastive_loss += contrast_loss.item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y)
                val_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        avg_contrast_loss = epoch_contrastive_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {avg_train_loss:.6f}')
        print(f'  验证损失: {avg_val_loss:.6f}')
        print(f'  对比损失: {avg_contrast_loss:.6f}')
        print(f'  对比权重: {contrast_weight:.6f}')
        print(f'  学习率: {optimizer.param_groups[0]["lr"]:.8f}')
        
        # 学习率调度
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # 早停检查
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'早停触发，在第 {epoch+1} 轮停止训练')
            break
    
    # 记录训练时间
    training_time = time.time() - training_start_time
    print(f'训练完成，总时间: {training_time:.2f} 秒')
    
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
    model.eval()
    
    # 预测
    print("开始预测...")
    prediction_start_time = time.time()
    
    with torch.no_grad():
        # 训练集预测
        train_predictions = []
        train_targets = []
        for batch_X, batch_y in train_loader:
            predictions = model(batch_X)
            train_predictions.extend(predictions.cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())
        
        # 测试集预测
        test_predictions = []
        test_targets = []
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
    
    prediction_time = time.time() - prediction_start_time
    print(f'预测完成，时间: {prediction_time:.2f} 秒')
    
    # 转换为numpy数组
    train_predictions = np.array(train_predictions).reshape(-1, 1)
    train_targets = np.array(train_targets).reshape(-1, 1)
    test_predictions = np.array(test_predictions).reshape(-1, 1)
    test_targets = np.array(test_targets).reshape(-1, 1)
    
    # 反归一化
    train_predictions_rescaled = scaler.inverse_transform(train_predictions)
    train_targets_rescaled = scaler.inverse_transform(train_targets)
    test_predictions_rescaled = scaler.inverse_transform(test_predictions)
    test_targets_rescaled = scaler.inverse_transform(test_targets)
    
    # 计算指标的函数
    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算MAPE
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        # 计算Log RMSE
        y_true_log = np.log1p(np.maximum(y_true, 0))
        y_pred_log = np.log1p(np.maximum(y_pred, 0))
        log_rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'log_rmse': float(log_rmse)
        }
    
    # 训练集指标
    train_metrics = calculate_metrics(train_targets_rescaled, train_predictions_rescaled)
    print(f"训练集 - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}")
    
    # 测试集指标
    test_metrics = calculate_metrics(test_targets_rescaled, test_predictions_rescaled)
    test_metrics['prediction_time_ms'] = float(prediction_time * 1000)
    test_metrics['per_sample_time_ms'] = float(prediction_time * 1000 / len(test_predictions))
    
    print(f"测试集 - RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2']:.4f}")
    
    # 计算步长RMSE
    max_steps = 90 if '30s' in dataset_name else 60 if '5m' in dataset_name else 30
    train_step_rmse = calculate_step_rmse(train_targets_rescaled, train_predictions_rescaled, max_steps)
    test_step_rmse = calculate_step_rmse(test_targets_rescaled, test_predictions_rescaled, max_steps)
    
    # 添加步长RMSE到测试指标中
    test_metrics['step_rmse'] = [float(x) for x in test_step_rmse]
    
    # 保存结果
    results = {
        'dataset': dataset_info['name'],
        'model_complexity': {
            'MACs': macs,
            'Parameters': params
        },
        'timing': {
            'training_time': float(training_time),
            'prediction_time': float(prediction_time)
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_step_rmse': train_step_rmse,
        'test_step_rmse': test_step_rmse
    }
    
    # 保存到文件
    np.save(os.path.join(dataset_info['results_dir'], 'train_predictions.npy'), train_predictions_rescaled)
    np.save(os.path.join(dataset_info['results_dir'], 'train_ground_truth.npy'), train_targets_rescaled)
    np.save(os.path.join(dataset_info['results_dir'], 'test_predictions.npy'), test_predictions_rescaled)
    np.save(os.path.join(dataset_info['results_dir'], 'test_ground_truth.npy'), test_targets_rescaled)
    np.save(os.path.join(dataset_info['results_dir'], 'train_step_rmse.npy'), train_step_rmse)
    np.save(os.path.join(dataset_info['results_dir'], 'test_step_rmse.npy'), test_step_rmse)
    
    # 保存模型复杂度和时间信息
    complexity_timing = {
        'MACs': macs,
        'Parameters': params,
        'training_time': float(training_time),
        'prediction_time': float(prediction_time)
    }
    
    with open(os.path.join(dataset_info['results_dir'], 'complexity_timing.json'), 'w') as f:
        json.dump(complexity_timing, f, indent=4)
    
    # 保存scaler和指标
    dump(scaler, os.path.join(dataset_info['results_dir'], 'scaler.joblib'))
    
    with open(os.path.join(dataset_info['results_dir'], 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 保存到统一结果目录
    np.save(os.path.join(unified_results_dir, f'Tfc_{dataset_info["name"]}_test.npy'), test_predictions_rescaled)
    np.save(os.path.join(unified_results_dir, f'Tfc_{dataset_info["name"]}_test_step_rmse.npy'), test_step_rmse)
    
    return results

# 主程序
if __name__ == "__main__":
    all_results = []
    
    for dataset_info in datasets:
        try:
            result = process_dataset(dataset_info)
            all_results.append(result)
        except Exception as e:
            print(f"处理数据集 {dataset_info['name']} 时出错: {str(e)}")
            continue
    
    # 保存所有结果
    with open(os.path.join(unified_results_dir, 'all_metrics.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\n所有数据集处理完成！")
    print(f"结果保存在: {unified_results_dir}")
    
    # 打印汇总结果
    print("\n=== 汇总结果 ===")
    for result in all_results:
        print(f"\n数据集: {result['dataset']}")
        print(f"测试集 RMSE: {result['test_metrics']['rmse']:.4f}")
        print(f"测试集 MAE: {result['test_metrics']['mae']:.4f}")
        print(f"测试集 R2: {result['test_metrics']['r2']:.4f}")
        print(f"测试集 MAPE: {result['test_metrics']['mape']:.4f}%")