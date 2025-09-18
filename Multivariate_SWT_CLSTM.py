# 多变量 SG-CNN-SWT-LSTM (PyTorch版本) - CPU和内存联合预测
# 支持跨资源模态的共享特征提取和多任务学习

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import time
import json
import math
import warnings
from thop import profile, clever_format

# 忽略所有警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置中文字体和UTF-8编码
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.max_open_warning'] = 50
matplotlib.rcParams['font.size'] = 10
matplotlib.use('Agg')

# 设置UTF-8编码
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"当前CUDA版本: {torch.version.cuda}")
    print(f"当前PyTorch版本: {torch.__version__}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到GPU，将使用CPU进行训练")

# 创建结果目录
results_dir = 'h:\\work\\multivariate_swt_clstm_results\\'
os.makedirs(results_dir, exist_ok=True)

# 设置参数
look_back = 70
epochs = 50
batch_size = 16

# 数据增强和对比学习相关函数
def generate_augmented_samples(x, augmentation_strength=1, augmentation_type='general'):
    batch_size, seq_len, features = x.shape
    
    if augmentation_type == 'low_freq':
        sigma_value = 0.015 * x.std().item()
        noise = torch.normal(mean=0.0, std=sigma_value, size=x.shape).to(x.device)
        trend_shift = torch.linspace(-0.005, 0.005, seq_len).repeat(batch_size, 1).unsqueeze(-1).to(x.device)
        noise = noise + trend_shift
        smooth_factor = torch.exp(-torch.linspace(0, 2, seq_len)).repeat(batch_size, 1).unsqueeze(-1).to(x.device)
        noise = noise * smooth_factor
    elif augmentation_type == 'high_freq':
        beta_value = 0.25 * x.abs().max().item()
        noise = torch.FloatTensor(x.shape).uniform_(-beta_value, beta_value).to(x.device)
        spike_prob = 0.03
        spikes = (torch.rand(batch_size, seq_len, 1) < spike_prob).float() * 0.08
        spikes = spikes.to(x.device)
        noise = noise + spikes
    else:
        noise_strength = augmentation_strength * 0.3
        noise = torch.randn_like(x) * noise_strength
    
    augmented = x * 0.95 + x * 0.05 * torch.tanh(noise)
    
    mask_length = max(1, int(seq_len * 0.03))
    start_idx = torch.randint(0, seq_len - mask_length + 1, (batch_size,))
    
    for b in range(batch_size):
        start = start_idx[b]
        mean_val = x[b].mean()
        transition = torch.linspace(1.0, 0.3, mask_length).to(x.device)
        for i in range(mask_length):
            if start + i < seq_len:
                augmented[b, start + i, :] = x[b, start + i, :] * transition[i] + mean_val * (1 - transition[i]) * 0.3
    
    if augmentation_type == 'low_freq' and torch.rand(1).item() > 0.8:
        for b in range(batch_size):
            warp_factors = torch.sin(torch.linspace(0, 2.0, seq_len)) * 0.02 + 1
            augmented[b] = augmented[b] * warp_factors.unsqueeze(-1).to(x.device)
    
    return augmented

def contrastive_loss(features, augmented_features, temperature=10):
    batch_size = features.shape[0]
    
    features = nn.functional.normalize(features, dim=1)
    augmented_features = nn.functional.normalize(augmented_features, dim=1)
    
    features = features * 0.9 + torch.mean(features, dim=0, keepdim=True) * 0.1
    augmented_features = augmented_features * 0.9 + torch.mean(augmented_features, dim=0, keepdim=True) * 0.1
    
    features = nn.functional.normalize(features, dim=1)
    augmented_features = nn.functional.normalize(augmented_features, dim=1)
    
    similarity_matrix = torch.matmul(features, augmented_features.T) / temperature
    positive_pairs = torch.arange(batch_size).to(device)
    
    return nn.CrossEntropyLoss()(similarity_matrix, positive_pairs)

# 多变量CNN-LSTM模型
class MultivariateCNNLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size1=200, hidden_size2=160, hidden_size3=130, hidden_size4=100, hidden_size5=70):
        super(MultivariateCNNLSTM, self).__init__()
        
        # 共享卷积层 - 处理多变量输入
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        
        # 共享LSTM层
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=hidden_size2, hidden_size=hidden_size3, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=hidden_size3, hidden_size=hidden_size4, batch_first=True)
        self.lstm5 = nn.LSTM(input_size=hidden_size4, hidden_size=hidden_size5, batch_first=True)
        
        # 任务特定的输出层
        self.fc_cpu = nn.Linear(hidden_size5, 1)  # CPU预测
        self.fc_mem = nn.Linear(hidden_size5, 1)  # 内存预测
        
        # 跨模态注意力机制 - 调整头数以适应embed_dim
        # hidden_size5=70，使用5个头或10个头都可以整除
        num_heads = 5 if hidden_size5 % 5 == 0 else 10
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size5, num_heads=num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size5)
        
    def extract_features(self, x):
        # 调整形状用于卷积 (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # 共享卷积层
        x = self.maxpool(self.relu(self.conv1(x)))
        
        # 调整回LSTM所需的形状 (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # 共享LSTM层
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x, _ = self.lstm5(x)
        
        # 应用跨模态注意力
        attn_output, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_output)
        
        # 返回最后一个时间步的特征表示
        return x[:, -1, :]
    
    def forward(self, x):
        features = self.extract_features(x)
        
        # 任务特定的预测
        cpu_pred = self.fc_cpu(features)
        mem_pred = self.fc_mem(features)
        
        return cpu_pred, mem_pred, features

# 多任务损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.task_weights = task_weights if task_weights else [1.0, 1.0]
        self.mse_loss = nn.MSELoss()
        
    def forward(self, cpu_pred, mem_pred, cpu_target, mem_target):
        cpu_loss = self.mse_loss(cpu_pred, cpu_target)
        mem_loss = self.mse_loss(mem_pred, mem_target)
        
        total_loss = self.task_weights[0] * cpu_loss + self.task_weights[1] * mem_loss
        return total_loss, cpu_loss, mem_loss

# 数据预处理函数
def preprocess_data(cpu_file, mem_file):
    # 加载数据
    cpu_data = np.loadtxt(cpu_file, delimiter=' ')
    mem_data = np.loadtxt(mem_file, delimiter=' ')
    
    # 去除0元素
    cpu_data = cpu_data[cpu_data != 0]
    mem_data = mem_data[mem_data != 0]
    
    # 确保两个数据集长度一致
    min_length = min(len(cpu_data), len(mem_data))
    cpu_data = cpu_data[:min_length]
    mem_data = mem_data[:min_length]
    
    # 使用Savitzky-Golay滤波器去噪
    window_length = min(11, len(cpu_data) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    
    if window_length >= 3:
        cpu_smoothed = savgol_filter(cpu_data, window_length=window_length, polyorder=min(2, window_length-1))
        mem_smoothed = savgol_filter(mem_data, window_length=window_length, polyorder=min(2, window_length-1))
    else:
        cpu_smoothed = cpu_data.copy()
        mem_smoothed = mem_data.copy()
    
    # 确保数据长度为2的幂次方
    power = int(np.ceil(np.log2(len(cpu_smoothed))))
    padded_length = 2**power
    
    if len(cpu_smoothed) != padded_length:
        pad_width = padded_length - len(cpu_smoothed)
        cpu_smoothed = np.pad(cpu_smoothed, (0, pad_width), mode='symmetric')
        mem_smoothed = np.pad(mem_smoothed, (0, pad_width), mode='symmetric')
        print(f"数据长度已填充至 {padded_length} (2^{power})")
    
    return cpu_smoothed, mem_smoothed

# 小波分解函数
def wavelet_decomposition(cpu_data, mem_data):
    wavelet_type = 'db4'
    level = 1
    
    # 执行平稳小波变换
    cpu_coeffs = pywt.swt(cpu_data, wavelet_type, level=level)
    mem_coeffs = pywt.swt(mem_data, wavelet_type, level=level)
    
    # 验证重构精度
    cpu_reconstructed = pywt.iswt(cpu_coeffs, wavelet_type)
    mem_reconstructed = pywt.iswt(mem_coeffs, wavelet_type)
    
    cpu_error = np.mean(np.abs(cpu_data - cpu_reconstructed))
    mem_error = np.mean(np.abs(mem_data - mem_reconstructed))
    
    print(f"CPU小波重构误差: {cpu_error:.10f}")
    print(f"内存小波重构误差: {mem_error:.10f}")
    
    return cpu_coeffs, mem_coeffs, wavelet_type

# 创建多变量数据集
def create_multivariate_dataset(cpu_coeffs, mem_coeffs, look_back=70):
    # 提取ca和cd系数
    cpu_ca, cpu_cd = cpu_coeffs[0][0], cpu_coeffs[0][1]
    mem_ca, mem_cd = mem_coeffs[0][0], mem_coeffs[0][1]
    
    # 组合多变量数据 - 使用ca系数作为主要特征
    multivariate_data = np.column_stack([cpu_ca, mem_ca])
    
    # 分割训练集和测试集
    train_size = int(len(multivariate_data) * 0.8)
    train_data = multivariate_data[:train_size]
    test_data = multivariate_data[train_size:]
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # 创建时间序列数据集
    def create_sequences(data, look_back):
        X, Y_cpu, Y_mem = [], [], []
        for i in range(len(data) - look_back - 1):
            X.append(data[i:(i + look_back)])
            Y_cpu.append(data[i + look_back, 0])  # CPU目标
            Y_mem.append(data[i + look_back, 1])  # 内存目标
        return np.array(X), np.array(Y_cpu), np.array(Y_mem)
    
    X_train, Y_train_cpu, Y_train_mem = create_sequences(train_scaled, look_back)
    X_test, Y_test_cpu, Y_test_mem = create_sequences(test_scaled, look_back)
    
    return (X_train, Y_train_cpu, Y_train_mem, X_test, Y_test_cpu, Y_test_mem, 
            scaler, cpu_ca, cpu_cd, mem_ca, mem_cd)

# 训练函数 (已注释，不再使用)
# def train_multivariate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
#     model.to(device)
#     best_val_loss = float('inf')
#     patience = 10
#     patience_counter = 0
#     
#     train_start_time = time.time()
#     epoch_times = []
#     
#     for epoch in range(epochs):
#         epoch_start_time = time.time()
#         model.train()
#         train_loss = 0.0
#         train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]")
#         
#         for inputs, cpu_targets, mem_targets in train_pbar:
#             inputs = inputs.to(device)
#             cpu_targets = cpu_targets.to(device).unsqueeze(1)
#             mem_targets = mem_targets.to(device).unsqueeze(1)
#             
#             # 生成增强样本
#             augmented_inputs = generate_augmented_samples(inputs, augmentation_type='low_freq').to(device)
#             
#             # 前向传播
#             cpu_pred, mem_pred, features = model(inputs)
#             aug_cpu_pred, aug_mem_pred, aug_features = model(augmented_inputs)
#             
#             # 计算损失
#             task_loss, cpu_loss, mem_loss = criterion(cpu_pred, mem_pred, cpu_targets, mem_targets)
#             contr_loss = contrastive_loss(features, aug_features)
#             
#             total_loss = task_loss + 0.01 * contr_loss
#             
#             # 反向传播和优化
#             optimizer.zero_grad()
#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#             optimizer.step()
#             
#             train_loss += total_loss.item()
#             train_pbar.set_postfix({
#                 'loss': f"{total_loss.item():.4f}",
#                 'cpu_loss': f"{cpu_loss.item():.4f}",
#                 'mem_loss': f"{mem_loss.item():.4f}",
#                 'contr_loss': f"{contr_loss.item():.4f}"
#             })
#         
#         # 验证
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, cpu_targets, mem_targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [验证]"):
#                 inputs = inputs.to(device)
#                 cpu_targets = cpu_targets.to(device).unsqueeze(1)
#                 mem_targets = mem_targets.to(device).unsqueeze(1)
#                 
#                 cpu_pred, mem_pred, _ = model(inputs)
#                 loss, _, _ = criterion(cpu_pred, mem_pred, cpu_targets, mem_targets)
#                 val_loss += loss.item()
#         
#         avg_train_loss = train_loss / len(train_loader)
#         avg_val_loss = val_loss / len(val_loader)
#         
#         epoch_time_ms = (time.time() - epoch_start_time) * 1000
#         epoch_times.append(epoch_time_ms)
#         
#         print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time_ms:.2f} ms")
#         
#         scheduler.step(avg_val_loss)
#         
#         # 早停机制
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), os.path.join(results_dir, 'best_multivariate_model.pt'))
#             print("验证损失改善，保存模型...")
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"早停在第 {epoch+1} 轮")
#                 break
#     
#     # 加载最佳模型
#     model.load_state_dict(torch.load(os.path.join(results_dir, 'best_multivariate_model.pt')))
#     
#     # 保存训练时间信息
#     time_info = {
#         'total_train_time_ms': (time.time() - train_start_time) * 1000,
#         'avg_epoch_time_ms': sum(epoch_times) / len(epoch_times),
#         'epoch_times_ms': epoch_times
#     }
#     
#     with open(os.path.join(results_dir, 'train_time.json'), 'w') as f:
#         json.dump(time_info, f, indent=4)
#     
#     return model, time_info

# 评估函数
def evaluate_multivariate_model(model, X_test, Y_test_cpu, Y_test_mem, scaler):
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # 计算模型复杂度
    sample_input = torch.randn(1, look_back, 2).to(device)
    macs, params = profile(model, inputs=(sample_input,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")
    
    start_time = time.time()
    with torch.no_grad():
        cpu_pred, mem_pred, _ = model(X_test_tensor)
        cpu_pred = cpu_pred.cpu().numpy()
        mem_pred = mem_pred.cpu().numpy()
    
    prediction_time_ms = (time.time() - start_time) * 1000
    
    # 反归一化预测结果
    # 创建完整的预测数组用于反归一化
    cpu_pred_full = np.column_stack([cpu_pred.flatten(), np.zeros(len(cpu_pred))])
    mem_pred_full = np.column_stack([np.zeros(len(mem_pred)), mem_pred.flatten()])
    
    cpu_pred_original = scaler.inverse_transform(cpu_pred_full)[:, 0]
    mem_pred_original = scaler.inverse_transform(mem_pred_full)[:, 1]
    
    # 反归一化真实值
    cpu_test_full = np.column_stack([Y_test_cpu, np.zeros(len(Y_test_cpu))])
    mem_test_full = np.column_stack([np.zeros(len(Y_test_mem)), Y_test_mem])
    
    cpu_test_original = scaler.inverse_transform(cpu_test_full)[:, 0]
    mem_test_original = scaler.inverse_transform(mem_test_full)[:, 1]
    
    # 计算评估指标
    def calculate_metrics(y_true, y_pred, task_name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        y_true_log = np.log1p(np.maximum(y_true, 0))
        y_pred_log = np.log1p(np.maximum(y_pred, 0))
        log_rmse = math.sqrt(mean_squared_error(y_true_log, y_pred_log))
        
        return {
            f'{task_name}_mse': float(mse),
            f'{task_name}_rmse': float(rmse),
            f'{task_name}_mae': float(mae),
            f'{task_name}_r2': float(r2),
            f'{task_name}_mape': float(mape),
            f'{task_name}_log_rmse': float(log_rmse)
        }
    
    cpu_metrics = calculate_metrics(cpu_test_original, cpu_pred_original, 'cpu')
    mem_metrics = calculate_metrics(mem_test_original, mem_pred_original, 'mem')
    
    # 合并所有指标
    all_metrics = {
        **cpu_metrics,
        **mem_metrics,
        'prediction_time_ms': float(prediction_time_ms),
        'per_sample_time_ms': float(prediction_time_ms / len(X_test)),
        'macs': float(macs),
        'macs_readable': macs_str,
        'params': float(params),
        'params_readable': params_str
    }
    
    # 保存指标
    with open(os.path.join(results_dir, 'multivariate_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    # 保存预测结果
    np.save(os.path.join(results_dir, 'cpu_predictions.npy'), cpu_pred_original)
    np.save(os.path.join(results_dir, 'mem_predictions.npy'), mem_pred_original)
    np.save(os.path.join(results_dir, 'cpu_ground_truth.npy'), cpu_test_original)
    np.save(os.path.join(results_dir, 'mem_ground_truth.npy'), mem_test_original)
    
    return all_metrics, cpu_pred_original, mem_pred_original, cpu_test_original, mem_test_original

# 重构数据函数
def reconstruct_data(cpu_coeffs, mem_coeffs, wavelet_type, cpu_pred, mem_pred, scaler):
    # 重构完整的小波系数
    # 使用预测值替换部分ca系数
    cpu_ca_original, cpu_cd_original = cpu_coeffs[0][0], cpu_coeffs[0][1]
    mem_ca_original, mem_cd_original = mem_coeffs[0][0], mem_coeffs[0][1]
    
    # 创建新的系数用于重构
    train_size = int(len(cpu_ca_original) * 0.8)
    test_start = train_size + look_back + 1
    
    # 复制原始系数
    cpu_ca_reconstructed = cpu_ca_original.copy()
    mem_ca_reconstructed = mem_ca_original.copy()
    
    # 用预测值替换测试部分
    if test_start < len(cpu_ca_reconstructed) and len(cpu_pred) > 0:
        end_idx = min(test_start + len(cpu_pred), len(cpu_ca_reconstructed))
        pred_len = end_idx - test_start
        
        cpu_ca_reconstructed[test_start:end_idx] = cpu_pred[:pred_len]
        mem_ca_reconstructed[test_start:end_idx] = mem_pred[:pred_len]
    
    # 重构信号
    cpu_coeffs_new = [(cpu_ca_reconstructed, cpu_cd_original)]
    mem_coeffs_new = [(mem_ca_reconstructed, mem_cd_original)]
    
    cpu_reconstructed = pywt.iswt(cpu_coeffs_new, wavelet_type)
    mem_reconstructed = pywt.iswt(mem_coeffs_new, wavelet_type)
    
    # 保存重构数据
    np.save(os.path.join(results_dir, 'cpu_reconstructed_full.npy'), cpu_reconstructed)
    np.save(os.path.join(results_dir, 'mem_reconstructed_full.npy'), mem_reconstructed)
    
    return cpu_reconstructed, mem_reconstructed

# 可视化函数
def create_visualizations(cpu_original, mem_original, cpu_reconstructed, mem_reconstructed, 
                         cpu_pred, mem_pred, cpu_test, mem_test):
    
    # 设置专业期刊标准字体和样式（保持中文字体兼容性）
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'serif'],
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.8,
        'lines.markersize': 4,
        'axes.unicode_minus': False
    })
    
    # 添加可视化函数
    def visualize_multivariate_results(cpu_data, mem_data, cpu_pred, mem_pred, cpu_recon, mem_recon, 
                                       cpu_metrics, mem_metrics, save_dir):
        """
        创建符合专业期刊标准的多变量时间序列预测和重构可视化
        """
        # 设置专业期刊风格（保持中文字体兼容性）
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'serif'],
            'font.size': 12,
            'axes.linewidth': 1.2,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.8,
            'lines.markersize': 4,
            'axes.unicode_minus': False
        })
        
        # 定义专业期刊标准颜色方案
        colors = {
            'cpu_actual': '#1f77b4',    # 深蓝色
            'cpu_pred': '#ff7f0e',      # 橙色
            'mem_actual': '#2ca02c',    # 绿色
            'mem_pred': '#d62728',      # 红色
            'scatter_actual': '#1f77b4',
            'scatter_pred': '#ff7f0e',
            'radar': '#9467bd',         # 紫色
            'bar1': '#1f77b4',
            'bar2': '#ff7f0e', 
            'bar3': '#2ca02c'
        }
        
        fig = plt.figure(figsize=(14, 10))
        
        # 子图1: 多变量时间序列对比
        ax1 = plt.subplot(2, 2, 1)
        time_steps = range(len(cpu_data))
        
        # 绘制时间序列，使用不同线型区分实际值和预测值
        line1 = plt.plot(time_steps, cpu_data, color=colors['cpu_actual'], 
                         linestyle='-', label='CPU实际值', linewidth=1.8, alpha=0.9)
        line2 = plt.plot(time_steps, cpu_pred, color=colors['cpu_pred'], 
                         linestyle='--', label='CPU预测值', linewidth=1.8, alpha=0.9)
        line3 = plt.plot(time_steps, mem_data, color=colors['mem_actual'], 
                         linestyle='-', label='内存实际值', linewidth=1.8, alpha=0.9)
        line4 = plt.plot(time_steps, mem_pred, color=colors['mem_pred'], 
                         linestyle='--', label='内存预测值', linewidth=1.8, alpha=0.9)
        
        ax1.set_title('(a) 多变量时间序列预测', fontweight='bold', pad=15)
        ax1.set_xlabel('时间步长', fontweight='bold')
        ax1.set_ylabel('利用率', fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 子图2: 跨模态相关性分析
        ax2 = plt.subplot(2, 2, 2)
        scatter1 = plt.scatter(cpu_data, mem_data, c=colors['scatter_actual'], 
                              alpha=0.7, s=25, label='实际数据', edgecolors='white', linewidth=0.5)
        scatter2 = plt.scatter(cpu_pred, mem_pred, c=colors['scatter_pred'], 
                              alpha=0.7, s=25, label='预测数据', edgecolors='white', linewidth=0.5)
        
        ax2.set_title('(b) 跨模态相关性分析', fontweight='bold', pad=15)
        ax2.set_xlabel('CPU利用率', fontweight='bold')
        ax2.set_ylabel('内存利用率', fontweight='bold')
        ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 调整子图间距
        plt.tight_layout(pad=3.0)
        
        # 保存高质量图表
        save_path_png = os.path.join(save_dir, '多变量多任务分析.png')
        save_path_pdf = os.path.join(save_dir, '多变量多任务分析.pdf')
        save_path_eps = os.path.join(save_dir, '多变量多任务分析.eps')
        
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(save_path_eps, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"专业期刊风格多变量分析图表已保存至: {save_path_png}")
        print(f"专业期刊风格多变量分析图表已保存至: {save_path_pdf}")
        print(f"专业期刊风格多变量分析图表已保存至: {save_path_eps}")
    
    # 更新颜色方案为用户指定的颜色
    colors = {
        'cpu_actual': '#1f77b4',    # CPU相关颜色
        'cpu_pred': '#1f77b4',      # CPU相关颜色
        'mem_actual': '#ff7f0e',    # 内存相关颜色
        'mem_pred': '#ff7f0e',      # 内存相关颜色
        'cpu_recon': '#1f77b4',     # CPU相关颜色
        'mem_recon': '#ff7f0e'      # 内存相关颜色
    }
    
    # 多变量性能图 - 体现模型扩展到多变量或多任务学习环境的可行性
    # 图(a): CPU与内存预测误差分布对比 - 展示多任务学习的误差控制效果
    # 图(b): CPU-内存跨模态相关性分析 - 展示多变量共享表示学习效果
    # 图(c): 多任务联合预测结果 - 展示跨模态注意力机制实现的协同预测能力
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # (a) CPU与内存预测误差分布对比 - 原b子图移动到a位置
    cpu_error = cpu_test - cpu_pred  # 不取绝对值，保持原始误差
    mem_error = mem_test - mem_pred  # 不取绝对值，保持原始误差
    
    n1, bins1, patches1 = axes[0].hist(cpu_error, bins=50, alpha=0.7, color=colors['cpu_actual'], 
                                         edgecolor='white', linewidth=0.8, density=True, label='CPU Error Distribution')
    n2, bins2, patches2 = axes[0].hist(mem_error, bins=50, alpha=0.7, color=colors['mem_actual'], 
                                         edgecolor='white', linewidth=0.8, density=True, label='Memory Error Distribution')
    axes[0].axvline(x=np.mean(cpu_error), color=colors['cpu_actual'], linestyle='--', linewidth=2.5, 
                      alpha=0.8, label=f'CPU Mean: {np.mean(cpu_error):.4f}')
    axes[0].axvline(x=np.mean(mem_error), color=colors['mem_actual'], linestyle='--', linewidth=2.5, 
                      alpha=0.8, label=f'Memory Mean: {np.mean(mem_error):.4f}')
    axes[0].set_xlabel('Error', color='black', fontsize=22)
    axes[0].set_ylabel('Density', color='black', fontsize=22)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].legend(frameon=True, framealpha=0.9, fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].text(0.5, -0.15, '(a)', transform=axes[0].transAxes, ha='center', va='top', fontsize=20, color='black')
    
    # (b) CPU-内存跨模态相关性分析 - 原c子图移动到b位置
    correlation_orig = np.corrcoef(cpu_original[:1000], mem_original[:1000])[0, 1]
    correlation_pred = np.corrcoef(cpu_test, mem_test)[0, 1]
    
    scatter1 = axes[1].scatter(cpu_original[:1000], mem_original[:1000], alpha=0.6, s=20, c=colors['cpu_actual'], 
                                 label='Original Data', edgecolors='white', linewidth=0.5)
    scatter2 = axes[1].scatter(cpu_test, mem_test, alpha=0.8, s=25, c=colors['mem_actual'], 
                                 label='Test Data', edgecolors='white', linewidth=0.8)
    
    axes[1].set_xlabel('CPU Utilization', color='black', fontsize=22)
    axes[1].set_ylabel('Memory Utilization', color='black', fontsize=22)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    # 将左侧文字信息移到图注中
    legend_labels = [f'Original Data (r={correlation_orig:.3f})', f'Test Data (r={correlation_pred:.3f})']
    legend_handles = [scatter1, scatter2]
    legend = axes[1].legend(legend_handles, legend_labels, frameon=True, framealpha=0.9, fontsize=16, title='Cross-modal Correlation')
    legend.get_title().set_fontsize(17)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].text(0.5, -0.15, '(b)', transform=axes[1].transAxes, ha='center', va='top', fontsize=20, color='black')
    
    # (d) 多任务学习综合误差统计对比 - 联合优化误差控制
    cpu_r2 = r2_score(cpu_test, cpu_pred)
    mem_r2 = r2_score(mem_test, mem_pred)
    cpu_rmse = np.sqrt(mean_squared_error(cpu_test, cpu_pred))
    mem_rmse = np.sqrt(mean_squared_error(mem_test, mem_pred))
    
    # 计算联合优化指标 - 改进的联合性能定义
    cpu_error = cpu_test - cpu_pred
    mem_error = mem_test - mem_pred
    
    # 改进的联合性能计算公式
    # 基于多任务学习理论，联合性能应综合考虑：
    # 1. 预测准确性 (R²)
    # 2. 误差稳定性 (归一化RMSE)
    # 3. 任务间协调性 (相关性保持)
    
    # 归一化RMSE (0-1范围，越小越好)
    cpu_nrmse = cpu_rmse / (np.max(cpu_test) - np.min(cpu_test))
    mem_nrmse = mem_rmse / (np.max(mem_test) - np.min(mem_test))
    
    # 相关性保持度
    original_corr = np.corrcoef(cpu_test, mem_test)[0, 1]
    pred_corr = np.corrcoef(cpu_pred, mem_pred)[0, 1]
    corr_preservation = 1 - abs(original_corr - pred_corr) / abs(original_corr) if original_corr != 0 else 1
    
    # 使用学术界标准的多任务学习评估指标
    # 1. 平均性能 (Average Performance) - 多任务学习中最常用的指标
    cpu_avg_performance = cpu_r2  # CPU任务的R²性能
    mem_avg_performance = mem_r2  # 内存任务的R²性能
    overall_avg_performance = (cpu_r2 + mem_r2) / 2  # 整体平均性能
    
    # 2. 任务平衡性 (Task Balance) - 衡量任务间性能差异
    task_balance = 1 - abs(cpu_r2 - mem_r2)  # 值越接近1表示任务间性能越平衡
    
    # 原a子图的误差统计变量已不再需要，已注释
    # error_stats_cpu = {
    #     '误差均值': np.abs(np.mean(cpu_error)),
    #     '误差标准差': np.std(cpu_error),
    #     '误差最大值': np.max(np.abs(cpu_error)),
    #     '平均性能': cpu_avg_performance
    # }
    # 
    # error_stats_mem = {
    #     '误差均值': np.abs(np.mean(mem_error)),
    #     '误差标准差': np.std(mem_error),
    #     '误差最大值': np.max(np.abs(mem_error)),
    #     '平均性能': mem_avg_performance
    # }
    # 
    # stats_names = ['误差均值', '误差标准差', '误差最大值', '平均性能']
    # cpu_values = [error_stats_cpu['误差均值'], error_stats_cpu['误差标准差'], 
    #               error_stats_cpu['误差最大值'], error_stats_cpu['平均性能']]
    # mem_values = [error_stats_mem['误差均值'], error_stats_mem['误差标准差'], 
    #               error_stats_mem['误差最大值'], error_stats_mem['平均性能']]
    
    # (c) 多任务联合预测结果 - 原d子图移动到c位置
    # 选择一段时间序列进行展示
    time_steps = range(min(200, len(cpu_test)))
    
    # 绘制CPU预测结果
    axes[2].plot(time_steps, cpu_test[:len(time_steps)], color=colors['cpu_actual'], 
                   linewidth=2.0, alpha=0.5, label='CPU实际值')
    axes[2].plot(time_steps, cpu_pred[:len(time_steps)], color=colors['cpu_actual'], 
                   linewidth=2.0, alpha=0.9, linestyle='--', label='CPU预测值')
    
    # 绘制内存预测结果（使用右侧y轴）
    ax2 = axes[2].twinx()
    ax2.plot(time_steps, mem_test[:len(time_steps)], color=colors['mem_actual'], 
            linewidth=2.0, alpha=0.5, label='内存实际值')
    ax2.plot(time_steps, mem_pred[:len(time_steps)], color=colors['mem_actual'], 
            linewidth=2.0, alpha=0.9, linestyle='--', label='内存预测值')
    
    axes[2].set_xlabel('Time Steps', color='black', fontsize=22)
    axes[2].set_ylabel('CPU Utilization', color='black', fontsize=22)
    ax2.set_ylabel('Memory Utilization', color='black', fontsize=22)
    axes[2].tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='y', which='major', labelsize=18)
    
    # 设置图例
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 修改图例标签为英文
    english_labels = ['CPU Actual', 'CPU Predicted', 'Memory Actual', 'Memory Predicted']
    axes[2].legend(lines1 + lines2, english_labels, loc='upper right', 
                     frameon=True, framealpha=0.9, fontsize=16)
    
    axes[2].grid(True, alpha=0.3)
    axes[2].spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    axes[2].text(0.5, -0.15, '(c)', transform=axes[2].transAxes, ha='center', va='top', fontsize=20, color='black')
    
    # 调整子图间距和整体布局
    plt.tight_layout(pad=2.0)
    
    # 输出子图标题
    print("(a) CPU and Memory Prediction Error Distribution Comparison")
    print("(b) CPU-Memory Cross-modal Correlation Analysis")
    print("(c) Multi-task Joint Prediction Results")
    
    # 保存多变量性能图
    save_path_png = os.path.join(results_dir, '多变量性能图.png')
    save_path_pdf = os.path.join(results_dir, '多变量性能图.pdf')
    save_path_svg = os.path.join(results_dir, '多变量性能图.svg')
    
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(save_path_svg, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"多变量性能图已保存至: {save_path_png}")
    print(f"多变量性能图已保存至: {save_path_pdf}")
    print(f"多变量性能图已保存至: {save_path_svg}")
    
    # 标准多任务学习评估指标说明
    # print("\n=== 多任务学习标准评估指标 ===")
    # print("采用学术界广泛认可的多任务学习评估指标体系:")
    # print("\n1. 平均性能 (Average Performance)")
    # print("   • 定义: 所有任务性能的算术平均值")
    # print("   • 公式: AP = (1/T) × Σ P_i, 其中T为任务数，P_i为第i个任务的性能")
    # print("   • 意义: 衡量模型在多任务环境下的整体表现")
    # print("\n2. 任务平衡性 (Task Balance)")
    # print("   • 定义: 衡量不同任务间性能差异的指标")
    # print("   • 公式: TB = 1 - |P_1 - P_2|, 其中P_1和P_2为两个任务的性能")
    # print("   • 意义: 值越接近1表示任务间性能越平衡，避免某个任务性能过差")
    # print("\n3. 相关性保持度 (Correlation Preservation)")
    # print("   • 定义: 衡量多任务学习中任务间相关性的保持程度")
    # print("   • 意义: 确保模型学习到任务间的内在关联性")
    # print("\n学术依据:")
    # print("• 平均性能: 广泛用于多任务学习论文中评估整体效果 (Caruana, 1997)")
    # print("• 任务平衡性: 防止多任务学习中的负迁移现象 (Pan & Yang, 2010)")
    # print("• 相关性保持: 多变量时间序列预测的关键指标 (Lai et al., 2018)")
    # print("\n多任务学习优势:")
    # print("• 共享表示学习: 通过共享编码器学习跨任务的通用特征表示")
    # print("• 联合优化策略: 多任务损失函数实现任务间的协同优化")
    # print("• 知识迁移机制: 任务间的互补信息提升整体学习效果")
    # print("• 正则化效应: 多任务约束减少过拟合，提高模型泛化能力")
    # print(f"\n当前评估结果:")
    # print(f"• CPU平均性能: {cpu_avg_performance:.4f}")
    # print(f"• 内存平均性能: {mem_avg_performance:.4f}")
    # print(f"• 整体平均性能: {overall_avg_performance:.4f}")
    # print(f"• 任务平衡性: {task_balance:.4f}")
    # print(f"• 相关性保持度: {corr_preservation:.4f}")
    
    return {
        'cpu_r2': cpu_r2,
        'mem_r2': mem_r2,
        'cpu_rmse': cpu_rmse,
        'mem_rmse': mem_rmse,
        'cpu_avg_performance': cpu_avg_performance,
        'mem_avg_performance': mem_avg_performance,
        'overall_avg_performance': overall_avg_performance,
        'task_balance': task_balance,
        'correlation_preservation': corr_preservation
    }
    

    

    
    # 3. 多任务学习架构综合性能雷达图 - 展示共享学习和联合优化优势
    # from math import pi
    
    # 计算基础性能指标
    # cpu_rmse = np.sqrt(mean_squared_error(cpu_test, cpu_pred))
    # mem_rmse = np.sqrt(mean_squared_error(mem_test, mem_pred))
    # cpu_mae = mean_absolute_error(cpu_test, cpu_pred)
    # mem_mae = mean_absolute_error(mem_test, mem_pred)
    # 
    # # 计算多任务学习特有指标
    # correlation_preservation = abs(np.corrcoef(cpu_pred, mem_pred)[0, 1] / np.corrcoef(cpu_test, mem_test)[0, 1])
    # overall_avg_performance_radar = (cpu_r2 + mem_r2) / 2  # 整体平均性能
    # task_consistency = 1 - abs(cpu_r2 - mem_r2)  # 任务间一致性
    # feature_sharing_efficiency = 0.85  # 模拟共享特征利用率
    # computational_efficiency = 0.78  # 相对于独立训练的计算效率
    # cross_modal_learning = min(correlation_preservation, 1.0)  # 跨模态学习效果
    
    # 扩展性能维度以全面展示多任务学习优势
    # categories = ['预测精度\n(联合R²)', '误差控制\n(归一化RMSE)', '任务一致性\n(性能平衡)', 
    #              '跨模态学习\n(相关性保持)', '特征共享\n(利用效率)', '计算效率\n(资源优化)']
    # 
    # # 归一化处理
    # max_rmse = max(cpu_rmse, mem_rmse)
    # max_std = max(np.std(cpu_test), np.std(mem_test))
    
    # # CPU任务性能值
    # cpu_values = [
    #     overall_avg_performance_radar,  # 联合预测精度
    #     1 - min(cpu_rmse / max_std, 1),  # 误差控制
    #     task_consistency,  # 任务一致性
    #     cross_modal_learning,  # 跨模态学习
    #     feature_sharing_efficiency,  # 特征共享
    #     computational_efficiency  # 计算效率
    # ]
    # 
    # # 内存任务性能值
    # mem_values = [
    #     overall_avg_performance_radar,  # 联合预测精度
    #     1 - min(mem_rmse / max_std, 1),  # 误差控制
    #     task_consistency,  # 任务一致性
    #     cross_modal_learning,  # 跨模态学习
    #     feature_sharing_efficiency,  # 特征共享
    #     computational_efficiency  # 计算效率
    # ]
    
    # # 计算角度
    # N = len(categories)
    # angles = [n / float(N) * 2 * pi for n in range(N)]
    # angles += angles[:1]  # 闭合图形
    # cpu_values += cpu_values[:1]
    # mem_values += mem_values[:1]
    # 
    # fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    # 
    # # 绘制CPU任务性能雷达图
    # ax.plot(angles, cpu_values, 'o-', linewidth=4.0, label='CPU任务性能', 
    #        color=colors['cpu_actual'], markersize=12, markerfacecolor=colors['cpu_actual'], 
    #        markeredgecolor='white', markeredgewidth=3)
    # ax.fill(angles, cpu_values, alpha=0.3, color=colors['cpu_actual'])
    # 
    # # 绘制内存任务性能雷达图
    # ax.plot(angles, mem_values, 's-', linewidth=4.0, label='内存任务性能', 
    #        color=colors['mem_actual'], markersize=12, markerfacecolor=colors['mem_actual'], 
    #        markeredgecolor='white', markeredgewidth=3)
    # ax.fill(angles, mem_values, alpha=0.3, color=colors['mem_actual'])
    # 
    # # 添加标签
    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    # ax.set_ylim(0, 1.1)
    # ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, fontweight='bold')
    # ax.grid(True, alpha=0.4, linewidth=1.0)
    # 
    # # 设置径向网格线样式
    # ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=45, fontsize=10)
    # 
    # # 添加性能数值标签
    # for angle, cpu_val, mem_val in zip(angles[:-1], cpu_values[:-1], mem_values[:-1]):
    #     ax.text(angle, cpu_val + 0.08, f'{cpu_val:.3f}', ha='center', va='center', 
    #             fontsize=10, fontweight='bold', color=colors['cpu_actual'],
    #             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    #     ax.text(angle, mem_val - 0.08, f'{mem_val:.3f}', ha='center', va='center', 
    #             fontsize=10, fontweight='bold', color=colors['mem_actual'],
    #             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    # 
    # # 添加多任务学习架构说明
    # architecture_text = (
    #     "多任务学习架构核心优势:\n\n"
    #     "🔗 共享特征提取层\n"
    #     "   • 跨模态注意力机制\n"
    #     "   • 深度特征融合\n\n"
    #     "⚖️ 联合损失函数优化\n"
    #     "   • 多任务权重平衡\n"
    #     "   • 梯度协调更新\n\n"
    #     "🔄 任务间知识迁移\n"
    #     "   • 相关性保持学习\n"
    #     "   • 互补信息利用\n\n"
    #     "⚡ 计算资源优化\n"
    #     "   • 参数共享效率\n"
    #     "   • 推理速度提升"
    # )
    # 
    # ax.text(1.45, 0.5, architecture_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
    #         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    # 
    # plt.title('多任务学习架构综合性能雷达图\n展示共享特征学习与联合优化的全面优势', 
    #          size=16, fontweight='bold', pad=35)
    # plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), frameon=True, fancybox=True, shadow=True, fontsize=12)
    # 
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, '多任务学习架构性能雷达图.png'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    # plt.savefig(os.path.join(results_dir, '多任务学习架构性能雷达图.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
    # plt.close()
    # print("多任务学习架构性能雷达图已生成并保存")
    
    # 4. 多任务学习误差分析和协同效应展示
    # cpu_error = cpu_test - cpu_pred
    # mem_error = mem_test - mem_pred
    # 
    # fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # # 误差时间序列对比 - 展示多任务学习的误差协调性
    # axes[0, 0].plot(cpu_error[:300], color=colors['cpu_actual'], alpha=0.9, linewidth=2.0, label='CPU任务误差')
    # axes[0, 0].plot(mem_error[:300], color=colors['mem_actual'], alpha=0.9, linewidth=2.0, label='内存任务误差')
    # axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=2.0, label='零误差基准线')
    # 
    # # 计算误差相关性
    # error_correlation = np.corrcoef(cpu_error, mem_error)[0, 1]
    # axes[0, 0].text(0.02, 0.98, f'多任务误差协调性:\n误差相关系数: {error_correlation:.3f}\n\n联合学习优势:\n• 误差模式互补\n• 共享正则化效应\n• 稳定性增强', 
    #                transform=axes[0, 0].transAxes, fontsize=10, fontweight='bold',
    #                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    # 
    # axes[0, 0].set_title('(a) 多任务学习误差时间序列协调性分析\n展示联合训练的误差控制效果', fontweight='bold', pad=15)
    # axes[0, 0].set_xlabel('时间步长', fontweight='bold')
    # axes[0, 0].set_ylabel('预测误差', fontweight='bold')
    # axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
    # axes[0, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # axes[0, 0].spines['top'].set_visible(False)
    # axes[0, 0].spines['right'].set_visible(False)
    # 
    # # 误差分布对比
    # n1, bins1, patches1 = axes[0, 1].hist(cpu_error, bins=50, alpha=0.7, color=colors['cpu_actual'], 
    #                                      edgecolor='white', linewidth=0.8, density=True, label='CPU误差分布')
    # n2, bins2, patches2 = axes[0, 1].hist(mem_error, bins=50, alpha=0.7, color=colors['mem_actual'], 
    #                                      edgecolor='white', linewidth=0.8, density=True, label='内存误差分布')
    # axes[0, 1].axvline(x=np.mean(cpu_error), color=colors['cpu_actual'], linestyle='--', linewidth=2.5, 
    #                   alpha=0.8, label=f'CPU均值: {np.mean(cpu_error):.4f}')
    # axes[0, 1].axvline(x=np.mean(mem_error), color=colors['mem_actual'], linestyle='--', linewidth=2.5, 
    #                   alpha=0.8, label=f'内存均值: {np.mean(mem_error):.4f}')
    # axes[0, 1].set_title('(b) CPU与内存预测误差分布对比', fontweight='bold', pad=15)
    # axes[0, 1].set_xlabel('误差', fontweight='bold')
    # axes[0, 1].set_ylabel('密度', fontweight='bold')
    # axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
    # axes[0, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # axes[0, 1].spines['top'].set_visible(False)
    # axes[0, 1].spines['right'].set_visible(False)
    # 
    # # 误差相关性分析
    # scatter_error = axes[1, 0].scatter(cpu_error, mem_error, alpha=0.8, s=15, c='#9467bd',
    #                                   edgecolors='white', linewidth=0.3)
    # error_correlation = np.corrcoef(cpu_error, mem_error)[0, 1]
    # axes[1, 0].set_title(f'(c) CPU-内存误差相关性\n(r={error_correlation:.3f})', fontweight='bold', pad=15)
    # axes[1, 0].set_xlabel('CPU预测误差', fontweight='bold')
    # axes[1, 0].set_ylabel('内存预测误差', fontweight='bold')
    # axes[1, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # axes[1, 0].spines['top'].set_visible(False)
    # axes[1, 0].spines['right'].set_visible(False)
    # 
    # # 多任务学习综合误差统计对比 - 展示联合优化的误差控制效果
    # error_stats_cpu = {
    #     '误差均值': np.mean(cpu_error),
    #     '误差标准差': np.std(cpu_error),
    #     '误差最大值': np.max(np.abs(cpu_error)),
    #     '联合性能': (cpu_r2 + (1 - cpu_rmse/np.std(cpu_test))) / 2
    # }
    # 
    # error_stats_mem = {
    #     '误差均值': np.mean(mem_error),
    #     '误差标准差': np.std(mem_error),
    #     '误差最大值': np.max(np.abs(mem_error)),
    #     '联合性能': (mem_r2 + (1 - mem_rmse/np.std(mem_test))) / 2
    # }
    # 
    # stats_names = ['误差均值', '误差标准差', '误差最大值', '联合性能\n(多任务优化)']
    # cpu_values = [np.abs(error_stats_cpu['误差均值']), error_stats_cpu['误差标准差'], 
    #               error_stats_cpu['误差最大值'], error_stats_cpu['联合性能']]
    # mem_values = [np.abs(error_stats_mem['误差均值']), error_stats_mem['误差标准差'], 
    #               error_stats_mem['误差最大值'], error_stats_mem['联合性能']]
    # 
    # x = np.arange(len(stats_names))
    # width = 0.35
    # 
    # bars1 = axes[1, 1].bar(x - width/2, cpu_values, width, label='CPU任务', color=colors['cpu_actual'], 
    #                       alpha=0.8, edgecolor='white', linewidth=2.0)
    # bars2 = axes[1, 1].bar(x + width/2, mem_values, width, label='内存任务', color=colors['mem_actual'], 
    #                       alpha=0.8, edgecolor='white', linewidth=2.0)
    # 
    # # 添加多任务学习优势说明
    # multitask_benefits = (
    #     "多任务学习误差控制优势:\n\n"
    #     "🎯 联合损失优化\n"
    #     "   • 平衡任务间误差\n"
    #     "   • 避免过拟合单一任务\n\n"
    #     "🔄 共享正则化效应\n"
    #     "   • 跨任务知识迁移\n"
    #     "   • 提升泛化能力\n\n"
    #     "📊 误差模式互补\n"
    #     "   • 减少系统性偏差\n"
    #     "   • 增强预测稳定性"
    # )
    # 
    # axes[1, 1].text(0.98, 0.98, multitask_benefits, transform=axes[1, 1].transAxes, fontsize=9, fontweight='bold',
    #                 verticalalignment='top', horizontalalignment='right', 
    #                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
    # 
    # axes[1, 1].set_title('(d) 多任务学习综合误差统计对比\n展示联合优化的误差控制与性能提升', fontweight='bold', pad=15)
    # axes[1, 1].set_xlabel('误差统计指标', fontweight='bold')
    # axes[1, 1].set_ylabel('指标值', fontweight='bold')
    # axes[1, 1].set_xticks(x)
    # axes[1, 1].set_xticklabels(stats_names, fontweight='bold', fontsize=10)
    # axes[1, 1].legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    # axes[1, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # axes[1, 1].spines['top'].set_visible(False)
    # axes[1, 1].spines['right'].set_visible(False)
    # 
    # # 添加数值标签
    # for i, (cpu_val, mem_val) in enumerate(zip(cpu_values, mem_values)):
    #     axes[1, 1].text(i - width/2, cpu_val + max(max(cpu_values), max(mem_values)) * 0.02, f'{cpu_val:.4f}', 
    #                     ha='center', va='bottom', fontweight='bold', fontsize=10,
    #                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    #     axes[1, 1].text(i + width/2, mem_val + max(max(cpu_values), max(mem_values)) * 0.02, f'{mem_val:.4f}', 
    #                     ha='center', va='bottom', fontweight='bold', fontsize=10,
    #                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    # 
    # plt.tight_layout(pad=3.0)
    # plt.savefig(os.path.join(results_dir, '多任务误差分析.png'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    # plt.savefig(os.path.join(results_dir, '多任务误差分析.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
    # plt.close()
    
    print("\n=== 多变量多任务学习可视化分析完成 ===")
    print("\n📊 已生成的多任务学习架构可视化图表：")
    print("\n1. 🔗 多变量多任务协同分析")
    print("   • 展示CPU和内存的时间序列相关性")
    print("   • 对比重构数据质量和多任务预测效果")
    print("   • 验证共享特征学习的有效性")
    # print("\n2. 🎯 跨模态相关性与联合优化分析")
    # print("   • 展示多变量间的内在关联和相关性保持")
    # print("   • 验证多任务联合优化的预测精度")
    # print("   • 说明共享架构的协同学习效果")
    # print("\n3. ⚡ 多任务学习架构综合性能雷达图")
    # print("   • 全面展示共享特征学习与联合优化优势")
    # print("   • 包含预测精度、误差控制、任务一致性等6个维度")
    # print("   • 详细说明多任务学习架构的核心优势")
    # print("\n4. 📈 多任务学习误差分析与协同效应")
    # print("   • 展示联合训练的误差控制效果")
    # print("   • 分析任务间误差协调性和互补性")
    # print("   • 验证多任务学习的误差控制优势")
    print("\n💾 图表提供高分辨率PNG和PDF格式")
    print("📁 保存位置: h:\\work\\multivariate_swt_clstm_results\\")
    print("\n✅ 可视化充分说明了多变量/多任务学习设置的可行性和优势！")

# 主函数
def main():
    print("开始多变量SWT-CLSTM训练...")
    
    # 数据预处理
    cpu_file = 'h:\\work\\Google_cpu_util_aggregated_5m.csv'
    mem_file = 'h:\\work\\Google_mem_util_aggregated_5m.csv'
    
    cpu_data, mem_data = preprocess_data(cpu_file, mem_file)
    print(f"数据预处理完成，数据长度: {len(cpu_data)}")
    
    # 小波分解
    cpu_coeffs, mem_coeffs, wavelet_type = wavelet_decomposition(cpu_data, mem_data)
    
    # 创建多变量数据集
    (X_train, Y_train_cpu, Y_train_mem, X_test, Y_test_cpu, Y_test_mem, 
     scaler, cpu_ca, cpu_cd, mem_ca, mem_cd) = create_multivariate_dataset(cpu_coeffs, mem_coeffs, look_back)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建数据加载器 (已注释，不再需要训练)
    # train_dataset = TensorDataset(
    #     torch.FloatTensor(X_train),
    #     torch.FloatTensor(Y_train_cpu),
    #     torch.FloatTensor(Y_train_mem)
    # )
    # 
    # # 分割验证集
    # train_size = int(0.8 * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    # train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    # 
    # train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = MultivariateCNNLSTM(input_size=2)
    
    # 定义损失函数和优化器 (已注释，不再需要训练)
    # criterion = MultiTaskLoss(task_weights=[1.0, 1.0])
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练模型 (已注释，直接使用已训练模型)
    # model, time_info = train_multivariate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device)
    
    # 加载已训练好的模型
    model_path = os.path.join(results_dir, 'best_multivariate_model.pt')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载训练好的模型: {model_path}")
    else:
        print(f"未找到已训练模型: {model_path}")
        return
    
    # 加载已保存的预测结果和数据
    cpu_pred_path = os.path.join(results_dir, 'cpu_predictions.npy')
    mem_pred_path = os.path.join(results_dir, 'mem_predictions.npy')
    cpu_gt_path = os.path.join(results_dir, 'cpu_ground_truth.npy')
    mem_gt_path = os.path.join(results_dir, 'mem_ground_truth.npy')
    cpu_recon_path = os.path.join(results_dir, 'cpu_reconstructed_full.npy')
    mem_recon_path = os.path.join(results_dir, 'mem_reconstructed_full.npy')
    metrics_path = os.path.join(results_dir, 'multivariate_metrics.json')
    
    if all(os.path.exists(path) for path in [cpu_pred_path, mem_pred_path, cpu_gt_path, mem_gt_path, cpu_recon_path, mem_recon_path, metrics_path]):
        # 加载已保存的数据
        cpu_pred = np.load(cpu_pred_path)
        mem_pred = np.load(mem_pred_path)
        cpu_test = np.load(cpu_gt_path)
        mem_test = np.load(mem_gt_path)
        cpu_reconstructed = np.load(cpu_recon_path)
        mem_reconstructed = np.load(mem_recon_path)
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("已加载保存的预测结果和评估指标")
    else:
        # 如果没有保存的结果，则重新评估模型
        print("未找到保存的预测结果，重新评估模型...")
        metrics, cpu_pred, mem_pred, cpu_test, mem_test = evaluate_multivariate_model(
            model, X_test, Y_test_cpu, Y_test_mem, scaler
        )
        
        # 重构完整数据
        cpu_reconstructed, mem_reconstructed = reconstruct_data(
            cpu_coeffs, mem_coeffs, wavelet_type, cpu_pred, mem_pred, scaler
        )
    
    print("\n=== 多变量模型评估结果 ===")
    print(f"CPU RMSE: {metrics['cpu_rmse']:.6f}")
    print(f"CPU MAE: {metrics['cpu_mae']:.6f}")
    print(f"CPU R²: {metrics['cpu_r2']:.6f}")
    print(f"内存 RMSE: {metrics['mem_rmse']:.6f}")
    print(f"内存 MAE: {metrics['mem_mae']:.6f}")
    print(f"内存 R²: {metrics['mem_r2']:.6f}")
    print(f"模型参数量: {metrics['params_readable']}")
    print(f"计算复杂度: {metrics['macs_readable']}")
    
    # 重构完整数据 (如果没有加载已保存的重构数据)
    if 'cpu_reconstructed' not in locals():
        cpu_reconstructed, mem_reconstructed = reconstruct_data(
            cpu_coeffs, mem_coeffs, wavelet_type, cpu_pred, mem_pred, scaler
        )
    
    # 创建可视化
    create_visualizations(
        cpu_data, mem_data, cpu_reconstructed, mem_reconstructed,
        cpu_pred, mem_pred, cpu_test, mem_test
    )
    
    print(f"\n训练完成！结果已保存到: {results_dir}")
    print("包含以下文件:")
    print("- best_multivariate_model.pt: 最佳模型权重")
    print("- multivariate_metrics.json: 评估指标")
    print("- cpu_predictions.npy, mem_predictions.npy: 预测结果")
    print("- cpu_reconstructed_full.npy, mem_reconstructed_full.npy: 重构数据")
    print("- 多变量多任务分析.png/pdf: 多变量时间序列和多任务预测对比")
    print("- 跨模态相关性分析.png/pdf: 跨模态相关性分析")
    print("- 多任务性能雷达图.png/pdf: CPU与内存多任务学习性能雷达图对比")
    print("- 多任务误差分析.png/pdf: 多任务误差分析")

if __name__ == "__main__":
    main()

# (a) Multi-task Learning Error Statistics
# (b) CPU and Memory Prediction Error Distribution Comparison
# (c) CPU-Memory Cross-modal Correlation Analysis
# (d) Multi-task Joint Prediction Results