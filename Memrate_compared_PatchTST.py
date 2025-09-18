# 基于PatchTST (2023)的内存利用率预测模型
# 替换LSTM为Transformer架构的PatchTST模型
# 保持原有数据处理流程不变

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
from thop import profile, clever_format

# 忽略所有警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置matplotlib参数，解决中文显示问题
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
        'file': 'Alibaba_mem_util_aggregated_30s.csv',
        'results_dir': 'h:\\work\\Mem_patchtst_alibaba_results_30s\\'
    },
    {
        'name': 'Google_5m',
        'file': 'Google_mem_util_aggregated_5m.csv',
        'results_dir': 'h:\\work\\Mem_patchtst_google_results_5m\\'
    }
]

# 创建统一的结果目录
unified_results_dir = 'h:\\work\\Mem_patchtst_unified_results\\'
os.makedirs(unified_results_dir, exist_ok=True)

# 创建结果目录
for dataset in datasets:
    os.makedirs(dataset['results_dir'], exist_ok=True)

# 准备数据集（保持原有逻辑不变）
def create_dataset(dataset, look_back=1, for_prediction=False):
    X, Y = [], []
    if for_prediction:
        for i in range(len(dataset) - look_back + 1):
            a = dataset[i:(i + look_back)]
            X.append(a)
            if i < len(dataset) - look_back:
                Y.append(dataset[i + look_back])
            else:
                Y.append(None)
    else:
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back)]
            X.append(a)
            Y.append(dataset[i + look_back])

    if for_prediction:
        return np.array(X), None
    else:
        return np.array(X), np.array(Y)

# PatchTST模型实现 - 优化版
class PatchTSTModel(nn.Module):
    def __init__(self, seq_len, patch_len=8, stride=4, d_model=256, n_heads=16, 
                 d_ff=1024, n_layers=6, dropout=0.15, output_dim=1):
        super(PatchTSTModel, self).__init__()
        
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # 计算patch数量
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        # 改进的Patch embedding层 - 添加LayerNorm
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 0.5)
        )
        
        # 改进的位置编码 - 使用可学习的正弦位置编码
        self.positional_encoding = self._generate_positional_encoding(self.n_patches, d_model)
        self.pos_dropout = nn.Dropout(dropout * 0.3)  # 位置编码dropout
        
        # 改进的Transformer编码器层 - 使用Pre-LN架构，优化注意力机制
        encoder_layers = []
        for i in range(n_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-LN架构
            )
            encoder_layers.append(encoder_layer)
        
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 进一步优化的输出层 - 使用更深的网络和改进的残差连接
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(d_model // 4, d_model // 8),
            nn.LayerNorm(d_model // 8),
            nn.GELU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(d_model // 8, output_dim)
        )
        
        # 改进的残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.05))
        
        # 输出缩放参数
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        
        # 初始化权重
        self._init_weights()
    
    def _generate_positional_encoding(self, max_len, d_model):
        """生成改进的正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=True)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def create_patches(self, x):
        # x shape: (batch_size, seq_len, 1)
        batch_size, seq_len, _ = x.shape
        
        # 创建patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, 0]  # (batch_size, patch_len)
            patches.append(patch)
        
        # 如果patches数量不足，用零填充最后一个patch
        while len(patches) < self.n_patches:
            patch = torch.zeros(batch_size, self.patch_len, device=x.device)
            patches.append(patch)
        
        # 只保留前n_patches个patches
        patches = patches[:self.n_patches]
        
        # 堆叠patches
        patches = torch.stack(patches, dim=1)  # (batch_size, n_patches, patch_len)
        
        return patches
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        batch_size = x.shape[0]
        
        # 创建patches
        patches = self.create_patches(x)  # (batch_size, n_patches, patch_len)
        
        # Patch embedding
        patch_embeddings = self.patch_embedding(patches)  # (batch_size, n_patches, d_model)
        
        # 添加改进的位置编码
        patch_embeddings = patch_embeddings + self.positional_encoding
        patch_embeddings = self.pos_dropout(patch_embeddings)
        
        # 逐层通过Transformer编码器
        encoded = patch_embeddings
        for encoder_layer in self.transformer_encoder:
            encoded = encoder_layer(encoded)
        
        # 最终层归一化
        encoded = self.final_norm(encoded)  # (batch_size, n_patches, d_model)
        
        # 全局平均池化
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)  # (batch_size, d_model)
        
        # 输出预测 - 使用缩放参数
        output = self.output_layer(pooled)
        output = output * self.output_scale  # 应用输出缩放
        
        return output

# 计算不同预测步长的RMSE（保持原有逻辑）
def calculate_step_rmse(y_true, y_pred, max_steps=30):
    step_rmse = []
    
    for step in range(1, max_steps + 1):
        if step < len(y_true) and step < len(y_pred):
            compare_length = min(len(y_true) - step, len(y_pred) - step)
            if compare_length <= 0:
                break
                
            y_true_segment = y_true[step:step+compare_length]
            y_pred_segment = y_pred[step:step+compare_length]
            
            if y_true_segment.ndim == 1:
                y_true_segment = y_true_segment.reshape(-1, 1)
            if y_pred_segment.ndim == 1:
                y_pred_segment = y_pred_segment.reshape(-1, 1)
                
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
    
    # 加载数据（保持原有逻辑）
    try:
        data = np.loadtxt(file_path, delimiter=' ')
        data = data[data != 0]
    except:
        print(f"无法加载文件，使用随机数据进行测试")
        np.random.seed(42)
        data = np.random.rand(1000) * 100
    
    # 使用Savitzky-Golay滤波器去噪（保持原有逻辑）
    window_length = min(11, len(data) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    
    if window_length < 3:
        smoothed_data = data.copy()
        print("数据长度过短，跳过滤波")
    else:
        smoothed_data = savgol_filter(data, window_length=window_length, polyorder=min(2, window_length-1))
    
    # 确保数据长度为2的幂次方（保持原有逻辑）
    power = int(np.ceil(np.log2(len(smoothed_data))))
    padded_length = 2**power
    
    if len(smoothed_data) != padded_length:
        pad_width = padded_length - len(smoothed_data)
        smoothed_data = np.pad(smoothed_data, (0, pad_width), mode='symmetric')
        print(f"数据长度已填充至 {padded_length} (2^{power})")
    else:
        print(f"数据长度已经是2的幂次方: {padded_length}")
    
    # 分割数据为训练集和测试集（保持原有逻辑）
    train_size = int(len(smoothed_data) * 0.8)
    train, test = smoothed_data[:train_size], smoothed_data[train_size:]
    
    print(f"训练集长度: {len(train)}")
    print(f"测试集长度: {len(test)}")
    
    # 数据归一化（保持原有逻辑）
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    scaled_test = scaler.transform(test.reshape(-1, 1)).flatten()
    
    # 保存scaler
    dump(scaler, os.path.join(results_dir, 'scaler.joblib'))
    
    # 设置进一步优化后的PatchTST参数
    look_back = 70  # 保持与原LSTM相同的序列长度
    patch_len = 6   # 进一步减小patch长度，捕获更精细的时间模式
    stride = 2      # 减小步长，最大化patch重叠和信息利用
    epochs = 50
    batch_size = 16
    
    # 创建数据集（保持原有逻辑）
    X_train, y_train = create_dataset(scaled_train, look_back)
    X_test, y_test = create_dataset(scaled_test, look_back)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2)  # (samples, seq_len, 1)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(2)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 分割训练集为训练和验证
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    # 初始化进一步优化后的PatchTST模型
    model = PatchTSTModel(
        seq_len=look_back,
        patch_len=patch_len,
        stride=stride,
        d_model=512,    # 进一步增加模型维度，提升模型容量和表达能力
        n_heads=16,     # 调整注意力头数，捕获更多样的注意力模式
        d_ff=2048,      # 增加前馈网络维度，提升非线性变换能力
        n_layers=10,    # 增加层数，提升模型深度
        dropout=0.06,   # 进一步减小dropout，减少信息损失
        output_dim=1
    ).to(device)
    
    # 进一步优化损失函数和优化器
    criterion = nn.SmoothL1Loss(beta=0.1)  # 调整beta参数，更好处理小误差
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.0003,      # 降低学习率以适应更深的模型
        weight_decay=5e-5,  # 减小权重衰减
        betas=(0.9, 0.999),  # 优化动量参数
        eps=1e-8
    )
    
    # 改进的学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=15,         # 调整初始重启周期
        T_mult=1,       # 调整重启周期倍数
        eta_min=5e-7    # 调整最小学习率
    )
    
    # 训练模型 - 优化版
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15  # 早停耐心值
    
    # 记录训练开始时间
    train_start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
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
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({
                'loss': loss.item(), 
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # 验证
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]')
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).view(-1, 1)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})
        
        # 更新学习率 - 使用新的调度器
        scheduler.step()
        
        # 记录损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_temp.pt'))
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 早停
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            # 加载最佳模型
            model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model_temp.pt')))
            break
    
    # 清理临时文件
    temp_model_path = os.path.join(results_dir, 'best_model_temp.pt')
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    
    # 计算训练时间（毫秒）
    train_time_ms = (time.time() - train_start_time) * 1000
    
    # 计算模型复杂度
    sample_input = torch.randn(1, look_back, 1).to(device)
    macs, params = profile(model, inputs=(sample_input,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")
    
    # 保存模型
    model_path = os.path.join(results_dir, f'patchtst_model_mem_{dataset_name}.pt')
    torch.save(model.state_dict(), model_path)
    
    # 保存训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - PatchTST - {dataset_name}')
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
    
    # 计算训练集MAPE
    epsilon = 1e-10
    train_mape = np.mean(np.abs((y_train_original - train_predict) / (y_train_original + epsilon))) * 100
    
    test_mse = mean_squared_error(y_test_original, test_predict)
    test_rmse = math.sqrt(test_mse)
    test_mae = mean_absolute_error(test_predict, y_test_original)
    test_r2 = r2_score(y_test_original, test_predict)
    
    # 计算测试集MAPE
    test_mape = np.mean(np.abs((y_test_original - test_predict) / (y_test_original + epsilon))) * 100
    
    # 计算不同预测步长的RMSE
    max_steps = 90 if '30s' in dataset_name else 60 if '5m' in dataset_name else 30
        
    train_step_rmse = calculate_step_rmse(y_train_original, train_predict, max_steps)
    test_step_rmse = calculate_step_rmse(y_test_original, test_predict, max_steps)
    
    # 保存不同步长的RMSE
    np.save(os.path.join(results_dir, 'train_step_rmse.npy'), np.array(train_step_rmse))
    np.save(os.path.join(results_dir, 'test_step_rmse.npy'), np.array(test_step_rmse))
    
    # 同时保存到统一目录
    np.save(os.path.join(unified_results_dir, f'PatchTST_{dataset_name}_test.npy'), test_predict)
    np.save(os.path.join(unified_results_dir, f'PatchTST_test_ground_truth_{dataset_name}.npy'), y_test_original)
    np.save(os.path.join(unified_results_dir, f'PatchTST_{dataset_name}_test_step_rmse.npy'), np.array(test_step_rmse))
    
    # 绘制不同预测步长的RMSE图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(test_step_rmse) + 1), test_step_rmse, marker='o', linestyle='-', linewidth=2, markersize=4)
    
    # 设置图表标题和标签
    if dataset_name == '30s':
        plt.xlabel('预测步长 (秒)')
        plt.title('PatchTST 30秒级内存利用率预测不同步长RMSE')
    elif dataset_name == '5m':
        plt.xlabel('预测步长 (分钟)')
        plt.title('PatchTST 5分钟级内存利用率预测不同步长RMSE')
    else:
        plt.xlabel('预测步长')
        plt.title(f'PatchTST {dataset_name} 内存利用率预测不同步长RMSE')
    
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'step_rmse_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制预测结果对比图
    plt.figure(figsize=(15, 8))
    
    # 选择一个合适的显示范围
    display_range = min(500, len(y_test_original))
    
    plt.subplot(2, 1, 1)
    plt.plot(y_test_original[:display_range], label='真实值', alpha=0.8, linewidth=1.5)
    plt.plot(test_predict[:display_range], label='预测值', alpha=0.8, linewidth=1.5)
    plt.title(f'PatchTST {dataset_name} 内存利用率预测结果对比 (前{display_range}个点)')
    plt.xlabel('时间点')
    plt.ylabel('内存利用率 (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制误差图
    plt.subplot(2, 1, 2)
    error = y_test_original[:display_range] - test_predict[:display_range]
    plt.plot(error, color='red', alpha=0.7, linewidth=1)
    plt.title('预测误差')
    plt.xlabel('时间点')
    plt.ylabel('误差 (%)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'prediction_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存详细的评估指标
    metrics = {
        'dataset_name': dataset_name,
        'model_name': 'PatchTST',
        'train_metrics': {
            'mse': float(train_mse),
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'r2': float(train_r2),
            'mape': float(train_mape)
        },
        'test_metrics': {
            'mse': float(test_mse),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2': float(test_r2),
            'mape': float(test_mape)
        },
        'model_complexity': {
            'parameters': params_str,
            'macs': macs_str,
            'parameters_count': int(params),
            'macs_count': int(macs)
        },
        'training_time_ms': float(train_time_ms),
        'prediction_time_ms': float(prediction_time_ms),
        'per_sample_time_ms': float(per_sample_time_ms),
        'hyperparameters': {
            'look_back': look_back,
            'patch_len': patch_len,
            'stride': stride,
            'd_model': 512,
            'n_heads': 16,
            'd_ff': 2048,
            'n_layers': 10,
            'dropout': 0.06,
            'epochs': epochs,
            'batch_size': batch_size
        }
    }
    
    # 保存指标到JSON文件
    with open(os.path.join(results_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\n{dataset_name} 数据集处理完成！")
    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    print(f"测试集 MAPE: {test_mape:.2f}%")
    print(f"模型参数量: {params_str}")
    print(f"计算复杂度: {macs_str} MACs")
    print(f"训练时间: {train_time_ms:.2f} ms")
    print(f"预测时间: {prediction_time_ms:.2f} ms")
    print(f"每样本预测时间: {per_sample_time_ms:.4f} ms")
    
    return metrics

# 主程序
if __name__ == "__main__":
    print("开始处理内存利用率数据集...")
    
    all_results = []
    
    # 处理每个数据集
    for dataset in datasets:
        try:
            result = process_dataset(dataset)
            all_results.append(result)
        except Exception as e:
            print(f"处理数据集 {dataset['name']} 时出错: {str(e)}")
            continue
    
    # 保存所有结果的汇总
    summary_results = {
        'model_name': 'PatchTST',
        'task': 'Memory Utilization Prediction',
        'datasets_processed': len(all_results),
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存到统一结果目录
    with open(os.path.join(unified_results_dir, 'all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("所有数据集处理完成！")
    print(f"结果已保存到: {unified_results_dir}")
    print("="*60)
    
    # 打印汇总信息
    if all_results:
        print("\n汇总结果:")
        for result in all_results:
            print(f"\n{result['dataset_name']}:")
            print(f"  测试集 RMSE: {result['test_metrics']['rmse']:.4f}")
            print(f"  测试集 R²: {result['test_metrics']['r2']:.4f}")
            print(f"  模型参数量: {result['model_complexity']['parameters']}")
            print(f"  训练时间: {result['training_time_ms']:.2f} ms")