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
from thop import profile, clever_format  # 添加用于计算MACs和Parameters的库

# 忽略所有警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置matplotlib参数
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.max_open_warning'] = 50
matplotlib.use('Agg')

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义要处理的数据集列表
datasets = [
    {
        'name': 'Alibaba_30s',
        'file': 'h:\\work\\Alibaba_mem_util_aggregated_30s.csv',
        'results_dir': 'h:\\work\\alibaba_mem_timemixer_results_30s\\'
    },
    {
        'name': 'Google_5m',
        'file': 'h:\\work\\Google_mem_util_aggregated_5m.csv',
        'results_dir': 'h:\\work\\google_mem_timemixer_results_5m\\'
    }
]

# 创建统一的结果目录
unified_results_dir = 'h:\\work\\Mem_timemixer_unified_results\\'
os.makedirs(unified_results_dir, exist_ok=True)

# 创建结果目录
for dataset in datasets:
    os.makedirs(dataset['results_dir'], exist_ok=True)

def create_dataset(data, look_back=60):
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
            
            if y_true_segment.ndim == 1:
                y_true_segment = y_true_segment.reshape(-1, 1)
            if y_pred_segment.ndim == 1:
                y_pred_segment = y_pred_segment.reshape(-1, 1)
            
            rmse = np.sqrt(mean_squared_error(y_true_segment, y_pred_segment))
            step_rmse.append(rmse)
        else:
            break
    
    return step_rmse

class PDMBlock(nn.Module):
    """Past-Decomposable-Mixing Block - 优化参数"""
    def __init__(self, d_model, d_model_out):
        super(PDMBlock, self).__init__()
        self.d_model = d_model
        self.d_model_out = d_model_out
        
        # 增加分解头数量
        self.num_heads = 8  # 增加到8个头
        self.head_dim = d_model_out // self.num_heads
        
        # 分解层
        self.decomp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, self.head_dim * 2),  # 增加中间层容量
                nn.GELU(),
                nn.Dropout(0.1),  # 降低dropout
                nn.Linear(self.head_dim * 2, self.head_dim)
            ) for _ in range(self.num_heads)
        ])
        
        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(d_model_out, d_model_out * 3),  # 增加融合网络容量
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model_out * 3, d_model_out * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model_out * 2, d_model_out)
        )
        
        self.norm1 = nn.LayerNorm(d_model_out)
        self.norm2 = nn.LayerNorm(d_model_out)
        self.dropout = nn.Dropout(0.15)
        
        # 注意力头数量
        num_attention_heads = 16  # 增加注意力头数量
        if d_model_out % num_attention_heads != 0:
            num_attention_heads = 8  # 备选方案
        self.attention = nn.MultiheadAttention(d_model_out, num_heads=num_attention_heads, dropout=0.1)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model_out, d_model_out * 4),  # 增加FFN容量
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model_out * 4, d_model_out * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model_out * 2, d_model_out)
        )
        
    def forward(self, x):
        # x shape: (batch_size, d_model)
        # 多头分解
        decomp_outputs = []
        for decomp_layer in self.decomp_layers:
            decomp_out = decomp_layer(x)
            decomp_outputs.append(decomp_out)
        
        # 拼接多头输出 - 现在维度正确
        x_decomp = torch.cat(decomp_outputs, dim=-1)  # (batch_size, d_model_out)
        
        # 融合
        x_fused = self.fusion(x_decomp)
        x_fused = self.norm1(x_fused + x_decomp)  # 残差连接
        
        # 自注意力
        x_att = x_fused.unsqueeze(0)  # (1, batch_size, d_model_out)
        x_att, _ = self.attention(x_att, x_att, x_att)
        x_att = x_att.squeeze(0)  # (batch_size, d_model_out)
        
        # 残差连接和归一化
        x_att = self.norm2(x_fused + x_att)
        
        # 前馈网络
        x_ffn = self.ffn(x_att)
        x_out = x_att + x_ffn  # 残差连接
        x_out = self.dropout(x_out)
        
        return x_out

class FMMBlock(nn.Module):
    """Future-Multipredictor-Mixing Block - 优化参数"""
    def __init__(self, d_model, pred_len):
        super(FMMBlock, self).__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        
        # 增加预测器数量
        self.num_predictors = 6  # 增加到6个预测器
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(d_model // 2, d_model // 3),  # 增加中间层
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(d_model // 3, d_model // 4),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(d_model // 4, pred_len)
            ) for _ in range(self.num_predictors)
        ])
        
        # 权重网络
        self.weight_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model // 3),  # 增加中间层
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 3, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 4, self.num_predictors),
            nn.Softmax(dim=-1)
        )
        
        # 注意力融合机制
        self.attention_embed_dim = max(16, pred_len * 8)  # 增加注意力维度
        self.attention_embed_dim = (self.attention_embed_dim // 8) * 8  # 确保能被8整除
        
        self.pre_attention_proj = nn.Linear(pred_len, self.attention_embed_dim)
        self.attention_fusion = nn.MultiheadAttention(self.attention_embed_dim, num_heads=8, dropout=0.1)  # 增加头数量
        self.post_attention_proj = nn.Linear(self.attention_embed_dim, pred_len)
        
    def forward(self, x):
        # x shape: (batch_size, d_model)
        predictions = []
        for predictor in self.predictors:
            pred = predictor(x)  # (batch_size, pred_len)
            predictions.append(pred)
        
        # 动态权重计算
        weights = self.weight_network(x)  # (batch_size, num_predictors)
        
        # 加权组合
        predictions = torch.stack(predictions, dim=-1)  # (batch_size, pred_len, num_predictors)
        weights = weights.unsqueeze(1)  # (batch_size, 1, num_predictors)
        output = torch.sum(predictions * weights, dim=-1)  # (batch_size, pred_len)
        
        # 注意力融合
        output_proj = self.pre_attention_proj(output)  # (batch_size, attention_embed_dim)
        output_att = output_proj.unsqueeze(0)  # (1, batch_size, attention_embed_dim)
        output_att, _ = self.attention_fusion(output_att, output_att, output_att)
        output_att = output_att.squeeze(0)  # (batch_size, attention_embed_dim)
        output = self.post_attention_proj(output_att)  # (batch_size, pred_len)
        
        return output

class TimeMixerModel(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=512, n_blocks=6):  # 将d_model从768改为512
        super(TimeMixerModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # 增强输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(0.1),  # 降低dropout以减少信息丢失
            nn.LayerNorm(d_model)
        )
        
        # 增加PDM块数量
        self.pdm_blocks = nn.ModuleList([
            PDMBlock(d_model, d_model) for _ in range(n_blocks)
        ])
        
        # 增强特征增强层
        self.feature_enhancement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1)  # 降低dropout
            ) for _ in range(3)  # 增加到3个
        ])
        
        # FMM块用于未来预测混合
        self.fmm_block = FMMBlock(d_model, pred_len)
        
        # 增强编码器
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # 进一步增加容量
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 4, d_model * 3),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.15)
        )
        
        # 增强解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 4, d_model // 8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 8, 1)
        )
        
        # 调整残差连接权重
        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.15)) for _ in range(n_blocks)  # 增加残差权重
        ])
        
        # 全局残差连接
        self.global_residual = nn.Parameter(torch.tensor(0.25))  # 增加全局残差权重
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        # 输入投影
        x_proj = self.input_projection(x)  # (batch_size, d_model)
        x_original = x_proj.clone()  # 保存原始投影用于全局残差
        
        # 通过PDM块进行过去信息分解
        x_pdm = x_proj
        for i, pdm in enumerate(self.pdm_blocks):
            x_pdm_new = pdm(x_pdm)
            # 残差连接
            x_pdm = x_pdm + self.residual_weights[i] * x_pdm_new
            
            # 特征增强（增加频率）
            if i % 2 == 1 and i // 2 < len(self.feature_enhancement):  # 每两层进行一次特征增强
                x_pdm = self.feature_enhancement[i // 2](x_pdm) + x_pdm
        
        # 编码
        x_encoded = self.encoder(x_pdm)
        # 残差连接
        x_encoded = x_encoded + x_pdm
        
        # 全局残差连接
        x_encoded = x_encoded + self.global_residual * x_original
        
        # 解码预测
        output = self.decoder(x_encoded)  # (batch_size, 1)
        return output.squeeze(-1)  # (batch_size,)

# 添加计算模型复杂度的函数
def calculate_model_complexity(model, input_tensor):
    """计算模型的MACs和Parameters"""
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor,))
        macs, params = clever_format([macs, params], "%.3f")
    return macs, params

# 在其他函数定义之后添加数据增强函数
def add_noise_augmentation(batch_x, batch_y, noise_factor=0.02):
    """
    为训练数据添加噪声增强
    
    Args:
        batch_x: 输入数据批次
        batch_y: 目标数据批次  
        noise_factor: 噪声强度因子
    
    Returns:
        增强后的输入和目标数据
    """
    # 为输入数据添加高斯噪声
    noise_x = torch.randn_like(batch_x) * noise_factor
    augmented_x = batch_x + noise_x
    
    # 为目标数据添加较小的噪声
    noise_y = torch.randn_like(batch_y) * (noise_factor * 0.5)
    augmented_y = batch_y + noise_y
    
    return augmented_x, augmented_y

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, results_dir):
    """训练TimeMixer模型"""
    model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15  # 增加patience，给模型更多训练时间
    patience_counter = 0
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 数据增强
            if np.random.random() > 0.3:  # 增加数据增强概率到70%
                batch_x, batch_y = add_noise_augmentation(batch_x, batch_y, noise_factor=0.015)  # 降低噪声强度
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # L2正则化
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += 0.0005 * l2_reg  # 降低L2正则化强度
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 放宽梯度裁剪
            
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_timemixer_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 记录训练结束时间
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_timemixer_model.pt')))
    
    # 返回时间信息
    time_info = {
        'training_time': training_time
    }
    
    return model, time_info, train_losses, val_losses

def evaluate_model(model, X_test, y_test, scaler, results_dir, dataset_name):
    """评估模型"""
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()
    
    # 反归一化
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # 计算评估指标
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    
    # 计算MAPE
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_mask = y_true != 0
        if np.sum(non_zero_mask) == 0:
            return 0
        return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
    
    # 计算Log RMSE
    log_rmse = np.sqrt(mean_squared_error(np.log1p(np.abs(y_test_original)), np.log1p(np.abs(y_pred_original))))
    
    # 计算步进RMSE
    max_steps = 30  # 默认最大预测步长
    if dataset_name == 'Alibaba_30s':
        max_steps = 90  # 30秒级别，最大90步
    elif dataset_name == 'Google_5m':
        max_steps = 60  # 5分钟级别，最大60步
        
    test_step_rmse = calculate_step_rmse(y_test_original, y_pred_original, max_steps)
    
    metrics = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'MAPE': float(mape),
        'Log_RMSE': float(log_rmse),
        'Step_RMSE_Mean': float(np.mean(test_step_rmse)) if test_step_rmse else 0.0
    }
    
    # 保存评估指标
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 保存预测结果
    np.save(os.path.join(results_dir, 'test_predictions.npy'), y_pred_original)
    np.save(os.path.join(results_dir, 'test_ground_truth.npy'), y_test_original)
    np.save(os.path.join(results_dir, 'test_step_rmse.npy'), np.array(test_step_rmse))
    
    print(f"TimeMixer模型评估完成:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Log RMSE: {log_rmse:.6f}")
    
    return metrics, y_pred_original, y_test_original

def process_dataset(dataset_info):
    """处理单个数据集"""
    print(f"\n处理数据集: {dataset_info['name']}")
    
    dataset_name = dataset_info['name']
    file_path = dataset_info['file']
    results_dir = dataset_info['results_dir']
    
    # 参数设置
    look_back = 70
    epochs = 50
    batch_size = 16
    results_dir = dataset['results_dir']
    
    # 加载数据 - 修改为与SWD-CLSTM一致的加载方式
    data_path = os.path.join('h:\\work', dataset['file'])
    try:
        data = np.loadtxt(data_path, delimiter=' ')
        data = data[data != 0]  # 去除0元素
    except:
        print(f"无法加载文件，使用随机数据进行测试")
        # 生成测试数据
        np.random.seed(42)
        data = np.random.rand(1000) * 100
    
    print(f"原始数据长度: {len(data)}")
    
    # 应用Savitzky-Golay滤波 - 与SWD-CLSTM完全一致
    window_length = min(11, len(data) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    
    if window_length < 3:
        smoothed_data = data.copy()
        print("数据长度过短，跳过滤波")
    else:
        smoothed_data = savgol_filter(data, window_length=window_length, polyorder=min(2, window_length-1))
    print(f"滤波后数据长度: {len(smoothed_data)}")
    
    # 确保数据长度为2的幂次方 - 与SWD-CLSTM完全一致
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
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 创建增强的TimeMixer模型
    model = TimeMixerModel(seq_len=look_back, d_model=512, pred_len=1, n_blocks=6)  # 将d_model从768改为512
    model.to(device)
    
    # 计算模型复杂度
    sample_input = torch.randn(1, look_back).to(device)
    macs, params = calculate_model_complexity(model, sample_input)
    print(f"模型复杂度 - MACs: {macs}, Parameters: {params}")
    
    # 优化器参数调整 - 与SWD-CLSTM一致
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)  # 降低学习率，增加权重衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)  # 调整学习率调度
    
    # 训练模型
    model, time_info, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, results_dir
    )
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(results_dir, 'timemixer_model.pt'))
    
    # 评估模型并记录预测时间
    prediction_start_time = time.time()
    
    model.eval()
    train_predictions = []
    train_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            train_predictions.extend(outputs.squeeze().cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())
    
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            test_predictions.extend(outputs.squeeze().cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
    
    prediction_end_time = time.time()
    prediction_time = prediction_end_time - prediction_start_time
    
    # 反归一化
    train_predictions = np.array(train_predictions).reshape(-1, 1)
    train_targets = np.array(train_targets).reshape(-1, 1)
    test_predictions = np.array(test_predictions).reshape(-1, 1)
    test_targets = np.array(test_targets).reshape(-1, 1)
    
    train_predictions_rescaled = scaler.inverse_transform(train_predictions)
    train_targets_rescaled = scaler.inverse_transform(train_targets)
    test_predictions_rescaled = scaler.inverse_transform(test_predictions)
    test_targets_rescaled = scaler.inverse_transform(test_targets)
    
    # 计算评估指标
    train_rmse = np.sqrt(mean_squared_error(train_targets_rescaled, train_predictions_rescaled))
    train_mae = mean_absolute_error(train_targets_rescaled, train_predictions_rescaled)
    train_r2 = r2_score(train_targets_rescaled, train_predictions_rescaled)
    train_mape = np.mean(np.abs((train_targets_rescaled - train_predictions_rescaled) / train_targets_rescaled)) * 100
    
    test_rmse = np.sqrt(mean_squared_error(test_targets_rescaled, test_predictions_rescaled))
    test_mae = mean_absolute_error(test_targets_rescaled, test_predictions_rescaled)
    test_r2 = r2_score(test_targets_rescaled, test_predictions_rescaled)
    test_mape = np.mean(np.abs((test_targets_rescaled - test_predictions_rescaled) / test_targets_rescaled)) * 100
    
    # 计算步进RMSE
    max_steps = 30  # 默认最大预测步长
    if dataset_name == 'Alibaba_30s':
        max_steps = 90  # 30秒级别，最大90步
    elif dataset_name == 'Google_5m':
        max_steps = 60  # 5分钟级别，最大60步
        
    train_step_rmse = calculate_step_rmse(train_targets_rescaled.flatten(), train_predictions_rescaled.flatten(), max_steps)
    test_step_rmse = calculate_step_rmse(test_targets_rescaled.flatten(), test_predictions_rescaled.flatten(), max_steps)
    
    # 组织结果
    train_metrics = {
        'rmse': float(train_rmse),
        'mae': float(train_mae),
        'r2': float(train_r2),
        'mape': float(train_mape)
    }
    
    test_metrics = {
        'rmse': float(test_rmse),
        'mae': float(test_mae),
        'r2': float(test_r2),
        'mape': float(test_mape)
    }
    
    results = {
        'dataset': dataset_name,
        'model_complexity': {
            'MACs': macs,
            'Parameters': params,
            'training_time': float(time_info['training_time']),
            'prediction_time': float(prediction_time)
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_step_rmse': train_step_rmse,
        'test_step_rmse': test_step_rmse
    }
    
    # 保存到文件
    np.save(os.path.join(results_dir, 'train_predictions.npy'), train_predictions_rescaled)
    np.save(os.path.join(results_dir, 'train_ground_truth.npy'), train_targets_rescaled)
    np.save(os.path.join(results_dir, 'test_predictions.npy'), test_predictions_rescaled)
    np.save(os.path.join(results_dir, 'test_ground_truth.npy'), test_targets_rescaled)
    np.save(os.path.join(results_dir, 'train_step_rmse.npy'), train_step_rmse)
    np.save(os.path.join(results_dir, 'test_step_rmse.npy'), test_step_rmse)
    
    # 保存模型复杂度和时间信息
    complexity_timing = {
        'MACs': macs,
        'Parameters': params,
        'training_time': float(time_info['training_time']),
        'prediction_time': float(prediction_time)
    }
    
    with open(os.path.join(results_dir, 'complexity_timing.json'), 'w') as f:
        json.dump(complexity_timing, f, indent=4)
    
    # 保存scaler和指标
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 保存到统一结果目录
    np.save(os.path.join(unified_results_dir, f'TimeMixer_{dataset_name}_test.npy'), test_predictions_rescaled)
    np.save(os.path.join(unified_results_dir, f'TimeMixer_{dataset_name}_test_step_rmse.npy'), test_step_rmse)
    
    return results

# 主程序入口点
if __name__ == "__main__":
    print("开始训练TimeMixer模型...")
    
    # 处理每个数据集
    all_metrics = {}
    for dataset in datasets:
        print(f"\n开始处理数据集: {dataset['name']}")
        results = process_dataset(dataset)  # 只接收一个返回值
        all_metrics[dataset['name']] = results
    
    # 保存所有数据集的评估指标到统一目录
    with open(os.path.join(unified_results_dir, 'timemixer_all_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print("\nTimeMixer模型处理完成！")