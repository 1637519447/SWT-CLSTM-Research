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
        'file': 'h:\\work\\Alibaba_cpu_util_aggregated_30s.csv',
        'results_dir': 'h:\\work\\alibaba_cpu_timemixerplusplus_results_30s\\'
    },
    {
        'name': 'Google_5m',
        'file': 'h:\\work\\Google_cpu_util_aggregated_5m.csv',
        'results_dir': 'h:\\work\\google_cpu_timemixerplusplus_results_5m\\'
    }
]

# 创建统一的结果目录
unified_results_dir = 'h:\\work\\CPU_timemixerplusplus_unified_results\\'
os.makedirs(unified_results_dir, exist_ok=True)

# 创建结果目录
for dataset in datasets:
    os.makedirs(dataset['results_dir'], exist_ok=True)

def create_dataset(dataset, look_back=60, for_prediction=False):
    """创建时间序列数据集"""
    X, Y = [], []
    if for_prediction:
        # 如果是用于预测，我们需要保留完整的输入序列
        for i in range(len(dataset) - look_back + 1):
            if dataset.ndim == 2:
                a = dataset[i:(i + look_back), 0]
            else:
                a = dataset[i:(i + look_back)]
            X.append(a)
            if i < len(dataset) - look_back:
                if dataset.ndim == 2:
                    Y.append(dataset[i + look_back, 0])
                else:
                    Y.append(dataset[i + look_back])
            else:
                # 在最后一个片段，Y可以是None或者最后一个值
                Y.append(None)
    else:
        # 如果不是用于预测，按照原来的逻辑
        for i in range(len(dataset) - look_back - 1):
            if dataset.ndim == 2:
                a = dataset[i:(i + look_back), 0]
                Y.append(dataset[i + look_back, 0])
            else:
                a = dataset[i:(i + look_back)]
                Y.append(dataset[i + look_back])
            X.append(a)

    # 返回X和Y，但在预测模式下只返回X
    if for_prediction:
        return np.array(X), None
    else:
        return np.array(X), np.array(Y)

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

class AdaptivePDMBlock(nn.Module):
    """Adaptive Past-Decomposable-Mixing Block - TimeMixer++核心组件"""
    def __init__(self, d_model, d_model_out, num_scales=4):
        super(AdaptivePDMBlock, self).__init__()
        self.d_model = d_model
        self.d_model_out = d_model_out
        self.num_scales = num_scales
        
        # 多尺度分解头 - 确保维度匹配
        self.num_heads = 2  # 进一步减少头数量以提高训练效率
        # 确保d_model_out能被num_heads整除
        if d_model_out % self.num_heads != 0:
            # 向上调整到最近的能被num_heads整除的数
            self.adjusted_d_model_out = ((d_model_out + self.num_heads - 1) // self.num_heads) * self.num_heads
            self.head_proj = nn.Linear(d_model_out, self.adjusted_d_model_out)
            self.head_proj_back = nn.Linear(self.adjusted_d_model_out, d_model_out)
        else:
            self.adjusted_d_model_out = d_model_out
            self.head_proj = None
            self.head_proj_back = None
        
        self.head_dim = self.adjusted_d_model_out // self.num_heads
        
        # 自适应尺度选择网络
        self.scale_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # 多尺度分解层 - 进一步简化网络结构
        self.multi_scale_decomp = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, self.head_dim),
                    nn.GELU()
                ) for _ in range(self.num_heads)
            ]) for _ in range(num_scales)
        ])
        
        # 跨尺度注意力融合 - 使用调整后的维度
        attention_heads = 2
        attention_dim = (self.adjusted_d_model_out // attention_heads) * attention_heads
        self.cross_scale_attention = nn.MultiheadAttention(
            attention_dim, num_heads=attention_heads, dropout=0.1, batch_first=True
        )
        
        # 如果adjusted_d_model_out不能被8整除，添加投影层
        if self.adjusted_d_model_out != attention_dim:
            self.attention_proj = nn.Linear(self.adjusted_d_model_out, attention_dim)
            self.attention_proj_back = nn.Linear(attention_dim, self.adjusted_d_model_out)
        else:
            self.attention_proj = None
            self.attention_proj_back = None
        
        # 增强融合网络 - 极简结构
        self.enhanced_fusion = nn.Sequential(
            nn.Linear(d_model_out, d_model_out),
            nn.GELU()
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model_out)
        self.norm2 = nn.LayerNorm(d_model_out)
        self.norm3 = nn.LayerNorm(d_model_out)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model_out * 2, d_model_out),
            nn.Sigmoid()
        )
        
        # 前馈网络 - 极简结构
        self.ffn = nn.Sequential(
            nn.Linear(d_model_out, d_model_out),
            nn.GELU()
        )
        
    def forward(self, x):
        # x shape: (batch_size, d_model)
        batch_size = x.size(0)
        
        # 自适应尺度选择
        scale_weights = self.scale_selector(x)  # (batch_size, num_scales)
        
        # 多尺度分解
        scale_outputs = []
        for scale_idx in range(self.num_scales):
            head_outputs = []
            for head_idx in range(self.num_heads):
                head_out = self.multi_scale_decomp[scale_idx][head_idx](x)
                head_outputs.append(head_out)
            
            scale_out = torch.cat(head_outputs, dim=-1)  # (batch_size, adjusted_d_model_out)
            scale_outputs.append(scale_out)
        
        # 加权组合多尺度输出
        scale_outputs = torch.stack(scale_outputs, dim=1)  # (batch_size, num_scales, adjusted_d_model_out)
        scale_weights = scale_weights.unsqueeze(-1)  # (batch_size, num_scales, 1)
        x_multi_scale = torch.sum(scale_outputs * scale_weights, dim=1)  # (batch_size, adjusted_d_model_out)
        
        # 跨尺度注意力
        if self.attention_proj is not None:
            x_multi_scale_proj = self.attention_proj(x_multi_scale)
            scale_outputs_proj = self.attention_proj(scale_outputs.view(-1, scale_outputs.size(-1))).view(scale_outputs.shape[0], scale_outputs.shape[1], -1)
        else:
            x_multi_scale_proj = x_multi_scale
            scale_outputs_proj = scale_outputs
            
        x_att, _ = self.cross_scale_attention(
            x_multi_scale_proj.unsqueeze(1), 
            scale_outputs_proj, 
            scale_outputs_proj
        )
        x_att = x_att.squeeze(1)  # (batch_size, attention_dim)
        
        if self.attention_proj_back is not None:
            x_att = self.attention_proj_back(x_att)  # (batch_size, adjusted_d_model_out)
        
        # 转换回原始维度
        if self.head_proj_back is not None:
            x_multi_scale_orig = self.head_proj_back(x_multi_scale)  # (batch_size, d_model_out)
        else:
            x_multi_scale_orig = x_multi_scale
            
        # 注意力输出的维度转换
        if self.attention_proj_back is not None:
            x_att_orig = x_att  # attention_proj_back已经将维度转换回adjusted_d_model_out
            if self.head_proj_back is not None:
                x_att_orig = self.head_proj_back(x_att_orig)  # 进一步转换到原始d_model_out
        else:
            if self.head_proj_back is not None:
                x_att_orig = self.head_proj_back(x_att)  # 直接转换到原始d_model_out
            else:
                x_att_orig = x_att
        
        # 残差连接和归一化
        x_att_final = self.norm1(x_multi_scale_orig + x_att_orig)
        
        # 增强融合
        x_fused = self.enhanced_fusion(x_att_final)
        x_fused = self.norm2(x_att_final + x_fused)
        
        # 门控机制
        gate_input = torch.cat([x_multi_scale_orig, x_fused], dim=-1)
        gate_weights = self.gate(gate_input)
        x_gated = gate_weights * x_fused + (1 - gate_weights) * x_multi_scale_orig
        
        # 前馈网络
        x_ffn = self.ffn(x_gated)
        x_out = self.norm3(x_gated + x_ffn)
        
        return x_out

class EnhancedFMMBlock(nn.Module):
    """Enhanced Future-Multipredictor-Mixing Block - TimeMixer++改进版"""
    def __init__(self, d_model, pred_len, num_experts=2):
        super(EnhancedFMMBlock, self).__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.num_experts = num_experts
        
        # 专家网络（极简结构）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, pred_len),
                nn.GELU()
            ) for _ in range(num_experts)
        ])
        
        # 动态路由网络（简化结构）
        self.router = nn.Sequential(
            nn.Linear(d_model, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 自适应聚合注意力 - 确保embed_dim能被num_heads整除
        attention_heads = 2
        attention_dim = max(2, (pred_len // attention_heads) * attention_heads)
        if attention_dim < 2:  # 最小维度保证
            attention_dim = 2
            attention_heads = min(2, pred_len)
        
        self.aggregation_attention = nn.MultiheadAttention(
            attention_dim, num_heads=attention_heads, dropout=0.1, batch_first=True
        )
        
        # 如果pred_len不等于attention_dim，添加投影层
        if pred_len != attention_dim:
            self.pred_proj = nn.Linear(pred_len, attention_dim)
            self.pred_proj_back = nn.Linear(attention_dim, pred_len)
        else:
            self.pred_proj = None
            self.pred_proj_back = None
        
        # 时序增强模块
        self.temporal_enhancement = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )
        
        # 残差缩放
        self.residual_scale = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, x):
        # x shape: (batch_size, d_model)
        batch_size = x.size(0)
        
        # 专家预测
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (batch_size, pred_len)
            expert_outputs.append(expert_out)
        
        # 动态路由权重
        routing_weights = self.router(x)  # (batch_size, num_experts)
        
        # 加权组合专家输出
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (batch_size, pred_len, num_experts)
        routing_weights = routing_weights.unsqueeze(1)  # (batch_size, 1, num_experts)
        weighted_output = torch.sum(expert_outputs * routing_weights, dim=-1)  # (batch_size, pred_len)
        
        # 自适应聚合注意力
        if self.pred_proj is not None:
            weighted_output_proj = self.pred_proj(weighted_output)
        else:
            weighted_output_proj = weighted_output
            
        att_input = weighted_output_proj.unsqueeze(1)  # (batch_size, 1, attention_dim)
        att_output, _ = self.aggregation_attention(att_input, att_input, att_input)
        att_output = att_output.squeeze(1)  # (batch_size, attention_dim)
        
        if self.pred_proj_back is not None:
            att_output = self.pred_proj_back(att_output)  # (batch_size, pred_len)
        
        # 时序增强
        temporal_input = weighted_output.unsqueeze(1)  # (batch_size, 1, pred_len)
        temporal_enhanced = self.temporal_enhancement(temporal_input)
        temporal_enhanced = temporal_enhanced.squeeze(1)  # (batch_size, pred_len)
        
        # 残差连接
        output = att_output + self.residual_scale * temporal_enhanced
        
        return output

class TimeMixerPlusPlusModel(nn.Module):
    """TimeMixer++ Model - ICLR 2025后继者"""
    def __init__(self, seq_len, pred_len, d_model=128, n_blocks=2, num_scales=3):
        super(TimeMixerPlusPlusModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # 增强输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(seq_len, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.LayerNorm(d_model)
        )
        
        # 位置编码
        self.pos_encoding = self._generate_positional_encoding(d_model)
        
        # 自适应PDM块
        self.adaptive_pdm_blocks = nn.ModuleList([
            AdaptivePDMBlock(d_model, d_model, num_scales) for _ in range(n_blocks)
        ])
        
        # 跨层连接权重
        self.cross_layer_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1 + 0.05 * i)) for i in range(n_blocks)
        ])
        
        # 多层特征增强
        self.feature_enhancement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.LayerNorm(d_model * 2),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model)
            ) for _ in range(4)
        ])
        
        # 增强FMM块
        self.enhanced_fmm_block = EnhancedFMMBlock(d_model, pred_len, num_experts=2)
        
        # 深度编码器 - 极简结构
        self.deep_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        # 深度解码器 - 极简结构
        self.deep_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # 全局残差连接
        self.global_residual = nn.Parameter(torch.tensor(0.3))
        
        # 输出缩放
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        
        # 输出投影层 - 简化结构
        self.output_projection = nn.Sequential(
            nn.Linear(1, 1),
            nn.Tanh()
        )
        
    def _generate_positional_encoding(self, d_model, max_len=1000):
        """生成位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size = x.size(0)
        
        # 输入投影
        x_proj = self.input_projection(x)  # (batch_size, d_model)
        
        # 添加位置编码
        x_proj = x_proj + self.pos_encoding[:, :1, :].squeeze(1)
        
        x_original = x_proj.clone()  # 保存原始投影用于全局残差
        
        # 通过自适应PDM块
        x_pdm = x_proj
        for i, pdm in enumerate(self.adaptive_pdm_blocks):
            x_pdm_new = pdm(x_pdm)
            # 跨层残差连接
            x_pdm = x_pdm + self.cross_layer_weights[i] * x_pdm_new
            
            # 特征增强（每两层进行一次）
            if i % 2 == 1 and i // 2 < len(self.feature_enhancement):
                x_pdm = self.feature_enhancement[i // 2](x_pdm) + x_pdm
        
        # 深度编码
        x_encoded = self.deep_encoder(x_pdm)
        x_encoded = x_encoded + x_pdm  # 残差连接
        
        # 全局残差连接
        x_encoded = x_encoded + self.global_residual * x_original
        
        # 深度解码预测
        output = self.deep_decoder(x_encoded)  # (batch_size, 1)
        output = output * self.output_scale  # 输出缩放
        
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
def add_noise_augmentation(batch_x, batch_y, noise_factor=0.015):
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
    noise_y = torch.randn_like(batch_y) * (noise_factor * 0.3)
    augmented_y = batch_y + noise_y
    
    return augmented_x, augmented_y

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, results_dir):
    """训练TimeMixer++模型"""
    model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 18  # 增加patience
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
            if np.random.random() > 0.25:  # 75%概率进行数据增强
                batch_x, batch_y = add_noise_augmentation(batch_x, batch_y, noise_factor=0.012)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # L2正则化
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += 0.0003 * l2_reg  # 降低L2正则化强度
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            
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
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_timemixerplusplus_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 记录训练结束时间
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_timemixerplusplus_model.pt')))
    
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
    
    print(f"TimeMixer++模型评估完成:")
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
    
    # 加载数据 - 修改为与SWD-CLSTM一致的加载方式
    data_path = os.path.join('h:\\work', dataset_info['file'])
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
    print(f"调整后数据长度: {len(smoothed_data)}")
    
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
    
    # 创建数据集
    X_train, y_train = create_dataset(scaled_train, look_back)
    X_test, y_test = create_dataset(scaled_test, look_back)
    print(f"训练数据集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试数据集形状: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 进一步分割训练集和验证集
    val_size = int(len(X_train) * 0.2)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建TimeMixer++模型
    model = TimeMixerPlusPlusModel(
        seq_len=look_back,
        pred_len=1,
        d_model=256,  # 简化模型维度以加快训练
        n_blocks=4,   # 减少块数量以加快训练
        num_scales=3  # 减少尺度数量
    )
    
    # 计算模型复杂度
    sample_input = torch.randn(1, look_back)
    macs, params = calculate_model_complexity(model, sample_input)
    print(f"模型复杂度 - MACs: {macs}, Parameters: {params}")
    
    # 定义损失函数和优化器
    criterion = nn.SmoothL1Loss(beta=0.1)  # 适配简化模型的beta参数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,  # 适当提高学习率以加快收敛
        weight_decay=1e-4,  # 适配简化模型的权重衰减
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # 适配简化模型的衰减因子
        patience=5,  # 减少patience以加快调整
        min_lr=1e-6,
        verbose=True
    )
    
    # 训练模型
    print("开始训练TimeMixer++模型...")
    model, time_info, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, results_dir
    )
    
    # 评估模型
    print("评估TimeMixer++模型...")
    metrics, y_pred, y_test_original = evaluate_model(model, X_test, y_test, scaler, results_dir, dataset_name)
    
    # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': time_info['training_time'],
        'model_complexity': {
            'MACs': macs,
            'Parameters': params
        }
    }
    
    with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=4)
    
    # 保存scaler
    dump(scaler, os.path.join(results_dir, 'scaler.joblib'))
    
    return metrics, time_info

def main():
    """主函数"""
    print("开始TimeMixer++模型训练和评估...")
    
    all_results = {}
    all_time_info = {}
    
    for dataset_info in datasets:
        try:
            metrics, time_info = process_dataset(dataset_info)
            all_results[dataset_info['name']] = metrics
            all_time_info[dataset_info['name']] = time_info
        except Exception as e:
            print(f"处理数据集 {dataset_info['name']} 时出错: {str(e)}")
            continue
    
    # 保存统一结果
    unified_results = {
        'model_name': 'TimeMixer++',
        'results': all_results,
        'time_info': all_time_info,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(unified_results_dir, 'timemixerplusplus_all_results.json'), 'w') as f:
        json.dump(unified_results, f, indent=4)
    
    print("\n=== TimeMixer++模型所有数据集结果汇总 ===")
    for dataset_name, metrics in all_results.items():
        print(f"\n{dataset_name}:")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE: {metrics['MAE']:.6f}")
        print(f"  R²: {metrics['R2']:.6f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        if dataset_name in all_time_info:
            print(f"  训练时间: {all_time_info[dataset_name]['training_time']:.2f}秒")
    
    print(f"\n所有结果已保存到: {unified_results_dir}")

if __name__ == "__main__":
    main()