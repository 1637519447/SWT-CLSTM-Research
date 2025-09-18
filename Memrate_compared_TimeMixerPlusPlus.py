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
        'results_dir': 'h:\\work\\alibaba_mem_timemixerplusplus_results_30s\\'
    },
    {
        'name': 'Google_5m',
        'file': 'h:\\work\\Google_mem_util_aggregated_5m.csv',
        'results_dir': 'h:\\work\\google_mem_timemixerplusplus_results_5m\\'
    }
]

# 创建统一的结果目录
unified_results_dir = 'h:\\work\\Mem_timemixerplusplus_unified_results\\'
os.makedirs(unified_results_dir, exist_ok=True)

# 创建结果目录
for dataset in datasets:
    os.makedirs(dataset['results_dir'], exist_ok=True)

def create_dataset(data, look_back=60, for_prediction=False):
    """创建时间序列数据集"""
    X, y = [], []
    if for_prediction:
        # 如果是用于预测，保留完整的输入序列
        for i in range(len(data) - look_back + 1):
            X.append(data[i:(i + look_back), 0])
            if i < len(data) - look_back:
                y.append(data[i + look_back, 0])
            else:
                y.append(None)
    else:
        # 如果不是用于预测，按照LSTM的逻辑
        for i in range(len(data) - look_back - 1):
            X.append(data[i:(i + look_back), 0])
            y.append(data[i + look_back, 0])
    
    if for_prediction:
        return np.array(X), None
    else:
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
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            # 加载最佳模型
            model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pt')))
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # 计算训练时间
    training_time = time.time() - training_start_time
    
    return train_losses, val_losses, training_time

def process_dataset(dataset_info):
    """处理单个数据集"""
    dataset_name = dataset_info['name']
    file_path = dataset_info['file']
    results_dir = dataset_info['results_dir']
    
    print(f"\n{'='*50}")
    print(f"处理数据集: {dataset_name} (文件: {file_path})")
    print(f"{'='*50}\n")
    
    # 加载数据
    try:
        data = np.loadtxt(file_path, delimiter=' ')
        data = data[data != 0]  # 移除零值
    except:
        print(f"无法加载文件 {file_path}，使用随机数据进行测试")
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
    
    # 确保数据长度为2的幂次方
    power = int(np.ceil(np.log2(len(smoothed_data))))
    padded_length = 2**power
    
    if len(smoothed_data) != padded_length:
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
    
    # 数据归一化 - 分别对训练集和测试集进行归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train.reshape(-1, 1))
    scaled_test = scaler.transform(test.reshape(-1, 1))
    
    # 保存scaler
    dump(scaler, os.path.join(results_dir, 'scaler.joblib'))
    
    # 设置参数
    look_back = 70
    epochs = 50
    batch_size = 16
    learning_rate = 0.0008
    
    # 创建数据集
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
    
    # 分割训练集为训练和验证
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = TimeMixerPlusPlusModel(
        seq_len=look_back,
        pred_len=1,
        d_model=128,
        n_blocks=2,
        num_scales=3
    )
    
    # 计算模型复杂度
    sample_input = torch.randn(1, look_back)
    macs, params = calculate_model_complexity(model, sample_input)
    print(f"模型参数量: {params}")
    print(f"计算复杂度: {macs} MACs")
    
    # 定义损失函数和优化器
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8, verbose=True)
    
    # 训练模型
    train_losses, val_losses, training_time = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, results_dir
    )
    
    # 预测
    model.eval()
    with torch.no_grad():
        train_predict = []
        for batch_x, _ in DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size):
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu().numpy()
            train_predict.extend(pred)
        
        test_predict = []
        for batch_x, _ in DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size):
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu().numpy()
            test_predict.extend(pred)
    
    train_predict = np.array(train_predict).reshape(-1, 1)
    test_predict = np.array(test_predict).reshape(-1, 1)
    
    # 反向转换预测值
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 计算评估指标
    train_mse = mean_squared_error(y_train_original, train_predict)
    train_rmse = math.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train_original, train_predict)
    train_r2 = r2_score(y_train_original, train_predict)
    
    test_mse = mean_squared_error(y_test_original, test_predict)
    test_rmse = math.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_original, test_predict)
    test_r2 = r2_score(y_test_original, test_predict)
    
    # 计算MAPE
    epsilon = 1e-10
    train_mape = np.mean(np.abs((y_train_original - train_predict) / (y_train_original + epsilon))) * 100
    test_mape = np.mean(np.abs((y_test_original - test_predict) / (y_test_original + epsilon))) * 100
    
    # 计算不同预测步长的RMSE
    max_steps = 90 if '30s' in dataset_name else 60 if '5m' in dataset_name else 30
        
    train_step_rmse = calculate_step_rmse(y_train_original, train_predict, max_steps)
    test_step_rmse = calculate_step_rmse(y_test_original, test_predict, max_steps)
    
    # 保存不同步长的RMSE
    np.save(os.path.join(results_dir, 'train_step_rmse.npy'), np.array(train_step_rmse))
    np.save(os.path.join(results_dir, 'test_step_rmse.npy'), np.array(test_step_rmse))
    
    # 同时保存到统一目录
    np.save(os.path.join(unified_results_dir, f'TimeMixerPlusPlus_{dataset_name}_test.npy'), test_predict)
    np.save(os.path.join(unified_results_dir, f'TimeMixerPlusPlus_test_ground_truth_{dataset_name}.npy'), y_test_original)
    np.save(os.path.join(unified_results_dir, f'TimeMixerPlusPlus_{dataset_name}_test_step_rmse.npy'), np.array(test_step_rmse))
    
    # 保存结果
    np.save(os.path.join(results_dir, 'train_predict.npy'), train_predict)
    np.save(os.path.join(results_dir, 'test_predict.npy'), test_predict)
    np.save(os.path.join(results_dir, 'train_ground_truth.npy'), y_train_original)
    np.save(os.path.join(results_dir, 'test_ground_truth.npy'), y_test_original)
    np.save(os.path.join(results_dir, 'train_step_rmse.npy'), np.array(train_step_rmse))
    np.save(os.path.join(results_dir, 'test_step_rmse.npy'), np.array(test_step_rmse))
    
    # 同时保存到统一目录
    np.save(os.path.join(unified_results_dir, f'TimeMixerPlusPlus_{dataset_name}_test.npy'), test_predict)
    np.save(os.path.join(unified_results_dir, f'TimeMixerPlusPlus_test_ground_truth_{dataset_name}.npy'), y_test_original)
    np.save(os.path.join(unified_results_dir, f'TimeMixerPlusPlus_{dataset_name}_test_step_rmse.npy'), np.array(test_step_rmse))
    
    # 绘制不同预测步长的RMSE图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(test_step_rmse) + 1), test_step_rmse, marker='o', linestyle='-', linewidth=2, markersize=4)
    
    # 设置图表标题和标签
    if '30s' in dataset_name:
        plt.xlabel('预测步长 (秒)')
        plt.title('TimeMixer++ 30秒级内存利用率预测不同步长RMSE')
    elif '5m' in dataset_name:
        plt.xlabel('预测步长 (分钟)')
        plt.title('TimeMixer++ 5分钟级内存利用率预测不同步长RMSE')
    else:
        plt.xlabel('预测步长')
        plt.title(f'TimeMixer++ {dataset_name} 内存利用率预测不同步长RMSE')
    
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
    plt.title(f'TimeMixer++ {dataset_name} 内存利用率预测结果对比 (前{display_range}个点)')
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
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', alpha=0.8)
    plt.plot(val_losses, label='验证损失', alpha=0.8)
    plt.title(f'TimeMixer++ {dataset_name} 训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存详细的评估指标
    metrics = {
        'dataset_name': dataset_name,
        'model_name': 'TimeMixer++',
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
            'parameters': params,
            'macs': macs
        },
        'training_time_seconds': float(training_time),
        'hyperparameters': {
            'look_back': look_back,
            'd_model': 128,
            'n_blocks': 2,
            'num_scales': 3,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
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
    print(f"模型参数量: {params}")
    print(f"计算复杂度: {macs} MACs")
    print(f"训练时间: {training_time:.2f} 秒")
    
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
        'model_name': 'TimeMixer++',
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
            print(f"  训练时间: {result['training_time_seconds']:.2f} 秒")