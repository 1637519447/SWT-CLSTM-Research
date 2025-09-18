# SWD-CLSTM参数影响实验
# 探究对比学习中不同参数组合对模型性能的影响
# 测试不同的低频噪声强度、高频噪声强度和温度参数

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
import itertools
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 忽略所有警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置matplotlib参数，解决中文显示问题
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rcParams['figure.max_open_warning'] = 50  # 避免图形过多警告
matplotlib.use('Agg')  # 确保使用Agg后端，避免中文显示问题

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"当前CUDA版本: {torch.version.cuda}")
    print(f"当前PyTorch版本: {torch.__version__}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到GPU，将使用CPU进行训练")

# 创建实验结果目录
experiment_dir = 'h:\\work\\param_experiment_results\\'
os.makedirs(experiment_dir, exist_ok=True)

# 设置实验参数
# 低频噪声强度参数范围
low_freq_noise_strengths = [0.02, 0.035, 0.05]
# low_freq_noise_strengths = [0.02, 0.06, 0.1, 0.2]
# 高频噪声强度参数范围
high_freq_noise_strengths = [0.7, 0.85, 1]
# high_freq_noise_strengths = [0.35, 0.7, 1.3, 2]

# 温度参数范围
temperatures = [4.5, 5, 5.5, 6]

# 固定的训练参数
look_back = 70
epochs = 10
batch_size = 16
dataset_name = '30s'  # 使用1h数据集进行实验
file_path = 'memory_rate_30s_257395954.csv'

# 准备LSTM模型训练数据
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

# 数据增强和对比学习相关函数 - 参数化版本
def generate_augmented_samples(x, low_freq_strength, high_freq_strength, augmentation_type='general'):
    batch_size, seq_len, features = x.shape

    # 根据频段类型调整增强策略
    if augmentation_type == 'low_freq':
        noise_strength = low_freq_strength
    elif augmentation_type == 'high_freq':
        noise_strength = high_freq_strength
    else:
        noise_strength = (low_freq_strength + high_freq_strength) / 2

    # 随机噪声增强
    noise = torch.randn_like(x) * noise_strength
    augmented = x + noise

    # 添加时间掩码增强
    mask_length = max(1, int(seq_len * 0.1))
    start_idx = torch.randint(0, seq_len - mask_length + 1, (batch_size,))

    for b in range(batch_size):
        start = start_idx[b]
        augmented[b, start:start + mask_length, :] = 0

    return augmented

def contrastive_loss(features, augmented_features, temperature):
    batch_size = features.shape[0]

    # 归一化特征向量
    features = nn.functional.normalize(features, dim=1)
    augmented_features = nn.functional.normalize(augmented_features, dim=1)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features, augmented_features.T) / temperature

    # 正样本对的索引（对角线元素）
    positive_pairs = torch.arange(batch_size).to(device)

    # 计算对比损失（InfoNCE损失）
    return nn.CrossEntropyLoss()(similarity_matrix, positive_pairs)

# 定义CNN-LSTM模型
class CNNLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size1=80, hidden_size2=60):
        super(CNNLSTM, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=90, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=1)

        # LSTM层
        self.lstm1 = nn.LSTM(input_size=90, hidden_size=hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size2, 1)

    def extract_features(self, x):
        # 调整形状用于卷积
        x = x.permute(0, 2, 1)

        # 卷积层
        x = self.maxpool(self.relu(self.conv1(x)))

        # 调整回LSTM所需的形状
        x = x.permute(0, 2, 1)

        # LSTM层
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # 返回最后一个时间步的特征表示
        return x[:, -1, :]

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc(features), features

# 参数化训练函数
def train_model_with_params(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, 
                           low_freq_strength, high_freq_strength, temperature, results_dir):
    model.to(device)
    best_val_loss = float('inf')
    
    # 固定对比学习权重
    contrastive_weight = 0.005
    
    # 记录训练时间和损失
    train_start_time = time.time()
    epoch_times = []
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_pred_loss = 0.0
        train_contr_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [训练]")

        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # 生成低频和高频增强样本
            low_freq_inputs = generate_augmented_samples(
                inputs, low_freq_strength, high_freq_strength, augmentation_type='low_freq').to(device)
            high_freq_inputs = generate_augmented_samples(
                inputs, low_freq_strength, high_freq_strength, augmentation_type='high_freq').to(device)

            # 前向传播
            outputs, features = model(inputs)
            low_freq_outputs, low_freq_features = model(low_freq_inputs)
            high_freq_outputs, high_freq_features = model(high_freq_inputs)

            # 计算损失
            pred_loss = criterion(outputs, targets)
            
            # 计算低频和高频对比损失
            low_freq_contr_loss = contrastive_loss(features, low_freq_features, temperature)
            high_freq_contr_loss = contrastive_loss(features, high_freq_features, temperature)
            
            # 总对比损失为两者平均
            contr_loss = (low_freq_contr_loss + high_freq_contr_loss) / 2
            
            # 总损失
            loss = pred_loss + contrastive_weight * contr_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pred_loss += pred_loss.item()
            train_contr_loss += contr_loss.item()
            
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'pred_loss': f"{pred_loss.item():.4f}",
                'contr_loss': f"{contr_loss.item():.4f}"
            })

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [验证]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_pred_loss = train_pred_loss / len(train_loader)
        avg_train_contr_loss = train_contr_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 记录每个epoch的时间
        epoch_time_ms = (time.time() - epoch_start_time) * 1000
        epoch_times.append(epoch_time_ms)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} (Pred: {avg_train_pred_loss:.4f}, "
              f"Contr: {avg_train_contr_loss:.4f}), Val Loss: {avg_val_loss:.4f}, Time: {epoch_time_ms:.2f} ms")

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))

    # 计算训练时间信息
    time_info = {
        'total_train_time_ms': (time.time() - train_start_time) * 1000,
        'avg_epoch_time_ms': sum(epoch_times) / len(epoch_times),
        'epoch_times_ms': epoch_times,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss)
    }

    return model, time_info

# 评估模型性能
def evaluate_model(model, X_test, y_test, scaler):
    model.eval()

    # 转换为PyTorch张量并预测
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        y_pred, _ = model(X_test_tensor)
        y_pred = y_pred.cpu().numpy()

    # 反归一化预测结果
    y_pred = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 计算评估指标
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }

    return metrics

# 主实验函数
def run_parameter_experiment():
    print(f"\n{'=' * 50}")
    print(f"开始参数影响实验")
    print(f"{'=' * 50}\n")
    
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
        smoothed_data = savgol_filter(data, window_length=window_length, polyorder=min(2, window_length - 1))

    # 确保数据长度为2的幂次方
    power = int(np.ceil(np.log2(len(smoothed_data))))
    padded_length = 2 ** power

    if len(smoothed_data) != padded_length:
        pad_width = padded_length - len(smoothed_data)
        smoothed_data = np.pad(smoothed_data, (0, pad_width), mode='symmetric')
        print(f"数据长度已填充至 {padded_length} (2^{power})")
    else:
        print(f"数据长度已经是2的幂次方: {padded_length}")

    # 分割数据为训练集和测试集
    train_size = int(len(smoothed_data) * 0.8)
    train, test = smoothed_data[:train_size], smoothed_data[train_size:]

    # 小波分解
    wavelet_type = 'db4'
    level = 1
    coeffs = pywt.swt(smoothed_data, wavelet_type, level=level)
    
    # 获取近似系数(ca)用于实验
    ca = coeffs[0][0]
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 分割训练集和测试集
    ca_train = ca[:train_size]
    ca_test = ca[train_size:]
    
    # 归一化
    ca_train_scaled = scaler.fit_transform(ca_train.reshape(-1, 1))
    ca_test_scaled = scaler.transform(ca_test.reshape(-1, 1))
    
    # 创建数据集
    X_train, y_train = create_dataset(ca_train_scaled, look_back)
    X_test, y_test = create_dataset(ca_test_scaled, look_back)
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 存储实验结果
    experiment_results = []
    
    # 创建参数组合
    param_combinations = list(itertools.product(low_freq_noise_strengths, high_freq_noise_strengths, temperatures))
    total_experiments = len(param_combinations)
    
    print(f"将进行 {total_experiments} 组参数实验")
    
    # 对每组参数进行实验
    for idx, (low_freq, high_freq, temp) in enumerate(param_combinations):
        print(f"\n实验 {idx+1}/{total_experiments}:")
        print(f"低频噪声强度: {low_freq}, 高频噪声强度: {high_freq}, 温度: {temp}")
        
        # 创建实验子目录
        experiment_subdir = os.path.join(experiment_dir, f"exp_low{low_freq}_high{high_freq}_temp{temp}")
        os.makedirs(experiment_subdir, exist_ok=True)
        
        # 初始化模型和训练
        model = CNNLSTM().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=0.00005)
        
        # 训练模型
        model, train_info = train_model_with_params(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            epochs, device, low_freq, high_freq, temp, experiment_subdir
        )
        
        # 评估模型
        metrics = evaluate_model(model, X_test, y_test, scaler)
        
        # 合并参数和结果
        result = {
            'low_freq_strength': low_freq,
            'high_freq_strength': high_freq,
            'temperature': temp,
            'best_val_loss': train_info['best_val_loss'],
            'train_time_ms': train_info['total_train_time_ms'],
            **metrics
        }
        
        experiment_results.append(result)
        
        # 保存当前实验结果
        with open(os.path.join(experiment_subdir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        
        # 绘制训练曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_info['train_losses'], label='训练损失')
        plt.plot(train_info['val_losses'], label='验证损失')
        plt.title(f'训练曲线 (低频={low_freq}, 高频={high_freq}, 温度={temp})')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(experiment_subdir, 'training_curve.png'))
        plt.close()
    
    # 保存所有实验结果
    with open(os.path.join(experiment_dir, 'all_experiment_results.json'), 'w') as f:
        json.dump(experiment_results, f, indent=4)
    
    return experiment_results

# 可视化实验结果
def visualize_experiment_results(results):
    print("\n开始可视化实验结果...")
    
    # 提取唯一的参数值
    unique_low_freq = sorted(list(set([r['low_freq_strength'] for r in results])))
    unique_high_freq = sorted(list(set([r['high_freq_strength'] for r in results])))
    unique_temp = sorted(list(set([r['temperature'] for r in results])))
    
    # 创建结果矩阵
    rmse_matrix = np.zeros((len(unique_low_freq), len(unique_high_freq), len(unique_temp)))
    r2_matrix = np.zeros((len(unique_low_freq), len(unique_high_freq), len(unique_temp)))
    
    # 填充结果矩阵
    for result in results:
        low_idx = unique_low_freq.index(result['low_freq_strength'])
        high_idx = unique_high_freq.index(result['high_freq_strength'])
        temp_idx = unique_temp.index(result['temperature'])
        
        rmse_matrix[low_idx, high_idx, temp_idx] = result['rmse']
        r2_matrix[low_idx, high_idx, temp_idx] = result['r2']
    
    # 1. 绘制不同温度下的RMSE热力图
    for temp_idx, temp in enumerate(unique_temp):
        plt.figure(figsize=(10, 8))
        plt.imshow(rmse_matrix[:, :, temp_idx], cmap='viridis', origin='lower', aspect='auto',
                  extent=[min(unique_high_freq), max(unique_high_freq), min(unique_low_freq), max(unique_low_freq)])
        plt.colorbar(label='RMSE')
        plt.title(f'RMSE热力图 (温度={temp})')
        plt.xlabel('高频噪声强度')
        plt.ylabel('低频噪声强度')
        
        # 添加数值标签
        for i in range(len(unique_low_freq)):
            for j in range(len(unique_high_freq)):
                plt.text(unique_high_freq[j], unique_low_freq[i], 
                        f'{rmse_matrix[i, j, temp_idx]:.4f}',
                        ha='center', va='center', color='white', fontsize=8)
        
        plt.savefig(os.path.join(experiment_dir, f'rmse_heatmap_temp{temp}.png'))
        plt.close()
    
    # 2. 绘制不同温度下的R²热力图
    for temp_idx, temp in enumerate(unique_temp):
        plt.figure(figsize=(10, 8))
        plt.imshow(r2_matrix[:, :, temp_idx], cmap='plasma', origin='lower', aspect='auto',
                  extent=[min(unique_high_freq), max(unique_high_freq), min(unique_low_freq), max(unique_low_freq)])
        plt.colorbar(label='R²')
        plt.title(f'R²热力图 (温度={temp})')
        plt.xlabel('高频噪声强度')
        plt.ylabel('低频噪声强度')
        
        # 添加数值标签
        for i in range(len(unique_low_freq)):
            for j in range(len(unique_high_freq)):
                plt.text(unique_high_freq[j], unique_low_freq[i], 
                        f'{r2_matrix[i, j, temp_idx]:.4f}',
                        ha='center', va='center', color='white', fontsize=8)
        
        plt.savefig(os.path.join(experiment_dir, f'r2_heatmap_temp{temp}.png'))
        plt.close()
    
    # 3. 绘制固定高频噪声强度下，低频噪声强度与温度的关系图
    for high_idx, high_freq in enumerate(unique_high_freq):
        plt.figure(figsize=(12, 8))
        
        for temp_idx, temp in enumerate(unique_temp):
            rmse_values = rmse_matrix[:, high_idx, temp_idx]
            plt.plot(unique_low_freq, rmse_values, marker='o', label=f'温度={temp}')
        
        plt.title(f'低频噪声强度对RMSE的影响 (高频噪声强度={high_freq})')
        plt.xlabel('低频噪声强度')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(experiment_dir, f'low_freq_vs_rmse_high{high_freq}.png'))
        plt.close()
    
    # 4. 绘制固定低频噪声强度下，高频噪声强度与温度的关系图
    for low_idx, low_freq in enumerate(unique_low_freq):
        plt.figure(figsize=(12, 8))
        
        for temp_idx, temp in enumerate(unique_temp):
            rmse_values = rmse_matrix[low_idx, :, temp_idx]
            plt.plot(unique_high_freq, rmse_values, marker='o', label=f'温度={temp}')
        
        plt.title(f'高频噪声强度对RMSE的影响 (低频噪声强度={low_freq})')
        plt.xlabel('高频噪声强度')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(experiment_dir, f'high_freq_vs_rmse_low{low_freq}.png'))
        plt.close()
    
    # 5. 绘制3D表面图
    for temp_idx, temp in enumerate(unique_temp):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(unique_high_freq, unique_low_freq)
        Z = rmse_matrix[:, :, temp_idx]
        
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        ax.set_xlabel('高频噪声强度')
        ax.set_ylabel('低频噪声强度')
        ax.set_zlabel('RMSE')
        ax.set_title(f'参数对RMSE的影响 (温度={temp})')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(os.path.join(experiment_dir, f'3d_surface_temp{temp}.png'))
        plt.close()
    
    # 6. 创建最佳参数组合的汇总表格
    # 找出RMSE最小的参数组合
    min_rmse = float('inf')
    best_params = None
    
    for result in results:
        if result['rmse'] < min_rmse:
            min_rmse = result['rmse']
            best_params = result
    
    # 创建HTML汇总报告
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>参数影响实验结果汇总</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .best { background-color: #d4edda; font-weight: bold; }
            .chart-container { margin: 20px 0; }
            .chart-container img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>SWD-CLSTM参数影响实验结果汇总</h1>
        
        <h2>最佳参数组合</h2>
        <table>
            <tr>
                <th>低频噪声强度</th>
                <th>高频噪声强度</th>
                <th>温度</th>
                <th>RMSE</th>
                <th>R²</th>
                <th>MAE</th>
                <th>训练时间(ms)</th>
            </tr>
            <tr class="best">
                <td>{result['high_freq_strength']}</td>
                <td>{result['temperature']}</td>
                <td>{result['rmse']:.4f}</td>
                <td>{result['r2']:.4f}</td>
                <td>{result['mae']:.4f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>可视化结果</h2>
        
        <div class="chart-container">
            <h3>RMSE热力图 (不同温度)</h3>
    """
    
    # 添加热力图
    for temp in unique_temp:
        html_content += f"""
            <div>
                <h4>温度 = {temp}</h4>
                <img src="rmse_heatmap_temp{temp}.png" alt="RMSE热力图 (温度={temp})">
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="chart-container">
            <h3>参数影响曲线</h3>
    """
    
    # 添加参数影响曲线图
    html_content += """
            <div>
                <h4>低频噪声强度对RMSE的影响 (不同高频噪声强度)</h4>
    """
    
    for high_freq in unique_high_freq:
        html_content += f"""
                <img src="low_freq_vs_rmse_high{high_freq}.png" alt="低频噪声强度对RMSE的影响 (高频噪声强度={high_freq})">
        """
    
    html_content += """
            </div>
            
            <div>
                <h4>高频噪声强度对RMSE的影响 (不同低频噪声强度)</h4>
    """
    
    for low_freq in unique_low_freq:
        html_content += f"""
                <img src="high_freq_vs_rmse_low{low_freq}.png" alt="高频噪声强度对RMSE的影响 (低频噪声强度={low_freq})">
        """
    
    html_content += """
            </div>
            
            <div>
                <h4>3D表面图 (不同温度)</h4>
    """
    
    for temp in unique_temp:
        html_content += f"""
                <img src="3d_surface_temp{temp}.png" alt="参数对RMSE的影响 (温度={temp})">
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML报告
    with open(os.path.join(experiment_dir, 'experiment_results_summary.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"实验结果可视化完成，报告已保存到: {os.path.join(experiment_dir, 'experiment_results_summary.html')}")
    
    # 返回最佳参数
    return best_params

# 主函数
if __name__ == "__main__":
    # 运行参数实验
    experiment_results = run_parameter_experiment()
    
    # 可视化实验结果
    best_params = visualize_experiment_results(experiment_results)
    
    print("\n实验完成!")
    print(f"最佳参数组合:")
    print(f"  低频噪声强度: {best_params['low_freq_strength']}")
    print(f"  高频噪声强度: {best_params['high_freq_strength']}")
    print(f"  温度: {best_params['temperature']}")
    print(f"  RMSE: {best_params['rmse']:.4f}")
    print(f"  R²: {best_params['r2']:.4f}")
    
    # 绘制最佳参数组合的特殊图表
    plt.figure(figsize=(15, 10))
    
    # 创建参数空间的3D散点图，突出显示最佳点
    ax = plt.subplot(111, projection='3d')
    
    # 提取所有参数和RMSE值
    low_freqs = [r['low_freq_strength'] for r in experiment_results]
    high_freqs = [r['high_freq_strength'] for r in experiment_results]
    temps = [r['temperature'] for r in experiment_results]
    rmses = [r['rmse'] for r in experiment_results]
    
    # 绘制所有点
    scatter = ax.scatter(low_freqs, high_freqs, temps, c=rmses, cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='k', linewidths=0.5)
    
    # 突出显示最佳点
    ax.scatter([best_params['low_freq_strength']], 
               [best_params['high_freq_strength']], 
               [best_params['temperature']], 
               color='red', s=200, marker='*', edgecolors='k', linewidths=1.5,
               label='最佳参数组合')
    
    # 设置坐标轴标签
    ax.set_xlabel('低频噪声强度')
    ax.set_ylabel('高频噪声强度')
    ax.set_zlabel('温度')
    ax.set_title('参数空间中的RMSE分布与最佳参数组合')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('RMSE')
    
    # 添加图例
    ax.legend()
    
    # 保存图表
    plt.savefig(os.path.join(experiment_dir, 'best_params_visualization.png'))
    plt.close()
    
    print(f"\n所有实验结果和可视化已保存到: {experiment_dir}")

#   1h最佳参数组合:
#   低频噪声强度: 0.005
#   高频噪声强度: 0.6
#   温度: 8
#   RMSE: 0.0754
#   R²: 0.9814


#   30s最佳参数组合:
#   低频噪声强度: 0.05
#   高频噪声强度: 0.85
#   温度: 8
#   RMSE: 0.0790
#   R²: 0.9795

  # 低频噪声强度: 0.035
  # 高频噪声强度: 0.85
  # 温度: 8.5
  # RMSE: 0.1226
  # R²: 0.9868

# 5min
# 最佳参数组合:
#   低频噪声强度: 0.035
#   高频噪声强度: 0.85
#   温度: 7
#   RMSE: 0.0365
#   R²: 0.9965

# Mem 1h
# 实验完成!
# 最佳参数组合:
#   低频噪声强度: 0.035
#   高频噪声强度: 1
#   温度: 7
#   RMSE: 7.5254
#   R²: 0.7766

# Mem 30s
# 实验完成!
# 最佳参数组合:
#   低频噪声强度: 0.035
#   高频噪声强度: 0.7
#   温度: 5.5
#   RMSE: 0.1667
#   R²: 0.9700