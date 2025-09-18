# 本体 SWT-CLSTM-SG (PyTorch版本) - 消融实验版本
# 支持多个时间间隔数据集的训练与评估
# 移除了SG滤波器模块，其他模块保持不变
# 移除了所有绘图功能，仅保留数据文件

import numpy as np
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
import pywt
from thop import profile, clever_format  # 添加thop库用于计算MACs和参数量

# 忽略所有警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
# 定义要处理的数据集列表
datasets = [
    {
        'name': 'Alibaba_30s',
        'file': 'Alibaba_cpu_util_aggregated_30s.csv',
        'results_dir': 'h:\\work\\Ablation\\CPU\\SWT-CLSTM-SG\\cpu_results_Alibaba_30s\\'
    },
    {
        'name': 'Google_5m',
        'file': 'Google_cpu_util_aggregated_5m.csv',
        'results_dir': 'h:\\work\\Ablation\\CPU\\SWT-CLSTM-SG\\cpu_results_Google_5m\\'
    }
]

# 创建统一的结果目录
unified_results_dir = 'h:\\work\\Ablation\\CPU\\SWT-CLSTM-SG\\CPU_swdclstm_unified_results\\'
os.makedirs(unified_results_dir, exist_ok=True)

# 创建结果目录
for dataset in datasets:
    os.makedirs(dataset['results_dir'], exist_ok=True)


# 准备LSTM模型训练数据
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)


# 设置LSTM参数
look_back = 70
epochs = 50
batch_size = 16


# 数据增强和对比学习相关函数
def generate_augmented_samples(x, augmentation_strength=1, augmentation_type='general'):
    batch_size, seq_len, features = x.shape

    # 根据频段类型调整增强策略
    if augmentation_type == 'low_freq':
        noise_strength = augmentation_strength * 0.035
    elif augmentation_type == 'high_freq':
        noise_strength = augmentation_strength * 0.85
    else:
        noise_strength = augmentation_strength

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


def contrastive_loss(features, augmented_features, temperature=7.5):
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
    def __init__(self, input_size=1, hidden_size1=200, hidden_size2=160, hidden_size3=130, hidden_size4=100, hidden_size5=70):
        super(CNNLSTM, self).__init__()
        # 卷积层 - 将out_channels改为128
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        # LSTM层 - 修改第一层的input_size以匹配Conv1d的输出
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=hidden_size2, hidden_size=hidden_size3, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=hidden_size3, hidden_size=hidden_size4, batch_first=True)
        self.lstm5 = nn.LSTM(input_size=hidden_size4, hidden_size=hidden_size5, batch_first=True)
        
        # 全连接层 - 输入维度改为最后一层LSTM的输出维度
        self.fc = nn.Linear(hidden_size5, 1)

    def extract_features(self, x):
        # 调整形状用于卷积
        x = x.permute(0, 2, 1)
        
        # 卷积层
        x = self.maxpool(self.relu(self.conv1(x)))
        
        # 调整回LSTM所需的形状
        x = x.permute(0, 2, 1)
        
        # LSTM层 - 修改为5层
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x, _ = self.lstm5(x)
        
        # 返回最后一个时间步的特征表示
        return x[:, -1, :]

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc(features), features


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, coeff_type='ca',
                results_dir=''):
    model.to(device)
    best_val_loss = float('inf')

    # 根据系数类型设置对比学习权重和增强类型
    contrastive_weight = 0.0007 if coeff_type == 'ca' else 0.0001
    augmentation_type = 'low_freq' if coeff_type == 'ca' else 'high_freq'
    #0.0005   0.0001

    # 记录训练时间
    train_start_time = time.time()
    epoch_times = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [训练-{coeff_type}]")

        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # 生成增强样本
            augmented_inputs = generate_augmented_samples(inputs, augmentation_type=augmentation_type).to(device)

            # 前向传播
            outputs, features = model(inputs)
            augmented_outputs, augmented_features = model(augmented_inputs)

            # 计算损失
            pred_loss = criterion(outputs, targets)
            contr_loss = contrastive_loss(features, augmented_features)
            loss = pred_loss + contrastive_weight * contr_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'pred_loss': f"{pred_loss.item():.4f}",
                                    'contr_loss': f"{contr_loss.item():.4f}"})

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [验证-{coeff_type}]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 记录每个epoch的时间
        epoch_time_ms = (time.time() - epoch_start_time) * 1000
        epoch_times.append(epoch_time_ms)

        print(
            f"Epoch {epoch + 1}/{epochs} - {coeff_type} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time_ms:.2f} ms")

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, f'best_model_{coeff_type}.pt'))
            print(f"验证损失改善，保存模型...")

    # 计算训练时间信息
    time_info = {
        'total_train_time_ms': (time.time() - train_start_time) * 1000,
        'avg_epoch_time_ms': sum(epoch_times) / len(epoch_times),
        'epoch_times_ms': epoch_times
    }

    with open(os.path.join(results_dir, f'train_time_{coeff_type}.json'), 'w', encoding='utf-8') as f:
        json.dump(time_info, f, indent=4)

    return model, time_info


# 计算不同预测步长的RMSE
def calculate_step_rmse(y_true, y_pred, max_steps=30):
    step_rmse = []

    for step in range(1, max_steps + 1):
        if step < len(y_true) and step < len(y_pred):
            compare_length = min(len(y_true) - step, len(y_pred) - step)
            if compare_length <= 0:
                break

            y_true_segment = y_true[step:step + compare_length]
            y_pred_segment = y_pred[step:step + compare_length]

            # 将数据转换为二维数组，如果它们是一维的
            if y_true_segment.ndim == 1:
                y_true_segment = y_true_segment.reshape(-1, 1)
            if y_pred_segment.ndim == 1:
                y_pred_segment = y_pred_segment.reshape(-1, 1)

            rmse = math.sqrt(mean_squared_error(y_true_segment, y_pred_segment))
            step_rmse.append(rmse)
        else:
            break

    return step_rmse


# 评估模型性能
def evaluate_model(model, X_test, y_test, scaler, results_dir, coeff_type='ca'):
    model.eval()

    # 转换为PyTorch张量并预测
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # 计算模型复杂度 (MACs和参数量)
    # 创建一个样本输入用于计算
    sample_input = torch.randn(1, look_back, 1).to(device)
    macs, params = profile(model, inputs=(sample_input,), verbose=False)
    
    # 格式化为更易读的形式
    macs_str, params_str = clever_format([macs, params], "%.3f")

    start_time = time.time()
    with torch.no_grad():
        y_pred, _ = model(X_test_tensor)
        y_pred = y_pred.cpu().numpy()

    prediction_time_ms = (time.time() - start_time) * 1000

    # 反归一化预测结果
    y_pred = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 计算评估指标
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    
    # 计算MAPE (平均绝对百分比误差)
    # 避免除以零，添加一个小的常数
    epsilon = 1e-10
    mape = np.mean(np.abs((y_test_original - y_pred) / (y_test_original + epsilon))) * 100

    # 计算不同预测步长的RMSE
    dataset_name = os.path.basename(os.path.dirname(results_dir))
    max_steps = 90 if '30s' in dataset_name else 60 if '5m' in dataset_name else 30
    step_rmse = calculate_step_rmse(y_test_original, y_pred, max_steps)

    # 保存RMSE数据
    np.save(os.path.join(results_dir, f'test_step_rmse_{coeff_type}.npy'), np.array(step_rmse))

    # 计算Log RMSE
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
        'per_sample_time_ms': float(prediction_time_ms / len(X_test)),
        'step_rmse': [float(x) for x in step_rmse],
        # 添加模型复杂度指标
        'macs': float(macs),
        'macs_readable': macs_str,
        'params': float(params),
        'params_readable': params_str
    }

    with open(os.path.join(results_dir, f'metrics_{coeff_type}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    # 保存预测结果
    np.save(os.path.join(results_dir, f'predictions_{coeff_type}.npy'), y_pred)
    np.save(os.path.join(results_dir, f'ground_truth_{coeff_type}.npy'), y_test_original)

    return metrics, y_pred, y_test_original


# 主函数：处理每个数据集
def process_dataset(dataset_info):
    dataset_name = dataset_info['name']
    file_path = dataset_info['file']
    results_dir = dataset_info['results_dir']

    print(f"\n{'=' * 50}")
    print(f"处理数据集: {dataset_name} (文件: {file_path})")
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

    # 消融实验：移除SG滤波器，直接使用原始数据
    smoothed_data = data.copy()
    print("消融实验：移除SG滤波器，直接使用原始数据")

    # 确保数据长度为2的幂次方，这对SWT很重要
    # 计算最接近的2的幂次方
    power = int(np.ceil(np.log2(len(smoothed_data))))
    padded_length = 2 ** power

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

    # 小波分解 - 使用更适合重构的小波类型和参数
    wavelet_type = 'db4'  # 修改为db4小波
    level = 1  # 保持单级分解

    # 执行平稳小波变换
    coeffs = pywt.swt(smoothed_data, wavelet_type, level=level)

    # 验证重构精度
    # 从系数重构信号
    reconstructed = pywt.iswt(coeffs, wavelet_type)

    # 计算重构误差
    reconstruction_error = np.mean(np.abs(smoothed_data - reconstructed))
    print(f"小波重构误差: {reconstruction_error:.10f}")

    # 如果重构误差过大，尝试其他小波类型
    if reconstruction_error > 1e-10:
        alternative_wavelets = ['db4', 'sym8', 'coif3']
        for alt_wavelet in alternative_wavelets:
            alt_coeffs = pywt.swt(smoothed_data, alt_wavelet, level=level)
            alt_reconstructed = pywt.iswt(alt_coeffs, alt_wavelet)
            alt_error = np.mean(np.abs(smoothed_data - alt_reconstructed))
            print(f"使用 {alt_wavelet} 的重构误差: {alt_error:.10f}")

            if alt_error < reconstruction_error:
                wavelet_type = alt_wavelet
                coeffs = alt_coeffs
                reconstructed = alt_reconstructed
                reconstruction_error = alt_error
                print(f"切换到更精确的小波类型: {wavelet_type}")

    print(f"最终使用的小波类型: {wavelet_type}, 重构误差: {reconstruction_error:.10f}")

    # 存储结果
    all_results = {}
    predictions = {}
    ground_truths = {}

    # 处理每种系数
    for coeff_idx, coeff_type in enumerate(['ca', 'cd']):
        coeff = coeffs[0][coeff_idx]  # 获取对应系数
        print(f"\n处理 {coeff_type} 系数...")

        # 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))

        # 分割训练集和测试集 - 确保与原始数据分割点一致
        coeff_train_size = train_size
        coeff_train = coeff[:coeff_train_size]
        coeff_test = coeff[coeff_train_size:]

        # 归一化
        coeff_train_scaled = scaler.fit_transform(coeff_train.reshape(-1, 1))
        coeff_test_scaled = scaler.transform(coeff_test.reshape(-1, 1))

        # 保存scaler
        dump(scaler, os.path.join(results_dir, f'scaler_{coeff_type}.joblib'))

        # 创建数据集
        X_train, y_train = create_dataset(coeff_train_scaled, look_back)
        X_test, y_test = create_dataset(coeff_test_scaled, look_back)

        # 创建数据加载器
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 初始化模型和训练
        model = CNNLSTM().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=0.00005)

        # 训练模型
        model, train_time_info = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            epochs, device, coeff_type, results_dir
        )

        # 评估模型
        metrics, y_pred, y_test_original = evaluate_model(model, X_test, y_test, scaler, results_dir, coeff_type)

        # 存储预测结果和真实值
        predictions[coeff_type] = y_pred
        ground_truths[coeff_type] = y_test_original

        # 合并训练时间和评估指标
        all_results[coeff_type] = {**train_time_info, **metrics}

    # 重构最终预测结果 - 使用与分解相同的方法确保一致性
    print("\n重构最终预测结果...")

    # 确保预测结果长度一致
    min_length = min(len(predictions['ca']), len(predictions['cd']))

    # 打印调试信息
    print(f"预测系数长度 - ca: {len(predictions['ca'])}, cd: {len(predictions['cd'])}")
    print(f"使用最小长度: {min_length}")

    # 确保两个系数数组长度完全一致
    ca_pred = predictions['ca'][:min_length].reshape(-1)
    cd_pred = predictions['cd'][:min_length].reshape(-1)

    # 再次检查形状
    print(f"重构前系数形状 - ca: {ca_pred.shape}, cd: {cd_pred.shape}")

    # 创建与原始SWT系数格式相同的结构
    pred_coeffs = [(ca_pred, cd_pred)]

    try:
        # 使用iswt进行精确重构
        reconstructed_pred = pywt.iswt(pred_coeffs, wavelet_type)[:min_length]

        # 同样处理真实值
        ca_truth = ground_truths['ca'][:min_length].reshape(-1)
        cd_truth = ground_truths['cd'][:min_length].reshape(-1)
        truth_coeffs = [(ca_truth, cd_truth)]
        reconstructed_truth = pywt.iswt(truth_coeffs, wavelet_type)[:min_length]

        print(f"重构成功! 重构结果形状: {reconstructed_pred.shape}")
    except Exception as e:
        print(f"重构过程中出现错误: {e}")
        print("尝试备用重构方法...")

        # 备用方法：确保长度为2的幂次方
        power = int(np.floor(np.log2(min_length)))
        safe_length = 2 ** power
        print(f"调整为2的幂次方长度: {safe_length}")

        ca_pred = predictions['ca'][:safe_length].reshape(-1)
        cd_pred = predictions['cd'][:safe_length].reshape(-1)
        pred_coeffs = [(ca_pred, cd_pred)]

        ca_truth = ground_truths['ca'][:safe_length].reshape(-1)
        cd_truth = ground_truths['cd'][:safe_length].reshape(-1)
        truth_coeffs = [(ca_truth, cd_truth)]

        # 重新尝试重构
        reconstructed_pred = pywt.iswt(pred_coeffs, wavelet_type)
        reconstructed_truth = pywt.iswt(truth_coeffs, wavelet_type)
        print(f"备用方法重构成功! 重构结果形状: {reconstructed_pred.shape}")

    # 计算重构后的评估指标
    recon_mse = mean_squared_error(reconstructed_truth, reconstructed_pred)
    recon_rmse = math.sqrt(recon_mse)
    recon_mae = mean_absolute_error(reconstructed_truth, reconstructed_pred)
    recon_r2 = r2_score(reconstructed_truth, reconstructed_pred)
    
    # 计算MAPE (平均绝对百分比误差)
    epsilon = 1e-10  # 避免除以零
    recon_mape = np.mean(np.abs((reconstructed_truth - reconstructed_pred) / (reconstructed_truth + epsilon))) * 100

    # 计算不同预测步长的RMSE
    max_steps = 90 if dataset_name == '30s' else 60 if dataset_name == '5m' else 30
    recon_step_rmse = calculate_step_rmse(reconstructed_truth, reconstructed_pred, max_steps)

    # 保存重构数据
    np.save(os.path.join(results_dir, 'reconstructed_test_step_rmse.npy'), np.array(recon_step_rmse))
    np.save(os.path.join(unified_results_dir, f'SWT-CLSTM-SG_{dataset_name}_reconstructed_test.npy'), reconstructed_pred)
    np.save(os.path.join(unified_results_dir, f'SWT-CLSTM-SG_{dataset_name}_reconstructed_test_ground_truth.npy'),
            reconstructed_truth)
    np.save(os.path.join(unified_results_dir, f'SWT-CLSTM-SG_{dataset_name}_reconstructed_test_step_rmse.npy'),
            np.array(recon_step_rmse))

    # 计算Log RMSE
    recon_truth_log = np.log1p(np.maximum(reconstructed_truth, 0))
    recon_pred_log = np.log1p(np.maximum(reconstructed_pred, 0))
    recon_log_rmse = math.sqrt(mean_squared_error(recon_truth_log, recon_pred_log))

    # 计算总体模型复杂度 (两个子模型的总和)
    total_macs = all_results['ca'].get('macs', 0) + all_results['cd'].get('macs', 0)
    total_params = all_results['ca'].get('params', 0) + all_results['cd'].get('params', 0)
    
    # 格式化为更易读的形式
    total_macs_str, total_params_str = clever_format([total_macs, total_params], "%.3f")
    
    # 计算总训练时间 (两个子模型的总和)
    total_train_time_ms = all_results['ca'].get('total_train_time_ms', 0) + all_results['cd'].get('total_train_time_ms', 0)
    
    # 计算总预测时间 (两个子模型的总和)
    total_prediction_time_ms = all_results['ca'].get('prediction_time_ms', 0) + all_results['cd'].get('prediction_time_ms', 0)

    # 保存重构后的评估指标
    recon_metrics = {
        'mse': float(recon_mse),
        'rmse': float(recon_rmse),
        'mae': float(recon_mae),
        'r2': float(recon_r2),
        'log_rmse': float(recon_log_rmse),
        'mape': float(recon_mape),  # 添加MAPE指标
        'step_rmse': [float(x) for x in recon_step_rmse],
        # 添加总体模型复杂度指标
        'total_macs': float(total_macs),
        'total_macs_readable': total_macs_str,
        'total_params': float(total_params),
        'total_params_readable': total_params_str,
        'total_train_time_ms': float(total_train_time_ms),
        'total_train_time_s': float(total_train_time_ms / 1000),
        'total_prediction_time_ms': float(total_prediction_time_ms),
        'avg_prediction_time_per_sample_ms': float(total_prediction_time_ms / min_length)
    }

    # 保存重构结果
    with open(os.path.join(results_dir, 'reconstructed_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(recon_metrics, f, indent=4)

    # 保存模型复杂度和时间信息到单独的文件
    model_complexity_info = {
        'model_name': 'SWT-CLSTM-SG',
        'dataset': dataset_name,
        'total_macs': float(total_macs),
        'total_macs_readable': total_macs_str,
        'total_params': float(total_params),
        'total_params_readable': total_params_str,
        'total_train_time_ms': float(total_train_time_ms),
        'total_train_time_s': float(total_train_time_ms / 1000),
        'total_train_time_min': float(total_train_time_ms / (1000 * 60)),
        'total_prediction_time_ms': float(total_prediction_time_ms),
        'avg_prediction_time_per_sample_ms': float(total_prediction_time_ms / min_length),
        'ca_macs': all_results['ca'].get('macs', 0),
        'ca_macs_readable': all_results['ca'].get('macs_readable', '0'),
        'ca_params': all_results['ca'].get('params', 0),
        'ca_params_readable': all_results['ca'].get('params_readable', '0'),
        'ca_train_time_ms': all_results['ca'].get('total_train_time_ms', 0),
        'ca_prediction_time_ms': all_results['ca'].get('prediction_time_ms', 0),
        'cd_macs': all_results['cd'].get('macs', 0),
        'cd_macs_readable': all_results['cd'].get('macs_readable', '0'),
        'cd_params': all_results['cd'].get('params', 0),
        'cd_params_readable': all_results['cd'].get('params_readable', '0'),
        'cd_train_time_ms': all_results['cd'].get('total_train_time_ms', 0),
        'cd_prediction_time_ms': all_results['cd'].get('prediction_time_ms', 0)
    }
    
    with open(os.path.join(results_dir, 'model_complexity_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_complexity_info, f, indent=4)
    
    # 同时保存到统一结果目录
    with open(os.path.join(unified_results_dir, f'SWT-CLSTM-SG_{dataset_name}_model_complexity_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_complexity_info, f, indent=4)

    np.save(os.path.join(results_dir, 'reconstructed_predictions.npy'), reconstructed_pred)
    np.save(os.path.join(results_dir, 'reconstructed_ground_truth.npy'), reconstructed_truth)

    # 将重构结果添加到总结果中
    all_results['reconstructed'] = recon_metrics

    # 保存所有结果的汇总
    with open(os.path.join(results_dir, 'all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    return all_results

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
    with open('h:\\work\\Ablation\\CPU\\SWT-CLSTM-SG\\SWT-CLSTM-SG_all_datasets_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_datasets_results, f, indent=4)

    with open(os.path.join(unified_results_dir, 'swt-clstm-sg_all_datasets_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_datasets_results, f, indent=4)
        
    # 保存所有数据集的复杂度信息汇总
    with open('h:\\work\\Ablation\\CPU\\SWT-CLSTM-SG\\SWT-CLSTM-SG_all_complexity_info.json', 'w', encoding='utf-8') as f:
        json.dump(all_complexity_info, f, indent=4)
        
    with open(os.path.join(unified_results_dir, 'swt-clstm-sg_all_complexity_info.json'), 'w', encoding='utf-8') as f:
        json.dump(all_complexity_info, f, indent=4)

    print("\n所有数据集处理完成！")
    print(f"结果已保存到各自的目录中，汇总结果保存在: {unified_results_dir}")
    print(f"模型复杂度信息已保存到: {unified_results_dir}/swt-clstm-sg_all_complexity_info.json")