# 本体 CLSTM-SSC (PyTorch版本) - 消融实验版本
# 支持多个时间间隔数据集的训练与评估
# 移除了SG滤波器、平稳小波变换(SWT)和对比学习模块，其他模块保持不变

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
datasets = [
    {
        'name': 'Alibaba_30s',
        'file': 'Alibaba_mem_util_aggregated_30s.csv',
        'results_dir': 'h:\\work\\Ablation\\Mem\\SWT-CLSTM-SSC\\mem_results_Alibaba_30s\\'
    },
    {
        'name': 'Google_5m',
        'file': 'Google_mem_util_aggregated_5m.csv',
        'results_dir': 'h:\\work\\Ablation\\Mem\\SWT-CLSTM-SSC\\mem_results_Google_5m\\'
    }
]

# 创建统一的结果目录
unified_results_dir = 'h:\\work\\Ablation\\Mem\\SWT-CLSTM-SSC\\Mem_clstm_unified_results\\'
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

    def forward(self, x):
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
        
        # 获取最后一个时间步的输出
        features = x[:, -1, :]
        
        # 全连接层
        output = self.fc(features)
        
        return output

# 训练函数 - 移除了对比学习相关代码
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, results_dir=''):
    model.to(device)
    best_val_loss = float('inf')
    
    # 记录训练时间
    train_start_time = time.time()
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]")

        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [验证]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 记录每个epoch的时间
        epoch_time_ms = (time.time() - epoch_start_time) * 1000
        epoch_times.append(epoch_time_ms)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time_ms:.2f} ms")

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, f'best_model.pt'))
            print(f"验证损失改善，保存模型...")
    
    # 计算训练时间信息
    time_info = {
        'total_train_time_ms': (time.time() - train_start_time) * 1000,
        'avg_epoch_time_ms': sum(epoch_times) / len(epoch_times),
        'epoch_times_ms': epoch_times
    }
    
    with open(os.path.join(results_dir, f'train_time.json'), 'w') as f:
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
                
            y_true_segment = y_true[step:step+compare_length]
            y_pred_segment = y_pred[step:step+compare_length]
            
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
def evaluate_model(model, X_test, y_test, scaler, results_dir):
    model.eval()
    
    # 转换为PyTorch张量并预测
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
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
    epsilon = 1e-10  # 避免除以零
    mape = np.mean(np.abs((y_test_original - y_pred) / (y_test_original + epsilon))) * 100
    
    # 计算不同预测步长的RMSE
    dataset_name = os.path.basename(os.path.dirname(results_dir))
    max_steps = 90 if '30s' in dataset_name else 60 if '5m' in dataset_name else 30
    step_rmse = calculate_step_rmse(y_test_original, y_pred, max_steps)
    
    # 保存RMSE数据
    np.save(os.path.join(results_dir, f'test_step_rmse.npy'), np.array(step_rmse))
    
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
        'step_rmse': [float(x) for x in step_rmse]
    }
    
    with open(os.path.join(results_dir, f'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 保存预测结果
    np.save(os.path.join(results_dir, f'predictions.npy'), y_pred)
    np.save(os.path.join(results_dir, f'ground_truth.npy'), y_test_original)
    
    # 保存到统一目录
    dataset_name = os.path.basename(os.path.dirname(results_dir)).split('_')[-1]
    np.save(os.path.join(unified_results_dir, f'CLSTM_SSC_{dataset_name}_predictions.npy'), y_pred)
    np.save(os.path.join(unified_results_dir, f'CLSTM_SSC_{dataset_name}_ground_truth.npy'), y_test_original)
    np.save(os.path.join(unified_results_dir, f'CLSTM_SSC_{dataset_name}_test_step_rmse.npy'), np.array(step_rmse))
    
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
    
    # 分割数据为训练集和测试集 - 直接使用原始数据，不进行SG滤波
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    print(f"训练集长度: {len(train)}")
    print(f"测试集长度: {len(test)}")
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))
    test_scaled = scaler.transform(test.reshape(-1, 1))
    
    # 保存scaler
    dump(scaler, os.path.join(results_dir, f'scaler.joblib'))
    
    # 创建数据集
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test, y_test = create_dataset(test_scaled, look_back)
    
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
        epochs, device, results_dir
    )
    
    # 评估模型
    metrics, y_pred, y_test_original = evaluate_model(model, X_test, y_test, scaler, results_dir)
    
    # 合并训练时间和评估指标
    all_results = {**train_time_info, **metrics}
    
    # 保存所有结果的汇总
    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    return all_results

# 主程序
if __name__ == "__main__":
    # 处理所有数据集
    all_datasets_results = {}
    
    for dataset in datasets:
        print(f"\n开始处理数据集: {dataset['name']}")
        results = process_dataset(dataset)
        all_datasets_results[dataset['name']] = results
    
    # 保存所有数据集的汇总结果
    with open('h:\\work\\Ablation\\Mem\\SWT-CLSTM-SSC\\CLSTM-SSC_all_datasets_results.json', 'w') as f:
        json.dump(all_datasets_results, f, indent=4)
    
    with open(os.path.join(unified_results_dir, 'clstm_ssc_all_datasets_results.json'), 'w') as f:
        json.dump(all_datasets_results, f, indent=4)
    
    print("\n所有数据集处理完成！")
    print(f"结果已保存到各自的目录中，汇总结果保存在: {unified_results_dir}")