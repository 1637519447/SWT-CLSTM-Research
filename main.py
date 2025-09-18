# 可视化性能比较
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# 读入不同模型测试集/预测值
test_y = np.load('test_y.npy')
train_y = np.load('train_y.npy')
CPU_test_y = np.load('CPUrate_test_y.npy')
CPU_train_y = np.load('CPUrate_train_y.npy')
Mem_test_y = np.load('Mem_test_y.npy')
Mem_train_y = np.load('Mem_train_y.npy')

# SWD-CLSTM
pre_test_Main = np.load('predicted_test_Main_PyTorch.npy')
pre_train_Main = np.load('predicted_train_Main_PyTorch.npy')
CPU_pre_test_Main = np.load('CPUrate_predicted_test_Main_PyTorch_30s.npy')
CPU_pre_train_Main = np.load('CPUrate_predicted_train_Main_PyTorch_30s.npy')
Mem_pre_test_Main = np.load('Mem_predicted_test_Main.npy')
Mem_pre_train_Main = np.load('Mem_predicted_train_Main.npy')


# SG-LSTM
pre_test_LSTM = np.load('predicted_test_LSTM.npy')
pre_train_LSTM = np.load('predicted_train_LSTM.npy')
CPU_pre_test_LSTM = np.load('CPUrate_predicted_test_LSTM.npy')
CPU_pre_train_LSTM = np.load('CPUrate_predicted_train_LSTM.npy')
Mem_pre_test_LSTM = np.load('Memrate_predicted_test_LSTM.npy')
Mem_pre_train_LSTM = np.load('Memrate_predicted_train_LSTM.npy')

# SG-1DCNN
pre_test_1DCNN = np.load('predicted_test_1DCNN.npy')[:-1]
pre_train_1DCNN = np.load('predicted_train_1DCNN.npy')[:-1]
CPU_pre_test_1DCNN = np.load('CPUrate_predicted_test_1DCNN.npy')[:-1]
CPU_pre_train_1DCNN = np.load('CPUrate_predicted_train_1DCNN.npy')[:-1]
Mem_pre_test_1DCNN = np.load('Memrate_predicted_test_1DCNN.npy')[:-1]
Mem_pre_train_1DCNN = np.load('Memrate_predicted_train_1DCNN.npy')[:-1]

# # SG-ARIMA
# pre_test_ARIMA = np.load('predicted_test_ARIMA.npy')[50:]
# pre_train_ARIMA = np.load('predicted_train_ARIMA.npy')[50:]
# CPU_pre_test_ARIMA = np.load('CPUrate_predicted_test_ARIMA.npy')[50:]
# CPU_pre_train_ARIMA = np.load('CPUrate_predicted_train_ARIMA.npy')[50:]
# Mem_pre_test_ARIMA = np.load('Memrate_predicted_test_ARIMA.npy')[50:]
# Mem_pre_train_ARIMA = np.load('Memrate_predicted_train_ARIMA.npy')[50:]

# Bi-LSTM
pre_test_BiLSTM = np.load('predicted_test_BiLSTM.npy')
pre_train_BiLSTM = np.load('predicted_train_BiLSTM.npy')
CPU_pre_test_BiLSTM = np.load('CPUrate_predicted_test_BiLSTM.npy')
CPU_pre_train_BiLSTM = np.load('CPUrate_predicted_train_BiLSTM.npy')
Mem_pre_test_BiLSTM = np.load('Memrate_predicted_test_BiLSTM.npy')
Mem_pre_train_BiLSTM = np.load('Memrate_predicted_train_BiLSTM.npy')

# 计算决定系数 (R^2) 和 对数均方根误差 (log RMSE)
def calculate_metrics(y_true, y_pred):
    # R^2决定系数
    r2 = r2_score(y_true, y_pred)
    # 均方根误差（RMSE）
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    # 对数均方根误差 (log RMSE)
    log_rmse = math.log(rmse + 1e-10)  # 防止log(0)错误
    return r2, log_rmse

# SWD-CLSTM
r2_Main, log_rmse_Main = calculate_metrics(test_y, pre_test_Main)
r2_CPU_Main, log_rmse_CPU_Main = calculate_metrics(CPU_test_y, CPU_pre_test_Main)
r2_Mem_Main, log_rmse_Mem_Main = calculate_metrics(Mem_test_y, Mem_pre_test_Main)

# SG-LSTM
r2_LSTM, log_rmse_LSTM = calculate_metrics(test_y, pre_test_LSTM)
r2_CPU_LSTM, log_rmse_CPU_LSTM = calculate_metrics(CPU_test_y, CPU_pre_test_LSTM)
r2_Mem_LSTM, log_rmse_Mem_LSTM = calculate_metrics(Mem_test_y, Mem_pre_test_LSTM)

# SG-1DCNN
r2_1DCNN, log_rmse_1DCNN = calculate_metrics(test_y, pre_test_1DCNN)
r2_CPU_1DCNN, log_rmse_CPU_1DCNN = calculate_metrics(CPU_test_y, CPU_pre_test_1DCNN)
r2_Mem_1DCNN, log_rmse_Mem_1DCNN = calculate_metrics(Mem_test_y, Mem_pre_test_1DCNN)

# Bi-LSTM
r2_BiLSTM, log_rmse_BiLSTM = calculate_metrics(test_y, pre_test_BiLSTM)
r2_CPU_BiLSTM, log_rmse_CPU_BiLSTM = calculate_metrics(CPU_test_y, CPU_pre_test_BiLSTM)
r2_Mem_BiLSTM, log_rmse_Mem_BiLSTM = calculate_metrics(Mem_test_y, Mem_pre_test_BiLSTM)

# 输出结果
print("SWD-CLSTM:")
print(f"  R2: {r2_Main}, Log RMSE: {log_rmse_Main}")
print(f"  CPU R2: {r2_CPU_Main}, Log RMSE: {log_rmse_CPU_Main}")
print(f"  Mem R2: {r2_Mem_Main}, Log RMSE: {log_rmse_Mem_Main}")

print("\nSG-LSTM:")
print(f"  R2: {r2_LSTM}, Log RMSE: {log_rmse_LSTM}")
print(f"  CPU R2: {r2_CPU_LSTM}, Log RMSE: {log_rmse_CPU_LSTM}")
print(f"  Mem R2: {r2_Mem_LSTM}, Log RMSE: {log_rmse_Mem_LSTM}")

print("\nSG-1DCNN:")
print(f"  R2: {r2_1DCNN}, Log RMSE: {log_rmse_1DCNN}")
print(f"  CPU R2: {r2_CPU_1DCNN}, Log RMSE: {log_rmse_CPU_1DCNN}")
print(f"  Mem R2: {r2_Mem_1DCNN}, Log RMSE: {log_rmse_Mem_1DCNN}")

print("\nBi-LSTM:")
print(f"  R2: {r2_BiLSTM}, Log RMSE: {log_rmse_BiLSTM}")
print(f"  CPU R2: {r2_CPU_BiLSTM}, Log RMSE: {log_rmse_CPU_BiLSTM}")
print(f"  Mem R2: {r2_Mem_BiLSTM}, Log RMSE: {log_rmse_Mem_BiLSTM}")


# 可视化预测结果
def draw_taskevents():
    plt.figure(figsize=(15, 8))

    # Subplot 1
    plt.subplot(221)
    plt.plot(test_y, label='Ground Truth of Test')
    plt.plot(pre_test_Main, label='Main', linestyle='-', linewidth=0.9)
    plt.title('Main model on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(30s interval)')
    plt.ylabel('Task Events')

    # Subplot 2
    plt.subplot(222)
    plt.plot(test_y, label='Ground Truth of Test')
    plt.plot(pre_test_LSTM, label='LSTM', linestyle='-', linewidth=0.9)
    plt.title('SG-LSTM on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(30s interval)')
    plt.ylabel('Task Events')

    # Subplot 3
    plt.subplot(223)
    plt.plot(test_y, label='Ground Truth of Test')
    plt.plot(pre_test_1DCNN, label='1DCNN', linestyle='-', linewidth=0.9)
    plt.title('SG-1DCNN on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(30s interval)')
    plt.ylabel('Task Events')

    # Subplot 4
    plt.subplot(224)
    plt.plot(test_y, label='Ground Truth of Test')
    plt.plot(pre_test_BiLSTM, label='BiLSTM', linestyle='-', linewidth=0.9)
    plt.title('SG-BiLSTM on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(30s interval)')
    plt.ylabel('Task Events')


    plt.tight_layout()
    plt.savefig('Compared_task_events.png')
    plt.show()

def draw_CPU():
    plt.figure(figsize=(15, 8))

    # Subplot 1
    plt.subplot(221)
    plt.plot(CPU_test_y, label='Ground Truth of Test')
    plt.plot(CPU_pre_test_Main, label='Main', linestyle='-', linewidth=0.9)
    plt.title('Main model (CPU rate) on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(5-min interval)')
    plt.ylabel('CPU usage(%)')

    # Subplot 2
    plt.subplot(222)
    plt.plot(CPU_test_y, label='Ground Truth of Test')
    plt.plot(CPU_pre_test_LSTM, label='LSTM', linestyle='-', linewidth=0.9)
    plt.title('SG-LSTM (CPU rate) on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(5-min interval)')
    plt.ylabel('CPU usage(%)')

    # Subplot 3
    plt.subplot(223)
    plt.plot(CPU_test_y, label='Ground Truth of Test')
    plt.plot(CPU_pre_test_1DCNN, label='1DCNN', linestyle='-', linewidth=0.9)
    plt.title('SG-1DCNN (CPU rate) on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(5-min interval)')
    plt.ylabel('CPU usage(%)')

    # Subplot 4
    plt.subplot(224)
    plt.plot(CPU_test_y, label='Ground Truth of Test')
    plt.plot(CPU_pre_test_BiLSTM, label='BiLSTM', linestyle='-', linewidth=0.9)
    plt.title('SG-BiLSTM (CPU rate) on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(5-min interval)')
    plt.ylabel('CPU usage(%)')


    plt.tight_layout()
    plt.savefig('Compared_CPU_rate.png')
    plt.show()

def draw_Mem():
    plt.figure(figsize=(15, 8))

    # Subplot 1
    plt.subplot(221)
    plt.plot(Mem_test_y, label='Ground Truth of Test')
    plt.plot(Mem_pre_test_Main, label='Main', linestyle='-', linewidth=0.9)
    plt.title('Main model (Mem rate) on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(5-min interval)')
    plt.ylabel('Mem usage(%)')

    # Subplot 2
    plt.subplot(222)
    plt.plot(Mem_test_y, label='Ground Truth of Test')
    plt.plot(Mem_pre_test_LSTM, label='LSTM', linestyle='-', linewidth=0.9)
    plt.title('SG-LSTM (Mem rate) on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(5-min interval)')
    plt.ylabel('Mem usage(%)')

    # Subplot 3
    plt.subplot(223)
    plt.plot(Mem_test_y, label='Ground Truth of Test')
    plt.plot(Mem_pre_test_1DCNN, label='1DCNN', linestyle='-', linewidth=0.9)
    plt.title('SG-1DCNN (Mem rate) on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(5-min interval)')
    plt.ylabel('Mem usage(%)')

    # Subplot 4
    plt.subplot(224)
    plt.plot(Mem_test_y, label='Ground Truth of Test')
    plt.plot(Mem_pre_test_BiLSTM, label='BiLSTM', linestyle='-', linewidth=0.9)
    plt.title('SG-BiLSTM (Mem rate) on test')
    plt.legend(loc='upper left')
    plt.xlabel('Time(5-min interval)')
    plt.ylabel('Mem usage(%)')


    plt.tight_layout()
    plt.savefig('Compared_Mem_rate.png')
    plt.show()

draw_taskevents()
draw_CPU()
draw_Mem()