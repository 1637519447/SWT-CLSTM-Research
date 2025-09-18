import numpy as np

# 数据目录
data_dir1 = "h:\\work\\Compared_data\\CPU\\"
data_dir2 = "h:\\work\\Compared_data\\Mem\\"


# 第一组：CPU
SWT_clstm_30s = np.load(data_dir1 + "SWT_CLSTM_Alibaba_30s_test_step_rmse.npy")
tcn_30s = np.load(data_dir1 + "Lstm_Alibaba_30s_test_step_rmse.npy")

# 第二组：内存
SWT_clstm_5m = np.load(data_dir2 + "SWT_CLSTM_Alibaba_30s_test_step_rmse.npy")
cnnlstm_5m = np.load(data_dir2 + "Lstm_30s_test_step_rmse.npy")


# 确保比较相同长度的数据
min_len_30s = min(len(SWT_clstm_30s), len(tcn_30s))
min_len_5m = min(len(SWT_clstm_5m), len(cnnlstm_5m))

# 截取相同长度
SWT_clstm_30s = SWT_clstm_30s[:min_len_30s]
tcn_30s = tcn_30s[:min_len_30s]

SWT_clstm_5m = SWT_clstm_5m[:min_len_5m]
cnnlstm_5m = cnnlstm_5m[:min_len_5m]

# 计算平均RMSE
avg_SWT_clstm_30s = np.mean(SWT_clstm_30s)
avg_tcn_30s = np.mean(tcn_30s)

avg_SWT_clstm_5m = np.mean(SWT_clstm_5m)
avg_cnnlstm_5m = np.mean(cnnlstm_5m)


# 计算改进百分比
improvement_30s = ((avg_tcn_30s - avg_SWT_clstm_30s) / avg_tcn_30s) * 100
improvement_5m = ((avg_cnnlstm_5m - avg_SWT_clstm_5m) / avg_cnnlstm_5m) * 100


print(f"阿里巴巴CPU利用率 - SWT-CLSTM vs LSTM:")
print(f"SWT-CLSTM平均RMSE: {avg_SWT_clstm_30s:.4f}")
print(f"TCN平均RMSE: {avg_tcn_30s:.4f}")
print(f"改进百分比: {improvement_30s:.2f}%\n")

print(f"阿里巴巴内存利用率 - SWT-CLSTM vs LSTM:")
print(f"SWT-CLSTM平均RMSE: {avg_SWT_clstm_5m:.4f}")
print(f"LSTM平均RMSE: {avg_cnnlstm_5m:.4f}")
print(f"改进百分比: {improvement_5m:.2f}%\n")

# 谷歌CPU利用率 - SWT-CLSTM vs LSTM:
# SWT-CLSTM平均RMSE: 0.0079
# TCN平均RMSE: 0.0101
# 改进百分比: 21.34%
#
# 谷歌内存利用率 - SWT-CLSTM vs LSTM:
# SWT-CLSTM平均RMSE: 0.0056
# LSTM平均RMSE: 0.0082
# 改进百分比: 32.47%

# 阿里巴巴CPU利用率 - SWT-CLSTM vs LSTM:
# SWT-CLSTM平均RMSE: 0.0047
# TCN平均RMSE: 0.0085
# 改进百分比: 44.72%
#
# 阿里巴巴内存利用率 - SWT-CLSTM vs LSTM:
# SWT-CLSTM平均RMSE: 0.0008
# LSTM平均RMSE: 0.0012
# 改进百分比: 33.38%
