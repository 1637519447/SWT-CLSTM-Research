import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

def process_azure_vm_data(file_paths, output_dir='h:/work/', num_vms=100):
    """
    处理Azure VM CPU利用率数据
    
    参数:
    file_paths: Azure VM数据文件路径列表
    output_dir: 输出目录
    num_vms: 选择的虚拟机数量
    """
    print(f"开始处理Azure VM CPU利用率数据...")
    print(f"数据文件数量: {len(file_paths)}")
    for i, path in enumerate(file_paths, 1):
        print(f"文件 {i}: {path}")
    print(f"选择虚拟机数量: {num_vms}")
    print("=" * 60)
    
    # 检查所有数据文件是否存在
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"错误: 数据文件 {file_path} 不存在")
            return
    
    # 读取并合并所有Azure VM数据文件
    print("正在读取Azure VM数据...")
    all_data = []
    
    for i, file_path in enumerate(file_paths, 1):
        try:
            print(f"正在读取文件 {i}/{len(file_paths)}: {os.path.basename(file_path)}")
            # Azure数据格式：timestamp, vm_id, min_cpu, max_cpu, avg_cpu
            data = pd.read_csv(file_path, header=None, 
                              names=['timestamp', 'vm_id', 'min_cpu', 'max_cpu', 'avg_cpu'])
            all_data.append(data)
            print(f"文件 {i} 读取成功，共 {len(data)} 行")
        except Exception as e:
            print(f"错误: 读取文件 {file_path} 时出现异常: {e}")
            return
    
    # 合并所有数据
    print("\n正在合并所有数据文件...")
    data = pd.concat(all_data, ignore_index=True)
    print(f"合并完成，总共 {len(data)} 行数据")
    
    # 数据预处理
    print("\n开始数据预处理...")
    
    # 转换数据类型
    data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')
    data['avg_cpu'] = pd.to_numeric(data['avg_cpu'], errors='coerce')
    
    # 移除无效数据
    data = data.dropna(subset=['timestamp', 'vm_id', 'avg_cpu'])
    print(f"清理后数据行数: {len(data)}")
    
    # 获取唯一的VM ID
    unique_vms = data['vm_id'].unique()
    print(f"数据中包含 {len(unique_vms)} 个唯一虚拟机")
    
    # 选择前num_vms个虚拟机
    selected_vms = unique_vms[:num_vms]
    print(f"选择前 {len(selected_vms)} 个虚拟机进行处理")
    
    # 过滤选定的虚拟机数据
    filtered_data = data[data['vm_id'].isin(selected_vms)]
    print(f"选定虚拟机的数据行数: {len(filtered_data)}")
    
    # 按时间戳排序
    filtered_data = filtered_data.sort_values('timestamp')
    
    # 计算5分钟间隔的聚合数据
    print("\n开始计算5分钟间隔聚合...")
    
    # 将时间戳转换为5分钟间隔（300秒）
    filtered_data['time_interval'] = (filtered_data['timestamp'] // 300) * 300
    
    # 按时间间隔聚合CPU利用率
    print("正在聚合CPU利用率数据...")
    aggregated_data = []
    
    # 获取所有时间间隔
    time_intervals = sorted(filtered_data['time_interval'].unique())
    
    for interval in tqdm(time_intervals, desc="处理时间间隔"):
        interval_data = filtered_data[filtered_data['time_interval'] == interval]
        
        # 计算该时间间隔内所有VM的平均CPU利用率
        avg_cpu_utilization = interval_data['avg_cpu'].mean()
        
        # 将CPU利用率乘以0.01转换为小数形式
        avg_cpu_utilization = avg_cpu_utilization * 0.01
        
        aggregated_data.append({
            'time_interval': interval,
            'cpu_utilization': avg_cpu_utilization,
            'vm_count': len(interval_data)
        })
    
    # 创建聚合数据DataFrame
    cpu_df = pd.DataFrame(aggregated_data)
    
    # 创建连续的时间索引
    cpu_df['time_index'] = range(len(cpu_df))
    
    print(f"\n聚合完成，共生成 {len(cpu_df)} 个时间点的数据")
    
    # 计算统计信息
    print("\n=== Azure VM CPU利用率统计信息 ===")
    print(cpu_df['cpu_utilization'].describe())
    
    # 显示数据范围
    print(f"\n数据时间范围:")
    print(f"起始时间戳: {cpu_df['time_interval'].min()}")
    print(f"结束时间戳: {cpu_df['time_interval'].max()}")
    print(f"总时间跨度: {(cpu_df['time_interval'].max() - cpu_df['time_interval'].min()) / 3600:.2f} 小时")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存聚合数据
    print("\n正在保存聚合数据...")
    output_path = os.path.join(output_dir, 'Azure_cpu_util_aggregated_5m.csv')
    
    # 保存为与Google数据相同的格式
    output_df = cpu_df[['time_index', 'cpu_utilization']].copy()
    output_df.to_csv(output_path, index=False)
    print(f"Azure CPU利用率聚合数据已保存到: {output_path}")
    
    # 保存只包含CPU利用率的文件（无列标题）
    cpu_only_path = os.path.join(output_dir, 'Azure_cpu_util_only.csv')
    cpu_df[['cpu_utilization']].to_csv(cpu_only_path, index=False, header=False)
    print(f"CPU利用率单独数据已保存到: {cpu_only_path}")
    
    # 可选：保存详细数据（包含时间间隔和VM数量信息）
    detailed_output_path = os.path.join(output_dir, 'Azure_cpu_util_detailed_5m.csv')
    cpu_df.to_csv(detailed_output_path, index=False)
    print(f"详细聚合数据已保存到: {detailed_output_path}")
    
    # 计算并保存滑动窗口平均值
    print("\n正在计算滑动窗口平均值...")
    window_size = 12  # 12个5分钟间隔 = 1小时
    
    if len(cpu_df) >= window_size:
        cpu_windowed = []
        time_windowed = []
        
        for i in tqdm(range(len(cpu_df) - window_size + 1), desc="计算滑动平均"):
            window_avg = cpu_df['cpu_utilization'].iloc[i:i+window_size].mean()
            cpu_windowed.append(window_avg)
            time_windowed.append(i + window_size // 2)
        
        # 创建滑动平均数据框
        windowed_df = pd.DataFrame({
            'time_index': time_windowed,
            'cpu_utilization_avg': cpu_windowed
        })
        
        # 保存滑动平均数据
        windowed_path = os.path.join(output_dir, 'Azure_cpu_util_windowed_1h.csv')
        windowed_df.to_csv(windowed_path, index=False)
        print(f"滑动平均数据已保存到: {windowed_path}")
    
    # 显示前几行数据
    print("\n=== Azure CPU利用率聚合数据前10行 ===")
    print(output_df.head(10))
    
    # 数据质量检查
    print("\n=== 数据质量检查 ===")
    print(f"CPU利用率范围: {cpu_df['cpu_utilization'].min():.4f} - {cpu_df['cpu_utilization'].max():.4f}")
    print(f"平均VM数量每个时间间隔: {cpu_df['vm_count'].mean():.2f}")
    print(f"数据完整性: {len(cpu_df)} / {len(time_intervals)} 时间间隔")
    
    return output_df

def main():
    """
    主函数
    """
    # Azure数据文件路径列表
    azure_file_paths = [
        r"h:/work/azure_datasets/vm_cpu_readings-file-1-of-195.csv",
        r"h:/work/azure_datasets/vm_cpu_readings-file-2-of-195.csv",
        r"h:/work/azure_datasets/vm_cpu_readings-file-3-of-195.csv"
    ]
    
    # 检查所有文件是否存在
    missing_files = []
    for file_path in azure_file_paths:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"错误: 以下Azure数据文件不存在:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("请确认文件路径是否正确")
        return
    
    print(f"开始处理Azure VM CPU利用率数据")
    print(f"将处理 {len(azure_file_paths)} 个数据文件")
    print("=" * 60)
    
    # 处理数据
    try:
        cpu_data = process_azure_vm_data(
            file_paths=azure_file_paths,
            output_dir='h:/work/',
            num_vms=100
        )
        
        print("\n" + "=" * 60)
        print("Azure数据处理完成！")
        print("=" * 60)
        
        if cpu_data is not None:
            print(f"\n处理结果摘要:")
            print(f"CPU数据点数量: {len(cpu_data)}")
            print(f"CPU利用率范围: {cpu_data['cpu_utilization'].min():.4f} - {cpu_data['cpu_utilization'].max():.4f}")
            print(f"平均CPU利用率: {cpu_data['cpu_utilization'].mean():.4f}")
            print(f"CPU利用率标准差: {cpu_data['cpu_utilization'].std():.4f}")
            
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()