import pandas as pd
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

def process_alibaba_cluster_data(input_file_path, output_dir='h:/work/'):
    """
    处理阿里巴巴集群数据
    
    参数:
    input_file_path: 输入CSV文件路径
    output_dir: 输出目录
    """
    print("开始读取数据...")
    
    # 读取CSV文件
    try:
        # 使用tqdm显示文件读取进度
        tqdm.pandas(desc="读取CSV文件")
        df = pd.read_csv(input_file_path)
        print(f"数据读取成功，共{len(df)}行数据")
        print(f"数据列数: {len(df.columns)}")
        print(f"列名: {list(df.columns)}")
        print("前5行数据:")
        print(df.head())
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 检查数据结构并设置正确的列名
    print("\n正在检查数据结构...")
    if len(df.columns) >= 4:
        # 如果有列名，使用现有列名；如果没有，设置列名
        if df.columns[0] == 0 or 'Unnamed' in str(df.columns[0]):
            # 没有列名的情况，根据实际列数设置列名
            if len(df.columns) == 4:
                df.columns = ['machine_id', 'timestamp', 'cpu_util_percent', 'mem_util_percent']
            else:
                # 对于有更多列的情况，只使用前4列
                df = df.iloc[:, :4]
                df.columns = ['machine_id', 'timestamp', 'cpu_util_percent', 'mem_util_percent']
        else:
            # 有列名的情况，根据位置选择需要的列
            if len(df.columns) >= 4:
                # 选择前4列并重命名
                df = df.iloc[:, :4]
                df.columns = ['machine_id', 'timestamp', 'cpu_util_percent', 'mem_util_percent']
    else:
        print(f"错误: 数据列数不足，期望至少4列，实际{len(df.columns)}列")
        return
    
    print(f"处理后数据形状: {df.shape}")
    print(f"机器数量: {df['machine_id'].nunique()}")
    
    # 选择前100台机器
    print("\n正在筛选前100台机器...")
    unique_machines = df['machine_id'].unique()[:100]
    df_filtered = df[df['machine_id'].isin(unique_machines)].copy()
    
    print(f"筛选前100台机器后数据形状: {df_filtered.shape}")
    
    # 检查数据类型并转换
    print("\n正在进行数据类型转换和清洗...")
    try:
        # 使用tqdm显示数据转换进度
        with tqdm(total=4, desc="数据类型转换") as pbar:
            # 确保数值列是数值类型
            df_filtered['cpu_util_percent'] = pd.to_numeric(df_filtered['cpu_util_percent'], errors='coerce')
            pbar.update(1)
            
            df_filtered['mem_util_percent'] = pd.to_numeric(df_filtered['mem_util_percent'], errors='coerce')
            pbar.update(1)
            
            df_filtered['timestamp'] = pd.to_numeric(df_filtered['timestamp'], errors='coerce')
            pbar.update(1)
            
            # 删除包含NaN的行
            original_len = len(df_filtered)
            df_filtered = df_filtered.dropna()
            pbar.update(1)
            
            print(f"数据清洗: 删除了{original_len - len(df_filtered)}行包含NaN的数据")
            print(f"数据清洗后形状: {df_filtered.shape}")
        
    except Exception as e:
        print(f"数据类型转换错误: {e}")
        return
    
    # 将百分比数据转换为小数（乘以0.01）
    print("\n正在转换百分比数据...")
    with tqdm(total=2, desc="百分比转换") as pbar:
        df_filtered['cpu_util'] = df_filtered['cpu_util_percent'] * 0.01
        pbar.update(1)
        
        df_filtered['mem_util'] = df_filtered['mem_util_percent'] * 0.01
        pbar.update(1)
    
    # 删除原始百分比列
    df_filtered = df_filtered.drop(['cpu_util_percent', 'mem_util_percent'], axis=1)
    
    print("\n开始按30秒间隔聚合数据...")
    
    # 将时间戳转换为30秒间隔的时间窗口
    print("正在创建时间窗口...")
    df_filtered['time_window'] = (df_filtered['timestamp'] // 30) * 30
    
    # 按时间窗口聚合数据（计算平均值）
    print("正在聚合CPU数据...")
    cpu_aggregated = df_filtered.groupby('time_window')['cpu_util'].mean().reset_index()
    
    print("正在聚合内存数据...")
    mem_aggregated = df_filtered.groupby('time_window')['mem_util'].mean().reset_index()
    
    # 重命名列
    cpu_aggregated.columns = ['timestamp', 'cpu_util']
    mem_aggregated.columns = ['timestamp', 'mem_util']
    
    # 按时间戳排序
    print("正在排序数据...")
    with tqdm(total=2, desc="数据排序") as pbar:
        cpu_aggregated = cpu_aggregated.sort_values('timestamp')
        pbar.update(1)
        
        mem_aggregated = mem_aggregated.sort_values('timestamp')
        pbar.update(1)
    
    print(f"CPU聚合数据形状: {cpu_aggregated.shape}")
    print(f"内存聚合数据形状: {mem_aggregated.shape}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为CSV文件
    print("\n正在保存结果文件...")
    cpu_output_path = os.path.join(output_dir, 'cpu_util_aggregated_30s.csv')
    mem_output_path = os.path.join(output_dir, 'mem_util_aggregated_30s.csv')
    
    with tqdm(total=2, desc="保存文件") as pbar:
        cpu_aggregated.to_csv(cpu_output_path, index=False)
        pbar.update(1)
        print(f"CPU利用率数据已保存到: {cpu_output_path}")
        
        mem_aggregated.to_csv(mem_output_path, index=False)
        pbar.update(1)
        print(f"内存利用率数据已保存到: {mem_output_path}")
    
    # 显示统计信息
    print("\n=== 数据统计信息 ===")
    print(f"CPU利用率统计:")
    print(cpu_aggregated['cpu_util'].describe())
    print(f"\n内存利用率统计:")
    print(mem_aggregated['mem_util'].describe())
    
    # 显示前几行数据
    print("\n=== CPU利用率数据前5行 ===")
    print(cpu_aggregated.head())
    print("\n=== 内存利用率数据前5行 ===")
    print(mem_aggregated.head())
    
    return cpu_aggregated, mem_aggregated

def main():
    """
    主函数
    """
    # 输入文件路径
    input_file = r"I:\Dataset\alibaba_clusterdata_v2018\machine_usage.csv"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        print("请确认文件路径是否正确")
        return
    
    print(f"开始处理文件: {input_file}")
    print("=" * 50)
    
    # 处理数据
    try:
        cpu_data, mem_data = process_alibaba_cluster_data(input_file)
        print("\n" + "=" * 50)
        print("数据处理完成！")
        print("=" * 50)
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()