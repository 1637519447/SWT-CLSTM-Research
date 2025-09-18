import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from datetime import datetime

def process_google_cluster_data(data_dir, output_dir='h:/work/', max_files=100):
    """
    处理谷歌云数据中心数据
    
    参数:
    data_dir: 数据目录路径
    output_dir: 输出目录
    max_files: 最大处理文件数量
    """
    print(f"开始处理谷歌云数据中心数据...")
    print(f"数据目录: {data_dir}")
    print(f"最大处理文件数: {max_files}")
    print("=" * 60)
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        return
    
    # 获取所有文件列表
    print("正在扫描数据文件...")
    file_pattern = os.path.join(data_dir, "*")
    all_files = glob.glob(file_pattern)
    
    # 过滤出文件（排除目录）
    data_files = [f for f in all_files if os.path.isfile(f)]
    
    if len(data_files) == 0:
        print(f"错误: 在目录 {data_dir} 中未找到任何数据文件")
        return
    
    print(f"找到 {len(data_files)} 个数据文件")
    
    # 限制处理文件数量
    files_to_process = data_files[:max_files]
    print(f"将处理前 {len(files_to_process)} 个文件")
    
    # 初始化数据存储
    all_cpu_data = []
    all_mem_data = []
    
    # 处理每个文件
    print("\n开始处理文件...")
    successful_files = 0
    failed_files = 0
    
    for file_path in tqdm(files_to_process, desc="处理文件", unit="文件"):
        try:
            # 读取文件数据
            # 假设文件格式为：第一列CPU利用率，第二列内存利用率
            # 根据实际文件格式调整读取方式
            
            # 尝试不同的读取方式
            data = None
            
            # 方式1: 尝试读取为CSV
            try:
                data = pd.read_csv(file_path, header=None)
            except:
                # 方式2: 尝试读取为空格分隔的文本文件
                try:
                    data = pd.read_csv(file_path, sep='\s+', header=None)
                except:
                    # 方式3: 尝试读取为制表符分隔的文件
                    try:
                        data = pd.read_csv(file_path, sep='\t', header=None)
                    except:
                        print(f"警告: 无法读取文件 {file_path}")
                        failed_files += 1
                        continue
            
            if data is None or len(data.columns) < 2:
                print(f"警告: 文件 {file_path} 格式不正确或列数不足")
                failed_files += 1
                continue
            
            # 提取CPU和内存利用率数据（假设前两列）
            cpu_data = data.iloc[:, 0].values
            mem_data = data.iloc[:, 1].values
            
            # 转换为数值类型并处理异常值
            cpu_data = pd.to_numeric(cpu_data, errors='coerce')
            mem_data = pd.to_numeric(mem_data, errors='coerce')
            
            # 移除NaN值
            cpu_data = cpu_data[~np.isnan(cpu_data)]
            mem_data = mem_data[~np.isnan(mem_data)]
            
            if len(cpu_data) > 0 and len(mem_data) > 0:
                all_cpu_data.extend(cpu_data)
                all_mem_data.extend(mem_data)
                successful_files += 1
            else:
                print(f"警告: 文件 {file_path} 中没有有效数据")
                failed_files += 1
                
        except Exception as e:
            print(f"错误: 处理文件 {file_path} 时出现异常: {e}")
            failed_files += 1
            continue
    
    print(f"\n文件处理完成:")
    print(f"成功处理: {successful_files} 个文件")
    print(f"处理失败: {failed_files} 个文件")
    print(f"总CPU数据点: {len(all_cpu_data)}")
    print(f"总内存数据点: {len(all_mem_data)}")
    
    if len(all_cpu_data) == 0 or len(all_mem_data) == 0:
        print("错误: 没有收集到有效数据")
        return
    
    # 数据聚合处理
    print("\n开始数据聚合...")
    
    # 将数据转换为numpy数组以便处理
    cpu_array = np.array(all_cpu_data)
    mem_array = np.array(all_mem_data)
    
    # 确保数据长度一致（取较短的长度）
    min_length = min(len(cpu_array), len(mem_array))
    cpu_array = cpu_array[:min_length]
    mem_array = mem_array[:min_length]
    
    print(f"统一数据长度: {min_length}")
    
    # 创建时间序列索引
    time_index = np.arange(min_length)
    
    # 创建聚合数据DataFrame
    print("正在创建聚合数据...")
    with tqdm(total=2, desc="创建数据框") as pbar:
        # CPU利用率聚合数据
        cpu_df = pd.DataFrame({
            'time_index': time_index,
            'cpu_utilization': cpu_array
        })
        pbar.update(1)
        
        # 内存利用率聚合数据
        mem_df = pd.DataFrame({
            'time_index': time_index,
            'memory_utilization': mem_array
        })
        pbar.update(1)
    
    # 计算统计信息
    print("\n=== 数据统计信息 ===")
    print(f"CPU利用率统计:")
    print(cpu_df['cpu_utilization'].describe())
    print(f"\n内存利用率统计:")
    print(mem_df['memory_utilization'].describe())
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存聚合数据
    print("\n正在保存聚合数据...")
    cpu_output_path = os.path.join(output_dir, 'google_cluster_cpu_aggregated.csv')
    mem_output_path = os.path.join(output_dir, 'google_cluster_mem_aggregated.csv')
    
    with tqdm(total=2, desc="保存文件") as pbar:
        cpu_df.to_csv(cpu_output_path, index=False)
        pbar.update(1)
        print(f"CPU利用率聚合数据已保存到: {cpu_output_path}")
        
        mem_df.to_csv(mem_output_path, index=False)
        pbar.update(1)
        print(f"内存利用率聚合数据已保存到: {mem_output_path}")
    
    # 可选：计算并保存平均利用率（按时间窗口）
    print("\n正在计算时间窗口平均值...")
    window_size = 100  # 每100个数据点计算一次平均值
    
    if min_length >= window_size:
        # 计算滑动平均
        cpu_windowed = []
        mem_windowed = []
        time_windowed = []
        
        for i in tqdm(range(0, min_length - window_size + 1, window_size), 
                     desc="计算窗口平均值"):
            cpu_window_avg = np.mean(cpu_array[i:i+window_size])
            mem_window_avg = np.mean(mem_array[i:i+window_size])
            
            cpu_windowed.append(cpu_window_avg)
            mem_windowed.append(mem_window_avg)
            time_windowed.append(i + window_size // 2)  # 窗口中心时间
        
        # 创建窗口平均数据框
        cpu_windowed_df = pd.DataFrame({
            'time_window': time_windowed,
            'cpu_utilization_avg': cpu_windowed
        })
        
        mem_windowed_df = pd.DataFrame({
            'time_window': time_windowed,
            'memory_utilization_avg': mem_windowed
        })
        
        # 保存窗口平均数据
        cpu_windowed_path = os.path.join(output_dir, 'google_cluster_cpu_windowed_avg.csv')
        mem_windowed_path = os.path.join(output_dir, 'google_cluster_mem_windowed_avg.csv')
        
        with tqdm(total=2, desc="保存窗口平均文件") as pbar:
            cpu_windowed_df.to_csv(cpu_windowed_path, index=False)
            pbar.update(1)
            print(f"CPU窗口平均数据已保存到: {cpu_windowed_path}")
            
            mem_windowed_df.to_csv(mem_windowed_path, index=False)
            pbar.update(1)
            print(f"内存窗口平均数据已保存到: {mem_windowed_path}")
    
    # 显示前几行数据
    print("\n=== CPU利用率聚合数据前5行 ===")
    print(cpu_df.head())
    print("\n=== 内存利用率聚合数据前5行 ===")
    print(mem_df.head())
    
    return cpu_df, mem_df

def main():
    """
    主函数
    """
    # 数据目录路径
    data_directory = r"H:\cluster2011-cpu-mem\\"
    
    # 检查目录是否存在
    if not os.path.exists(data_directory):
        print(f"错误: 数据目录 {data_directory} 不存在")
        print("请确认目录路径是否正确")
        return
    
    print(f"开始处理谷歌云数据中心数据")
    print(f"数据目录: {data_directory}")
    print("=" * 60)
    
    # 处理数据
    try:
        cpu_data, mem_data = process_google_cluster_data(
            data_dir=data_directory,
            output_dir='h:/work/',
            max_files=100
        )
        
        print("\n" + "=" * 60)
        print("数据处理完成！")
        print("=" * 60)
        
        if cpu_data is not None and mem_data is not None:
            print(f"\n处理结果摘要:")
            print(f"CPU数据点数量: {len(cpu_data)}")
            print(f"内存数据点数量: {len(mem_data)}")
            print(f"CPU利用率范围: {cpu_data['cpu_utilization'].min():.4f} - {cpu_data['cpu_utilization'].max():.4f}")
            print(f"内存利用率范围: {mem_data['memory_utilization'].min():.4f} - {mem_data['memory_utilization'].max():.4f}")
            
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()