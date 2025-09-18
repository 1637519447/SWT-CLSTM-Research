import pandas as pd
import os

def clean_csv_files():
    """
    删除指定CSV文件的列名和第一列内容，保留纯数据
    """
    # 定义要处理的文件列表
    files_to_process = [
        r'h:\work\Alibaba_cpu_util_aggregated_30s.csv',
        r'h:\work\Alibaba_mem_util_aggregated_30s.csv',
        r'h:\work\Google_cpu_util_aggregated_5m.csv',
        r'h:\work\Google_mem_util_aggregated_5m.csv'
    ]
    
    for file_path in files_to_process:
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
            
            print(f"正在处理文件: {file_path}")
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 删除第一列（索引为0的列）
            df_cleaned = df.iloc[:, 1:]
            
            # 生成新的文件名（添加_cleaned后缀）
            file_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(file_name)[0]
            new_file_path = os.path.join(file_dir, f"{name_without_ext}_cleaned.csv")
            
            # 保存清理后的数据（不包含列名和索引）
            df_cleaned.to_csv(new_file_path, header=False, index=False)
            
            print(f"已保存清理后的文件: {new_file_path}")
            print(f"原始数据形状: {df.shape}")
            print(f"清理后数据形状: {df_cleaned.shape}")
            print("-" * 50)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            continue

def preview_cleaned_data():
    """
    预览清理后的数据
    """
    cleaned_files = [
        r'h:\work\Alibaba_cpu_util_aggregated_30s_cleaned.csv',
        r'h:\work\Alibaba_mem_util_aggregated_30s_cleaned.csv',
        r'h:\work\Google_cpu_util_aggregated_5m_cleaned.csv',
        r'h:\work\Google_mem_util_aggregated_5m_cleaned.csv'
    ]
    
    for file_path in cleaned_files:
        if os.path.exists(file_path):
            print(f"\n预览文件: {file_path}")
            # 读取前5行数据进行预览
            with open(file_path, 'r') as f:
                lines = f.readlines()[:5]
                for i, line in enumerate(lines, 1):
                    print(f"第{i}行: {line.strip()}")
            print("-" * 30)

if __name__ == "__main__":
    print("开始清理CSV文件...")
    print("=" * 60)
    
    # 清理文件
    clean_csv_files()
    
    print("\n文件清理完成！")
    print("=" * 60)
    
    # 预览清理后的数据
    print("\n预览清理后的数据:")
    preview_cleaned_data()
    
    print("\n所有操作完成！")
    print("清理后的文件已保存，文件名添加了'_cleaned'后缀")
    print("这些文件现在可以直接用作机器学习数据集")