"""
将股票数据按年份划分为训练集和测试集
- 2026年数据作为测试集 (test.csv)
- 其他年份数据作为训练集 (train.csv)

Args:
    input_path: 原始数据文件路径
    output_dir: 输出目录
"""
import pandas as pd
from pathlib import Path


def split_stock_data(input_path: str = "../data/stock_data.csv", 
                     output_dir: str = "../data") -> None:

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 读取数据
    print(f"正在读取数据: {input_path}")
    df = pd.read_csv(input_path)
    
    # 验证必要列
    required_columns = {"股票代码", "日期"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"输入文件缺少必要列: {sorted(missing_columns)}")
    
    # 解析日期
    print("正在解析日期...")
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    
    # 检查是否有无法解析的日期
    if df["日期"].isna().any():
        bad_rows = int(df["日期"].isna().sum())
        print(f"警告: 发现 {bad_rows} 行无法解析的日期，将被移除")
        df = df.dropna(subset=["日期"])
    
    # 提取年份
    df["年份"] = df["日期"].dt.year
    
    # 划分训练集和测试集
    print("正在划分训练集和测试集...")
    test_df = df[df["年份"] == 2026].copy()
    train_df = df[df["年份"] != 2026].copy()
    
    # 删除临时列
    train_df = train_df.drop(columns=["年份"])
    test_df = test_df.drop(columns=["年份"])
    
    # 格式化日期为字符串
    train_df["日期"] = train_df["日期"].dt.strftime("%Y-%m-%d")
    test_df["日期"] = test_df["日期"].dt.strftime("%Y-%m-%d")
    
    # 按股票代码和日期排序
    train_df = train_df.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    test_df = test_df.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    
    # 保存文件
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    print(f"正在保存训练集到: {train_path}")
    train_df.to_csv(train_path, index=False)
    
    print(f"正在保存测试集到: {test_path}")
    test_df.to_csv(test_path, index=False)
    
    # 输出统计信息
    print("数据划分完成!")
    print(f"训练集: {train_path}")
    print(f"测试集: {test_path}")
    
    # 警告检查
    if train_df.empty:
        print("警告: 训练集为空!")
    if test_df.empty:
        print("警告: 测试集为空! 请检查数据中是否包含2026年的数据")


if __name__ == "__main__":
    split_stock_data()
