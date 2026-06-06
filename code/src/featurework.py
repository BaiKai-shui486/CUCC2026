"""
featurework.py — 向前预测脚本

基于 test.csv 全部数据（截至 2026-03-13），使用已训练的 GRU 模型
预测下一持有期的 Top-5 持仓及权重。

输出: app/output/result.csv (stock_id, weight)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import CausalGRUStockScorer
from utils import (
    FEATURE_COLUMNS_MAP,
    FEATURE_ENGINEER_FUNC_MAP,
    cross_sectional_standardize,
)


def load_config():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.dirname(os.path.dirname(src_dir))
    config_dir = os.path.join(app_root, 'config')
    with open(os.path.join(config_dir, 'model.json'), 'r') as f:
        return json.load(f)


def normalize_weights(weights):
    """归一化权重，确保和 ≤ 1，浮点舍入修正"""
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0, 1)
    w = w / (w.sum() + 1e-12)
    w = np.round(w, 6)
    excess = w.sum() - 1.0
    if excess > 0:
        imax = np.argmax(w)
        w[imax] = max(0.0, w[imax] - excess)
    return np.clip(w, 0, 1)


def main():
    config = load_config()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.dirname(os.path.dirname(src_dir))
    model_dir = os.path.join(app_root, 'model')
    output_dir = os.path.join(app_root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # ---- 设备 ----
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用设备: CUDA - {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用设备: MPS")
    else:
        device = torch.device('cpu')
        print("使用设备: CPU")

    # ========================================================================
    # 1. 加载测试数据
    # ========================================================================
    test_file = os.path.join(app_root, 'data', 'test.csv')
    print(f"\n加载测试数据: {test_file}")
    raw_df = pd.read_csv(test_file)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    raw_df = raw_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    all_dates = sorted(raw_df['日期'].unique())
    print(f"  日期范围: {all_dates[0].date()} ~ {all_dates[-1].date()}")
    print(f"  总交易日: {len(all_dates)}")
    print(f"  股票数量: {raw_df['股票代码'].nunique()}")

    last_date = all_dates[-1]
    print(f"  预测参考日 (窗口结束日): {last_date.date()}")
    print(f"  预测目标: {last_date.date()} 之后的下一个持有期")

    # ========================================================================
    # 2. 特征工程
    # ========================================================================
    feature_num = config['feature_num']
    feature_engineer = FEATURE_ENGINEER_FUNC_MAP[feature_num]
    feature_columns_template = FEATURE_COLUMNS_MAP[feature_num]

    print(f"\n特征工程: {feature_num}")
    groups = [group for _, group in raw_df.groupby('股票代码', sort=False)]
    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(
            pool.imap(feature_engineer, groups),
            total=len(groups), desc='特征工程'
        ))
    processed = pd.concat(processed_list).reset_index(drop=True)

    # ========================================================================
    # 3. 数据清洗 + 截面标准化
    # ========================================================================
    exclude_cols = {'日期', '股票代码', 'instrument', 'label', 'datetime'}
    actual_features = [
        f for f in feature_columns_template
        if f in processed.columns and f not in exclude_cols
    ]
    print(f"  实际特征数: {len(actual_features)}")

    # NaN/inf 处理（前向填充，不用 bfill 避免未来数据泄露）
    processed[actual_features] = processed[actual_features].replace(
        [np.inf, -np.inf], np.nan
    )
    processed[actual_features] = processed[actual_features].ffill().fillna(0)
    processed.rename(columns={'日期': 'datetime'}, inplace=True)
    processed['datetime'] = pd.to_datetime(processed['datetime'])

    # 截面 Z-score 标准化（逐日、全股票）
    print("截面标准化...")
    processed = cross_sectional_standardize(processed, actual_features)

    # ========================================================================
    # 4. 构建预测窗口（每只股票取最后 seq_length 天）
    # ========================================================================
    seq_length = config['sequence_length']
    n_features = len(actual_features)
    print(f"\n构建预测窗口 (seq_length={seq_length})...")

    stock_codes = []
    sequences = []
    seq_lengths = []

    for stock_code, group in tqdm(
        processed.groupby('股票代码', sort=False), desc='构建窗口'
    ):
        group = group.sort_values('datetime')
        feat_arr = group[actual_features].values.astype(np.float32)  # (T, F)
        T = len(group)

        if T < 3:  # 数据太少，跳过
            continue

        # 取最后 min(T, seq_length) 天作为序列
        actual_len = min(T, seq_length)
        seq = feat_arr[-actual_len:].copy()  # (actual_len, F)

        # 左侧零填充至 seq_length
        if actual_len < seq_length:
            pad = np.zeros((seq_length - actual_len, n_features), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)

        stock_codes.append(stock_code)
        sequences.append(seq)
        seq_lengths.append(actual_len)

    if len(sequences) < 5:
        raise RuntimeError(f"可预测股票不足 5 只 (仅 {len(sequences)} 只)")

    sequences_arr = np.stack(sequences)  # (S, L, F)
    seq_lengths_arr = np.array(seq_lengths, dtype=np.int64)
    print(f"  预测股票数: {len(stock_codes)}")
    print(f"  序列矩阵: {sequences_arr.shape}")
    print(f"  实际长度: min={seq_lengths_arr.min()}, max={seq_lengths_arr.max()}, "
          f"mean={seq_lengths_arr.mean():.1f}")

    # ========================================================================
    # 5. 模型加载 & 推理
    # ========================================================================
    model_path = os.path.join(model_dir, f'best_model_s{config["seed"]}.pth')
    if not os.path.exists(model_path):
        candidates = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith('best_model_s') and f.endswith('.pth')
            and '_wf_' not in f
        ])
        if candidates:
            model_path = os.path.join(model_dir, candidates[0])
        else:
            raise FileNotFoundError(f"未找到模型文件于 {model_dir}")

    print(f"\n加载模型: {os.path.basename(model_path)}")

    model = CausalGRUStockScorer(
        input_dim=n_features,
        d_model=config.get('d_model', 192),
        gru_hidden=config.get('gru_hidden', 128),
        gru_layers=config.get('gru_layers', 2),
        dropout=config.get('dropout', 0.15)
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 推理
    batch_size = 512
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(sequences_arr), batch_size):
            batch = torch.FloatTensor(
                sequences_arr[i:i + batch_size]
            ).to(device)
            lb = torch.LongTensor(
                seq_lengths_arr[i:i + batch_size]
            ).to(device)
            out = model(batch, lengths=lb)
            all_scores.append(out.cpu().numpy())

    scores = np.concatenate(all_scores)

    # ========================================================================
    # 6. 可交易性过滤 & Top-5 选股
    # ========================================================================
    close_pivot = raw_df.pivot_table(
        index='股票代码', columns='日期', values='收盘', aggfunc='first'
    )
    open_pivot = raw_df.pivot_table(
        index='股票代码', columns='日期', values='开盘', aggfunc='first'
    )

    # 检查最后一天是否一字涨停（今日涨停 → 次日大概率继续涨停，应排除）
    tradable = np.ones(len(stock_codes), dtype=bool)
    last_date_str = last_date.strftime('%Y-%m-%d')

    for i, sc in enumerate(stock_codes):
        if sc in open_pivot.index and sc in close_pivot.index:
            if last_date_str in open_pivot.columns and last_date_str in close_pivot.columns:
                t_close = close_pivot.loc[sc, last_date_str]
                t_open = open_pivot.loc[sc, last_date_str]
                if not pd.isna(t_close) and not pd.isna(t_open) and t_close > 1e-6:
                    gap = (t_open / t_close) - 1.0
                    if gap >= 0.095:
                        tradable[i] = False

    n_untradable = (~tradable).sum()
    if n_untradable > 0:
        print(f"  当日一字涨停 (排除): {n_untradable} 只")

    # 按分数排序，跳过不可交易股票
    order = np.argsort(scores)[::-1]
    top_stocks = []
    top_scores_list = []
    for idx in order:
        if tradable[idx]:
            top_stocks.append(stock_codes[idx])
            top_scores_list.append(scores[idx])
            if len(top_stocks) >= 5:
                break

    if len(top_stocks) < 5:
        print(f"  ⚠ 可交易股票不足 5 只，实际选出 {len(top_stocks)} 只")

    # Softmax 权重
    top_scores_arr = np.array(top_scores_list[:5])
    shifted = top_scores_arr - top_scores_arr.max()
    w = np.exp(shifted * 10.0)
    w = normalize_weights(w)

    # ========================================================================
    # 7. 输出 result.csv (UTF-8)
    # ========================================================================
    result_path = os.path.join(output_dir, 'result.csv')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write('stock_id,weight\n')
        for stock, weight in zip(top_stocks[:5], w[:5]):
            f.write(f'{stock},{weight:.6f}\n')

    print(f"\n{'=' * 50}")
    print(f"  预测结果 (下一持有期)")
    print(f"{'=' * 50}")
    for i, (stock, weight) in enumerate(zip(top_stocks[:5], w[:5])):
        print(f"  {i + 1}. {stock:>8}  权重: {weight:.6f}")
    print(f"  权重之和: {w[:5].sum():.6f}")
    print(f"  现金 (1 - 权重和): {1 - w[:5].sum():.6f}")
    print(f"\n  结果已保存: {result_path}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
