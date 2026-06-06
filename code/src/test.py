"""
测试脚本 V2 — 滚动窗口预测 + 完整业绩归因

对测试集所有可能日期进行滚动预测（每窗口选 Top-5 持仓，持有 T+1→T+5），
计算：平均日收益率、胜率、累计净值曲线、最大回撤、夏普比率。

使用方法:
    python test.py                     # 滚动预测 + 完整报告
    python test.py --mode single       # 仅预测最后一个窗口（输出 result.csv）

输出文件:
    app/output/result.csv              — 最后窗口预测结果
    app/output/rolling_predictions.csv — 每日预测明细
    app/output/rolling_report.txt      — 滚动业绩报告
"""

import os
import sys
import json
import argparse
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


# ============================================================================
# 配置加载
# ============================================================================

def load_config():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.dirname(os.path.dirname(src_dir))
    config_dir = os.path.join(app_root, 'config')

    with open(os.path.join(config_dir, 'model.json'), 'r') as f:
        model_config = json.load(f)
    with open(os.path.join(config_dir, 'train.json'), 'r') as f:
        train_config = json.load(f)

    return {**model_config, **train_config}


# ============================================================================
# 评分与验证
# ============================================================================

def normalize_weights(weights):
    """归一化权重到 [0,1] 且和 ≤ 1，修正浮点舍入误差"""
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0, 1)
    w = w / (w.sum() + 1e-12)
    w = np.round(w, 6)
    excess = w.sum() - 1.0
    if excess > 0:
        imax = np.argmax(w)
        w[imax] = max(0.0, w[imax] - excess)
    return np.clip(w, 0, 1)


def is_valid_prediction(prediction_data):
    """验证预测结果: ≤5 只股票, 权重之和 ∈ [0, 1]"""
    weight_col = ('weight' if 'weight' in prediction_data.columns
                  else '权重' if '权重' in prediction_data.columns else None)
    id_col = ('stock_id' if 'stock_id' in prediction_data.columns
              else '股票代码' if '股票代码' in prediction_data.columns else None)

    if id_col is None or weight_col is None:
        raise ValueError('缺少 stock_id/股票代码 或 weight/权重 列')
    if len(prediction_data) > 5:
        raise ValueError(f'最多 5 只股票，当前 {len(prediction_data)} 只')
    ws = float(prediction_data[weight_col].sum())
    if not (-1e-9 <= ws <= 1.0 + 1e-9):
        raise ValueError(f'权重之和需 ∈ [0,1]，当前 {ws:.4f}')


def select_top_k(scores, stock_codes, k=5, temperature=10.0):
    """从分数中选 Top-K 股票并分配 Softmax 权重"""
    order = np.argsort(scores)[::-1]
    top_idx = order[:k]
    top_scores = scores[top_idx]
    shifted = top_scores - top_scores.max()
    w = np.exp(shifted * temperature)
    w = normalize_weights(w)
    result = pd.DataFrame({
        '股票代码': [int(stock_codes[i]) for i in top_idx],
        '权重': w
    })
    return result


# ============================================================================
# 业绩指标计算
# ============================================================================

def compute_metrics(daily_returns):
    """
    计算滚动预测的业绩指标。

    参数:
      daily_returns: np.array, 每日加权收益率

    返回:
      dict: 各项指标
    """
    dr = np.asarray(daily_returns, dtype=np.float64)

    n = len(dr)
    if n == 0:
        return {'error': '无有效交易日'}

    mean_ret = dr.mean()
    std_ret = dr.std(ddof=1) if n > 1 else 0.0
    win_rate = (dr > 0).mean()
    total_ret = dr.sum()
    # 累计净值 (简单加和, 非复利)
    cumulative = np.cumsum(dr)
    cum_ret = cumulative[-1]

    # 最大回撤 (基于累计净值)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = drawdowns.min()

    # 夏普比率 (假设无风险利率=0, 年化因子=52 周, 每周约5天选一次)
    sharpe = (mean_ret / (std_ret + 1e-12)) * np.sqrt(52) if std_ret > 1e-12 else 0.0

    # Calmar 比率
    calmar = cum_ret / (abs(max_dd) + 1e-12) if abs(max_dd) > 1e-12 else 0.0

    # 正收益均值 / 负收益均值
    pos_rets = dr[dr > 0]
    neg_rets = dr[dr < 0]
    avg_win = pos_rets.mean() if len(pos_rets) > 0 else 0.0
    avg_loss = neg_rets.mean() if len(neg_rets) > 0 else 0.0
    profit_factor = abs(avg_win * len(pos_rets) / (avg_loss * len(neg_rets) + 1e-12))

    return {
        'n_days': n,
        'mean_return': mean_ret,
        'std_return': std_ret,
        'win_rate': win_rate,
        'total_return': total_ret,
        'cumulative_return': cum_ret,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'cumulative_curve': cumulative,
        'drawdowns': drawdowns,
    }


def print_report(metrics, output_dir):
    """打印并保存业绩报告"""
    report_lines = []
    def emit(s):
        print(s)
        report_lines.append(s)

    emit("")
    emit("=" * 70)
    emit("  滚动预测业绩报告")
    emit("=" * 70)
    emit(f"  预测天数:        {metrics['n_days']}")
    emit(f"  平均收益率:      {metrics['mean_return']*100:+.4f}%")
    emit(f"  收益率标准差:    {metrics['std_return']*100:.4f}%")
    emit(f"  累计收益率:      {metrics['cumulative_return']*100:+.4f}%")
    emit(f"  胜率 (正收益):   {metrics['win_rate']*100:.2f}%")
    emit(f"  最大回撤:        {metrics['max_drawdown']*100:.4f}%")
    emit(f"  夏普比率 (年化): {metrics['sharpe_ratio']:.4f}")
    emit(f"  Calmar 比率:     {metrics['calmar_ratio']:.4f}")
    emit(f"  平均盈利:        {metrics['avg_win']*100:.4f}%")
    emit(f"  平均亏损:        {metrics['avg_loss']*100:.4f}%")
    emit(f"  盈亏比:          {metrics['profit_factor']:.4f}")
    emit("=" * 70)

    # 保存
    with open(os.path.join(output_dir, 'rolling_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines))


# ============================================================================
# 主流程
# ============================================================================

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
    # 1. 加载原始测试数据 (用于评估收益)
    # ========================================================================
    test_file = os.path.join(app_root, 'data', 'test.csv')
    print(f"\n加载测试数据: {test_file}")
    raw_df = pd.read_csv(test_file)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    raw_df = raw_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 全局日期序列（所有股票一致）
    all_dates = sorted(raw_df['日期'].unique())
    n_dates = len(all_dates)
    print(f"  日期范围: {all_dates[0].date()} ~ {all_dates[-1].date()}")
    print(f"  总交易日: {n_dates}")
    print(f"  股票数量: {raw_df['股票代码'].nunique()}")

    # 构建 (stock × date) → open_price / close_price 查询表
    open_pivot = raw_df.pivot_table(
        index='股票代码', columns='日期', values='开盘', aggfunc='first'
    )
    close_pivot = raw_df.pivot_table(
        index='股票代码', columns='日期', values='收盘', aggfunc='first'
    )
    stock_list = sorted(open_pivot.index.tolist())
    date_list = sorted(open_pivot.columns.tolist())

    # 预计算每个 (stock, date_idx) 的 5 日 forward return
    # return = (open[i+5] - open[i+1]) / open[i+1]
    open_mat = open_pivot.loc[stock_list, date_list].values  # (S, T)
    close_mat = close_pivot.loc[stock_list, date_list].values  # (S, T)
    n_stocks = len(stock_list)
    n_total_dates = len(date_list)

    # 可预测日期: i 从 0 到 T-6 (保证 i+5 存在), 共 T-5 个
    n_pred_dates = n_total_dates - 5

    # forward_returns[s, i] = (open[s, i+5] - open[s, i+1]) / open[s, i+1]
    forward_returns = np.full((n_stocks, n_pred_dates), np.nan, dtype=np.float64)
    for i in range(n_pred_dates):
        t1_open = open_mat[:, i + 1]   # T+1 开盘
        t5_open = open_mat[:, i + 5]   # T+5 开盘
        forward_returns[:, i] = (t5_open - t1_open) / (t1_open + 1e-12)

    # 预计算可交易性: T+1 是否一字涨停（开盘涨幅 ≥ 9.5% 则不可买入）
    # tradable[s, i] = True 当 T+1 开盘可买入
    tradable_mask = np.ones((n_stocks, n_pred_dates), dtype=bool)
    for i in range(n_pred_dates):
        t1_open = open_mat[:, i + 1]        # T+1 开盘价
        t_close = close_mat[:, i]            # T 收盘价
        t1_gap = (t1_open / (t_close + 1e-12)) - 1.0
        # 涨停（gap ≥ 9.5%）或数据缺失 → 不可交易
        untradable = (t1_gap >= 0.095) | np.isnan(t1_gap) | np.isnan(t_close)
        tradable_mask[untradable, i] = False

    n_untradable = (~tradable_mask).sum()
    print(f"  不可交易 (涨停/停牌) 样本: {n_untradable} / {n_stocks * n_pred_dates} "
          f"({100*n_untradable/max(1,n_stocks*n_pred_dates):.1f}%)")

    # 交易成本: 双边 0.2% (佣金+滑点)
    transaction_cost = config.get('transaction_cost', 0.002)
    print(f"  交易成本: {transaction_cost*100:.1f}% (双边)")
    print(f"  可预测窗口数: {n_pred_dates} (需要 T+1~T+5 未来数据)")

    # ========================================================================
    # 2. 特征工程 (多进程, 一次处理全部数据)
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
    # 3. 数据清洗与特征确认
    # ========================================================================
    exclude_cols = {'日期', '股票代码', 'instrument', 'label', 'datetime'}
    actual_features = [
        f for f in feature_columns_template
        if f in processed.columns and f not in exclude_cols
    ]
    print(f"  实际特征数: {len(actual_features)}")

    processed[actual_features] = processed[actual_features].replace(
        [np.inf, -np.inf], np.nan
    )
    processed[actual_features] = processed[actual_features].ffill().fillna(0)
    processed.rename(columns={'日期': 'datetime'}, inplace=True)
    processed['datetime'] = pd.to_datetime(processed['datetime'])

    # ========================================================================
    # 4. 截面标准化
    # ========================================================================
    print("\n截面标准化...")
    processed = cross_sectional_standardize(processed, actual_features)

    # ========================================================================
    # 5. 构建所有预测窗口
    # ========================================================================
    seq_length = config['sequence_length']  # 60
    n_features = len(actual_features)

    print(f"\n构建滚动预测窗口 (seq_length={seq_length})...")
    print(f"  每个股票生成 {n_pred_dates} 个窗口，共 ~{n_stocks * n_pred_dates} 个序列")

    # 按股票分组，为每个股票构建所有窗口
    all_sequences = []   # list of np.array (L, F)
    all_lengths = []     # list of int
    all_stock_indices = []  # list of int (0..S-1)
    all_date_indices = []   # list of int (0..n_pred_dates-1)

    for stock_code, group in tqdm(
        processed.groupby('股票代码', sort=False), desc='构建窗口'
    ):
        group = group.sort_values('datetime')
        feat_arr = group[actual_features].values.astype(np.float32)  # (T, F)

        # 确保股票在 stock_list 中
        if stock_code not in stock_list:
            continue
        s_idx = stock_list.index(stock_code)

        T = len(group)
        for d in range(n_pred_dates):
            # 窗口结束于 date index d (0-indexed)
            # 需要数据: [max(0, d-59), d]
            window_end = d
            if window_end >= T:
                continue  # 该股票没有这个日期的数据

            actual_len = min(window_end + 1, seq_length)
            start = window_end - actual_len + 1

            seq = feat_arr[start:window_end + 1].copy()  # (actual_len, F)

            # 左填充至 seq_length
            if actual_len < seq_length:
                pad = np.zeros((seq_length - actual_len, n_features), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)

            all_sequences.append(seq)
            all_lengths.append(actual_len)
            all_stock_indices.append(s_idx)
            all_date_indices.append(d)

    n_total_seq = len(all_sequences)
    print(f"  生成序列总数: {n_total_seq}")

    if n_total_seq == 0:
        raise RuntimeError("未生成任何有效序列！")

    sequences = np.stack(all_sequences)  # (N, L, F)
    seq_lengths_arr = np.array(all_lengths, dtype=np.int64)
    stock_idx_arr = np.array(all_stock_indices, dtype=np.int64)
    date_idx_arr = np.array(all_date_indices, dtype=np.int64)

    print(f"  序列矩阵: {sequences.shape} ({sequences.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  实际长度: min={seq_lengths_arr.min()}, max={seq_lengths_arr.max()}, "
          f"mean={seq_lengths_arr.mean():.1f}")

    # ========================================================================
    # 6. 模型集成推理
    # ========================================================================
    ensemble_seeds = config.get('ensemble_seeds', [config.get('seed', 42)])
    available_models = []
    for seed in ensemble_seeds:
        mp_path = os.path.join(model_dir, f'best_model_s{seed}.pth')
        if os.path.exists(mp_path):
            available_models.append((seed, mp_path))

    if not available_models:
        all_pth = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith('best_model') and f.endswith('.pth')
        ])
        if all_pth:
            for f in all_pth:
                available_models.append((f, os.path.join(model_dir, f)))
        else:
            raise FileNotFoundError(f"未找到模型文件于 {model_dir}")

    print(f"\n集成模型 ({len(available_models)} 个):")
    for seed, path in available_models:
        print(f"  seed={seed}: {os.path.basename(path)}")

    # 累计所有模型分数 (之后取平均)
    all_model_scores = []

    for seed, model_path in available_models:
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

        scores_list = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, n_total_seq, batch_size):
                batch = torch.FloatTensor(
                    sequences[i:i + batch_size]
                ).to(device)
                len_batch = torch.LongTensor(
                    seq_lengths_arr[i:i + batch_size]
                ).to(device)
                out = model(batch, lengths=len_batch)
                scores_list.append(out.cpu().numpy())

        model_scores = np.concatenate(scores_list)
        all_model_scores.append(model_scores)
        print(f"    seed={seed}: 推理完成")

    ensemble_scores = np.mean(all_model_scores, axis=0)  # (N,)

    if len(all_model_scores) > 1:
        stacked = np.stack(all_model_scores)
        corr = np.corrcoef(stacked)
        upper = corr[np.triu_indices_from(corr, k=1)]
        print(f"  模型间平均相关性: {upper.mean():.4f}")

    # ========================================================================
    # 7. 逐日选股 & 计算收益
    # ========================================================================
    print(f"\n逐日选股评分...")

    daily_records = []
    daily_returns_list = []
    n_excluded = 0
    n_untradable_skipped = 0  # 因涨停被跳过的股票数

    for d in tqdm(range(n_pred_dates), desc='逐日评估'):
        # 找出该日期所有股票的序列索引
        mask = date_idx_arr == d
        if mask.sum() < 5:
            n_excluded += 1
            continue

        s_scores = ensemble_scores[mask]           # 该日各股票分数
        s_indices = stock_idx_arr[mask]             # 对应股票在 stock_list 中的索引

        # Top-5 选股（跳过 T+1 一字涨停的股票，用后续股票补上）
        order = np.argsort(s_scores)[::-1]          # 分数从高到低
        selected_s_idx = []
        selected_scores = []
        for idx_in_mask in order:
            s = s_indices[idx_in_mask]
            # 检查可交易性
            if tradable_mask[s, d]:
                selected_s_idx.append(s)
                selected_scores.append(s_scores[idx_in_mask])
                if len(selected_s_idx) >= 5:
                    break
            else:
                n_untradable_skipped += 1

        if len(selected_s_idx) < 5:
            n_excluded += 1
            continue

        top_s_idx = np.array(selected_s_idx)
        top_scores = np.array(selected_scores)

        # Softmax 权重
        w = np.exp((top_scores - top_scores.max()) * 10.0)
        w = normalize_weights(w)

        # 获取各股票的 5 日收益
        rets = forward_returns[top_s_idx, d]  # (top_k,)
        # 排除 nan (股票在 T+1 或 T+5 无数据)
        valid = ~np.isnan(rets)
        if valid.sum() == 0:
            n_excluded += 1
            continue

        w_valid = w[valid] / w[valid].sum()
        r_valid = rets[valid]

        # 加权收益 − 双边交易成本
        weighted_ret = (w_valid * r_valid).sum() - transaction_cost
        daily_returns_list.append(weighted_ret)
        daily_records.append({
            '日期': date_list[d],  # 窗口结束日
            'weighted_return': weighted_ret,
            'selected_stocks': ','.join(
                str(stock_list[s]) for s in top_s_idx
            ),
            'weights': ','.join(f'{w[i]:.4f}' for i in range(5)),
        })

    print(f"  有效评估日: {len(daily_returns_list)} (排除 {n_excluded} 个)")
    if n_untradable_skipped > 0:
        print(f"  因涨停跳过的股票候选: {n_untradable_skipped} 次")

    daily_returns = np.array(daily_returns_list, dtype=np.float64)

    # ========================================================================
    # 8. 业绩报告 (分段：按实际序列长度过滤)
    # ========================================================================
    # 计算每日平均实际序列长度
    daily_avg_len = []
    for d in range(n_pred_dates):
        mask = date_idx_arr == d
        if mask.sum() > 0:
            daily_avg_len.append(seq_lengths_arr[mask].mean())
    daily_avg_len = np.array(daily_avg_len)

    def compute_len_filtered_metrics(daily_returns, daily_avg_len, min_len, label):
        """计算指定最小实际长度阈值下的指标"""
        mask = daily_avg_len >= min_len
        dr = daily_returns[mask]
        if len(dr) < 5:
            return None, mask
        m = compute_metrics(dr)
        print(f"\n  --- {label} (≥{min_len}天实际数据) ---")
        print(f"  窗口数: {m['n_days']}/{len(daily_returns)}")
        print(f"  平均收益率: {m['mean_return']*100:+.4f}%")
        print(f"  累计收益率: {m['cumulative_return']*100:+.4f}%")
        print(f"  胜率:       {m['win_rate']*100:.1f}%")
        print(f"  最大回撤:   {m['max_drawdown']*100:.2f}%")
        print(f"  夏普比率:   {m['sharpe_ratio']:.4f}")
        return m, mask

    # 全部窗口
    print(f"\n{'=' * 70}")
    print(f"  滚动预测业绩报告 (分段分析)")
    print(f"{'=' * 70}")

    metrics_full = compute_metrics(daily_returns)
    print(f"\n  --- 全部窗口 ({metrics_full['n_days']}个) ---")
    print(f"  平均收益率: {metrics_full['mean_return']*100:+.4f}%")
    print(f"  累计收益率: {metrics_full['cumulative_return']*100:+.4f}%")
    print(f"  胜率:       {metrics_full['win_rate']*100:.1f}%")
    print(f"  最大回撤:   {metrics_full['max_drawdown']*100:.2f}%")
    print(f"  夏普:       {metrics_full['sharpe_ratio']:.2f}")
    print(f"  收益分布:   "
          f"min={daily_returns.min()*100:.2f}%  "
          f"Q25={np.percentile(daily_returns,25)*100:.2f}%  "
          f"med={np.percentile(daily_returns,50)*100:.2f}%  "
          f"Q75={np.percentile(daily_returns,75)*100:.2f}%  "
          f"max={daily_returns.max()*100:.2f}%")

    # 按不同长度阈值分段
    thresholds = [30, 25, 20, 15, 10]
    best_metrics = None
    best_mask = None
    for thresh in thresholds:
        m, mask = compute_len_filtered_metrics(daily_returns, daily_avg_len, thresh, f"len≥{thresh}")
        if m is not None and best_metrics is None:
            best_metrics = m
            best_mask = mask

    # 使用 ≥20 天的窗口作为主报告 (平衡窗口数和可靠性)
    main_metrics, main_mask = compute_len_filtered_metrics(
        daily_returns, daily_avg_len, 20, "主报告"
    )
    if main_metrics is not None:
        print(f"\n{'=' * 70}")
        # 保存主报告
        print_report(main_metrics, output_dir)
    else:
        metrics_full = compute_metrics(daily_returns)
        print_report(metrics_full, output_dir)

    # 保存每日明细 (含实际长度信息)
    daily_df = pd.DataFrame(daily_records)
    daily_df['avg_actual_len'] = daily_avg_len
    daily_df.to_csv(os.path.join(output_dir, 'rolling_predictions.csv'), index=False)
    print(f"\n每日明细已保存至: {os.path.join(output_dir, 'rolling_predictions.csv')}")
    print(f"  (含 avg_actual_len 列，表示该预测日所有股票的平均实际序列长度)")

    # ========================================================================
    # 8b. 随机基线对比
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"  随机基线对比 (模拟 5000 次随机选股)")
    print(f"{'=' * 70}")

    rng = np.random.RandomState(42)
    n_sim = 5000
    random_means = []
    for _ in range(n_sim):
        rand_rets = []
        for d in range(n_pred_dates):
            mask = date_idx_arr == d
            if mask.sum() < 5:
                continue
            s_indices = np.unique(stock_idx_arr[mask])
            if len(s_indices) < 5:
                continue
            picked = rng.choice(s_indices, size=5, replace=False)
            rets = forward_returns[picked, d]
            valid = ~np.isnan(rets)
            if valid.sum() == 0:
                continue
            rand_rets.append(rets[valid].mean())
        if rand_rets:
            random_means.append(np.mean(rand_rets))

    random_means = np.array(random_means)
    print(f"  随机选股平均收益: {random_means.mean()*100:+.4f}%")
    print(f"  随机选股标准差:   {random_means.std()*100:.4f}%")
    print(f"  随机胜率:         {(random_means>0).mean()*100:.1f}%")

    # 模型在不同长度阈值下 vs 随机
    for thresh in [30, 25, 20, 15, 10]:
        len_mask = daily_avg_len >= thresh
        if len_mask.sum() < 5:
            continue
        model_mean = daily_returns[len_mask].mean()
        # 对随机也做同样的长度过滤... 但随机不需要长度，直接比较
        pval = (np.abs(random_means - random_means.mean()) >=
                np.abs(model_mean - random_means.mean())).mean()
        direction = "优于" if model_mean > random_means.mean() else "差于"
        print(f"  len≥{thresh}: 模型{model_mean*100:+.3f}% vs 随机{random_means.mean()*100:+.3f}%  "
              f"({direction}随机, p≈{pval:.4f})")

    # ========================================================================
    # 9. 最后一个窗口的预测结果 (兼容原输出格式)
    # ========================================================================
    print(f"\n{'=' * 60}")
    print(f"  最后一个预测窗口 (日期: {date_list[-6].date()})")
    print(f"{'=' * 60}")

    last_d = n_pred_dates - 1
    last_mask = date_idx_arr == last_d
    last_scores = ensemble_scores[last_mask]
    last_s_indices = stock_idx_arr[last_mask]

    last_result = select_top_k(last_scores, [stock_list[i] for i in last_s_indices], k=5)
    last_result.to_csv(os.path.join(output_dir, 'result.csv'), index=False)

    for i, (_, row) in enumerate(last_result.iterrows()):
        print(f"  {i + 1}. 股票 {int(row['股票代码']):6d}  权重: {row['权重']:.4f}")
    print(f"  权重之和: {last_result['权重'].sum():.4f}")

    # 最后一个窗口的独立得分
    last_date_str = date_list[last_d]
    last_top_stocks = last_result['股票代码'].tolist()
    last_ret = forward_returns[[stock_list.index(s) for s in last_top_stocks], last_d]
    last_w = last_result['权重'].values
    last_score = (last_w * last_ret).sum()
    print(f"  该窗口加权收益: {last_score*100:+.4f}%")

    print(f"\n  各窗口收益分布: "
          f"min={daily_returns.min()*100:.3f}%, "
          f"Q25={np.percentile(daily_returns, 25)*100:.3f}%, "
          f"median={np.median(daily_returns)*100:.3f}%, "
          f"Q75={np.percentile(daily_returns, 75)*100:.3f}%, "
          f"max={daily_returns.max()*100:.3f}%")

    return daily_returns, main_metrics if main_metrics is not None else metrics_full


# ============================================================================
# 单窗口模式 (兼容旧接口)
# ============================================================================

def single_window_mode():
    """仅对最后一个窗口进行预测（原逻辑）"""
    config = load_config()
    src_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.dirname(os.path.dirname(src_dir))
    model_dir = os.path.join(app_root, 'model')
    output_dir = os.path.join(app_root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")

    test_file = os.path.join(app_root, 'data', 'test.csv')
    print(f"\n加载测试数据: {test_file}")
    test_df = pd.read_csv(test_file)
    test_df['日期'] = pd.to_datetime(test_df['日期'])
    test_df = test_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    print(f"  日期范围: {test_df['日期'].min().date()} ~ {test_df['日期'].max().date()}")
    print(f"  股票数量: {test_df['股票代码'].nunique()}")

    feature_num = config['feature_num']
    feature_engineer = FEATURE_ENGINEER_FUNC_MAP[feature_num]
    feature_columns_template = FEATURE_COLUMNS_MAP[feature_num]

    print(f"\n特征工程: {feature_num}")
    groups = [group for _, group in test_df.groupby('股票代码', sort=False)]
    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(
            pool.imap(feature_engineer, groups),
            total=len(groups), desc='特征工程'
        ))
    processed = pd.concat(processed_list).reset_index(drop=True)

    exclude_cols = {'日期', '股票代码', 'instrument', 'label', 'datetime'}
    actual_features = [
        f for f in feature_columns_template
        if f in processed.columns and f not in exclude_cols
    ]
    print(f"  实际特征数: {len(actual_features)}")

    processed[actual_features] = processed[actual_features].replace(
        [np.inf, -np.inf], np.nan
    )
    processed[actual_features] = processed[actual_features].ffill().fillna(0)
    processed.rename(columns={'日期': 'datetime'}, inplace=True)
    processed['datetime'] = pd.to_datetime(processed['datetime'])

    print("\n截面标准化...")
    processed = cross_sectional_standardize(processed, actual_features)

    seq_length = config['sequence_length']
    print(f"\n构建预测序列...")

    stock_seqs = []
    skipped = 0
    for stock_code, group in tqdm(
        processed.groupby('股票代码', sort=False), desc='构建窗口'
    ):
        group = group.sort_values('datetime')
        n = len(group)
        if n < 7:
            skipped += 1
            continue
        window_end = n - 6
        actual_len = min(window_end + 1, seq_length)
        start = window_end - actual_len + 1
        seq = group.iloc[start:window_end + 1][actual_features].values.astype(np.float32)
        if actual_len < seq_length:
            pad = np.zeros((seq_length - actual_len, len(actual_features)), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        stock_seqs.append((stock_code, seq, actual_len))

    print(f"  有效预测: {len(stock_seqs)}, 跳过: {skipped}")

    if len(stock_seqs) < 5:
        raise RuntimeError(f"不足 5 只可预测股票")

    ensemble_seeds = config.get('ensemble_seeds', [config.get('seed', 42)])
    available_models = []
    for seed in ensemble_seeds:
        mp_path = os.path.join(model_dir, f'best_model_s{seed}.pth')
        if os.path.exists(mp_path):
            available_models.append((seed, mp_path))
    if not available_models:
        all_pth = sorted([f for f in os.listdir(model_dir)
                          if f.startswith('best_model') and f.endswith('.pth')])
        for f in all_pth:
            available_models.append((f, os.path.join(model_dir, f)))

    stock_codes = [s[0] for s in stock_seqs]
    sequences = np.stack([s[1] for s in stock_seqs])
    seq_lengths_vals = np.array([s[2] for s in stock_seqs], dtype=np.int64)

    all_scores = []
    for seed, mp_path in available_models:
        model = CausalGRUStockScorer(
            input_dim=len(actual_features),
            d_model=config.get('d_model', 192),
            gru_hidden=config.get('gru_hidden', 128),
            gru_layers=config.get('gru_layers', 2),
            dropout=config.get('dropout', 0.15)
        )
        model.load_state_dict(torch.load(mp_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        scores_l = []
        with torch.no_grad():
            for i in range(0, len(sequences), 512):
                batch = torch.FloatTensor(sequences[i:i + 512]).to(device)
                lb = torch.LongTensor(seq_lengths_vals[i:i + 512]).to(device)
                scores_l.append(model(batch, lengths=lb).cpu().numpy())
        all_scores.append(np.concatenate(scores_l))

    ensemble = np.mean(all_scores, axis=0)
    result_df = select_top_k(ensemble, stock_codes, k=5)
    result_df.to_csv(os.path.join(output_dir, 'result.csv'), index=False)

    print(f"\n{'=' * 60}")
    print(f"  Top-5 预测结果:")
    for i, (_, row) in enumerate(result_df.iterrows()):
        print(f"  {i + 1}. 股票 {int(row['股票代码']):6d}  权重: {row['权重']:.4f}")
    print(f"  权重之和: {result_df['权重'].sum():.4f}")

    return result_df


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDC2026 测试预测')
    parser.add_argument('--mode', type=str, default='rolling',
                        choices=['rolling', 'single'],
                        help='rolling=滚动多窗口, single=单窗口')
    parser.add_argument('--skip_score_check', action='store_true',
                        help='[single模式下] 跳过自评')
    args, _ = parser.parse_known_args()

    mp.set_start_method('spawn', force=True)

    if args.mode == 'single':
        result_df = single_window_mode()
        if not args.skip_score_check:
            # 简单自评
            from pathlib import Path
            test_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), 'data', 'test.csv'
            )
            raw = pd.read_csv(test_file)
            raw_sub = raw[['股票代码', '日期', '开盘']].copy()
            out_sub = result_df.rename(columns={'stock_id': '股票代码', 'weight': '权重'})
            sel = raw_sub[raw_sub['股票代码'].isin(out_sub['股票代码'])]
            sel = sel.groupby('股票代码').tail(5)
            grp = sel.groupby('股票代码')
            rets = grp.apply(
                lambda g: (g.iloc[-1]['开盘'] - g.iloc[0]['开盘']) / g.iloc[0]['开盘'],
                include_groups=False
            ).reset_index().rename(columns={0: '收益率'})
            merged = rets.merge(out_sub, on='股票代码')
            score = (merged['收益率'] * merged['权重']).sum()
            print(f"\n  自评得分: {score:.6f}")
    else:
        daily_returns, metrics = main()

    print("\n测试完成!")
