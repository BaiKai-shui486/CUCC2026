"""
特征工程与数据集工具函数
基于 THU-BDC2026 基线：

特征集:
  - '39':   10 基础行情 + 29 技术指标 (TA-Lib)
  - '158+39': 158 Alpha 因子 + 39 技术指标 (完整特征集)

数据集创建:
  - create_ranking_dataset_vectorized: 向量化加速版本 (推荐)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


# ============================================================================
# 特征工程
# ============================================================================

def engineer_features_39(df):
    """
    计算 39 个技术指标特征。

    特征列表:
      基础(10): 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅
      技术(29): sma_5, sma_20, ema_12, ema_26, rsi, macd, macd_signal,
               volume_change, obv, volume_ma_5, volume_ma_20, volume_ratio,
               kdj_k, kdj_d, kdj_j, boll_mid, boll_std, atr_14, ema_60,
               volatility_10, volatility_20, return_1, return_5, return_10,
               high_low_spread, open_close_spread, high_close_spread, low_close_spread
    """
    try:
        import talib
    except ImportError:
        print("请安装 TA-Lib 库: pip install TA-Lib")
        raise

    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    # 移动平均线 (SMA, EMA)
    df['sma_5'] = talib.SMA(close, timeperiod=5)
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['ema_26'] = talib.EMA(close, timeperiod=26)
    # ema_60 已移除：60天窗口在短序列测试集中产生NaN

    # MACD
    macd_line, macd_signal_line, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal_line

    # RSI
    df['rsi'] = talib.RSI(close, timeperiod=14)

    # KDJ
    df['kdj_k'], df['kdj_d'] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # Bollinger Bands
    df['boll_mid'], df['boll_upper'], df['boll_lower'] = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df['boll_std'] = (df['boll_upper'] - df['boll_mid']) / 2
    df.drop(columns=['boll_upper', 'boll_lower'], inplace=True)

    # ATR
    df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

    # OBV (On-Balance Volume)
    df['obv'] = talib.OBV(close, volume)

    # Volume-related features
    df['volume_change'] = volume.pct_change(fill_method=None)
    df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
    df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = df['volume_ma_5'] / df['volume_ma_20']

    # Returns and Volatility
    df['return_1'] = close.pct_change(1)
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()

    # Spreads
    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close

    # 处理 inf 和 NaN：前向填充保留时序信息（不用bfill避免未来数据泄露）
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill()
    df.fillna(0, inplace=True)

    return df


def engineer_features(df):
    """
    计算 158 个 Alpha 因子特征 (使用 TA-Lib 加速)。

    特征分组:
      1.  K线特征 (9):   KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
      2.  价格特征 (4):   OPEN0, HIGH0, LOW0, VWAP0
      3.  价格变动 (5):   ROC5, ROC10, ROC20, ROC30, ROC60
      4.  移动平均 (5):   MA5, MA10, MA20, MA30, MA60
      5.  标准差 (5):     STD5, STD10, STD20, STD30, STD60
      6.  回归特征 (15):  BETA, RSQR, RESI (各5个窗口)
      7.  最大/最小 (10): MAX, MIN (各5个窗口)
      8.  分位数 (10):    QTLU, QTLD (各5个窗口)
      9.  排名 (5):       RANK5~RANK60
      10. 随机振荡 (5):   RSV5~RSV60
      11. 极值索引 (15):  IMAX, IMIN, IMXD (各5个窗口)
      12. 相关性 (10):    CORR, CORD (各5个窗口)
      13. 计数特征 (15):  CNTP, CNTN, CNTD (各5个窗口)
      14. 价格变动和 (15): SUMP, SUMN, SUMD (各5个窗口)
      15. 成交量特征 (10): VMA, VSTD (各5个窗口)
      16. 加权成交量 (5):  WVMA5~WVMA60
      17. 成交量变动和 (15): VSUMP, VSUMN, VSUMD (各5个窗口)
    """
    try:
        import talib
    except ImportError:
        print("请安装 TA-Lib 库: pip install TA-Lib")
        raise

    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'] / (volume + 1e-12)

    features = []
    feature_names = []

    # 1. K-line features (9)
    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12)
    ])
    feature_names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

    # 2. Price-related features (4)
    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12)
    ])
    feature_names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    windows = [5, 10, 20]  # 精简：移除30/60天窗口，避免短序列产生NaN填充

    # 3. Price change features (5)
    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f'ROC{w}')

    # 4. Moving average features (5)
    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MA{w}')

    # 5. Standard deviation features (5)
    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'STD{w}')

    # 6. Regression-based features (15)
    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f'BETA{w}')

        # RSQR: 需要至少 w 条数据才能计算
        if len(close) >= w:
            time_period_series = pd.Series(range(w), index=close.index[:w])
            rolling_corr = close.rolling(w).corr(time_period_series)
            rsquare = rolling_corr ** 2
        else:
            rsquare = pd.Series(np.nan, index=close.index)
        features.append(rsquare)
        feature_names.append(f'RSQR{w}')

        # RESI: 需要至少 w 条数据
        if len(close) >= w:
            intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
            predicted = slope * (w - 1) + intercept
            resi = close - predicted
        else:
            resi = pd.Series(np.nan, index=close.index)
        features.append(resi / (close + 1e-12))
        feature_names.append(f'RESI{w}')

    # 7. Max/Min features (10)
    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MAX{w}')
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MIN{w}')

    # 8. Quantile features (10) - TA-Lib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f'QTLU{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f'QTLD{w}')

    # 9. Rank features (5)
    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f'RANK{w}')

    # 10. Stochastic oscillator features (5)
    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f'RSV{w}')

    # 11. Index of Max/Min features (15)
    for w in windows:
        if len(high) >= w:
            imax_val = high.rolling(w).apply(np.argmax, raw=True)
        else:
            imax_val = pd.Series(np.nan, index=high.index)
        features.append(imax_val / w)
        feature_names.append(f'IMAX{w}')
    for w in windows:
        if len(low) >= w:
            imin_val = low.rolling(w).apply(np.argmin, raw=True)
        else:
            imin_val = pd.Series(np.nan, index=low.index)
        features.append(imin_val / w)
        feature_names.append(f'IMIN{w}')
    for w in windows:
        if len(high) >= w:
            imax = high.rolling(w).apply(np.argmax, raw=True)
            imin = low.rolling(w).apply(np.argmin, raw=True)
        else:
            imax = pd.Series(np.nan, index=high.index)
            imin = pd.Series(np.nan, index=low.index)
        features.append((imax - imin) / w)
        feature_names.append(f'IMXD{w}')

    # 12. Correlation features (10)
    log_volume = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f'CORR{w}')

    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows:
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1).fillna(0)
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        feature_names.append(f'CORD{w}')

    # 13. Count features (15)
    close_diff_pos = (close > close.shift(1))
    close_diff_neg = (close < close.shift(1))
    for w in windows:
        features.append(close_diff_pos.rolling(w).mean())
        feature_names.append(f'CNTP{w}')
    for w in windows:
        features.append(close_diff_neg.rolling(w).mean())
        feature_names.append(f'CNTN{w}')
    for w in windows:
        cntp = close_diff_pos.rolling(w).mean()
        cntn = close_diff_neg.rolling(w).mean()
        features.append(cntp - cntn)
        feature_names.append(f'CNTD{w}')

    # 14. Sum of price change features (15)
    close_diff_abs = (close - close.shift(1)).abs()
    close_diff_up = (close - close.shift(1)).clip(lower=0)
    close_diff_down = -(close - close.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'SUMP{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'SUMN{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'SUMD{w}')

    # 15. Volume-related features (10)
    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VMA{w}')
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VSTD{w}')

    # 16. Weighted volume features (5)
    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows:
        mean_vol_w_ret = vol_weighted_ret.rolling(w).mean()
        std_vol_w_ret = vol_weighted_ret.rolling(w).std()
        features.append(std_vol_w_ret / (mean_vol_w_ret + 1e-12))
        feature_names.append(f'WVMA{w}')

    # 17. Volume change sum features (15)
    volume_diff_abs = (volume - volume.shift(1)).abs()
    volume_diff_up = (volume - volume.shift(1)).clip(lower=0)
    volume_diff_down = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'VSUMP{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'VSUMN{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'VSUMD{w}')

    # 合并所有特征
    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names

    df = pd.concat([df, feature_df], axis=1)

    # 填充缺失值（不用bfill避免未来数据泄露）
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill()
    df.fillna(0, inplace=True)
    return df


def engineer_features_158plus39(df):
    """
    计算 158+39 完整特征集:
      先计算 158 Alpha 因子, 再追加 39 个技术指标, 去重后返回。
    """
    df_copy = df.copy()

    # 1. 158 Alpha 特征
    df_158 = engineer_features(df_copy)

    # 2. 39 技术指标特征
    df_39 = engineer_features_39(df_copy)

    # 3. 合并 (从 df_39 中仅选取新增的技术指标列)
    feature_cols_39 = [
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
        'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14',
        'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
    feature_cols_39_exist = [col for col in feature_cols_39 if col in df_39.columns]

    df_final = pd.concat([df_158, df_39[feature_cols_39_exist]], axis=1)

    # 去重
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    # 处理 inf 和 NaN（不用bfill避免未来数据泄露）
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final = df_final.ffill()
    df_final.fillna(0, inplace=True)

    return df_final


# ============================================================================
# 特征列名映射
# ============================================================================

FEATURE_COLUMNS_MAP = {
    '39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
        'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14',
        'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ],
    '158+39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2',
        'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
        'ROC5', 'ROC10', 'ROC20',
        'MA5', 'MA10', 'MA20',
        'STD5', 'STD10', 'STD20',
        'BETA5', 'BETA10', 'BETA20',
        'RSQR5', 'RSQR10', 'RSQR20',
        'RESI5', 'RESI10', 'RESI20',
        'MAX5', 'MAX10', 'MAX20',
        'MIN5', 'MIN10', 'MIN20',
        'QTLU5', 'QTLU10', 'QTLU20',
        'QTLD5', 'QTLD10', 'QTLD20',
        'RANK5', 'RANK10', 'RANK20',
        'RSV5', 'RSV10', 'RSV20',
        'IMAX5', 'IMAX10', 'IMAX20',
        'IMIN5', 'IMIN10', 'IMIN20',
        'IMXD5', 'IMXD10', 'IMXD20',
        'CORR5', 'CORR10', 'CORR20',
        'CORD5', 'CORD10', 'CORD20',
        'CNTP5', 'CNTP10', 'CNTP20',
        'CNTN5', 'CNTN10', 'CNTN20',
        'CNTD5', 'CNTD10', 'CNTD20',
        'SUMP5', 'SUMP10', 'SUMP20',
        'SUMN5', 'SUMN10', 'SUMN20',
        'SUMD5', 'SUMD10', 'SUMD20',
        'VMA5', 'VMA10', 'VMA20',
        'VSTD5', 'VSTD10', 'VSTD20',
        'WVMA5', 'WVMA10', 'WVMA20',
        'VSUMP5', 'VSUMP10', 'VSUMP20',
        'VSUMN5', 'VSUMN10', 'VSUMN20',
        'VSUMD5', 'VSUMD10', 'VSUMD20',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
        'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14',
        'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
}

FEATURE_ENGINEER_FUNC_MAP = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


# ============================================================================
# 数据集创建 - 向量化加速版本
# ============================================================================

def create_ranking_dataset_vectorized(data, features, sequence_length,
                                       ranking_data_path=None, min_window_end_date=None):
    """
    向量化加速版本: 预计算每只股票的所有滑动窗口，再按日期聚合。

    参数:
      data:               DataFrame, 含 '日期'/'instrument'/'label' 列
      features:           特征列名列表
      sequence_length:    序列长度 (窗口大小)
      ranking_data_path:  可选缓存路径 (未启用)
      min_window_end_date: 窗口结束日下限 (验证集使用)

    返回:
      sequences:         list of np.array [N, L, F]
      targets:           list of np.array [N,]
      relevance_scores:  list of np.array [N,]
      stock_indices:     list of list [N,]
    """
    print("正在创建排序数据集（向量化加速版本）...")

    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    # 1. 按股票和时间排序
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)

    # 2. 剔除无 label 的行
    data = data.dropna(subset=['label'])

    # 3. 为每只股票生成所有滑动窗口
    all_windows = []  # (end_date, stock_code, sequence, target)

    print("Step 1: 为每只股票生成滑动窗口...")
    grouped = data.groupby('instrument')

    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue

        feature_values = group[features].values.astype(np.float32)  # (T, F)
        labels = group['label'].values.astype(np.float32)           # (T,)
        dates = group['datetime'].values                            # (T,)
        dates_day = group['datetime'].values.astype('datetime64[D]')

        num_windows = len(group) - sequence_length + 1
        n = len(group)
        for i in range(num_windows):
            end_idx = i + sequence_length - 1

            # 需要有未来 5 条数据
            if end_idx + 5 >= n:
                continue

            # 未来 5 条数据需在合理时间范围内（排除长期停牌）
            last_future_date = dates_day[end_idx + 5]
            total_span = (last_future_date - dates_day[end_idx]).astype(np.int64)
            if total_span > 12:  # 正常5个交易日约7天，允许少量假期延长
                continue

            seq = feature_values[i: i + sequence_length]   # (L, F)
            target = labels[end_idx]                        # label 对应窗口最后一天
            end_date = dates[end_idx]
            all_windows.append((end_date, stock_code, seq, target))

    # 4. 转为 DataFrame 按日期聚合
    print("Step 2: 按日期聚合窗口...")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'seq', 'target'])

    # 5. 按 date 分组构建每日样本
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []

    print("Step 3: 构建每日样本并计算 relevance...")
    grouped_by_date = window_df.groupby('date')

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)

    for date, group in tqdm(grouped_by_date, desc="Aggregating by date"):
        if min_window_end_date is not None and pd.to_datetime(date) < min_window_end_date:
            continue

        if len(group) < 10:
            continue

        day_seqs = np.stack(group['seq'].values)          # (N, L, F)
        day_targets = group['target'].values              # (N,)

        # 使用 rank-based relevance: 收益率越高，得分越高
        sorted_indices = np.argsort(day_targets)[::-1]
        relevance = np.zeros_like(day_targets, dtype=np.float32)
        for rank, idx in enumerate(sorted_indices):
            relevance[idx] = len(day_targets) - rank

        sequences.append(day_seqs)
        targets.append(day_targets)
        relevance_scores.append(relevance)
        stock_indices.append(group['stock_code'].tolist())

    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        avg_stocks = np.mean([len(seq) for seq in sequences])
        print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")

    return sequences, targets, relevance_scores, stock_indices


# ============================================================================
# 截面标准化
# ============================================================================

def cross_sectional_standardize(data, features):
    """
    每个交易日，对所有股票的每个特征独立做 Z-score 标准化。
    单向 GRU 无股票间交互能力，必须先让特征在截面上可比。

    参数:
      data:     DataFrame, 含 'datetime' 和 features 列
      features: 特征列名列表

    返回: 标准化后的 DataFrame (copy)
    """
    data = data.copy()
    for f in features:
        if f not in data.columns:
            continue
        # 逐日计算截面均值/标准差并标准化
        stats = data.groupby('datetime')[f].agg(['mean', 'std'])
        # 合并回原数据
        date_stats = data[['datetime']].join(stats, on='datetime')
        mean_val = date_stats['mean'].values
        std_val = date_stats['std'].values
        # 避免除零
        std_val = np.where(std_val > 1e-8, std_val, 1.0)
        data[f] = (data[f].values - mean_val) / std_val
        # 将 std≈0 的特征置零
        data.loc[date_stats['std'].values <= 1e-8, f] = 0.0
    return data


# ============================================================================
# Per-Stock 数据集创建（断点感知）
# ============================================================================

def add_cross_sectional_rank_features(data, features):
    """
    为每个特征添加横截面 rank 特征（0~1）。
    对每个交易日，计算每个特征在当日所有股票间的百分位排名。
    这让单向 GRU 能够感知股票间的相对位置，部分替代 Transformer 的 cross-stock attention。

    参数:
      data:     DataFrame, 含 'datetime' 和 features 列
      features: 原始特征列名列表

    返回:
      data:     添加了 CSRANK_{f} 列的 DataFrame
      new_features: 所有特征列名（原始 + 新增）
    """
    data = data.copy()
    rank_features = []

    for f in features:
        if f not in data.columns:
            continue
        rank_col = f'RK_{f}'
        # 逐日计算百分位排名
        data[rank_col] = data.groupby('datetime')[f].rank(pct=True)
        rank_features.append(rank_col)

    # 填充可能的 NaN（某天只有 1 只股票时）
    data[rank_features] = data[rank_features].fillna(0.5)

    all_features = list(features) + rank_features
    print(f"横截面 Rank 特征: {len(features)} → {len(all_features)} (+{len(rank_features)})")
    return data, all_features


def create_per_stock_dataset(data, features, sequence_length,
                              min_window_end_date=None):
    """
    每只股票独立构建滑动窗口，不按日期聚合。
    处理停牌断点：只在中国A股交易周内的连续段中构建窗口。

    参数:
      data:               DataFrame, 含 'datetime'/'instrument'/'label' 列
      features:           特征列名列表
      sequence_length:    序列长度 (窗口大小)
      min_window_end_date: 窗口结束日下限 (验证集使用)

    返回:
      sequences:          np.array [M, L, F]  扁平化的 per-stock 样本
      targets:            np.array [M]         未来 5 日收益率标签
      stock_ids:          np.array [M]         instrument 索引
      window_end_dates:   np.array [M]         datetime64 窗口结束日
      tradable:           np.array [M] bool    T+1 开盘是否可买入
    """
    import pandas as pd
    import numpy as np

    print("正在创建 per-stock 数据集（断点感知）...")

    data = data.copy()
    data['datetime'] = pd.to_datetime(data['datetime'])

    # 按股票和时间排序
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)

    # 剔除无 label 的行
    data = data.dropna(subset=['label'])

    all_seqs = []
    all_targets = []
    all_stock_ids = []
    all_end_dates = []
    all_tradable = []

    # 找到 '开盘' 和 '收盘' 在 features 中的位置（用于可交易性检查）
    try:
        open_col_idx = list(features).index('开盘')
        close_col_idx = list(features).index('收盘')
        do_tradability_check = True
    except ValueError:
        do_tradability_check = False
        print("警告: features 中无 '开盘'/'收盘' 列，跳过可交易性检查")

    print("Step 1: 为每只股票检测连续段并生成窗口...")
    grouped = data.groupby('instrument')

    total_skipped_breaks = 0
    total_skipped_short = 0

    for stock_id, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length + 5:
            total_skipped_short += 1
            continue

        feature_values = group[features].values.astype(np.float32)  # (T, F)
        labels = group['label'].values.astype(np.float32)           # (T,)
        dates = group['datetime'].values                            # datetime64[ns]
        dates_day = dates.astype('datetime64[D]')

        # 检测连续段：自然日差 > 10 视为断点（允许国庆/春节等长假，排除长期停牌）
        n = len(group)
        date_diffs = np.diff(dates_day).astype(np.int64)
        break_indices = np.where(date_diffs > 10)[0]

        segments = []
        seg_start = 0
        for b_idx in break_indices:
            segments.append((seg_start, b_idx + 1))  # [start, end) 左闭右开
            seg_start = b_idx + 1
        segments.append((seg_start, n))

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len < sequence_length + 5:
                total_skipped_short += 1
                continue

            num_windows = seg_len - sequence_length - 5 + 1
            seg_feat = feature_values[seg_start:seg_end]
            seg_labels = labels[seg_start:seg_end]
            seg_dates_day = dates_day[seg_start:seg_end]

            for i in range(num_windows):
                win_end_idx = i + sequence_length - 1  # 窗口在段内的结束索引

                # 窗口最后一天到第 5 个未来日跨度需合理（排除长期停牌）
                last_future_date = seg_dates_day[win_end_idx + 5]
                total_span = (last_future_date - seg_dates_day[win_end_idx]).astype(np.int64)
                if total_span > 12:
                    total_skipped_breaks += 1
                    continue

                seq = seg_feat[i: i + sequence_length]          # (L, F)
                target = seg_labels[win_end_idx]                  # 标量
                end_date = dates[seg_start + win_end_idx]

                # 可交易性检查：T+1 开盘是否一字涨停（无法买入）
                tradable = True
                if do_tradability_check:
                    t1_open = seg_feat[win_end_idx + 1, open_col_idx]   # T+1 开盘
                    t_close = seg_feat[win_end_idx, close_col_idx]       # T 收盘
                    t1_gap = (t1_open / (t_close + 1e-12)) - 1.0
                    if t1_gap >= 0.095:  # 开盘涨幅 ≥ 9.5%，接近一字板
                        tradable = False

                # ASSERT: 窗口内日期单调递增
                if i == 0:
                    win_dates = seg_dates_day[i: i + sequence_length]
                    assert np.all(np.diff(win_dates.astype(np.int64)) > 0), \
                        f"窗口内日期非严格递增! stock={stock_id}, end_date={end_date}"

                all_seqs.append(seq)
                all_targets.append(target)
                all_stock_ids.append(stock_id)
                all_end_dates.append(end_date)
                all_tradable.append(tradable)

    if total_skipped_breaks > 0 or total_skipped_short > 0:
        print(f"跳过的短段: {total_skipped_short}, 跳过的大跨度窗口: {total_skipped_breaks}")

    if len(all_seqs) == 0:
        print("警告: 未生成任何 per-stock 窗口！请检查数据时间范围是否足够。")
        return (np.empty((0, sequence_length, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.array([], dtype=object),
                np.array([], dtype='datetime64[D]'),
                np.array([], dtype=bool))

    sequences = np.array(all_seqs, dtype=np.float32)            # (M, L, F)
    targets = np.array(all_targets, dtype=np.float32)           # (M,)
    stock_ids = np.array(all_stock_ids)                          # (M,)
    window_end_dates = np.array(all_end_dates)                   # (M,)
    tradable = np.array(all_tradable, dtype=bool)                # (M,)

    # 按 min_window_end_date 过滤（验证集时间隔离）
    if min_window_end_date is not None:
        min_dt = pd.to_datetime(min_window_end_date)
        mask = window_end_dates >= np.datetime64(min_dt)
        print(f"min_window_end_date 过滤: {mask.sum()} / {len(mask)} 个样本保留")
        sequences = sequences[mask]
        targets = targets[mask]
        stock_ids = stock_ids[mask]
        window_end_dates = window_end_dates[mask]
        tradable = tradable[mask]

    print(f"成功创建 {len(sequences)} 个 per-stock 样本")
    print(f"  序列形状: {sequences.shape}")
    if do_tradability_check:
        print(f"  可交易样本: {tradable.sum()} / {len(tradable)} ({100*tradable.sum()/max(1,len(tradable)):.1f}%)")
    print(f"  窗口日期范围: {window_end_dates.min()} ~ {window_end_dates.max()}")

    return sequences, targets, stock_ids, window_end_dates, tradable
