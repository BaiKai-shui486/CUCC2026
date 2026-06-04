"""
训练脚本 - 基于 THU-BDC2026 的 StockTransformer 排序模型训练

使用方法:
    conda activate THU-BDC
    python train.py

配置文件:
    config/model.json  - 模型超参数 (d_model, nhead, num_layers, ...)
    config/train.json  - 训练超参数 (batch_size, lr, epochs, ...)
"""

import os
import sys
import json
import math
import random
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import joblib

# 将当前目录加入 path，确保能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import StockTransformer
from utils import (
    FEATURE_COLUMNS_MAP,
    FEATURE_ENGINEER_FUNC_MAP,
    create_ranking_dataset_vectorized,
)


def load_config():
    """加载 model.json 和 train.json，合并为统一配置"""
    # __file__ = app/code/src/train.py
    # src_dir   = app/code/src
    # app_root  = app
    src_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.dirname(os.path.dirname(src_dir))
    config_dir = os.path.join(app_root, 'config')

    with open(os.path.join(config_dir, 'model.json'), 'r') as f:
        model_config = json.load(f)

    with open(os.path.join(config_dir, 'train.json'), 'r') as f:
        train_config = json.load(f)

    # 合并配置
    config = {**model_config, **train_config}
    return config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class WeightedRankingLoss(nn.Module):
    """LambdaRank: pairwise logistic + |Δrank| 权重"""
    def __init__(self, sigma=1.0):
        super(WeightedRankingLoss, self).__init__()
        self.sigma = sigma

    def forward(self, y_pred, y_true, masks):
        masks = masks.float()
        y_pred = y_pred * masks + (1 - masks) * (-1e4)
        y_true = y_true * masks

        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)

        pair_mask = (true_diff != 0).float() * masks.unsqueeze(2) * masks.unsqueeze(1)
        lambda_weights = torch.abs(true_diff)

        sign = torch.sign(true_diff)
        logistic_loss = torch.log(1.0 + torch.exp(-self.sigma * sign * pred_diff))

        weighted_loss = logistic_loss * lambda_weights * pair_mask
        num_pairs = pair_mask.sum().clamp(min=1)
        return weighted_loss.sum() / num_pairs


def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """计算 Top-5 收益和、与理论最高值的比值、final_score"""
    batch_size = y_pred.size(0)

    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()

        if valid_indices.numel() < k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]

        # Predicted Top 5
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()

        # True Top 5 (理论最高)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()

        # Random 5 (期望值)
        random_return_sum = k * valid_true.mean().item()

        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0

        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)

    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
        'ratio_pred': np.mean(ratio_pred_list) if ratio_pred_list else 0.0,
        'ratio_random': np.mean(ratio_random_list) if ratio_random_list else 0.0,
        'final_score': np.mean(final_score_list) if final_score_list else 0.0,
    }
    return metrics


class RankingDataset(torch.utils.data.Dataset):
    """排序数据集"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),
            'targets': torch.FloatTensor(self.targets[idx]),
            'relevance': torch.FloatTensor(self.relevance_scores[idx]),
            'stock_indices': torch.LongTensor(self.stock_indices[idx])
        }


def collate_fn(batch):
    """自定义 collate: padding 到相同股票数"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]

    max_stocks = max(seq.size(0) for seq in sequences)

    padded_sequences, padded_targets, padded_relevance, padded_stock_indices, masks = [], [], [], [], []

    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)

        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size)
            stock_pad = torch.full((pad_size,), -1, dtype=torch.long)

            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)

        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)

    return {
        'sequences': torch.stack(padded_sequences),
        'targets': torch.stack(padded_targets),
        'relevance': torch.stack(padded_relevance),
        'stock_indices': torch.stack(padded_stock_indices),
        'masks': torch.stack(masks)
    }


def _build_label_and_clean(processed, drop_small_open=True):
    """构建标签 (未来5日收益率) 并清洗无效样本"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])

    # Winsorize：截断极端异常值，防止单个样本主导 softmax 损失
    low, high = processed['label'].quantile([0.005, 0.995]).values
    processed['label'] = processed['label'].clip(low, high)

    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, config, drop_small_open=True):
    """通用预处理: 特征工程 + 标签构建"""
    assert config['feature_num'] in FEATURE_ENGINEER_FUNC_MAP, \
        f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"

    feature_engineer = FEATURE_ENGINEER_FUNC_MAP[config['feature_num']]
    feature_columns = FEATURE_COLUMNS_MAP[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行 {desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc} 输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None, config=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", config=config, drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", config=config, drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None, config=None):
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", config=config, drop_small_open=True)


def split_train_val_by_last_month(df, sequence_length, val_months=2):
    """按最后 N 个月做验证集划分"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=val_months)).normalize()

    # 验证集需保留前 sequence_length-1 个交易日作为序列上下文
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后 {val_months} 月): {val_start.date()} 到 {last_date.date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer, config, scheduler=None):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0

    accum_steps = config.get('accum_steps', 1)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        relevance = batch['relevance'].to(device)
        masks = batch['masks'].to(device)

        stock_idx = batch['stock_indices'].to(device)
        outputs = model(sequences, stock_idx)  # [B, N]

        # 向量化损失，按累积步数缩放
        batch_loss = criterion(outputs, relevance, masks) / accum_steps
        
        batch_loss.backward()

        total_loss += batch_loss.item() * accum_steps

        # 每 accum_steps 步更新一次参数
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            optimizer.zero_grad()

            # 每步更新学习率（warmup + cosine）
            if scheduler is not None:
                scheduler.step()

            if writer:
                writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch * len(dataloader) + local_step)

            local_step += 1

        with torch.no_grad():
            # Mask 输出用于计算指标
            masked_outputs = outputs * masks + (1 - masks) * (-1e4)
            masked_targets = targets * masks
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

        if writer and (batch_idx + 1) % accum_steps == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', batch_loss.item() * accum_steps, global_step=step)
            for k, v in metrics.items():
                writer.add_scalar(f'train/{k}', v, global_step=step)

    num_batches = len(dataloader)
    if num_batches > 0:
        for k in total_metrics:
            total_metrics[k] /= num_batches

    return total_loss / num_batches if num_batches > 0 else 0, total_metrics


def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            relevance = batch['relevance'].to(device)
            masks = batch['masks'].to(device)

            stock_idx = batch['stock_indices'].to(device)
            outputs = model(sequences, stock_idx)  # [B, N]

            # 向量化损失：直接使用原始收益率作为目标
            batch_loss = criterion(outputs, relevance, masks)

            total_loss += batch_loss.item()

            # 计算排序指标
            masked_outputs = outputs * masks + (1 - masks) * (-1e4)
            masked_targets = targets * masks
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches

    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)

    return avg_loss, total_metrics


def main():
    config = load_config()

    # 设置随机种子
    set_seed(config.get('seed', 42))

    # 日志和评分输出目录（固定到 app/output）
    src_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.dirname(os.path.dirname(src_dir))
    model_dir = os.path.join(app_root, 'model')    # 存放需重新加载的文件 (.pth, .pkl)
    output_dir = os.path.join(app_root, 'output')  # 存放不需重新加载的文件 (log, final_score)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"使用设备: CUDA GPU - {gpu_name}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"使用设备: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"使用设备: CPU (无 GPU)")

    # 1. 数据加载
    data_file = os.path.join(app_root, 'data', 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(
        full_df, config['sequence_length'], config.get('val_months', 2)
    )

    # 股票 ID 映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    print(f"股票数量: {num_stocks}")

    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx, config=config)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx, config=config)

    # 3. 标准化
    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data[features] = train_data[features].ffill().bfill().fillna(0)
    val_data[features] = val_data[features].ffill().bfill().fillna(0)
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])

    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    # 4. 创建排序数据集
    train_sequences, train_targets, train_relevance, train_stock_indices = \
        create_ranking_dataset_vectorized(train_data, features, config['sequence_length'])
    val_sequences, val_targets, val_relevance, val_stock_indices = \
        create_ranking_dataset_vectorized(
            val_data, features, config['sequence_length'],
            min_window_end_date=val_start.strftime('%Y-%m-%d')
        )

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数: {len(val_sequences)}")

    # 5. DataLoader
    # 注意: num_workers=0 避免 spawn 模式下的 CUDA 多进程问题
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    # 6. 模型初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 7. 损失函数、优化器和调度器
    criterion = WeightedRankingLoss(sigma=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # Warmup + Cosine 退火 (按步更新)
    total_steps = config['num_epochs'] * len(train_loader)
    warmup_steps = int(config.get('warmup_ratio', 0.05) * total_steps)

    def lr_lambda(current_step):
        # current_step 从 0 开始，+1 避免第一步 LR=0
        step = current_step + 1
        if step <= warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 8. 早停
    class EarlyStopping:
        def __init__(self, patience=15, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_score = None

        def __call__(self, score):
            if self.best_score is None:
                self.best_score = score
            elif score < self.best_score + self.min_delta:
                self.counter += 1
                return self.counter >= self.patience
            else:
                self.best_score = score
                self.counter = 0
            return False

    early_stopping = EarlyStopping(patience=15, min_delta=0.001)

    # 9. 训练循环
    best_score = -float('inf')
    best_epoch = -1

    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
        current_lr = scheduler.get_last_lr()[0]
        print(f"当前学习率: {current_lr:.2e}")

        # 训练
        train_loss, train_metrics = train_ranking_model(
            model, train_loader, criterion, optimizer, device, epoch, writer, config, scheduler
        )
        print(f"Train Loss: {train_loss:.4f}")
        for k, v in train_metrics.items():
            print(f"Train {k}: {v:.4f}")

        # 验证
        eval_loss, eval_metrics = evaluate_ranking_model(
            model, val_loader, criterion, device, writer, epoch
        )
        print(f"Eval Loss: {eval_loss:.4f}")
        for k, v in eval_metrics.items():
            print(f"Eval {k}: {v:.4f}")

        # 记录学习率
        if writer:
            writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)

        # 保存最佳模型
        current_final_score = eval_metrics.get('final_score', 0.0)
        if current_final_score > best_score:
            best_score = current_final_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            print(f"保存最佳模型 - final score: {best_score:.4f}")

        # 早停检查
        if early_stopping(current_final_score):
            print(f"早停触发于 epoch {epoch+1} (patience=15)")
            break

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 final score: {best_score:.4f}")

    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")

    writer.close()
    return best_score


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 final score: {best_score:.4f} ##########")
