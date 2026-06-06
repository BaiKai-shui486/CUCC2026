"""
训练脚本 V2 — Per-Stock 样本 + 单向 GRU 模型

两阶段:
  Phase 1 (MSE):  标准回归损失，快速验证 pipeline 和 score 稳定性
  Phase 2 (LambdaRank): 按日期分组 pairwise 排序损失，提升排序上限

配置文件:
    config/model.json  - 模型超参数
    config/train.json  - 训练超参数

使用方法:
    python train.py
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
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import CausalGRUStockScorer
from utils import (
    FEATURE_COLUMNS_MAP,
    FEATURE_ENGINEER_FUNC_MAP,
    cross_sectional_standardize,
    create_per_stock_dataset,
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

    config = {**model_config, **train_config}
    return config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# EMA
# ============================================================================

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ============================================================================
# 损失函数
# ============================================================================

class WeightedRankingLoss(nn.Module):
    """
    LambdaRank: pairwise logistic + |Δrank| 权重 + Top-K 聚焦

    对涉及真实 Top-K 股票的 pair 赋予更高权重，让梯度集中优化头部排序，
    减少尾部股票对之间的无效梯度。
    """
    def __init__(self, sigma=1.0, top_k=5, topk_weight=5.0):
        super().__init__()
        self.sigma = sigma
        self.top_k = top_k
        self.topk_weight = topk_weight

    def forward(self, y_pred, y_true, masks=None):
        if masks is not None:
            masks = masks.float()
            y_pred = y_pred * masks + (1 - masks) * (-1e4)
            y_true = y_true * masks
        else:
            masks = torch.ones_like(y_pred)

        pred_diff = y_pred.unsqueeze(-1) - y_pred.unsqueeze(-2)  # (..., N, N)
        true_diff = y_true.unsqueeze(-1) - y_true.unsqueeze(-2)

        pair_mask = (true_diff != 0).float()
        if masks is not None:
            pair_mask = pair_mask * masks.unsqueeze(-1) * masks.unsqueeze(-2)

        lambda_weights = torch.abs(true_diff)

        # Top-K 聚焦：涉及头部股票的 pair 获得更高权重
        if self.top_k > 0 and y_true.size(-1) > self.top_k:
            k = min(self.top_k, y_true.size(-1))
            _, topk_idx = torch.topk(y_true, k, dim=-1)
            topk_mask = torch.zeros_like(y_true)
            topk_mask.scatter_(-1, topk_idx, 1.0)
            # pair_has_topk[i,j] = 1 当 i 或 j 在 top-K 中
            pair_has_topk = (topk_mask.unsqueeze(-1) + topk_mask.unsqueeze(-2) > 0).float()
            # 非 top-K 的 pair 降权，top-K 相关 pair 加权
            boost = 1.0 + (self.topk_weight - 1.0) * pair_has_topk
            lambda_weights = lambda_weights * boost

        sign = torch.sign(true_diff)
        logistic_loss = torch.log(1.0 + torch.exp(-self.sigma * sign * pred_diff))

        weighted_loss = logistic_loss * lambda_weights * pair_mask
        num_pairs = pair_mask.sum().clamp(min=1)
        return weighted_loss.sum() / num_pairs


# ============================================================================
# 排序指标
# ============================================================================

def calculate_ranking_metrics(y_pred, y_true, k=5):
    """计算 Top-5 final_score (单 batch，不需要 masks)"""
    valid_pred = y_pred
    valid_true = y_true

    if len(valid_pred) < k:
        return {'final_score': 0.0, 'pred_return_sum': 0.0, 'max_return_sum': 0.0,
                'ratio_pred': 0.0, 'ratio_random': 0.0, 'random_return_sum': 0.0}

    _, pred_indices = torch.topk(valid_pred, k)
    pred_top_returns = valid_true[pred_indices]
    pred_return_sum = pred_top_returns.sum().item()

    _, true_indices = torch.topk(valid_true, k)
    true_top_returns = valid_true[true_indices]
    max_return_sum = true_top_returns.sum().item()

    random_return_sum = k * valid_true.mean().item()

    ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
    ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
    denom = max_return_sum - random_return_sum
    final_score = (pred_return_sum - random_return_sum) / (denom + 1e-12) if abs(denom) > 1e-6 else 0.0

    return {
        'pred_return_sum': pred_return_sum,
        'max_return_sum': max_return_sum,
        'random_return_sum': random_return_sum,
        'ratio_pred': ratio_pred,
        'ratio_random': ratio_random,
        'final_score': final_score,
    }


# ============================================================================
# 数据集
# ============================================================================

class PerStockDataset(Dataset):
    """Phase 1 MSE: 每个样本独立，可随机 shuffle，支持随机截断"""
    def __init__(self, sequences, targets, min_seq_len=None):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = sequences.shape[1]
        self.min_seq_len = min_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].clone()  # (L, F)
        target = self.targets[idx]

        if self.min_seq_len is not None and self.min_seq_len < self.seq_len:
            actual_len = random.randint(self.min_seq_len, self.seq_len)
            # 左填充零：模拟较短的历史序列（最近 actual_len 天保留在右侧）
            seq[:self.seq_len - actual_len] = 0
        else:
            actual_len = self.seq_len

        return seq, target, actual_len


class DateGroupedDataset(Dataset):
    """Phase 2 LambdaRank: 按 window_end_date 分组，支持随机截断（同日股票截断长度一致）"""
    def __init__(self, sequences, targets, window_end_dates, min_seq_len=None):
        self.sequences = sequences
        self.targets = targets
        self.seq_len = sequences.shape[1]
        self.min_seq_len = min_seq_len
        # 将日期转为字符串 key
        self.dates = pd.to_datetime(window_end_dates).strftime('%Y-%m-%d').values
        self.unique_dates = np.unique(self.dates)

        # 预构建 date → indices 映射
        self.date_to_indices = {}
        for i, d in enumerate(self.dates):
            self.date_to_indices.setdefault(d, []).append(i)

    def __len__(self):
        return len(self.unique_dates)

    def __getitem__(self, idx):
        date = self.unique_dates[idx]
        indices = self.date_to_indices[date]
        seqs = torch.FloatTensor(self.sequences[indices].copy())   # (N_day, L, F)
        tgt = torch.FloatTensor(self.targets[indices])              # (N_day,)

        n_stocks = len(indices)
        if self.min_seq_len is not None and self.min_seq_len < self.seq_len:
            # 同一日期所有股票使用完全相同的截断长度
            actual_len = random.randint(self.min_seq_len, self.seq_len)
            seqs[:, :self.seq_len - actual_len, :] = 0
        else:
            actual_len = self.seq_len

        lengths = torch.LongTensor([actual_len] * n_stocks)  # (N_day,)
        return seqs, tgt, date, lengths


def collate_per_stock(batch):
    """标准 collate: 所有样本同 shape，支持变长序列"""
    seqs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    lengths = torch.LongTensor([item[2] for item in batch])
    return seqs, targets, lengths


def collate_date_grouped(batch):
    """Phase 2: 每个 batch 就是一天的数据，含变长信息"""
    # batch 里只有一个元素（因为 batch_size=1，每个 date 一个 batch）
    return batch[0][0], batch[0][1], batch[0][2], batch[0][3]


# ============================================================================
# 验证函数（按日期聚合评估）
# ============================================================================

@torch.no_grad()
def evaluate_on_val(model, val_sequences, val_targets, val_dates, device,
                     criterion=None, val_tradable=None):
    """
    按日期聚合验证：对每天的所有股票独立打分，然后按天计算 final_score。
    支持可交易性过滤：排除 T+1 一字涨停无法买入的股票。
    """
    model.eval()
    dates_str = pd.to_datetime(val_dates).strftime('%Y-%m-%d').values
    unique_dates = np.unique(dates_str)

    all_metrics = []
    total_loss = 0
    num_dates = 0

    for date in unique_dates:
        mask = dates_str == date
        indices = np.where(mask)[0]

        # 可交易性过滤：只保留 T+1 开盘可买入的股票
        if val_tradable is not None:
            tradable_indices = indices[val_tradable[indices]]
            if len(tradable_indices) < 5:
                continue  # 可交易股票不足 5 只，跳过该日
            indices = tradable_indices

        seqs = torch.FloatTensor(val_sequences[indices]).to(device)
        tgt = torch.FloatTensor(val_targets[indices]).to(device)

        scores = model(seqs)  # (N_day,)

        if criterion is not None:
            loss = criterion(scores, tgt)
            total_loss += loss.item()

        metrics = calculate_ranking_metrics(scores, tgt, k=5)
        all_metrics.append(metrics)
        num_dates += 1

    # 聚合所有日期的指标
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    if criterion is not None:
        avg_metrics['loss'] = total_loss / max(num_dates, 1)

    return avg_metrics


# ============================================================================
# 训练循环
# ============================================================================

def train_phase1_mse(model, train_loader, val_data, optimizer, scheduler,
                     device, config, writer, model_dir, output_dir, save_suffix=''):
    """
    Phase 1: MSE 回归训练
    - 标准 shuffle DataLoader
    - 按 samples_processed 设置验证频率
    - 保留 EMA、早停、warmup+cosine LR
    """
    val_sequences, val_targets, val_dates, val_tradable = val_data
    criterion_mse = nn.MSELoss()
    ema_decay = config.get('ema_decay', 0.995)
    ema_model = EMAModel(model, decay=ema_decay)
    use_ema = config.get('use_ema', True)

    accum_steps = config.get('accum_steps', 1)
    val_freq = config.get('val_freq_samples', 2000)
    patience = config.get('early_stop_patience', 30)
    best_score = -float('inf')
    best_epoch = -1
    early_stop_counter = 0
    samples_seen = 0
    global_step = 0
    optimizer_step = 0

    print(f"Phase 1 MSE 训练开始")
    print(f"  训练样本: {len(train_loader.dataset)}, batch_size: {config['batch_size']}")
    print(f"  验证频率: 每 {val_freq} 样本, 早停 patience: {patience}")

    last_val_samples = 0

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        epoch_samples = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch_idx, (seqs, targets, lengths) in enumerate(pbar):
            seqs = seqs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            # 前向（传入 lengths 以正确处理变长序列）
            scores = model(seqs, lengths=lengths)
            loss = criterion_mse(scores, targets) / accum_steps
            loss.backward()

            epoch_loss += loss.item() * accum_steps
            epoch_samples += seqs.size(0)
            samples_seen += seqs.size(0)

            # 梯度累积
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                optimizer_step += 1

                if use_ema:
                    ema_model.update(model)

                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=optimizer_step)

                global_step += 1

            pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}', 'samples': samples_seen})

            # 按样本数验证（用 last_val_samples 避免每个 batch 都触发）
            if samples_seen - last_val_samples >= val_freq:
                # 验证
                if use_ema:
                    ema_model.apply(model)

                val_metrics = evaluate_on_val(model, val_sequences, val_targets, val_dates,
                                              device, criterion=criterion_mse,
                                              val_tradable=val_tradable)

                if use_ema:
                    ema_model.restore(model)

                current_score = val_metrics.get('final_score', 0.0)
                val_loss = val_metrics.get('loss', float('inf'))

                if writer:
                    writer.add_scalar('eval/loss', val_loss, global_step=optimizer_step)
                    for k, v in val_metrics.items():
                        if k != 'loss':
                            writer.add_scalar(f'eval/{k}', v, global_step=optimizer_step)

                print(f"\n  [验证 @{samples_seen}样本] loss={val_loss:.4f}, "
                      f"final_score={current_score:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

                last_val_samples = samples_seen

                # 保存最佳
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch + 1
                    early_stop_counter = 0
                    if use_ema:
                        ema_model.apply(model)
                        torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_s{config["seed"]}{save_suffix}.pth'))
                        ema_model.restore(model)
                    else:
                        torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_s{config["seed"]}{save_suffix}.pth'))
                    print(f"  >>> 保存最佳模型, final_score={best_score:.4f}")
                else:
                    early_stop_counter += 1

                model.train()

                if early_stop_counter >= patience:
                    print(f"早停触发 @{samples_seen} 样本 (patience={patience})")
                    break

        if early_stop_counter >= patience:
            break

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} 完成: avg_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\nPhase 1 完成! 最佳 final_score={best_score:.4f} (epoch {best_epoch})")
    with open(os.path.join(output_dir, f'final_score_s{config["seed"]}{save_suffix}.txt'), 'w') as f:
        f.write(f"Phase: 1 (MSE)\nBest epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")
    return best_score


def train_phase2_lambdarank(model, train_loader, val_data, optimizer, scheduler,
                             device, config, writer, model_dir, output_dir, save_suffix=''):
    """
    Phase 2: LambdaRank 训练
    - 按日期分组 batch (每个 batch = 一天所有股票)
    - 同日股票 pairwise 比较
    """
    val_sequences, val_targets, val_dates, val_tradable = val_data
    criterion_rank = WeightedRankingLoss(
        sigma=config.get('lambda_sigma', 2.0),
        top_k=config.get('loss_top_k', 5),
        topk_weight=config.get('loss_topk_weight', 5.0)
    )
    ema_decay = config.get('ema_decay', 0.995)
    ema_model = EMAModel(model, decay=ema_decay)
    use_ema = config.get('use_ema', True)

    accum_steps = config.get('accum_steps', 2)
    patience = config.get('early_stop_patience', 30)
    best_score = -float('inf')
    best_epoch = -1
    early_stop_counter = 0
    global_step = 0

    print(f"Phase 2 LambdaRank 训练开始")
    print(f"  训练日期数: {len(train_loader.dataset)}, 梯度累积: {accum_steps}")

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch_idx, (seqs, targets, date_str, lengths) in enumerate(pbar):
            seqs = seqs.squeeze(0).to(device)      # (1, N, L, F) → (N, L, F)
            targets = targets.squeeze(0).to(device)  # (1, N) → (N,)
            lengths = lengths.squeeze(0).to(device)   # (1, N) → (N,)

            scores = model(seqs, lengths=lengths)  # (N,)

            loss = criterion_rank(scores, targets) / accum_steps
            loss.backward()

            epoch_loss += loss.item() * accum_steps

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if use_ema:
                    ema_model.update(model)

                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=global_step)
                    writer.add_scalar('train/loss', loss.item() * accum_steps, global_step=global_step)

                global_step += 1

            pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})

        # 每个 epoch 结束时验证
        if use_ema:
            ema_model.apply(model)

        val_metrics = evaluate_on_val(model, val_sequences, val_targets, val_dates,
                                      device, criterion=None, val_tradable=val_tradable)
        if use_ema:
            ema_model.restore(model)

        current_score = val_metrics.get('final_score', 0.0)

        if writer:
            for k, v in val_metrics.items():
                writer.add_scalar(f'eval/{k}', v, global_step=epoch)

        print(f"Epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f}, "
              f"final_score={current_score:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            early_stop_counter = 0
            if use_ema:
                ema_model.apply(model)
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_s{config["seed"]}{save_suffix}.pth'))
                ema_model.restore(model)
            else:
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_s{config["seed"]}{save_suffix}.pth'))
            print(f"  >>> 保存最佳模型, final_score={best_score:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"早停触发 @epoch {epoch+1} (patience={patience})")
                break

    print(f"\nPhase 2 完成! 最佳 final_score={best_score:.4f} (epoch {best_epoch})")
    with open(os.path.join(output_dir, f'final_score_s{config["seed"]}{save_suffix}.txt'), 'a') as f:
        f.write(f"\nPhase: 2 (LambdaRank)\nBest epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")
    return best_score


# ============================================================================
# 单 Fold 训练流水线（可复用，无跨期特征泄露）
# ============================================================================

def run_training_pipeline(train_df, val_df, val_start, stockid2idx, config,
                          device, model_dir, output_dir, fold_name, is_final=False):
    """
    完整的单 fold 训练流水线。

    对 train_df/val_df 独立进行特征工程和截面标准化，确保无跨期泄露。
    训练集使用随机截断（min_seq_len），验证集使用完整序列。

    参数:
      train_df:    训练集 DataFrame (仅含训练日期)
      val_df:      验证集 DataFrame (含上下文日期)
      val_start:   验证窗口起始日 (用于 min_window_end_date 隔离)
      stockid2idx: 股票代码→整数索引映射
      fold_name:   Fold 名称 (用于日志和模型保存)
      is_final:    是否保存为最终生产模型
    返回:
      best_score:  float, 验证集最佳 final_score
    """
    # 特征工程配置
    feature_engineer = FEATURE_ENGINEER_FUNC_MAP[config['feature_num']]
    feature_columns = FEATURE_COLUMNS_MAP[config['feature_num']]

    def preprocess(df, desc):
        df = df.copy()
        df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

        print(f"  多进程特征工程 ({desc})...")
        groups = [group for _, group in df.groupby('股票代码', sort=False)]
        num_processes = min(10, mp.cpu_count())
        with mp.Pool(processes=num_processes) as pool:
            processed_list = list(tqdm(pool.imap(feature_engineer, groups),
                                       total=len(groups), desc=f"  {desc}"))
        processed = pd.concat(processed_list).reset_index(drop=True)

        # 映射股票索引
        processed['instrument'] = processed['股票代码'].map(stockid2idx)
        processed = processed.dropna(subset=['instrument']).copy()
        processed['instrument'] = processed['instrument'].astype(np.int64)

        # 标签构建
        processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
        processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)
        if desc == '训练集':
            processed = processed[processed['open_t1'] > 1e-4]
        processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
        processed = processed.dropna(subset=['label'])

        # Winsorize
        low, high = processed['label'].quantile([0.005, 0.995]).values
        processed['label'] = processed['label'].clip(low, high)
        processed.drop(columns=['open_t1', 'open_t5'], inplace=True)

        return processed

    # 独立特征工程（train/val 各自处理，无跨期泄露）
    train_data = preprocess(train_df, f'{fold_name} 训练集')
    val_data = preprocess(val_df, f'{fold_name} 验证集')

    # 确定实际特征列
    exclude_cols = {'日期', '股票代码', 'instrument', 'label', 'datetime'}
    actual_features = [f for f in feature_columns if f in train_data.columns and f not in exclude_cols]
    print(f"  [{fold_name}] 特征数: {len(actual_features)}")

    # 处理 inf/NaN
    for df in [train_data, val_data]:
        df[actual_features] = df[actual_features].replace([np.inf, -np.inf], np.nan)
        df[actual_features] = df[actual_features].ffill().fillna(0)

    # 统一日期列名
    for df in [train_data, val_data]:
        df.rename(columns={'日期': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])

    # 独立截面标准化（train/val 各自标准化，无跨期泄露）
    print(f"  [{fold_name}] 截面标准化...")
    train_data = cross_sectional_standardize(train_data, actual_features)
    val_data = cross_sectional_standardize(val_data, actual_features)

    # Per-Stock 数据集创建
    train_sequences, train_targets, train_stock_ids, train_dates, train_tradable = \
        create_per_stock_dataset(train_data, actual_features, config['sequence_length'])

    val_sequences, val_targets, val_stock_ids, val_dates, val_tradable = \
        create_per_stock_dataset(val_data, actual_features, config['sequence_length'],
                                  min_window_end_date=val_start.strftime('%Y-%m-%d'))

    # 窗口结束日隔离断言
    max_train_end = pd.to_datetime(train_dates).max()
    min_val_end = pd.to_datetime(val_dates).min()
    assert max_train_end < min_val_end, \
        f"[{fold_name}] 窗口结束日重叠! train max end={max_train_end.date()}, val min end={min_val_end.date()}"

    print(f"  [{fold_name}] 训练样本: {len(train_sequences)}, 验证样本: {len(val_sequences)}")
    print(f"  [{fold_name}] 训练窗口: {pd.to_datetime(train_dates).min().date()} ~ {max_train_end.date()}")
    print(f"  [{fold_name}] 验证窗口: {min_val_end.date()} ~ {pd.to_datetime(val_dates).max().date()}")

    # 标签截面去均值（MSE 阶段用）
    train_targets_centered = train_targets.copy()
    val_targets_centered = val_targets.copy()
    train_dates_str = pd.to_datetime(train_dates).strftime('%Y-%m-%d').values
    val_dates_str = pd.to_datetime(val_dates).strftime('%Y-%m-%d').values

    for date in np.unique(train_dates_str):
        mask = train_dates_str == date
        train_targets_centered[mask] -= train_targets[mask].mean()
    for date in np.unique(val_dates_str):
        mask = val_dates_str == date
        val_targets_centered[mask] -= val_targets[mask].mean()

    # 模型初始化
    model = CausalGRUStockScorer(
        input_dim=len(actual_features),
        d_model=config.get('d_model', 192),
        gru_hidden=config.get('gru_hidden', 128),
        gru_layers=config.get('gru_layers', 2),
        dropout=config.get('dropout', 0.15)
    )
    model.to(device)
    print(f"  [{fold_name}] 模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 随机截断参数（仅训练集，验证集使用完整序列）
    use_truncation = config.get('use_random_truncation', False)
    train_min_seq_len = config.get('min_seq_len', None) if use_truncation else None

    # 优化器 & 调度器 & 数据加载器
    train_phase = config.get('train_phase', 1)

    if train_phase == 1:
        train_dataset = PerStockDataset(train_sequences, train_targets_centered,
                                         min_seq_len=train_min_seq_len)
        train_loader = DataLoader(
            train_dataset, batch_size=config.get('batch_size', 128),
            shuffle=True, collate_fn=collate_per_stock, num_workers=0, pin_memory=False
        )
        total_steps = (len(train_loader) + config.get('accum_steps', 1) - 1) // config.get('accum_steps', 1) \
                       * config['num_epochs']
        val_data_tuple = (val_sequences, val_targets_centered, val_dates, val_tradable)
    else:
        train_dataset = DateGroupedDataset(train_sequences, train_targets, train_dates,
                                            min_seq_len=train_min_seq_len)
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            collate_fn=collate_date_grouped, num_workers=0, pin_memory=False
        )
        total_steps = len(train_loader) * config['num_epochs']
        val_data_tuple = (val_sequences, val_targets, val_dates, val_tradable)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )

    warmup_steps = int(config.get('warmup_ratio', 0.08) * total_steps)
    print(f"  [{fold_name}] LR 调度: total_steps={total_steps}, warmup_steps={warmup_steps}")

    def lr_lambda(current_step):
        step = current_step + 1
        if step <= warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # TensorBoard (按 fold 分目录)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log', fold_name))

    # 模型保存后缀 (walk-forward fold 加前缀，最终模型不加)
    save_suffix = f'_wf_{fold_name}' if (config.get('walk_forward') and not is_final) else ''

    if train_phase == 1:
        best_score = train_phase1_mse(
            model, train_loader, val_data_tuple, optimizer, scheduler,
            device, config, writer, model_dir, output_dir, save_suffix=save_suffix
        )
    else:
        best_score = train_phase2_lambdarank(
            model, train_loader, val_data_tuple, optimizer, scheduler,
            device, config, writer, model_dir, output_dir, save_suffix=save_suffix
        )

    writer.close()

    if is_final:
        print(f"  [{fold_name}] 最终生产模型训练完成, best_score={best_score:.4f}")

    return best_score


# ============================================================================
# Walk-Forward 验证编排
# ============================================================================

def train_with_walk_forward(config, full_df, stockid2idx, device,
                             model_dir, output_dir):
    """
    Walk-Forward 验证：多 fold 训练 + 最终生产模型。

    每个 fold 使用独立的时间段，train/val 特征工程完全隔离。
    最终模型使用全部训练数据（直到原 val_start）。
    """
    folds = config.get('walk_forward_folds', [])
    fold_scores = []
    fold_details = []

    print(f"\n{'=' * 70}")
    print(f"  Walk-Forward 验证 ({len(folds)} folds)")
    print(f"{'=' * 70}")

    for fold_idx, fold in enumerate(folds):
        fold_name = fold['name']
        train_end = pd.Timestamp(fold['train_end'])
        val_start = pd.Timestamp(fold['val_start'])
        val_end = pd.Timestamp(fold['val_end'])
        val_context_start = val_start - pd.tseries.offsets.BDay(config['sequence_length'] - 1)

        print(f"\n{'=' * 60}")
        print(f"  Fold {fold_idx + 1}/{len(folds)}: {fold_name}")
        print(f"    训练截止: {train_end.date()}, 验证期: {val_start.date()} ~ {val_end.date()}")
        print(f"{'=' * 60}")

        # 划分 fold 数据（严格按日期，train/val 无重叠）
        fold_train = full_df[full_df['日期'] <= train_end].copy()
        fold_val = full_df[(full_df['日期'] >= val_context_start) &
                            (full_df['日期'] <= val_end)].copy()

        if len(fold_train) == 0 or len(fold_val) == 0:
            print(f"  ⚠ [{fold_name}] 数据为空，跳过")
            continue

        score = run_training_pipeline(
            fold_train, fold_val, val_start, stockid2idx, config,
            device, model_dir, output_dir, fold_name, is_final=False
        )
        fold_scores.append(score)
        fold_details.append({'name': fold_name, 'score': float(score)})
        print(f"  [{fold_name}] best_score = {score:.4f}")

    # Walk-Forward 汇总
    if fold_scores:
        avg_score = float(np.mean(fold_scores))
        print(f"\n{'=' * 70}")
        print(f"  Walk-Forward 验证结果")
        print(f"{'=' * 70}")
        for detail in fold_details:
            print(f"  {detail['name']}: final_score = {detail['score']:.4f}")
        print(f"  平均 final_score: {avg_score:.4f}")
        print(f"{'=' * 70}")

        # 保存 Walk-Forward 结果
        wf_results = {
            'folds': fold_details,
            'avg_score': avg_score,
            'n_folds': len(fold_scores)
        }
        wf_path = os.path.join(output_dir, 'walk_forward_scores.json')
        with open(wf_path, 'w') as f:
            json.dump(wf_results, f, indent=2, ensure_ascii=False)
        print(f"  Walk-Forward 结果已保存: {wf_path}")
    else:
        avg_score = 0.0
        print("\n  ⚠ 无有效 Walk-Forward fold")

    # ========================================================================
    # 最终生产模型：使用全部训练数据（直到原 val_start）
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"  训练最终生产模型（全量训练数据）")
    print(f"{'=' * 70}")

    last_date = full_df['日期'].max()
    final_val_start = (last_date - pd.DateOffset(months=config.get('val_months', 2))).normalize()
    final_val_context_start = final_val_start - pd.tseries.offsets.BDay(config['sequence_length'] - 1)

    final_train = full_df[full_df['日期'] < final_val_start].copy()
    final_val = full_df[full_df['日期'] >= final_val_context_start].copy()

    print(f"  最终训练集: {final_train['日期'].min().date()} ~ {final_train['日期'].max().date()}")
    print(f"  最终验证集: {final_val['日期'].min().date()} ~ {final_val['日期'].max().date()}")
    print(f"  验证起始: {final_val_start.date()}")

    final_score = run_training_pipeline(
        final_train, final_val, final_val_start, stockid2idx, config,
        device, model_dir, output_dir, 'Final', is_final=True
    )

    return avg_score, final_score


# ============================================================================
# 主函数
# ============================================================================

def main():
    config = load_config()
    set_seed(config.get('seed', 42))

    src_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.dirname(os.path.dirname(src_dir))
    model_dir = os.path.join(app_root, 'model')
    output_dir = os.path.join(app_root, 'output')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用设备: CUDA - {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用设备: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("使用设备: CPU")

    # ========================================================================
    # 1. 数据加载
    # ========================================================================
    data_file = os.path.join(app_root, 'data', 'train.csv')
    full_df = pd.read_csv(data_file)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    full_df = full_df.sort_values(['日期', '股票代码']).reset_index(drop=True)
    print(f"全量数据: {full_df['日期'].min().date()} ~ {full_df['日期'].max().date()}")

    # ========================================================================
    # 2. 股票 ID 映射（全量数据，所有 fold 共享）
    # ========================================================================
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    print(f"股票数量: {len(stockid2idx)}")

    # 验证特征集配置
    assert config['feature_num'] in FEATURE_ENGINEER_FUNC_MAP, \
        f"不支持的特征集: {config['feature_num']}"

    # ========================================================================
    # 3. 训练（Walk-Forward 或 单次）
    # ========================================================================
    if config.get('walk_forward', False):
        avg_score, final_score = train_with_walk_forward(
            config, full_df, stockid2idx, device, model_dir, output_dir
        )
        return final_score
    else:
        # 原始单 fold 逻辑（通过 run_training_pipeline）
        last_date = full_df['日期'].max()
        val_start = (last_date - pd.DateOffset(months=config.get('val_months', 2))).normalize()
        val_context_start = val_start - pd.tseries.offsets.BDay(config['sequence_length'] - 1)

        train_df = full_df[full_df['日期'] < val_start].copy()
        val_df = full_df[full_df['日期'] >= val_context_start].copy()

        print(f"训练集: {train_df['日期'].min().date()} ~ {train_df['日期'].max().date()}")
        print(f"验证集 (含上下文): {val_df['日期'].min().date()} ~ {val_df['日期'].max().date()}")
        print(f"验证集目标期起始 (val_start): {val_start.date()}")

        best_score = run_training_pipeline(
            train_df, val_df, val_start, stockid2idx, config,
            device, model_dir, output_dir, 'default', is_final=True
        )
        return best_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='覆盖配置文件中的 seed')
    parser.add_argument('--ensemble_id', type=int, default=None, help='集成训练的模型编号')
    args, _ = parser.parse_known_args()

    mp.set_start_method('spawn', force=True)

    # 如果指定了 seed，修改配置
    if args.seed is not None:
        import json
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'model.json')
        with open(config_path) as f:
            cfg = json.load(f)
        cfg['seed'] = args.seed
        if args.ensemble_id is not None:
            cfg['ensemble_id'] = args.ensemble_id
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"Seed 已设置为: {args.seed}")

    best_score = main()
    print(f"\n########## 训练完成！最佳 final score: {best_score:.4f} ##########")
