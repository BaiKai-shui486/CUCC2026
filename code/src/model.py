"""
CausalGRUStockScorer — 单向 GRU + 三路池化的轻量时序排序模型

设计原则:
  1. 单向 GRU (causal): 隐藏状态 h_t 只依赖 x_1...x_t，模拟实时预测
  2. 三路池化: last_hidden + max_pool(GRU_out) + mean_pool(GRU_out)
     全部基于 GRU 输出（已融合时序上下文），不做原始特征的池化
  3. 支持 pack_padded_sequence 处理变长序列（推理时停牌/新股场景）
  4. 参数量 ~200-300K，适合 10 万级 per-stock 样本
"""

import torch
import torch.nn as nn


class CausalGRUStockScorer(nn.Module):
    """
    单向 GRU 股票评分模型

    Input:  [B, L, F]  (batch, 序列长度=60, 特征数)
    Output: [B]         标量排序分数

    Pipeline:
      Input → Linear(F→d_model) → Unidirectional GRU →
      {last_hidden, max_pool(GRU_out), mean_pool(GRU_out)} → concat →
      MLP → score
    """

    def __init__(self, input_dim, d_model=192, gru_hidden=128,
                 gru_layers=2, dropout=0.15):
        super().__init__()

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 单向 GRU（causal）
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,  # 单向，天然 causal
            dropout=dropout if gru_layers > 1 else 0
        )

        # 三路池化拼接维度: gru_hidden * 3
        concat_dim = gru_hidden * 3

        # 排序 MLP 头
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x, lengths=None):
        """
        参数:
          x:       [B, L, F] 特征序列
          lengths: [B] 每个样本的实际长度（可选，用于 pack_padded_sequence）
                   如果为 None，假设所有样本长度 = L（无 padding）
        返回:
          scores: [B] 标量分数
        """
        # 输入投影
        x = self.input_proj(x)  # [B, L, d_model]

        # 优化 GRU 内存布局（避免每次调用重新排列权重）
        self.gru.flatten_parameters()

        # GRU 编码
        if lengths is not None:
            # 使用 pack_padded_sequence 忽略填充位
            lengths_cpu = lengths.cpu().to(torch.int64)
            # enforce_sorted=False: 输入不需要按长度降序排列
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            gru_out_packed, h_last = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out_packed, batch_first=True, total_length=x.size(1)
            )
        else:
            gru_out, h_last = self.gru(x)  # gru_out: [B, L, H], h_last: [D*L, B, H]

        # 三路池化（全部基于 GRU 输出，已融合时序上下文）
        # h_last shape: [num_layers * num_directions, B, H] = [gru_layers, B, H]
        last_hidden = h_last[-1]           # [B, H] — 最后一层最后时间步（最新信息）

        if lengths is not None:
            # 对变长序列，max/mean 只在有效长度内计算
            lengths = lengths.to(gru_out.device)
            mask = torch.arange(gru_out.size(1), device=gru_out.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # [B, L, 1]

            # Masked max
            masked_out = gru_out * mask + (1 - mask) * (-1e9)
            max_pooled = masked_out.max(dim=1)[0]  # [B, H]

            # Masked mean
            sum_out = (gru_out * mask).sum(dim=1)  # [B, H]
            mean_pooled = sum_out / lengths.unsqueeze(-1).float().clamp(min=1)  # [B, H]
        else:
            max_pooled = gru_out.max(dim=1)[0]   # [B, H] — 最大值（捕获强信号）
            mean_pooled = gru_out.mean(dim=1)    # [B, H] — 均值（捕获整体趋势）

        # 拼接三路池化
        combined = torch.cat([last_hidden, max_pooled, mean_pooled], dim=-1)  # [B, H*3]

        # MLP → 标量分数
        scores = self.mlp(combined).squeeze(-1)  # [B]
        return scores
