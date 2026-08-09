# THU-BDC2026 — CausalGRUStockScorer 股票排序模型（队伍：GEMN）

## 环境配置

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.10 ~ 3.12 | 推荐 3.12 |
| PyTorch | 2.6.0+cu124 | CUDA 12.4 |
| pandas | ≥ 2.3.0 | 数据处理 |
| numpy | ≥ 2.0.0 | 数值计算 |
| scikit-learn | ≥ 1.6.0 | 标准化 |
| TA-Lib | ≥ 0.6.0 | 技术指标（需先安装 C 库） |
| tensorboard | ≥ 2.20.0 | 训练日志 |
| tensorboardX | ≥ 2.6.0 | TensorBoard 写入器 |
| tqdm | ≥ 4.67.0 | 进度条 |
| joblib | ≥ 1.5.0 | 模型持久化 |

**环境安装**（三选一）：

```bash
# 方式 1: 直接加载预构建镜像（推荐，无需编译）
docker load -i app/docker/GEMN.tar
docker compose -f app/docker/docker-compose.yml up -d
docker exec -it dbc2026 bash /app/init.sh

# 方式 2: 本地构建镜像（需要网络下载 PyTorch 基础镜像）
bash app/docker/export.sh build     # 仅构建
# 或
bash app/docker/export.sh           # 构建 + 导出 GEMN.tar

# 方式 3: 本地 Conda
conda create -n THU-BDC python=3.12 -y
conda activate THU-BDC
bash init.sh
```

详见 `app/requirements.txt`。

## 数据

使用 **沪深 300 成分股** 日频行情数据（公开数据），数据获取自 比赛提供的[stock_data.csv](https://github.com/Sherlock1956/THU-BDC2026/blob/main/data/stock_data.csv)和[hs300_stock_list.csv](https://github.com/Sherlock1956/THU-BDC2026/blob/main/data/hs300_stock_list.csv)

- **原始数据**: `app/data/stock_data.csv` — 包含开盘价、收盘价、最高价、最低价、成交量、成交额、振幅、涨跌额、换手率、涨跌幅等基础字段
- **数据划分**: 由 `app/code/split_train_test.py` 按年份划分：
  - 训练集 `app/data/train.csv`：2026 年之前的数据
  - 测试集 `app/data/test.csv`：2026 年数据
- **股票列表**: `app/data/hs300_stock_list.csv`

## 预训练模型

本方案为**端到端训练**，不使用外部预训练模型。模型权重通过 Xavier 初始化，从零开始在训练集上训练。训练完成后模型保存至 `app/model/` 目录。

Walk-Forward 模式会产出每个 fold 的模型权重文件：
- `best_model_s{seed}.pth` — 单 fold 最佳模型
- `best_model_s{seed}_wf_{fold_name}.pth` — Walk-Forward 各 fold 模型

## 算法

### 整体思路介绍

将股票量化选股建模为 **Learning to Rank (LTR)** 问题：给定每只股票过去 30 个交易日的时序特征序列，预测该股票未来的超额收益排序分数。训练采用两阶段课程式策略，先用 MSE 回归做快速收敛，再用 LambdaRank 做排序优化。推理时按日期滚动预测，每窗口选取 Top-5 股票等权持仓。

### 方法的创新点

1. **Causal GRU 时序编码**：使用单向 GRU（非双向），保证 `h_t` 仅依赖 `x_1...x_t`，严格模拟实时预测场景，避免未来信息泄露
2. **三路池化**：对 GRU 输出序列同时做 `last_hidden`（最新信息）+ `max_pool`（强信号捕获）+ `mean_pool`（整体趋势），三种池化全部基于 GRU 隐状态（已融合时序上下文）
3. **两阶段课程训练**：Phase 1 MSE 快速收敛 → Phase 2 LambdaRank 排序优化，先学回归再学排序
4. **Walk-Forward 交叉验证**：严格按时间顺序划分训练/验证 fold，真实模拟滚动调仓回测

### 网络结构

```
Input: [B, L=30, F=197]          # 158 Alpha + 39 技术指标
  ↓
Linear(F → d_model=192)          # 输入投影
  ↓
Unidirectional GRU               # 2层, hidden=128, causal
  ↓
Three-way Pooling                # last_hidden + max_pool + mean_pool
  ↓  concat [B, 384]
MLP Head                         # 192 → 96 → 1
  ↓
Score [B]                        # 标量排序分数
```

- 参数量：约 200-300K
- 支持 `pack_padded_sequence` 处理变长序列（停牌/新股场景）

### 损失函数

**Phase 1 — MSE Loss**：标准均方误差回归损失，目标为未来收益率，用于快速验证 pipeline 和分数稳定性。

**Phase 2 — LambdaRank Loss** (`WeightedRankingLoss`)：
```
λ_ij = |1/(rank_i + 1) - 1/(rank_j + 1)|       # ΔNDCG 权重
L = Σ_{i,j: y_i > y_j} λ_ij · log(1 + exp(-σ·(s_i - s_j)))
```
- `σ` (sigma)：温度参数，控制 sigmoid 陡峭程度
- `top_k`：Top-K 聚焦增强，头部样本损失权重放大
- 按日期分组 pairwise 比较，同日期内股票互相对比

### 数据扩增

**随机截断（Random Truncation）**：训练时对每个样本在 `[min_seq_len, sequence_length]` 范围内随机截断序列长度，增强模型对不同历史窗口长度的鲁棒性，等效于时间维度的数据增强。

### 模型集成

**Multi-Seed 集成**：使用不同随机种子（`ensemble_seeds: [114514]`）训练多个模型，推理时对各模型输出的排序分数取均值。当仅一个 seed 时退化为单模型。

### 算法的其他细节

- **EMA (指数移动平均)**：训练过程中维护模型参数的 EMA 副本（decay=0.995），验证和推理时使用 EMA 参数
- **交叉截面标准化**：每个交易日对全部股票的特征做截面 z-score 标准化，消除市场整体波动对特征的影响
- **特征相关性过滤**：训练前按 `feature_corr_threshold` 剔除与标签相关性过低的特征
- **梯度裁剪**：`max_grad_norm=1.0`，防止梯度爆炸
- **Warmup**：学习率前 8% 步数线性预热，之后余弦退火

## 训练流程

`app/code/src/train.py` 的训练流程：

1. **加载配置**：读取 `config/model.json` 和 `config/train.json`
2. **加载数据**：读取 `data/train.csv`，解析日期
3. **股票 ID 映射**：为每只股票分配连续整数 ID
4. **特征工程**（多进程并行）：计算 158 Alpha 因子 + 39 技术指标
5. **数据集构建**：按股票分组，滑动窗口构建 (序列, 标签) 样本
6. **Walk-Forward 训练**（当 `walk_forward: true`）：
   - 按配置的 folds 严格按时间划分训练/验证集
   - 每个 fold 独立运行两阶段训练（MSE → LambdaRank）
   - 记录各 fold 的 best score 并计算均值
7. **最终模型训练**：使用全量训练数据训练生产模型
8. **输出**：模型保存至 `app/model/`，Walk-Forward 结果保存至 `app/output/walk_forward_scores.json`

```bash
# 启动训练
bash train.sh

# 或指定 seed / ensemble_id
python code/src/train.py --seed 114514 --ensemble_id 0
```

## 推理与回测

项目提供两个脚本，底层共享同一模型和特征工程管线：

| 脚本 | 用途 | 对应命令 |
|---|---|---|
| `code/src/featurework.py` | **向前推理** — 取截止最新日期的窗口，预测下一持有期 Top-5，产出最终 `result.csv` | `python code/src/featurework.py` |
| `code/src/test.py --mode single` | 同上（等价功能），仅预测最后一个窗口 | `bash test.sh --mode single` |
| `code/src/test.py`（默认） | **滚动回测** — 遍历全部可预测日期，逐窗口 Top-5 选股 → T+1 买 T+5 卖 → 业绩归因 | `bash test.sh` |

### 推理流程（`featurework.py` / `test.py --mode single`）

1. **加载配置**：读取 `config/model.json` 获取模型超参数和特征集配置
2. **加载测试数据**：读取 `data/test.csv`（截止最新日期），解析日期和股票列表
3. **特征工程**（多进程并行）：计算 158 Alpha + 39 技术指标
4. **数据清洗 + 截面标准化**：NaN/inf 前向填充，逐日 z-score 标准化
5. **构建预测窗口**：每只股票取最后 `seq_length` 天序列，不足的左侧零填充
6. **模型加载**：自动匹配 `model/` 下最佳模型权重（优先 `best_model_s{seed}.pth`）
7. **模型推理**：批量推理输出排序分数，支持变长序列
8. **可交易性过滤**：排除最后交易日一字涨停（gap ≥ 9.5%）的股票，按分数降序取 Top-5
9. **Softmax 权重分配**（temperature=10），归一化使权重和 ≤ 1
10. **输出**：`app/output/result.csv`（stock_id, weight）

```bash
# 向前推理预测
python code/src/featurework.py
# 或等价地
bash test.sh --mode single
```

### 回测流程（`test.py` 默认模式）

在推理基础上，遍历测试集所有可预测窗口（共 T-5 天），对每个窗口重复 推理→Top-5 选股→模拟交易，最后汇总业绩指标：

1. 构建全量 (stock × date) 矩阵，预计算每个 `(股票, 日期)` 的 T+1→T+5 前向收益率
2. 预计算可交易性掩码（T+1 一字涨停/停牌 → 不可买入）
3. 为每个可预测日期运行一次推理 → Top-5 选股
4. 记录每窗口的实际收益率，扣除双边交易成本（0.2%）
5. **业绩归因**：平均日收益率、胜率、累计净值曲线、最大回撤、夏普比率（年化×√52）、Calmar 比率、盈亏比
6. **输出文件**：
   - `app/output/rolling_predictions.csv` — 每日预测明细
   - `app/output/rolling_report.txt` — 滚动业绩报告
   - `app/output/walk_forward_scores.json` — Walk-Forward 各 fold 评分（训练时产出）

```bash
# 完整滚动回测 + 业绩报告
bash test.sh
```

## 其他注意事项

- **数据划分**：严格按年份划分训练/测试集（2026 年为测试集），Walk-Forward folds 按日期顺序无重叠划分
- **停牌/新股处理**：支持变长序列（`pack_padded_sequence`），长度不足 `min_seq_len` 的样本会被过滤
- **交易约束**：T+1 开盘涨幅 ≥ 9.5% 视为一字涨停不可买入；双边交易成本 0.2%（可配置）
- **复现性**：所有随机种子固定（`seed=114514`），PyTorch deterministic 模式开启
- **GPU 检测**：运行 `python code/test-GPU.py` 可独立检测 GPU 是否可用
