# Assignment 3 Scaling Note

这份笔记面向第一次做 scaling-law 作业的读者。目标不是复述 PDF，而是把 Assignment3 的任务拆成一条可执行路线：先理解 scaling-law 问题，再理解代码框架，最后知道如何用 API 提交实验和最终方案。

## 1. 全局地图

Assignment3 的核心问题是：在有限训练预算下，怎样选择模型规模、训练 token 数和训练配置，让最终验证 loss 尽可能低。作业不是让你从零写一个训练系统，而是给你一个训练 API、调度器和一份 synthetic IsoFLOPs 数据，让你用 scaling laws 指导实验。

本目录的关键文件如下：

- `cs336_assignment3_scaling.pdf`：作业说明，先读它。它定义了 scaling laws、IsoFLOPs、训练 API、budget 和 leaderboard。
- `README.md`：环境启动方式、学生 API key、hosted API 地址。
- `data/isoflops_curves.json`：合成 IsoFLOPs 曲线数据，用来练习从固定 compute budget 曲线中找到最优模型规模。
- `scripts/fit_isoflops.py`：本地新增的 scaling-law 分析脚本。
- `cs336_scaling/client.py`：学生侧调用 hosted API 的客户端。
- `cs336_scaling/api/public.py`：公开 API，包括 submit、budget、experiments、final_submission。
- `cs336_scaling/scheduler/experiment_selector.py`：服务端调度逻辑，决定哪些 queued 实验被派发。
- `cs336_scaling/training/training_config.py`：训练配置的校验入口。
- `tests/`：API、budget、去重、final submission 和调度公平性的测试。

可以把整个系统看成这条链路：

```text
TrainingConfig
  -> POST /submit
  -> 数据库记录 queued experiment
  -> dispatcher 选择 queued jobs
  -> Modal/B200 worker 训练
  -> worker 回报 val loss / finish status
  -> GET /experiments 查看结果
  -> POST /final_submission 提交最终方案
```

## 2. 先理解 Scaling Laws

### 2.1 什么是 IsoFLOPs

IsoFLOPs 的意思是：固定总训练 compute budget `C`，改变模型参数量 `N` 和训练 token 数 `D`，观察最终验证 loss。

对 dense Transformer，常用近似是：

```text
C ≈ 6 * N * D
```

所以在固定 `C` 时，模型越大，能训练的 token 数通常越少；模型越小，能训练的 token 数越多。loss 曲线通常会出现一个最低点，这个点就是该 compute budget 下的 compute-optimal 配置。

### 2.2 本地 synthetic 数据怎么用

`data/isoflops_curves.json` 有 72 个点：9 个 compute budget，每个 budget 8 个模型规模。新增脚本会做三件事：

1. 按 `compute_budget` 分组。
2. 在每组里选择 `final_loss` 最低的点。
3. 拟合三条简单规律：

```text
N_opt(C) = a_N * C^b_N
D_opt(C) = a_D * C^b_D
L_opt(C) = E + A * C^(-alpha)
```

运行方式：

```sh
uv run python scripts/fit_isoflops.py --target-compute-budget 1e22
```

当前 synthetic 数据给出的拟合结果是：

```text
N_opt(C) = 1.163411e+00 * C^0.468683
D_opt(C) = 1.432570e-01 * C^0.531317
L_opt(C) = 2.708387 + 6.530819e+03 * C^-0.176344
```

如果外推到 `C=1e22`，脚本估计：

```text
parameters  ≈ 2.38e10
train_tokens ≈ 7.00e10
final_loss   ≈ 3.570
```

这只是 synthetic curves 的基线，不等于 hosted API 的最终答案。真正的最终提交还需要用你的 12 B200-hour budget 做实验校准。

## 3. 人类完成路径

### 3.1 读 PDF 时抓住四件事

第一，作业的目标不是训练最大模型，而是在预算内找到最优 loss。大模型如果 token 不够，会 undertrained；小模型如果参数不够，会 capacity-limited。

第二，budget 是按实验预留的。`POST /submit` 时，服务端会把 `max_runtime_seconds` 计入预算；queued/running 实验按最大运行时长占用预算，completed/failed 实验按实际使用时间计入，但至少计 1 秒且不超过 `max_runtime_seconds`。

第三，API 是工作流入口。你需要提交多个 `TrainingConfig`，查看每个实验的 validation loss，再基于结果更新下一轮实验设计。

第四，最终提交由 `POST /final_submission` 保存：它包含一个完整 `training_config` 和你预测的 `predicted_final_loss`。

### 3.2 设计实验的建议流程

先用 synthetic curves 建立直觉：

```sh
uv run python scripts/fit_isoflops.py
```

然后根据拟合结果设计小规模试探实验。不要一开始把 12 小时全押在一个配置上。更稳的方式是：

1. 选 2 到 3 个相近模型规模，训练较短时间，看 loss 曲线和速度。
2. 固定架构族，扫描学习率、batch size、total tokens。
3. 用已有结果估计更大预算下的 loss，而不是只看最后一个点。
4. 留出预算给最终方案或修正实验。

学生侧 API 使用方式：

```sh
export A3_API_KEY=06123456
```

```python
from cs336_scaling.client import get_budget, list_experiments, submit_experiment

print(get_budget())
response = submit_experiment(training_config)
print(response)
print(list_experiments())
```

最终提交：

```python
from cs336_scaling.client import save_final_submission

save_final_submission(
    training_config=final_training_config,
    predicted_final_loss=predicted_loss,
)
```

## 4. 代码框架设计

本次本地改动保持两个原则：

- 服务端 API、调度器、训练 loop 已经有清晰边界，不把 scaling-law 分析塞进这些核心路径。
- IsoFLOPs 分析是离线研究工具，放在 `scripts/fit_isoflops.py`，输入是 JSON，输出是可读表格和可选 JSON 文件。

这样做的好处是：训练服务仍然只负责提交、排队、调度、记录状态；实验设计逻辑可以独立迭代，不会影响 API 行为和测试。

## 5. 实现细节

### 5.1 TrainingConfig 校验

`cs336_scaling/training/training_config.py` 用 Pydantic 定义训练配置，并在 `validate_training_config` 里检查关键约束：

- `total_train_tokens` 必须能被 `seq_len * train_batch_size` 整除。
- optimizer steps 必须能被 `n_evals` 整除。
- validation token 数必须能被 validation batch 整除。
- `hidden_size == num_attention_heads * head_dim`。
- 当前不支持 GQA，所以 `num_key_value_heads == num_attention_heads`。
- RoPE 要求 `head_dim` 是偶数。
- AdamW 的 `weight_decay`、`beta1`、`beta2`、`eps` 等参数有合法范围。

这些校验很重要，因为 hosted API 不应该接收无法训练或 shape 不一致的配置。

### 5.2 稳定去重

`TrainingConfig.unique_id` 来自 `stable_json_hash`。它会把 Pydantic model 转成带类型信息的 JSON，再用 Blake2s 哈希。

这让服务端能拒绝同一用户重复提交完全相同的训练配置：

```text
same user + same training_config_unique_id -> 409 conflict
```

### 5.3 Public API

`cs336_scaling/api/public.py` 是学生侧最重要的服务端入口：

- `POST /submit`：提交训练配置，检查预算，拒绝重复配置，创建 queued experiment。
- `GET /budget`：查看已用、剩余和总预算。
- `GET /experiments`：列出当前用户的所有实验。
- `GET /experiment/{experiment_id}`：查看某个实验。
- `POST /final_submission`：保存或覆盖最终提交。
- `GET /final_submission`：读取当前最终提交。

这里有一个容易忽略的点：`final_submission` 使用 PostgreSQL upsert，所以重复提交会覆盖旧的最终方案，而不是创建多条记录。

### 5.4 Budget 计算

`cs336_scaling/budget.py` 按实验状态计算预算：

- `queued` / `running`：按 `max_runtime_seconds` 预留。
- `completed` / `failed`：按实际 `used_runtime_seconds`，但限制在 `[1, max_runtime_seconds]`。
- 其他状态：不计入预算。

这会鼓励你设置真实的 `max_runtime_seconds`。如果随手填很大，实验一提交就会占掉大量预算。

### 5.5 调度公平性

`cs336_scaling/scheduler/experiment_selector.py` 的核心思路是：优先照顾当前 running job 少的用户，然后再按排队时间排序。

它先统计每个用户当前 running 数量，再给 queued job 计算一个 `effective_running_count`。同一用户连续排多个任务时，后面的任务会被视作“更占用并发”，避免一个用户把队列前部全占满。

测试 `test_select_experiments_for_dispatch_orders_by_running_count_then_queue_time` 正是在验证这个公平性。

### 5.6 Internal API 和 worker

训练 worker 不直接写公开 API，而是通过 internal API 上报事件：

- validation loss 更新走 `/internal/worker/{experiment_id}/event`。
- 完成或失败走 `/internal/worker/{experiment_id}/finish`。

这些接口需要 `X-Internal-Key`。本地修复了一个配置问题：如果没有设置 `INTERNAL_API_KEY`，`settings_from_env()` 现在会默认用空字符串，而不是在导入阶段直接抛 `KeyError`。这让测试和本地探索能走到数据库初始化阶段。

## 6. 本地验证状态

已成功运行：

```sh
uv run python scripts/fit_isoflops.py --target-compute-budget 1e22
```

脚本能从 synthetic curves 中稳定提取 compute-optimal 点并拟合 scaling laws。

服务端测试命令是：

```sh
uv run --extra server pytest
```

当前机器缺少本地 PostgreSQL server/tooling，测试 fixture 无法连接：

```text
connection to server on socket "/tmp/.s.PGSQL.5432" failed
```

因此 API 和调度集成测试已经推进到数据库依赖处，但不能在当前环境完整跑通。要完整验证，需要先安装并启动 PostgreSQL，然后重新运行上面的 pytest 命令。

## 7. 最小行动清单

如果你要继续真正冲 leaderboard，按这个顺序做：

1. 运行 `scripts/fit_isoflops.py`，确认 compute-optimal 趋势。
2. 设置 `A3_API_KEY`，用 `get_budget()` 确认 12 小时预算。
3. 提交小规模探索实验，记录每个配置的 loss、runtime 和训练速度。
4. 根据实验结果更新 scaling-law 估计。
5. 只在最后提交一个经过预算核算的 final `TrainingConfig`。
6. 用 `get_final_submission()` 确认最终提交已保存。
