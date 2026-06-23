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

## 第二章：如何完成 Human 路径

这一章只说明 Human 路径要在哪些文件写代码或实验记录、每一步完成什么行为、怎样测试。Assignment3 有两类工作：本地代码/服务端 correctness，以及 hosted API 上的真实 scaling-law 实验。不要用 synthetic 曲线结果冒充最终 leaderboard 结论。

### 1. Task 1：读清楚作业边界和测试入口

需要查看的文件：`Human/assignment3-scaling/cs336_assignment3_scaling.pdf`、`Human/assignment3-scaling/README.md`、`Human/assignment3-scaling/tests/test_api.py`、`Human/assignment3-scaling/tests/test_scheduler.py`。

先弄清楚四件事：

- 作业目标是在有限预算内找到更低 validation loss，而不是训练最大模型。
- `TrainingConfig` 的合法性会影响 API 是否接受实验。
- queued/running/completed/failed 实验对 budget 的计算规则不同。
- 最终提交由 `POST /final_submission` 保存，并包含完整配置和预测 loss。

怎样测试：

```sh
cd Human/assignment3-scaling
uv run python -c "import cs336_scaling; print('ok')"
```

### 2. Task 2：理解并实现 TrainingConfig 校验

需要写代码的文件：`Human/assignment3-scaling/cs336_scaling/training/training_config.py`、`Human/assignment3-scaling/cs336_scaling/stable_hash.py`。

你要确保训练配置在进入 API 前能被可靠校验，例如 token 数整除关系、attention head 维度关系、RoPE 维度要求、optimizer 参数范围、稳定 hash 和重复提交识别。这里不要为了单个测试绕过校验；hosted API 也依赖这些约束保护真实训练。

怎样测试：

```sh
cd Human/assignment3-scaling
uv run --extra server pytest tests/test_api.py::test_submit_jobs -q
```

如果本机没有 PostgreSQL，这类 API 测试会停在数据库连接；先安装并启动 PostgreSQL，再重跑。

### 3. Task 3：实现 budget 和 public API 行为

需要写代码的文件：`Human/assignment3-scaling/cs336_scaling/api/public.py`、`Human/assignment3-scaling/cs336_scaling/budget.py`、`Human/assignment3-scaling/cs336_scaling/client.py`、`Human/assignment3-scaling/cs336_scaling/schemas/`。

你要让 submit、budget、experiments、experiment detail、final submission 这些公开接口行为一致。重点是预算预留、重复配置拒绝、final submission 覆盖语义，以及 client 侧返回结构。

怎样测试：

```sh
cd Human/assignment3-scaling
uv run --extra server pytest tests/test_api.py -q
```

这同样需要可用 PostgreSQL 测试数据库。

### 4. Task 4：实现调度公平性

需要写代码的文件：`Human/assignment3-scaling/cs336_scaling/scheduler/experiment_selector.py`。

调度器应该优先照顾当前 running job 少的用户，再按排队时间排序。同一用户连续排多个任务时，要避免该用户把队列前部全部占满。

怎样测试：

```sh
cd Human/assignment3-scaling
uv run --extra server pytest tests/test_scheduler.py -q
```

### 5. Task 5：实现和使用 IsoFLOPs 分析脚本

需要写代码的文件：`Human/assignment3-scaling/scripts/fit_isoflops.py`，输入数据来自 `Human/assignment3-scaling/data/isoflops_curves.json`。

IsoFLOPs 的意思是：固定总训练 compute budget `C`，改变模型参数量 `N` 和训练 token 数 `D`，观察最终验证 loss。

对 dense Transformer，常用近似是：

```text
C ≈ 6 * N * D
```

所以在固定 `C` 时，模型越大，能训练的 token 数通常越少；模型越小，能训练的 token 数越多。loss 曲线通常会出现一个最低点，这个点就是该 compute budget 下的 compute-optimal 配置。

脚本应该做三件事：

1. 按 `compute_budget` 分组。
2. 在每组里选择 `final_loss` 最低的点。
3. 拟合三条简单规律：

```text
N_opt(C) = a_N * C^b_N
D_opt(C) = a_D * C^b_D
L_opt(C) = E + A * C^(-alpha)
```

怎样测试：

```sh
cd Human/assignment3-scaling
uv run python scripts/fit_isoflops.py --target-compute-budget 1e22
```

### 6. Task 6：设计 hosted API 实验和最终提交

需要使用或更新的文件：`Human/assignment3-scaling/cs336_scaling/client.py`、你的实验记录、最终 `TrainingConfig` 记录。

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

怎样测试：

```sh
cd Human/assignment3-scaling
export A3_API_KEY=<your_api_key>
uv run python - <<'PY'
from cs336_scaling.client import get_budget, list_experiments
print(get_budget())
print(list_experiments())
PY
```

这一步依赖 hosted API 和你的真实 API key；不能用本地 synthetic 结果替代真实实验记录。

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
