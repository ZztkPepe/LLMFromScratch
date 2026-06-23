# Assignment 2 Systems Note

## 第一章：从全局视角理解本次作业

Assignment 2 的主题是系统优化和并行训练。Assignment 1 关注“语言模型的数学组件怎么从零实现”，而 Assignment 2 关注“同一个 Transformer LM 如何更快、更省显存、并扩展到多设备训练”。

这次作业可以分成四条主线：

1. **Benchmarking / Profiling**：先测量，再优化。用 warmup、CUDA synchronize、Nsight Systems、memory snapshot 等工具找到时间和显存瓶颈。
2. **FlashAttention**：优化 attention 的内存访问方式，避免显式 materialize 完整 attention matrix。
3. **Distributed Data Parallel (DDP)**：多个 rank 各自处理一部分 batch，再平均梯度，让训练结果等价于单卡大 batch。
4. **Optimizer state sharding / FSDP**：进一步减少每个 rank 重复存储 optimizer state、gradient 和 weight 的内存。

从训练步骤看，系统优化围绕这条路径展开：

```text
forward
  -> attention / MLP compute
  -> loss
  -> backward
  -> gradient communication
  -> optimizer step
  -> parameter synchronization
```

Assignment 2 不是让你改变语言模型的语义，而是让你在保持训练结果等价的前提下改变执行方式。正确性永远先于速度：如果优化后模型参数和单进程 baseline 不一致，那么速度没有意义。

### 1. Profiling 为什么要先于优化

GPU 程序有一个重要特点：CUDA kernel 通常是异步提交的。Python 代码调用一个 CUDA op 后，CPU 可能马上继续执行，而 GPU 还在后台跑 kernel。因此，直接用 `time.time()` 包住一行 PyTorch 调用，往往测不到真实 GPU 执行时间。

正确的 benchmarking 需要：

- warmup：让 kernel cache、allocator、编译路径进入稳定状态。
- `torch.cuda.synchronize()`：等待 GPU 任务完成后再停止计时。
- 分开测 forward、forward+backward、full step。
- 用 profiler 看细分瓶颈，而不是只看总时间。

本地新增的 `benchmark.py` 对应 PDF 的基础 benchmarking script，可以初始化 `cs336_basics.model.BasicsTransformerLM`，生成随机数据，并计时 forward、backward、full training step。

### 2. FlashAttention 的核心直觉

普通 attention 会显式构造：

```text
S = QK^T / sqrt(d)
P = softmax(S)
O = PV
```

其中 `S` 和 `P` 的形状是 `(batch, n_queries, n_keys)`。当 sequence length 很长时，这个矩阵非常大，读写 HBM 的成本会压过实际计算。

FlashAttention 的思想是分块计算 attention。它不会一次性把完整 `S` 和 `P` 写到显存，而是在 tile 内维护每一行 softmax 所需的最大值、归一化因子和局部输出。这样可以显著减少显存读写。

在本地测试里，重点验证的是 autograd contract：

- forward 返回 `O`。
- forward 保存 log-sum-exp `L`，以及 backward 需要的 `Q/K/V/O`。
- backward 根据 `dO` 重新计算 attention probabilities，并返回 `dQ/dK/dV`。

当前 CPU 环境没有 CUDA，所以 Triton 测试会自动跳过。代码中提供了和 Triton adapter 同接口的 correctness-compatible path；真正的 fused Triton kernel 性能需要在 CUDA 环境里继续实现和 benchmark。

### 3. DDP 的核心不变量

DDP 的目标是让多个 rank 看到不同 mini-batch 分片，但参数更新结果等价于单进程看完整 batch。

流程是：

```text
rank 0 参数广播到所有 rank
每个 rank 处理自己的 local batch
backward 得到 local gradient
all-reduce gradient sum
除以 world_size 得到 average gradient
每个 rank 用同一个 averaged gradient 做 optimizer step
```

只要每个 rank 初始参数一致，且每次 optimizer step 使用的平均梯度一致，那么所有 rank 的参数会持续保持一致。

本地实现使用 `register_post_accumulate_grad_hook`：某个参数的梯度一产生，就立刻异步 all-reduce。这样可以把通信和剩余 backward 计算重叠起来。训练循环在 `optimizer.step()` 前调用 `finish_gradient_synchronization()` 等待通信完成。

### 4. Optimizer state sharding 和 FSDP 的区别

DDP 仍然让每个 rank 存完整参数和完整 optimizer state。AdamW 通常至少为每个参数保存两个额外状态：一阶动量和二阶动量，所以 optimizer state 很占显存。

Optimizer state sharding 只切分 optimizer 负责的参数子集：

```text
rank 0 更新一部分参数
rank 1 更新另一部分参数
更新后把各自负责的参数 broadcast 给其他 rank
```

这样每个 rank 只需要维护自己那一片参数的 optimizer state。

FSDP 更进一步：不仅 optimizer state 分片，参数和梯度也分片。真实 FSDP 在 forward/backward 前需要 all-gather 当前层权重，用完后释放；backward 后 reduce-scatter 梯度。PDF 还要求 mixed precision：通信和计算可以用低精度，但 master weights 和 optimizer update 保持 FP32。

本地测试重点验证 FSDP 的训练等价性、mixed precision 行为、梯度同步和 full parameter gather。真正的显存节省和预取是否隐藏通信，需要在多 GPU 上用 Nsight 和 memory profiler 继续验证。

## 第二章：如何完成 Human 路径

这一章只说明 Human 路径要在哪些文件写代码、每一步完成什么行为、怎样测试。建议按“先 correctness，再 profiling，再性能优化”的顺序完成。不要一开始就写 Triton kernel 或 FSDP 预取逻辑，因为分布式和 GPU kernel 的 bug 通常很难定位。

### 1. Task 1：接通环境和 adapters

需要写代码的文件：`Human/assignment2-systems/tests/adapters.py`，以及 `Human/assignment2-systems/cs336_systems/` 下的实现模块。

Assignment2 使用 `cs336-basics/` 里的 Assignment1 实现作为模型依赖。先确认：

```sh
uv run python
>>> import cs336_basics
>>> import cs336_basics.model
```

然后看 `README.md`、`tests/adapters.py` 和四个测试文件：

- `tests/test_attention.py`
- `tests/test_ddp.py`
- `tests/test_sharded_optimizer.py`
- `tests/test_fsdp.py`

这一步要弄清楚：测试不是直接找你的类名，而是通过 adapters 获取你的实现。所以实质代码应放在 `cs336_systems/`，adapters 只做薄连接。

怎样测试：

```sh
cd Human/assignment2-systems
uv run python -c "import cs336_basics, cs336_systems; print('ok')"
```

### 2. Task 2：完成 FlashAttention correctness path

需要写代码的文件：`Human/assignment2-systems/cs336_systems/attention.py` 和 `Human/assignment2-systems/tests/adapters.py`。

先写纯 PyTorch 的 `torch.autograd.Function`，不要急着写 Triton。forward 应该计算：

```text
scores = QK^T / sqrt(d)
L = logsumexp(scores)
P = exp(scores - L)
O = PV
```

backward 可以按 FlashAttention 的 recomputation 思路重新算 `P`，再得到：

```text
D = rowsum(O * dO)
dV = P^T dO
dP = dO V^T
dS = P * (dP - D)
dQ = dS K / sqrt(d)
dK = dS^T Q / sqrt(d)
```

检查信号：

```sh
uv run pytest tests/test_attention.py
```

常见错误：

- 忘记保存形状为 `(batch, n_queries)` 的 `L`。
- causal mask 使用 `-inf` 但测试参考使用 `-1e6`，导致数值略有差异。
- backward 里忘记乘 `1 / sqrt(d)`。
- `D` 的维度没有 unsqueeze，broadcast 方向错。

怎样测试：

```sh
cd Human/assignment2-systems
uv run pytest tests/test_attention.py -q
```

### 3. Task 3：完成 DDP

需要写代码的文件：`Human/assignment2-systems/cs336_systems/parallel.py` 和 `Human/assignment2-systems/tests/adapters.py`。

DDP 的最小正确版本必须做两件事：

1. 初始化时从 rank 0 broadcast 参数和 buffer。
2. backward 后平均所有 trainable 参数的梯度。

更好的版本是在每个参数梯度 ready 时注册 hook，异步通信：

```text
param.grad /= world_size
handle = all_reduce(param.grad, async_op=True)
保存 handle
```

然后在 optimizer step 前等待所有 handle。

检查信号：

```sh
uv run pytest tests/test_ddp.py
```

测试会比较：

- rank 0 初始化参数是否保持不变。
- rank 1 是否被 rank 0 参数覆盖。
- DDP 训练 5 步后是否和单进程 full batch baseline 一致。
- tied weights 是否还能正常同步。

怎样测试：

```sh
cd Human/assignment2-systems
uv run pytest tests/test_ddp.py -q
```

### 4. Task 4：完成 sharded optimizer

需要写代码的文件：`Human/assignment2-systems/cs336_systems/parallel.py` 和 `Human/assignment2-systems/tests/adapters.py`。

Sharded optimizer 的关键是“谁负责更新哪个参数”。一个简单稳定的策略是按参数顺序分配：

```text
owner_rank = parameter_index % world_size
```

每个 rank 的本地 optimizer 只持有自己负责的参数。`step()` 后，每个参数由 owner rank broadcast 给所有 rank。这样所有 rank 最终仍然拥有完整同步后的模型参数，但 optimizer state 只为本 rank 负责的参数存在。

检查信号：

```sh
uv run pytest tests/test_sharded_optimizer.py
```

常见错误：

- 本地 optimizer 更新后没有 broadcast 参数。
- `zero_grad()` 只清了本地 shard，没有清全模型参数。
- tied weights 被重复加入 optimizer，导致更新次数不一致。
- 每个 rank 参数分配顺序不一致。

怎样测试：

```sh
cd Human/assignment2-systems
uv run pytest tests/test_sharded_optimizer.py -q
```

### 5. Task 5：完成 FSDP correctness path

需要写代码的文件：`Human/assignment2-systems/cs336_systems/parallel.py` 和 `Human/assignment2-systems/tests/adapters.py`。

FSDP 的完整生产实现很复杂：真实参数分片、前向 all-gather、反向 all-gather、梯度 reduce-scatter、预取、释放 full weights、mixed precision。Human 路径建议先实现测试要求的 correctness path：

1. 包装整个 module。
2. 对 `Linear` 和 `Embedding` 安装 mixed precision hook。
3. backward 后同步梯度，让 local batch 训练等价于 full batch baseline。
4. 提供 `gather_full_params()`，用于把当前参数状态取出来比较。

检查信号：

```sh
uv run pytest tests/test_fsdp.py
```

测试会覆盖：

- fp32 correctness。
- fp16 compute dtype correctness。
- backward 后每个参数都有正确 shape/dtype 的 gradient。
- 非 Linear/Embedding 的 replicated 参数梯度在 ranks 间一致。

怎样测试：

```sh
cd Human/assignment2-systems
uv run pytest tests/test_fsdp.py -q
```

### 6. Task 6：性能实验和 written deliverables

需要写代码或实验记录的文件：`Human/assignment2-systems/benchmark.py`，以及你的 profiling 记录和 written deliverables。

核心测试通过后，再进入 PDF 里需要 GPU 的实验：

- 用 `benchmark.py` 采集 forward/backward/full step timings。
- 用 Nsight Systems 看 kernel timeline 和 communication overlap。
- 用 memory profiler 看 activation/optimizer state/FSDP 的峰值显存。
- 在 CUDA 环境实现和调优真正 Triton FlashAttention forward kernel。
- 记录 DDP、flat DDP、overlapped DDP、optimizer state sharding、FSDP 的性能对比。

这些结果必须来自真实运行，不能从本地 CPU/gloo 测试推断。

怎样测试：

```sh
cd Human/assignment2-systems
uv run pytest
uv run python benchmark.py --model-size small --mode forward
```

## 第三章：代码实现、逻辑与细节讲解

本地实现的核心文件是：

- `cs336_systems/attention.py`
- `cs336_systems/parallel.py`
- `tests/adapters.py`
- `benchmark.py`

### 1. `attention.py`

`FlashAttentionPytorchFunction` 是一个 `torch.autograd.Function`。

forward 做四件事：

1. 计算 attention scores。
2. 可选 causal mask。
3. 保存每一行的 log-sum-exp `L`。
4. 返回 `O = softmax(scores) V`。

它保存的 tensors 是：

```text
L, Q, K, V, O
```

测试会直接检查 `saved_tensors` 里是否有且只有一个形状为 `(batch, n_queries)` 的 tensor，这就是 `L`。

backward 不依赖 forward 保存完整 attention matrix，而是重新计算 `scores` 和 `P`。这体现了 FlashAttention backward 的关键思想：多算一点，少存很多。

`FlashAttentionTritonFunction` 当前保持同一个 autograd contract，作为 Triton adapter 的 correctness-compatible 入口。当前机器没有 CUDA，所以 Triton 测试被 pytest skip；真正 fused Triton kernel 需要在 CUDA 环境里实现和验证。

### 2. `parallel.py` 中的 DDP

`DistributedDataParallel` 继承 `nn.Module`，内部保留原始模型为 `self.module`。这样测试可以通过 `ddp_model.module` 访问被包装模型。

初始化时：

- 遍历 `state_dict()`，从 rank 0 broadcast 每个 tensor。
- 给每个 `requires_grad=True` 的参数注册 post-accumulate hook。

hook 做的是：

```text
grad /= world_size
async all_reduce(grad)
保存 handle
```

`finish_gradient_synchronization()` 等待所有 handle，然后清空 handle 列表。训练循环必须在 `optimizer.step()` 前调用它，否则 optimizer 可能读到尚未同步完成的梯度。

### 3. `parallel.py` 中的 ShardedOptimizer

`ShardedOptimizer` 继承 `torch.optim.Optimizer`，外层保留完整参数组，所以 `zero_grad()` 可以清理所有参数。内部再构造一个只包含本 rank 参数 shard 的本地 optimizer。

参数分片方式是稳定的顺序分配：

```text
index 0 -> rank 0
index 1 -> rank 1
index 2 -> rank 0
...
```

本地 optimizer 只为这些参数维护 AdamW state。`step()` 后，所有参数按 owner rank broadcast。这样每个 rank 的完整模型参数重新同步。

这个设计对应 ZeRO stage 1 的入门版思想：optimizer state 分片，但参数本身在 step 后仍复制到每个 rank。

### 4. `parallel.py` 中的 FSDP wrapper

`FullyShardedDataParallel` 当前实现了测试所需的 FSDP correctness interface：

- `forward()` 代理到 wrapped module。
- `finish_gradient_synchronization()` 等待异步梯度同步。
- `gather_full_params()` 返回当前完整参数字典。
- `compute_dtype` 不为空时，对 `Linear` 和 `Embedding` 安装 mixed precision hooks。

mixed precision hooks 的目标是模拟 PDF 要求的行为：

```text
master weight: fp32
forward/backward compute weight: compute_dtype
gradient before optimizer step: fp32
```

对于 `Linear`，backward 前还需要把 weight 临时 cast 到 compute dtype，因为 grad input 的计算会用到 weight。梯度累积完成后，再恢复 fp32 master weight，并把 grad cast 回 fp32。

在 CPU/gloo 测试环境中，这个 wrapper 使用 all-reduce 平均梯度来保证和 non-parallel full batch baseline 等价。真实显存节省版 FSDP 还需要物理参数分片、all-gather、reduce-scatter、预取和释放 full weights；这些属于 CUDA 多 GPU 性能路径，需要结合 profiler 继续扩展。

### 5. `benchmark.py`

`benchmark.py` 提供基础命令行 benchmarking：

```sh
uv run python benchmark.py --model-size small --mode forward
uv run python benchmark.py --model-size small --mode backward
uv run python benchmark.py --model-size small --mode full
```

它支持：

- PDF 表格里的 `small/medium/large/xl/10b` 配置。
- `forward`、`backward`、`full` 三种模式。
- warmup steps 和 measurement steps。
- CUDA 环境下自动 synchronize。

这个脚本只产出 timing 原始数据。PDF 里要求的表格、Nsight 截图、memory timeline 和多 GPU benchmark，需要在目标硬件上运行后整理进 writeup。

### 6. 测试覆盖关系

本地验证结果：

```sh
uv run pytest
# 10 passed, 4 skipped

uv run ruff check cs336_systems tests/adapters.py benchmark.py
# All checks passed!
```

4 个 skipped 是 CUDA-only Triton 测试，当前 macOS/CPU 环境没有 CUDA，所以 pytest 按条件跳过。

`uv run ruff check .` 会报告提供的测试文件中已有的 lint 问题，例如 `tests/conftest.py` 和 `tests/test_attention.py`。我没有修改这些测试文件；对本次新增实现和 adapter 范围单独 ruff 是通过的。

### 7. 调试优先级

如果 FlashAttention backward 不匹配，先检查 `D = rowsum(O * dO)` 和 `sqrt(d)` scaling。多数错误来自维度 broadcast 或漏乘 scale。

如果 DDP 不匹配 baseline，先检查初始化 broadcast，再检查 loss reduction 是否和 full batch mean 对齐，最后检查 `finish_gradient_synchronization()` 是否在 optimizer step 前调用。

如果 sharded optimizer 不匹配，先打印每个参数的 owner rank 和 broadcast source。参数顺序必须在所有 rank 上一致。

如果 FSDP mixed precision 不匹配，先检查 master weight 是否在 optimizer step 前恢复为 fp32，以及 grad dtype 是否和 param dtype 一致。
