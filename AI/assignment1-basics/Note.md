# Assignment 1 Basics Note

## 第一章：从全局视角理解本次作业

这次作业的目标是从零搭出一个可以训练小型语言模型的最小系统。它不是只实现一个 Transformer 层，而是把语言模型训练链路中最核心的部件串起来：

1. 把原始文本变成 token ID：训练并使用 byte-level BPE tokenizer。
2. 把 token ID 变成 logits：实现 decoder-only Transformer LM。
3. 把 logits 变成训练信号：实现 softmax、cross-entropy、AdamW、学习率调度和梯度裁剪。
4. 把训练跑起来：实现 batch 采样、checkpoint、训练循环和文本生成。

这条链路可以写成：

```text
raw text
  -> UTF-8 bytes
  -> BPE token IDs
  -> TransformerLM logits
  -> cross-entropy loss
  -> AdamW update
  -> checkpoint / generation / experiments
```

### 1. 为什么先做 tokenizer

语言模型不能直接读 Python 字符串。模型输入必须是整数序列，每个整数表示词表中的一个 token。

本作业使用 byte-level BPE。它先把文本编码成 UTF-8 bytes，所以不会出现 unknown token：任何 Unicode 文本最终都可以表示成 0 到 255 的 byte 序列。然后 BPE 会把高频相邻 byte 序列合并成更长的 subword token，减少序列长度。

这里有三个容易混淆的概念：

- **Unicode code point**：字符的抽象编号，比如 `"牛"` 对应一个 Unicode 编号。
- **UTF-8 bytes**：字符在计算机里的实际字节表示，一个字符可能对应多个 byte。
- **BPE token**：一个或多个 byte 组成的词表项，最终会映射成整数 ID。

special token，例如 `<|endoftext|>`，在训练 tokenizer 时表示硬边界：它分隔文档，不能让 BPE merge 跨过它；在编码时它又必须整体保留成一个 token。

### 2. Transformer LM 在这里做什么

Transformer LM 的输入是形状为 `(batch_size, sequence_length)` 的 token ID。模型输出是形状为 `(batch_size, sequence_length, vocab_size)` 的 logits。每个位置的 logits 表示“看到当前位置及之前的 token 后，下一个 token 是词表中每个 token 的未归一化分数”。

本作业实现的是 pre-norm decoder-only Transformer：

```text
token IDs
  -> token embedding
  -> Transformer block x num_layers
  -> final RMSNorm
  -> LM head
  -> logits
```

每个 Transformer block 内部是：

```text
x
  -> x + causal self-attention(RMSNorm(x))
  -> z + feed-forward(RMSNorm(z))
```

这里的关键机制是：

- **causal mask**：第 `i` 个 token 只能看见 `0..i` 的 token，不能偷看未来。
- **multi-head self-attention**：把同一个 hidden state 分成多个 head，让不同 head 独立计算注意力。
- **RoPE**：给 query 和 key 加入位置信息，但不作用在 value 上。
- **SwiGLU FFN**：用门控结构增强逐位置非线性变换能力。
- **RMSNorm**：只按均方根缩放 hidden state，不减均值，比 LayerNorm 更简洁。

### 3. 训练基础设施为什么也属于作业核心

只写模型 forward 还不能训练语言模型。训练还需要：

- `cross_entropy`：把 logits 和正确下一个 token 变成一个标量 loss。
- `AdamW`：根据梯度更新参数，并维护一阶、二阶动量。
- `cosine_lr_schedule`：先 warmup，再按 cosine 退火学习率。
- `get_batch`：从一整条 token 序列中随机采样 `(x, y)`，其中 `y` 是 `x` 向右偏移一位。
- checkpoint：保存 model state、optimizer state 和 iteration，支持恢复训练。
- generation：从 prompt 出发，反复采样下一个 token，直到达到长度限制或生成 EOS。

PDF 后半部分还要求 TinyStories、OpenWebText、消融实验和 leaderboard。这些实验依赖真实数据、训练时间和硬件。本地代码提供了训练和生成的基础设施；实验报告本身需要在下载数据并实际跑训练后填写，不能凭空编造结果。

## 第二章：如何完成 Human 路径

这一章只说明 Human 路径要在哪些文件写代码、每一步完成什么行为、怎样测试。不要把具体实现写进笔记里；真正作业代码应写在 `Human/assignment1-basics` 下，`tests/adapters.py` 只做薄适配。

### 1. Task 1：接通测试适配层

需要写代码的文件：`Human/assignment1-basics/tests/adapters.py`，以及你在 `Human/assignment1-basics/cs336_basics/` 下创建或补齐的实现模块。

先读 `cs336_assignment1_basics.pdf` 和 `tests/adapters.py`。测试只通过 `run_*` adapter 调你的代码，所以第一步是决定实现模块边界，然后让 adapter 转发到这些模块。adapter 不应该承载 BPE、模型、优化器等业务逻辑。

建议的实现文件边界：

- `cs336_basics/bpe.py`：BPE 训练。
- `cs336_basics/tokenizer.py`：Tokenizer。
- `cs336_basics/model.py`：Transformer 组件。
- `cs336_basics/nn_utils.py`：softmax、cross entropy、gradient clipping。
- `cs336_basics/optim.py`：AdamW 和学习率 schedule。
- `cs336_basics/data.py`：batch sampling。
- `cs336_basics/serialization.py`：checkpoint 保存和加载。
- `cs336_basics/generation.py`、`cs336_basics/training.py`：生成和长跑训练脚手架。

怎样测试：

```sh
cd Human/assignment1-basics
uv run pytest tests/test_train_bpe.py::test_train_bpe -q
```

这条测试可能仍会因为 BPE 未实现而失败，但失败点应该进入你的实现，而不是停在 adapter 的 `NotImplementedError`。

### 2. Task 2：实现 BPE 训练

需要写代码的文件：`Human/assignment1-basics/cs336_basics/bpe.py` 和 `Human/assignment1-basics/tests/adapters.py`。

你要实现 byte-level BPE training，并让 `run_train_bpe` 调用它。注意 special token 切分、GPT-2 regex pre-tokenization、byte tuple 计数、pair 选择规则和 merge 更新。这里不要直接把逻辑写在 adapter 里，也不要为了某个 fixture 硬编码输出。

常见错误：

- 没有在训练前用 special token 切分文本，导致词表里出现包含 `<|` 的普通 token。
- tie-break 用了字典序更小的 pair。
- 直接每轮全量扫描所有字符，速度过慢。

怎样测试：

```sh
cd Human/assignment1-basics
uv run pytest tests/test_train_bpe.py -q
```

### 3. Task 3：实现 Tokenizer

需要写代码的文件：`Human/assignment1-basics/cs336_basics/tokenizer.py` 和 `Human/assignment1-basics/tests/adapters.py`。

你要实现 vocab/merges 加载、`encode`、`encode_iterable`、`decode` 和 special token 处理。special token 要在普通 regex pre-tokenization 前处理；重叠 special token 要有稳定优先级。

怎样测试：

```sh
cd Human/assignment1-basics
uv run pytest tests/test_tokenizer.py -q
```

### 4. Task 4：实现 Transformer 模型模块

需要写代码的文件：`Human/assignment1-basics/cs336_basics/model.py` 和 `Human/assignment1-basics/tests/adapters.py`。

建议按依赖顺序完成：`Linear`、`Embedding`、`RMSNorm`、`silu`、`SwiGLU`、scaled dot-product attention、RoPE、multi-head self-attention、`TransformerBlock`、`TransformerLM`。每个模块都要遵守测试 adapter 给定的权重形状和输出 shape。

常见错误：

- `Linear` 权重方向写反。
- RoPE 只支持二维输入，没有处理 batch-like 维度。
- causal mask 方向反了。
- attention head reshape 后忘了转回 hidden 维。
- final RMSNorm 漏掉。

怎样测试：

```sh
cd Human/assignment1-basics
uv run pytest -k test_linear
uv run pytest -k test_rope
uv run pytest -k test_transformer_lm
uv run pytest tests/test_model.py -q
```

### 5. Task 5：实现训练工具

需要写代码的文件：`Human/assignment1-basics/cs336_basics/nn_utils.py`、`Human/assignment1-basics/cs336_basics/optim.py`、`Human/assignment1-basics/cs336_basics/data.py`、`Human/assignment1-basics/cs336_basics/serialization.py` 和 `Human/assignment1-basics/tests/adapters.py`。

训练工具看起来简单，但容易出现数值或状态问题：

- `softmax` 和 `cross_entropy` 都要先减最大值，避免 overflow。
- `AdamW` 的 step 从 1 开始，moment state 要按参数保存。
- weight decay 是 decoupled weight decay，不要混进 gradient。
- gradient clipping 要按所有参数梯度的整体 L2 norm 缩放。
- `get_batch` 的 `y` 必须是 `x` 向右偏移一位。
- checkpoint 必须同时保存 model、optimizer 和 iteration。

怎样测试：

```sh
cd Human/assignment1-basics
uv run pytest tests/test_nn_utils.py
uv run pytest tests/test_optimizer.py
uv run pytest tests/test_data.py
uv run pytest tests/test_serialization.py
```

### 6. Task 6：实验、训练和生成

需要写代码或脚本的文件：`Human/assignment1-basics/cs336_basics/training.py`、`Human/assignment1-basics/cs336_basics/generation.py`，以及你自己用于 TinyStories/OpenWebText 的训练入口或实验记录文件。

当核心测试全部通过后，再进入 PDF 实验部分：

1. 下载 TinyStories / OpenWebText。
2. 训练对应 vocab size 的 BPE tokenizer。
3. 用 tokenizer 把 train/valid 文本编码成 `uint16` 或合适 dtype 的 NumPy array。
4. 用 `np.load(..., mmap_mode="r")` 或 `np.memmap` 读大数组。
5. 训练小模型，记录 train/valid loss、wall-clock time 和 learning curve。
6. 使用 generation 函数生成文本，再做学习率、batch size、RMSNorm、pre-norm、RoPE、SwiGLU 等实验。

这里的重点是不要把“代码能跑”误认为“实验已完成”。实验交付需要真实曲线、真实生成样例和真实观察。

怎样测试：

```sh
cd Human/assignment1-basics
uv run pytest
```

## 第三章：代码实现、逻辑与细节讲解

本地实现把实质逻辑放在 `cs336_basics/`，把测试桥接放在 `tests/adapters.py`。

### 1. 文件职责

- `cs336_basics/bpe.py`：训练 byte-level BPE，包含 GPT-2 regex、special token 切分、pair count 和 merge 更新。
- `cs336_basics/tokenizer.py`：实现 `Tokenizer`，支持 `encode`、`encode_iterable`、`decode` 和 GPT-2 vocab/merges 文件加载。
- `cs336_basics/model.py`：实现 Transformer LM 的全部神经网络组件。
- `cs336_basics/nn_utils.py`：实现 SiLU、softmax、cross-entropy、gradient clipping 和 cosine LR。
- `cs336_basics/optim.py`：实现 AdamW optimizer。
- `cs336_basics/data.py`：实现语言模型 batch 采样。
- `cs336_basics/serialization.py`：实现 checkpoint 保存和加载。
- `cs336_basics/generation.py`：实现 temperature 和 top-p generation。
- `cs336_basics/training.py`：提供最小训练循环和验证 loss 估计。
- `tests/adapters.py`：只做薄适配，让测试调用上述模块。

### 2. BPE 训练实现

`train_bpe` 先创建初始词表：special tokens 在前，之后是 256 个单 byte。这样训练一开始就能覆盖所有 UTF-8 byte。

pre-tokenization 使用 PDF 给出的 GPT-2 regex。训练前会先按 special token 切开文本，所以 `<|endoftext|>` 这类边界不会参与普通 pair 统计，也不会被 merge 进普通 token。

merge 阶段维护两类结构：

- `pair_counts`：每个相邻 token pair 的总频次。
- `pair_to_word_ids`：某个 pair 出现在哪些 pre-token 中。

每次选择最高频 pair 时，用 `(frequency, pair)` 作为比较键，因此频率相同会选择字典序更大的 pair。merge 后只更新受影响的 pre-token，而不是每轮从头扫描全部文本。这是 `test_train_bpe_speed` 能通过的关键。

### 3. Tokenizer 实现

`Tokenizer` 会把 `merges` 转成 `merge_ranks`，也就是“哪个 pair 更早被学习到”。编码一个 pre-token 时，它不是选择当前最高频 pair，而是反复选择 rank 最小的可用 merge。这对应 BPE 推理阶段：训练时已经决定了 merge 顺序，编码时只按顺序应用。

special token 处理采用“长 token 优先”。例如同时存在 `<|endoftext|>` 和 `<|endoftext|><|endoftext|>` 时，后者应该优先作为一个整体匹配。`test_overlapping_special_tokens` 专门检查这个行为。

`decode` 不假设输入 token IDs 一定能组成合法 UTF-8；它用 `errors="replace"`，所以非法 byte 会变成 Unicode replacement character。这符合 PDF 对 decoding 的要求。

### 4. Transformer 组件实现

`Linear` 的参数形状是 `(out_features, in_features)`，forward 使用：

```text
... d_in, d_out d_in -> ... d_out
```

这让任意 batch-like leading dimensions 都可以自然保留。

`Embedding` 直接用 token IDs 索引形状为 `(vocab_size, d_model)` 的参数矩阵。`RMSNorm` 会先把输入 upcast 到 `float32`，计算均方根后再 cast 回原 dtype，避免平方时溢出。

`SwiGLU` 按 PDF 的结构实现：

```text
W2(SiLU(W1(x)) * W3(x))
```

`RotaryPositionalEmbedding` 预先缓存 cos/sin buffer。forward 时根据 `token_positions` 取出对应位置，并在缺少 head 维度时自动 unsqueeze，让同一组位置可以 broadcast 到每个 attention head。

`scaled_dot_product_attention` 做三件事：

1. 计算 `QK^T / sqrt(d_k)`。
2. 对 mask 为 False 的位置填入很小的数，使 softmax 后概率接近 0。
3. 用 attention probabilities 加权 `V`。

`MultiHeadSelfAttention` 先分别做 Q/K/V projection，再把最后一维拆成 `(num_heads, d_head)`。RoPE 只作用在 Q/K 上，value 不旋转。最后用 causal mask 保证位置 `i` 不能看未来位置 `j > i`。

`TransformerBlock` 是标准 pre-norm residual：

```text
x = x + attn(ln1(x))
x = x + ffn(ln2(x))
```

`TransformerLM` 将 token embedding、多个 block、final RMSNorm 和 LM head 串起来。它还检查输入长度不能超过 `context_length`，因为 RoPE buffer 只为最大上下文长度预计算。

### 5. 训练工具实现

`cross_entropy` 不直接先算 softmax 再取 log，而是使用 shifted logits 计算：

```text
log(sum(exp(shifted_logits))) - shifted_target_logit
```

这样可以避免大 logits 导致 `exp` overflow。

`AdamW` 继承 `torch.optim.Optimizer`，每个参数单独维护：

- `step`
- `exp_avg`
- `exp_avg_sq`

更新顺序是先 decoupled weight decay，再更新 moment，最后用 bias-corrected learning rate 做参数更新。测试允许匹配本实现或 PyTorch AdamW 的数值。

`get_batch` 从一维 token array 中随机采样起点，返回：

```text
x = tokens[start : start + context_length]
y = tokens[start + 1 : start + context_length + 1]
```

这正是 next-token prediction 的训练样本。

`save_checkpoint` 和 `load_checkpoint` 保存/恢复三件事：model state、optimizer state 和 iteration。没有保存随机数状态；如果要完全复现实验，可以在训练脚本里额外记录 seed 和 RNG state。

### 6. 测试如何覆盖这些行为

- `tests/test_train_bpe.py` 覆盖 BPE merge 正确性、速度和 special token 边界。
- `tests/test_tokenizer.py` 覆盖 roundtrip、GPT-2/tiktoken 对齐、special token、streaming encode。
- `tests/test_model.py` 用 snapshot 覆盖 Linear、Embedding、SwiGLU、RoPE、attention、TransformerBlock 和 TransformerLM。
- `tests/test_nn_utils.py` 覆盖数值稳定 softmax/cross-entropy 和 gradient clipping。
- `tests/test_optimizer.py` 覆盖 AdamW 和 cosine schedule。
- `tests/test_data.py` 覆盖 batch shape、target offset、随机起点范围和 device 参数。
- `tests/test_serialization.py` 覆盖 checkpoint 能恢复 model 和 optimizer。

当前本地验证结果：

```sh
uv run pytest
# 46 passed, 2 skipped

uv run ruff check .
# All checks passed!
```

两个 skipped 测试是 Linux-only 的 `rlimit` 内存测试；当前环境是 macOS，所以 pytest 按测试条件跳过它们。

### 7. 调试时优先看什么

如果 tokenizer 对不上，先看 special token 是否被普通文本 merge 了，再看 GPT-2 regex 是否完全一致，最后看 merge rank 应用顺序。

如果模型 snapshot 对不上，优先检查 shape 和权重名。测试提供的 state dict key 是本实现命名的重要约束，例如 `attn.q_proj.weight`、`ffn.w1.weight`、`ln_final.weight`。只要名字不对，`load_state_dict` 就无法可靠对齐参考权重。

如果训练 loss 不下降，先尝试在一个 minibatch 上过拟合。过拟合不过去通常说明 forward、loss、optimizer 或 causal mask 有问题；能过拟合但验证差，才进入学习率、batch size、模型规模和 tokenizer 质量的实验阶段。
