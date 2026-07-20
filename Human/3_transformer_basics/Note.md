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

这一章只说明 Human 路径要在哪些文件写代码、每一步完成什么行为、怎样测试。不要把具体实现写进笔记里；真正作业代码应写在 `Human/3_transformer_basics` 下，`tests/adapters.py` 只做薄适配。

### 1. Task 1：接通测试适配层

需要写代码的文件：`Human/3_transformer_basics/tests/adapters.py`，以及你在 `Human/3_transformer_basics/cs336_basics/` 下创建或补齐的实现模块。

先读 `cs336_assignment1_basics.pdf` 和 `tests/adapters.py`。测试只通过 `run_*` adapter 调你的代码，所以第一步是决定实现模块边界，然后让 adapter 转发到这些模块。adapter 不应该承载 BPE、模型、优化器等业务逻辑。

#### 知识点：adapter 是稳定接口层

测试 adapter 的作用是把课程规定的函数签名与学生自己的模块结构连接起来。它相当于一个很薄的协议转换层：测试只依赖稳定入口，而实现可以按职责拆分。若把算法写进 adapter，代码虽然可能通过局部测试，却会失去可复用、可调试的模块边界。

#### 大概实现逻辑

先逐个阅读 `run_*` 的参数与返回契约，为每类能力选择唯一的实现模块，再让 adapter 完成参数转发和必要的对象构造。初期可以让调用进入尚未完成的实现并产生明确失败，以证明接线正确；之后每完成一个模块，只替换实现内部，不继续膨胀 adapter。

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
cd Human/3_transformer_basics
uv run pytest tests/test_train_bpe.py::test_train_bpe -q
```

这条测试可能仍会因为 BPE 未实现而失败，但失败点应该进入你的实现，而不是停在 adapter 的 `NotImplementedError`。

### 2. Task 2：实现 BPE 训练

需要写代码的文件：`Human/3_transformer_basics/cs336_basics/bpe.py` 和 `Human/3_transformer_basics/tests/adapters.py`。

你要实现 byte-level BPE training，并让 `run_train_bpe` 调用它。注意 special token 切分、GPT-2 regex pre-tokenization、byte tuple 计数、pair 选择规则和 merge 更新。这里不要直接把逻辑写在 adapter 里，也不要为了某个 fixture 硬编码输出。

#### 知识点：BPE 的贪心词表学习

Byte-level BPE 从单字节符号开始，反复把语料中最值得合并的相邻符号对变成一个新 token。每轮合并都会改变后续 pair 的统计，因此词表和 merge 顺序共同定义 tokenizer 行为。special token 必须先被隔离，因为它们应作为不可拆分的控制符号，而不是参与普通 pair 竞争。

#### 大概实现逻辑

先把语料按 special token 边界和预分词规则切成带频次的 byte 序列，再建立相邻 pair 到出现位置或词条的统计。每轮按频次和规定的 tie-break 选择 pair，创建新 token，只更新受该合并影响的局部统计，并记录 merge 顺序。达到目标词表大小后返回 vocab 与 merges，并用小语料检查 special token、并列 pair 和重复词频。

常见错误：

- 没有在训练前用 special token 切分文本，导致词表里出现包含 `<|` 的普通 token。
- tie-break 用了字典序更小的 pair。
- 直接每轮全量扫描所有字符，速度过慢。

怎样测试：

```sh
cd Human/3_transformer_basics
uv run pytest tests/test_train_bpe.py -q
```

### 3. Task 3：实现 Tokenizer

需要写代码的文件：`Human/3_transformer_basics/cs336_basics/tokenizer.py` 和 `Human/3_transformer_basics/tests/adapters.py`。

你要实现 vocab/merges 加载、`encode`、`encode_iterable`、`decode` 和 special token 处理。special token 要在普通 regex pre-tokenization 前处理；重叠 special token 要有稳定优先级。

#### 知识点：训练规则与编码规则必须一致

Tokenizer 编码是在新文本上重放 BPE 学到的 merge 优先级。vocab 决定 token id 与 bytes 的映射，merges 决定相邻符号按什么顺序组合。decode 则把 token bytes 连接后统一解码；只要编码阶段随意改变 special token 或 merge 顺序，就无法保证 round trip。

#### 大概实现逻辑

加载阶段建立双向 vocab 和 pair 的 merge rank。编码时先稳定识别 special token，再对普通片段执行与训练一致的预分词，把每段转成 bytes，并反复应用当前 rank 最靠前的可用 merge。流式接口逐块复用同一编码逻辑；解码按 id 找回 bytes、拼接后使用明确的错误处理策略还原文本。

怎样测试：

```sh
cd Human/3_transformer_basics
uv run pytest tests/test_tokenizer.py -q
```

### 4. Task 4：实现 Transformer 模型模块

需要写代码的文件：`Human/3_transformer_basics/cs336_basics/model.py` 和 `Human/3_transformer_basics/tests/adapters.py`。

建议按依赖顺序完成：`Linear`、`Embedding`、`RMSNorm`、`silu`、`SwiGLU`、scaled dot-product attention、RoPE、multi-head self-attention、`TransformerBlock`、`TransformerLM`。每个模块都要遵守测试 adapter 给定的权重形状和输出 shape。

#### 知识点：Transformer 是 shape 契约的组合

Transformer 的复杂度主要来自多个简单模块对张量维度的约定。Embedding 把 token id 变成 hidden vector；attention 在 sequence 维混合信息；SwiGLU 在特征维变换；residual 要求输入输出 shape 完全一致；RoPE 只旋转成对的特征维。只要其中一个 reshape、转置或广播方向错，后续模块都会出现看似无关的失败。

#### 大概实现逻辑

按最小依赖顺序逐个实现并独立测试，每个函数入口先写清输入与输出 shape。attention 中分开追踪 batch、head、sequence 和 head-dim，完成打分、mask、归一化和值聚合后再合并 heads；Block 只负责 norm、子层和 residual 的组合；LM 最后连接 embedding、若干 Block、final norm 与词表投影。每通过一层再进入下一层，避免整模调试。

常见错误：

- `Linear` 权重方向写反。
- RoPE 只支持二维输入，没有处理 batch-like 维度。
- causal mask 方向反了。
- attention head reshape 后忘了转回 hidden 维。
- final RMSNorm 漏掉。

怎样测试：

```sh
cd Human/3_transformer_basics
uv run pytest -k test_linear
uv run pytest -k test_rope
uv run pytest -k test_transformer_lm
uv run pytest tests/test_model.py -q
```

### 5. Task 5：实现训练工具

需要写代码的文件：`Human/3_transformer_basics/cs336_basics/nn_utils.py`、`Human/3_transformer_basics/cs336_basics/optim.py`、`Human/3_transformer_basics/cs336_basics/data.py`、`Human/3_transformer_basics/cs336_basics/serialization.py` 和 `Human/3_transformer_basics/tests/adapters.py`。

训练工具看起来简单，但容易出现数值或状态问题：

#### 知识点：数值稳定性与可恢复训练状态

训练工具一部分控制数值尺度，例如稳定 softmax、交叉熵和全局梯度裁剪；另一部分维护跨 step 状态，例如 AdamW 的一阶/二阶矩、学习率进度和 checkpoint iteration。正确的单步公式如果缺失状态或恢复顺序，长训练仍会悄悄偏离。

#### 大概实现逻辑

先实现无状态纯函数并用极端 logits 检查有限输出，再实现按参数保存状态的 optimizer 和 schedule。batch sampler 必须保证输入、目标错开一个 token且不越界。checkpoint 以一个一致快照保存模型、优化器和迭代位置，加载时恢复到调用者提供的对象。最后做“连续训练若干步”与“中途保存再恢复”的结果对比。

- `softmax` 和 `cross_entropy` 都要先减最大值，避免 overflow。
- `AdamW` 的 step 从 1 开始，moment state 要按参数保存。
- weight decay 是 decoupled weight decay，不要混进 gradient。
- gradient clipping 要按所有参数梯度的整体 L2 norm 缩放。
- `get_batch` 的 `y` 必须是 `x` 向右偏移一位。
- checkpoint 必须同时保存 model、optimizer 和 iteration。

怎样测试：

```sh
cd Human/3_transformer_basics
uv run pytest tests/test_nn_utils.py
uv run pytest tests/test_optimizer.py
uv run pytest tests/test_data.py
uv run pytest tests/test_serialization.py
```

### 6. Task 6：实验、训练和生成

需要写代码或脚本的文件：`Human/3_transformer_basics/cs336_basics/training.py`、`Human/3_transformer_basics/cs336_basics/generation.py`，以及你自己用于 TinyStories/OpenWebText 的训练入口或实验记录文件。

当核心测试全部通过后，再进入 PDF 实验部分：

#### 知识点：训练实验是受控比较

实验的目标不是只得到一个能生成文本的模型，而是判断某个配置变化如何影响速度、稳定性和验证 loss。一次只改变少量因素、保留基线和完整元数据，才能把结果归因到学习率、batch size、归一化或位置编码，而不是随机种子或数据处理差异。

#### 大概实现逻辑

先用极小语料打通 tokenizer、二进制数据、训练、checkpoint 和生成的端到端链路，再固定随机种子和基线配置扩大规模。每次运行记录数据版本、模型配置、优化器、token 数、wall-clock 与 train/valid loss；生成时从明确 checkpoint 和采样参数出发。实验结束后比较曲线和样例，并把观察与原始记录对应起来。

1. 下载 TinyStories / OpenWebText。
2. 训练对应 vocab size 的 BPE tokenizer。
3. 用 tokenizer 把 train/valid 文本编码成 `uint16` 或合适 dtype 的 NumPy array。
4. 用 `np.load(..., mmap_mode="r")` 或 `np.memmap` 读大数组。
5. 训练小模型，记录 train/valid loss、wall-clock time 和 learning curve。
6. 使用 generation 函数生成文本，再做学习率、batch size、RMSNorm、pre-norm、RoPE、SwiGLU 等实验。

这里的重点是不要把“代码能跑”误认为“实验已完成”。实验交付需要真实曲线、真实生成样例和真实观察。

怎样测试：

```sh
cd Human/3_transformer_basics
uv run pytest
```

### 7. 全面验收：确认整个 Assignment 1 已完成

先从锁定依赖的干净环境开始，确认 adapters 已全部接到你自己的实现，再运行完整测试和静态检查：

```sh
cd Human/3_transformer_basics
uv sync
rg -n 'raise NotImplementedError' tests/adapters.py
uv run pytest -q
uv run ruff check cs336_basics tests
```

`tests/adapters.py` 中不应再有待实现的 adapter；完整 pytest 必须一次性通过，而不是只通过若干 `-k` 子集。macOS 上因平台条件跳过的 Linux `rlimit` 测试应记录原因；若要做严格跨平台验收，还需在 Linux 重跑这些测试。

自动测试之外，再做一次最小端到端流程：用小语料训练 BPE，保存并重新加载 tokenizer，检查 encode/decode；用极小 token 数据完成若干训练 step，保存 checkpoint，恢复后继续训练，并用恢复的模型生成文本。确认 loss 有限、checkpoint 中 model/optimizer/iteration 齐全、恢复前后状态连续。

最后核对作业要求的 TinyStories/OpenWebText 实验、训练/验证曲线、运行时间、生成样例和消融结论都来自真实运行并已记录。只有 adapters、全套测试、静态检查、端到端 smoke test 和实验交付物都齐全，才算 Assignment 1 完成。

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
