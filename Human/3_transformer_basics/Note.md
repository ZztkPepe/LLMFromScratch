# Assignment 1 Basics Note

## 第一章：从全局视角理解本次作业

本作业要从零搭出训练小型语言模型的最小闭环，而不只是实现一个 Transformer 层：

```text
原始文本 -> UTF-8 bytes -> BPE token IDs -> TransformerLM logits
       -> cross-entropy loss -> AdamW 更新 -> checkpoint / 文本生成
```

这条链路由四类部件组成：

1. **Tokenizer**：把任意文本映射为整数 ID，并能还原为文本。
2. **Transformer language model**：输入已有 token，输出每个位置的下一个 token 的全词表 logits。
3. **训练工具**：稳定计算 loss、采样 `(x, y)`、更新参数、保存训练状态。
4. **训练和生成入口**：在真实数据上串起上述部件。

这里使用 byte-level BPE。任意 Unicode 文本先编码为 UTF-8 bytes，所以不存在 unknown token；BPE 再合并高频相邻 byte 序列，形成更长的 subword token。模型使用 pre-norm、decoder-only Transformer：

```text
token IDs -> token embedding -> TransformerBlock x num_layers
          -> final RMSNorm -> LM head -> logits

x -> x + causal self-attention(RMSNorm(x))
  -> z + SwiGLU(RMSNorm(z))
```

必须先理解三件事：causal mask 禁止当前位置读取未来 token；logits 是未归一化分数而不是概率；训练状态除了权重，还有 AdamW 的动量和当前 iteration。

## 第二章：Human 路径——按文件、接口与数据流完成实现

本章合并原先第二、三章。它既说明做什么，也说明每项实现如何协作。边界固定：算法写入 `cs336_basics/`；`tests/adapters.py` 只负责构造对象、装入测试权重并转发调用，不能承载业务逻辑。

### 0. 作业边界与完整交付物

这是一个 **from-scratch** 作业：不能直接调用 `torch.nn.Linear`、`torch.nn.Embedding`、`torch.nn.functional.linear`、`torch.nn.functional.embedding`、`torch.nn.functional.softmax`、`torch.nn.functional.cross_entropy` 或 `torch.optim.AdamW` 来代替自己的实现。允许使用 `torch.nn.Parameter`、`nn.Module`/`nn.ModuleList` 等容器、`torch.optim.Optimizer` 基类，以及普通 PyTorch tensor 运算；手册也明确指定 `torch.nn.init.trunc_normal_` 用于初始化。测试所用的 PyTorch 参考实现只能拿来比较结果，不能成为你的实现。

“完成作业”除代码外还包括 `writeup.pdf` 中的书面回答和真实实验记录。下面是无需再查手册的交付清单；答案、曲线和耗时必须来自自己实际观察，不能编造：

| 模块 | 需要提交/回答的内容 |
| --- | --- |
| Unicode | `chr(0)` 是什么、`repr` 与打印为何不同、它出现在字符串中时的表现；比较 UTF-8/16/32，给出错误逐 byte UTF-8 解码的反例，以及一个非法两 byte 序列。 |
| BPE | 在 TinyStories 上训练 10K vocab（含 `<|endoftext|>`），序列化 vocab/merges，报告时间、内存、最长 token 和 profiling 瓶颈；在 OpenWebText 上训练 32K vocab，比较两种词表。 |
| Tokenizer | 对两种数据各抽 10 篇文档，报告 bytes/token；用 TinyStories tokenizer 编 OpenWebText 的变化；报告 bytes/s 并估算 825GB The Pile 的耗时；将 train/dev 编为 `uint16` 并说明 16 位足够的原因。 |
| Transformer | 完成 GPT-2 XL、small、medium、large 的参数/FLOPs 分析；再把 XL context length 改为 16,384，说明总 FLOPs 和各组成比例变化。 |
| 训练 | 记录不同 SGD learning rate 的行为；完成 AdamW 训练内存、FLOPs、MFU 时间的书面分析；实现学习率曲线、数据加载、checkpoint。 |
| 实验 | 提交带 step 与 wall-clock 横轴的日志；TinyStories 学习率 sweep、batch-size sweep、至少 256 tokens 的生成；RMSNorm、post-norm、NoPE、SwiGLU vs SiLU 消融；OpenWebText 训练曲线与生成。Leaderboard 是额外的最终挑战。 |

不要把“pytest 通过”当成交付完成：它只覆盖一部分核心接口，不会替你产生上述书面答案、曲线、模型样例或性能数据。

### 1. 文件地图、依赖和推荐顺序

| 顺序 | 文件 | 要完成的内容 | 公开入口 |
| --- | --- | --- | --- |
| 0 | `tests/adapters.py` | 测试适配、加载权重、调用实现 | 所有 `run_*` / `get_*` |
| 1 | `cs336_basics/bpe.py` | byte-level BPE 训练 | `train_bpe` |
| 2 | `cs336_basics/tokenizer.py` | BPE 编码与解码 | `Tokenizer` |
| 3 | `cs336_basics/model.py` | 全部 Transformer 模块 | 模块类、attention 函数 |
| 4 | `cs336_basics/nn_utils.py`（创建） | 无状态数值工具 | `softmax`、`cross_entropy`、`gradient_clipping` |
| 5 | `cs336_basics/optim.py`（创建） | 有状态优化器与学习率曲线 | `AdamW`、`get_lr_cosine_schedule` |
| 6 | `cs336_basics/data.py`（创建） | next-token batch 采样 | `get_batch` |
| 7 | `cs336_basics/serialization.py`（创建） | checkpoint 保存和恢复 | `save_checkpoint`、`load_checkpoint` |
| 8 | `cs336_basics/training.py`、`generation.py`（实验时创建） | 真实训练、评估和生成 | 自行定义的入口 |

按这个顺序完成：BPE 与 Tokenizer 是一条独立链；模型是另一条独立链；训练工具依赖模型产生 logits；最后才在真实数据上串联。不要先写训练循环，因为那会把 tokenizer、模型、loss 和优化器的错误混在一起。

### 2. Step 0：接通 `tests/adapters.py`

**要修改的文件**：`Human/3_transformer_basics/tests/adapters.py`。

每个 adapter 只做三件事：

1. 用传入的维度创建 `cs336_basics` 中的对象，或调用对应纯函数。
2. 用测试传入的 `weights` 写入参数；整份 state dict 时优先调用 `load_state_dict`。
3. 调用 forward/函数，并原样返回结果。

例如 `run_linear(...)` 创建 `Linear(d_in, d_out)`，装入形状 `(d_out, d_in)` 的权重，再调用它；它不应自己实现矩阵乘法。`get_adamw_cls()` 必须返回你的 `AdamW` **类**，不是已经实例化的对象。

| Adapter 入口 | 应转发到 |
| --- | --- |
| `run_train_bpe` | `bpe.train_bpe` |
| `get_tokenizer` | `tokenizer.Tokenizer(...)` |
| `run_linear`、`run_embedding`、`run_swiglu`、`run_rmsnorm` | `model.py` 的对应模块 |
| `run_silu`、`run_scaled_dot_product_attention` | `model.py` 的纯函数 |
| `run_rope`、两个 `run_multihead_self_attention` | RoPE、MHA 模块 |
| `run_transformer_block`、`run_transformer_lm` | 对应模块，加载权重后 forward |
| `run_softmax`、`run_cross_entropy`、`run_gradient_clipping` | `nn_utils.py` |
| `get_adamw_cls`、`run_get_lr_cosine_schedule` | `optim.py` |
| `run_get_batch` | `data.py` |
| `run_save_checkpoint`、`run_load_checkpoint` | `serialization.py` |

**完成信号**：失败调用栈已进入目标实现，不会停在 adapter 的 `NotImplementedError`；所有 `run_*` 与 `get_*` 都有明确转发目标。

### 3. Step 1：在 `bpe.py` 训练 byte-level BPE

**要修改的文件**：`Human/3_transformer_basics/cs336_basics/bpe.py`。

**必须实现的函数**：

```python
train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
```

| 输入/输出 | 含义 |
| --- | --- |
| `input_path` | UTF-8 训练语料路径。 |
| `vocab_size` | 最终词表总大小，包含 special tokens 和 256 个单 byte。 |
| `special_tokens` | 如 `['<|endoftext|>']`；必须整体保留，不能被普通 merge 跨越。 |
| `vocab` 返回值 | `token_id -> token_bytes` 的映射。 |
| `merges` 返回值 | 按学习先后排列的 `(left_bytes, right_bytes)` 列表。 |

**函数要求**：

1. 初始化词表时按 `special_tokens` 给定顺序先放入 special token，再按数值 `0..255` 放入 256 个单 byte；ID 顺序必须稳定，`vocab_size` 计入这些初始项。
2. 先按 special token 隔开文本，再用 GPT-2 pre-tokenization regex 切普通片段；special token 只作边界，不能进入普通 pair 统计。
3. 每轮选择频次最高的相邻 pair；并列时选字典序更大的 pair；创建拼接后的新 token，并记录这次 merge。
4. merge 会改变后续 pair 频次，不能只做一次全局排序。
5. 为通过速度测试，维护 pair 频次和 pair 所在的 pre-token，只更新受本轮 merge 影响的条目，不能每轮重扫整份语料。

预切词必须使用下列模式，并使用第三方 `regex` 包而非标准库 `re`，因为模式包含 `\p{L}`、`\p{N}`：

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

应使用 `finditer` 边迭代边累计 `Counter[tuple[bytes, ...]]`，避免先把全部 pre-token 存入列表。并行只适合 pre-tokenization：按 special token 的**起始位置**切文件块，分别统计后合并 Counter；BPE merge 本身是顺序相关的，不能并行执行。空语料、`vocab_size` 小于等于初始词表大小、没有剩余 pair 时都应自然停止。

**清晰的实现逻辑**：

```text
语料 -> special-token 边界切分 -> GPT-2 regex 预切词
     -> 每个预切词转为单 byte 元组并计数
     -> 建立全局 pair_counts 与 pair_to_word_ids
     -> 选择最佳 pair，创建新 token
     -> 仅重写包含该 pair 的预切词，增量更新统计
     -> 重复直到达到 vocab_size 或没有 pair
```

验证：`uv run pytest tests/test_train_bpe.py -q`。special-token 测试要求除 special token 本身外，普通词表项不包含 `b'<|'`；结果不一致时优先检查边界切分、tie-break 和局部统计更新。

### 4. Step 2：在 `tokenizer.py` 重放 BPE 规则

**要修改的文件**：`Human/3_transformer_basics/cs336_basics/tokenizer.py`。

| 名称 | 输入 | 输出 | 必须满足的要求 |
| --- | --- | --- | --- |
| `Tokenizer(vocab, merges, special_tokens)` | `id -> bytes`、有序 merge、可选 special token | tokenizer 对象 | 建 `bytes -> id` 反查表与 `pair -> rank`；若 special token 不在 vocab 中，追加它；special token 按长度降序匹配。 |
| `Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=None)` | 序列化 vocab/merge 文件路径 | tokenizer 对象 | 读回 `train_bpe` 保存的内容，转换为 `dict[int, bytes]` 和 `list[tuple[bytes, bytes]]` 后委托给构造器。 |
| `encode(text)` | 一个 `str` | `list[int]` | 先识别 special token；普通片段按 GPT-2 regex 切分，再做 BPE。 |
| `encode_iterable(iterable)` | `Iterable[str]` | `Iterator[int]` | 逐块产出，不先拼接全部输入。 |
| `decode(ids)` | token ID 序列 | `str` | 查回 bytes、拼接、一次 UTF-8 解码；非法 byte 用 `errors='replace'`。 |

**编码逻辑**：训练得到的 `merges` 顺序就是推理时的优先级。每个普通 pre-token 先拆成单 bytes；在当前相邻 pair 中选择 **rank 最小**（最早学习）的 pair 合并，直到没有可用 pair。不要按当前频次合并，频次只属于训练阶段。

```text
' hello' -> GPT-2 pre-token ' hello' -> bytes
         -> 反复合并 rank 最小的相邻 pair -> vocab ID
```

若同时存在 `<|endoftext|>` 与 `<|endoftext|><|endoftext|>`，必须先匹配更长者，否则会错误切为两个短 token。验证：`uv run pytest tests/test_tokenizer.py -q`。应覆盖空串、英文、Unicode、special token 和流式编码；`decode(encode(text))` 必须还原原文，流式 ID 必须和一次性编码一致。

`encode_iterable` 的难点不是 `yield from self.encode(chunk)` 本身，而是**不能让一个 special token 或普通 pre-token 被任意 chunk 边界切断**。实现应保留尚不能确认结束的尾部缓冲区：只产出已经完整结束的片段，把可能与下一块连接的残余文本带到下一轮。内存目标是相对于整个文件近似常量，而不是一次性读入 5MB/更大文件。将 token ID 写盘时推荐 `np.uint16`，前提是 vocab ID 最大值小于 `2**16`；10K 和 32K 词表均满足，较 `int64` 节省空间。

### 5. Step 3：在 `model.py` 搭出 Transformer

**要修改的文件**：`Human/3_transformer_basics/cs336_basics/model.py`。

先完成基础模块，再完成 attention，最后组合为 block 和 LM。形状中的 `...` 表示任意前导 batch 维度；实现不能只支持二维输入。

#### 5.1 基础模块与输入输出

| 名称 | 输入 -> 输出 | 实现要求 |
| --- | --- | --- |
| `Linear(in_features, out_features, device=None, dtype=None)` | `(..., d_in) -> (..., d_out)` | `weight` 形状为 `(d_out, d_in)`；沿最后一维投影；没有 bias。 |
| `Embedding(num_embeddings, embedding_dim, device=None, dtype=None)` | 整数 ID `(...) -> (..., d_model)` | 参数形状 `(num_embeddings, embedding_dim)`；按 ID 索引，不需要 one-hot。 |
| `silu(x)` | `(...) -> (...)` | 逐元素 `x * sigmoid(x)`。 |
| `RMSNorm(d_model, eps=1e-5, device=None, dtype=None)` | `(..., d_model) -> (..., d_model)` | 最后一维的均方根归一化后乘 `(d_model,)` 可学习 gain；`eps` 在开方前加入。 |
| `SwiGLU(d_model, d_ff)` | `(..., d_model) -> (..., d_model)` | 三个无 bias 线性层：`W2(SiLU(W1(x)) * W3(x))`；`W1/W3` 升到 `d_ff`，`W2` 投回 `d_model`。 |

所有模块先 `super().__init__()`，再将可训练 tensor 包为 `nn.Parameter`，创建时把 `device` 和 `dtype` 传给底层 `torch.empty`。下表给出了手册规定的初始化；这部分即使 snapshot 测试会覆盖权重，也会直接影响后续真实训练：

| 参数 | 初始分布 | 具体调用/公式 |
| --- | --- | --- |
| 任意 `Linear.weight`（包括 Q/K/V/O、W1/W2/W3、LM head） | 截断正态 | `std = sqrt(2 / (d_in + d_out))`；从 `N(0, std²)` 截断到 `[-3*std, 3*std]`，使用 `torch.nn.init.trunc_normal_`。 |
| `Embedding.weight` | 截断标准正态 | 从 `N(0, 1)` 截断到 `[-3, 3]`。 |
| `RMSNorm.weight` | 常数 | 全部初始化为 1。 |

Linear 在数学上存储的是 `W ∈ R^(d_out x d_in)`，将每个输入视为列向量时有 `y = W x`；但 PyTorch 输入的最后一维是 row-major，因此 forward 应等价于 `y = x W^T`，即 `x @ weight.T`，或写成 `einsum(x, weight, '... d_in, d_out d_in -> ... d_out')`。直接写 `x @ weight` 会维度不符或隐蔽地产生错误结果。Embedding 的 forward 只做 `weight[token_ids]`：若 `token_ids` 是 `(B, T)`，输出必为 `(B, T, d_model)`。

RMSNorm 的逐元素公式为：

```text
rms(x) = sqrt(mean(x², dim=-1, keepdim=True) + eps)
output_i = x_i / rms(x) * gain_i
```

它不减均值。应先保存 `in_dtype`，把输入 upcast 到 `float32` 完成平方、均值、开方和缩放，再 cast 回 `in_dtype`，避免 `float16`/`bfloat16` 求平方溢出。

#### 5.2 RoPE 与 scaled dot-product attention

**需要实现的对象/函数**：

```text
RotaryPositionalEmbedding(d_k, theta, max_seq_len)
    forward(in_query_or_key, token_positions)

scaled_dot_product_attention(Q, K, V, mask=None)
```

| 名称 | 输入 | 输出 | 要求 |
| --- | --- | --- | --- |
| RoPE forward | `(..., T, d_k)` 与 `(..., T)` 位置 ID | 同形状 tensor | 最后一维两两旋转；缓存 `0..max_seq_len-1` 的 cos/sin，按 token position 取出并广播。 |
| attention | `Q(..., Q, d_k)`、`K(..., K, d_k)`、`V(..., K, d_v)`、可选 `mask(..., Q, K)` | `(..., Q, d_v)` | 计算 `QK^T / sqrt(d_k)`；`False` mask 位置不可被关注；沿 keys 维 softmax 后加权 V。 |

RoPE 的 `d_k` 必须为偶数。对位置 `i`、第 `k` 对特征，先计算

```text
angle(i, k) = i / theta ** (2*k / d_k),  k = 0, ..., d_k/2 - 1
```

若一对输入为 `(u, v)`，手册采用的旋转为：

```text
(u', v') = (cos(angle) * u + sin(angle) * v,
            -sin(angle) * u + cos(angle) * v)
```

不要构造每个位置完整的 `(d_k, d_k)` 旋转矩阵。初始化时预计算二维 sin/cos buffer，并以 `register_buffer(..., persistent=False)` 保存；它不是参数、不参与优化、不应进入 checkpoint。forward 只用 `token_positions` 从 buffer 取对应行，再 `unsqueeze` 到能广播给输入的 batch/head 维。RoPE 只作用在 Q/K，绝不作用在 V。

attention 的完整计算为 `scores = Q @ K.transpose(-2, -1) / sqrt(d_k)`，随后对所有 `mask == False` 的 score 填 `-inf`（或 dtype 可表示的最小数），再在最后一个 keys 维 softmax，最后 `probs @ V`。mask 中 `True` 表示允许，`False` 表示屏蔽；屏蔽位置的概率必须为 0，允许位置的概率和为 1。转置必须发生在最后两个维度，所以函数要同时支持三维和带 batch/head 的四维输入。

#### 5.3 Multi-head self-attention

**需要实现的类**：可命名为 `MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, use_rope)`。名称可调整，但属性命名最好支持 state dict 键：`q_proj.weight`、`k_proj.weight`、`v_proj.weight`、`output_proj.weight`。

**输入输出**：`x(..., T, d_model) -> (..., T, d_model)`；必须验证 `d_model % num_heads == 0`，其中 `d_head = d_model / num_heads`，并按手册取 `d_k = d_v = d_head`。Q/K/V 三个投影各是一次完整矩阵乘法；不要为每个 head 写 Python 循环。

**实现逻辑**：

```text
x
  -> q_proj / k_proj / v_proj: (..., T, d_model)
  -> reshape + transpose: (..., num_heads, T, d_head)
  -> （启用时）仅对 Q、K 应用 RoPE
  -> 下三角 causal mask + scaled_dot_product_attention
  -> transpose + reshape 回 (..., T, d_model)
  -> output_proj
```

causal mask 第 `i` 行只能允许 `0..i` 列；反过来会让模型偷看未来。无 RoPE 和有 RoPE 的 adapter 分别验收两种模式。

#### 5.4 `TransformerBlock` 与 `TransformerLM`

| 类 | 输入 -> 输出 | 组成与要求 |
| --- | --- | --- |
| `TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)` | `(..., T, d_model) -> (..., T, d_model)` | 两个 RMSNorm、一个带 RoPE 的 MHA、一个 SwiGLU；严格 pre-norm residual：`x = x + attn(ln1(x))`，再 `x = x + ffn(ln2(x))`。 |
| `TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)` | ID `(B, T) -> logits (B, T, vocab_size)` | embedding、`num_layers` 个 block、final RMSNorm、LM head；`T <= context_length`。 |

`run_transformer_block` 和 `run_transformer_lm` 会传入参考 state dict。最稳妥的实现是使属性路径与其一致，例如 `token_embeddings`、`layers`、`ln_final`、`lm_head`，然后 adapter 调用 `load_state_dict(weights)`；若名称不同，adapter 必须逐项映射，不能使用随机初始化。

**模型验证顺序**：先 `test_linear`、`test_embedding`、`test_swiglu`、`test_rope`，再 attention、block，最后 `test_transformer_lm`。整模不一致时，先查 shape/state dict 键、reshape/transpose、RoPE 的作用范围、mask 方向和 residual 顺序。

### 6. Step 4：创建训练工具文件

这四个文件都创建在 `Human/3_transformer_basics/cs336_basics/`。数值函数不持有状态；optimizer 持有跨 step 状态；sampler 只取数据；checkpoint 只保存状态。

#### 6.1 `nn_utils.py`

| 函数 | 输入 -> 输出 | 实现逻辑与要求 |
| --- | --- | --- |
| `softmax(x, dim)` | 任意 tensor -> 同形状概率 tensor | `shifted = x - x.max(dim, keepdim=True)`，再算 `exp(shifted) / exp(shifted).sum(dim, keepdim=True)`；输入整体加常数后结果不变。 |
| `cross_entropy(inputs, targets)` | logits `(..., V)`、目标 `(...)` -> 标量 | 对每个位置取正确类别负对数概率后，跨全部 batch-like 维平均；最后一维必须是 vocab。 |
| `gradient_clipping(parameters, max_l2_norm)` | 参数 iterable -> `None` | 汇总所有非空 `.grad` 的全局 L2 norm；超过阈值时以同一比例原地缩放所有 gradient。 |

交叉熵不需要显式构造概率。令 `z = inputs - inputs.max(dim=-1, keepdim=True).values`，每个样本的 loss 可以直接写成：

```text
loss = log(sum(exp(z))) - z[target]
```

这正是 `-logsoftmax(inputs)[target]`，其中减最大值不会改变结果，并且避免大 logit 的 `exp` overflow。语言模型训练时，先把 `(B, T, V)` logits 展平为 `(-1, V)`，把 `(B, T)` 的下一个 token targets 展平为 `(-1,)`，再调用该函数。

梯度裁剪的全局范数不是逐参数分别裁剪：`total_norm = sqrt(sum(sum(p.grad**2) for p if p.grad is not None))`。仅在 `total_norm > max_l2_norm` 时，将每一个非空梯度原地乘以 `max_l2_norm / (total_norm + 1e-6)`。这一步必须位于 `loss.backward()` 之后、`optimizer.step()` 之前。

#### 6.2 `optim.py`

**类**：`AdamW(torch.optim.Optimizer)`。构造器至少支持测试使用的 `params`、`lr`、`weight_decay`、`betas`、`eps`；`step()` 更新参数。

每个参数独立保存 `step`、一阶矩 `exp_avg`、二阶矩 `exp_avg_sq`，并以全 0、与参数同形状的 tensor 初始化。跳过 `p.grad is None` 的参数。对于当前梯度 `g`，第 `t` 步（**从 1 开始**）必须按以下顺序执行：

```text
alpha_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)
p       = p - lr * weight_decay * p
m       = beta1 * m + (1 - beta1) * g
v       = beta2 * v + (1 - beta2) * g**2
p       = p - alpha_t * m / (sqrt(v) + eps)
```

这说明 weight decay 与梯度更新解耦，且发生在 moment 更新之前。不要把 `weight_decay * p` 加到 `g`，否则那是 L2 regularization，不是 AdamW。实现时要在无梯度跟踪的上下文中原地更新参数，以免把 optimizer 运算接进 autograd 图。

**函数**：

```python
get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float
```

规则和公式如下，边界要严格按不等号实现：

```text
it < T_w:
    lr = (it / T_w) * lr_max
T_w <= it <= T_c:
    lr = lr_min + 0.5 * (1 + cos(pi * (it - T_w) / (T_c - T_w))) * (lr_max - lr_min)
it > T_c:
    lr = lr_min
```

因此 `it=0` 的学习率为 0，`it=T_w` 为 `lr_max`，`it=T_c` 为 `lr_min`。训练循环每个 iteration 都应把函数结果写入每个 optimizer parameter group 的 `lr`。

#### 6.3 `data.py`

**函数**：

```python
get_batch(dataset, batch_size, context_length, device) -> tuple[torch.Tensor, torch.Tensor]
```

输入是一维 token ID NumPy array。对随机起点 `s` 返回：

```text
x = dataset[s : s + context_length]
y = dataset[s + 1 : s + context_length + 1]
```

输出为两个形状 `(batch_size, context_length)` 的 `torch.long` tensor，并移到 `device`。合法起点是 `0` 至 `len(dataset) - context_length - 1`：多取一位会越界，少取一位会遗漏最后一个合法样本。真实数据不要整体载入内存：把 ID 保存为 `.npy` 后通过 `np.load(path, mmap_mode='r')` 读取，或直接使用 `np.memmap` 并明确匹配原始 `dtype`。抽样前可检查 ID 均小于 vocab size，防止以错误 dtype 读取导致 silent corruption。

#### 6.4 `serialization.py`

**函数**：

```python
save_checkpoint(model, optimizer, iteration, out) -> None
load_checkpoint(src, model, optimizer) -> int
```

保存时把 `model.state_dict()`、`optimizer.state_dict()`、`iteration` 放入同一个可由 `torch.save` 序列化的对象；`out` 可是路径或二进制文件对象。加载时恢复传入对象的前两项，并返回保存的 iteration。

checkpoint 不只是权重副本：缺 optimizer state 会丢失 AdamW 动量；缺 iteration 会让学习率 schedule 无法接上。字典键名称可以自定，但保存与加载必须一致，例如 `{'model_state_dict': ..., 'optimizer_state_dict': ..., 'iteration': ...}`；`out`/`src` 都应同时接受路径与二进制 file-like object。

验证：

```sh
uv run pytest tests/test_nn_utils.py -q
uv run pytest tests/test_optimizer.py -q
uv run pytest tests/test_data.py -q
uv run pytest tests/test_serialization.py -q
```

### 7. Step 5：核心接口通过后，创建真实训练与生成入口

`training.py` 与 `generation.py` 不是当前 adapter 的测试接口。只有开始 TinyStories/OpenWebText 实验时才创建，且应保持最小职责：

| 文件 | 函数 | 输入 -> 输出 | 实现逻辑 |
| --- | --- | --- | --- |
| `training.py` | `train(...)` | 模型、token 数据、optimizer、配置 -> 训练记录/checkpoint | 循环执行 `get_batch -> model -> cross_entropy -> backward -> clipping -> optimizer.step`，定期记录 train/valid loss。 |
| `training.py` | `estimate_loss(...)` | 模型、验证数据、采样配置 -> 平均 loss | 临时进入 eval/no-grad，多次采样平均，再恢复训练模式。 |
| `generation.py` | `generate(...)` | 模型、初始 IDs、最大新 token 数、`temperature`、`top_p`、EOS ID -> 新 IDs | 每轮截取最近不超过 `context_length` 的 ID，取最后位置 logits，按 temperature/top-p 采样并追加；到长度或 EOS 停止。 |

完整数据流如下：

```text
训练语料
  -> train_bpe 得到 vocab / merges
  -> Tokenizer.encode_iterable 得到 token 数据
  -> get_batch 得到 x 与右移一位的 y
  -> TransformerLM(x) 得到 logits
  -> cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
  -> backward / gradient clipping / AdamW.step
  -> save_checkpoint
  -> load_checkpoint 后 generate
```

实验记录必须来自真实运行：数据版本、模型配置、随机种子、token 数、训练/验证 loss、耗时和生成样例。单元测试通过只证明接口行为正确，不等于实验已经完成。

#### 7.1 训练循环的每一步及其顺序

一次 iteration 的顺序不能随意交换：

```text
1. optimizer.zero_grad()                 # 清除上一轮梯度
2. x, y = get_batch(train_data, ...)     # x 和 y 均为 (B, T)
3. logits = model(x)                     # (B, T, V)
4. loss = cross_entropy(logits.flatten(0, 1), y.flatten())
5. loss.backward()                       # 让 autograd 填充 p.grad
6. gradient_clipping(model.parameters(), max_norm)
7. 为每个 parameter group 更新本 step 的 schedule lr
8. optimizer.step()                      # 读取 grad 和自己的 moment state
9. 定期：estimate_loss、记录日志、save_checkpoint
```

验证函数应使用 `model.eval()` 和 `torch.no_grad()`，多次独立采样 validation batch 后平均 loss；结束时恢复原来的训练模式。日志每条至少包括 iteration、累计 tokens、train loss、valid loss、当前 lr、wall-clock seconds；这样每条实验曲线都能以 step 和真实时间作横轴。

TinyStories 的手册起始配置是：`vocab_size=10_000`、`context_length=256`、`d_model=512`、`d_ff=1344`、`rope_theta=10_000`、4 layers、16 heads，约 17M 个非 embedding 参数；总处理 token 数约为 327,680,000，即 `batch_size * total_steps * context_length` 应大致相等。学习率、warmup、AdamW 的 `betas`/`eps` 与 weight decay 需要自行调参。资源有限时，CPU/MPS 可以将总 token 数降至约 40M，同时把 TinyStories 验证 loss 目标从 1.45 放宽到 2.00；cosine 衰减终点必须随总 steps 改到最后一步。

#### 7.2 生成：temperature 和 top-p 的准确逻辑

只使用当前上下文的**最后一个位置** logits：`v = model(context)[:, -1, :]`。先按 temperature 缩放并 softmax：

```text
q_i = exp(v_i / temperature) / sum_j exp(v_j / temperature)
```

temperature 趋近 0 时分布趋向 argmax；温度较高时采样更随机。若启用 nucleus/top-p，按 `q` 从大到小排序，取累计概率首次达到 `p` 的最小 token 集合 `V(p)`，把其余概率置 0，再对保留概率重新归一化，最后用多项式采样选一个 ID。不能只截断而不重归一化。将它追加到完整已生成序列；送入模型前只截取最近 `context_length` 个 ID；遇到 `<|endoftext|>` 的 ID 或达到 `max_new_tokens` 立即停止。

生成实验的交付物是至少 256 个 token 的文本，或在首次 EOS 前停止，并附上流畅度评价及至少两个影响质量的因素。小模型的效果受训练 token 数、数据质量、tokenizer、模型容量、学习率，以及 temperature/top-p 等采样参数共同影响。

#### 7.3 消融和真实实验的最小执行清单

每个实验仅改变一个自变量，并保留基线的随机种子、数据、总 token 数、验证频率与日志格式。必须至少完成：

1. 学习率 sweep：包含至少一次发散运行，说明“稳定边缘”和最佳收敛速度的关系。
2. batch-size sweep：从 1 到硬件可容纳上限，含典型 64/128，并在必要时重新调学习率。
3. RMSNorm 移除：在原最优学习率和更低学习率下各观察稳定性。
4. post-norm：改为 `z = RMSNorm(x + MHA(x))`、`y = RMSNorm(z + FFN(z))`，与 pre-norm 曲线对比。
5. NoPE：完全不对 Q/K 使用 RoPE，与基线对比。
6. SwiGLU vs SiLU：无门控版本为 `FFN_SiLU(x) = W2(SiLU(W1 x))`；为近似匹配参数量，将其 `d_ff` 设为 `4 * d_model`，而默认 SwiGLU 约为 `8/3 * d_model`（并向 64 的倍数取整）。
7. OpenWebText：使用和 TinyStories 相同的模型架构与总 iteration，重新调必要超参，解释为何 loss 不可直接横向等同、为何同一计算预算下生成更差。

### 8. 最终验收与排错顺序

核心接口完成后运行：

```sh
cd Human/3_transformer_basics
uv run pytest -q
```

提交前还应执行 `make_submission.sh`，确认生成的 `code.zip` 不含数据集、checkpoint 等大文件，并单独准备写完上述所有书面题与实验图表的 `writeup.pdf`。课程手册将 AI 的使用限制单列为课程政策；提交前需自行确认你的实现和材料符合该政策。

定位失败时按依赖倒推：

1. **BPE 不一致**：检查 special-token 边界、字典序 tie-break、merge 后的局部统计。
2. **Tokenizer 不一致**：检查 GPT-2 regex、长 special token 优先级、rank 最小的 merge 是否优先。
3. **模型 snapshot 不一致**：先查 shape 与 state dict 键，再查 reshape/transpose、RoPE 只用于 Q/K、causal mask 和 pre-norm residual。
4. **loss 为 `nan`**：检查 softmax/cross-entropy 是否先减最大 logit。
5. **恢复后训练偏离**：检查 checkpoint 是否同时保存 model、optimizer、iteration。

最终应能把实现闭环讲清：BPE 决定文本如何离散化；Tokenizer 在新文本上重放 merge 顺序；TransformerLM 用上下文 ID 预测下一个 token 的 logits；训练工具将 logits 变为稳定梯度并维护优化状态；checkpoint 保存这条训练轨迹；generation 再把模型预测的 ID 还原为文本。每个文件都服务于这条完整链路。
