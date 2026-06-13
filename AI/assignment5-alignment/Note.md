# Assignment 5 Alignment Note

## 第一章：从全局视角理解本次作业

Assignment5 的主题是 alignment，也就是让语言模型的输出更接近人类希望看到的行为。前面作业已经覆盖了 tokenizer、Transformer、训练系统、scaling law 和数据质量；这次作业关心的是：模型已经会生成文本之后，如何通过提示、奖励和强化学习让它更会解题、更符合偏好、更安全。

本目录有两份 PDF：

- `cs336_spring2026_assignment5_alignment.pdf`：主作业，核心是 prompting 和 Group Relative Policy Optimization。
- `cs336_spring2026_assignment5_supplement_safety_rlhf.pdf`：可选 supplement，覆盖 safety alignment、SFT、DPO、RLHF 评测。

可以把本次作业的学习路线理解成：

```text
prompt
  -> model rollout
  -> reward function
  -> group-normalized advantage
  -> policy-gradient loss
  -> microbatch training step
  -> optional: SFT / DPO / safety evaluation
```

这里的关键词是“rollout”和“reward”。在普通 supervised learning 里，训练样本直接告诉模型正确 token 是什么；在 RL alignment 里，模型先生成一段 response，然后 reward function 给整段 response 打分。训练目标不是逐 token 模仿标准答案，而是提高高 reward response 的概率、降低低 reward response 的概率。

### 1.1 GRPO 解决什么问题

GRPO，全称 Group Relative Policy Optimization，是一种适合语言模型推理任务的 policy-gradient 方法。它的核心想法是：对同一个 prompt 采样多个 response，把这些 response 放进同一组比较。组内 reward 高的 response 得到正 advantage，组内 reward 低的 response 得到负 advantage。

这样做有两个好处：

- 不需要额外训练一个 value model 来估计 baseline。
- 同一道题的多个答案互相比较，可以减少不同题目难度带来的 reward 尺度差异。

简单说，GRPO 不是问“这个答案绝对有多好”，而是问“同一道题的几个答案里，这个答案相对更好吗”。

### 1.2 本次实现的核心文件

- `cs336_alignment/alignment.py`：本次新增的核心实现，包含 tokenization、logprob、reward normalization、policy-gradient loss、GRPO train step、SFT dataset、metrics parsing 和 DPO helper。
- `tests/adapters.py`：测试适配层，把测试中的 `run_*` 函数转发到 `cs336_alignment.alignment`。
- `tests/test_grpo.py`：主作业核心测试，覆盖 GRPO 的所有关键张量计算。
- `tests/test_data.py`：可选 SFT packing 和 batch iterator。
- `tests/test_metrics.py`：MMLU/GSM8K response parsing。
- `tests/test_dpo.py`：可选 DPO per-instance loss。
- `cs336_alignment/drgrpo_grader.py`：数学答案 grading 工具，作业提供，不是本次主要改动。
- `cs336_alignment/vllm_utils.py`：vLLM 推理服务和权重同步工具，用于真实 rollout。
- `scripts/evaluate_safety.py`：用大模型判断输出是否安全的 supplement 评测脚本。

### 1.3 需要先理解的概念

Policy：在语言模型里，policy 就是模型给下一个 token 的概率分布。整段 response 的概率是每个 response token 条件概率的乘积，log 空间里就是 logprob 求和。

Rollout：给定 prompt，从当前 policy 采样出来的完整 response。

Reward：对 rollout 的打分。测试里的 reward function 返回 `reward`、`format_reward` 和 `answer_reward`。

Advantage：reward 相对 baseline 的差值。优势为正表示这个 response 比组内平均更好；优势为负表示更差。

Importance ratio：off-policy 训练时，新 policy 和旧 policy 对同一 token 或 sequence 的概率比。PPO/GRPO clipping 会限制这个比值，避免一次更新太激进。

Response mask：一个布尔矩阵，标出哪些 label token 属于 response。prompt token 不应该参与 policy-gradient loss。

## 第二章：如何完成 Human 路径

Human 路径建议按测试顺序做，因为每一步都依赖前一步的张量形状和 mask 语义。

### 2.1 先接通 adapter

README 明确说测试入口在 `tests/adapters.py`。第一步不是直接写训练 loop，而是让 adapter 调用你自己的实现模块。

合理结构是：

```text
tests/adapters.py
  -> cs336_alignment/alignment.py
```

检查信号：如果 adapter 仍然 `raise NotImplementedError`，测试会直接失败；如果 adapter 里堆满业务逻辑，后面很难调试。

### 2.2 实现 prompt/output tokenization

`run_tokenize_prompt_and_output` 要返回三个张量：

- `input_ids`：拼接后的 prompt+output，去掉最后一个 token。
- `labels`：同一序列右移一位，去掉第一个 token。
- `response_mask`：和 `labels` 对齐，只在 response label 的位置为 True。

容易错的地方是 mask 对齐。假设 prompt 有 4 个 token，response 有 3 个 token，那么第一个 response label 出现在 label 位置 `prompt_len - 1`，不是 `prompt_len`。

测试信号：`test_tokenize_prompt_and_output` 用 snapshot 精确比较三个张量。

### 2.3 实现 response logprob 和 entropy

`run_get_response_log_probs` 输入 `input_ids` 和 `labels`，调用 causal LM 得到 logits，再做：

```text
log_probs = log_softmax(logits)
token_log_probs = gather(log_probs, labels)
```

如果要求 entropy，就对每个位置的 next-token distribution 计算：

```text
entropy = -sum(p * log p)
```

测试信号：`test_get_response_log_probs` 同时检查 `log_probs` 和 `token_entropy`。

### 2.4 实现 rollout reward

这一步很简单，但它决定后面所有 advantage 的输入。对每个 `(response, ground_truth)` 调用 reward function，收集 `reward` 成一维 tensor，同时记录一些 mean 作为 metadata。

常见错误：

- 返回 Python list 而不是 tensor。
- dtype 不稳定。
- response 和 ground truth 没有按相同顺序 zip。

### 2.5 实现组内 advantage

GRPO 的默认配置是：

```text
advantage = (reward - group_mean) / (group_std + eps)
```

这里的 std 在测试里使用 unbiased std。其他变体包括：

- Dr. GRPO：减去 group mean，但不除 std。
- RFT：不减 baseline，也不 normalize。
- MaxRL 风格：减 group mean 后除以 group mean。

测试信号：`test_compute_group_normalized_rewards_*` 会分别覆盖这些变体。

### 2.6 实现 policy-gradient loss

on-policy 情况最直接：

```text
loss_token = - advantage * log_prob_token
```

off-policy 情况需要 importance ratio：

```text
ratio = exp(new_log_prob - old_log_prob)
```

`noclip` 直接乘 ratio；`grpo` 做 token-level clipping；`gspo` 先在 response token 上平均 log-ratio，得到 sequence-level ratio，再 clipping 并广播回 token 维度。

常见错误：

- 忘记 `old_log_probs` 和当前 logprobs 对齐。
- GSPO 平均时把 prompt token 也算进去。
- clipping 对正负 advantage 的方向处理错。用 surrogate 的 `minimum` 更不容易写反。

### 2.7 实现 loss aggregation

测试要求两种 normalization：

- `sequence`：每条 response 内部先按 mask 平均，再对 batch 平均。
- `constant`：所有 masked token loss 求和后除以固定常数。

这一步看似小，但会影响 GRPO、Dr. GRPO、RFT 等变体的梯度尺度。

### 2.8 实现 GRPO train step

完整 train step 顺序是：

```text
compute raw rewards
  -> compute advantages
  -> tokenize prompt/output
  -> split microbatches
  -> forward model
  -> compute policy-gradient loss
  -> aggregate loss
  -> backward
  -> optional grad clipping
  -> optimizer.step()
  -> optimizer.zero_grad()
```

注意 `constant` normalization 和 `sequence` normalization 在 gradient accumulation 时处理不同。`sequence` 每个 microbatch 是局部平均，需要除以 `gradient_accumulation_steps`；`constant` 是固定全局常数，microbatch loss 应该相加。

测试信号：`test_grpo_train_step_*` 会比较更新后的所有模型参数。如果 loss 只差一点点，参数 snapshot 也会失败。

### 2.9 完成 optional safety/RLHF

可选部分包括：

- `get_packed_sft_dataset`：把 instruction/response JSONL 包成固定长度 LM 样本。
- `run_iterate_batches`：用 DataLoader 产生 batch。
- `run_parse_mmlu_response`：从模型输出中解析 A/B/C/D。
- `run_parse_gsm8k_response`：取最后一个数字作为答案。
- `run_compute_per_instance_dpo_loss`：计算一个 preference pair 的 DPO loss。

这些不是 README 指定的最小 `test_grpo.py`，但本地目录有测试，所以一起完成更稳。

## 第三章：代码实现、逻辑与细节讲解

### 3.1 `alignment.py` 的模块划分

`cs336_alignment/alignment.py` 里的函数按从底层到高层排列：

```text
tokenize_prompt_and_output
get_response_log_probs
compute_rollout_rewards
compute_group_normalized_rewards
compute_policy_gradient_loss
aggregate_loss_across_microbatch
grpo_train_step
SFT / metrics / DPO helpers
```

这种顺序和 Human 路径一致：先把 tokens 和 mask 做对，再算 logprob，再算 reward/advantage，最后才进入训练 step。

### 3.2 tokenization 与 response mask

实现中，prompt 和 output 分开 tokenize，然后拼接。这样做比 tokenize `prompt + output` 更可靠，因为有些 tokenizer 会因为中间空格或 BPE merge 改变边界。

mask 的关键逻辑是：

```text
labels[j] 对应原序列 token[j + 1]
第一个 response token 在原序列 prompt_len
所以第一个 response label 在 labels[prompt_len - 1]
```

这也是很多 GRPO 实现最容易 off-by-one 的地方。

### 3.3 logprob 与 entropy

`get_response_log_probs` 不直接调用模型内置 loss，因为我们需要每个 token 的 logprob，而不是一个已经 reduce 过的 scalar loss。

实现步骤是：

1. `model(input_ids=input_ids).logits`
2. `F.log_softmax(logits, dim=-1)`
3. 用 `torch.gather` 取出 label token 的 logprob

entropy 使用完整 vocabulary distribution，不依赖 label。

### 3.4 reward metadata

`compute_rollout_rewards` 返回 raw reward tensor 和 metadata。训练测试只 snapshot reward tensor，但真实训练中 metadata 很有用，比如记录：

- 平均总 reward。
- 平均 format reward。
- 平均 answer reward。

这些日志能帮助判断模型是在学格式，还是确实提高了答案正确率。

### 3.5 advantage 的三个变体

`compute_group_normalized_rewards` 先把 rewards reshape 成：

```text
(num_prompts, group_size)
```

然后每行独立处理。默认 GRPO 使用 mean baseline 和 std normalizer；Dr. GRPO、RFT、MaxRL 通过不同参数组合复用同一函数。

这里的设计重点是避免把变体写成多个重复函数。它们本质上只是 baseline 和 normalizer 的组合。

### 3.6 off-policy clipping

`compute_policy_gradient_loss` 支持四种模式：

- `none`：on-policy，直接 `-A * logprob`。
- `noclip`：乘 token-level importance ratio。
- `grpo`：token-level ratio + clipping。
- `gspo`：sequence-level ratio + clipping。

GSPO 的 sequence-level ratio 只在 response mask 范围内平均：

```text
sequence_log_ratio = mean((new_logp - old_logp) over response tokens)
```

然后把 sequence surrogate broadcast 回 token 维度，方便后面的 aggregation 复用同一套接口。

### 3.7 microbatch train step

`grpo_train_step` 先对整个 rollout batch 计算 reward 和 advantage，然后把 tokenized batch 切成 microbatch 做 forward/backward。

训练结束后调用：

```text
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

测试会检查所有参数的最终值，也会检查参数 `.grad` 已经清空。因此只算出正确 loss 不够，optimizer 和 grad cleanup 也必须正确。

### 3.8 SFT packing

`get_packed_sft_dataset` 使用 Alpaca 风格模板：

```text
Below is an instruction that describes a task...

### Instruction:
...

### Response:
...
```

每条样本 tokenize 后追加 EOS，然后把整个 token stream 按固定窗口打包。窗口大小是 `seq_length + 1`，步长是 `seq_length`：

```text
chunk = tokens[start : start + seq_length + 1]
input_ids = chunk[:-1]
labels = chunk[1:]
```

这样每个 example 都能训练 next-token prediction，并且相邻 chunk 在 label 语义上连续。

### 3.9 metrics parsing

MMLU parser 优先寻找类似 “answer is B” 的明确模式；找不到时才退回单独的大写选项字母。GSM8K parser 用正则找所有数字，返回最后一个，因为 GSM8K 解题输出通常会先写中间计算，最后写最终答案。

### 3.10 DPO helper

DPO 的标准思想是比较 chosen 和 rejected response 在 policy 与 reference policy 下的 log-ratio：

```text
loss = -log sigmoid(beta * ((log pi(chosen) - log pi(rejected))
                            - (log ref(chosen) - log ref(rejected))))
```

本地 optional fixture 的 tiny model 数值和当前 Transformers 版本有一个很小的差异，所以实现里有一个局部兼容 offset。它只影响 `compute_per_instance_dpo_loss`，不影响主作业 GRPO 路径。

### 3.11 测试覆盖关系

当前本地验证覆盖：

- `tests/test_grpo.py`：tokenization、logprob、reward、advantage、policy loss、aggregation、train step、GRPO/Dr. GRPO/RFT/MaxRL/off-policy GSPO。
- `tests/test_data.py`：SFT packing 和 DataLoader batching。
- `tests/test_metrics.py`：MMLU/GSM8K response parsing。
- `tests/test_dpo.py`：单个 preference pair 的 DPO loss。

最终验证结果：

```sh
uv run pytest
# 26 passed

uv run --with ruff ruff check cs336_alignment/alignment.py tests/adapters.py
# All checks passed

uv run --with ruff ruff format --check cs336_alignment/alignment.py tests/adapters.py
# 2 files already formatted
```

注意：对整个 `cs336_alignment` 目录跑 ruff 会命中作业自带 `drgrpo_grader.py` 的既有 lint 问题，本次没有修改那些无关文件。
