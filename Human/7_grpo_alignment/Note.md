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

这一章只说明 Human 路径要在哪些文件写代码、每一步完成什么行为、怎样测试。建议按测试顺序做，因为每一步都依赖前一步的张量形状和 mask 语义。

### 1. Task 1：接通 adapter

需要写代码的文件：`Human/7_grpo_alignment/tests/adapters.py`，以及 `Human/7_grpo_alignment/cs336_alignment/alignment.py`。如果 `alignment.py` 不存在，就在 Human 路径下创建它。

README 明确说测试入口在 `tests/adapters.py`。第一步不是直接写训练 loop，而是让 adapter 调用你自己的实现模块。

#### 知识点：alignment 流程由多个可独立验证的纯步骤组成

GRPO train step 看起来是一条大流程，但 tokenization、logprob、reward、advantage、loss 和 aggregation 都有独立的 shape 与数值契约。adapter 把这些步骤暴露给测试，使问题能定位到某一层，而不是只能通过最终参数更新猜测错误来源。

#### 大概实现逻辑

先为每个 `run_*` 找到 `alignment.py` 中唯一对应函数，保持参数名、默认值和返回结构一致。adapter 只做转发，核心函数尽量不依赖全局状态。先让最早的 tokenization 测试进入实现，再按数据流顺序补后续步骤；不要一开始写一个绕过中间接口的整体训练函数。

合理结构是：

```text
tests/adapters.py
  -> cs336_alignment/alignment.py
```

检查信号：如果 adapter 仍然 `raise NotImplementedError`，测试会直接失败；如果 adapter 里堆满业务逻辑，后面很难调试。

怎样测试：

```sh
cd Human/7_grpo_alignment
uv run pytest tests/test_grpo.py::test_tokenize_prompt_and_output -q
```

这条测试可能仍会因为 tokenization 未实现而失败，但失败点应该进入你的实现，而不是 adapter 的 `NotImplementedError`。

### 2. Task 2：实现 prompt/output tokenization

需要写代码的文件：`Human/7_grpo_alignment/cs336_alignment/alignment.py` 和 `Human/7_grpo_alignment/tests/adapters.py`。

`run_tokenize_prompt_and_output` 要返回三个张量：

- `input_ids`：拼接后的 prompt+output，去掉最后一个 token。
- `labels`：同一序列右移一位，去掉第一个 token。
- `response_mask`：和 `labels` 对齐，只在 response label 的位置为 True。

#### 知识点：causal LM 训练是错位一格的预测

模型在位置 `t` 的 logits 用来预测位置 `t+1` 的 token，所以输入与 labels 来自同一完整序列的左右错位切片。response mask 对齐的是“哪些 label 属于回答”，而不是原始拼接序列中回答从哪里开始；这就是边界会提前一个位置的原因。

#### 大概实现逻辑

先分别 tokenize prompt 与 output，记录每个样本真实长度，再拼接并按 batch 需要 padding。由完整序列构造错位后的 input/labels，最后在 labels 坐标系中标记 response token，padding 和 prompt label 都为 False。用极短 prompt/output 手工画索引表，确认三张量 shape 相同且 mask 的 True 数等于有效 response token 数。

容易错的地方是 mask 对齐。假设 prompt 有 4 个 token，response 有 3 个 token，那么第一个 response label 出现在 label 位置 `prompt_len - 1`，不是 `prompt_len`。

测试信号：`test_tokenize_prompt_and_output` 用 snapshot 精确比较三个张量。

怎样测试：

```sh
cd Human/7_grpo_alignment
uv run pytest tests/test_grpo.py::test_tokenize_prompt_and_output -q
```

### 3. Task 3：实现 response logprob 和 entropy

需要写代码的文件：`Human/7_grpo_alignment/cs336_alignment/alignment.py` 和 `Human/7_grpo_alignment/tests/adapters.py`。

`run_get_response_log_probs` 输入 `input_ids` 和 `labels`，调用 causal LM 得到 logits，再做：

#### 知识点：每个 token 的策略概率来自整份词表分布

模型对每个位置输出 vocab 维 logits。训练只需要该位置真实 label 的 log-prob，但 entropy 使用完整分布衡量不确定性。直接对概率取 log 容易产生数值问题，因此应从稳定的 log-softmax 表示出发，并沿 vocab 维选择 label。

#### 大概实现逻辑

先确认模型 logits 的 batch、sequence、vocab 维与 labels 前两维对齐，再在 vocab 维计算稳定 log-prob。通过 labels 索引每个位置对应元素，并去掉被 gather 引入的单元素维。若请求 entropy，从同一 log-prob 恢复概率并沿 vocab 聚合；mask 留给上层 loss/metric 使用，不在这里改变 token 排列。

```text
log_probs = log_softmax(logits)
token_log_probs = gather(log_probs, labels)
```

如果要求 entropy，就对每个位置的 next-token distribution 计算：

```text
entropy = -sum(p * log p)
```

测试信号：`test_get_response_log_probs` 同时检查 `log_probs` 和 `token_entropy`。

怎样测试：

```sh
cd Human/7_grpo_alignment
uv run pytest tests/test_grpo.py::test_get_response_log_probs -q
```

### 4. Task 4：实现 rollout reward

需要写代码的文件：`Human/7_grpo_alignment/cs336_alignment/alignment.py` 和 `Human/7_grpo_alignment/tests/adapters.py`。

这一步很简单，但它决定后面所有 advantage 的输入。对每个 `(response, ground_truth)` 调用 reward function，收集 `reward` 成一维 tensor，同时记录一些 mean 作为 metadata。

#### 知识点：reward 是文本世界到优化信号的边界

reward function 可能返回正确性分数和额外 metadata；训练真正消费的是与 rollout 顺序一一对应的标量向量，监控则消费聚合指标。若顺序、dtype 或设备不稳定，后面的 advantage 即使公式正确也会配错样本。

#### 大概实现逻辑

按原 batch 顺序逐对调用 reward function，把主 reward 与附加字段分开收集。最后一次性构造指定 dtype/device 的一维 Tensor；metadata 仅对存在且可数值聚合的字段求 batch 统计，并使用清晰名称。用不同 reward 的小样本确认位置不被排序或分组打乱。

常见错误：

- 返回 Python list 而不是 tensor。
- dtype 不稳定。
- response 和 ground truth 没有按相同顺序 zip。

怎样测试：

```sh
cd Human/7_grpo_alignment
uv run pytest tests/test_grpo.py::test_compute_rollout_rewards -q
```

### 5. Task 5：实现组内 advantage

需要写代码的文件：`Human/7_grpo_alignment/cs336_alignment/alignment.py` 和 `Human/7_grpo_alignment/tests/adapters.py`。

GRPO 的默认配置是：

```text
advantage = (reward - group_mean) / (group_std + eps)
```

#### 知识点：组内 baseline 降低的是相对难度差异

同一 prompt 的多个 response 构成一个 group。减去组均值后，advantage 表示回答相对同组表现，而不是绝对 reward；是否再除标准差决定不同组的梯度尺度是否被归一。不同 GRPO 变体改变的是 baseline/scale 规则，但都必须保持原 batch 顺序。

#### 大概实现逻辑

先根据 group size 把一维 rewards 还原成“prompt × samples”视图，在组维计算所需均值和标准差，再按所选变体变换并展平回原顺序。显式处理组内全相同、标准差为零和不能整分组的输入；同时返回 raw/normalized reward 等 metadata，便于检查尺度是否合理。

这里的 std 在测试里使用 unbiased std。其他变体包括：

- Dr. GRPO：减去 group mean，但不除 std。
- RFT：不减 baseline，也不 normalize。
- MaxRL 风格：减 group mean 后除以 group mean。

测试信号：`test_compute_group_normalized_rewards_*` 会分别覆盖这些变体。

怎样测试：

```sh
cd Human/7_grpo_alignment
uv run pytest tests/test_grpo.py -k compute_group_normalized_rewards -q
```

### 6. Task 6：实现 policy-gradient loss

需要写代码的文件：`Human/7_grpo_alignment/cs336_alignment/alignment.py` 和 `Human/7_grpo_alignment/tests/adapters.py`。

on-policy 情况最直接：

```text
loss_token = - advantage * log_prob_token
```

#### 知识点：importance ratio 修正新旧策略分布差异

on-policy 数据由当前策略生成，不需要分布修正；off-policy 更新时，新策略对同一 token 的概率已经变化，概率比率衡量这种偏移。clipping 限制单次更新对目标的影响，GRPO 在 token 粒度限制，GSPO 则先把整段 response 的 log-ratio 汇成 sequence 粒度，再用于该序列各 token。

#### 大概实现逻辑

先让 advantage 广播到 token 维，并只在有效 response mask 上解释 loss。需要旧策略时，在 log 空间相减后再指数化；根据 loss type 选择 token ratio 或 masked sequence ratio，并构造 unclipped/clipped 两个 surrogate。按目标定义选择保守一侧，返回逐 token loss，让后续 aggregation 统一处理 mask 和归一化。

off-policy 情况需要 importance ratio：

```text
ratio = exp(new_log_prob - old_log_prob)
```

`noclip` 直接乘 ratio；`grpo` 做 token-level clipping；`gspo` 先在 response token 上平均 log-ratio，得到 sequence-level ratio，再 clipping 并广播回 token 维度。

常见错误：

- 忘记 `old_log_probs` 和当前 logprobs 对齐。
- GSPO 平均时把 prompt token 也算进去。
- clipping 对正负 advantage 的方向处理错。用 surrogate 的 `minimum` 更不容易写反。

怎样测试：

```sh
cd Human/7_grpo_alignment
uv run pytest tests/test_grpo.py -k compute_policy_gradient_loss -q
```

### 7. Task 7：实现 loss aggregation

需要写代码的文件：`Human/7_grpo_alignment/cs336_alignment/alignment.py` 和 `Human/7_grpo_alignment/tests/adapters.py`。

测试要求两种 normalization：

- `sequence`：每条 response 内部先按 mask 平均，再对 batch 平均。
- `constant`：所有 masked token loss 求和后除以固定常数。

#### 知识点：归一化定义决定长短 response 的权重

sequence normalization 让每条回答先贡献一个平均 loss，因此长回答不会仅因 token 多就占更大权重；constant normalization 保留 token loss 的总和比例，只用固定尺度控制梯度。两者数值都可能看起来合理，但表达的是不同训练目标，不能混用分母。

#### 大概实现逻辑

先用 response mask 把 prompt/padding loss 清零。sequence 模式为每条样本计算有效 token 数和 masked 平均，再对 batch 聚合，并处理零有效 token；constant 模式直接汇总所有有效 token 后除指定常数。用不同 response 长度的构造样例检查两种模式产生预期的相对权重。

这一步看似小，但会影响 GRPO、Dr. GRPO、RFT 等变体的梯度尺度。

怎样测试：

```sh
cd Human/7_grpo_alignment
uv run pytest tests/test_grpo.py -k aggregate_loss_across_microbatch -q
```

### 8. Task 8：实现 GRPO train step

需要写代码的文件：`Human/7_grpo_alignment/cs336_alignment/alignment.py` 和 `Human/7_grpo_alignment/tests/adapters.py`。

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

#### 知识点：microbatching 必须保持完整 batch 的梯度语义

切 microbatch 是为降低峰值显存，不应改变一次 optimizer update 代表的目标函数。各 microbatch 的 loss 缩放取决于 aggregation 定义：局部平均需要按累计步数或全局样本数校正，固定分母的局部和则应直接累加。只有所有 microbatch backward 完成后才能裁剪并更新。

#### 大概实现逻辑

先在不切 batch 的路径上组合前面已验证的纯函数，得到参考 loss 和参数更新。再按相同顺序切分所有对齐张量，每块执行 forward、policy loss、aggregation 与 backward，并根据 normalization 选择正确缩放。循环结束后统一 grad clipping、step 和清梯度；用固定随机种子比较切分与不切分后的梯度或参数 snapshot。

注意 `constant` normalization 和 `sequence` normalization 在 gradient accumulation 时处理不同。`sequence` 每个 microbatch 是局部平均，需要除以 `gradient_accumulation_steps`；`constant` 是固定全局常数，microbatch loss 应该相加。

测试信号：`test_grpo_train_step_*` 会比较更新后的所有模型参数。如果 loss 只差一点点，参数 snapshot 也会失败。

怎样测试：

```sh
cd Human/7_grpo_alignment
uv run pytest tests/test_grpo.py -k grpo_train_step -q
```

### 9. Task 9：完成 optional safety/RLHF

需要写代码的文件：`Human/7_grpo_alignment/cs336_alignment/alignment.py`、`Human/7_grpo_alignment/scripts/evaluate_safety.py` 和 `Human/7_grpo_alignment/tests/adapters.py`。

可选部分包括：

- `get_packed_sft_dataset`：把 instruction/response JSONL 包成固定长度 LM 样本。
- `run_iterate_batches`：用 DataLoader 产生 batch。
- `run_parse_mmlu_response`：从模型输出中解析 A/B/C/D。
- `run_parse_gsm8k_response`：取最后一个数字作为答案。
- `run_compute_per_instance_dpo_loss`：计算一个 preference pair 的 DPO loss。

#### 知识点：SFT、评估解析与 DPO 是不同的数据契约

SFT packing 把多条文本组织成固定长度的 next-token 样本；答案解析把自由文本转成可评分结果；DPO 比较 chosen/rejected 在策略与 reference 下的相对偏好。这三类功能共享 tokenizer/model，但 mask、截断和返回语义不同，不应塞进一个通用函数后靠分支猜测。

#### 大概实现逻辑

分别沿测试契约实现：packing 先 tokenize 每条样本并明确 EOS、截断与跨样本边界，再产生固定长度块；batch iterator 只负责确定顺序与张量化；解析函数对格式噪声设置清晰 fallback；DPO 先得到四组序列 log-prob，再按 chosen 相对 rejected 的策略变化构造逐样本 loss。每条路径先用最小手工例子验证边界。

这些不是 README 指定的最小 `test_grpo.py`，但本地目录有测试，所以一起完成更稳。

怎样测试：

```sh
cd Human/7_grpo_alignment
uv run pytest tests/test_data.py tests/test_metrics.py tests/test_dpo.py -q
uv run pytest
```

### 10. 全面验收：确认整个 Assignment 5 已完成

先在锁定环境中完成 adapter、全套单元测试和静态检查：

```sh
cd Human/7_grpo_alignment
uv sync
rg -n 'raise NotImplementedError' tests/adapters.py
uv run pytest -q
uv run python -m compileall -q cs336_alignment tests scripts
```

adapter 不应再有占位实现。完整 pytest 要同时覆盖 tokenization、response logprob、reward、group normalization、policy-gradient loss、microbatch aggregation、GRPO train step，以及本地存在的 data/metrics/DPO 测试。任何参数 snapshot、mask、shape 或 normalization 失败都不能用近似结果忽略。

随后在小模型和极小数据集上做真实 train-step smoke test：固定随机种子，确认 loss/reward/advantage/logprob 都是有限值，只有 response token 参与 loss，梯度累积与不切 microbatch 的结果一致，optimizer step 后目标参数确实变化且没有意外更新 reference model。保存运行配置和关键数值，确保结果可复现。

若提交包含 GPU rollout、GRPO 训练或 safety evaluation，还必须在目标 GPU 环境运行真实模型，确认生成、reward 计算、训练、checkpoint 恢复和评估脚本端到端可用，并检查输出 JSONL 完整、无损坏记录。只有 CPU 测试、GPU smoke test、训练/评估产物和 written deliverables 全部完成，才算 Assignment 5 完成。

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
