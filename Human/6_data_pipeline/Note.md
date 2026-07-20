# Assignment 4 Data Note

## 第一章：从全局视角理解本次作业

Assignment4 的主题是数据。前几个作业主要关注 tokenizer、Transformer、训练系统和 scaling law；这次作业换了一个角度：如果训练数据本身很差，模型再大、训练再久，也只是在更高成本地学习噪声、重复内容、隐私信息和有害文本。

这份作业的目标是搭建一个语言模型训练前的数据处理管线。PDF 的主线可以概括为：

```text
Common Crawl / HTML / WET
  -> 提取正文
  -> 语言识别
  -> PII 脱敏
  -> 有害内容过滤
  -> 质量规则过滤
  -> 质量分类
  -> 精确去重和近似去重
  -> tokenization
  -> 用固定训练代码训练 LM
```

本目录里，`cs336_basics` 是作业提供的语言模型训练实现，原则上不应该改。你真正要写的是 `cs336_data`：它负责把原始网页文本处理成更适合训练的语料。`scripts/train.py` 后面会消费你生成的 GPT-2-tokenized `.bin` 文件，但训练代码不是这次测试的重点。

### 1.1 为什么数据过滤重要

预训练数据通常来自网页，网页数据会有几类典型问题：

- HTML 里混有导航栏、脚本、样式和链接结构，不能直接作为自然语言文本。
- Common Crawl 覆盖多种语言，如果目标是英文 LM，需要过滤非英文文本。
- 网页可能包含邮箱、手机号、IP 等个人信息，需要脱敏。
- 数据里可能有 NSFW、辱骂、仇恨或其他有害内容，需要识别和过滤。
- 很多网页是论坛模板、登录页、菜单页、重复版权声明，质量不适合训练。
- 大规模抓取会包含大量重复内容，重复样本会让模型过拟合局部模式，也会浪费训练预算。

所以 Assignment4 的学习重点不是“写一个复杂模型”，而是理解数据质量如何决定训练效果。

### 1.2 本次实现涉及的关键文件

- `cs336_data/processing.py`：正文抽取、语言识别、PII masking、内容分类、Gopher 质量规则。
- `cs336_data/deduplication.py`：精确行去重和基于 n-gram Jaccard 的文档级近似去重。
- `cs336_data/wet_files.py`：下载 WET 文件时复用 `is_english` 做英文过滤。
- `cs336_data/modal_utils.py`：Modal 运行配置。当前从 `CS336_SUNET_ID` 环境变量读取用户 ID，默认 `local` 以保证本地导入可用。
- `tests/adapters.py`：测试适配层，把测试调用转发到 `cs336_data` 的实现。
- `tests/test_*.py`：定义每个数据处理步骤的预期行为。

### 1.3 需要先理解的概念

HTML to text conversion：HTML 是网页结构，不等于正文。正文抽取工具会丢掉标签、脚本和样式，保留可读文本。

Language identification：语言识别器输入一段文本，输出语言标签和置信度。本实现优先使用本地 fastText 模型；模型不存在时，用简单规则兜底，保证离线测试可运行。

PII：Personally Identifiable Information，指能识别个人的信息。本作业测试邮箱、手机号和 IP 地址。

Gopher quality rules：一组启发式质量规则，例如文本不能太短或太长，平均词长不能异常，太多行不能以省略号结尾，大多数词应该包含字母。

Deduplication：去重。精确去重处理完全相同的行或文档；近似去重处理“几乎一样”的文本，例如不同项目里的 MIT License。

MinHash / LSH：大规模近似去重常用技术。MinHash 用短签名近似 Jaccard 相似度，LSH 用分桶快速找候选重复项。本地测试规模很小，所以实现用确定性的 n-gram Jaccard 判断重复，接口仍保留 `num_hashes`、`num_bands`、`ngrams` 和 `jaccard_threshold`。

## 第二章：如何完成 Human 路径

这一章只说明 Human 路径要在哪些文件写代码、每一步完成什么行为、怎样测试。数据作业应该按数据管线顺序完成，不要一开始就碰训练脚本。训练只有在数据清洗稳定后才有意义。

### 1. Task 1：接通测试适配层

需要写代码的文件：`Human/6_data_pipeline/tests/adapters.py`，以及你在 `Human/6_data_pipeline/cs336_data/` 下创建或补齐的实现模块。

测试不会直接猜你的函数名，而是调用 `tests/adapters.py` 中的 `run_*` 函数。因此第一步是决定实现模块的位置，然后让 adapter 做薄转发。

#### 知识点：数据函数需要纯净、稳定的测试边界

数据管线往往依赖模型文件、网络和批量文件系统，但单元测试需要用小 fixture 验证确定行为。adapter 把外部测试契约与实现模块连接起来，使正文抽取、分类、mask 和去重都能作为独立函数测试，而不必启动整条下载训练流程。

#### 大概实现逻辑

先按 `run_*` 契约把能力分到 processing 与 deduplication，明确每个函数的输入类型、返回结构和可选依赖。adapter 只转发参数并做必要的轻量类型适配；模型加载、正则和算法留在实现模块。先让失败进入目标函数，再逐项实现，避免一个 adapter 同时承担多阶段管线。

推荐结构是：

```text
tests/adapters.py
  -> cs336_data.processing
  -> cs336_data.deduplication
```

检查信号：如果 adapter 还在 `raise NotImplementedError`，所有测试都会直接失败；如果 adapter 做了太多逻辑，后面维护会变乱。

怎样测试：

```sh
cd Human/6_data_pipeline
uv run pytest tests/test_extract.py -q
```

这条测试可能仍会因为正文抽取未实现而失败，但失败点应该进入你的实现，而不是停在 adapter 的 `NotImplementedError`。

### 2. Task 2：HTML 正文抽取

需要写代码的文件：`Human/6_data_pipeline/cs336_data/processing.py` 和 `Human/6_data_pipeline/tests/adapters.py`。

先用 `tests/fixtures/moby.html` 和 `moby_extracted.txt` 做逐字对比。不要手写一堆正则解析 HTML；这类任务应该使用 HTML parser 或正文抽取库。

#### 知识点：HTML DOM 与可见正文不是同一个东西

HTML 同时包含结构、脚本、样式、导航和正文。解析器负责把不规则标记恢复成树，正文抽取规则再决定哪些节点可见、哪些换行和列表结构要保留。直接用正则删除标签会丢失嵌套关系，也很难稳定处理实体、损坏标记和编码。

#### 大概实现逻辑

先把 bytes 用明确的容错策略解码，再交给成熟 parser/正文抽取入口。用 fixture 逐段比较空白、列表与标点，调整库选项而不是为单个页面硬编码替换。对空输入、损坏 HTML 和非 UTF-8 内容设定可预测行为，确保单个坏页面不会终止整个数据批次。

常见错误：

- 输入是 `bytes`，抽取库可能需要 `str`。
- 解码错误没有处理，导致网页里少数字符让整个函数失败。
- 开启 main-content 过滤后，可能误删测试期望保留的列表项。

检查信号：`test_extract_text_from_html_bytes` 要求输出和 fixture 完全相同。

怎样测试：

```sh
cd Human/6_data_pipeline
uv run pytest tests/test_extract.py -q
```

### 3. Task 3：语言识别

需要写代码的文件：`Human/6_data_pipeline/cs336_data/processing.py`、`Human/6_data_pipeline/cs336_data/wet_files.py` 和 `Human/6_data_pipeline/tests/adapters.py`。

作业下载脚本会下载 fastText 的 `lid.176.bin`。真实数据处理时应该优先用这个模型，并用概率阈值决定是否保留英文。

#### 知识点：语言识别输出包含标签与置信度

语言分类器给出的是模型判断和分数，不是绝对事实。管线通常同时关心预测语言与置信度，并在阈值附近接受一定误差。离线 fallback 只是保证有限 fixture 可测试，不能假装与覆盖 176 种语言的真实模型等价。

#### 大概实现逻辑

把模型加载与单条预测分开，避免每篇文档重复读取权重。统一清理模型标签前缀并把 score 转成稳定 float；模型不存在时进入明确、范围有限的规则分支。调用方根据目标语言和阈值决定保留，不把筛选策略塞进预测函数。分别测试真实模型路径与 fallback 路径。

离线测试环境可能没有模型文件，所以需要有可解释的 fallback：中文字符明显存在时返回 `zh`，否则对英文 fixture 返回 `en`。

检查信号：

- Moby-Dick fixture 应该识别为 `en`。
- `"欢迎来到我们的网站"` 应该识别为 `zh`。
- score 要是正的 `float`。

怎样测试：

```sh
cd Human/6_data_pipeline
uv run pytest tests/test_langid.py -q
```

### 4. Task 4：PII masking

需要写代码的文件：`Human/6_data_pipeline/cs336_data/processing.py` 和 `Human/6_data_pipeline/tests/adapters.py`。

PII masking 的重点是“替换并计数”。每个函数都应该返回：

#### 知识点：PII masking 要区分识别、替换和审计

mask 的目标是不泄露原值，同时保留“这里曾有某类信息”的结构信号。计数用于审计过滤强度，因此必须统计本次识别到的原始 PII，而不是结果文本里出现了多少占位符。模式还要考虑词边界和句末标点，避免过度替换普通数字。

#### 大概实现逻辑

为每类 PII 单独定义匹配边界和标准占位符，先用回调或匹配结果完成替换并累计实际命中。已有占位符在扫描前应被保护或天然不匹配。准备正例、相邻标点、多种电话格式、无效 IP 和重复运行样例，确认第二次运行不会把旧 mask 计成新发现。

```text
(masked_text, num_masked)
```

测试覆盖三类：

- email -> `|||EMAIL_ADDRESS|||`
- phone number -> `|||PHONE_NUMBER|||`
- IP address -> `|||IP_ADDRESS|||`

常见错误：

- 已经存在的 mask 字符串不能被算作新 mask。
- 电话格式有多种：纯数字、括号、空格、短横线。
- IP 后面可能跟句号，正则不能因为句末标点漏匹配。

怎样测试：

```sh
cd Human/6_data_pipeline
uv run pytest tests/test_pii.py -q
```

### 5. Task 5：有害内容与质量分类

需要写代码的文件：`Human/6_data_pipeline/cs336_data/processing.py` 和 `Human/6_data_pipeline/tests/adapters.py`。

真实系统可以使用下载的 fastText 分类器。本地实现保留这个入口：模型文件存在时加载模型；模型不存在时使用关键词和结构规则兜底。

#### 知识点：不同分类目标需要不同决策边界

NSFW、toxic speech 和文本质量衡量的是不同属性，不能共用一个“坏文本”标签。模型分数通常需要按任务解释和设阈值；规则 fallback 只能覆盖少量高精度信号，并应允许返回不确定或低置信结果，而不是把所有未知文本判成同一类。

#### 大概实现逻辑

为三个任务建立独立入口、标签映射和阈值配置，共享的只应是模型加载等基础设施。优先调用可用分类器并规范化标签/score；fallback 使用少量可解释特征，避免宽泛关键词误伤。用能区分三种目标的对照样例验证：成人但不辱骂、辱骂但非成人、结构低质但内容安全。

Human 实现时要先区分两个目标：

- `classify_nsfw` 判断是否 NSFW。
- `classify_toxic_speech` 判断是否 toxic。

这两个任务不能完全混在一起。一个文本可能有成人内容但不辱骂，也可能有辱骂但不是 NSFW。

质量分类的测试只要求区分低质量 Common Crawl 风格文本和高质量 wiki/reference 风格文本。你可以先用明显特征建立基线，再替换成更强分类器。

怎样测试：

```sh
cd Human/6_data_pipeline
uv run pytest tests/test_toxicity.py tests/test_quality.py::test_classify_quality -q
```

### 6. Task 6：Gopher 质量规则

需要写代码的文件：`Human/6_data_pipeline/cs336_data/processing.py` 和 `Human/6_data_pipeline/tests/adapters.py`。

Gopher 规则适合按“先解析统计量，再依次判断”的方式写。不要把所有条件挤进一个大表达式，否则调试会很痛苦。

#### 知识点：启发式质量过滤是可解释的特征管线

Gopher 规则不是训练分类器，而是从词、行和字符统计中寻找明显低质量信号。关键在于分母、token 定义和边界条件一致：空行是否计入、符号是否算词、比例刚好等于阈值时是否通过，都会改变结果。

#### 大概实现逻辑

先把文本解析成后续规则共享的行、词和字符统计，集中处理空文本与零分母。每条规则独立比较一个指标并保留可调试的中间值，最后组合成统一判定。使用“一次只触发一条规则”的 fixture 检查边界，再用正常长文确认规则组合不会意外全拒绝。

建议先计算：

- non-symbol words 数量。
- 平均词长。
- 非空行中以 `...` 结尾的比例。
- 包含字母的词的比例。

检查信号来自 `tests/test_quality.py`：每个测试只打一个规则点，所以失败时通常能直接定位到对应条件。

怎样测试：

```sh
cd Human/6_data_pipeline
uv run pytest tests/test_quality.py -q
```

### 7. Task 7：去重

需要写代码的文件：`Human/6_data_pipeline/cs336_data/deduplication.py` 和 `Human/6_data_pipeline/tests/adapters.py`。

精确行去重的含义是：如果某一整行在整个输入集合中出现超过一次，就从所有文档里删除这行。测试中的导航行 `- home`、`- menu` 就是这种情况。

#### 知识点：精确去重与近似去重解决不同污染

精确行去重针对跨网页反复出现的模板行，需要先看完整语料的全局频次；文档近似去重针对内容大体相同但局部有差异的样本，用集合相似度表达重叠程度。两者都要规定“保留谁”和输出顺序，否则同一输入可能得到不稳定结果。

#### 大概实现逻辑

行去重采用两阶段流程：先统计规范化后整行的全局出现次数，再按原文档顺序删除所有重复行并写出结果。近似去重先为每篇文档生成稳定 n-gram 集合，再按输入顺序与已保留文档比较相似度，超过阈值时丢弃后出现者。实现时隔离空文档、短文档和空集合的相似度语义。

文档级近似去重的思路是：

1. 把文档转成小写 token。
2. 构造 word n-gram 集合。
3. 计算两个文档的 Jaccard 相似度。
4. 相似度超过阈值时保留先出现的文档，丢弃后出现的文档。

测试里 `rails_mit_license.txt` 和 `react_mit_license.txt` 是近似重复；`pytorch_license.txt` 应该保留。

怎样测试：

```sh
cd Human/6_data_pipeline
uv run pytest tests/test_deduplication.py -q
```

### 8. Task 8：真实数据和训练

需要使用或更新的文件：`Human/6_data_pipeline/scripts/download_data.py`、`Human/6_data_pipeline/scripts/train.py`、`Human/6_data_pipeline/configs/experiment/your_data.yaml` 和你的实验记录。

单元测试通过后，再考虑下载数据和训练：

#### 知识点：数据管线质量要靠逐阶段可观测性

真实数据经过抽取、语言过滤、PII、有害内容、质量规则和去重，每一步都会改变样本分布。最终训练 loss 无法告诉你是哪一步误删或漏删，因此每个阶段都要记录输入数、保留数、拒绝原因和少量人工样本，同时保证相同输入能得到相同输出。

#### 大概实现逻辑

先用小批 WARC/WET 数据串起所有纯函数，并为每阶段保存计数和可抽查输出。确认失败文档被隔离、PII 不进入日志、重复运行幂等后，再生成 tokenized `.bin` 与 metadata。训练前验证 dtype、token 数和配置路径；先做短 smoke run，只有数据统计和 loss 都合理时才扩大到远端全量任务。

```sh
uv run scripts/download_data.py --offline-only
```

如果要跑 Modal，需要先设置真实身份：

```sh
export CS336_SUNET_ID=<your_sunet_id>
```

然后才考虑全量下载、tokenization 和：

```sh
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```

不要在测试未通过时进入训练阶段；训练很贵，数据 bug 会被放大。

怎样测试：

```sh
cd Human/6_data_pipeline
uv run pytest
uv run scripts/download_data.py --offline-only
```

### 9. 全面验收：确认整个 Assignment 4 已完成

先验证所有 adapter 和纯函数行为，再做真实数据管线验收：

```sh
cd Human/6_data_pipeline
uv sync
rg -n 'raise NotImplementedError' tests/adapters.py
uv run pytest -q
uv run python -m compileall -q cs336_data tests scripts
uv run scripts/download_data.py --offline-only
```

adapter 不应再有占位实现；完整 pytest 必须一次性覆盖 HTML 抽取、langid、PII、toxicity、质量规则和去重。offline-only 下载应可重复运行，不能破坏已有文件，也不能依赖未声明的本地状态。

接着对一小批真实 WARC/WET 文档跑完整管线，逐阶段记录输入/保留/过滤数量，并人工抽查：HTML 正文没有明显模板污染，语言阈值符合要求，email/phone/IP 被 mask，有害和低质量样本被正确处理，精确与近似去重不会大量误删。输出中不能泄露抽查到的 PII，重复运行同一输入应得到一致结果。

最后才在真实数据上 tokenization 和训练，确认生成的 `.bin`/metadata 可被训练脚本读取、训练 loss 有限且总体下降，并保存数据统计、过滤比例、训练曲线和配置。Modal/full-data 步骤需要真实身份与远端资源；只通过单元测试和 offline-only 下载，不能算整个 Assignment 4 完成。

## 第三章：代码实现、逻辑与细节讲解

### 3.1 `processing.py` 的职责

`cs336_data/processing.py` 放所有“单条文本级”的处理：

- `extract_text_from_html_bytes`
- `identify_language`
- `is_english`
- `mask_emails`
- `mask_phone_numbers`
- `mask_ips`
- `classify_nsfw`
- `classify_toxic_speech`
- `classify_quality`
- `gopher_quality_filter`

这样设计的好处是：WET 下载、adapter 和未来的数据流水线都能复用同一套文本处理函数。

### 3.2 HTML 抽取

实现使用 `resiliparse.extract.html2text.extract_plain_text`。输入先用 UTF-8 解码，遇到坏字符时用 replacement 处理：

```text
bytes -> str -> extract_plain_text -> plain text
```

这样可以处理真实网页中常见的编码脏数据，不会因为单个坏字节让整篇文档失败。

### 3.3 fastText 优先，规则兜底

语言识别和有害内容分类都采用同一种结构：

```text
如果本地模型文件存在：
    fastText predict
否则：
    使用轻量规则返回稳定结果
```

模型路径来自 `get_shared_assets_path()` 下的 `classifiers/`。这和 `scripts/download_data.py` 下载离线文件的位置一致。

这种设计兼顾两点：

- 单元测试不依赖外部模型文件。
- 下载真实模型后，生产路径可以自动使用更强分类器。

### 3.4 PII 正则

三个 masking 函数都通过 `re.subn` 实现。`subn` 会同时返回替换后的文本和替换次数，正好对应测试接口。

IP 正则额外限制每段在 `0..255`，并允许 IP 后面紧跟句号这类标点。这是因为真实文本里 IP 常常出现在句尾：

```text
192.0.2.146.
```

这里要 mask 的是 `192.0.2.146`，不是最后的句号。

### 3.5 Gopher 质量过滤

`gopher_quality_filter` 的实现顺序是：

1. 用 word regex 找 non-symbol words。
2. 检查词数范围 `[50, 100000]`。
3. 检查平均词长 `[3, 10]`。
4. 检查以省略号结尾的非空行比例是否超过 `0.3`。
5. 检查至少 `80%` 的词包含英文字母。

每个规则都独立返回，调试时可以根据失败测试快速定位。

### 3.6 `deduplication.py` 的职责

`cs336_data/deduplication.py` 放“跨文档集合”的处理。它和 `processing.py` 分开，是因为去重必须同时看多个文件，而 PII、语言识别、质量过滤通常只看单条文本。

`exact_line_deduplication` 的流程：

```text
读入所有文件的所有行
统计每一行在全局出现次数
每个输出文件只保留出现次数为 1 的行
```

这会删除模板导航行，也会让两个完全相同的一行文档变成空文档。测试正是这样定义 expected output 的。

`minhash_deduplication` 的本地实现流程：

```text
document text
  -> lowercase tokens
  -> word n-gram set
  -> pairwise Jaccard
  -> similarity >= threshold 时删除后出现的文档
```

虽然函数名保留 MinHash，但在小型测试数据上直接计算 Jaccard 更清晰、更确定。真实大规模数据中可以把 pairwise Jaccard 替换成 MinHash + LSH candidate generation，再对候选对做精确 Jaccard 复核。

### 3.7 `wet_files.py` 如何接入

`cs336_data/wet_files.py` 原来有一个 `is_english` TODO。现在它从 `cs336_data.processing` 导入 `is_english`，在处理每条 WET conversion record 时判断是否保留。

这让离线测试里的语言识别函数和真实 WET 过滤路径使用同一个标准，不会出现“测试通过但下载数据时逻辑不同”的问题。

### 3.8 `modal_utils.py` 的本地默认值

原始文件要求直接填 `SUNET_ID`，否则导入时报错。这会阻断本地导入 `wet_files.py`。当前实现改为：

```text
SUNET_ID = os.getenv("CS336_SUNET_ID", "local")
```

这样本地测试和探索不需要真实 SUNET；真正跑 Modal 时，再通过环境变量设置真实值。

### 3.9 测试覆盖关系

测试和实现的对应关系：

- `test_extract.py`：验证 HTML bytes 到纯文本的精确输出。
- `test_langid.py`：验证英文和中文识别。
- `test_pii.py`：验证 email、phone、IP masking 和计数。
- `test_toxicity.py`：验证 NSFW/toxic 的基本分类行为。
- `test_quality.py`：验证 quality classifier 和 Gopher 质量规则。
- `test_deduplication.py`：验证精确行去重、完全重复文档去重、MIT license 近似重复去重。

当前本地验证结果：

```sh
uv run pytest
# 21 passed

uv run --with ruff ruff check cs336_data tests/adapters.py
# All checks passed

uv run --with ruff ruff format --check cs336_data tests/adapters.py
# 7 files already formatted
```

这说明测试要求的 Assignment4 本地实现已经闭环。剩下的真实 leaderboard 工作，是下载更大规模数据、选择过滤策略、tokenize 并用固定训练脚本训练，这一步需要你的真实 Modal/SUNET 配置和计算资源。
