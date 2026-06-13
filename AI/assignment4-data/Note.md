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

Human 路径应该按数据管线顺序完成，不要一开始就碰训练脚本。训练只有在数据清洗稳定后才有意义。

### 2.1 第一步：接通测试适配层

测试不会直接猜你的函数名，而是调用 `tests/adapters.py` 中的 `run_*` 函数。因此第一步是决定实现模块的位置，然后让 adapter 做薄转发。

推荐结构是：

```text
tests/adapters.py
  -> cs336_data.processing
  -> cs336_data.deduplication
```

检查信号：如果 adapter 还在 `raise NotImplementedError`，所有测试都会直接失败；如果 adapter 做了太多逻辑，后面维护会变乱。

### 2.2 第二步：HTML 正文抽取

先用 `tests/fixtures/moby.html` 和 `moby_extracted.txt` 做逐字对比。不要手写一堆正则解析 HTML；这类任务应该使用 HTML parser 或正文抽取库。

常见错误：

- 输入是 `bytes`，抽取库可能需要 `str`。
- 解码错误没有处理，导致网页里少数字符让整个函数失败。
- 开启 main-content 过滤后，可能误删测试期望保留的列表项。

检查信号：`test_extract_text_from_html_bytes` 要求输出和 fixture 完全相同。

### 2.3 第三步：语言识别

作业下载脚本会下载 fastText 的 `lid.176.bin`。真实数据处理时应该优先用这个模型，并用概率阈值决定是否保留英文。

离线测试环境可能没有模型文件，所以需要有可解释的 fallback：中文字符明显存在时返回 `zh`，否则对英文 fixture 返回 `en`。

检查信号：

- Moby-Dick fixture 应该识别为 `en`。
- `"欢迎来到我们的网站"` 应该识别为 `zh`。
- score 要是正的 `float`。

### 2.4 第四步：PII masking

PII masking 的重点是“替换并计数”。每个函数都应该返回：

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

### 2.5 第五步：有害内容与质量分类

真实系统可以使用下载的 fastText 分类器。本地实现保留这个入口：模型文件存在时加载模型；模型不存在时使用关键词和结构规则兜底。

Human 实现时要先区分两个目标：

- `classify_nsfw` 判断是否 NSFW。
- `classify_toxic_speech` 判断是否 toxic。

这两个任务不能完全混在一起。一个文本可能有成人内容但不辱骂，也可能有辱骂但不是 NSFW。

质量分类的测试只要求区分低质量 Common Crawl 风格文本和高质量 wiki/reference 风格文本。你可以先用明显特征建立基线，再替换成更强分类器。

### 2.6 第六步：Gopher 质量规则

Gopher 规则适合按“先解析统计量，再依次判断”的方式写。不要把所有条件挤进一个大表达式，否则调试会很痛苦。

建议先计算：

- non-symbol words 数量。
- 平均词长。
- 非空行中以 `...` 结尾的比例。
- 包含字母的词的比例。

检查信号来自 `tests/test_quality.py`：每个测试只打一个规则点，所以失败时通常能直接定位到对应条件。

### 2.7 第七步：去重

精确行去重的含义是：如果某一整行在整个输入集合中出现超过一次，就从所有文档里删除这行。测试中的导航行 `- home`、`- menu` 就是这种情况。

文档级近似去重的思路是：

1. 把文档转成小写 token。
2. 构造 word n-gram 集合。
3. 计算两个文档的 Jaccard 相似度。
4. 相似度超过阈值时保留先出现的文档，丢弃后出现的文档。

测试里 `rails_mit_license.txt` 和 `react_mit_license.txt` 是近似重复；`pytorch_license.txt` 应该保留。

### 2.8 第八步：真实数据和训练

单元测试通过后，再考虑下载数据和训练：

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
