# Module 2 Note

## 第一章：从全局视角理解本次作业

Module 2 的目标是把 Module 1 中“一个数字的自动微分”推广到“一整块多维数据的自动微分”。

在 Module 1 中，我们用 `Scalar` 表示一个会记录计算历史的数字。这个设计很适合教学，但如果神经网络里有几千、几百万个数，用一个个 `Scalar` 来算会非常慢，也很难表达矩阵、批量样本、图像这些数据。

所以 Module 2 引入 `Tensor`。

Tensor 可以理解成“有形状的数字表”：

- 一个标量可以看成 shape 为 `()` 或 `(1,)` 的 Tensor。
- 一个向量可以看成 shape 为 `(n,)` 的 Tensor。
- 一个矩阵可以看成 shape 为 `(m, n)` 的 Tensor。
- 一批二维点可以看成 shape 为 `(batch, 2)` 的 Tensor。

深度学习框架真正处理的大部分数据都是 Tensor。

### 1. 为什么需要 Tensor

神经网络训练通常不是一次处理一个数字，而是一次处理一批样本。

例如一个二分类数据集里，每个点有两个坐标：

```text
(x0, x1)
```

如果一次处理 30 个点，就可以把输入放进一个二维 Tensor：

```text
shape = (30, 2)
```

第一维是样本数，第二维是每个样本的特征数。

Tensor 的意义是：让我们用统一的方式表达批量数据，并让加法、乘法、sum、relu、sigmoid 这些操作一次作用在很多元素上。

### 2. 本模块在整个项目中的角色

Module 1 已经证明：只要每个基础函数知道自己的 backward，系统就能自动求导。

Module 2 要解决的新问题是：当值不再是单个数字，而是多维数组时，如何仍然做到这一点？

这会引出几个核心机制：

- TensorData：如何存储多维数组。
- shape：Tensor 每个维度有多长。
- strides：如何从多维索引找到一维 storage 位置。
- broadcasting：不同形状的 Tensor 如何一起运算。
- map/zip/reduce：如何把标量函数推广到整个 Tensor。
- Tensor 自动微分：forward 如何构建 Tensor 计算图，backward 如何传播 Tensor 梯度。

这些机制是 PyTorch、NumPy、TensorFlow 等系统的基础。

### 3. Tensor 为什么需要 shape 和 strides

先看直觉。

一个矩阵：

```text
[[2, 3, 4],
 [4, 5, 7]]
```

看起来是二维的，但计算机内存更像一条长线。它可以存成：

```text
[2, 3, 4, 4, 5, 7]
```

那么问题来了：二维索引 `(1, 2)` 怎么找到一维位置？

这就需要 strides。

如果 shape 是 `(2, 3)`，普通连续布局的 strides 是 `(3, 1)`：

```text
position = row * 3 + col * 1
```

所以 `(1, 2)` 的位置是：

```text
1 * 3 + 2 * 1 = 5
```

strides 的重要性在于：有些操作不需要复制数据，只需要改变“怎么解释这段 storage”。例如转置矩阵时，可以交换 shape 和 strides，而不用真的搬动每个数。

### 4. Broadcasting 是什么，为什么需要

Broadcasting 可以理解成“自动扩展形状较小的 Tensor”。

例如：

```text
矩阵 shape = (30, 8)
bias shape = (8,)
```

我们想给矩阵每一行都加同一个 bias。直觉上，这是合理的：每个输出单元都有一个 bias，应该应用到所有样本。

Broadcasting 让系统自动把 `(8,)` 看成可以扩展到 `(30, 8)`。

注意它不是真的复制 30 份 bias，而是在索引时把需要扩展的维度映射回原来的第 0 个位置。

如果没有 broadcasting，写神经网络会很笨重：每次加 bias 都要手动复制成和 batch 一样大的矩阵。

### 5. forward 和 backward 在 Tensor 中有什么变化

概念上没有变化：

- forward 还是从输入算输出，并记录计算历史。
- backward 还是从输出梯度反向传回输入和参数。

变化在于：梯度本身也变成 Tensor。

例如：

```text
out = a * b
```

如果 `a` 和 `b` 都是 Tensor，输出也是 Tensor。反向传播时，传回 `a` 和 `b` 的梯度也应该是对应形状的 Tensor。

这带来一个新问题：如果 forward 中发生了 broadcasting，backward 时梯度形状可能比原输入更大。系统必须把梯度“还原”回原输入形状。

这就是 `Tensor.expand` 的作用。

### 6. 为什么仍然需要拓扑排序和链式法则

Module 2 继承了 Module 1 的自动微分思想。Tensor 只是把每个节点里的值从单个数字换成多维数组。

计算图仍然存在：

```text
X -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid -> Loss
```

链式法则仍然是反向传播的数学基础。区别只是每一步的局部导数不再只是一个数字，而是一批元素上的梯度传播。

拓扑排序仍然必要，因为 Tensor 计算图也可能有分叉、复用和汇合。系统必须按正确顺序把梯度从输出传回所有叶子参数。

### 7. 本模块和深度学习框架核心机制的关系

Module 2 对应真实框架中的这些机制：

- Tensor storage/layout：NumPy 和 PyTorch 都有类似的 storage、shape、stride 思想。
- Broadcasting：NumPy/PyTorch 中常见的形状自动扩展规则。
- Elementwise ops：逐元素加法、乘法、sigmoid、relu。
- Reduction ops：sum、mean 这类沿维度折叠的操作。
- Autograd Function：每个操作保存 forward 信息，并定义 backward。
- Batch training：一次处理多个样本，而不是一个样本一个样本循环。

如果 Module 1 是“理解自动微分的心脏”，Module 2 就是“让这颗心脏能泵动多维数据”。

### 8. 完成本模块前需要理解的知识

学生需要先理解：

- Module 1 的计算图和反向传播。
- 多维数组和矩阵的基本概念。
- shape：每个维度的长度。
- indexing：用多维坐标访问元素。
- storage：底层一维存储。
- strides：多维坐标如何映射到 storage 位置。
- broadcasting：形状不同但兼容的 Tensor 如何一起运算。
- reduce：沿某个维度把多个值合成一个值。

不需要一开始就懂高性能 GPU 编程。本模块实现的是简单 Python 版本，目标是理解机制。

## 第二章：如何完成 Human 路径

这一章只给 Human 路径的文件级路线图：在哪些文件写代码、每个文件承担什么任务、如何验证。它不会直接给出实现代码。真正写作业时请在 `Human/Module-2` 下完成；`AI/Module-2` 只适合作为做完后的对照阅读。

### 1. 总体顺序

开始前先处理前置条件：`Human/Module-2` 依赖 Module 0 和 Module 1 的实现。如果 `Human/Module-2/minitorch/operators.py`、`Human/Module-2/minitorch/module.py`、`Human/Module-2/minitorch/autodiff.py`、`Human/Module-2/minitorch/scalar.py`、`Human/Module-2/project/run_scalar.py` 仍然是“Need to include this file from past assignment”，先把你自己在前面 Human 模块完成的版本同步过来。

推荐顺序：

1. 在 `Human/Module-2/minitorch/tensor_data.py` 完成 TensorData 的索引、枚举和 permute。
2. 仍在 `Human/Module-2/minitorch/tensor_data.py` 完成 broadcasting 相关函数。
3. 在 `Human/Module-2/minitorch/tensor_ops.py` 完成低层 `tensor_map`、`tensor_zip`、`tensor_reduce`。
4. 在 `Human/Module-2/minitorch/tensor_functions.py` 完成 Tensor Function 的 forward。
5. 在 `Human/Module-2/minitorch/tensor_functions.py` 完成 Tensor Function 的 backward。
6. 在 `Human/Module-2/project/run_tensor.py` 完成 Tensor 训练网络。

这个顺序很重要。因为后面的所有 Tensor 运算都依赖前面的索引规则。

### 2. Task 2.1：TensorData indexing

需要写代码的文件：`Human/Module-2/minitorch/tensor_data.py`。

你要完成三个位置：

- `index_to_position`：把多维 index 和 strides 转成 storage 位置。
- `to_index`：把线性 ordinal 转成某个 shape 下的多维 index。
- `TensorData.permute`：返回共享同一 storage、但 shape/strides 维度顺序改变的新 `TensorData`。

不要复制 storage 来实现 permute；本任务要训练的是“同一段底层数据可以用不同 strides 解释”的思想。

验证位置：`Human/Module-2/tests/test_tensor_data.py` 中标记为 `task2_1` 的测试。

检查重点：

- `indices()` 是否枚举了所有位置。
- 枚举结果是否没有重复。
- index 是否越界时报错。
- permute 两次是否能回到原布局。

容易出错的地方：

- 把 shape 和 strides 混淆。
- permute 时复制了 storage，破坏共享布局的意义。
- `to_index` 的维度顺序写反。

怎样测试：

```sh
cd Human/Module-2
python -m pytest tests/test_tensor_data.py -m task2_1 -q
```

### 3. Task 2.2：Broadcasting

需要写代码的文件：`Human/Module-2/minitorch/tensor_data.py`。

你要完成两个位置：

- `shape_broadcast`：根据两个输入 shape 计算共同输出 shape，无法兼容时抛出 `IndexingError`。
- `broadcast_index`：把较大输出 shape 的 index 映射回某个较小输入 shape 的 index。

写这部分时先从右侧维度对齐的规则出发，不要真的复制 Tensor 数据。broadcasting 在这里是索引映射，不是数据扩容。

验证位置：`Human/Module-2/tests/test_tensor_data.py` 中标记为 `task2_2` 的测试。

检查重点：

- `shape_broadcast` 是否返回正确输出形状。
- 不兼容形状是否抛出 `IndexingError`。
- `broadcast_index` 是否把扩展维度映射到 0。

容易出错的地方：

- 从左往右对齐维度。
- 真的复制数据，而不是通过索引映射复用。
- 对少维 Tensor 没有加 offset。

怎样测试：

```sh
cd Human/Module-2
python -m pytest tests/test_tensor_data.py -m task2_2 -q
```

### 4. Task 2.3：Tensor map、zip、reduce

需要写代码的文件：`Human/Module-2/minitorch/tensor_ops.py`、`Human/Module-2/minitorch/tensor_functions.py`。

在 `tensor_ops.py` 中，你要完成低层 storage 版本的三个执行器：

- `tensor_map`：枚举输出位置，把输入 index 映射到对应 storage，再写入输出 storage。
- `tensor_zip`：同时处理两个输入 Tensor，支持 broadcasting 后的逐元素二元函数。
- `tensor_reduce`：沿指定维度折叠输入，把结果写到输出 storage。

在 `tensor_functions.py` 中，你要完成 Task 2.3 标记的 forward 方法，例如乘法、sigmoid、ReLU、log、exp、比较、近似相等和 permute。这些 forward 应该调用 backend 上已经封装好的 map/zip/reduce 能力，而不是重新手写一遍 storage 遍历。

验证位置：`Human/Module-2/tests/test_tensor.py` 中标记为 `task2_3` 的测试。

检查重点：

- contiguous Tensor 是否正确。
- permuted Tensor 是否正确。
- broadcasted Tensor 是否正确。
- reduce 后被折叠的维度是否变成 1。

容易出错的地方：

- 假设 storage 一定连续。
- 输出和输入 shape 不同时没有 broadcast。
- reduce 时没有从初始值开始累积。
- reduce 只处理二维情况，无法推广到更多维。

怎样测试：

```sh
cd Human/Module-2
python -m pytest tests/test_tensor.py -m task2_3 -q
```

### 5. Task 2.4：Tensor backward

需要写代码的文件：`Human/Module-2/minitorch/tensor_functions.py`。

你要补齐 Task 2.4 标记的 backward 方法。每个方法只负责本 Tensor Function 的局部反向规则：

- 需要 forward 保存输入或输出时，使用 `ctx.save_for_backward`。
- 返回值数量要和 forward 输入数量一致。
- 对比较类、索引类或形状参数这类不可导输入，返回零梯度或非 Tensor 占位。
- 对 permute 这类重排操作，backward 要把梯度维度恢复到输入顺序。

本任务通常不需要修改 `Human/Module-2/minitorch/tensor.py`。`Tensor.expand`、`Tensor.chain_rule` 等接口已经为 broadcasting 后的梯度还原提供入口；你要做的是让各个 Function 返回正确形态的局部梯度。

验证位置：`Human/Module-2/tests/test_tensor.py` 中标记为 `task2_4` 的测试。

检查重点：

- backward 返回数量是否和 forward 输入数量一致。
- 广播输入的梯度形状是否能还原。
- 比较函数是否不会向输入传递有效梯度。
- `grad_check` 是否通过。

容易出错的地方：

- 忘记保存 backward 需要的 forward 输入或输出。
- 用 Python 数字表达式时触发 Tensor 不支持的反向操作。
- permute 的 backward 没有使用逆排列。
- sum 的 backward 没考虑被 reduce 的维度。

怎样测试：

```sh
cd Human/Module-2
python -m pytest tests/test_tensor.py -m task2_4 -q
```

### 6. Task 2.5：Tensor 训练

需要写代码的文件：`Human/Module-2/project/run_tensor.py`。

你要完成两个位置：

- `Linear.forward`：用 Tensor 运算实现一层线性变换，输入是一批样本，输出是一批 hidden/output 表示。
- `Network.forward`：串起三层线性层，前两层接 ReLU，最后一层输出概率。

本模块还没有矩阵乘法，所以不要去改 `tensor_ops.py` 里的 `matrix_multiply`，也不要为了训练脚本临时引入外部矩阵库。线性层应当用本模块已有的 Tensor 操作表达，这样才能真正测试 broadcasting、sum、view、relu、sigmoid 是否协同工作。

验证方式：先通过 `task2_1` 到 `task2_4` 的测试，再运行 `Human/Module-2/project/run_tensor.py` 或项目 app 观察训练是否能正常降低 loss。

检查重点：

- 每层输出 shape 是否正确。
- 权重和 bias 是否是 `Parameter`。
- bias 是否正确 broadcast 到 batch。
- loss 是否能 backward。
- optimizer 是否能更新 Tensor 参数。

容易出错的地方：

- `sum(1)` 后忘记把 `(batch, 1, out)` reshape 成 `(batch, out)`。
- 参数 Tensor 没有注册，优化器找不到。
- 训练时没有清空旧梯度。

怎样测试：

```sh
cd Human/Module-2
python project/run_tensor.py
```

## 第三章：代码实现、逻辑与细节讲解

这一章解释当前实现如何完成 Module 2。

### 1. 整体代码结构

核心文件如下：

- `minitorch/tensor_data.py`：Tensor 的底层 storage、shape、strides、indexing、broadcasting。
- `minitorch/tensor_ops.py`：低层 Tensor map、zip、reduce。
- `minitorch/tensor_functions.py`：Tensor 版本的可微 Function。
- `minitorch/tensor.py`：用户直接操作的 Tensor 类。
- `minitorch/autodiff.py`：继承 Module 1 的拓扑排序和反向传播。
- `project/run_tensor.py`：Tensor 版三层神经网络训练。
- `tests/test_tensor_data.py`：验证布局、索引、permute、broadcast。
- `tests/test_tensor.py`：验证 Tensor 前向、梯度、view、reduce 和训练所需行为。

可以把架构看成四层：

```text
TensorData      管 storage 和索引
TensorOps       管逐元素和 reduce 遍历
TensorFunction  管 forward/backward 规则
Tensor          给用户提供 +、*、sum、view 等接口
```

### 2. `tensor_data.py`：多维数组的底层解释器

`TensorData` 保存：

- `_storage`：一维数据。
- `_shape`：每个维度大小。
- `_strides`：每个维度移动一步要跨过多少 storage 位置。
- `size`：元素总数。
- `dims`：维度数量。

`index_to_position(index, strides)` 把多维 index 转成 storage 位置。

`to_index(ordinal, shape, out_index)` 把线性编号转成多维 index。它用于枚举所有元素。

`permute` 创建新的 `TensorData`，但复用同一个 storage，只重排 shape 和 strides。

这就是为什么 permute 很轻量：它不搬数据，只改变解释方式。

### 3. Broadcasting 的实现

`shape_broadcast` 从右向左合并两个 shape：

- 如果两个维度相等，保留这个维度。
- 如果其中一个是 1，使用另一个维度。
- 如果缺失，按 1 处理。
- 否则抛出 `IndexingError`。

`broadcast_index` 则把输出的大 index 映射回输入的小 index。对于输入 shape 中等于 1 的维度，index 固定为 0。

这使得 bias 这类小 Tensor 能参与 batch 级运算。

### 4. `tensor_ops.py`：把标量函数推广到整块 Tensor

`tensor_map(fn)` 返回一个底层函数 `_map`。它遍历输出 storage，对每个输出位置：

1. 算出输出 index。
2. broadcast 到输入 index。
3. 找到输入和输出 storage 位置。
4. 写入 `fn(input_value)`。

`tensor_zip(fn)` 类似，但有两个输入，需要分别 broadcast。

`tensor_reduce(fn)` 沿某个维度折叠。输出 shape 和输入 shape 基本相同，只是 reduce 的维度变成 1。

这里的关键是：所有访问都通过 index 和 strides 完成，而不是假设 storage 顺序就是逻辑顺序。因此 permuted Tensor 也能正确运算。

### 5. `TensorBackend` 和 `SimpleOps`

`TensorBackend` 把具体函数组合成一组后端操作，例如：

- `neg_map`
- `sigmoid_map`
- `add_zip`
- `mul_zip`
- `add_reduce`
- `mul_reduce`

`SimpleOps` 是本模块的简单 Python 后端。后续模块可以替换成更快的后端，但上层 Tensor Function 不需要改。这就是分层设计的好处。

### 6. `tensor_functions.py`：Tensor 的可微操作

每个 Tensor Function 和 Module 1 的 Scalar Function 类似，都有 forward 和 backward。

例如 `Mul`：

- forward 使用 `mul_zip` 做逐元素乘法。
- backward 返回 `b * grad_output` 和 `a * grad_output`。

`Sigmoid`：

- forward 保存 sigmoid 输出。
- backward 使用 `out * (1 - out) * grad_output`。

当前代码中为了避免 `1.0 - Tensor` 这种未定义的反向减法写法，使用了等价形式：

```text
1.0 + (-out)
```

`Permute`：

- forward 调用 TensorData 的 permute。
- backward 使用逆排列把梯度维度换回去。

`LT`、`EQ`：

- forward 做比较。
- backward 返回零梯度，因为比较操作在这里不作为可训练连续函数处理。

### 7. `Tensor` 类如何协作

`Tensor` 是用户看到的对象。

当用户写：

```text
c = a * b
```

实际会调用：

```text
Mul.apply(a, b)
```

`Function.apply` 会：

1. 检查输入是否需要梯度。
2. detach 输入，执行 forward。
3. 如果需要梯度，保存 `History`。
4. 返回带 history 的新 Tensor。

这和 Module 1 的 Scalar 非常类似，只是数据变成了 TensorData。

### 8. backward、拓扑排序和梯度累积在哪里发生

`Tensor.backward()` 调用的是 `backpropagate`。这个函数来自 Module 1 的 `autodiff.py`。

Tensor 节点也实现了自动微分需要的接口：

- `is_leaf`
- `is_constant`
- `parents`
- `chain_rule`
- `accumulate_derivative`

因此同一个通用 `backpropagate` 可以服务 Scalar，也可以服务 Tensor。

Tensor 的 `chain_rule` 有一个额外工作：调用 `expand` 把梯度还原成输入需要的形状。

这一步对于 broadcasting 非常重要。例如一个 bias 被广播到整个 batch，反向时每个样本都会给 bias 一份梯度，这些梯度必须沿 batch 维度求和。

### 9. `run_tensor.py`：用 Tensor 训练网络

`Network` 结构是：

```text
Linear(2, hidden)
ReLU
Linear(hidden, hidden)
ReLU
Linear(hidden, 1)
Sigmoid
```

`Linear.forward` 没有使用矩阵乘法，而是用 broadcasting 实现：

```text
x.view(batch, in_size, 1)
weights.view(1, in_size, out_size)
逐元素相乘
沿 in_size 维度 sum
view(batch, out_size)
加 bias.view(1, out_size)
```

这样得到的输出 shape 是 `(batch, out_size)`。

训练循环和 Scalar 版本相同：

- 前向计算所有样本预测。
- 计算二分类概率和 loss。
- 对平均 loss 调用 backward。
- 用 SGD 更新参数。

Tensor 版本的优势是一次处理整个 batch，而不是在 Python 中逐个样本循环。

### 10. 测试在验证什么

`tests/test_tensor_data.py` 验证：

- strides 和 index 是否正确。
- indices 是否完整枚举。
- permute 是否保持 storage 对应关系。
- broadcast shape 是否符合规则。

`tests/test_tensor.py` 验证：

- Tensor 创建和索引。
- 一元函数和二元函数前向结果。
- 每个函数的梯度是否通过 `grad_check`。
- reduce 的前向和 backward。
- permute、view、contiguous 的行为。
- broadcast 场景下梯度形状是否正确。

如果 tensor 前向测试失败，优先看 `tensor_data.py` 和 `tensor_ops.py`。

如果前向通过但梯度失败，优先看 `tensor_functions.py` 的 backward 和 `Tensor.expand`。

### 11. 常见 bug 和排查方式

如果很多测试同时失败，并且错误来自 `index_to_position` 或 `to_index`，先修 TensorData，不要改上层函数。

如果只有 broadcast 测试失败，检查 shape 是否从右往左对齐。

如果 permute 后数值错，检查是否正确重排 strides，而不是只重排 shape。

如果 reduce 前向错，检查输出 shape 的 reduce 维是否变成 1，以及初始值是否正确。

如果 `grad_check` 失败但前向正确，检查 backward 是否保存了需要的 forward 值。

如果训练脚本 shape 报错，逐层打印 shape。Tensor 训练里的大多数错误都不是数学公式错，而是 shape 推导不一致。

Module 2 的关键收获是：Tensor 自动微分不是另一个全新的魔法系统。它仍然是 Module 1 的计算图和链式法则，只是每个节点里的值从单个数字变成了多维数组。
