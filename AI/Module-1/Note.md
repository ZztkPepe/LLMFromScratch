# Module 1 Note

## 第一章：从全局视角理解本次作业

Module 1 的核心目标是：让一个普通数字变成“会记住自己怎么被算出来，并且能自动求导”的数字。

在本模块里，这种数字叫 `Scalar`。它看起来像 Python 里的 `float`，可以加、减、乘、除、取 `log`、取 `sigmoid`，但它额外保存了一件普通数字不会保存的信息：这个值是由哪些上一步的值、通过哪个函数计算出来的。

这正是自动微分系统的开始。

### 1. 为什么训练神经网络需要自动微分

训练神经网络时，我们有一堆参数，比如权重和偏置。训练的目标是让损失变小。损失可以理解为“模型错得有多严重”。

如果某个参数稍微变大一点，损失变大了，那我们就应该把这个参数往小调。如果参数稍微变大一点，损失变小了，那我们就应该把这个参数往大调。

这里的“稍微变一点会造成多大影响”，就是梯度。

梯度不是神秘概念。它回答的是：

```text
这个变量往上动一点，最终损失会怎么动？
```

如果模型只有一个参数，我们也许可以手算导数。但真实模型有大量参数，而且每个参数都会经过很多层函数组合。手写每个参数的导数不仅麻烦，还非常容易出错。

所以深度学习框架要做一件事：让用户只写 forward，系统自动完成 backward。

Module 1 就是在实现这个机制的最小版本。

### 2. 计算图是什么

计算图是一张记录计算过程的图。

例如：

```text
x ----\
      multiply ---- z ---- sigmoid ---- y
w ----/
```

这张图表达了：

1. `x` 和 `w` 先相乘。
2. 乘法结果得到 `z`。
3. `z` 经过 sigmoid 得到 `y`。

图中的每个值都可以看成一个节点。每个操作，比如乘法和 sigmoid，也会留下记录：它的输入是谁，输出是谁，反向传播时该怎么把梯度传回去。

普通 Python 数字只保存结果，不保存过程。`Scalar` 保存结果，也保存过程。

### 3. forward 和 backward 分别在做什么

forward 是“从输入算到输出”。

如果你写：

```text
y = (x * w + b).sigmoid()
```

forward 会计算出 `y` 的数值。

backward 是“从输出反推每个输入的影响”。

如果最终损失是 `loss`，调用 `loss.backward()` 后，系统应该知道：

- `x` 对 `loss` 的影响是多少；
- `w` 对 `loss` 的影响是多少；
- `b` 对 `loss` 的影响是多少。

在训练中，真正需要更新的是参数，例如 `w` 和 `b`。它们的梯度会交给优化器，优化器再根据学习率更新参数。

### 4. 链式法则为什么是反向传播的数学基础

先讲直觉。

假设你从宿舍到教室要经过两段路：

```text
宿舍 -> 校门 -> 教室
```

如果宿舍到校门多走 1 分钟会让总时间多 1 分钟，校门到教室多走 1 分钟也会让总时间多 1 分钟，那么每段变化都会影响最终总时间。

函数组合也是这样。假设：

```text
z = f(x)
y = g(z)
```

`x` 影响 `z`，`z` 又影响 `y`。如果想知道 `x` 对 `y` 的影响，就要把两段影响乘起来：

```text
dy/dx = dy/dz * dz/dx
```

这就是链式法则。

反向传播就是在计算图上反复应用链式法则：从最终输出开始，沿着图往回走，把每一步的局部导数乘上从后面传来的梯度。

### 5. 为什么需要拓扑排序

计算图可能不是一条直线，而是有分叉和汇合。

例如：

```text
      -> f1 ---
x ---          + ---> y
      -> f2 ---
```

同一个变量 `x` 通过两条路径影响 `y`。反向传播时，`x` 的总梯度应该是两条路径贡献的和。

如果顺序处理错了，可能还没等所有路径的梯度都回来，就提前把 `x` 往前传播了。

拓扑排序就是为了解决“应该按什么顺序反向处理节点”。它保证：当我们处理某个节点时，它后面的影响已经被汇总过。

在 Module 1 中，`topological_sort` 负责给计算图排顺序，`backpropagate` 按这个顺序传播梯度。

### 6. `Value` / `Node` / `Scalar` 这类抽象为什么存在

不同课程或框架会用不同名字：

- 有的叫 `Value`。
- 有的叫 `Node`。
- MiniTorch 里 Module 1 叫 `Scalar`。

它们表达的是同一个思想：一个值不仅有数值，还要能回答下面几个问题：

- 我是谁？也就是唯一 id。
- 我是常量、叶子节点，还是由函数算出来的中间值？
- 我的父节点是谁？
- 我该如何把梯度传回父节点？
- 如果我是叶子节点，我该如何累积梯度？

这就是为什么不能只用 Python float。float 没有这些信息。

### 7. 本模块和深度学习框架的关系

PyTorch 中的 `torch.Tensor` 也会记录计算历史。你写 forward 时，它在背后构建计算图。你调用 `loss.backward()` 时，它沿图反向传播梯度。

Module 1 实现的是同样思想的标量版本。它没有 Tensor 的多维数组能力，但它完整展示了自动微分的基本机制：

- forward 构建计算图；
- backward 使用链式法则；
- 拓扑排序决定反向顺序；
- 梯度累积到叶子变量；
- 优化器用梯度更新参数。

### 8. 完成本模块前需要理解的知识

学生需要先掌握：

- 函数组合：一个函数的输出可以作为另一个函数的输入。
- 导数直觉：输出对输入变化的敏感度。
- 链式法则：组合函数的导数如何相乘。
- 图结构：节点、边、父节点、子节点。
- 递归或深度优先搜索：用于遍历计算图。
- Python 运算符重载：让 `Scalar` 支持 `+`、`*`、`-` 等语法。
- 类方法和静态方法：`ScalarFunction.apply`、`forward`、`backward` 都会用到。

## 第二章：如何完成 Human 路径

这一章只说明 Human 路径要在哪些文件里写什么，不直接给出实现代码。所有实际作业代码都应写在 `Human/Module-1` 下；`AI/Module-1` 只能作为完成后的阅读材料，不建议边看边抄。

### 1. 总体路线

开始前先处理前置条件：`Human/Module-1` 依赖 Module 0 的实现。如果 `Human/Module-1/minitorch/operators.py`、`Human/Module-1/minitorch/module.py`、`Human/Module-1/tests/test_operators.py`、`Human/Module-1/tests/test_module.py` 仍然是“Need to include this file from past assignment”，先把你自己在 `Human/Module-0` 完成的版本同步过来，再做 Module 1。

建议按下面顺序推进：

1. 在 `Human/Module-1/minitorch/autodiff.py` 完成 `central_difference`。
2. 在 `Human/Module-1/minitorch/scalar.py` 和 `Human/Module-1/minitorch/scalar_functions.py` 完成 Scalar 的前向运算。
3. 在 `Human/Module-1/minitorch/scalar.py` 完成 `chain_rule`。
4. 在 `Human/Module-1/minitorch/autodiff.py` 完成拓扑排序和整图反向传播。
5. 在 `Human/Module-1/minitorch/scalar_functions.py` 补齐各个 ScalarFunction 的 backward。
6. 在 `Human/Module-1/project/run_scalar.py` 完成三层 Scalar 网络和线性层 forward。

不要一开始就写训练。训练失败时很难判断问题在网络、优化器、forward，还是 backward。应该先让最小自动微分系统通过测试。

### 2. Task 1.1：中心差分

需要写代码的文件：`Human/Module-1/minitorch/autodiff.py`。

你要完成 `central_difference`，让它能对任意一个指定参数位置做数值导数近似。这个函数后面会被 `derivative_check` 用来检查自动微分结果。

写这部分时只关心数值检查工具本身，不要引入计算图、`ScalarHistory` 或 backward 逻辑。

验证位置：`Human/Module-1/tests/test_scalar.py` 中标记为 `task1_1` 的测试。

容易出错的地方：

- 只改变第一个参数，忽略 `arg` 指定的参数位置。
- 使用前向差分而不是中心差分，误差更大。
- 忘记除以 `2 * epsilon`。

怎样测试：

```sh
cd Human/Module-1
python -m pytest tests/test_scalar.py -m task1_1 -q
```

### 3. Task 1.2：Scalar 前向计算

需要写代码的文件：`Human/Module-1/minitorch/scalar.py`、`Human/Module-1/minitorch/scalar_functions.py`。

在 `scalar.py` 中，你要补齐 `Scalar` 的用户接口：

- 算术运算符：加、减、取负、除法相关组合。
- 比较运算符：小于、大于、相等。
- 方法调用：`log()`、`exp()`、`sigmoid()`、`relu()`。

在 `scalar_functions.py` 中，你要补齐各个 `ScalarFunction.forward`，让 `ScalarFunction.apply` 能创建带 history 的新 `Scalar`。这里不要绕过 `apply` 直接返回 float；否则 forward 数值看起来对，后面 backward 会没有计算历史。

验证位置：`Human/Module-1/tests/test_scalar.py` 中标记为 `task1_2` 的测试。

容易出错的地方：

- 混合 `Scalar` 和普通数字时没有统一转换。
- 运算结果丢失 history。
- `__gt__` 和 `__lt__` 的方向写反。

怎样测试：

```sh
cd Human/Module-1
python -m pytest tests/test_scalar.py -m task1_2 -q
```

### 4. Task 1.3：单节点链式法则

需要写代码的文件：`Human/Module-1/minitorch/scalar.py`。

你要完成 `Scalar.chain_rule`。它只负责一个节点：从当前节点的 `history` 找到最后一个函数、上下文和输入变量，调用该函数的 backward，再把每个非 constant 输入和对应梯度配对返回。

不要在 `chain_rule` 里遍历整张图，也不要在这里直接写入叶子变量的 `derivative`。整图遍历和梯度累积属于 Task 1.4。

检查重点：

- 输入顺序和梯度顺序必须一致。
- 常量输入不需要累积梯度。
- 返回的是一组 `(变量, 梯度)`。

验证位置：`Human/Module-1/tests/test_autodiff.py` 中标记为 `task1_3` 的测试。

怎样测试：

```sh
cd Human/Module-1
python -m pytest tests/test_autodiff.py -m task1_3 -q
```

### 5. Task 1.4：整张图的反向传播

需要写代码的文件：`Human/Module-1/minitorch/autodiff.py`、`Human/Module-1/minitorch/scalar_functions.py`。

在 `autodiff.py` 中，你要完成：

- `topological_sort`：从输出节点出发，得到适合反向传播的非 constant 节点顺序。
- `backpropagate`：沿拓扑顺序传递梯度，把叶子变量的梯度累积起来，把中间变量的梯度继续传给父节点。

在 `scalar_functions.py` 中，你要补齐 Task 1.4 标记的 backward 方法。每个 backward 只描述本函数的局部梯度规则，并返回和 forward 输入数量一致的梯度结果。

检查重点：

- 是否跳过常量节点。
- 是否按正确顺序遍历计算图。
- 同一个叶子变量的梯度是否累加，而不是覆盖。
- 非叶子节点是否继续向父节点传播。

验证位置：

- `Human/Module-1/tests/test_autodiff.py` 中标记为 `task1_4` 的反向传播结构测试。
- `Human/Module-1/tests/test_scalar.py` 中标记为 `task1_4` 的导数检查测试。

怎样测试：

```sh
cd Human/Module-1
python -m pytest tests/test_autodiff.py tests/test_scalar.py -m task1_4 -q
```

### 6. ScalarFunction 的 backward 工作边界

需要写代码的文件：`Human/Module-1/minitorch/scalar_functions.py`。

这一节单独强调边界：`ScalarFunction.backward` 不应该知道整张计算图，也不应该修改任何变量的 `derivative`。它只根据 forward 保存的上下文和传入的上游梯度，返回本函数各个输入位置应该收到的局部梯度。

需要处理的类包括 `Mul`、`Inv`、`Neg`、`Sigmoid`、`ReLU`、`Exp`、`LT`、`EQ`；`Add` 和 `Log` 已经给出示例，可以作为接口风格参考，但不要直接把别的函数硬套成同一种返回形状。

容易出错的地方：

- backward 返回数量和 forward 输入数量不一致。
- 忘记在 forward 中保存 backward 需要的值。
- 比较函数 `LT`、`EQ` 这类不可导或离散函数没有返回零梯度。

怎样测试：

```sh
cd Human/Module-1
python -m pytest tests/test_scalar.py -m task1_4 -q
```

### 7. Task 1.5：Scalar 训练

需要写代码的文件：`Human/Module-1/project/run_scalar.py`。

你要完成两个位置：

- `Network.__init__`：注册三层线性层，结构应和已有 `forward` 逻辑匹配。
- `Linear.forward`：用输入列表、权重参数和 bias 参数计算每个输出单元。

不要改 `ScalarTrain.train` 来掩盖网络实现问题。训练循环已经负责清梯度、计算 loss、调用 backward 和执行 optimizer；你要补的是模型结构和线性层计算。

验证方式：先通过前面所有 `task1_*` 测试，再运行 `Human/Module-1/project/run_scalar.py` 或项目 app 观察训练是否能降低 loss。

容易出错的地方：

- 参数没有注册，导致 `SGD` 找不到。
- 每轮训练前没有 `zero_grad`，导致梯度跨 epoch 累积。
- 最后一层没有 sigmoid，概率不在 0 到 1 之间。

怎样测试：

```sh
cd Human/Module-1
python project/run_scalar.py
```

## 第三章：代码实现、逻辑与细节讲解

这一章解释当前代码如何对应上面的概念。

### 1. 整体代码结构

核心文件如下：

- `minitorch/autodiff.py`：自动微分的通用算法，包括中心差分、拓扑排序、反向传播、`Context`。
- `minitorch/scalar.py`：`Scalar` 数据结构和运算符重载。
- `minitorch/scalar_functions.py`：每个标量函数的 forward 和 backward。
- `minitorch/operators.py`：从 Module 0 继承的基础数学函数。
- `project/run_scalar.py`：使用 Scalar 自动微分训练小型神经网络。
- `tests/test_scalar.py`：检查 Scalar 前向和导数。
- `tests/test_autodiff.py`：检查链式法则和反向传播。

可以把它们分成三层：

```text
数学函数层：operators.py
自动微分层：autodiff.py + scalar.py + scalar_functions.py
训练应用层：run_scalar.py
```

### 2. `Scalar`：带历史记录的数字

`Scalar` 保存：

- `data`：实际数值。
- `history`：这个值怎么来的。
- `derivative`：作为叶子节点时累计到的梯度。
- `unique_id`：区分计算图中的不同节点。
- `name`：调试时可读的名字。

`ScalarHistory` 保存：

- `last_fn`：最后产生这个值的函数。
- `ctx`：forward 时保存给 backward 用的信息。
- `inputs`：这个函数的输入。

如果一个 `Scalar` 是用户直接创建的，它是叶子节点。如果它是某个函数算出来的，它就有 `last_fn`。

### 3. `ScalarFunction.apply`：forward 时构建计算图

当你写：

```text
c = a * b
```

实际发生的是：

1. `Scalar.__mul__` 调用 `Mul.apply(a, b)`。
2. `apply` 把输入统一转换成 `Scalar`。
3. `Mul.forward` 计算数值结果。
4. `apply` 创建新的 `ScalarHistory`，记录 `Mul`、`ctx` 和输入。
5. 返回新的 `Scalar`。

所以 forward 不只是算结果，还在搭计算图。

### 4. `Context`：forward 给 backward 留便签

有些 backward 需要知道 forward 时的输入或输出。

例如乘法：

```text
y = a * b
```

反向时要知道 `a` 和 `b` 的值。因此 `Mul.forward` 会把它们保存到 `ctx`。

Sigmoid 的 backward 可以用 sigmoid 的输出值计算，所以 `Sigmoid.forward` 保存的是输出。

`Context` 就像 forward 留给 backward 的便签。如果忘记保存，backward 时就缺信息。

### 5. `chain_rule`：一个节点如何往回传

`Scalar.chain_rule(d_output)` 做的是局部反向传播。

它先找到当前节点的 `last_fn`，再调用这个函数的 `_backward`。然后它把得到的每个梯度和对应输入配对。

注意它会过滤常量输入。因为常量不是训练参数，也不需要累计梯度。

### 6. `topological_sort`：决定反向顺序

`topological_sort` 从最终输出开始做深度优先搜索。

它跳过常量，避免重复访问同一个节点，然后把节点放入列表。最终返回的顺序适合从输出往输入反向传播。

为什么这样重要？因为一个节点可能被多个下游节点使用。只有等下游贡献都汇总完，才能正确继续往前传。

### 7. `backpropagate`：整张图的 backward

`backpropagate(variable, deriv)` 使用一个字典保存每个节点当前累计到的梯度。

流程是：

1. 输出节点的初始梯度通常是 1。
2. 按拓扑顺序处理节点。
3. 如果是叶子节点，就把梯度累积到 `derivative`。
4. 如果不是叶子节点，就调用 `chain_rule`，把梯度分发给父节点。
5. 如果同一个父节点收到多次梯度，就相加。

这就是反向传播的核心。

### 8. `scalar_functions.py`：每个函数只负责自己的局部导数

`Add.backward` 返回两个 `d_output`，因为加法对两个输入的导数都是 1。

`Mul.backward` 返回和另一个输入相关的梯度。

`Inv.backward` 使用倒数函数的导数。

`Neg.backward` 把梯度取反。

`Sigmoid.backward` 使用 sigmoid 的局部导数。

`ReLU.backward` 只在输入大于 0 时传递梯度。

这些 backward 函数都不需要知道整张图。它们只需要解决当前这一小步。整张图的组合由 `backpropagate` 完成。

### 9. `run_scalar.py`：把自动微分用于训练

`Network` 是三层结构：

```text
Linear(2, hidden)
ReLU
Linear(hidden, hidden)
ReLU
Linear(hidden, 1)
Sigmoid
```

`Linear.forward` 对每个输出单元计算：

```text
bias + 输入和权重的加权和
```

训练循环中：

- `optim.zero_grad()` 清空旧梯度。
- forward 得到预测。
- 根据标签计算概率和 loss。
- 对 loss 调用 backward。
- `optim.step()` 更新所有参数。

这就是一个最小深度学习训练循环。

### 10. 测试在验证什么

`test_scalar.py` 验证：

- `central_difference` 是否近似正确。
- `Scalar` 前向结果是否和普通数学函数一致。
- 每个函数的梯度是否和数值导数接近。

`test_autodiff.py` 验证：

- `chain_rule` 是否能返回正确父节点梯度。
- 常量和变量是否被区分。
- 共享路径上的梯度是否能正确累加。

如果 `derivative_check` 失败，通常说明某个 backward 公式错了，或者计算图没有正确记录 history。

如果 `backprop` 测试失败，通常说明拓扑排序、梯度累加或常量过滤有问题。

### 11. 常见 bug 和排查方式

如果前向值错，先检查 `Scalar` 运算符是否调用了正确的 `ScalarFunction`。

如果前向值对但导数错，重点检查对应函数的 backward。

如果共享变量导数少了一倍，通常是梯度覆盖而不是累加。

如果训练 loss 不变，检查参数是否注册到 `Module`，以及每轮是否真正调用了 backward 和 optimizer step。

Module 1 的关键收获是：自动微分并不是魔法。它是 forward 记录历史，backward 按链式法则把局部导数组合起来。
