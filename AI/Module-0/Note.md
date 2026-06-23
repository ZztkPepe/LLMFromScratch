# Module 0 Note

## 第一章：从全局视角理解本次作业

Module 0 是 MiniTorch 的地基。它还没有真正进入“自动微分”或“反向传播”的实现，但它在为后面的 Module 做准备：后面所有 Scalar、Tensor、神经网络层、优化器，最后都会建立在本模块的基本数学函数、参数管理和测试习惯之上。

可以把一个深度学习框架想象成一座房子。Module 0 做的不是搭屋顶，而是在铺地基：

- 数学算子是砖块，例如加法、乘法、取反、sigmoid、ReLU。
- 测试是水平尺，用来检查砖块是不是直的。
- 高阶函数是简单工具，用来把“一个数上的操作”推广到“一组数上的操作”。
- `Module` 和 `Parameter` 是将来搭神经网络时的结构框架，用来组织层和参数。
- 数据集和手工分类器让学生第一次看到：这些小函数最后会服务于分类任务。

### 1. 本模块到底要做什么

本模块主要完成五件事：

1. 实现基础数学函数。
2. 为这些函数写性质测试。
3. 实现 `map`、`zipWith`、`reduce` 这类函数式工具。
4. 实现一个能保存子模块和参数的 `Module` 系统。
5. 用手工设置的线性分类器可视化一个简单数据集。

这五件事看起来分散，其实都在回答同一个问题：一个深度学习框架最小需要什么？

答案是：它至少需要能做数学计算，能验证数学计算，能组织参数，还能把这些参数放进一个模型结构里。

### 2. 它在整个课程中的位置

后续 Module 会逐步加入：

- Module 1：Scalar 自动微分。
- Module 2：Tensor 自动微分。
- Module 3：更快的 Tensor 运算。
- Module 4：更真实的深度学习模型。

Module 0 不是最终系统，但它定义了最基础的语言。例如 Module 1 中的 `ScalarFunction` 会调用这里的 `operators`；Module 2 的 Tensor 后端也会继续复用这些数学函数。

如果 Module 0 的 `sigmoid`、`relu`、`log_back` 这些函数写错，后面的自动微分即使框架结构正确，也会算出错误梯度。

### 3. 为什么深度学习框架需要这些基础件

一个神经网络本质上是一串函数组合。比如一个非常小的模型可以写成：

```text
输入 x
先做线性组合 z = w * x + b
再做非线性变换 y = sigmoid(z)
```

这里面需要加法、乘法、sigmoid。训练时还需要知道输出对参数 `w` 和 `b` 的变化方向，这就会涉及导数和反向传播。

Module 0 还没有实现反向传播，但已经实现了一些“将来反向传播会用到的局部导数函数”，例如：

- `log_back`
- `inv_back`
- `relu_back`

它们的作用可以先这样理解：当后面系统知道某个函数输出端的梯度时，这些函数负责把梯度往输入端传回去。

### 4. 计算图、forward、backward 和本模块的关系

本模块不会真的构建计算图，但学习时需要提前知道这个概念，因为后面 Module 1 和 Module 2 会直接使用它。

计算图可以理解成一张“计算流程图”。每个圆点是一个值，每条边表示这个值是如何由前面的值计算出来的。例如：

```text
x ----\
      multiply ---- z ---- sigmoid ---- y
w ----/
```

在深度学习框架里：

- forward 指从输入一路算到输出。
- backward 指从输出的误差一路反推每个参数应该如何改变。
- 梯度表示“如果这个值稍微变大一点，最终结果会怎么变”。

为什么不手写每个参数的导数？因为真实神经网络有成千上万个参数，手写不现实，也容易错。框架需要把每一步基础运算都写正确，然后让系统自动组合它们的导数。

Module 0 做的正是“把每一步基础运算写正确”。

### 5. `Module` 和 `Parameter` 为什么存在

一个神经网络通常不是一堆散落的变量，而是有层次结构的：

```text
Network
  layer1
    weight
    bias
  layer2
    weight
    bias
```

`Module` 就是这种树状结构的容器。`Parameter` 是需要训练的值。后面优化器会问模型：“请把你所有参数都交出来。” 如果没有 `Module.parameters()` 和 `Module.named_parameters()`，优化器就不知道该更新哪些值。

这就是 Module 0 中参数树的意义：它为后面的训练循环做准备。

### 6. 完成本模块前需要理解的知识

学生不需要提前懂完整深度学习框架，但需要先理解这些基础：

- Python 函数：输入、输出、返回值。
- 浮点数：计算机里的小数有精度误差，所以测试不能总用严格相等。
- 简单导数直觉：导数描述“输出对输入变化有多敏感”。
- 列表和循环：后面高阶函数会把循环抽象成函数。
- 类和对象：`Module`、`Parameter` 都是对象。
- 树结构：一个模块可以包含子模块，子模块还可以继续包含参数。

## 第二章：如何完成 Human 路径

这一章只给 Human 路径的施工图：你需要在哪些文件里写代码、每个文件要完成什么行为、用哪些测试确认方向正确。这里不会展开具体实现代码；真正写作业时请在 `Human/Module-0` 下完成，而不是照抄 `AI/Module-0` 的实现。

### 1. 总体顺序

建议按下面顺序推进，每一步都先让对应测试通过，再进入下一步：

1. 在 `Human/Module-0/minitorch/operators.py` 完成 Task 0.1 的基础算子。
2. 在 `Human/Module-0/tests/test_operators.py` 补齐 Task 0.2 的性质测试。
3. 回到 `Human/Module-0/minitorch/operators.py` 完成 Task 0.3 的高阶函数。
4. 在 `Human/Module-0/minitorch/module.py` 完成 Task 0.4 的参数树和模式切换。
5. 用 `Human/Module-0/project/run_manual.py` 观察手工分类器效果，并把需要提交的记录写进 `Human/Module-0/README.md`。

不要先写 `Module`，也不要先碰可视化。因为后面的任务依赖前面的基础函数和测试习惯。

### 2. Task 0.1：基础算子

需要写代码的文件：`Human/Module-0/minitorch/operators.py`。

你要在这个文件里完成基础标量函数，包括：

- 算术类函数：让加法、乘法、取反、取倒数等函数返回普通数值结果。
- 比较类函数：让小于、相等、最大值这类函数返回作业要求的数值信号。
- 非线性函数：实现 sigmoid、ReLU、log、exp 这类后续 Scalar/Tensor 会复用的函数。
- 局部反向函数：实现 log、inv、ReLU 对应的 backward helper，让后续自动微分模块能调用。

不要在这里写测试逻辑，也不要修改函数签名。测试会直接 import 这些函数，因此最小目标是：保留接口，只填函数体。

验证位置：`Human/Module-0/tests/test_operators.py` 中标记为 `task0_1` 的测试。

检查时重点看：

- 函数是否返回数字，而不是布尔值、字符串或列表。
- 极端输入下的 sigmoid 是否仍然稳定。
- backward helper 是否只做本函数局部梯度的事，不掺入整张计算图逻辑。

怎样测试：

```sh
cd Human/Module-0
python -m pytest tests/test_operators.py -m task0_1 -q
```

### 3. Task 0.2：性质测试

需要写代码的文件：`Human/Module-0/tests/test_operators.py`。

你要补齐 Task 0.2 中带 `NotImplementedError` 的 property tests。这里不是实现库函数，而是用 Hypothesis 描述函数应该长期满足的性质。

需要覆盖的测试意图包括：

- sigmoid 的输出范围和基本形状。
- 小于关系的传递性。
- 乘法的交换性。
- 加法和乘法之间的分配关系。
- 列表级别的求和、逐元素相加、高阶函数组合是否一致。

容易出错的地方：

- 把 property test 写成固定样例，失去 Hypothesis 的意义。
- 忽略浮点误差，用严格相等比较所有小数。
- 测试 sigmoid 单调性时不考虑浮点饱和。

验证位置：同一个文件中标记为 `task0_2` 的测试。目标不是让测试“随便通过”，而是让测试真的能抓到 Task 0.1 中常见的错误实现。

如果 Hypothesis 给出反例，先判断是性质写错、浮点容差不合理，还是 `operators.py` 的函数行为确实有问题。

怎样测试：

```sh
cd Human/Module-0
python -m pytest tests/test_operators.py -m task0_2 -q
```

### 4. Task 0.3：高阶函数

需要写代码的文件：`Human/Module-0/minitorch/operators.py`。

你要在这个文件里完成两层内容：

1. 通用高阶函数：`map`、`zipWith`、`reduce`。
2. 由高阶函数组合出来的列表工具：`negList`、`addLists`、`sum`、`prod`。

这里的重点是让通用函数承担遍历责任，具体算子只作为参数传进去。不要为每个列表工具单独写一套循环逻辑；那样虽然可能过一两个例子，但会绕开本任务真正要练的抽象。

容易出错的地方：

- `reduce` 忘记使用初始值。
- `zipWith` 没有按位置配对。
- `map` 返回的不是新列表，而是修改原列表导致副作用。

验证位置：`Human/Module-0/tests/test_operators.py` 中标记为 `task0_3` 的测试。

怎样测试：

```sh
cd Human/Module-0
python -m pytest tests/test_operators.py -m task0_3 -q
```

### 5. Task 0.4：Module 和 Parameter

需要写代码的文件：`Human/Module-0/minitorch/module.py`。

你要完成 `Module` 对参数和子模块的递归管理。具体工作包括：

- `train()`：把当前模块和所有子模块切到训练模式。
- `eval()`：把当前模块和所有子模块切到评估模式。
- `parameters()`：收集当前模块和所有后代模块中的 `Parameter`。
- `named_parameters()`：收集参数时保留层级路径，避免重名参数混在一起。

容易出错的地方：

- 只收集当前模块参数，忘记递归子模块。
- 子模块参数没有加前缀，导致名字冲突。
- `train()` 和 `eval()` 只改当前节点，不改子节点。
- 把普通属性误当参数，或者把参数没有放进 `_parameters`。

验证位置：`Human/Module-0/tests/test_module.py` 中标记为 `task0_4` 的测试。

调试时可以用 `Human/Module-0/project/module_interface.py` 的可视化理解模块树，但真正要改的核心文件仍然是 `minitorch/module.py`。

怎样测试：

```sh
cd Human/Module-0
python -m pytest tests/test_module.py -m task0_4 -q
```

### 6. Task 0.5：数据集和手工分类器

主要查看和记录的文件：`Human/Module-0/project/run_manual.py`、`Human/Module-0/minitorch/datasets.py`、`Human/Module-0/README.md`。

这一节通常不是继续补核心库的 `NotImplementedError`，而是把前面写好的函数和模块系统放到一个手工分类器里观察。你需要做的是：

- 阅读 `Human/Module-0/minitorch/datasets.py`，确认 Simple 数据集的标签规则。
- 运行或观察 `Human/Module-0/project/run_manual.py`，理解 `Network`、`Linear` 和手工参数如何形成一条分类边界。
- 在 `Human/Module-0/README.md` 记录你用于 Simple 数据集的参数或截图信息，确保结果可复现。

不要在 `datasets.py` 里改标签规则来迁就模型，也不要把可视化结果当成核心库实现的一部分。这个任务的重点是解释和记录：你用了哪些参数，为什么图像能对应数据规则。

验证位置：本节更多依赖人工检查和 README 记录；前四个 Task 的自动测试通过后，再做这一节才有意义。

怎样测试：

```sh
cd Human/Module-0
python -m pytest tests/test_operators.py tests/test_module.py -q
streamlit run project/app.py 0
```

## 第三章：代码实现、逻辑与细节讲解

这一章解释当前代码如何实现上面的目标。现在可以进入实现细节，因为前两章已经建立了为什么要这样做。

### 1. 整体代码结构

本模块核心文件如下：

- `minitorch/operators.py`：基础数学函数和高阶函数。
- `tests/test_operators.py`：算子和性质测试。
- `minitorch/module.py`：`Module` 和 `Parameter`。
- `tests/test_module.py`：参数树和模式切换测试。
- `minitorch/datasets.py`：二维分类数据集。
- `project/run_manual.py`：手工线性分类器。
- `README.md`：训练或可视化交付记录。

这套结构的重点是：实现文件负责行为，测试文件负责定义正确性，README 负责记录结果。

### 2. `operators.py`：深度学习框架的最小数学语言

`operators.py` 中的函数分三类。

第一类是普通数值函数：

- `mul`
- `add`
- `neg`
- `id`
- `max`
- `lt`
- `eq`
- `inv`
- `log`
- `exp`

它们让后续 Scalar 和 Tensor 可以复用同一套数学定义。

第二类是神经网络常用非线性：

- `sigmoid`
- `relu`

没有非线性，多层神经网络本质上还是一个线性函数，表达能力会很弱。sigmoid 常用于把输出压到 0 到 1 之间，ReLU 常用于隐藏层。

第三类是局部反向函数：

- `log_back`
- `inv_back`
- `relu_back`

它们在 Module 0 中看起来有点超前，但后面自动微分会用到。它们表示：如果输出端传来一个梯度 `d`，这个函数应该把多少梯度传给输入端。

### 3. 高阶函数：把标量操作推广到列表

`map(fn)` 返回一个新函数，这个新函数会把 `fn` 应用到列表每个元素。

`zipWith(fn)` 返回一个新函数，这个新函数会同时遍历两个列表。

`reduce(fn, start)` 返回一个新函数，这个新函数会从 `start` 开始累积。

因此：

- `negList` 是把 `neg` 提升到列表上。
- `addLists` 是把 `add` 提升到两个列表上。
- `sum` 是用 `add` 折叠列表。
- `prod` 是用 `mul` 折叠列表。

这个思想在 Module 2 会再次出现：Tensor 的 `map`、`zip`、`reduce` 本质上也是同样想法，只是数据从 Python 列表变成了多维 storage。

### 4. `module.py`：参数树

`Module` 的关键设计是把参数和子模块分别存起来：

- `_parameters` 保存当前模块直接拥有的参数。
- `_modules` 保存当前模块直接拥有的子模块。

当执行：

```text
self.p = Parameter(...)
self.layer = Module(...)
```

`__setattr__` 会根据类型决定放到哪个字典里。

`named_parameters()` 的逻辑是：

1. 先收集当前模块自己的参数。
2. 再遍历每个子模块。
3. 对子模块返回的参数名加上 `子模块名.` 前缀。

这就是为什么测试能找到：

```text
a.p2
b.c.p3
```

`train()` 和 `eval()` 则是树上的递归操作。它们先改变当前模块状态，再对子模块调用同名方法。

### 5. `run_manual.py`：手工分类器

当前手工模型是：

```text
output = sigmoid(5.0 - 10.0 * x0)
```

当 `x0 = 0.5` 时，括号里正好是 0，sigmoid 输出 0.5。这就是分类边界。

当 `x0 < 0.5` 时输出大于 0.5；当 `x0 > 0.5` 时输出小于 0.5。因此它能匹配 Simple 数据集的竖直分割规则。

### 6. 测试在验证什么

`tests/test_operators.py` 主要验证：

- 基础函数是否和 Python 行为一致。
- sigmoid、lt、mul、add 是否满足数学性质。
- 高阶函数是否能正确处理列表。
- 反向局部函数是否至少能运行并满足简单行为。

`tests/test_module.py` 主要验证：

- 参数能否被访问。
- 嵌套模块参数名是否正确。
- `parameters()` 数量是否正确。
- `train()` 和 `eval()` 是否递归传播。
- `__call__` 是否转发到 `forward()`。

### 7. 常见 bug 和排查方式

如果基础算子测试失败，先用普通 Python 表达式对照，不要立刻怀疑测试。

如果 property test 失败，先看 Hypothesis 给出的最小反例。它通常比随机猜测更有价值。

如果参数树测试失败，画出模块树，再逐层检查是否递归。

如果 README 图和模型行为不一致，直接代入边界点 `x0 = 0.5`，看模型输出是否接近 0.5。

Module 0 的最终目标不是写很多代码，而是建立一种习惯：先理解数学行为，再用测试固定行为，最后再把行为放进框架结构中。
