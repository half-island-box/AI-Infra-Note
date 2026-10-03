1. **机制（Mechanics）：** 掌握 PyTorch 的基础操作（自底向上：张量 -> 构建模型 -> 优化器 -> 训练循环）。
2. **思维模式（Mindset）：** 养成**资源核算**的习惯。你需要清楚每一行代码消耗了多少内存（GB）和算力（FLOPs）。

## 一个简单直观的资源估算例子（时间估算）
> **问题：** 假设你是一个 AI 工程师，老板问你：“在 1024 张 H100 显卡上，训练一个 70B（700亿参数）的模型，数据量是 15T（15万亿 tokens），大概要多久？”

如果你直接跑去写代码测试，那可能几天甚至几个月都出不来结果。因此，我们需要学会“Napkin math”（餐巾纸计算，即**快速估算**）。

### 1. 计算总工作量
FLOPs（Floating Point Operations，浮点运算次数）是衡量计算复杂度的常用指标，表示一个算法或模型在执行过程中所需的浮点加减乘除等运算的总次数。而训练模型本质上是进行浮点运算（FLOPs）。有一个经验公式：

总计算量 = 6 * 参数量 * Token数量

> 为什么是 **6** 倍？
> C=A * B
> - **前向传播 (Forward Pass):** 计算一次约为  参数量（乘法+加法）。
> - **反向传播 (Backward Pass):** 计算梯度需要计算关于A,和关于B的。约为前向传播的 2 倍工作量，即  参数量。

代入数据：

![[Pasted image 20260906160622.png]]

### 2. 计算硬件算力
990FLOPS 是 H100 的理论峰值，但实际运行模型时，
由于各种软硬件开销，你几乎不可能达到 100% **模型算力利用率 (MFU, Model FLOPs Utilization)**，通常按 30%–60% 的利用率估算更现实。这里取 50% 用作后续估计。
![[Pasted image 20260906160928.png]]

### 3.计算结果：总工作量/硬件算力
![[Pasted image 20260906161631.png]]


## Pytorch概念和入门
### 张量
#### 基础概念
- 张量是深度学习中存储一切的基本构建块，包括：
	数据（data）
	模型参数（parameters）
	梯度（gradients）
	优化器状态（optimizer state）
	激活值（activations）

- 创建方式
	```python
	x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # 从Python列表创建
	x = torch.zeros(4, 8)  # 4x8的零矩阵
	x = torch.ones(4, 8)   # 4x8的全1矩阵
	x = torch.randn(4, 8)  # 4x8的正态分布随机数
	
	# 分配但不初始化值（用于自定义初始化的值）
	x = torch.empty(4, 8)  # 未初始化的4x8矩阵
	nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)  # 截断正态分布初始化，分布均值为0，标准差为1，截断范围，只保留[-2, 2]区间内的值，超出此范围的值会被重新采样
	```



- 张量的秩
	张量的秩就是维度数：
	```python
	x = torch.zeros(4)        # 秩 1 张量（向量），shape = (4,)
	x = torch.zeros(4, 8)     # 秩 2 张量（矩阵），shape = (4, 8)
	x = torch.zeros(4, 8, 2)  # 秩 3 张量，shape = (4, 8, 2)
	```
	
	在 Transformer 中，常见的张量是秩 4 的：
	
	```python
	B = 32   # Batch size（批次大小）
	S = 16   # Sequence length（序列长度）
	H = 16   # Number of heads（注意力头数）
	D = 64   # Hidden dimension per head（每个头的隐藏维度）
	x = torch.zeros(B, S, H, D)
	```

#### 张量操作
大多数张量都是**通过对其他张量执行操作**来创建的，每个操作都会占用一定的内存和计算资源。

##### 张量视图 view
许多操作只是提供张量的一个不同视图
**不会创建副本，因此一个张量的修改会影响相关视图张量**

- 非连续张量
	正常情况下对于连续张量，如果你按**内存地址从小到大**遍历张量的数据，
	得到的结果和按**行优先（row-major）顺序**读取张量元素的结果相同，那么这个张量就是连续的。

	一些view操作会改变原张量的shape、stride等，导致变成非连续，此时无法再次view
	（除非调用contiguous()方法创建一个新张量）


##### 逐元素操作
对每个元素独立应用函数，并返回相同形状的新张量。
这意味着，当你对一个张量执行 `x.pow(2)` 或 `x + x` 时，
**操作会独立地作用于张量中的每一个数字，而不会像矩阵乘法那样考虑元素之间的行列关系。**


##### 矩阵乘法
一个标准的矩阵乘法形式为：一个 M×K 的矩阵乘以一个 K×N 的矩阵，得到一个 M×N 的结果矩阵。示例如下：

```python
# 基本矩阵乘法
x = torch.ones(16, 32)  # 16x32矩阵
w = torch.ones(32, 2)   # 32x2权重矩阵
y = x @ w               # 结果：16x2矩阵
assert y.size() == torch.Size([16, 2])
```



- 批处理
在实际应用中，我们很少只处理单个数据样本。为了提高效率，我们会将多个样本打包成一个“批次”（batch），并对整个批次同时进行计算。下图展示了批量处理的操作：


![[Pasted image 20260906163437.png]]

- `batch` 标签指向堆叠起来的多个矩形，代表一个批次中包含的多个样本。
- `sequence` 标签指向每个矩形内部，代表每个样本可能包含的序列长度（例如，一个句子中的词元数量）。

下面案例中，我们对前两个维度（批次和序列）中的每个样本执行矩阵乘法

```python
x = torch.ones(4, 8, 16, 32) # 形状为 [批大小, 序列长度, 样本特征, 特征维度]
w = torch.ones(32, 2)        # 权重矩阵
y = x @ w                    # 矩阵乘法
assert y.size() == torch.Size([4, 8, 16, 2]) # 结果形状
```

在 PyTorch 进行上述运算时，会自动遍历 x 的前两个维度（即 4 和 8），对每一个 16×32 的子矩阵执行与 w 的矩阵乘法。最终结果 y 的形状是 (4, 8, 16, 2)，其中前两个维度保持不变，后两个维度根据矩阵乘法规则变化。


### 内存
张量的内存占用由两个因素决定：

- 元素数量：张量的形状（如4×8矩阵有32个元素）
- 数据类型：每个元素占用的字节数

#### 浮点类型（大多数张量都是浮点形式）
在 LLM 训练中，选择数据类型本质上是在**做显存占用、计算速度与数值稳定性之间的权衡**（Trade-off）。

##### Float32（FP32 / 单精度浮点数）
![[Pasted image 20260906163851.png | 1000]]

- **规格**：占用 **4 字节** (32 bits)。结构为：1位符号位 + **8位指数位** + 23位尾数位。
- **地位**：PyTorch 的默认数据类型，也是科学计算领域的“黄金标准”。
- **优点**：数值精度高，动态范围大，训练最稳定，几乎不会出现数值溢出问题。
- **缺点**：对于大模型而言太“奢侈”。它占用的显存是 16位格式的两倍，且在现代 GPU（如 H100）上的计算吞吐量远低于低精度格式。
- **炼丹用途**：通常用于存储**参数的主副本 (Master Weights)** 和 **优化器状态 (Optimizer States)**，以确保在梯度累积和参数更新时不会因为精度丢失而导致模型无法收敛。

##### Float16 （FP16 / 半精度浮点数）
![[Pasted image 20260906164014.png]]
- **规格**：占用 **2 字节** (16 bits)。结构为：1位符号位 + **5位指数位** + 10位尾数位。
- **优点**：显存占用比 FP32 减少一半，计算速度快。
- **致命缺陷**：**动态范围（Dynamic Range）太窄**。
    - 由于指数位只有 5 位，它无法表示非常小的数（会发生下溢 Underflow，直接变成 0）或非常大的数（会发生上溢 Overflow，变成 Infinity）。
    - 例如：在 FP16 中，`1e-8` 这样的小数会被直接当作 `0` 处理，导致梯度消失。
- **用途**：这是上一代 GPU（如 V100）混合精度训练的主流。为了解决溢出问题，必须使用复杂的**损失缩放 (Loss Scaling)** 技术。目前在 LLM 训练中正逐渐被 BF16 取代。

##### BFloat16（BF16 / 指数位扩大，精度降低）
![[Pasted image 20260906164159.png]]
- **规格**：占用 **2 字节** (16 bits)。结构为：1位符号位 + **8位指数位** + 7位尾数位。
- **来源**：由 Google Brain 专为深度学习设计。
- **设计逻辑**：**“要范围，不要精度”**。
    - 深度学习模型（尤其是神经网络）对小数点的后几位精度不敏感，但对数值的范围非常敏感。
    - BF16 直接截断了 FP32 的尾数，但**保留了和 FP32 相同的 8 位指数位**。
- **优点**：
    - 拥有和 FP32 一样宽广的动态范围，**不需要 Loss Scaling** 也能稳定训练。
    - 显存占用和 FP16 一样少。
    - 在 A100/H100 等新硬件上计算速度极快。
- **用途**：**当前 LLM 训练的绝对主流选择**。通常用于存储**激活值 (Activations)** 以及进行前向和反向传播的矩阵乘法计算。


#### 三种浮点类型简单对比

| 数据类型 | 总位数 | 符号位 | 指数位 | 尾数位 | 显存占用（相对 FP32） | 动态范围 | 数值精度 | 是否需要 Loss Scaling | 典型用途 |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-- |
| **FP32** | 32 | 1 | 8 | 23 | 100%（4 字节） | 宽 | 高 | 否 | 参数主副本、优化器状态、对精度要求高的计算 |
| **FP16** | 16 | 1 | 5 | 10 | 50%（2 字节） | 窄 | 中 | 是 | 旧混合精度训练、V100 等上一代 GPU |
| **BF16** | 16 | 1 | 8 | 7 | 50%（2 字节） | 宽（与 FP32 相同） | 较低 | 否 | 当前 LLM 训练主流：激活值、前向/反向矩阵乘法 |


#### 张量在内存的存储机制
Tensor 对象本身成为一个集成了“**指针+元数据**”的单一实体。
这意味着一个张量本身并不直接“包含”数据，而是指向一块连续的内存区域，并附带一套规则（元数据），告诉程序如何根据你请求的索引去这块内存里找到对应的数据。

早期是元数据+数据区分开两块内存，数据区公开存在隐患



## 计算效率
### 浮点运算总数（FLOPs）
**一个浮点运算（FLOP, Floating Point Operation）是一个基本的算术操作**，
比如加法 x + y 或乘法 x * y，理解 FLOPs 对性能分析至关重要。

#### 概念区分（工程量 VS 性能）
- **FLOPs**（小写s）：执行的浮点运算总数，这是一个衡量计算量的指标，
- **FLOP/s**（每秒浮点运算）：指硬件每秒能执行的浮点运算次数，也写作 **FLOPS**（大写S）。这是一个衡量“速度”的指标。例如，H100 GPU峰值性能为 990 TFLOP/s。

#### 常见运算的计算量
- 矩阵乘法
	![[Pasted image 20260906211310.png]]
	（2表一次矩阵运算的乘法和加法、token数代表处理的数据长度、模型参数量代表一次浮点运算涉及到的变更范围）

- 逐元素操作
	若张量形状为 m × n，则逐元素操作的 FLOPs 为 O(m × n)。

- 张量加法
	两个相同形状张量的对应元素相加，如 z = x + y。若 x 和 y 均为 m × n，则总 FLOPs = m × n


### MFU (Model FLOPs Utilization 衡量实际工作的效率)
MFU = (实际FLOP/s) / (硬件峰值FLOP/s)，即：
![[Pasted image 20260906211649.png]]
- MFU >= 0.5：被认为是相当不错的性能
- MFU 接近 1.0：非常难达到，因为硬件不可能100%满负荷运转，总会有内存访问、数据传输等开销

## 模型构建与训练基础
### 参数初始化
我们将构建一个简单的**深度线性网络（Deep Linear Network）**，并手动实现优化器。

### 模型参数的存储

在 PyTorch 中，**可训练的模型参数被封装为 nn.Parameter 对象：**

```python
w = nn.Parameter(torch.randn(input_dim, output_dim))
```

nn.Parameter 是 torch.Tensor 的子类，
因此它“表现得像一个张量”，可以进行所有张量操作。
它有一个 .data 属性，用于访问其底层的 torch.Tensor 数据。


### 参数初始化的重要性及常用方法

如果直接用 `torch.randn` 进行**标准高斯分布初始化**参数，
随着层数变深，数值会变得非常大（爆炸）或非常小（消失）。

```python
x = nn.Parameter(torch.randn(input_dim)) # 输入向量
output = x @ w # 输出向量
```

当 input_dim = 16384 时，输出值的大小约为 18.9，这是一个非常大的数值。这种大数值会逐层放大，导致梯度爆炸（gradient explosion），使训练过程变得极不稳定，甚至无法收敛。


为了克服这个问题，我们需要一种对输入维度 input_dim 不敏感的初始化方法。
**解决方法是使用 Xavier初始化([paper](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) [stackexchange社区](https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head))。通过除以 输入维度 来缩放权重，保持数值稳定性。**

```python
w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
```

经过缩放后，输出 output 的每个元素的值变得稳定在一个较小的范围内，不再随 input_dim 增长。

即使使用了 Xavier 初始化，由于正态分布的尾部是无界的，仍然存在产生极端值（outliers）的可能性。解决方案是使用**截断正态分布（truncated normal distribution）**，将生成的随机数限制在一个合理的范围内（如 [-3, 3]）。

```python
w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim), 
                                      std=1 / np.sqrt(input_dim), 
                                      a=-3, b=3))
```


### 数据加载
在语言建模中，**输入数据通常是经过分词器（tokenizer）处理后的整数序列。**
例如，句子 "Hello world" 可能被编码为 [1, 2]。
为了方便处理，这些整数序列通常会被保存为 NumPy 数组文件（.npy 格式）

```python
orig_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
orig_data.tofile("data.npy") # 将数组保存到文件
```

对于像 LLaMA 这样高达 2.8TB 的超大数据集，不可能将其全部加载到内存中。
**解决方案是使用 `numpy.memmap` 创建一个“内存映射”对象。**

```python
data = np.memmap("data.npy", dtype=np.int32) # 创建内存映射
assert np.array_equal(data, orig_data) # 验证加载的数据是否正确
```

data 对象本身不包含数据，而是一个指向磁盘上文件的“指针”。当你访问 data[0] 或 data[1:10] 时，系统才会按需将这部分数据从磁盘加载到内存中。这种方式极大地节省了内存占用，允许你在内存有限的情况下处理远超内存大小的数据集。

在实际训练中，我们不会直接操作整个数据集，而是通过一个“数据加载器”来生成训练所需的批次。
```python
B = 2  # 批次大小，即一次训练处理多少个样本
L = 4  # 序列长度，即每个样本包含多少个词元（token）
x = get_batch(data, batch_size=B, sequence_length=L, device=get_device())
assert x.size() == torch.Size([B, L]) # 验证输出张量的形状
```

调用 `get_batch` 函数会从 data 中随机采样 B 个起始位置，然后截取每个位置后长度为 L 的序列，最终返回一个形状为 (B, L) 的张量 x


### 优化器


#### 常用优化器介绍

- SGD (随机梯度下降)：最基础的优化器，直接用**学习率乘以梯度更新参数。**
- Momentum (动量法)：在 SGD 基础上增加了一个“动量”项，
	即**梯度的指数移动平均值，有助于加速收敛并减少震荡。**
- AdaGrad：根据历史梯度的平方值来调整每个参数的学习率，对稀疏特征更友好。
- RMSProp：对 AdaGrad 的改进，使用梯度平方的指数加权平均代替简单累加，避免学习率过早衰减。
- Adam：融合了 RMSProp 和 Momentum 的思想，是目前最流行的优化器。

我们以 AdaGrad 为例（虽然现在常用 AdamW，但原理类似）。
**优化器不仅要更新参数，还要记住每个参数的历史梯度信息（状态）**。
```python
class AdaGrad(torch.optim.Optimizer):
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                # 获取状态（梯度平方和）
                state = self.state[p]
                if 'sum_squared_grad' not in state:
                    state['sum_squared_grad'] = torch.zeros_like(p.data)
                
                # 更新状态：累加梯度的平方
                state['sum_squared_grad'] += grad ** 2
                
                # 更新参数：除以 根号下(状态)
                std = state['sum_squared_grad'].sqrt() + 1e-10
                p.data -= group['lr'] * grad / std
```

#### 优化器的使用

在 PyTorch 中实例化和使用一个 AdaGrad 优化器：

```python
# 实例化优化器
optimizer = AdaGrad(model.parameters(), lr=0.01) # model.parameters()：将模型中所有可学习的参数传递给优化器

# 计算梯度
loss.backward() # 计算损失函数对所有参数的梯度

# 执行一步更新
optimizer.step() # 根据梯度和优化器内部状态，更新模型参数
```

- 释放内存（可选）
	**在每次迭代开始前，需要清空上一次计算得到的梯度，否则梯度会累积**。
	优化器在调用 `zero_grad` 函数清空梯度时，设置 set_to_none=True 是一种更高效的内存管理方式，它会将梯度指针设为 None，而不是将其置零，可以节省内存。

```python
optimizer.zero_grad(set_to_none=True)
```

### 资源核算
#### 内存占用分析
对于一个深度线性模型，**总内存需求由四部分组成**：

- 参数 (Parameters)：模型中所有可学习权重的数量。
- 激活值 (Activations)：前向传播过程中产生的中间结果，需要保存下来用于反向传播。
- 梯度 (Gradients)：反向传播计算出的梯度，其数量与参数相同。
- 优化器状态 (Optimizer States)：优化器维护的额外状态信息（如 AdaGrad 的 g2），其数量也与参数相同。

假设所有数据都使用 float32 格式（每个元素占 4 字节），则总内存为：

```python
total_memory = 4 * (num_parameters + num_activations + num_gradients + num_optimizer_states)
```

#### 计算量（FLOPs）分析
们知道训练一次（前向+反向）的总 FLOPs 约等于 6 × (数据量) × (参数量)。因此，对于一个训练步骤，其计算量为：

```python
flops = 6 * B * num_parameters
```

### 训练循环：整合所有组件


#### 训练循环

```python
def train_loop():
    # 数据生成
    D = 16
    true_w = torch.arange(D, dtype=torch.float32, device=get_device()) # 创建一个真实的权重向量 [0, 1, 2, ..., 15]

    # 数据加载器，用于生成训练批次
    def get_batch(B: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(B, D).to(get_device()) # 从标准正态分布中随机采样 B 个 D 维的输入样本
        true_y = x @ true_w # 根据真实的权重 true_w 计算出对应的“真实标签”，即 y = x @ w_true
        return (x, true_y) # 作为模型的输入和目标输出

    # 执行训练
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.01)
    # 超参数调优（hyperparameter tuning）
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.1) # 通过改变学习率 lr 从 0.01 到 0.1，观察不同学习率对训练效果的影响
```

#### train()的内部实现

```python
def train(name: str, get_batch,
          D: int, num_layers: int,
          B: int, num_train_steps: int, lr: float):

    # 初始化模型和优化器
    model = Cruncher(dim=D, num_layers=0).to(get_device())
    optimizer = SGD(model.parameters(), lr=lr)

    # 主训练循环
    for t in range(num_train_steps): # 循环执行 num_train_steps 次，每次迭代称为一个“训练步”
        # 获取数据
        x, y = get_batch(B=B) # 获取当前批次的输入 x 和目标输出 y

        # 前向传播（计算损失）
        pred_y = model(x)
        loss = F.mse_loss(pred_y, y) # 计算预测值 pred_y 和真实值 y 之间的均方误差（Mean Squared Error），作为损失值 loss

        # 反向传播（计算梯度）
        loss.backward() # 调用 loss.backward()，触发自动微分机制，计算损失函数对所有模型参数的梯度，并将这些梯度存储在 .grad 属性中
        
        # 更新参数
        optimizer.step() # 根据计算出的梯度和优化器的更新规则（这里是 SGD），更新模型参数
        optimizer.zero_grad(set_to_none=True) # 清空上一步计算得到的梯度，为下一次迭代做准备
```

`train()` 函数参数说明：

- "simple": 实验名称。
- get_batch: 数据生成函数。
- D=16: 输入维度。
- num_layers=0: 模型层数为0，意味着这是一个单层线性模型。
- B=4: 批次大小。
- num_train_steps=10: 训练步数。
- lr=0.01: 学习率。

### 检查点CheckPoint
#### 为什么需要检查点？

- 训练时间长：训练一个大型语言模型可能需要数天、数周甚至数月。
- 系统不稳定：硬件故障、软件错误、电源中断、人为操作失误等都可能导致训练进程意外终止。
- 避免损失：**如果没有任何备份，一次崩溃就意味着之前所有的计算和时间都付诸东流。**

因此，定期保存“检查点”是保证项目顺利进行的基本保障。

#### 检查点包含哪些内容？

为了能够完整地恢复训练，检查点必须包含所有必要的状态信息。主要包括两个核心部分：

- **模型参数**：
	这是模型的核心，包含了所有可学习权重（nn.Parameter）的当前值。通过 model.state_dict() 可以获取一个字典，其中键是参数名称，值是对应的张量数据。
- **优化器状态** (optimizer.state_dict())：
	这是很多人容易忽略但极其关键的部分。优化器（如 Adam、AdaGrad）不仅存储了当前的学习率，还维护着一些内部状态变量。例如：
    - Adam：存储了动量（momentum）和方差（variance）的移动平均值。
    - AdaGrad：存储了历史梯度平方的累加和。

如果只保存了模型参数而没有保存优化器状态，那么在恢复训练时，优化器会从零开始，这会导致训练过程不连续，性能下降，甚至可能无法收敛。

#### 如何保存和加载检查点？

保存检查点

```python
# 1. 创建一个字典，包含模型和优化器的状态
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

# 2. 使用 torch.save 将字典序列化并保存到磁盘文件
torch.save(checkpoint, "model_checkpoint.pt") # 文件名通常使用 .pt 或 .pth 后缀
```

加载检查点

```python
# 1. 从磁盘文件加载保存的字典
loaded_checkpoint = torch.load("model_checkpoint.pt")

# 2. （后续步骤）将状态加载回模型和优化器
# model.load_state_dict(loaded_checkpoint["model"])
# optimizer.load_state_dict(loaded_checkpoint["optimizer"])
```

> 注意：上述代码只展示了加载步骤，完整的恢复流程还需要调用 load_state_dict() 方法将加载的数据应用到相应的对象上

### 混合精度计算
**fp32 能稳定训练但内存太大；fp16/bf16 省内存但有数值不稳定风险。**
如何在“高精度的稳定性”和“低精度的效率”之间取得平衡？ 混合精度计算

#### 主流方案
- 前向传播（Forward Pass）：使用 bfloat16 或 fp8。这包括所有中间激活值（activations）。因为激活值通常不需要极高的精度，使用低精度可以显著节省内存。
- 其余部分：使用 float32。这包括模型参数（parameters）、梯度（gradients）以及优化器状态（optimizer states）。这些是训练的核心，需要更高的精度来保证数值稳定性和收敛性。

> 核心思想：将低精度用于“消耗大但对精度要求不高”的部分（激活值），将高精度用于“对精度敏感”的部分（参数和梯度）。

## 算术强度

**计算不仅包含FLOPS浮点运算，还需要考虑数据从GPU Memory写入Compute Unit再读出的时间**

#### 定义
**算术强度（Arithmetic Intensity）**：

```
加速器强度 = FLOP/s / Bytes/s  → H100 约 295 FLOP/byte
算术强度 = FLOPs / Bytes        → 该操作每搬运 1 字节能做多少 FLOP
```

- 算术强度 < 加速器强度 → **Memory-bound**（瓶颈在数据传输 可以增大batch size）
- 算术强度 > 加速器强度 → **Compute-bound**（瓶颈在计算）

> "对于 H100，加速器强度大约是 295。这个数字值得记一下——对于 bf16 来说，每搬运 1 字节你需要做大约 300 次浮点运算才能摆脱内存瓶颈。"


### 训练 vs 推理的算术强度差异

- **训练时**：输入是整个序列，相当于矩阵乘法。处理一个序列的所有 token 的矩阵乘法有很高的算术强度，是 **compute-bound**。
- **推理时**：逐 token 生成，相当于**矩阵-向量乘**（一个 vector 和一个 matrix 做点积）。如前面分析，矩阵-向量乘是 **memory-bound**。

"这解释了为什么推理的 MFU 远低于训练：你不是在喂饱计算单元，而是在等待数据从 HBM 搬过来。"