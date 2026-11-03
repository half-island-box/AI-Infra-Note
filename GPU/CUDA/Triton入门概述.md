>和CUDA从thread视角编写计算逻辑不同，
>Triton 让开发者以 tile（数据块）为单位描述加载、计算和存储操作，
>将大量线程与 warp 级别的实现细节交给编译器。



## 基础语法介绍
Triton kernel 中的**大部分操作并不是只处理一个标量，而是同时处理一个数据块。**
理解 Triton 代码时，可以按照下面的顺序分析：

- 使用 tl.program_id 确定当前 program **负责的数据区域**；
- 使用 tl.arange **生成一个数据块内部的元素偏移**；
- 通过指针运算得到一组内存地址；
- 使用 tl.load 将**数据加载为 Triton 张量**；
- 对张量执行逐元素运算、规约或矩阵乘法；
- 使用 tl.**store 将结果写回显存**。

调包通常写为`import triton.language as tl`

### tl.program_id（类别block内id）
`tl.program_id` 返回当前 program instance **在启动 grid 指定维度上的编号**，
作用类似于 CUDA 中的 blockIdx：

```
# Triton 的启动 grid 最多可以包含三个维度，因此 axis 的取值为 0、1、2
# 类似 cuda 中的 blockIdx.x / blockIdx.y / blockIdx.z 
pid = tl.program_id(axis=0)
```

### tl.num_programs（类比grid内id）
`tl.num_programs(axis)` 返回启动 grid 在指定维度上的 program 数量，作用类似于 CUDA 中的 gridDim：

```

# 类似地 axis 的取值为 0、1、2
# 对应 cuda 中的 gridDim.x / gridDim.y / gridDim.z
num_programs = tl.num_programs(axis=0)
```

### tl.arange
`tl.arange(start, end)` 用于创建连续的逻辑索引
```

offsets = tl.arange(0, BLOCK_SIZE)

```

其中 `BLOCK_SIZE` 通常是通过 `tl.constexpr` 传入的 2 的幂。


### tl.range
`tl.range` 用法类似于原生 python 中的 range 迭代器，常用于循环遍历中：

```

@triton.jit
def kernel(M, num_stages):
    for idx in tl.range(0, M, num_stages=num_stages) :
        ...

```

其中 `num_stages` 参数表示软件流水线级数，具体的使用案例可参考随后的 Triton Softmax 实现。

### tl.load
`tl.load` 根据一个指针或一组指针从显存中读取数据：

```

values = tl.load(ptrs)

```

如果 `ptrs` 是一个形状为 `[BLOCK_SIZE]` 的指针张量，那么返回的 values 也具有相同形状。  
对于可能越界的位置，可以使用 mask 屏蔽：

```

values = tl.load(
    ptrs,
    mask=offsets < n_elements,
    other=0.0,
)

```

对于 `mask=False` 的位置，代码不会访问对应的内存位置，而是使用 `other` 来填充返回，如果使用了 `mask`，通常应当显式指定 `other`。否则，被屏蔽位置的返回值是未定义的，可能污染后续计算

### tl.store
`tl.store` 将标量或 Triton 张量写入指定的内存位置，同样可以使用 mask 避免越界写入：

```

tl.store(output_ptrs, output, mask=mask)

```

### tl.where
`tl.where(condition, x, y)` 根据条件逐元素选择结果：

```

output = tl.where(condition, x, y)

```

它的语义类似于：`output[i] = x[i] if condition[i] else y[i]`，一个常见的例子是 `ReLU` 的实现：
```
output = tl.where(x > 0.0, x, 0.0)
```

### tl.dot
Triton 提供了大量类似 NumPy 和 PyTorch 的逐元素数学函数，例如：

```
y = tl.exp(x)
y = tl.log(x)
y = tl.sqrt(x)
y = tl.abs(x)
y = tl.maximum(x, 0.0)
y = tl.dot(x, y)
```

其中 `tl.dot` 用于计算两个二维或三维数据块的矩阵乘法，
在 GEMM kernel 中，`tl.dot` 用于沿 K 维分块逐步做乘法求和



## 通过Triton Softmax讲述算子实现
### 格式和传参
```python
import triton
import triton.language as tl
from triton.runtime import driver

# 声明这是一个 triton kernel
@triton.jit
def _softmax(
    input_ptr,                    # 输入矩阵的首指针
    output_ptr,                   # 输出矩阵的首指针
    input_row_stride,             # 输入矩阵中，相邻两行起点的间隔
    output_row_stride,            # 输出矩阵中，相邻两行起点的间隔
    n_rows,                       # 矩阵的行数
    n_cols,                       # 矩阵的列数
    BLOCK_SIZE: tl.constexpr,     # 每次循环的逻辑元素数，编译期常量
    num_stages: tl.constexpr      # 传给 tl.range 的软件流水线级数
) :
```

`BLOCK_SIZE`表示本例**每次循环处理的逻辑元素数量**
`tl.constexpr`表示该参数值在编译时已知


### 处理机制
假设启动了P 个 program，编号为 p 的 program 依次处理行 p,p+P,p+2P,…
直到行号达到 M，**每次循环只计算其中一行。**

对于每一行，先计算各列元素的指针，再通过 `tl.load` 加载数据、完成规约和逐元素运算，最后通过 `tl.store` 写回结果：

```python

    # 步长为 row_step；row_idx 表示当前 program 所处理的行数据索引
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages) :
        # 当前所处理的行数据的首指针
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # col_offset: [0, 1, 2, ..., BLOCK_SIZE-1]
        # 此处 BLOCK_SIZE 必须是编译期已知的 2 的幂
        col_offset = tl.arange(0, BLOCK_SIZE)

        # 构造 BLOCK_SIZE 个逻辑位置的指针，越界位置由下面的 mask 屏蔽
        input_ptrs = row_start_ptr + col_offset

        # 实际矩阵的列数不一定是2的幂数，需要掩码来控制读取的范围
        mask = col_offset < n_cols

        # mask=False 的位置不读取内存，填 -inf；行最大值有限时，这些位置的 exp 为 0
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # softmax 计算
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # 计算写入的行起始指针
        output_row_ptr = output_ptr + row_idx * output_row_stride
        # 计算每一个数据的具体指针偏移
        output_ptrs = output_row_ptr + col_offset
        tl.store(output_ptrs, softmax_output, mask=mask)
```

注意，这里的 `row` 是长度为 `BLOCK_SIZE` 的一维 Triton 张量，因此 `tl.max(row, axis=0)` 和 `tl.sum(..., axis=0)` 都沿当前行的逻辑列位置规约，得到一个标量。

`tl.range(..., num_stages=num_stages)` 为编译器提供软件流水线级数的提示，尝试让不同循环迭代的加载与计算重叠，提升资源利用率。级数越高，共享内存和寄存器的压力越大。


### Triton Kernel的启动配置
与 CUDA 类似，Triton kernel 也需要指定启动 grid。下面沿用官方教程的思路：先预编译，查询资源占用，再估算要启动的 program 数量。program 在循环中继续处理后续行，这种写法通常称为 persistent kernel（可持久内核）。

```python
DEVICE = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(DEVICE)
NUM_SM = properties['multiprocessor_count']   # SM 数量
NUM_REGS = properties['max_num_regs']         # 寄存器容量预算，以 32 位寄存器计
SIZE_SMEM = properties['max_shared_mem']      # 共享内存容量预算，单位为字节
WARP_SIZE = properties['warpSize']            # NVIDIA GPU 上为 32
kernels = {}

def softmax(x):
    assert x.is_cuda

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    # 向上取到不小于 n_cols 的最小 2 的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8       # 一个 program 使用 8 个 warp，即 256 个 CUDA 线程
    num_stages = 4 if SIZE_SMEM > 200000 else 2       # 流水线的级数
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))

    if kernel is None:
        # warmup 只编译、不真正执行。返回一个编译好的 kernel 对象
        kernel = _softmax.warmup(
            y, x, 
            x.stride(0), y.stride(0), 
            n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, 
            num_stages=num_stages, num_warps=num_warps, 
            grid=(1,))

        # 初始化 CUDA kernel 句柄，以便查询编译产物的资源占用。
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # 启动 softmax kernel
    # 其中 x 和 y 会被隐式地转换成矩阵的首地址
    kernel[(num_programs, 1, 1)](
        x,
        y,
        x.stride(0),   # 矩阵同列相邻元素的地址差
        y.stride(0),
        n_rows,
        n_cols,
    )

    return y
```
