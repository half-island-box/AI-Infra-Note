## Warp和SIMT执行模型
### 硬件视角的Warp
#### Warp形成规则
当一个 **Thread Block 被分配到 SM 上时**，（即软件层面的任务 分配给 硬件层面的处理单元）
硬件会自动将其中的线程按 `threadIdx` 顺序分组为 Warp：

|Block 中的线程 ID|所属 Warp|
|---|---|
|0 ~ 31|Warp 0|
|32 ~ 63|Warp 1|
|64 ~ 95|Warp 2|
|…|…|

 - **关键点**：
	 Warp 的划分是硬件行为，程序员无法干预。
	 一个包含 256 个线程的 Block 会被拆为 8 个 Warp。
	 **如果 Block 大小不是 32 的整数倍，最后一个 Warp 中多余的线程位置会被”填充”**
	 但不执行有效工作——这会浪费硬件资源。

#### Warp Scheduler
每个 SM 配备若干 Warp Scheduler（现代架构通常 4 个），
**每个Scheduler 在每个时钟周期 能从就绪的 Warp 中选择一个发射指令。
这带来一个关键的延迟隐藏机制：**

```
Warp A 发射内存读取指令 → 等待数据返回（~400 cycles）
Warp Scheduler 切换到 Warp B → 执行计算指令
切换到 Warp C → 执行计算指令
...
Warp A 数据就绪 → 重新调度执行（Warp A可能在前几个时钟周期时已经就绪）
```

💡 **提示**：Warp 之间的切换是零开销的（zero-overhead context switch），因为每个 Warp 的寄存器状态始终驻留在 SM 上，不需要保存/恢复。这与 CPU 的线程切换有本质区别。


### SIMT执行模型（Single Instruction Multi Threads 单指令多线程）
#### 特点
| 📊 对比维度 | SIMD（如 AVX-512）     | SIMT（CUDA Warp）  |
| ------- | ------------------- | ---------------- |
| 编程视角    | 程序员显式操作向量寄存器        | 每个线程拥有独立的程序计数器和栈 |
| 分支处理    | 使用掩码（mask）跳过特定 lane | **硬件自动序列化分支路径**  |
| 寻址灵活性   | 所有 lane 通常访问连续数据    | 每个线程可独立寻址        |
| 编程难度    | 需手动向量化              | 标量代码自动在 32 线程上并行 |

#### Warp的执行流水线
一个 Warp 的指令执行流程：
**取指 -> 译码 -> 发射-> 执行-> 写回**

关键特征：
- **同步执行**：一个 Warp 内的 32 个线程在同一时钟周期执行相同的指令
- **独立状态**：每个线程拥有自己的寄存器文件和程序计数器（逻辑上）
- **lockstep 执行**：**在没有分支时，所有线程步调一致**

### Warp的分支代价
**同个Warp的所有线程会并行执行相同代码**
如果代码分支条件设置不当（eg 用操作数值 或者 线程序号作为条件）
**会导致 if块 和 else 块实际上是串行执行** 执行时间翻倍

```cpp
__global__ void divergent_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid % 2 == 0) {
        // 偶数线程：路径A（10条指令）
        data[tid] = expf(data[tid]);
    } else {
        // 奇数线程：路径B（10条指令）
        data[tid] = logf(data[tid]);
    }
}

```

#### 规避策略
##### 分支条件按照Warp 粒度对齐
```cpp
// ❌ 不推荐：相邻线程走不同分支
if (threadIdx.x % 2 == 0) { ... }

// ✅ 推荐：同一 Warp 内的线程走相同分支
if (threadIdx.x / 32 % 2 == 0) { ... }
// 或按 warpId 分支
int warpId = threadIdx.x / 32;
if (warpId < 4) { ... }

```

##### 算术运算代替分支
```cpp
// ❌ 不推荐：条件分支
if (x > 0) {
    y = x;
} else {
    y = 0;
}

// ✅ 推荐：无分支实现 ReLU
y = x * (x > 0);  // 利用 bool → int 隐式转换
// 或使用内建函数
y = fmaxf(x, 0.0f);

```

##### 数据重排
**将需要走同一路径的数据提前重排汇聚到连续位置**
保证同个Warp内的线程执行相同路径
eg
```cpp
// 预处理：将正数和负数分别聚集
// 正数放 data[0..pos_count-1]
// 负数放 data[pos_count..N-1]
// 这样处理正数的 Warp 不会有分支分歧

```

### Warp Shuffle：线程间的数据交换
#### 概念
传统的线程间数据交换主要通过共享内存：
```
线程A写入共享内存 → __syncthreads() → 线程B读取共享内存
```
- 存在两个问题
	消耗内存资源
	需要同步屏障 会带来延迟

Warp Shuffle 让**同一 Warp 内的线程**可以**直接读取彼此的寄存器值**，
无需经过任何共享内存，
也不需要显式同步——就像同学之间直接传纸条，不用先放到讲台上再取。


#### 主要的Shuffle指令
所有 Shuffle 指令的签名形式：

```cpp
T __shfl_sync(unsigned mask, T var, int srcLane, int width=32);
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width=32);
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=32);
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=32);
```

| 📊 指令              | 功能                | 源 Lane 计算           |
| ------------------ | ----------------- | ------------------- |
| `__shfl_sync`      | 从指定 lane 读取       | `srcLane`           |
| `__shfl_up_sync`   | 从低编号 lane 读取      | `laneId - delta`    |
| `__shfl_down_sync` | 从高编号 lane 读取      | `laneId + delta`    |
| `__shfl_xor_sync`  | 与 XOR 配对的 lane 交换 | `laneId ^ laneMask` |

参数说明：

- `mask`：参与 Shuffle 的线程掩码，通常用 `0xFFFFFFFF` 表示所有 32 个线程参与
- `var`：当前线程要分享的值
- `width`：逻辑子 Warp 宽度（可以是 2、4、8、16、32）

eg
```cpp
初始状态：Lane[0]=A  Lane[1]=B  Lane[2]=C  Lane[3]=D  Lane[4]=E  Lane[5]=F  Lane[6]=G  Lane[7]=H

__shfl_down_sync(mask, var, 2):
  Lane[0] 得到 Lane[2] 的值 C
  Lane[1] 得到 Lane[3] 的值 D
  Lane[2] 得到 Lane[4] 的值 E
  ...

__shfl_xor_sync(mask, var, 1):  （相邻两两交换）
  Lane[0] ↔ Lane[1]
  Lane[2] ↔ Lane[3]
  Lane[4] ↔ Lane[5]
  Lane[6] ↔ Lane[7]

```

#### 常用示例
- Warp 内广播
```cpp
__device__ float warp_broadcast(float val, int srcLane) {
    // 所有线程获取 srcLane 线程的 val 值
    return __shfl_sync(0xFFFFFFFF, val, srcLane);
}

// 使用：让 Warp 内所有线程知道 lane 0 的计算结果
float result = compute_something();
float broadcast_val = warp_broadcast(result, 0);
```

- Warp内求和
```cpp
__device__ float warp_reduce_sum(float val) {
    // 蝶形归约：log2(32) = 5 步完成
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // 只有 lane 0 持有正确结果
}

```

### Warp级原语和进阶
#### Warp Vote函数
Warp Vote 函数**让整个 Warp 对一个布尔条件进行”投票”**：

```cpp
// 所有线程的 predicate 都为 true 时返回 true
int __all_sync(unsigned mask, int predicate);

// 任意线程的 predicate 为 true 时返回 true
int __any_sync(unsigned mask, int predicate);

// 返回一个 32-bit 位图，每个 bit 对应一个 lane 的 predicate 值
unsigned __ballot_sync(unsigned mask, int predicate);
```

实用场景——提前退出优化：

```cpp
__global__ void search_kernel(int* data, int target, int* found) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int pred = (data[tid] == target);

    // 如果 Warp 内所有线程都没找到，整个 Warp 可以跳过后续处理
    if (__any_sync(0xFFFFFFFF, pred)) {
        // 至少一个线程找到了目标
        if (pred) {
            atomicExch(found, tid);
        }
    }
}
```

#### 5.2 __activemask() 与 __syncwarp()
```cpp
// 获取当前活跃线程的掩码（Volta+ 才有意义）
unsigned mask = __activemask();

// Warp 内同步屏障（确保指定线程完成前序操作）
__syncwarp(0xFFFFFFFF);
```

💡 **提示**：`__syncwarp()` 在 Volta 之前的架构上通常是空操作（因为 Warp 天然是 lockstep 的），但从 Volta 开始由于独立线程调度，显式的 Warp 同步变得必要。

---

## 内存访问优化
Roofline 模型将 Kernel 分为两类：

- **Memory-Bound**：受带宽限制，优化方向是减少内存访问、提高访问效率
- **Compute-Bound**：受计算限制，优化方向是提高计算并行度

GPU可认为是巨大的工厂，车间里有几千台机器（计算核心）同时运转，但是原料只有一条搬运通道（内存总线）
**所以大多数 Kernel（特别是 Elementwise、归约、矩阵转置等）都是 Memory-Bound。**
对它们而言，合并访问和高效的内存使用模式才是性能关键。

### 全局内存和共享内存
全局内存是GPU的主存，存储所有必要数据
共享内存位于单个SM内，是单个Block对应的所有Warp的共享存储

二者的性能问题与优化：
[[入门——环境搭建和GPU模型#全局内存（显存 HBM）]]
[[入门——环境搭建和GPU模型#共享内存（单个SM内）]]

### 向量化加载（优化减少指令数）
向量化加载指的是用一条指令读取多个连续元素（如 `float4` 一次读 4 个 float，16 Bytes）。好处是：

1. **减少指令数**：4 次 `LDG.32` → 1 次 `LDG.128`
2. **提高带宽利用率**：单次事务传输更多有效数据
3. **减少调度开销**：更少的内存指令排队
#### 常用向量类型
|📊 类型|大小|等价|
|---|---|---|
|`float2`|8 Bytes|2 个 float|
|`float4`|16 Bytes|4 个 float|
|`int4`|16 Bytes|4 个 int|
|`double2`|16 Bytes|2 个 double|

- **提示**：
	GPU 一条 cache line 为 128 Bytes。
	一个 Warp 的 32 个线程每人加载一个 `float4`（16B）= 512B，需要请求 4 条 cache line。对比每人加载一个 `float`（4B）= 128B 只需 1 条 cache line。
	**虽然 cache line 请求数增多了，但每条加载指令处理的数据量是 4 倍，总指令数减少 75%。**


#### 基本用法
```cpp
// 标量版本：每个线程加载 1 个 float
__global__ void scalar_copy(float* dst, float* src, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        dst[tid] = src[tid];
    }
}

// 向量化版本：每个线程加载 4 个 float
__global__ void vectorized_copy(float* dst, float* src, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid * 4;  // 每个线程负责4个元素

    if (idx + 3 < N) {
        // 将 float* 强制转换为 float4*，一条指令读取 16B
        float4 val = reinterpret_cast<float4*>(src)[tid];
        reinterpret_cast<float4*>(dst)[tid] = val;
    }
}

```

- 注意对齐要求
	比如float4 要求16B对齐
	**如果头尾有偏移，需要先标量处理头部未对齐的部分，再向量化处理主体**

### 内存访问优化实战
#### 矩阵转置
- 朴素版本

```cpp
__global__ void transpose_naive(float* out, float* in, int width, int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < width && row < height) {
        out[col * height + row] = in[row * width + col];
        // 读取：合并（连续线程访问连续列）
        // 写入：不合并（stride = height）
    }
}
```

- 共享内存优化版本

```cpp
// 每个线程处理 TILE_DIM / BLOCK_ROWS 个元素
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_shared(float* out, float* in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 Padding 消除 Bank Conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 合并读取：每个 warp 读取连续元素
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }

    __syncthreads();

    // 合并写入：转置后的索引，连续线程写连续地址
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// 调用方式：
// dim3 block(TILE_DIM, BLOCK_ROWS);
// dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

```

核心思路：**用共享内存做”中转站”，将全局内存的非合并写入转化为共享内存的列读取（通过 Padding 消除 Bank Conflict），再以合并方式写出。**

#### 规约操作
```cpp
// ❌ 朴素归约：交错访问导致 Bank Conflict 和低带宽利用
__global__ void reduce_interleaved(float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    __shared__ float sdata[256];
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ✅ 优化归约：连续线程操作连续地址
__global__ void reduce_sequential(float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    __shared__ float sdata[256];
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```


## Occupancy（SM占有率）和资源分配
### Occupancy的概念
想象一个停车场有 64 个车位（SM 最大可驻留的 Warp 数），
如果当前只有 32 辆车停着，停车场利用率就是 50%。
**Occupancy 就是 GPU 中 SM 的”停车场利用率”——它告诉你硬件资源被用了多少。**

#### 重要性--延迟隐藏
**GPU的执行模型是依赖Warp的切换来掩盖内存访问延迟：**

```cpp
时间线：
Warp 0: [计算] [等待内存...400cycles...] [计算]
Warp 1:        [计算] [等待内存...400cycles...] [计算]
Warp 2:               [计算] [等待内存...400cycles...]
...

如果活跃 Warp 足够多，调度器总能找到就绪的 Warp 来填充等待期
```

- 延迟隐藏所需空闲Warp数量的粗略估算
	![[Pasted image 20261007155436.png]]
	由于一个SM最多拥有64个Warp，
	**所以实际上延迟是无法完全隐藏的**

### SM的资源清单
每个SM有一组固定的硬件资源，所有驻留在其上的Block共享这些资源

#### 资源和架构对比
| 📊 资源           | Ampere (A100) | Ada (RTX 4090) | Hopper (H100) |
| --------------- | ------------- | -------------- | ------------- |
| 每 SM 最大线程数      | 2048          | 1536           | 2048          |
| 每 SM 最大 Warp 数  | 64            | 48             | 64            |
| 每 SM 最大 Block 数 | 32            | 16             | 32            |
| 每 SM 寄存器总数      | 65536         | 65536          | 65536         |
| 每线程最大寄存器数       | 255           | 255            | 255           |
| 每 SM 共享内存上限     | 164 KB        | 100 KB         | 228 KB        |
| 每 Block 最大共享内存  | 163 KB        | 99 KB          | 227 KB        |
| 每 Block 最大线程数   | 1024          | 1024           | 1024          |
#### 资源分配层级
```cpp
Kernel（一次 kernel 启动，对应一个 grid）
 └── Grid
      └── Block（线程块，一批线程）
           └── Warp（32 线程，调度单位）
                └── Thread（单个线程）
```

![[Pasted image 20261007155844.png]]
1. 资源以Block为单元进行分配
	一个Block要么完整进入SM，要么完全不进入

2. 对Block分配的Warp资源是连续分配的

### 影响Occupancy的三大因素
#### 寄存器用量
- 每线程的寄存器数量是常见的限制因素之一：
	每线程寄存器数量 -> 每个Warp的寄存器数（32*）
	-> SM可以容纳的Warp数（65536/）-> Occupancy(/64)
	
eg

|每线程寄存器数|每 Warp 实际分配|SM 可容纳 Warp 数|Occupancy|
|---|---|---|---|
|32|32×32=1024|65536/1024=64|100%|
|48|48×32=1536|65536/1536=42|42/64≈65%|
|64|64×32=2048|65536/2048=32|50%|
|128|128×32=4096|65536/4096=16|25%|
|255|255×32=8160 → 对齐到 8192|65536/8192=8|12.5%|
#### 共享内存用量
- 每个Block声明的共享内存总量（静态+动态）会限制SM能容纳的Blcok数
	每个Block声明的共享内存量 -> SM能容纳的Block数量 
	-> SM能容纳的总线程数（总Warp数量） ->  Occupancy

eg
**示例**（A100，164KB 共享内存/SM）：

| 每 Block 共享内存 | SM 可容纳 Block 数         | 每 Block 256线程时总线程               | Occupancy |
| ------------ | ---------------------- | ------------------------------- | --------- |
| 8 KB         | 20（但受最大 Block 数 32 限制） | min(20,32)×256=5120→cap at 2048 | 100%      |
| 32 KB        | 5                      | 5×256=1280                      | 62.5%     |
| 64 KB        | 2                      | 2×256=512                       | 25%       |
| 164 KB       | 1                      | 1×256=256                       | 12.5%     |

#### Block的大小
Block 大小同时受两个限制：

1. 每 Block 最大 1024 个线程
2. 每 SM 最大 Block 数量（A100 为 32）
Block的大小 -> 总的Warp数量（SM的Block上限*BLock大小）

eg

| Block 大小        | 每 SM Block 数上限 | 总 Warp 数 | Occupancy |
| --------------- | -------------- | -------- | --------- |
| 32 (1 Warp)     | 32（Block 数限制）  | 32       | 50%       |
| 64 (2 Warps)    | 32（Block 数限制）  | 64       | 100%      |
| 128 (4 Warps)   | 16（2048/128）   | 64       | 100%      |
| 256 (8 Warps)   | 8（2048/256）    | 64       | 100%      |
| 512 (16 Warps)  | 4（2048/512）    | 64       | 100%      |
| 1024 (32 Warps) | 2（2048/1024）   | 64       | 100%      |

### Occupancy的使用
#### 使用CUDA API
```cpp
#include <cuda_runtime.h>

int main() {
    int blockSize = 256;
    int minGridSize, gridSize;

    // 自动计算最佳 Block 大小
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize,
        my_kernel,  // kernel 函数指针 指定Block数量、大小
        0,          // 动态共享内存大小
        0           // Block 大小上限（0=无限制）
    );

    // 查询给定配置的 Occupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,
        my_kernel,
        blockSize,
        0  // 动态共享内存
    );

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    float occupancy = (float)(maxActiveBlocks * blockSize) /
                      prop.maxThreadsPerMultiProcessor;
    printf("Occupancy: %.1f%%\n", occupancy * 100);
    return 0;
}

```

#### 高Occupancy 不等于 高性能
- 一个例子

| 📊 版本     | 寄存器/线程 | 共享内存/Block | Occupancy | 实际 GFLOPS |
| --------- | ------ | ---------- | --------- | --------- |
| Version A | 32     | 8 KB       | 100%      | 800       |
| Version B | 96     | 48 KB      | 33%       | 1200      |

Version B 的 Occupancy 更低，但性能更高！
原因是它使用更多寄存器和共享内存来**增加数据复用**，
**每次从全局内存加载的数据被反复使用多次，减少了总的内存访问量**。


#### 高Occupancy的代价
通常情况下哎，每线程寄存器数越小、Block声明共享内存越小、Block大小越小
Occupancy越高

1. **寄存器溢出（Register Spilling）**：
	为降低寄存器用量强制限制 `maxrregcount`，
	**编译器将变量溢出到局部内存（实质是全局内存 + L1 缓存），速度慢数十倍**
    
2. **共享内存复用不足**：
	使用更少的共享内存意味着 Tile 更小，
	**数据复用率降低，总的全局内存访问量反而增加**
    
3. **缓存抖动（Cache Thrashing）**：
	**太多活跃 Warp 争抢有限的 L1/L2 缓存，导致 cache miss 率上升**

#### 什么情况下Occupancy是关键？
Occupancy 对**内存访问密集且复用度低**的 Kernel 最重要：

- ✅ Elementwise 操作（向量加法、激活函数等）：几乎无数据复用，纯靠延迟隐藏
- ✅ 归约操作：每个元素只被读一次
- ✅ 简单的 Stencil 操作

相反情况下，对大量数据复用、寄存器和共享内存换来计算效率的场景，Occupancy不适用：
- ✅ GEMM / 矩阵乘法：大量数据复用，寄存器和共享内存换来计算效率
- ✅ FlashAttention：用更多共享内存减少 HBM 访问
- ✅ 计算密集型 Kernel：瓶颈在计算而非访存

## 同步和原子操作
CUDA的同步机制解决两个问题：
1. 执行顺序：确保线程A写入的数据在线程B读取前已经完成
2. 内存可见性：**确保一个线程的写入对其他线程“可见”**（不被缓存遮挡）
![[Pasted image 20261007173952.png]]

### Block内同步：__syncthreads()
是 CUDA 中最常用的同步原语，其语义为：

> Block 内的所有线程都到达此屏障点后，才能继续向下执行。
> 同时**保证屏障前的所有内存写入对 Block 内所有线程可见**。

#### 使用场景例子
```cpp
__global__ void shared_memory_example(float* input, float* output, int N) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // 阶段一：所有线程将数据从全局内存加载到共享内存
    smem[tid] = (gid < N) ? input[gid] : 0.0f;

    // 必须同步！确保所有线程的写入完成后再读取
    __syncthreads();

    // 阶段二：每个线程读取其他线程写入的数据
    float left = (tid > 0) ? smem[tid - 1] : 0.0f;
    float right = (tid < blockDim.x - 1) ? smem[tid + 1] : 0.0f;
    output[gid] = left + smem[tid] + right;
}
```
如果删除 `__syncthreads()`，线程**可能读到未初始化的共享内存值**（其他线程尚未完成写入），导致结果错误且难以调试（因为行为取决于时序，可能偶尔正确）。

#### 使用规则和陷阱
1. 所有线程必须到达同一个 __synthreads
	如果在分支中使用，
	会导致一部分线程永远无法到达 -> 出现死锁

2. 在循环中使用时，需要确保所有线程循环次数相同
	风险和上点一致

#### 性能代价
本身硬件开销低，真正的代价在于：
1. 等待最慢的线程
	已经到达的线程会导致空等

2. 流水线断裂
	**屏障点会打断编译器的指令重排优化**

因此需要尽量减少使用次数
如果kernel中有多个阶段的共享内存读写，考虑合并为一次

### Warp级同步
#### 隐式同步（Pre-Volta）
在 Volta 之前的架构（Pascal、Maxwell 等），
**同一 Warp 内的线程以 lockstep 方式执行——硬件保证它们始终在同一条指令上**。
因此 Warp 内的线程天然同步，不需要显式屏障。

#### 显式同步（Volta+）
从 Volta 开始引入独立线程调度，同一 Warp 内的线程**可能**不再步调一致。
因此需要显式同步：

```cpp
__shared__ float smem[256];
smem[threadIdx.x] = value;

// Volta+ 架构必须显式同步
__syncwarp(0xFFFFFFFF);  // 确保 Warp 内 32 个线程都完成写入

float neighbor = smem[threadIdx.x ^ 1];
```

`__syncwarp(mask)` 的参数是参与同步的线程掩码。
通常使用 `0xFFFFFFFF`（所有 32 个线程），但如果你知道只有部分线程参与，可以使用更精确的掩码。

#### 使用场景
|场景|Pre-Volta|Volta+|
|---|---|---|
|Warp 内通过共享内存通信|隐式安全（但不推荐）|必须 `__syncwarp()`|
|Warp Shuffle 前后|不需要|不需要（Shuffle 自带同步语义）|
|Warp Vote 前后|不需要|不需要|
|Warp 内的归约循环|不需要额外同步|Shuffle 版本不需要|


### Memory Fence和内存可见性
`__syncthreads()` 解决了 Block 内的同步问题，但跨 Block 通信怎么办？
GPU 没有全局屏障（传统意义上），但**提供了 Memory Fence 保证内存写入的可见性。**

- 不阻塞线程执行
	只会保证：Fence之前的**内存写入在Fence之后对目标范围内的其他线程可见**

```cpp
// Block 级：确保写入对同一 Block 内的线程可见
__threadfence_block();

// Device 级：确保写入对同一 GPU 上所有线程可见
__threadfence();

// System 级：确保写入对系统内所有设备（包括 CPU、其他 GPU）可见
__threadfence_system();

```


