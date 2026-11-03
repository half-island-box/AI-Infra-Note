![[Pasted image 20261103194626.png]]

Reduce规约，即把某个给定位置在不同进程的不同数据，
收集到一起进行如 sum、max、min、dot product等运算

本文以 **Sum Reduce** 为例（其他操作原理相通），输入为长度 NN 的 `float` 数组，输出为所有元素之和。


## 性能瓶颈分析
对于 Sum Reduce：

- 每个元素读取一次（1 次 load）
- 每次读取后做一次加法（1 次 FLOP）
- 算术强度 =1 FLOP4 Byte=0.25 FLOP/Byte（每个 `float` 元素 4 字节，对应 1 次加法）

这表明 Reduce 是典型的**访存密集型**（Memory-Bound）操作，
优化的核心在于**提升内存带宽利用率**，而不是减少计算次数。

## 版本方法和优化
### V0: 朴素的树形并行规约

1. 每个tid拿到`input[tid]`的数据
2. 满足 `tid % (2*step) == 0` 的线程将自己的值与偏移 `step` 处的值相加，
	步长每次翻倍，直到所有值汇聚到第 0 号元素

3. tid 0的线程将结果复制回output
![[Pasted image 20261103195248.png]]

#### 性能问题：warp内分支
由[[Kernel性能优化基础#Warp的分支代价]]可知，
统一warp内的线程永远同步执行一个指令，

**所以当前版本的if会串行执行，**
**并且随着step增大，线程利用率会不断下降**

### V1: 活跃tid保持连续，消除Warp内分支
一个直接的优化思路是：
**不改变规约的步长方向**（仍然从小到大），但改变线程映射关系
——让**相邻的线程**负责相邻步长对应的加法，而不是”按 tid 间隔挑选”。

**即数据的处理位置，不再直接按tid映射，**
**而是保证每一轮处理的tid都是连续的**
![[Pasted image 20261103195601.png]]

这种做法，前期的并行数量>32(一个warp的线程数)时，
可以保证warp内不会分化
直到活跃线程数不足32个，才会出现分化

#### 性能问题：bank conflict
前版本依靠Block内，**工作的Warps之间的Shared Memory来传递数据**
但是
Shared Memory 被划分为 32 个 **Bank**（默认 32-bit 模式，每个 Bank 宽度为 4 字节）。相邻的 4 字节地址依次落到 Bank 0、Bank 1、…、Bank 31、Bank 0、Bank 1、……如此循环

这会导致smem[0] 和smem[32]，实际上都落在了Bank0中，并且处于不同位置
如果不同的tid，分别要访问smem[0] smem[32]，则会发生bank conflict

> 如果一个 Warp 内有两个或多个线程访问**同一个 Bank 的不同地址**，访问就会**串行化**，这就是 **Bank Conflict**。
⚠️ **注意**：同一 Warp 内多个线程访问**同一地址**不会产生冲突（硬件可以广播），只有访问同一 Bank 的**不同地址**才会冲突。

### V2: 步长反转为递减，解决Bank Conflict
![[Pasted image 20261103200252.png]]
在这种模式下，以 Warp 0 为例：

- **第 1 轮（step=128）**：tid 0 访问 smem[0, 128]，tid 1 访问 smem[1, 129]，…，tid 31 访问 smem[31, 159]。这 32 个线程分别访问 Bank 0~31 的不同地址 → **无 Bank Conflict**
- **第 2 轮（step=64）**、**第 3 轮（step=32）** 同理，32 个线程刚好覆盖 32 个 Bank
- 当 step ≤ 16 时，活跃线程减少，剩下的访问也完全落在不同 Bank 上

可以发现，每一个step，都保证一个bank的数据只被一个tid访问，消除了bank conflict


#### 性能问题：线程的空闲问题
前版本的数据记载，
都只是让每个线程负责一个元素，即每个 Block 处理 `blockDim.x` 个数据

问题在于，**规约的第一轮，只有前面一半blockDim的线程参与了后续的迭代计算**
后面一半的线程彻底闲置，它们只在开头将对应数据搬到了Shared Memory

### V3： 每个线程负责加载2个元素，保证规约迭代时全部线程参加
```cpp
// V3: 每线程处理 2 个元素，减少空闲线程
__global__ void reduce_v3(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 每个线程加载 2 个相距 blockDim.x 的元素并求和
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();
	
	...
}

```

优化的本质，即在不增加规约轮次的前提下，让单个线程承担更多工作

#### 性能问题：多余的warp间同步
V3 在规约阶段仍然使用完整的 for 循环，当 `step <= 32` 时，只有 1 个 Warp 的线程（tid 0~31）还在工作。

**此时循环里的 `__syncthreads()` 完全是多余的**，
	因为同一 Warp 内的线程在执行相同指令路径时本身就是同步的，不需要额外的屏障。
	这些多余的 `__syncthreads()` 带来了不必要的开销。

### V4: 特殊处理：展开最后一个Warp，直接做树形规约
```cpp
// 辅助函数：展开最后 32 个线程的规约
__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid +  8];
    smem[tid] += smem[tid +  4];
    smem[tid] += smem[tid +  2];
    smem[tid] += smem[tid +  1];
}

// V4: 展开最后一个 Warp（在 V3 基础上）
__global__ void reduce_v4(float* input, float* output, int n) {
    extern __shared__ float smem[];


    // 规约循环仅执行到 step > 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // 最后一个 Warp 内的规约，无需 __syncthreads()
    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

```

因为剩下64个部分和开始，每一轮活跃处理的线程依次是：64、32、16 ...
并且前版本的方法已经保证每一轮迭代tid连续
-> 后续处理的每轮线程都在一个warp内，所以不需要__synthreads()

⚠️ **注意**：`warpReduce` 中的 `smem` 必须声明为 `volatile`，防止编译器将中间结果缓存到寄存器，导致其他线程读取到旧值。这是保证 Warp 内线程间数据可见性的关键。


#### 性能问题：Block大小已知的话可以消除分支判断
V4 展开了最后一个 Warp，但前几轮（step = 128, 64）仍然是运行时循环。每次循环迭代都包含：

- 边界判断（`s > 32`）
- 分支跳转
- 循环变量更新（`s >>= 1`）

对于 Block 大小在编译期已知的情况（如固定为 256），这些开销完全可以消除。