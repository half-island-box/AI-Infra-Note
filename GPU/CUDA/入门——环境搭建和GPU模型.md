
![[Pasted image 20261013184528.png]]
## 开发环境
### GPU 驱动
驱动就像是你手机的操作系统固件——它让硬件能正常工作

```
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090       Off | 00000000:01:00.0  Off |                  Off |
|  0%   35C    P8              12W / 450W |       1MiB / 24564MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+

```

- **Driver Version**：当前驱动版本
- **CUDA Version**：驱动支持的最高 CUDA 版本
- **Memory-Usage**：显存使用情况

### CUDA Toolkit
 CUDA Toolkit 则是你在这个操作系统上安装的开发工具包——它提供编译器、库和调试器，让你能写出在 GPU 上跑的程序。
#### 包含的组件
|📦 组件|📝 作用|
|---|---|
|nvcc|CUDA C/C++ 编译器|
|cuBLAS|线性代数运算库|
|cuDNN|深度学习原语库（需单独下载）|
|cuFFT|快速傅里叶变换库|
|cuRAND|随机数生成库|
|Nsight Systems|系统级性能分析工具|
|Nsight Compute|Kernel 级性能分析工具|
|CUDA Runtime (libcudart)|运行时 API 库|
|cuda-gdb|GPU 调试器|

### NVCC编译工具链
nvcc 是 CUDA 程序的编译器驱动程序。它不是一个独立的编译器，而是一个**编译协调器**：
将 `.cu` 文件中的设备代码和主机代码分离，
**分别用 NVIDIA PTX 编译器和系统 C++ 编译器处理，最终链接成可执行文件。**

#### 编译流程
![[Pasted image 20261001110212.png]]

#### 常用编译选项

```
# 基本编译
nvcc hello.cu -o hello

# 指定 GPU 架构（为 RTX 4090 的 sm_89 编译）
nvcc -arch=sm_89 hello.cu -o hello

# 生成多架构兼容的 fatbinary
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_89,code=sm_89 \
     hello.cu -o hello

# 开启调试信息（用于 cuda-gdb）
nvcc -g -G hello.cu -o hello_debug

# 开启优化
nvcc -O3 hello.cu -o hello_opt

# 显示编译过程的详细信息
nvcc --verbose hello.cu -o hello

```

|选项|作用|
|---|---|
|`-arch=sm_XX`|指定目标 GPU 架构的计算能力|
|`-gencode`|生成多架构代码，适配不同 GPU|
|`-G`|生成设备代码调试信息|
|`-lineinfo`|在优化代码中保留行号信息（性能分析用）|
|`-Xcompiler`|将后续选项传递给主机编译器|
|`-std=c++17`|指定 C++ 标准版本|
|`-maxrregcount=N`|限制每个线程使用的寄存器数量|
|`--ptxas-options=-v`|显示每个 kernel 的寄存器和共享内存用量|

#### GPU架构速查
| 计算能力            | 架构代号         | 代表产品             |
| --------------- | ------------ | ---------------- |
| sm_70           | Volta        | Tesla V100       |
| sm_75           | Turing       | RTX 20 系列        |
| sm_80           | Ampere       | A100             |
| sm_86           | Ampere       | RTX 30 系列        |
| sm_89           | Ada Lovelace | RTX 40 系列        |
| sm_90           | Hopper       | H100             |
| sm_100 / sm_120 | Blackwell    | B200 / RTX 50 系列 |

## 编程模型
### 第一个CUDA程序
``` cpp
#include<cstdio>

  
  

// __global__ 标记为kernel函数，由CPU调用、GPU执行

__global__ void helloKernel(){

// 分别打印Block编号 和 块内线程的编号

printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);

}

  

int main(){


helloKernel<<<2, 4>>>(); //表示启动配置：2个Block、每个Block4个线程

cudaDeviceSynchronize(); //CPU阻塞等待GPU完成所有任务

printf("Done");

return 0;

}

```


### 异构计算模型
CUDA 程序运行在一个**异构系统**上：
**CPU（Host）负责控制逻辑和串行代码，GPU（Device）负责大规模并行计算。**
想象一个工厂——CPU 是厂长，负责下达指令、调配资源；
GPU 是拥有上万工人的车间，一旦接到任务就全员并行开工。

![[Pasted image 20261001112744.png]]

#### 三种函数修饰符
| 修饰符          | 执行位置 | 调用方             | 用途               |
| ------------ | ---- | --------------- | ---------------- |
| `__global__` | GPU  | CPU（或 GPU 动态并行） | Kernel 入口函数      |
| `__device__` | GPU  | GPU             | Kernel 内部调用的辅助函数 |
| `__host__`   | CPU  | CPU             | 普通 CPU 函数（默认）    |
这些修饰符可以组合使用：
```cpp
// 可以组合使用：同时为 CPU 和 GPU 编译
__host__ __device__ float square(float x) {
    return x * x;
}
```


### 三级线程层次
![[Pasted image 20261001113752.png]]

- **Grid（网格）** = 整个学校——一次 Kernel 启动产生的所有线程的集合
- **Block（线程块）** = 一个班级——Block 内线程可以协作（共享内存、同步）
- **Thread（线程）** = 一个学生——最小的执行单位

#### 硬件限制
| 限制项                | 最大值（Compute Capability ≥ 8.0）  |
| ------------------ | ------------------------------ |
| Block 每维最大线程数      | x:1024, y:1024, z:64           |
| Block 内最大线程总数      | 1024                           |
| Grid 每维最大 Block 数  | x:231−1231−1, y:65535, z:65535 |
| 每个 SM 最大活跃线程数      | 2048（sm_80）/ 1536（sm_89）       |
| 每个 SM 最大活跃 Block 数 | 16~32（取决于架构）                   |
**Block 内线程总数不能超过 1024，即 `blockDim.x * blockDim.y * blockDim.z ≤ 1024`。**


### 线程的索引计算
**由于Kernel函数的执行主体粒度为Block内的单个线程，**
**所以每次执行时需要先获取线程的索引**

每个线程都能通过**内置变量获取自身在层次结构中的位置**，进而计算出它应该处理哪一份数据。

#### 内置变量
|变量|类型|含义|
|---|---|---|
|`threadIdx`|`dim3`|线程在所属 Block 内的索引|
|`blockIdx`|`dim3`|Block 在 Grid 内的索引|
|`blockDim`|`dim3`|每个 Block 的维度（线程数）|
|`gridDim`|`dim3`|Grid 的维度（Block 数）|

#### 一维索引计算
eg 对于一个gridDim=4, blockDim=8的配置，全局索引分布如下：
```cpp
Block 0: [ 0  1  2  3  4  5  6  7]
Block 1: [ 8  9 10 11 12 13 14 15]
Block 2: [16 17 18 19 20 21 22 23]
Block 3: [24 25 26 27 28 29 30 31]
```

所以线程id计算：
```cpp
// 全局线程 ID = Block 编号 × Block 大小 + Block 内线程编号
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
```


#### 二维索引计算
类比一维，先算出每个维度的index,
然后用**LinearIdx=row*width+col**计算出一维先行索引

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

// 转换为一维线性索引（行主序）
int linearIdx = row * width + col;
```

#### Grid-stride Loop 模式
当数据量大于线程总数时，使用循环让每个线程处理多个元素：
```cpp
__global__ void processLargeArray(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // 总线程数

    // 每个线程以 stride 为步长遍历
    for (int i = idx; i < N; i += stride) {
        data[i] = data[i] * 2.0f;
    }
}
```

### Kernel函数与启动语法
#### 定义规则
```cpp
__global__ void myKernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}

```

Kernel 函数的约束：
- 返回类型必须是 `void`
- 不能使用可变参数（variadic）
- 不能是类的虚函数
- 不能递归（Compute Capability < 2.0，现代 GPU 均已支持）
- 参数通过值传递，指针必须指向设备内存

#### 启动语法
```cpp
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

| 参数               | 类型             | 必填  | 说明               |
| ---------------- | -------------- | --- | ---------------- |
| `gridDim`        | `dim3` 或 `int` | 是   | Grid 中 Block 的数量 |
| `blockDim`       | `dim3` 或 `int` | 是   | 每个 Block 中线程的数量  |
| `sharedMemBytes` | `size_t`       | 否   | 动态共享内存大小（默认 0）   |
| `stream`         | `cudaStream_t` | 否   | 执行流（默认 0，即默认流）   |


## 内存模型
### 内存层次总览
![[Pasted image 20261001143147.png]]

| 📊 内存类型    | 位置      | 容量                 | 延迟           | 带宽         | 作用域     | 生命周期       |
| ---------- | ------- | ------------------ | ------------ | ---------- | ------- | ---------- |
| 寄存器        | SM 芯片上  | ~256KB/SM          | 1 cycle      | 最高         | 单线程     | 线程生命周期     |
| 共享内存       | SM 芯片上  | 48~228KB/SM（因架构而异） | ~20 cycles   | ~19 TB/s   | Block 内 | Block 生命周期 |
| L1 Cache   | SM 芯片上  | 128~256KB/SM       | ~30 cycles   | ~19 TB/s   | 自动      | 自动管理       |
| L2 Cache   | GPU 芯片上 | 6~50MB             | ~200 cycles  | ~5 TB/s    | 所有 SM   | 自动管理       |
| 全局内存 (HBM) | 芯片外     | 16~80GB            | ~400 cycles  | 1~3.4 TB/s | 全局      | 应用生命周期     |
| 常量内存       | 芯片外+缓存  | 64KB               | 1~400 cycles | 有缓存时极高     | 全局只读    | 应用生命周期     |

### GPU的内存事务：Memory Transaction
GPU 的显存系统并非以单个字节或者单个浮点数为单位与核函数交互，**而是以内存事务（Memory Transaction） 为单位。**

以常见的 NVIDIA GPU 架构为例，L2 缓存与显存控制器之间的最小传输单元通常为 32 字节。
当 SM发出一次全局内存访问请求时，**硬件会尝试将同一 Warp 的所有请求合并成尽可能少的事务。**

- 理想情况：一个Warp的32个线程访问连续且对齐的32个float，也就是128字节，那么只需要发送4次32字节的事务，或者1次128字节的事务。
- 最坏情况：32个线程访问的内存地址在显存各处，并不连续，那么一次Warp的访存操作最差为32次独立的事务。

由于单次事务的延迟高达数百个时钟周期，拆分事务意味着带宽利用率断崖式下跌，核函数则需要等待数据。


### 全局内存（显存 HBM）
全局内存（Global Memory）是 GPU 的”主存”，对应物理上的 HBM（高带宽内存）。
**它容量最大但延迟最高，是 CPU-GPU 数据交换的枢纽。**

#### 基本操作
```cpp
// 分配设备内存
float* d_data;
size_t size = N * sizeof(float);
cudaMalloc(&d_data, size);

// Host → Device 传输
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Device → Host 传输
cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);

// 释放设备内存
cudaFree(d_data);

```

#### 合并访问
全局内存的访问效率取决于**一个 Warp 内 32 个线程的访问模式是否连续对齐**。
GPU 以 32B/64B/128B 的粒度从全局内存取数据（一次”事务”），
**如果 Warp 内线程访问的地址连续，一次事务就能满足所有线程**——这就是**合并访问**。

即可以用线程Idx对应到访存数据的Idx：
![[Pasted image 20261001143622.png]]

- 访问模式对性能的影响

|访问模式|事务数（Warp 访问 32 个 float）|有效带宽利用率|
|---|---|---|
|完美合并（连续对齐）|1 次 128B 事务|100%|
|连续但未对齐|2 次 128B 事务|50%|
|随机散列访问|最多 32 次 32B 事务|~3%|

### 共享内存（单个SM内）
共享内存（Shared Memory）位于 SM 芯片上，48~228KB/SM（因架构而异），
**是 Block 内所有线程共享的高速暂存**。
Block分配给某个SM后，SM中属于同个Block的所有Warp共享该内存

把它想象成一个团队的白板——**团队成员都可以快速读写，但其他团队看不到。**

#### 声明使用
- 静态声明
```cpp
__global__ void sharedMemDemo(float* input, float* output, int N) {
    // 静态声明：编译时确定大小
    __shared__ float tile[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 将全局内存数据加载到共享内存
        tile[threadIdx.x] = input[idx];

        // 同步：确保所有线程都完成加载
        __syncthreads();

        // 现在可以安全地访问 tile 中其他线程加载的数据
        int partner = blockDim.x - 1 - threadIdx.x;
        output[idx] = tile[partner];
    }
}

```

- 动态声明
```cpp
// 动态声明：运行时确定大小
__global__ void dynamicSharedMem(float* data, int N) {
    extern __shared__ float sharedData[];  // 大小在启动时指定

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        sharedData[threadIdx.x] = data[idx];
    }
    __syncthreads();
    // ...
}

// 启动时通过第三个参数指定动态共享内存大小
int sharedMemSize = blockSize * sizeof(float);
dynamicSharedMem<<<grid, block, sharedMemSize>>>(data, N);
```

#### Bank Conflict
共享内存被划分为 32 个 **Bank**（对应 32 个 Warp 线程），每个 Bank 宽度为 4 bytes。
即
**Bank编号=（地址/4 bytes）%32**

**当一个 Warp 中多个线程同时访问同一个 Bank 的不同地址时，**
访问必须串行化——这就是 **Bank Conflict**。

```cpp
// ✅ 无 Bank Conflict：每个线程访问不同 Bank
float val = shared[threadIdx.x];         // 线程 i → Bank i

// ❌ 2-way Bank Conflict：步长为 2，线程 0 和 16 冲突
float val = shared[threadIdx.x * 2];     // 线程 0 → Bank 0, 线程 16 → Bank 0

// ✅ 广播无冲突：所有线程读同一地址
float val = shared[0];                   // 广播，无冲突

```

#### 共享内存配置
**现代GPU（sm_80+）的L1 Cache和共享内存  共享同一块物理SRAM**
可以通过API配置比例：

```cpp
// 优先分配更多共享内存（最大 164KB on A100）
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 164 * 1024);

// 设置 L1/Shared Memory 偏好
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferShared);

```

### 寄存器内存
寄存器是每个线程私有的最快存储。
**Kernel中的局部变量默认存放在寄存器中：**
```cpp
__global__ void compute(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 存在寄存器
    float temp = 0.0f;                                  // 存在寄存器
    float acc = 0.0f;                                   // 存在寄存器

    for (int i = 0; i < 10; i++) {
        temp = data[idx + i * N];
        acc += temp * temp;
    }
    data[idx] = acc;
}
```

#### 寄存器溢出
每个 SM 的寄存器总量有限（如 A100 为 65536 个 32-bit 寄存器/SM）。
当一个线程使用的寄存器超过 SM 能分配给它的数量时，
**多出的变量会”溢出”到局部内存（实际是全局内存的一块区域，有 L1/L2 缓存加速但延迟仍然高）。**

```cpp
# 查看 Kernel 的寄存器用量
nvcc --ptxas-options=-v mykernel.cu
# ptxas info: Used 48 registers, 0 bytes smem

# 限制最大寄存器数（可能导致溢出但提高占用率）
nvcc --maxrregcount=32 mykernel.cu

```

### 内存选型决策（新Kernel的数据放在哪里）
![[Pasted image 20261001145908.png]]

| 需求             | 首选方案                     | 次选方案                   |
| -------------- | ------------------------ | ---------------------- |
| Block 内线程间数据交换 | 共享内存 + `__syncthreads()` | Warp Shuffle（同 Warp 内） |
| 全局只读广播         | 常量内存                     | `__ldg()`              |
| 大数组逐元素处理       | 全局内存（合并访问）               | —                      |
| 局部累加器/计数器      | 寄存器                      | —                      |
| 复杂数据结构快速原型     | 统一内存                     | —                      |
| 减少全局内存事务       | Tiling + 共享内存            | L1 缓存                  |


## 第一个实用kernel：向量加法
### Kernel的一次完整执行流程
![[Pasted image 20261001152247.png]]

|步骤|API|说明|
|---|---|---|
|分配 GPU 内存|`cudaMalloc()`|类似 CPU 的 `malloc()`|
|Host→Device 传输|`cudaMemcpy(..., HostToDevice)`|CPU 数据拷贝到 GPU|
|启动 Kernel|`kernel<<<grid, block>>>(args)`|GPU 并行执行|
|Device→Host 传输|`cudaMemcpy(..., DeviceToHost)`|结果拷回 CPU|
|释放 GPU 内存|`cudaFree()`|类似 CPU 的 `free()`|


