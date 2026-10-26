## 常见算子总结

| 类别          | 算子名称                              | 数学表达式 / 描述                                                     | 典型应用场景                       |
| :---------- | :-------------------------------- | :------------------------------------------------------------- | :--------------------------- |
| **基础算术**    | **1. 加法Add**                      | C[i]=A[i]+B[i]                                                 | 特征融合、残差连接                    |
|             | **2. 减法Sub**                      | C[i]=A[i]−B[i]                                                 | 误差计算、差分运算                    |
|             | **3. 乘法Mul**                      | C[i]=A[i]×B[i]                                                 | 注意力掩码、逐通道缩放                  |
|             | **4. 除法Div**                      | C[i]=A[i]/B[i]                                                 | 归一化中间步骤、比值计算                 |
|             | **5. 绝对值Abs**                     | y[i] = [x[i]]                                                  | $                            |
|             | **6. 取反Neg**                      | y[i]=−x[i]                                                     | 梯度反转层 (GRL)、相位翻转             |
|             | **7. 平方 Square**                  | y[i]=x[i]^2                                                    | 均方误差 (MSE) 计算                |
|             | **8. 平方根Sqrt**                    | y[i]=x[i]y[i]=x[i]​                                            | 标准差计算、RMSNorm                |
|             | **9. 幂运算Pow**                     | y[i]=x[i]py[i]=x[i]p                                           | 对比度调整 (Gamma 校正)             |
| **激活函数**    | **10. ReLU**                      | y=max⁡(0,x)y=max(0,x)                                          | 深度神经网络标准激活函数                 |
|             | **11. Leaky ReLU**                | y=max⁡(αx,x)y=max(αx,x)                                        | 解决神经元死亡问题 (GAN 常用)           |
|             | **12. Sigmoid**                   | y=11+e−xy=1+e−x1​                                              | 二分类概率输出、门控机制                 |
|             | **13. Tanh**                      | y=ex−e−xex+e−xy=ex+e−xex−e−x​                                  | RNN/LSTM 内部状态激活              |
|             | **14. Swish / SiLU**              | y=x⋅σ(x)y=x⋅σ(x)                                               | EfficientNet、LLaMA 模型        |
|             | **15. GELU**                      | y=x⋅Φ(x)y=x⋅Φ(x)  <br>(近似公式: 0.5x(1+tanh⁡(… ))0.5x(1+tanh(…))) | Transformer (BERT, GPT) 标准激活 |
|             | **16. Hard Swish**                | y=x⋅ReLU6(x+3)6y=x⋅6ReLU6(x+3)​                                | MobileNetV3 移动端高效激活          |
|             | **17. ELU**                       | y={xx>0α(ex−1)x≤0y={xα(ex−1)​x>0x≤0​                           | 加速收敛且输出均值接近零                 |
|             | **18. Softplus**                  | y=ln⁡(1+ex)y=ln(1+ex)                                          | ReLU 的平滑近似                   |
| **裁剪与归一化**  | **19. Clip / Clamp**              | y=min⁡(max⁡(x,min),max)y=min(max(x,min),max)                   | 梯度裁剪、像素值范围约束                 |
|             | **20. 取最大值Max**                   | C[i]=max⁡(A[i],B[i])C[i]=max(A[i],B[i])                        | 最大池化辅助操作、ReLU 变体             |
|             | **21. 取最小值Min**                   | C[i]=min⁡(A[i],B[i])C[i]=min(A[i],B[i])                        | 对偶操作、距离场计算                   |
|             | **22. 舍入 (Round / Ceil / Floor)** | y=⌊x⌉y=⌊x⌉                                                     | 量化感知训练 (QAT) 中的伪量化           |
| **特殊函数与图像** | **23. 指数运算Exp**                   | y[i]=ex[i]y[i]=ex[i]                                           | Softmax 内部计算、对数似然            |
|             | **24. 对数运算Log**                   | y[i]=ln⁡(x[i])y[i]=ln(x[i])                                    | 交叉熵损失内部计算                    |
|             | **25. 取倒数Reciprocal**             | y[i]=1/x[i]y[i]=1/x[i]                                         | 除法优化、调和平均数                   |
|             | **26. 取符号Sign**                   | y[i]=sgn(x[i])y[i]=sgn(x[i])                                   | 二值化神经网络 (BNN)                |
|             | **27. 数值比较 (Eq / Ne / Gt / Lt)**  | y[i]=(A[i]==B[i])y[i]=(A[i]==B[i])                             | 生成布尔掩码、准确率统计                 |
|             | **28. RGB 转灰度**                   | Y=0.299R+0.587G+0.114BY=0.299R+0.587G+0.114B                   | 计算机视觉预处理、单通道特征提取             |
|             | **29. 颜色反转**                      | y=255−xy=255−x                                                 | 图像负片效果、数据增强                  |
|             | **30. Dropout**                   | y=x⋅mask1−py=1−px⋅mask​                                        | 神经网络正则化 (掩码为逐元素伯努利采样)        |
|             | **31. 缩放与偏置Scale & Bias**         | y=γx+βy=γx+β                                                   | BatchNorm 的仿射变换部分            |


## 核心特性
### 数据并行
以上所有算子的共同核心在于：
输出位置 i 的值仅取决于输入数据在同一位置 i 的值（单目运算）或对应位置 i 的值（双目运算），
计算过程**不涉及相邻位置数据的求和、滑动窗口或矩阵乘法**。

- 对于GPU而言 意味着：
	不需要共享内存：线程之间无需交换数据和同步指令
	不需要复杂索引：除了计算线程的全局idx之外，没有额外的寻址

因此这类Kernel天生**拥有优秀的数据并行性，**
**最容易优化到理论峰值带宽**

### 极低的计算访存比
- 逐元素算子最大的瓶颈在于访存：
	在 GPU 上执行一条算术指令（如浮点加法、乘法）的耗时，往往只有几个时钟周期；而访问一次全局显存的耗时，则是几百个时钟周期。

所以优化访存是最重要的一个方向

## 优化方向
根据[[入门——环境搭建和GPU模型#GPU的内存事务：Memory Transaction]]可知，Element-wise的访存延迟是最大的问题

### 基础代码模版
```cpp
// naive：一个线程只负责一个元素
__global__ void add_kernel(const float* A, const float* B, float* C, int N) {
    // 1. 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 边界保护
    if (idx < N) {
        // 3. 逐元素计算
        C[idx] = A[idx] + B[idx];
    }
}
```

索引计算、边界检查、算子实现是通用不变的顺序框架
基本只需要特殊化第三步的运算即可

通常启动配置为：
```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

### 隐藏访存延迟：Grid-Stride-Loop

既然运算单元在等数据，那就让每一个线程多负责几个元素的处理。

这也就是非常著名的网格跨步循环(Grid-Stride Loop)
即不再启动和数据量一致的线程数量，让每个线程处理一个索引位置
而是用总线程量作为步长stride，
**每个线程迭代处理`i+n*stride`位置的数据**
（因此启动的线程数量，最好能被数据数量整除）

```cpp
__global__ void add_kernel_v2(const float* A, const float* B, float* C, int N) {
    // 起始索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 总线程数（网格跨度）
    int stride = gridDim.x * blockDim.x;

    // 跨步循环处理多个元素
    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
```

启动配置并不需要覆盖整个N，而是可以让gridSize更小，这样每个线程就可以处理更多的数据。

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = 128 * 2;   // 故意只启动 SM 数量 × 若干倍，而非 N/256
add_kernel_v2<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

这种方案实际上已经成为了Element Wise核函数的事实标准，这样无论数据量多大，核函数都能够自适应，无需调整启动参数。


### 向量化加载
即使用了 Grid-Stride-Loop，每个线程每次循环依然只处理一个float
因此可以使用向量化加载：
[[Kernel性能优化基础#向量化加载（优化减少指令数）]] 

一个 float4 变量包含 4 个连续的 float，共 16 字节。使用它可以将 4 次 32-bit 访存合并为 1 次 128-bit 访存。

```cpp
__global__ void add_kernel_v3(const float* A, const float* B, float* C, int N) {
    // 向量化：一次处理 4 个元素
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // 以 float4 为单位循环
    for (int i = idx; i < N / 4; i += stride) {
        float4 a = A4[i];
        float4 b = B4[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[i] = c;
    }

    // 处理尾部不足 4 的剩余元素（略）
}
```


