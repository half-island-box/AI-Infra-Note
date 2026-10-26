>同样一批请求的一次前向计算，怎样跑的更快？
>1. 选对高效的Attention Kernel
>2. CUDA Graph+torch.compile消除Decode的CPU开销


## CPU-Lanch Bound：Decode不仅受限于Memory
当模型不大、Batch 也不大时，还有一个更隐蔽的瓶颈会冒出来：
**CPU 发射 Kernel 的开销**。

- GPU 自己不会主动干活，
	它每做一个操作（一次矩阵乘、一次 LayerNorm、一次激活……），都要由 CPU 通过一次”启动（launch）“命令告诉它


因为 Prefill 一步要处理成千 Token，每个 GPU Kernel 一跑就是几百微秒甚至毫秒，CPU 那点启动开销被淹没了。
但 **Decode 每步只处理 1 个（或很少几个）Token**，每个 Kernel 的实际计算可能就几微秒，结果**CPU 发射这个 Kernel 的时间比 GPU 执行它的时间还长**。
于是 GPU 干几微秒就停下来，眼巴巴等 CPU 发下一条命令——算力再次闲置，只不过这次不是等数据，是**等指令**。


## Attention后端
Attention 是 Transformer 里最重、最讲究的算子，它的 Kernel 实现直接决定推理速度。

不同的后端Kernel，本质是同一个数学操作的不同工程实现
它们算出的结果在数学上等价
区别在于访存模式、并行策略等

### 可插拔的Attention后端
vLLM 的设计哲学是**不绑死一种实现，而是提供可插拔的 Attention 后端**，
根据硬件、模型结构和场景自动或手动选择最优的那一个

| 🏷️ 后端             | 特点                                    | 典型适用                   |
| ------------------ | ------------------------------------- | ---------------------- |
| **FlashAttention** | IO 感知的经典高性能实现，支持到 FlashAttention 3    | 主流 NVIDIA GPU 的通用首选    |
| **FlashInfer**     | 面向推理优化，KV Cache 处理灵活，支持多种数据类型         | 高吞吐服务、需要灵活 KV 布局       |
| **FlashMLA**       | 针对 MLA（Multi-head Latent Attention）优化 | DeepSeek 等采用 MLA 结构的模型 |
| **Triton**         | 用 Triton 语言实现，可移植性好                   | 非 NVIDIA 硬件或需自定义时的兜底   |

### Attention后端要求处理PD混合批次
上一节讲到，vLLM调度器会把PD塞进同个batch，[[Chunked Prefil与PD分离：长Prompt的prefill加速#调度过程]]
这就要求Attention后端原生支持混合批次——不需要把Prefill和Decode拆成两次Kernel

## CUDA Graph：一次打包固定多次Kernel启动

把一连串固定的 GPU 操作**录制（capture）**成一张”图”，之后只需一条命令**重放（replay）**整张图，GPU 就会依次执行录好的所有 Kernel——**成百上千次 CPU 启动被压缩成一次**。

用类比来说：
没有 CUDA Graph，就像老师每讲一句话都要等秘书传一次话（每个 Kernel 都要 CPU 单独发射）；
有了 CUDA Graph，相当于老师提前把整段讲稿录好，上课时按一下播放键，一气呵成放完，中间不用秘书再一句句传。


#### 硬性前提
但 CUDA Graph 有个硬性前提：**录制的操作序列和张量形状必须固定**。

这对 Decode 是天作之合——每步都是”处理固定 Batch 个 Token”，形状稳定，
可以针对不同的 Batch Size 分别录制好图，运行时按当前 Batch Size 选对应的图重放。

而多变的Prefill较难直接套用

- vLLM默认只捕获一组预设batch_size的CUDA Graph
	实际size命中时才走graph replay
	否则回退到逐个Kernel启动

#### Pecewise CUDA Graph
CUDA Graph 要求整段操作形状固定，
而 **Attention 恰恰是最难被 CUDA Graph 兼容的部分**——它涉及变长的 KV、复杂的 Mask，形状不那么规整。
如果因为 Attention 不好处理就放弃整张图的 CUDA Graph，就太可惜了。


vLLM V1 的巧解是 **Piecewise CUDA Graph（分段 CUDA Graph）**：

- 把整个前向图**在 Attention 算子处切开**（Attention 被包装成一个”不透明”的自定义算子，编译器不去动它）。
- **Attention 之外的部分**（LayerNorm、各种 GEMM、激活、残差……这些都是规整的 token-wise 操作）用 CUDA Graph 捕获、重放。
- **Attention 本身保持 eager（即时）执行**，保留它处理变长 KV 的灵活性。