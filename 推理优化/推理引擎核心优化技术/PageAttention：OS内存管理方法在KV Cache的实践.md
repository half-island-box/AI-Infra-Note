## 问题回顾：KV Cache的问题出在哪里
[[引言-大模型推理#碎片化问题]]

- **内部碎片（Internal Fragmentation）**：
	 一个请求最大可能生成 2048 个 Token，就先占好 2048 个 Token 的空间，但它实际可能只生成了 100 个就遇到 EOS 停了。
	剩下 1948 个 Token 的空间被这个请求”锁着”却没用上，其它请求也进不来。
- **外部碎片（External Fragmentation）**：
	不同请求预留的大块之间会留下大小不一的空隙，
	这些零碎空间加起来可能不小，但因为每一块都不够放下一个完整请求，等于全废了。

**问题的根源在于连续+预留这两个约束叠加**


## 基本原理
### 核心思想
仿照OS的虚拟内存分页机制：
- 把 KV Cache 切成固定大小的 **KV Block（块）**，
	**每个块存固定数量 Token 的 Key 和 Value**
	（这个数量由 `block_size` 决定，vLLM 中常见取值为 16）。
	
- 一个请求的 KV Cache 在逻辑上是连续的”逻辑块”序列，物理上却可以散落在显存的任意位置。
- 对于每个seq，用一张 逻辑**Block Table（块表）** 记录”
	这个请求的第几个逻辑块，对应物理显存里的哪一个物理块”。

即一次请求的Token序列被切分为多个物理的KV Block分散存储
并且用块表映射，显存不再要求连续，也不需要提前按最大长度预留


### Block Table：每个seq维护的逻辑块到物理块的映射
#### Prefill：分配物理Block、建立Block Table
我们用一个具体的例子走一遍。

假设 `block_size = 4`（每块存 4 个 Token 的 KV，真实场景常用 16，这里为了好画取 4）。一个请求 Prefill 了 9 个 Token，那么它需要 ⌈9/4⌉=3⌈9/4⌉=3 个逻辑块：

- 逻辑块 0：Token 0~3
- 逻辑块 1：Token 4~7
- 逻辑块 2：Token 8（只用了 1 格，还剩 3 格）

vLLM 从空闲物理块池里分配 3 个物理块给它
——**注意物理块编号可以完全不连续，比如物理块 `#7`、`#2`、`#5`**。
Block Table 就长这样：

![[Pasted image 20261007204619.png]]

#### Decode：查阅映射表获取KV，新Token追加
- Attention计算，获取历史KV
	CUDA Kernel 通过 Block Table 拿到每个逻辑位置对应的 `physical_block_number`（物理块号）和 `physical_block_offset`（块内偏移），
	就能从分散的物理块里正确读到所有历史 KV。

eg 
```cpp
// 根据物理块号 + 头偏移 + 块内偏移，算出这个 token 的 Key 起始地址
const scalar_t* k_ptr = k_cache
                      + physical_block_number * kv_block_stride   // 定位到物理块
                      + kv_head_idx           * kv_head_stride    // 定位到对应 KV 头
                      + physical_block_offset * x;                // 定位块内 token

```

进入 Decode 阶段后，**每生成一个新 Token 就往当前逻辑块的空位里填。**
逻辑块 2 还剩 3 格，能再接 3 个 Token；
填满后（生成到第 12 个 Token），vLLM 才**按需**再申请一个新物理块挂到逻辑块 3——
**用多少申请多少，绝不提前预留**。


### 内存共享与Copy-On-Write
分页机制还顺带解锁了一个连续分配时代做不到的能力：**多个请求共享同一份物理块**。

#### 共享前缀
这是共享物理块的一个典型场景

eg 一批请求来自同一个LLM，每个seq都携带相同的system prompt，
**那么这些请求的前缀KV完全相同，没有必要存储多份**
直接让这些seq的Block Table都指向同个物理Block即可

#### 写时复制（Copy-On-Write）
如果共享同一个块的两个请求，后来生成的内容分叉了怎么办？
答案是操作系统里另一个经典机制——**写时复制**


- 只读共享
	大家指向同一块，
	并且依靠 引用计数 来记录有几个seq请求在用

- 写时复制
	当某个请求要往一个被共享的块里写新内容时，
	先把这个块复制一份成为它的私有块，在副本上写，再更新自己的块表指向副本。
	其他请求不受影响。

## 抢占（Preemption）与恢复：显存不够时如何解决
当某个时刻并发请求过多，导致空闲的物理块不足以分配，
就需要用抢占Peempt来解决

当某一步Decode需要为若干请求追加新块，但空闲物理块不够时
会采取以下两种 方式：

|🔄 恢复方式|做法|✅ 优点|❌ 代价|
|---|---|---|---|
|**换出（Swapping）**|把 KV Cache 从 GPU 显存拷贝到 CPU 内存暂存，恢复时再拷回来|不用重算，省算力|占用 CPU 内存，来回拷贝有 PCIe 传输开销|
|**重计算（Recomputation）**|直接丢弃 KV，恢复时把该请求已生成的 Token 当成新 Prompt 重新 Prefill 一遍|不占 CPU 内存，实现简单|要重跑一次 Prefill，浪费算力|


## 工程实践
### vLLM中可以调整的相关参数
```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    # KV Cache 能用多少显存的总闸门：模型权重之外，
    # 剩余显存的这个比例划给 KV Cache 块池。默认 0.9 左右。
    gpu_memory_utilization=0.90,
    # 每个 KV Block 存多少个 token 的 KV。分页的基本粒度。
    block_size=16,
    # 开启前缀缓存，让相同前缀的请求复用物理块（见 2.3 节）。
    enable_prefix_caching=True,
    # 一个 batch 里最多同时处理多少条序列。
    max_num_seqs=256,
)

```

