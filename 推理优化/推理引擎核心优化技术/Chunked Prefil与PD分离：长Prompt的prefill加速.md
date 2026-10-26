Batching批处理时，如果有一个超长Prompt的Prefill存在，会抢占计算资源，导致同批的Decode请求拖慢。

## Prefill是如何拖慢Decode的？
回顾上一章的两条结论：

- **Prefill 是 Compute Bound**：
	一次要并行处理整段 Prompt 的所有 Token，
	Prompt 越长，这一步的矩阵运算量越大、耗时越久。
- **Decode 是 Memory Bound**：
	每步只处理 1 个 Token，本身很快，
	用户感受到的出字速度（TPOT）就靠它一步一步稳定推进。

朴素版的continupus batching，规定一个step为：
其中的所有seq，decode的输出一个token，prefill的完全prefill完成

所以这会导致长prompt的prefill会拖慢整个batch的decode输出速度


## Chunked Prefill：长Prefill切块
思路非常直接：
**不要求一个长 Prompt 在一步里 Prefill 完，而是把它切成若干个固定大小的 Chunk（块），分散到多个 step 里逐块处理。**

比如一个 4000 Token 的 Prompt，切成每块 512 Token，就分成 8 个 Chunk。每一步只处理其中一块，剩下的算力**留给同批的 Decode 请求正常出字**。这样：

- 每一步的计算量被控制在一个可预期的范围内，不会被超长 Prefill 突然撑爆。
- Decode 请求得以在长 Prefill 进行的同时**持续、平稳地推进**，TPOT 尖刺被抹平。

### Token Budget：调度器决定Chunk大小的度量
**无论 Prefill 还是 Decode，对 GPU 来说本质都是”处理若干个 Token 的前向计算”**。

- 一个做 Decode 的请求，这一步贡献 **1 个 Token**（生成下一个）。
- 一个做 Prefill 的请求（或它的一个 Chunk），这一步贡献 **N 个 Token**（这一块的长度）。

于是调度器给每一步设定一个总的 Token 预算上限（比如”这一步最多处理 2048 个 Token”），
然后往里面装请求：
先保证正在生成的 Decode 请求（每个占 1 个 Token 额度），
剩下的预算再分给 Prefill 的 Chunk。预算用完，这一步就发车。

![[Pasted image 20261008145644.png]]


### vLLM是如何用Budget切分出Chunk？
#### 调度决策的字典结构
```python
# V1 调度器每一步的决策，本质是这样一个映射
{
    "req_A": 1,      # 正在 Decode，处理 1 个新 token
    "req_B": 1,      # 正在 Decode，处理 1 个新 token
    "req_C": 512,    # 长 Prompt 的一个 Prefill chunk
    "req_D": 128,    # 短 Prompt，一步就能 Prefill 完
}
# 约束：所有 value 之和 ≤ token_budget

```

**`{request_id: num_tokens}` 这一个统一表示，同时涵盖了 Chunked Prefill、Prefix Cache 和 Speculative Decoding。**

因为它们本质都是”某个请求这一步要处理多少 Token”的不同取值——Prefill chunk 是一个较大的值，
Decode 是 1，
投机解码验证是若干个候选 Token。

一套调度框架，优雅地统一了所有情况。


#### 调度过程
```python
# 伪代码：还原 schedule() 里"预算截断即切块"的逻辑（非逐字源码）
def schedule(self):
    token_budget = self.max_num_batched_tokens
    scheduled = {}

    # 正在运行的请求：Decode 每个只要 1 个 token，也可能是上一步没填完的 prefill 续块
    for req in self.running:
        # 这个请求还差多少 token 才算完（prefill 未完成部分 或 decode 的 1 个）
        need = req.num_tokens_with_spec - req.num_computed_tokens
        # 关键：被剩余预算截断 → 长 prefill 在这里被"切"出一个 chunk
        num_tokens = min(need, token_budget)
        if num_tokens == 0:
            break
        blocks = self.kv_cache_manager.allocate_slots(req, num_tokens)
        if blocks is None:
            self._preempt_last(); continue
        scheduled[req.id] = num_tokens          # {request_id: num_tokens}
        token_budget -= num_tokens

    # 预算有剩，再从 waiting 拉新请求（同样会被预算截断成首个 chunk）
    while self.waiting and token_budget > 0 and self._can_admit():
        req = self.waiting.pop_front()
        need = req.num_prompt_tokens
        num_tokens = min(need, token_budget)     # 长 prompt 首个 chunk 也在此截断
        self.running.append(req)
        scheduled[req.id] = num_tokens
        token_budget -= num_tokens

    return SchedulerOutput(scheduled)

```

- min(need, token_budget)就是切块的本质：
	vLLM没有专门的分块器，prefiill的切块是自然截断的结果

- chunk的续传依靠num_computed_tokens:
	请求的”已计算到第几个 Token”被持久记录，
	下一步从断点继续，直到 `num_computed_tokens` 追上 `num_prompt_tokens`（Prefill 完成）后自然转入每步 +1 的 Decode。

## PD分离：Prefill和Decode实例解偶

典型的PD分离系统
会将Prefill和Decode分开到不同的GPU实例上运行：

```
        请求
         ↓
   ┌─────────────┐
   │  Prefill 实例 │  ← 只做 prefill，算 prompt 的 KV
   └─────────────┘
         ↓ 传输 KV cache
   ┌─────────────┐
   │  Decode 实例  │  ← 只做 decode，逐 token 生成
   └─────────────┘
         ↓
        输出
```

**关键流程：**

1. 请求先到 **Prefill 实例**，做一次 prefill，算出 prompt 的 KV cache。
    
2. 把这份 KV cache **传输到 Decode 实例**。
    
3. Decode 实例拿着 KV，逐 token 生成，直到结束。
    

Prefill 实例和 Decode 实例可以是**不同的 GPU、不同的机器，甚至不同型号的卡**

### 核心挑战：KV Cache传输
PD 分离最大的代价就是**要把 KV cache 从 prefill 实例搬到 decode 实例**。

问题在于 KV cache 很大：

- 一个长 prompt 的 KV，可能几百 MB 到几 GB；
    
- 传输走网络（如 RDMA、NVLink），带宽有限；
    
- 传输时间可能抵消掉分离带来的收益。
    

所以 PD 分离的关键技术点就是**如何高效传 KV**：

| 手段                    | 说明                      |
| --------------------- | ----------------------- |
| **RDMA / InfiniBand** | 高带宽低延迟网络，减少传输开销         |
| **NVLink**            | 同机多卡时用，带宽更高             |
| **分层传输 / 压缩**         | 压缩 KV、或只传必要部分           |
| **传输与计算重叠**           | prefill 算完一块就传一块，不等全部算完 |

### Prefill和Decode各自和共同存什么
- KV Cache只存在Decode实例中
	因为KV的使用实际只在Decode阶段起作用

- 共同存储
	模型权重：因为Prefill和Decode都是一次模型的forward
	分词器、框架代码等

### 和Chunked Prefill的关系？
两者解决同一个问题，但思路不同：

|方案|思路|优点|缺点|
|---|---|---|---|
|**Chunked Prefill**|同一批 GPU 上，把 prefill 切块，和 decode 错开|不传 KV，实现简单|仍在同一资源池，隔离不彻底|
|**PD 分离**|不同实例物理隔离，各干各的|隔离彻底，各自优化|要传 KV，架构复杂|

实践中可以**结合使用**：prefill 实例内部用 chunked prefill 提高吞吐，decode 实例专注低延迟生成。