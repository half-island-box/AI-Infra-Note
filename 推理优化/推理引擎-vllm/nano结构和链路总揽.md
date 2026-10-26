

## 四层架构

![[Pasted image 20261014165024.png]]

**🔌 第一层：接口层（`llm.py` + `sampling_params.py` + `config.py`）**
用户只接触 `LLM`（提供 `generate`）和 `SamplingParams`（温度、最大长度等）。
`Config` 则在引擎初始化时一次性把显存比例、最大批大小、张量并行度、KV 块大小等全局约束固化下来。

这一层的设计哲学是**对外 API 最小化**——你不需要懂调度就能跑通推理。


**🧠 第二层：引擎层（`engine/llm_engine.py`）**
`LLMEngine` 是整个系统的「业务逻辑心脏」。
**它持有调度器、模型执行器和分词器，**

跑一个循环：**从调度器拿一批可执行的序列 → 交给执行器做一次前向 → 采样出下一个 token → 更新序列状态**，直到所有请求结束。

**🗂️ 第三层：调度层（`scheduler.py` + `sequence.py` + `block_manager.py`）**
这一层回答「这一步该跑谁、它们的 KV 放哪儿」。

`Scheduler` 决定本轮是 prefill 还是 decode、把哪些序列拼成一个 batch；
`Sequence` 跟踪每条请求的 token、步数、是否结束；
`BlockManager` 用 PagedAttention 的思想给 KV 分配物理块。

注意：**它只管资源分配与生命周期，不碰矩阵乘法**——这也是它属于调度层而非 `layers/` 的原因。

**⚙️ 第四层：执行层（`model_runner.py` + `models/qwen3.py` + `layers/*`）**
`ModelRunner` 把**调度器选出的序列「张量化」**：

构造 `input_ids`、`positions`、`slot_mapping`、`block_tables` 等张量，
设置全局 `Context`，然后调用 `Qwen3` 前向。

`layers/` 是算子级实现，其中 `attention.py` 身兼两职
——既负责把新的 K/V 写进 Cache，又负责调用 FlashAttention 做注意力计算。


## 一次请求的生命周期
![[nano-vllm  |1000]]

1. 业务脚本（example.py）用分词器将prompt编码为token_id序列，调用`llm.generate`
2. LLM Engine将prompt列表，转换为Seq列表后，交给Sceduler的waiting_queue
	Seq包含包含p/d标记、BlockManager维护映射表等
	
3. 主循环每执行一次 `step()`：
	`Scheduler` 根据 Token Budget 和 KV 余量选出本轮要跑的序列，
	并标记这是一次 **prefill**（处理 prompt）还是 **decode**（逐 token 生成）。


4. `ModelRunner` 把这批序列转成 GPU 张量，写好全局 `Context`，跑一遍 `Qwen3`。
	前向过程中 `Attention` 把**新算出的 K/V 存进 KV Cache**，
	再用 **FlashAttention 算注意力。**

5. 模型输出的 hidden state 过 `ParallelLMHead` 得到 logits，`Sampler` 按温度采样出下一个 token。

6. `Scheduler.postprocess` 把新 token 写回对应 `Sequence`，
	判断是否命中 EOS 或达到 `max_tokens`
	结束的序列释放 KV 块，
	未结束的留到下一轮 decode。

7. 请求的所有序列结束，按请求顺序deocode为结果字符串返回
