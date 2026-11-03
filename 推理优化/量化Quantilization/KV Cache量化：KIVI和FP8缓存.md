> **KV为什么不能照搬权重量化？**
> 权重可以离线长时间跑一次GPTQ，然后反复使用
> 但是KV是在在线推理时不断产生的，没生成一个Token就要写入一份


## KIVI：Key按通道，Value按Token
### KV误差影响的区别
由 `P = softmax(QK^t/sqrt(d) O = PV` 可知，

**Key的误差可能让模型关注错位置**
**Value的误差按注意力权重加权后影响输出**

### KIVI的量化思路
[KIVI](https://arxiv.org/abs/2402.02750) 是一种免调参的 2-bit KV Cache 量化算法。
它的出发点是一个观察：
在其研究的模型中，**Key 有明显的通道离群值，Value 没有**。

因此论文**主张 Key 按通道量化、Value 按 Token 量化。**
固定一层、一个请求和一个 KV Head，把 K、V 都看成 T×dT×d 矩阵，
”行 = Token，列 = 通道”的约定：

- **Key Per-channel**：
	固定一个通道，沿连续的若干 Token 分组求量化参数（竖着切）。
	大幅值通道就被单独隔离开，不会拖累其他通道
	
- **Value Per-token**：
	固定一个 Token，沿通道分组求量化参数（横着切）。
	每个Token用自己的范围
	

## vLLM的FP8 KV Cache
工程上**更常用的路线是把 KV 存成 FP8：**
	数据量减半，精度压力比 2-bit 小得多，
	也不需要 KIVI 那样的特殊缓存结构。

### 常用格式
`kv_cache_dtype` 的常用取值：

|📊 取值|含义|支持说明（v0.32.0 文档）|
|---|---|---|
|`auto`|与模型数据类型一致|默认值|
|`fp8` / `fp8_e4m3`|E4M3，精度更高|CUDA 11.8+ 与 ROCm|
|`fp8_e5m2`|E5M2，范围更大、精度更低|CUDA 11.8+|

源码里的 `CacheDType` 还列出了 `int8_per_token_head`、`fp8_per_token_head`、`nvfp4` 等更多选项，它们依赖特定的 Attention 后端，使用前要查对应后端的支持情况。另外，DeepSeek-V3.2 等少数模型默认就用 FP8 KV。


### 存储精度不等于计算精度
FP8 KV 首先减少的是存储；
读出来以后，后端可能转回高精度再算，也可能直接在 FP8 域做 Attention。

vLLM 文档注明，**Flash Attention 3 后端配合 FP8 KV 时，Attention 在 FP8 域计算，Query 也会被量化成 FP8**。
所以只看 `kv_cache_dtype` 不能断定整段 Attention 的精度，
要结合 GPU、后端和启动日志一起判断（第 2 章讲过后端选择）