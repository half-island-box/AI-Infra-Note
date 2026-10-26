
> 在[[第六章 GPU和相关优化]]中，详细介绍了单个GPU的基本结构
> 而NVIDIA GPU的架构演进，
> 本质上是在理解AI硬件和AI算法的协同进化


## AI架构演进概览
![[Pasted image 20261019125604.png]]

|📊 架构|📅 年份|🖥️ 代表产品|⚡ 核心突破|
|---|---|---|---|
|Volta|2017|V100|首次引入 Tensor Core|
|Turing|2018|T4|INT8/INT4 推理加速、RT Core|
|Ampere|2020|A100|第三代 Tensor Core、MIG、TF32、BF16|
|Hopper|2022|H100|FP8、Transformer Engine、NVLink 4.0|
|Blackwell|2024|B200 / GB200|第五代 Tensor Core、NVLink 5.0、双芯封装|

## Volta架构：AI加速的开端

在Volta之前，深度学习训练基本依赖于GPU的基础核心CUDA Core。
CUDA Core什么都能算，
但是没有针对**矩阵乘法**（AI的核心操作）进行优化


### Tensor Core张量核心：GEMM的专门优化
Volta最重要的贡献是引入了Tensor Core
是专门为 矩阵乘加运算（MMA） 设计的硬件单元

- 传统CUDA Core
	每个时钟周期，**只能处理一次 标量的乘加运算（FMA）**

- Tensor Core
	每个时钟周期，可以完成一个`4*4*4`的混合精度矩阵乘加


### V100关键规格
| 参数             | 规格                      |
| -------------- | ----------------------- |
| Tensor Core 数量 | 640 个                   |
| FP16 Tensor 算力 | 125 TFLOPS              |
| FP32 CUDA 算力   | 15.7 TFLOPS             |
| 显存             | 16 GB / 32 GB HBM2      |
| 显存带宽           | 900 GB/s                |
| NVLink         | NVLink 2.0，300 GB/s（双向） 
### 实践意义：充分利用Tensor Core
Volta 开启了一个重要的编程范式转变：
要想充分利用 Tensor Core，你的矩阵维度需要是 8 的倍数（后续架构进一步放宽），数据类型需要使用混合精度（FP16 计算 + FP32 累加）。

这也是 NVIDIA 混合精度训练（AMP）的硬件基础。

```python
# PyTorch 中使用混合精度训练（Volta 时代开始受益）
import torch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, target in dataloader:
    optimizer.zero_grad()
    with autocast():  # 自动选择 FP16/FP32
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```


## Turing架构：RT Core和推理加速

Volta 主要面向训练场景。
但到了模型部署阶段，**推理对延迟和吞吐量的要求更加苛刻**，
且推理负载通常不需要 FP16 那么高的精度——很多模型用 INT8 就能保持足够的准确率。

Volta 的 Tensor Core 只支持 FP16，在推理场景下算力利用不够充分。


### 关键创新
- **INT8 / INT4 Tensor Core 支持**：第二代 Tensor Core 新增了整数精度运算能力，推理吞吐量相比 FP16 可再翻倍
- **RT Core**：光线追踪加速单元（主要面向图形渲染，AI Infra 领域较少涉及）
- **更高效的推理定位**：T4 卡的 TDP 仅 70W，适合部署在推理服务器中

### T4 关键规格
|参数|规格|
|---|---|
|Tensor Core 数量|320 个|
|FP16 Tensor 算力|65 TFLOPS|
|INT8 Tensor 算力|130 TOPS|
|显存|16 GB GDDR6|
|显存带宽|320 GB/s|
|TDP|70W|
T4 虽然绝对算力不如 V100，但其低功耗、高 INT8 吞吐的特性使它成为推理部署的经典选择。
很多云厂商的推理实例至今仍在使用 T4。


### 实践意义：模型量化
Turing 架构推动了 **模型量化（Quantization）** 在工业界的普及。TensorRT 的 INT8 量化 pipeline 正是以 Turing 架构为目标设计的：

```python
# 使用 TensorRT 进行 INT8 量化推理的典型流程
# 1. 导出 ONNX 模型
python export_onnx.py --model resnet50 --output model.onnx

# 2. 用 trtexec 构建 INT8 engine（需要校准数据集）
trtexec --onnx=model.onnx --int8 --calib=calibration_cache.bin --saveEngine=model_int8.engine
```


## Ampere架构：数据中心 一卡多用
2020 年，AI 模型规模急剧膨胀（GPT-3 有 1750 亿参数），训练所需的算力和显存同步增长。Volta/Turing 时代的两个痛点变得突出：

1. **精度格式不够灵活**：FP16 在某些模型上存在数值溢出问题，FP32 又太慢
2. **单卡利用率不高**：一块 GPU 只能跑一个任务，多租户场景下资源浪费严重

### 关键创新
#### TF32—— 隐式提速

Ampere 引入了 **TF32（TensorFloat-32）** 精度格式。TF32 使用 FP32 的 8 位指数位（保持相同的数值范围）加上 FP16 的 10 位尾数位，在不修改任何代码的情况下，对 FP32 运算自动提速：

| 格式   | 符号位 | 指数位 | 尾数位 | 总位宽 |
| ---- | --- | --- | --- | --- |
| FP32 | 1   | 8   | 23  | 32  |
| TF32 | 1   | 8   | 10  | 19  |
| FP16 | 1   | 5   | 10  | 16  |
| BF16 | 1   | 8   | 7   | 16  |
TF32 在 Ampere 上默认开启。当你在 A100 上跑 `torch.matmul` 时，如果输入是 FP32，Tensor Core 会自动使用 TF32 加速，不需要你显式开启混合精度。

#### MIG——一卡多用
Multi-Instacne GPU，**允许将一块A100物理隔离为最多7个独立的GPU实例**
每个实例拥有独立的显存、缓存和计算资源

```bash
# 查看 A100 支持的 MIG 配置
nvidia-smi mig -lgip

# 创建一个 MIG 实例（以 3g.20gb 为例：3 个 Compute Slice + 20GB 显存）
sudo nvidia-smi mig -cgi 9,9 -C

# 查看已创建的实例
nvidia-smi mig -lgi

```


### A100关键规格
| 参数                    | 规格                      |
| --------------------- | ----------------------- |
| Tensor Core 数量        | 432 个（第三代）              |
| FP16 / BF16 Tensor 算力 | 312 TFLOPS              |
| TF32 Tensor 算力        | 156 TFLOPS              |
| FP32 CUDA 算力          | 19.5 TFLOPS             |
| 显存                    | 40 GB / 80 GB HBM2e     |
| 显存带宽                  | 2.0 TB/s                |
| NVLink                | NVLink 3.0，600 GB/s（双向） |
### 实践意义
Ampere 架构让 **BF16 混合精度训练**成为主流。BF16 与 FP32 有相同的数值范围（8 位指数），避免了 FP16 常见的溢出问题，同时将显存占用和通信量减半。大部分 LLM 训练从 Ampere 时代起使用 BF16 作为默认训练精度：

```python
# 在 A100 上使用 BF16 训练（PyTorch 原生支持）
model = model.to(dtype=torch.bfloat16, device="cuda")

# 或使用 PyTorch 的 autocast
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(input_ids)
    loss = criterion(output, labels)
```


## Hooper架构：专门适配大模型

大语言模型（LLM）时代到来后，模型训练面临双重压力：

1. **算力饥渴**：百亿、千亿参数模型训练需要成百上千块 **GPU 并行**
2. **通信瓶颈**：多卡并行训练中，**GPU 之间的数据同步**（All-Reduce、All-to-All）成为重要瓶颈

Hopper 架构几乎每个重要特性都在回应这两个挑战。

### 关键创新
#### FP8 训练 —— 精度再压一刀

Hopper 的第四代 Tensor Core 支持 **FP8** 精度（E4M3 和 E5M2 两种格式），将训练精度从 BF16 的 16 位进一步压缩到 8 位。

|格式|符号位|指数位|尾数位|适用场景|
|---|---|---|---|---|
|E4M3|1|4|3|前向传播（精度优先）|
|E5M2|1|5|2|反向传播（范围优先）|

FP8 的理论算力是 BF16 的两倍。
但精度降低了，模型质量不会受损吗？这就引出了 Hopper 的另一个核心技术。


#### Transformer Engine——自动精度管理
**Transformer Engine** 是一个软硬件协同的机制：
它在每一层的计算前，动态分析张量的数值分布，自动决定这一层用 FP8 还是 BF16/FP16。

可以理解为一个”自动挡”——
- 在直路上（数值稳定的层）挂高速档（FP8），
- 在弯道上（数值敏感的层）自动降档（BF16）。

```python
# 使用 Transformer Engine 的 FP8 训练
import transformer_engine.pytorch as te

# 替换标准 Linear 层为 TE 的 FP8 版本
model.layer = te.Linear(hidden_size, hidden_size)

# 启用 FP8 自动混合精度
with te.fp8_autocast(enabled=True):
    output = model(input_data)

```

#### NVLink 4.0 和 NVSwitch
| 📊 指标         | Ampere (A100) | Hopper (H100)                | 提升   |
| ------------- | ------------- | ---------------------------- | ---- |
| NVLink 带宽（双向） | 600 GB/s      | 900 GB/s                     | 1.5x |
| All-Reduce 带宽 | 600 GB/s      | 900 GB/s                     | 1.5x |
| NVSwitch 支持   | 节点内 8 卡       | 节点内 8 卡 + 跨节点 NVLink Network | 全新   |
### 关键规格
|参数|规格|
|---|---|
|Tensor Core 数量|528 个（第四代）|
|FP8 Tensor 算力|1,979 TFLOPS（Dense）/ 3,958 TFLOPS（稀疏）|
|FP16 / BF16 Tensor 算力|989 TFLOPS（Dense）/ 1,979 TFLOPS（稀疏）|
|TF32 Tensor 算力|495 TFLOPS（Dense）/ 989 TFLOPS（稀疏）|
|显存|80 GB HBM3|
|显存带宽|3.35 TB/s|
|NVLink|NVLink 4.0，900 GB/s（双向，SXM 版）|

### 实践意义
Hopper 是当前大模型训练的主力架构。在实际的千卡训练集群中，以下特性的价值最为显著：

1. **FP8 + Transformer Engine** 将训练吞吐提升 30%-60%（取决于模型结构）
2. **NVLink 4.0** 大幅降低了多卡 All-Reduce 的通信延迟
3. **HBM3 的 3.35 TB/s 带宽** 缓解了 Attention 计算的访存瓶颈

```bash
# 查看 H100 的 NVLink 拓扑
nvidia-smi topo -m

# 查看当前 GPU 的 NVLink 状态
nvidia-smi nvlink -s
```

## Blackwell架构：万亿参数时代的基石

进入万亿参数模型时代，单块 GPU 的算力增长已经跟不上模型膨胀的速度。
Blackwell 的设计哲学是：
与其一味堆晶体管提升单 die 性能，不如**在芯片互联层面做革命性突破。**

### 关键创新
#### 双芯封装（Dual-Die）
B200 的核心由**两块 die 通过 10 TB/s 的片间互联**封装在同一块芯片上。
对外表现为一块完整的 GPU，但实际上拥有两块 die 的算力。

这就像把两匹马套在一起拉车，对车夫来说操作方式不变，但拉力翻倍。

#### [第五代 Tensor Core](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/gpu/nvidia-gpu-evolution/#%E7%AC%AC%E4%BA%94%E4%BB%A3-tensor-core)

FP8 Dense 算力相比 H100 提升约 14%（双 die 加持），含稀疏性加速后可达 4,500 TFLOPS。**FP4 精度首次获得硬件级支持**，Dense 算力 4,500 TOPS，开启稀疏性后可达 9,000 TOPS。FP4 的引入为推理场景提供了更极致的算力密度。

#### [NVLink 5.0](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/gpu/nvidia-gpu-evolution/#nvlink-50)

|📊 指标|Hopper (H100)|Blackwell (B200)|提升|
|---|---|---|---|
|NVLink 带宽（双向）|900 GB/s|1,800 GB/s|2x|
|NVLink 连接的 GPU 数|8|72（通过 NVLink Switch）|9x|

1.8 TB/s 的互联带宽意味着 GPU 之间的通信速度已经接近单块 GPU 的显存带宽。这从根本上改变了分布式训练的通信格局——过去需要精心设计通信拓扑来规避的瓶颈，现在可以用”暴力”带宽直接碾过。


### B200/GB200关键规格

GB200 是一个 **CPU-GPU 超级芯片**，
将 NVIDIA 自研的 **Grace ARM CPU 与 Blackwell GPU 通过高速 NVLink-C2C 互联封装**在一起。
这种设计**消除了传统 PCIe 的 CPU-GPU 通信瓶颈**。

| 参数            | B200                                  | GB200（Grace-Blackwell）             |
| ------------- | ------------------------------------- | ---------------------------------- |
| FP8 Tensor 算力 | 2,250 TFLOPS（Dense）/ 4,500 TFLOPS（稀疏） | 同左（GPU 部分）                         |
| FP4 Tensor 算力 | 4,500 TOPS（Dense）/ 9,000 TOPS（稀疏）     | 同左（GPU 部分）                         |
| 显存            | 192 GB HBM3e                          | 192 GB HBM3e + 480 GB LPDDR5x（CPU） |
| 显存带宽          | 8 TB/s                                | 8 TB/s（GPU 部分）                     |
| NVLink        | NVLink 5.0，1,800 GB/s                 | NVLink 5.0，1,800 GB/s              |
| 芯片设计          | 双 die 封装                              | Grace CPU + Blackwell GPU 紧耦合      |