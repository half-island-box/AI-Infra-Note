# AI Infra Note

学习 AI Infra（AI 基础设施）的个人笔记仓库，基于 [Obsidian](https://obsidian.md/) 管理，内容涵盖 GPU 体系结构、CUDA 算子、大模型结构、分布式训练与推理优化等方向。

> 学习路线与计划见 [路线计划.md](路线计划.md)。

## 目录结构

```
.
├── CS336/          # Stanford CS336 课程笔记
│                   #   分词器、PyTorch 与资源核算、语言模型架构、GPU 优化、分布式训练
├── GPU/            # GPU 体系结构与 CUDA
│   └── CUDA/       #   CUDA 入门、Kernel 性能优化基础
├── LLM结构/        # 大模型结构
│   ├── Layers/     #   Attention 及其变体、位置编码、归一化、激活函数
│   └── Models/     #   具体模型适配（如 Qwen3）
├── 分布式训练/      # 分布式训练
│   └── 并行方式/    #   数据并行、张量并行、流水线并行、3D 并行
│                   #   另有 ZeRO、集合通信原语、优化器显存开销等
├── 推理优化/        # 推理优化
│   ├── 推理引擎-vllm/        # vLLM 结构与核心算子
│   ├── 推理引擎核心优化技术/  # Continuous Batching、PageAttention、Prefix Cache 等
│   ├── 投机解码/              # 投机解码、Medusa 与 EAGLE
│   └── 量化Quantilization/   # 量化基础、SmoothQuant
└── Excalidraw/     # Excalidraw 手绘图（模型结构、推理链路等）
```

其余说明：

- `欢迎.md`、`未命名.md` 等为 Obsidian 初始或草稿笔记。
- `.obsidian/` 为 Obsidian 配置目录，`.claude/`、`.copilot/` 为 AI 工具配置。
