
## 传统Dense稠密模型 VS MoE
### 架构对比

![[Pasted image 20261107170200.png]]

![[Pasted image 20261107170316.png]]


由图可知，
- 稠密模型的MLP层
	FFN的升维和降维矩阵，由于需要中间高维度的特征提取
	所以参数量非常大
	每次token计算，都要加载这两个大矩阵的全部参数
	在decode场景下，严重的memory bound

- MoE模型
	通过将每个专家构建成同纬度的小MLP + 一个专家路由层
	在token计算时，只需激活部分专家的权重参数
	显著降低访存压力



## MoE的构成和简单实现（给予Qwen3Moe）
> 一个MoE块，可以认为是 “专家路由+专家列表”的组合
> 
> 每个token首先通过路由，计算出对于每个专家的logits
> 经过softmax后，选出top_k个专家mlp进行计算
> 最后将计算结果根据softmax的权重加和


### 专家路由
```python
class Qwen3MoeTopKRouter(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        nn.init.normal_(self.weight, std=config.initializer_range)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(x, self.weight) # 经过路由层得到logits

        # Following Qwen3MoE, compute probabilities in float32 for stable routing.
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        router_scores, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True)

        return router_logits, router_scores.to(x.dtype), selected_experts

```

### 单个专家MLP
单个专家MLP本质上仍然是一个FFN结构，包含up、activation、down
区别主要在于`config.intermediate_size`一般较小
```python
lass Qwen3MoeMLP(nn.Module):
    """Qwen3-style gated MLP, also used as one routed expert."""

    def __init__(self, config: Qwen3MoeConfig, intermediate_size: int | None = None):
        super().__init__()
        intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(x), self.up_proj(x)))
```

### 组合的MoE Block
```python
class Qwen3MoeSparseMoeBlock(nn.Module):
    """A straightforward dense-loop implementation of Qwen3MoE routing."""

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.gate = Qwen3MoeTopKRouter(config)
        self.experts = nn.ModuleList(
            Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])

        router_logits, routing_weights, selected_experts = self.gate(x_flat)
        output = torch.zeros_like(x_flat)

        # The Python loop is intentionally easy to read. It executes only selected
        # experts, but can be replaced later by grouped GEMM for better performance.
        for expert_idx, expert in enumerate(self.experts):
            token_idx, _ = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue

            expert_output = expert(x_flat[token_idx])
            weights = torch.gather(
                routing_weights[token_idx],
                1,
                torch.where(selected_experts[token_idx] == expert_idx)[1].unsqueeze(1),
            ).squeeze(1)
            output.index_add_(0, token_idx, expert_output * weights.unsqueeze(1))

        return output.reshape(shape), router_logits.reshape(*shape[:-1], -1)

```