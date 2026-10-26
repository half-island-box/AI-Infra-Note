## MHA-MQA-GQA
![[Pasted image 20261016131550.png]]

### MHA：多头注意力
![[Pasted image 20261016131626.png]]

本质即沿着hidden_size维度，进行d/num_heads的维度切分
每个每个注意力头，
都有自己独特的QKV矩阵

每个头计算出自己的Attention后，结果进行拼接
#### 优缺点
优点：多个注意力头之间可以并行执行Attention，提速计算
缺点：KV Cache压力显著增加


### MQA：多查询Q的注意力
继续沿用多头方案，**每个头的Q仍然不同，但是K、V在多头间共用**
计算过程类似

### 优缺点
优点：解决了KV Cache的显存爆炸问题
缺点：所有头共用一组KV，语义信息被压缩，模型的表达能力下降

## GQA：分组查询注意力
既然所有头共用一份KV，模型能力下降，则采取折中方案：
1Q：m KV（num_heads % m = 0）

这样既可以提速计算，同时不会使KV Cache显存占用过多


## 代码分析
```python
class Attention(nn.Module):

def __init__(self, num_heads:int, head_dim:int, num_kv_heads:int):

	super().__init__()
	
	  
	
	if num_heads % num_kv_heads !=0: # 保证可以实现GQA
	
	raise ValueError("num_heads must be divisible by num_kv_heads")
	
	  
	
	self.num_heads=num_heads # 多头
	self.head_dim=head_dim # 单头处理维度
	self.num_kv_heads=num_kv_heads # KV组数
	
	  
	# 头数/KV数 -> 一个组的头数量（即Q数量）
	self.groups=self.num_heads //self.num_kv_heads
	self.scale=head_dim ** -0.5 # 缩放因子（～根号下的维度d）
 
  

def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:

#[B, S, H, D] [batch, seq, head, head_dim]

# head * head_dim == hidden_size 符合分组处理特征的直觉

  

B, S, H, D=query.shape

# transpose: [B, H, S, D] 因为要在 head 维度作矩阵运算

# reshape: [B, H_kv, G, S, D] 方便和后续同一 KV 组内的 query 复用 K/V

query = query.transpose(1, 2).reshape(B, self.num_kv_heads, self.groups, S, D)

  

# [B, S, H_kv, D] -> [B, H_kv, S, D] -> [B, H_kv, 1, S, D]

# K/V 的 head 数是 H_kv，不能在转置后直接 unsqueeze 而不先归并到 H_kv。

key = key.transpose(1, 2).reshape(B, self.num_kv_heads, 1, S, D)

value = value.transpose(1, 2).reshape(B, self.num_kv_heads, 1, S, D)

  

# 每个句子作自注意力 -> 矩阵大小为 S*S

# 下三角为False不需要遮掩：i >= j 表示q可以看到当下之前的token

# 相反， 上三角为true

mask = ~torch.ones(S, S, dtype=torch.bool, device=query.device).tril()

  

# scores = Q @ K^T / sqrt(head_dim)

scores = query @ key.transpose(-2, -1) * self.scale

  

scores=scores.masked_fill(mask, float("-inf")) #掩码

  

out=F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype) @ value

  

return out.reshape(B, H, S, D).transpose(1,2)

  
  

if __name__ == "__main__":

torch.manual_seed(42)

  

B, S, H, Hkv, D=1, 1024, 16, 8, 128

# device="cuda"

  

q=torch.randn(B, S, H, D)

k=torch.randn(B, S, Hkv, D)

  

attn=Attention(H, D, Hkv)

  

out=attn(q,k,k)

  

reference=F.scaled_dot_product_attention(

q.transpose(1,2),k.transpose(1,2), k.transpose(1,2),

is_causal=True, scale=D ** -0.5, enable_gqa=True,

).transpose(1,2)

  

print(torch.allclose(out, reference, atol=1e-4, rtol=1e-3))
```