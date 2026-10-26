

![[Pasted image 20261014204754.png]]

BN和LN都基于同一个计算公式：
![[Pasted image 20261015110056.png]]
只是计算出发的维度不同

## Batch Norm批次归一化
在batch的维度上进行归一化，
即**固定特征，对一个batch内的所有样本进行计算**

### 代码说明
```python
bn_input=torch.rand((2,3,4,4)) # （样本数，特征数，高，宽）

  
def batch_norm(x:torch.Tensor, eps=1e-05)-> torch.Tensor:

	alpha=torch.nn.Parameter(torch.ones(x.shape[1]))
	
	beta=torch.nn.Parameter(torch.zeros(x.shape[1]))
	
	  
	# 选择维度（0，2，3），代表固定特征维度，计算norm
	mean=torch.mean(x, dim=(0,2,3), keepdim=True)
	
	var=torch.var(x, dim=(0,2,3), keepdim=True, unbiased=False)
	
	  
	
	output=(x-mean)/torch.sqrt(var+eps)
	
	output=output*alpha.view(1,-1,1,1)+beta.view(1,-1,1,1)
	
	return output
```

## LayerNorm
在特征维度上进行归一化
即**固定样本，对同一样本内的所有特征进行计算**

### 代码分析
```python
import torch

import torch.nn as nn

  

ln_input=torch.rand((2,3,4)) #(样本数，序列长度，特征数量)
  

def layer_norm(x: torch.Tensor, eps=1e-05) -> torch.Tensor:

alpha=nn.Parameter(torch.ones(x.shape[-1])) # 选择最后一个维度

beta=nn.Parameter(torch.zeros(x.shape[-1]))

  

mean=torch.mean(x, dim=-1, keepdim=True)

var=torch.var(x, dim=-1, keepdim=True, unbiased=False)

  

output=(x-mean)/torch.sqrt(var+eps)

output=output*alpha+beta

return output

  

ln=nn.LayerNorm(4)

print("torch LN output: \n", ln(ln_input))

print("Our Ln output: \n", layer_norm(ln_input))
```


## RMSNorm：LN砍掉减均值![[Pasted image 20261015112205.png]]

由对比公式可知：
LN为：去均值+除以标准差
RMS为：直接除以均方根

### 为什么去掉均值还能work？

1. **均值平移不变性在实践中不重要**。
   理论上 LN 对"输入加常数"不敏感（减均值抵消了），但经验发现在深层网络里，这一性质带来的收益很小——尤其是 LN 后面本来就跟一个可学习的 β 平移项，均值信息可以部分被吸收。
    
2. **换来的是实打实的效率**。少
	3算一次均值、少一轮减法/广播，前向和反向都快约 7%~64%（原论文数据，视实现而定）。在大模型动辄数十亿的归一化调用次数下，这个节省很可观。
    
3. **缩放不变性是核心，保住它就够**。
	4真正重要的是对特征"幅度/尺度"的归一化——除 RMS 把向量长度归一，这个性质 LN 和 RMSNorm 都有。

### 代码分析
```python
class RMSNorm(nn.Module):

def __init__(self, hidden_size:int, eps:float=1e-6):

super().__init__()

  

self.weight=nn.Parameter(torch.ones(hidden_size))

self.eps=eps

  
  

def forward(self, x:torch.Tensor, )-> torch.Tensor:

ori_dtype=x.dtype #保留精度

  

x32=x.float()

  
#mean的维度选择-1 dim：表示和LN一样在特征维度计算norm
rms=torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True)+self.eps) 


x_normed=x32*rms

out32=x_normed*self.weight.float()

  

return out32.to(ori_dtype)
```