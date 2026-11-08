
## Nano项目的schedule调度过程
![[Scheduler调度过程.excalidraw]]
### scheduler.schedule的简单总结
1. 扫描waiting和running中，状态为finished的seq
	将这些seq移动到finished队列
	（seq状态机优化的和改进 防止超长prompt和用户abort）

2. 如果waiting队列中有seq，默先进行prefill
	否则再进行decode
	一次schedule，永远不混合pd

3. 某个seq进行prefill完成，则移动到running队列

4. 某个seq如果decode过程需要append新的block
	有则添加
	没有则preempt抢占waiting队尾的seq
	释放这个seq的KV Cache


## 测例运行数据检测
### 场景构造

- 先放 128 条短请求（prompt 128 tok，生成 256 tok），step 到全部进入稳态 decode；
- 之后每 20 个 engine step 注入一条 3072 tok 的长 prompt，默认共 5 条；
- 手动驱动 `LLMEngine.step()`，每步前后 `cuda.synchronize`，记录 step / token 落地时间。

### 输出指标

短请求 TBT（相邻 completion token 间隔）分桶：

- `baseline`：第一条长请求到达前，纯 decode
- `quiet`：未跨过 prefill 步的 TBT
- `interrupted`：跨过长 prompt prefill 步的 TBT，也就是被打断的尖峰
- `all`：稳态内全部样本

每个桶输出 `n / p50 / p99 / max`，并打印：

- `p99 - p50`
- `p99 / p50`
- baseline 的 `p99 / p50` 对比
- 长请求逐条 TTFT、p50 / p99、prefill 被切成几步

同时写 JSONL trace（`meta / request / step / preempt / finish / summary`），可以脱离 bench 代码重新算 TBT / TTFT。KV 不足导致抢占时，会把重算空档排除并给出警告。


### 运行结果分析
```bash
  稳态起始 step: 1 | 稳态步数: 261 | 长请求已注入: 5/5 | 结束原因: finished
  稳态 step: decode=255 prefill=5 empty=1
  decode step 耗时: p50=170.50ms p99=220.32ms | prefill step 耗时: p50=461.17ms p99=1273.71ms

  短请求 TBT（token 落地间隔，ms）
    baseline（首条长请求前，纯 decode）: n=2432 p50=109.45 p99=120.15 max=120.15
    quiet（所有未跨 prefill 的间隔）: n=31872 p50=171.70 p99=220.40 max=221.13
    interrupted（跨过 prefill 步）: n=640 p50=612.31 p99=1453.62 max=1453.62
    all（稳态内全部，排除被抢占的样本）: n=32512 p50=172.43 p99=612.31 max=1453.62
    -> all: p99 - p50 = 439.88ms，p99 / p50 = 3.55x
    -> baseline: p99 / p50 = 1.10x

  长请求 TTFT（add_request -> 第一个 token，ms）
    seq132: submit_step=21 first_token_step=21 prefill_steps=1 TTFT=1307.57ms finish=length
    seq133: submit_step=41 first_token_step=41 prefill_steps=1 TTFT=461.25ms finish=length
    seq134: submit_step=61 first_token_step=61 prefill_steps=1 TTFT=463.11ms finish=length
    seq135: submit_step=81 first_token_step=81 prefill_steps=1 TTFT=461.08ms finish=length
    seq136: submit_step=101 first_token_step=101 prefill_steps=1 TTFT=460.52ms finish=length
    -> n=5 p50=461.25ms p99=1273.79ms max=1307.57ms

  请求: 133/133 完成（short=128 long=5），未完成=0，被抢占=0
```

可以看到decode阶段的p99达到220ms，远大于p50的170ms

很容易推测：
	**当前的prefill机制，会导致prefill长seq时，**
	**严重阻塞decode的TBT指标**

## 解决和优化
当前prefill对于decode速度的两个最大影响因素，是：
- scheduler对prefill优先的强限制
	导致每一轮step中，waiting有seq
	就会不断阻塞decode进度

- decode时需要抢占prefill的KV Cache
	token产出时间会加上Cache的释放时间

### interval尝试点：减少prefill的调度频率
设置prefill_interval为某个值，
只有当`decode_step % interval ==0 `时，
才尝试进行prefill，否则默认decode


### watermark：减少prefill被抢占的概率，减少decode抢占
设置block使用量的水位线，
轮到prefill的step时，如果当前Block用量已经达到水位线了，
则继续decode

### 优化效果数据

```bash
  稳态起始 step: 1 | 稳态步数: 258 | 长请求已注入: 5/5 | 结束原因: finished
  稳态 step: decode=255 prefill=2 empty=1
  decode step 耗时: p50=166.12ms p99=212.72ms | prefill step 耗时: p50=1122.60ms p99=1334.63ms

  短请求 TBT（token 落地间隔，ms）
    baseline（首条长请求前，纯 decode）: n=2432 p50=105.29 p99=106.55 max=106.55
    quiet（所有未跨 prefill 的间隔）: n=32256 p50=155.02 p99=212.85 max=214.47
    interrupted（跨过 prefill 步）: n=256 p50=1305.43 p99=1493.00 max=1493.00
    all（稳态内全部，排除被抢占的样本）: n=32512 p50=166.40 p99=214.47 max=1493.00
    -> all: p99 - p50 = 48.07ms，p99 / p50 = 1.29x
    -> baseline: p99 / p50 = 1.01x

  长请求 TTFT（add_request -> 第一个 token，ms）
    seq132: submit_step=21 first_token_step=71 prefill_steps=1 TTFT=6607.58ms finish=length
    seq133: submit_step=41 first_token_step=71 prefill_steps=1 TTFT=4503.26ms finish=length
    seq134: submit_step=61 first_token_step=71 prefill_steps=1 TTFT=2393.54ms finish=length
    seq135: submit_step=81 first_token_step=142 prefill_steps=1 TTFT=10089.59ms finish=length
    seq136: submit_step=101 first_token_step=142 prefill_steps=1 TTFT=7252.08ms finish=length
    -> n=5 p50=6607.58ms p99=9976.09ms max=10089.59ms

  请求: 133/133 完成（short=128 long=5），未完成=0，被抢占=0
```
