# 05. Optimizer、Distributed DDP 和 Distributed Optimizer 带读

这一节接在 Pipeline Parallel 与 1F1B schedule 后面。

前五节你已经读到：

```text
GPT model
-> TransformerBlock / TransformerLayer
-> SelfAttention / MLP
-> Tensor Parallel
-> Pipeline Parallel 1F1B forward/backward
```

现在还缺一块：

```text
backward 已经把梯度算出来了
这些梯度怎么在 Data Parallel ranks 之间同步？
同步后的梯度怎么被 optimizer 消费？
```

这一节只读普通 DDP 梯度同步主线。先不要展开：

- distributed optimizer state sharding
- FSDP
- CPU offload
- MoE expert parallel 的完整细节
- FP8 / Transformer Engine 细节
- layer-wise optimizer

建议打开这些文件对照读：

- `megatron/training/training.py`
- `megatron/core/pipeline_parallel/schedules.py`
- `megatron/core/distributed/distributed_data_parallel.py`
- `megatron/core/distributed/param_and_grad_buffer.py`
- `megatron/core/distributed/finalize_model_grads.py`
- `megatron/core/optimizer/optimizer.py`

---

## A. 不看代码版：Optimizer 和 Distributed DDP 运行过程

这一节解决的问题是：多张 GPU 各自算了梯度，怎么把它们同步成一致的模型更新。

Data Parallel 下，每个 DP rank 都有一份相同模型，但拿到不同数据：

```text
rank0: batch shard 0 -> forward/backward -> grads0
rank1: batch shard 1 -> forward/backward -> grads1
rank2: batch shard 2 -> forward/backward -> grads2
```

这些梯度必须合并，否则每张卡会把模型更新到不同方向。普通 DDP 的核心动作是 all-reduce：

```text
grads0, grads1, grads2
  -> all-reduce / average
  -> 每个 rank 得到相同 grads
  -> optimizer step
  -> 每个 rank 参数继续一致
```

Megatron 会把参数和梯度组织进连续 buffer，再按 bucket 同步。这样做是为了减少碎片化的小通信，把很多小参数的梯度合并成较大的通信单位。

训练 step 可以简化成：

```text
forward 得到 loss
  -> backward 产生梯度
  -> 梯度进入 main_grad / bucket
  -> DDP 同步梯度
  -> finalize_model_grads 做收口处理
  -> optimizer 消费梯度并更新参数
  -> zero_grad 清理下一轮状态
```

如果有 microbatch 梯度累积，中间若干个 microbatch 先不急着同步，最后再统一同步。

不看代码时记住：DDP 保证“同一个 DP group 内，每张卡看到不同数据，但更新同一份模型”。

## B. 代码带读版：Optimizer 和 Distributed DDP 实现路径

### B1. 先把问题摆清楚

Pipeline schedule 负责组织 forward/backward：

```text
microbatch 0 forward
microbatch 1 forward
microbatch 0 backward
microbatch 2 forward
microbatch 1 backward
...
```

但是 backward 只是在当前 rank 上算出了本地梯度。

如果 Data Parallel size 大于 1，不同 DP rank 看到的是不同 data batch。它们各自 backward 后的梯度需要被同步，否则每个 DP replica 会走向不同参数。

所以这一节的主线是：

```text
local backward
-> param.grad
-> param.main_grad
-> ParamAndGradBuffer / bucket
-> all-reduce 或 reduce-scatter
-> finalize_model_grads
-> MegatronOptimizer.step()
```

读这一节时，先把它当成一个普通 DDP 系统：

```text
模型计算由 TP/PP 切开
梯度一致性由 DP/DDP 负责
参数更新由 optimizer 负责
```

---

### B2. 训练入口把 DDP 钩到 schedule 上

先看 `megatron/training/training.py` 里训练开始前对 `config` 的设置。

关键关系是：

```python
config.grad_scale_func = optimizer.scale_loss if optimizer is not None else None
...
config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
...
config.finalize_model_grads_func = finalize_model_grads
```

这几行很关键，因为 Pipeline schedule 不直接知道 DDP 的内部实现。

它只拿到几个函数：

```text
grad_scale_func
no_sync_func
grad_sync_func
finalize_model_grads_func
```

它们的作用分别是：

```text
grad_scale_func:
  forward loss 反传前做 loss scaling

no_sync_func:
  梯度累积期间暂时不触发 DP 同步

grad_sync_func:
  需要对齐 overlap grad reduce 时，提前启动梯度同步

finalize_model_grads_func:
  整个 forward-backward 结束后，把梯度同步和一些特殊梯度 all-reduce 收口
```

这里先记住一个设计：

```text
training.py 负责把 DDP/optimizer 的函数塞进 config
schedules.py 负责在正确时机调用这些函数
distributed/ 目录负责真正做梯度通信
optimizer/ 目录负责真正更新参数
```

---

### B3. schedule 在哪里调用 finalize

上一节读过 `forward_backward_pipelining_without_interleaving`。

在 `megatron/core/pipeline_parallel/schedules.py` 里，forward/backward 都跑完之后，会出现这样的逻辑：

```python
if config.finalize_model_grads_func is not None and not forward_only:
    config.finalize_model_grads_func(...)
```

这说明 DDP 梯度同步不是 schedule 主循环的主要逻辑。

主循环负责：

```text
recv forward
forward
send forward / recv backward
backward
send backward
```

等这些都完成后，再由 `finalize_model_grads` 做梯度收口。

如果启用了 overlap grad reduce，部分 bucket 的通信可能已经在 backward hook 中提前发出。此时 finalize 的职责就变成：

```text
等待或补发还没完成的 grad sync
处理 TP / PP / embedding / sequence parallel 相关的特殊梯度同步
按 token 数做最终缩放
```

所以不要把 `finalize_model_grads` 理解成“开始同步所有梯度”的唯一入口。

更准确地说：

```text
它是梯度同步的收口点。
```

---

### B4. DDP wrapper 做了什么

进入 `megatron/core/distributed/distributed_data_parallel.py`。

主类是：

```python
class DistributedDataParallel(_BaseDataParallel):
```

它不是 PyTorch 原生 `torch.nn.parallel.DistributedDataParallel` 的简单封装，而是 Megatron Core 自己的 DDP wrapper。

它的核心目标是：

```text
把参数的梯度放进连续 buffer
按 bucket 切分
在合适时机做 all-reduce 或 reduce-scatter
支持通信与 backward 计算重叠
```

初始化里先收集所有可训练参数：

```python
for name, param in self.module.named_parameters():
    if not param.requires_grad:
        continue
    self.params_with_grad.append(param)
    param.grad_added_to_main_grad = False
    param_to_name[param] = name
    all_params.append(param)
```

注意这里不是直接依赖 `param.grad` 到最后。

Megatron 会给参数准备 `main_grad`，让梯度累积和通信 buffer 更可控。

可以先把它理解成：

```text
param.grad:
  PyTorch autograd 本次 backward 产生的临时梯度

param.main_grad:
  Megatron 用来累计 microbatch 梯度、并交给 optimizer 使用的主梯度
```

---

### B5. 参数为什么要放进 buffer

DDP 初始化会调用：

```python
buffer_groups = group_params_for_buffers(...)
```

然后为每组参数创建：

```python
buffer = _ParamAndGradBuffer(...)
```

这些 buffer 的意义是：

```text
不要为每个 parameter 单独发一个通信
把很多 parameter 的 grad 连续放到一大块内存里
再按 bucket 组织通信
```

这带来几个好处：

```text
通信 kernel 数量更少
内存访问更连续
可以按 bucket 早发通信，从而和后续 backward 重叠
```

读 `group_params_for_buffers` 时，先看它按哪些维度分组：

```text
param dtype
grad dtype
是否 expert parallel
是否由 layer-wise optimizer 管理
```

本轮先只理解普通 dense 参数：

```text
同一类 dtype 的普通参数
-> 放进一个或多个 ParamAndGradBuffer
-> 每个 buffer 里再切 bucket
```

---

### B6. bucket 是梯度同步的基本单位

在 `param_and_grad_buffer.py` 里，`_ParamAndGradBuffer` 会为参数建立布局。

你可以把它想成：

```text
grad_data: 一整块连续梯度内存

param A main_grad -> grad_data 的一段 view
param B main_grad -> grad_data 的一段 view
param C main_grad -> grad_data 的一段 view
...
```

bucket 是这块连续内存中的一段。

DDP 初始化时会把 buffer 进一步分到 `bucket_groups`：

```python
self.bucket_groups = partition_buckets(...)
```

后面真正通信时，常见调用链是：

```text
DistributedDataParallel.finish_grad_sync
-> bucket_group.finish_grad_sync
-> bucket_group.start_grad_sync
-> all_reduce 或 reduce_scatter
```

这一层最重要的概念不是具体 layout 细节，而是：

```text
optimizer 读的是 main_grad
main_grad 常常是连续 grad buffer 的 view
bucket 通信改变的是 grad buffer
因此通信完成后，optimizer 看到的就是同步后的梯度
```

---

### B7. backward hook 如何把梯度搬进 main_grad

继续看 `DistributedDataParallel.__init__` 后半段。

它会给每个可训练参数注册 backward hook：

```python
grad_acc.register_hook(self._make_backward_post_hook(param))
```

hook 的核心逻辑在 `_make_backward_post_hook`：

```python
if param.grad is not None and (
    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
):
    param.main_grad.add_(param.grad.data)
param.grad = None

if self.ddp_config.overlap_grad_reduce:
    self.param_to_bucket_group[param].register_grad_ready(...)
```

这里发生了三件事：

```text
1. PyTorch autograd 算出 param.grad
2. hook 把 param.grad 累加到 param.main_grad
3. 清空 param.grad，避免保留很多临时梯度
```

如果开启 `overlap_grad_reduce`，hook 还会通知 bucket：

```text
这个 param 的梯度 ready 了
```

等一个 bucket group 里的参数梯度都 ready 后，就可以提前发起通信。

这就是“梯度通信和 backward 计算重叠”的基础。

---

### B8. no_sync 控制 microbatch 梯度累积

Pipeline 训练通常会有多个 microbatch。

如果每个 microbatch backward 后都立刻做 DP all-reduce，通信会太频繁，而且语义也不一定是想要的梯度累积。

所以 DDP 提供：

```python
def no_sync(self):
```

它会把 bucket group 的：

```python
is_last_microbatch = False
```

等退出 context 后再恢复：

```python
is_last_microbatch = True
```

对应到 `register_grad_ready`：

```python
if self.is_last_microbatch:
    ...
```

也就是说：

```text
不是最后一个 microbatch 时：
  只把 param.grad 累到 main_grad
  不触发 bucket ready 通信

最后一个 microbatch 时：
  累完 main_grad
  允许 bucket 进入 grad sync
```

这和上一节的 1F1B schedule 正好接上。

schedule 在做 microbatch backward 时，可以用 `no_sync_func` 控制什么时候允许 DDP 同步。

---

### B9. start_grad_sync 做什么

进入 `param_and_grad_buffer.py` 的 bucket group：

```python
def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
```

它的职责是发起通信。

核心步骤可以概括成：

```text
1. 检查是否已有未完成通信
2. 必要时把额外 main_grad copy 到通信 buffer
3. 检查 NaN/Inf 或过大梯度
4. 按 gradient_scaling_factor 缩放
5. 选择 all-reduce 或 reduce-scatter
6. 根据 overlap_grad_reduce 决定同步或异步通信
```

普通 DDP 主线先看 all-reduce：

```python
torch.distributed.all_reduce(bucket.grad_data, ...)
```

如果启用 distributed optimizer，则可能走 reduce-scatter：

```python
dist_reduce_scatter_func(...)
```

但这一节先不要展开 reduce-scatter 的 optimizer state sharding 含义。

先记住：

```text
普通 DDP:
  all-reduce 后每个 DP rank 都有完整同步梯度

Distributed optimizer:
  reduce-scatter 后每个 rank 可能只保留梯度 shard
```

这一节重点是普通 DDP，所以只要求读懂 all-reduce 路径。

---

### B10. finish_grad_sync 做什么

同一个文件里还有：

```python
def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
```

它的职责是确保通信完成。

如果没有开启 `overlap_grad_reduce`：

```python
if not self.ddp_config.overlap_grad_reduce:
    self.start_grad_sync(...)
    self._copy_back_extra_main_grads()
    return
```

也就是说：

```text
不 overlap:
  finish_grad_sync 现场发起同步通信，并等它完成
```

如果开启了 `overlap_grad_reduce`，通信可能已经在 backward hook 中发出。

这时 `finish_grad_sync` 主要做：

```text
等待异步通信 handle
清理 handle
必要时把通信 buffer 的结果 copy 回 main_grad
```

所以 `start` 和 `finish` 的关系是：

```text
start_grad_sync:
  发通信

finish_grad_sync:
  确保通信结束；如果还没发，就补发
```

---

### B11. DDP wrapper 的 finish_grad_sync 是分发层

`distributed_data_parallel.py` 里也有：

```python
def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
```

它不是直接做 all-reduce，而是遍历 bucket group：

```python
for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
    bucket_group.finish_grad_sync(force_all_reduce=force_all_reduce)
```

这一层的作用是：

```text
DDP wrapper:
  管整个 model/module 的所有 bucket groups

bucket group:
  管一组 buckets 的通信

bucket:
  对应连续 grad buffer 的一段
```

所以调用栈可以写成：

```text
finalize_model_grads(model)
-> model_chunk.finish_grad_sync()
-> DDP.finish_grad_sync()
-> bucket_group.finish_grad_sync()
-> bucket_group.start_grad_sync()
-> torch.distributed.all_reduce(bucket.grad_data)
```

这是本节最重要的一条调用链。

---

### B12. finalize_model_grads 是总收口

进入 `megatron/core/distributed/finalize_model_grads.py`。

主函数：

```python
def finalize_model_grads(model, num_tokens=None, pg_collection=None, force_all_reduce=False):
```

函数 docstring 直接说明了它做几件事：

```text
All-reduce all model grads across DP replicas
All-reduce layernorm grads for sequence parallelism
All-reduce embedding grads across first and last pipeline stages
Scale gradients by num_tokens
```

第一步就是：

```python
for model_chunk in model:
    model_chunk.finish_grad_sync(force_all_reduce=force_all_reduce)
```

这一步就是普通 DDP 梯度同步的收口。

后面还有几类特殊同步：

```text
conditional embedding grads
non tensor-parallel / layernorm grads
word embedding grads
position embedding grads
```

为什么还有这些？

因为 Megatron 里不是所有参数都只属于普通 DP 维度。

例如：

```text
sequence parallel 下，有些 LayerNorm 梯度需要跨 TP 相关 group 同步
pipeline parallel 下，embedding 可能在首尾 stage 间共享或关联
```

本节不展开它们的通信细节，只记住：

```text
finish_grad_sync 解决 DDP bucket 梯度同步
finalize_model_grads 还顺手处理 TP/PP 特殊梯度同步和 token-based scaling
```

---

### B13. num_tokens 为什么会进 finalize

`finalize_model_grads` 还有一个参数：

```python
num_tokens
```

这和 loss normalization 有关。

如果按 token 数计算 loss，而每个 microbatch 或 rank 的有效 token 数可能不同，那么最终梯度需要按全局 token 数做一致缩放。

所以 finalize 阶段适合做这件事：

```text
forward/backward 已经结束
各 rank 的 token 统计也能汇总
梯度同步也在这里收口
```

这一点先不用深究。

只要知道：

```text
DDP 同步解决的是不同 DP rank 的梯度一致性
num_tokens scaling 解决的是 loss/梯度归一化口径一致性
```

---

### B14. optimizer step 消费什么梯度

进入 `megatron/core/optimizer/optimizer.py`。

基类是：

```python
class MegatronOptimizer(ABC):
```

它定义了几类关键接口：

```python
prepare_grads()
step_with_ready_grads()
zero_grad()
step()
```

普通训练步大致是：

```text
forward/backward
-> finalize_model_grads
-> optimizer.step
-> optimizer.zero_grad
```

optimizer 不负责帮你做 DDP 梯度同步。

它默认消费的是已经准备好的梯度。

在 Megatron 里，这些梯度通常已经在：

```text
param.main_grad
```

或者 optimizer 内部对应的 main parameter / grad 结构里。

所以你要把职责边界分清：

```text
DDP:
  让梯度在 DP ranks 间同步

Optimizer:
  检查梯度、clip、更新参数、维护 optimizer state
```

---

### B15. prepare_grads 和 step 的职责

`MegatronOptimizer` 抽象类里有：

```python
def prepare_grads(self) -> bool:
```

和：

```python
def step_with_ready_grads(self) -> bool:
```

名字很直白：

```text
prepare_grads:
  在真正 step 前处理梯度，比如 unscale、检查 inf/nan、clip grad、统计 norm

step_with_ready_grads:
  假设梯度已经准备好，调用底层 torch optimizer 更新参数
```

不同 optimizer 子类会根据精度和参数形态实现这些函数。

本节只建立主线：

```text
DDP 同步后的 main_grad
-> optimizer.prepare_grads
-> optimizer.step_with_ready_grads
-> 参数更新
```

如果你现在直接跳到 `distrib_optimizer.py`，会看到大量 param shard、grad shard、range map、bucket map，容易把主线淹没。

所以这里先停在 `optimizer.py` 的通用接口。

---

### B16. zero_grad 清理的是下一轮状态

optimizer 还有：

```python
zero_grad(set_to_none=True)
```

普通 PyTorch 里，`zero_grad` 多数是清 `param.grad`。

Megatron 里因为使用 `main_grad` 和连续 buffer，清理对象会更复杂。

但从训练循环角度看，它的作用还是：

```text
当前 step 已经用完梯度
清理梯度累计区
准备下一轮 forward/backward
```

结合前面 backward hook 的逻辑：

```text
backward hook:
  param.grad -> param.main_grad
  param.grad = None

optimizer.zero_grad:
  清理 main_grad / buffer 中的累计值
```

所以不要只盯着 `param.grad`。

在 Megatron 中，真正长期持有梯度的往往是：

```text
main_grad / grad buffer
```

---

### B17. overlap_grad_reduce 的两种路径

现在可以回头整理 `overlap_grad_reduce`。

#### 不开启 overlap

```text
每个 microbatch backward:
  param.grad -> param.main_grad

forward-backward 全部结束:
  finalize_model_grads
  -> finish_grad_sync
  -> start_grad_sync
  -> all_reduce
  -> optimizer.step
```

这种路径容易理解：

```text
先算完所有梯度
再统一通信
```

#### 开启 overlap

```text
backward hook:
  param.grad -> param.main_grad
  register_grad_ready

bucket ready:
  start_grad_sync async

forward-backward 结束:
  finalize_model_grads
  -> finish_grad_sync
  -> wait async handle
  -> optimizer.step
```

这种路径性能更好，但读代码更绕。

建议学习时先按不 overlap 的路径读一遍，再读 overlap。

---

### B18. 一条完整调用链

把本节主线串起来：

```text
training.py
  set config.finalize_model_grads_func = finalize_model_grads
  set config.no_sync_func = model_chunk.no_sync

schedules.py
  forward_backward_pipelining_without_interleaving
  -> backward_step
  -> autograd backward

DistributedDataParallel backward hook
  param.grad -> param.main_grad
  param.grad = None
  maybe register_grad_ready

schedules.py
  forward/backward 全部结束
  -> config.finalize_model_grads_func(...)

finalize_model_grads.py
  for model_chunk in model:
      model_chunk.finish_grad_sync()

DistributedDataParallel.finish_grad_sync
  for bucket_group:
      bucket_group.finish_grad_sync()

ParamAndGradBuffer bucket group
  start_grad_sync if needed
  all_reduce(bucket.grad_data)
  wait handle if async

optimizer.py
  optimizer.step()
  optimizer.zero_grad()
```

这条链路读顺后，你就能回答：

```text
backward 算出来的梯度在哪里？
什么时候从 param.grad 变成 main_grad？
什么时候通信？
谁等待通信完成？
optimizer 用的是哪个梯度？
```

---

### B19. 读代码时的检查点

建议你带着这些问题复读：

```text
1. DistributedDataParallel 初始化时，哪些 parameter 被收集？

2. group_params_for_buffers 按什么维度把参数分组？

3. _ParamAndGradBuffer 里的 grad_data 和 param.main_grad 是什么关系？

4. backward hook 为什么要把 param.grad 加到 param.main_grad 后清空？

5. no_sync 如何避免每个 microbatch 都触发通信？

6. overlap_grad_reduce=False 时，通信在哪一步发生？

7. overlap_grad_reduce=True 时，通信在哪一步提前发出？

8. finalize_model_grads 除了 DDP grad sync，还处理哪些特殊梯度？

9. optimizer.step 前，为什么可以假设梯度已经同步好？

10. zero_grad 清理的是 param.grad，还是 main_grad / buffer？
```

如果这些问题能答出来，这一节就算读通了。

---

### B20. 第六轮总图

可以把目前已经读过的训练主线画成这样：

```text
tokens
  |
  v
GPTModel.forward
  |
  v
TransformerBlock
  |
  v
TransformerLayer
  |
  +--> SelfAttention
  |      |
  |      +--> QKV projection
  |      +--> RoPE
  |      +--> DotProductAttention
  |      +--> output projection
  |
  +--> MLP
  |
  +--> bias-dropout-add / layernorm
  |
  v
Pipeline schedule
  |
  +--> warmup
  +--> 1F1B
  +--> cooldown
  |
  v
backward
  |
  v
DDP backward hook
  |
  +--> param.grad -> param.main_grad
  +--> bucket ready
  |
  v
finalize_model_grads
  |
  +--> finish_grad_sync
  +--> all-reduce / wait async reduce
  +--> special TP/PP grad sync
  |
  v
MegatronOptimizer.step
  |
  v
updated parameters
```

这张图把计算、通信、更新三件事分开了：

```text
TP/PP:
  负责把模型计算切开

DDP:
  负责让梯度在 data parallel replicas 间一致

Optimizer:
  负责用同步后的梯度更新参数
```

---

### B21. 下一步带读建议

下一节建议进入：

```text
Distributed Optimizer
```

但是不要一上来就读完整 `distrib_optimizer.py`。

建议第七节只解决一个问题：

```text
普通 DDP 是 all-reduce 完整梯度；
distributed optimizer 为什么要 reduce-scatter 梯度，
又如何让每个 rank 只更新自己负责的参数 shard？
```

建议先看：

- `megatron/core/optimizer/distrib_optimizer.py`
- `megatron/core/optimizer/param_layout.py`
- `megatron/core/distributed/param_and_grad_buffer.py`

目标是理解：

```text
param range
grad shard
main param shard
optimizer state shard
```

等这条线读通，再去看 checkpoint、FP8、CPU offload、layer-wise optimizer。

---

## C. Distributed Optimizer 带读

这一节接在第 05 节的普通 DDP 梯度同步后面。

第 05 节你已经读到：

```text
backward
-> param.main_grad
-> ParamAndGradBuffer / bucket
-> all-reduce
-> MegatronOptimizer.step()
```

现在要把一个问题单独拆出来：

```text
普通 DDP 已经能让 DP ranks 的梯度一致；
为什么 Megatron 还要 Distributed Optimizer？
```

这一节只读 Distributed Optimizer 的最小闭环。先不要展开：

- distributed checkpointing
- FP8 / NVFP4 参数 gather 细节
- CPU offload
- layer-wise optimizer
- Megatron-FSDP
- 多 distributed optimizer instance

建议打开这些文件对照读：

- `megatron/core/distributed/param_and_grad_buffer.py`
- `megatron/core/optimizer/distrib_optimizer.py`
- `megatron/core/optimizer/optimizer.py`

---

## A. 不看代码版：Distributed Optimizer 运行过程

Distributed Optimizer 解决的问题是：普通 DDP 每张卡都保存完整 optimizer state，显存浪费很大。

普通 DDP 的状态更像这样：

```text
每个 DP rank:
  完整参数
  完整梯度
  完整 optimizer state
```

Distributed Optimizer 把 optimizer 相关状态按 DP rank 切片：

```text
rank0 负责一段参数 shard 的 optimizer state
rank1 负责另一段参数 shard 的 optimizer state
rank2 负责另一段参数 shard 的 optimizer state
```

梯度同步也从 all-reduce 完整梯度，变成 reduce-scatter：

```text
所有 rank 的梯度
  -> reduce-scatter
  -> 每个 rank 只拿到自己负责的梯度 shard
  -> 只更新自己负责的参数 shard
```

更新完后，下一轮 forward 又需要完整模型参数参与计算，所以会在需要时把参数 shard all-gather 回来。

整体流程是：

```text
backward 产生完整梯度贡献
  -> reduce-scatter 得到 grad shard
  -> optimizer 只更新本 rank 的 param shard
  -> all-gather 参数供下一轮 forward 使用
```

不看代码时记住：Distributed Optimizer 不是换优化算法，而是把 optimizer state 和梯度更新责任分摊到 DP ranks 上，主要目的是省显存。

## B. 代码带读版：Distributed Optimizer 实现路径

### B1. 先把普通 DDP 的问题说清楚

普通 DDP 的数据并行逻辑很直接：

```text
DP rank 0: full params + full grads + full optimizer states
DP rank 1: full params + full grads + full optimizer states
DP rank 2: full params + full grads + full optimizer states
DP rank 3: full params + full grads + full optimizer states
```

每个 DP rank 都有一份完整模型参数。backward 后，每个 rank 算出自己 batch 上的本地梯度。

然后 DDP 做 all-reduce：

```text
local full grad
-> all-reduce across DP group
-> synced full grad on every DP rank
```

这样做的好处是简单：每个 rank 都拿到完整同步梯度，然后本地 optimizer 更新完整参数。

问题也很明显：optimizer state 被重复存了很多份。

以 Adam 为例，一个参数通常至少关联：

```text
model param
main fp32 param
exp_avg
exp_avg_sq
```

如果 DP size 是 8，普通 DDP 会让这几类 optimizer 相关状态在 8 个 DP rank 上各存一份。

Distributed Optimizer 要解决的是这件事：

```text
不要让每个 DP rank 都更新完整参数；
每个 DP rank 只负责一片参数 shard 和对应 optimizer state shard。
```

---

### B2. all-reduce 和 reduce-scatter 的区别

先把通信语义拆开。

普通 DDP all-reduce 可以理解为：

```text
输入:
  rank 0: grad[0:N]
  rank 1: grad[0:N]
  rank 2: grad[0:N]
  rank 3: grad[0:N]

输出:
  rank 0: reduced_grad[0:N]
  rank 1: reduced_grad[0:N]
  rank 2: reduced_grad[0:N]
  rank 3: reduced_grad[0:N]
```

每个 rank 最后都拿到完整结果。

Distributed Optimizer 更关心 reduce-scatter：

```text
输入:
  rank 0: grad[0:N]
  rank 1: grad[0:N]
  rank 2: grad[0:N]
  rank 3: grad[0:N]

输出:
  rank 0: reduced_grad[0:N/4]
  rank 1: reduced_grad[N/4:N/2]
  rank 2: reduced_grad[N/2:3N/4]
  rank 3: reduced_grad[3N/4:N]
```

reduce-scatter 做了两件事：

```text
reduce:
  把不同 DP rank 的梯度相加或平均

scatter:
  每个 DP rank 只保留 reduce 后的一片 shard
```

所以 Distributed Optimizer 的关键变化是：

```text
普通 DDP:
  每个 rank 得到完整同步梯度

Distributed Optimizer:
  每个 rank 得到自己负责的同步梯度 shard
```

---

### B3. 从 `shard_buffer` 开始读

先看 `megatron/core/distributed/param_and_grad_buffer.py` 里的 `shard_buffer`：

```python
def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer
```

这段代码只做一件事：

```text
把一个连续 buffer 按 DP world size 均分。
```

假设一个 bucket 的 `grad_data` 有 16 个元素，DP size 是 4：

```text
bucket.grad_data:
  [0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15]

rank 0 shard: [0 1 2 3]
rank 1 shard: [4 5 6 7]
rank 2 shard: [8 9 10 11]
rank 3 shard: [12 13 14 15]
```

注意这里切的是 bucket buffer，不是按参数边界切。

这点很重要，因为一个 shard 可能刚好落在某个参数中间：

```text
param A: [0 1 2 3 4 5]
param B: [6 7 8 9 10 11 12 13]
param C: [14 15]

rank 1 shard: [4 5 6 7]
  -> 包含 param A 的尾部
  -> 包含 param B 的头部
```

所以 Distributed Optimizer 后面需要一套 range map，把 bucket shard 映射回具体参数里的局部区间。

---

### B4. 梯度同步从 all-reduce 变成 reduce-scatter

继续看 `_ParamAndGradBucketGroup.start_grad_sync()`。

这段逻辑会根据 `ddp_config.use_distributed_optimizer` 选择通信方式：

```text
use_distributed_optimizer = False
  -> all_reduce(bucket.grad_data)

use_distributed_optimizer = True
  -> reduce_scatter(local_data_view, bucket.grad_data)
```

核心代码路径是：

```python
if self.ddp_config.use_distributed_optimizer and not force_all_reduce:
    local_data_view = shard_buffer(bucket.grad_data, dp_size)[dp_rank]
    dist_reduce_scatter_func(
        local_data_view,
        bucket.grad_data,
        op=reduce_op,
        group=communication_group,
        async_op=async_op,
    )
else:
    torch.distributed.all_reduce(bucket.grad_data, ...)
```

读这段时重点看两个张量：

```text
bucket.grad_data:
  输入，是当前 rank 上完整 bucket 梯度 buffer

local_data_view:
  输出，是 bucket.grad_data 中当前 DP rank 对应的 shard view
```

也就是说，reduce-scatter 完成后，当前 rank 只保证自己的 `local_data_view` 是同步后的结果。

这就是 Distributed Optimizer 可以只更新 shard 的前提。

---

### B5. `Range` 是这条线的核心小工具

接着进入 `megatron/core/optimizer/distrib_optimizer.py`。

先看 `Range`：

```python
class Range:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.size = end - start
```

这里没有复杂抽象，只是用 `[start, end)` 表示一段连续区间。

Distributed Optimizer 需要同时描述几种区间：

```text
gbuf_world:
  这个 shard 在完整 grad buffer 里的位置

gbuf_world_in_bucket:
  这个 shard 在当前 bucket 里的位置

gbuf_local:
  这个 shard 在当前 rank local shard view 里的位置

param:
  这个 shard 在原始参数自身 flatten 后的位置
```

你可以把它们理解成同一段数据在不同坐标系下的名字。

---

### B6. `_build_gbuf_range_map`：当前 rank 拥有哪些 shard

`DistributedOptimizer.__init__` 里会为每个 `_ParamAndGradBuffer` 建立 range map：

```python
self.gbuf_ranges.append(self._build_gbuf_range_map(buffer))
self.model_param_gbuf_map = self._build_model_param_gbuf_map(self.gbuf_ranges)
```

`_build_gbuf_range_map()` 会遍历 buffer 里的每个 bucket：

```python
return {
    (param_dtype, grad_dtype): [
        cls._build_model_gbuf_range(param_and_grad_buffer, bucket_index)
        for bucket_index in range(len(param_and_grad_buffer.buckets))
    ]
}
```

真正算当前 rank shard 区间的是 `_build_model_gbuf_range()`：

```text
gbuf_size = bucket.grad_data.numel()
max_gbuf_range_size = gbuf_size // data_parallel_world_size

rank r owns:
  [r * max_gbuf_range_size, (r + 1) * max_gbuf_range_size)
```

再加上 bucket 在全局 grad buffer 里的 `bucket.offset`，就得到当前 rank 在整个 grad buffer 坐标里的 `gbuf_world_range`。

这一层回答的问题是：

```text
当前 DP rank 在每个 bucket 里负责哪一段？
```

---

### B7. `_build_model_gbuf_param_range_map`：把 bucket shard 映射回参数

现在有了当前 rank 负责的 bucket range，还要知道这段 range 覆盖了哪些参数。

`_build_model_gbuf_param_range_map()` 会遍历 `param_world_index_map`：

```text
param_world_index_map:
  param -> (param_world_start, param_world_end, bucket_id)
```

对每个参数，它会判断：

```text
这个参数的 world range
和当前 rank 的 gbuf_world_range
有没有交集？
```

如果有交集，就为这个参数建立四个 range：

```text
gbuf_world:
  交集在完整 grad buffer 里的位置

gbuf_world_in_bucket:
  交集在当前 bucket 里的位置

gbuf_local:
  交集在当前 rank local shard view 里的位置

param:
  交集在该参数自身 flatten 后的位置
```

举一个小例子：

```text
bucket:
  [0 ... 15]

DP size = 4
rank 1 owns:
  [4 ... 7]

param A:
  [0 ... 5]

param B:
  [6 ... 13]
```

rank 1 的 shard 覆盖：

```text
param A 的 [4 ... 5]
param B 的 [0 ... 1]
```

这就是 `param` range 要表达的东西。

---

### B8. `_build_model_and_main_param_groups`：optimizer 只看 shard

接着看 `_build_model_and_main_param_groups()`。

这个函数的注释很关键：

```text
the optimizer operates on shards of the model parameters,
rather than the full parameters.
```

它会根据前面算好的 `param_range`，为当前 rank 创建参数 shard：

```python
shard_model_param = model_param.detach().view(-1)[
    param_range.start : param_range.end
]
```

如果原始模型参数是 fp16 / bf16，它还会创建 fp32 main param shard：

```python
shard_main_param = shard_model_param.clone().float()
model_param.main_param = shard_main_param
model_param.main_param_sharded = True
```

最后，真正传给 inner optimizer 的不再是完整参数，而是 shard 参数组：

```python
group_range["orig_group"]["params"] = [
    *shard_fp32_params_this_group,
    *shard_fp32_from_float16_params_this_group,
]
```

这一层回答的问题是：

```text
为什么 optimizer state 也能被 shard？
```

因为 inner optimizer 看到的参数本身就是 shard：

```text
optimizer param:
  shard_main_param

optimizer state:
  exp_avg shard
  exp_avg_sq shard
```

optimizer 从来不需要为当前 rank 没有负责的参数片段创建 state。

---

### B9. step 前：把 grad shard 挂到 main param shard 上

读 `_copy_model_grads_to_main_grads()`。

它的注释已经把背景说清楚：

```text
Since this step follows a reduce-scatter through the DDP's grad buffer,
this method is responsible for copying the updated grads
from the grad buffer to the main shard's grad field.
```

核心逻辑是：

```python
param_range_map = self._get_model_param_range_map(model_param)
param_range = param_range_map["param"]

model_grad = model_param.main_grad
shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
shard_main_param.grad = shard_model_grad.float()
```

这里要注意：

```text
model_param.main_grad:
  从 DDP grad buffer 来，是当前参数对应的梯度视图

param_range:
  当前 DP rank 负责的那段参数区间

shard_main_param.grad:
  inner optimizer 真正消费的 shard 梯度
```

所以 step 前的数据流是：

```text
reduce-scatter 后的 grad buffer shard
-> model_param.main_grad 的局部区间
-> shard_main_param.grad
-> inner optimizer
```

---

### B10. step 后：把 shard 更新结果写回 param buffer

读 `_copy_main_params_to_model_params()`。

inner optimizer 更新的是 `shard_main_param`。但是下一轮 forward 需要模型参数可用，所以更新后的 shard 要写回参数 buffer 中正确位置。

核心逻辑是：

```python
param_range_map = self._get_model_param_range_map(model_param)
world_range = param_range_map["gbuf_world_in_bucket"]

model_param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data
shard_model_param = model_param_buffer.view(-1)[
    world_range.start : world_range.end
]
shard_model_param.data.copy_(shard_main_param)
```

这里使用的是 `gbuf_world_in_bucket`，因为要写回的是 bucket 的 `param_data` buffer。

写回后还没有结束。每个 rank 现在只更新了自己负责的参数 shard，所以还需要参数同步：

```text
updated local param shard
-> all-gather
-> every DP rank gets full updated params for next forward
```

这个 all-gather 在 `_ParamAndGradBucketGroup.start_param_sync()` 里：

```python
local_data_view = shard_buffer(bucket.param_data, dp_size)[dp_rank]
dist_all_gather_func(
    bucket.param_data,
    local_data_view,
    group=...,
)
```

这里和梯度 reduce-scatter 正好是反方向：

```text
grad sync:
  full grad buffer on each rank
  -> reduce-scatter
  -> local grad shard

param sync:
  local updated param shard
  -> all-gather
  -> full param buffer on each rank
```

---

### B11. 把整条链串起来

Distributed Optimizer 的一轮核心数据流是：

```text
forward
  |
  v
backward
  |
  v
local grads accumulated into ParamAndGradBuffer.grad_data
  |
  v
reduce-scatter grad_data
  |
  v
each DP rank owns local grad shard
  |
  v
_copy_model_grads_to_main_grads
  |
  v
shard_main_param.grad
  |
  v
inner optimizer.step()
  |
  v
updated shard_main_param
  |
  v
_copy_main_params_to_model_params
  |
  v
updated shard written into bucket.param_data
  |
  v
all-gather param_data
  |
  v
full updated params available for next forward
```

对比普通 DDP：

```text
普通 DDP:
  all-reduce full grads
  every rank updates full params
  every rank stores full optimizer states

Distributed Optimizer:
  reduce-scatter grad shards
  each rank updates param shards
  each rank stores optimizer state shards
  all-gather params before next forward
```

---

### B12. 四个关键词

读完这一节，至少要能解释这四个词。

```text
param range:
  当前 DP rank 负责的 shard 在原始参数 flatten 后的位置。

grad shard:
  reduce-scatter 后，当前 DP rank 拿到的同步梯度片段。

main param shard:
  当前 DP rank 持有并交给 optimizer 更新的 fp32 参数片段。

optimizer state shard:
  inner optimizer 为 main param shard 创建的 Adam 状态片段，
  比如 exp_avg shard 和 exp_avg_sq shard。
```

这四个词连起来就是：

```text
grad shard 对应 param range；
param range 对应 main param shard；
optimizer state shard 跟着 main param shard 走。
```

---

### B13. 这一节先不要被分支带偏

`distrib_optimizer.py` 很长，原因不是这条主线复杂，而是它还要兼容很多能力：

- checkpoint state dict
- FP8 / NVFP4 参数
- precision-aware optimizer
- CPU offload
- layer-wise optimizer
- Megatron-FSDP
- MoE expert 相关处理

第一次读不要跟进去。

只要你能沿着下面这条线走通，就已经读到了 Distributed Optimizer 的核心：

```text
shard_buffer
-> start_grad_sync reduce-scatter
-> Range / gbuf range map
-> _build_model_and_main_param_groups
-> _copy_model_grads_to_main_grads
-> optimizer.step
-> _copy_main_params_to_model_params
-> start_param_sync all-gather
```

---

### B14. 自检问题

读完后可以用这些问题检查自己有没有读通：

1. 为什么普通 DDP 的 optimizer state 会在 DP ranks 间重复？
2. reduce-scatter 和 all-reduce 的输出有什么区别？
3. 为什么 Megatron 切 shard 时按 bucket buffer 切，而不是天然按参数边界切？
4. `gbuf_world`、`gbuf_world_in_bucket`、`gbuf_local`、`param` 分别是哪种坐标系？
5. 为什么 inner optimizer 只需要看到 shard 参数，就自然只会创建 shard optimizer state？
6. step 后为什么还需要 param all-gather？

---

### B15. 下一步带读建议

下一节可以有两个方向。

如果继续沿训练主线读，建议进入：

```text
distributed checkpointing
```

因为 Distributed Optimizer 引入了 optimizer state shard，checkpoint 就必须回答：

```text
这些分散在 DP ranks 上的 param shard 和 optimizer state shard
如何保存、加载、reshard？
```

如果想先补齐并行策略，也可以进入：

```text
context parallel 或 expert parallel / MoE
```

但建议优先读 distributed checkpointing，因为它直接接在本节的 shard state 后面。

