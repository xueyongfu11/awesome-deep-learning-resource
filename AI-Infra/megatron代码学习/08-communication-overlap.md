# 08. Communication Overlap 带读

这一节不引入新的并行维度，而是读 Megatron 如何把已有通信藏到计算后面。

前面几节你已经见过这些同步点：

- DP / Distributed Optimizer：梯度需要 all-reduce 或 reduce-scatter。
- Distributed Optimizer：下一轮 forward 前，参数 shard 需要 all-gather 回完整参数 buffer。
- PP：stage 之间要发送 activation 和 activation grad。
- TP / SP：linear 前后可能要 all-gather、reduce-scatter。

如果这些通信都同步执行，训练时间会变成：

```text
compute -> communication -> compute -> communication
```

overlap 的目标是改成：

```text
compute ---- compute ---- compute
     communication ---- communication
```

也就是通信仍然要发生，但尽量早发出，真正需要结果时再 wait。

## A. 不看代码版：Communication Overlap 运行过程

Communication Overlap 解决的问题是：分布式训练里通信很贵，如果总是“算完再通信”，GPU 会经常等待。

它的思路是：通信尽早发起，真正需要结果时再等待。

```text
普通方式:
  计算 A
  等通信 A
  计算 B
  等通信 B

Overlap:
  计算 A
  发起通信 A，不立刻等
  同时计算 B
  到必须使用 A 的结果时再 wait
```

Megatron 里常见几类 overlap：

```text
DP 梯度 overlap:
  backward 某些梯度 ready 后就开始 reduce。

参数 gather overlap:
  下一层参数提前 all-gather，用到前再 wait。

PP P2P overlap:
  pipeline stage 的 send/recv 先发出，后面再等待。

TP Linear overlap:
  Linear 内部的 TP 通信和 GEMM 尽量重叠。
```

它不改变模型数学结果，只改变通信和计算的时间安排。

不看代码时记住：overlap 的核心是“尽早 start，尽晚 wait”。效果好不好取决于通信能不能被后续计算时间盖住。

## B. 代码带读版：Communication Overlap 实现路径

### B1. 先分清四类 overlap

不要把所有 overlap 混成一个概念。Megatron 里至少有四条线：

| 类型 | 开关 | 通信对象 | 主要位置 |
| --- | --- | --- | --- |
| DP 梯度 overlap | `overlap_grad_reduce` | 梯度 all-reduce / reduce-scatter | `param_and_grad_buffer.py` |
| 参数 gather overlap | `overlap_param_gather` | Distributed Optimizer 参数 all-gather | `param_and_grad_buffer.py`、`distributed_data_parallel.py` |
| PP P2P overlap | `overlap_p2p_comm` | pipeline activation / activation grad send-recv | `schedules.py`、`p2p_communication.py` |
| TP Linear overlap | `tp_comm_overlap` | Linear 内 TP all-gather / reduce-scatter | `model_parallel_config.py`、`extensions/transformer_engine.py` |

第一遍建议按这个顺序读：

1. DP 梯度 overlap
2. 参数 gather overlap
3. PP P2P overlap
4. TP Linear overlap

前两条接在第 6、7 节后面，最容易读通。

### B2. DP 梯度 overlap：从 finalize 提前到 backward hook

先看普通路径。

在 `megatron/core/distributed/finalize_model_grads.py` 里，`finalize_model_grads()` 会调用：

```python
for model_chunk in model:
    model_chunk.finish_grad_sync(force_all_reduce=force_all_reduce)
```

如果没有开启 `overlap_grad_reduce`，`finish_grad_sync()` 里才真正启动梯度通信：

```python
if not self.ddp_config.overlap_grad_reduce:
    self.start_grad_sync(force_all_reduce=force_all_reduce)
    self._copy_back_extra_main_grads()
    return
```

也就是：

```text
所有 backward 算完
  -> finalize_model_grads
  -> finish_grad_sync
  -> start_grad_sync 同步 all-reduce / reduce-scatter
```

开启 `overlap_grad_reduce` 后，核心变化是：

```text
某个 bucket 的梯度在 backward 中准备好了
  -> backward hook 标记 ready
  -> 这个 bucket 直接异步发起通信
  -> finalize_model_grads 只负责 wait
```

关键代码在 `distributed_data_parallel.py`：

```python
grad_acc.register_hook(self._make_backward_post_hook(param))
```

这个 hook 在参数梯度产生后触发：

```python
if self.ddp_config.overlap_grad_reduce:
    self.param_to_bucket_group[param].register_grad_ready(
        param, self.force_all_reduce
    )
```

再进入 `_ParamAndGradBucketGroup.register_grad_ready()`：

```python
if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
    self.start_grad_sync(force_all_reduce=force_all_reduce)
```

这句话是 DP 梯度 overlap 的核心：

> 不是等整个模型 backward 完，而是等某个 bucket group 里的参数梯度都 ready，就发起这个 bucket 的通信。

### B3. 为什么有 first batch / golden counts

`param_and_grad_buffer.py` 里有两个计数字典：

```python
self.golden_per_param_grad_ready_counts = {}
self.per_param_grad_ready_counts = {}
self.is_first_batch = True
```

原因是一个参数在复杂图里可能不止一次产生梯度 ready 事件。Megatron 第一轮先记录“正常应该 ready 几次”，后面每轮用这个 golden count 判断 bucket 是否真的完整 ready。

所以第一轮 overlap 不一定完全展开，主要是在建立 ready 计数基准。

### B4. start_grad_sync 真正在发什么

`start_grad_sync()` 会根据配置选择通信方式。

如果使用 Distributed Optimizer，梯度通信是 reduce-scatter：

```python
dist_reduce_scatter_func(
    local_data_view,
    bucket.grad_data,
    op=reduce_op,
    group=communication_group,
    async_op=async_op,
)
```

如果不是 Distributed Optimizer，就走 all-reduce：

```python
torch.distributed.all_reduce(
    bucket.grad_data,
    op=reduce_op,
    group=communication_group,
    async_op=async_op,
)
```

这里的 `async_op` 来自：

```python
async_op = (
    self.ddp_config.overlap_grad_reduce
    and self.ddp_config.num_distributed_optimizer_instances == 1
)
```

第一遍可以先记：

- `overlap_grad_reduce=False`：`finish_grad_sync()` 同步发通信。
- `overlap_grad_reduce=True`：backward hook 提前 `start_grad_sync()`，`finish_grad_sync()` 等 handle。

### B5. finish_grad_sync 的职责变了

不开 overlap 时：

```text
finish_grad_sync = 发起通信 + 等通信完成
```

开 overlap 时：

```text
finish_grad_sync = 确认通信已经发出 + wait
```

对应代码：

```python
assert self.grad_reduce_handle is not None
self.grad_reduce_handle.wait()
self.grad_reduce_handle = None
```

这就是 Megatron overlap 很常见的模式：

```text
start_xxx_sync()   提前发异步通信
finish_xxx_sync()  真正需要结果时 wait
```

后面参数 gather 和 PP P2P 都是这个心智模型。

### B6. 参数 gather overlap：下一层用参数前再 wait

Distributed Optimizer 会把参数 shard 分散在 DP ranks 上。optimizer step 后，每个 rank 只更新自己那一片参数。下一轮 forward 前，需要把参数 all-gather 回完整参数 buffer。

不开 overlap 时，这个 all-gather 可以同步完成。

开 `overlap_param_gather` 后，`start_param_sync()` 会异步发起 all-gather：

```python
async_op = self.ddp_config.overlap_param_gather and not force_sync
```

Distributed Optimizer 标准路径里：

```python
dist_all_gather_func(
    bucket.param_data,
    local_data_view,
    group=self.intra_distributed_optimizer_instance_group,
    async_op=async_op,
)
```

真正等它完成的位置，不是随便等，而是在 forward pre-hook 里：

```python
module.register_forward_pre_hook(self._make_forward_pre_hook())
```

hook 的语义是：

> 当前 module 马上要用自己的参数了，如果这个参数所在 bucket 的 all-gather 还没完成，就在这里 wait。

对应：

```python
self.param_to_bucket_group[param].finish_param_sync(...)
```

所以参数 gather overlap 的时间线是：

```text
optimizer step 更新本 rank 参数 shard
  -> start_param_sync 异步 all-gather 后续 bucket
  -> forward 继续推进
  -> 某层真的要用参数时，forward pre-hook 调 finish_param_sync wait
```

### B7. overlap_param_gather 依赖 overlap_grad_reduce

在 `megatron/training/arguments.py` 里有约束：

```python
if args.overlap_param_gather:
    assert args.use_distributed_optimizer or args.use_megatron_fsdp \
        or args.optimizer == 'dist_muon'
    assert args.overlap_grad_reduce
```

也就是说普通 dense 训练里，参数 gather overlap 通常建立在 Distributed Optimizer / FSDP 这类参数 shard 方案之上，并且要求梯度 reduce overlap 打开。

读代码时可以先不要展开 FSDP，只看 Distributed Optimizer 路径。

### B8. PP P2P overlap：send/recv 先发，后面再 wait

Pipeline Parallel 里 stage 之间要传：

- forward activation：上一 stage -> 下一 stage
- backward activation grad：下一 stage -> 上一 stage

`p2p_communication.py` 里，`send_forward_recv_forward()` 有一个参数：

```python
overlap_p2p_comm: bool = False
```

不开 overlap 时：

```python
wait_on_reqs=(not overlap_p2p_comm)
```

也就是函数内部等 send/recv 完成再返回。

开 overlap 时，函数返回 tensor 和 wait handles：

```python
if overlap_p2p_comm:
    return input_tensor, wait_handles
```

然后 `schedules.py` 在合适位置 wait：

```python
recv_prev_wait_handle.wait()
send_next_wait_handle.wait()
recv_next_wait_handle.wait()
send_prev_wait_handle.wait()
```

它和 DP overlap 的模式一样：

```text
发通信 -> 继续做别的 microbatch 计算 -> 真正需要 recv 结果或释放 send buffer 前 wait
```

不过要注意一个限制：当前参数检查里，非 interleaved schedule 会关闭 `overlap_p2p_comm`：

```python
args.overlap_p2p_comm = False
args.align_param_gather = False
```

所以 PP P2P overlap 不是所有 pipeline schedule 都可用。

### B9. TP Linear overlap：主要交给 Transformer Engine

TP Linear overlap 是另一类：它发生在线性层内部。

配置在 `model_parallel_config.py`：

```python
tp_comm_overlap: bool = False
```

注释说明它用于让 Linear layer execution 和 TP collective overlap，例如 AllGather / ReduceScatter。

训练参数检查里还有一个重要约束：

```python
if args.tp_comm_overlap:
    assert args.sequence_parallel == True
```

所以在 Megatron 这条主线里，TP comm overlap 通常和 Sequence Parallel 绑定理解。

真正把这些开关传给后端的是 `extensions/transformer_engine.py`。例如 TE Linear 会设置：

```python
extra_kwargs["ub_overlap_ag"] = self.config.tp_comm_overlap_ag
extra_kwargs["ub_overlap_rs"] = self.config.tp_comm_overlap_rs
extra_kwargs["ub_name"] = tp_comm_buffer_name
```

这里的 `ub` 可以先理解成 TE 侧的 user buffer 机制。第一次读不用深入 TE 内部实现，只需要知道：

> DP / PP overlap 主要在 Megatron Python 调度层控制；TP Linear overlap 更多交给 Transformer Engine 的 Linear kernel / user buffer 机制。

### B10. 一张总图

```text
Backward compute
  |
  |-- param grad ready hook
  |     |
  |     |-- overlap_grad_reduce:
  |          bucket ready -> async all-reduce / reduce-scatter
  |
  |-- finalize_model_grads
        |
        |-- finish_grad_sync waits outstanding grad communication

Optimizer step
  |
  |-- updates local optimizer shard / param shard
  |
  |-- overlap_param_gather:
        start_param_sync -> async all-gather params

Next forward
  |
  |-- forward pre-hook
        |
        |-- finish_param_sync waits if this module's params are needed

Pipeline schedule
  |
  |-- overlap_p2p_comm:
        send/recv returns wait handles
        schedule waits only before using recv result or freeing send buffer

TE Linear
  |
  |-- tp_comm_overlap:
        TE overlaps GEMM with TP AG/RS when supported
```

### B11. 读代码时抓这几个问题

1. `overlap_grad_reduce=False` 时，梯度通信到底在哪里发起？
2. `overlap_grad_reduce=True` 时，哪个 hook 负责提前发起通信？
3. 为什么 bucket 要等所有参数 grad ready 才能发？
4. `start_grad_sync()` 在 Distributed Optimizer 下发的是 all-reduce 还是 reduce-scatter？
5. `finish_grad_sync()` 在 overlap 开关前后职责有什么变化？
6. `overlap_param_gather` 为什么需要 forward pre-hook？
7. PP P2P overlap 为什么要保存 wait handles？
8. `tp_comm_overlap` 为什么要求 sequence parallel？

### B12. 本节先记住的结论

Communication overlap 不是新的切分方式，而是调度方式。

最核心的模式只有一句：

```text
尽早 start，尽晚 finish。
```

具体到 Megatron：

- 梯度 overlap：backward hook 里 start，`finalize_model_grads()` 里 finish。
- 参数 gather overlap：optimizer 后或 forward 前 start，module forward pre-hook 里 finish。
- PP P2P overlap：schedule 里 send/recv 先返回 handle，后续用到数据或释放 buffer 前 finish。
- TP Linear overlap：Megatron 设置 TE overlap 参数，由 TE 在 Linear 内部做 GEMM 和 AG/RS overlap。

下一节如果继续深入，可以读 MoE / Expert Parallel；如果想把训练系统更闭环，也可以先读 Distributed Checkpointing。

