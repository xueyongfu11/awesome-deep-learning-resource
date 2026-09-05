# 04. Pipeline Parallel 与 1F1B Schedule 带读

这一节继续读 Pipeline Parallel。

上一节已经看清楚了单个 `TransformerLayer` 如何把 LayerNorm、SelfAttention、BDA 和 MLP 串成一个 decoder block。现在往外走一步，看一组 decoder blocks 如何被切到不同 pipeline ranks，以及 non-interleaved 1F1B schedule 如何安排 microbatch。

本节先建立三件事：

1. Pipeline Parallel 切的是 layer 序列。
2. forward hidden states 从前一个 PP rank 流向下一个 PP rank。
3. backward gradients 从后一个 PP rank 反向流回前一个 PP rank。

这一节解决两个问题：

```text
一组 Transformer layers 如何分到不同 PP ranks？
一个 global batch 被拆成多个 microbatch 后，Megatron 如何安排这些 microbatch 做 forward 和 backward？
```

本节只看 dense GPT 主线里的 **non-interleaved pipeline schedule**。

暂时不展开：

- interleaved pipeline
- virtual pipeline stage
- combined 1F1B
- bridge communicator
- multi-module pipeline
- MoE
- Transformer Engine / FP8
- distributed optimizer

建议打开这些文件对照读：

- `megatron/core/transformer/transformer_block.py`
- `megatron/core/pipeline_parallel/schedules.py`
- `megatron/core/pipeline_parallel/p2p_communication.py`

---

## A. 不看代码版：Pipeline Parallel 1F1B 运行过程

1F1B 解决的问题是：pipeline stage 不能只等一个完整 batch 全部 forward 完再 backward，否则很多 GPU 会空等。

Megatron 会把一个 batch 切成多个 microbatch，让它们像流水线一样经过不同 stage：

```text
microbatch 0 -> stage0 -> stage1 -> stage2
microbatch 1 -> stage0 -> stage1 -> stage2
microbatch 2 -> stage0 -> stage1 -> stage2
```

1F1B 的意思是稳定阶段里每个 stage 尽量交替做：

```text
1 个 forward
1 个 backward
1 个 forward
1 个 backward
```

整个 schedule 分三段：

```text
warmup:
  先塞入足够多的 forward，让 pipeline 填起来。

steady 1F1B:
  一边继续 forward 新 microbatch，一边 backward 旧 microbatch。

cooldown:
  不再塞新 forward，把剩下的 backward 做完。
```

stage 之间传的是 activation 和 activation gradient：

```text
forward:  当前 stage 输出 hidden，发给下一个 stage
backward: 当前 stage 收到输出梯度，算完后把输入梯度发回上一个 stage
```

不看代码时记住：1F1B 的核心不是新模型结构，而是安排 microbatch 的前后向顺序，减少 pipeline 空泡。

## B. 代码带读版：Pipeline Parallel 实现路径

### B1. 从 TransformerLayer 到 TransformerBlock

入口：

- `megatron/core/transformer/transformer_block.py`
- 重点函数：`get_num_layers_to_build()`、`_build_layers()`、`forward()`

`TransformerLayer` 是一个 decoder block；`TransformerBlock` 是一组 decoder blocks。

不开 pipeline parallel 时，可以先理解成：

```text
TransformerBlock
  layers = [TransformerLayer 1, TransformerLayer 2, ..., TransformerLayer N]

forward:
  for layer in layers:
      hidden_states, context = layer(hidden_states, ...)
```

开 pipeline parallel 后，每个 pipeline rank 只构建自己负责的那一段 layers。关键逻辑在 `get_num_layers_to_build()`：

```text
num_layers_per_pipeline_rank = num_layers / pipeline_model_parallel_size
```

非 interleaved 主线里，每个 PP stage 拿连续的一段 layers。例如 8 层、PP size = 4：

```text
PP rank 0: layer 1, 2
PP rank 1: layer 3, 4
PP rank 2: layer 5, 6
PP rank 3: layer 7, 8
```

这样每个 rank 上的 `TransformerBlock.forward()` 仍然是本地顺序执行自己的 layers，只是输入 hidden states 来自前一个 pipeline stage，输出 hidden states 发给下一个 pipeline stage。

### B2. Pipeline stage 的输入输出

不开 PP 时，GPT 主线是：

```text
embedding
  -> TransformerBlock(all layers)
  -> final layernorm
  -> output layer / loss
```

开 PP 后，可以理解成：

```text
PP rank 0:
  embedding
  local Transformer layers
  send hidden_states to rank 1

PP middle rank:
  recv hidden_states from previous rank
  local Transformer layers
  send hidden_states to next rank

PP last rank:
  recv hidden_states from previous rank
  local Transformer layers
  final layernorm / output layer / loss
```

在 `GPTModel._preprocess()` 里，之前已经看到过：

```text
pre_process=False 时，中间 stage 不持有 embedding；
decoder_input 会从 pipeline 通信里传入。
```

所以 PP 的核心不是改变单个 layer 内部计算，而是把 layer 序列切成多段，让不同 rank 接力处理 hidden states。

### B3. 先定位 schedule 入口

入口函数在：

```python
def get_forward_backward_func(pp_size=None, vp_size=None):
```

它负责根据并行配置选择 forward-backward 函数：

```text
if pp_size > 1:
    if vp_size is not None:
        forward_backward_pipelining_with_interleaving
    else:
        forward_backward_pipelining_without_interleaving
else:
    forward_backward_no_pipelining
```

这说明：

- 没有 PP：走 `forward_backward_no_pipelining()`。
- 有 PP，且没有 virtual pipeline：走 `forward_backward_pipelining_without_interleaving()`。
- 有 PP，且有 virtual pipeline：走 interleaved 版本。

本节只读：

```python
forward_backward_pipelining_without_interleaving()
```

这个函数的 docstring 已经把它定位得很清楚：

```text
Run non-interleaved 1F1B schedule, with communication between pipeline stages.
```

也就是：

> 在 pipeline stages 之间通信，并执行非 interleaved 的 1F1B schedule。

---

### B4. 1F1B 是什么

1F1B 的意思是：

```text
one forward, one backward
```

进入稳定阶段后，每个 pipeline rank 尽量交替做：

```text
forward 一个 microbatch
backward 一个更早的 microbatch
```

它不是一个 microbatch 完整 forward 到最后，然后立刻完整 backward 回来。

如果那样做，其他 pipeline rank 会经常空等。

Pipeline Parallel 的核心问题是：

```text
rank 0 做前几层
rank 1 做中间几层
rank 2 做最后几层
```

单个 microbatch 的 forward 必须按顺序走：

```text
rank 0 -> rank 1 -> rank 2
```

backward 必须按反方向走：

```text
rank 2 -> rank 1 -> rank 0
```

如果只有一个 microbatch，pipeline 没有办法被填满。

所以训练时会把 global batch 拆成多个 microbatch，让不同 microbatch 同时处在不同 PP rank 上。

---

### B5. 整体结构：warmup、1F1B、cooldown

`forward_backward_pipelining_without_interleaving()` 的主结构可以压缩成：

```text
准备 communicator、process groups、tensor shapes

计算 num_warmup_microbatches

warmup:
    只做 forward
    保存 input/output tensor，留给之后 backward

steady 1F1B:
    forward 当前 microbatch
    send forward output，并接收 backward grad
    backward 一个较早的 microbatch
    send backward grad，并接收下一个 forward input

cooldown:
    不再做 forward
    把 warmup 阶段剩下的 backward 补完

finalize grads
return forward_data_store
```

三段的含义：

| 阶段 | 做什么 | 为什么需要 |
| --- | --- | --- |
| warmup | 先做若干个 forward | 把 pipeline 填起来 |
| 1F1B | forward 和 backward 交替 | 让 rank 尽量持续工作 |
| cooldown | 补剩下的 backward | 把之前只 forward 的 microbatch 反传完 |

这一节最重要的直觉：

> warmup 是填管道，1F1B 是稳定流动，cooldown 是排空管道。

---

### B6. warmup 数量怎么来

代码里 warmup microbatch 数量是：

```python
num_warmup_microbatches = (
    p2p_communicator.total_stages
    - p2p_communicator.current_stage
    - 1
)
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
num_microbatches_remaining = num_microbatches - num_warmup_microbatches
```

假设一共有 3 个 PP stages：

```text
rank 0: current_stage = 0 -> warmup = 2
rank 1: current_stage = 1 -> warmup = 1
rank 2: current_stage = 2 -> warmup = 0
```

为什么越靠前的 rank warmup 越多？

因为 backward 最早只能从最后一个 rank 开始。

最后一个 rank 拿到第一个 microbatch 的 forward 输出后，就可以算 loss 并开始 backward。

但 rank 0 在收到 backward grad 之前，需要先把足够多的 microbatch 往后送。

所以：

```text
rank 0 需要先 forward 更多 microbatch
rank 1 少一些
rank 2 不需要 warmup，拿到 forward 后可以更早进入 backward
```

---

### B7. warmup 阶段在做什么

warmup 循环里，每轮核心步骤是：

```python
input_tensor = p2p_communicator.recv_forward(...)

output_tensor, num_tokens = forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    ...
)

p2p_communicator.send_forward(output_tensor, ...)

input_tensors.append(input_tensor)
output_tensors.append(output_tensor)
deallocate_output_tensor(output_tensor, ...)
```

翻译成读代码时的动作：

```text
1. 从前一个 PP rank 接收 forward input
2. 跑本 rank 的 local layers
3. 把 output 发给下一个 PP rank
4. 保存 input/output，后面 backward 要用
```

第一个 rank 的 `recv_forward()` 会返回 `None`。

因为 rank 0 没有前一个 PP rank，它的输入来自 data iterator 和 embedding。

最后一个 rank 的 `send_forward()` 不会真的往下发。

因为 last stage 没有下一个 PP rank，它会在 `forward_step()` 里进入 loss 相关逻辑。

---

### B8. forward_step 做了什么

`forward_step()` 是 schedule 和模型 forward 之间的桥。

它做的关键事情是：

```python
set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
set_input_tensor(input_tensor)

output_tensor, loss_func = forward_step_func(data_iterator, model)

output_tensor, num_tokens = forward_step_calc_loss(...)
```

理解成三步：

```text
1. 把来自上一个 PP rank 的 input_tensor 放进 model
2. 调用训练脚本传进来的 forward_step_func
3. 如果当前是 last stage，就计算 loss 或收集 loss data
```

这里容易混淆的是两个名字：

```text
forward_step()
forward_step_func()
```

区别是：

| 名字 | 来自哪里 | 负责什么 |
| --- | --- | --- |
| `forward_step()` | Megatron schedule | 统一处理 PP 输入、autocast、loss 收集 |
| `forward_step_func()` | 训练入口传入 | 真正调用 model 和 data iterator |

对 PP 来说，`input_tensor` 很重要。

中间 rank 的模型不是直接从 token ids 开始算，而是接收上一个 rank 发来的 hidden states。

所以 `forward_step()` 开头要调用：

```python
set_input_tensor(input_tensor)
```

---

### B9. backward_step 做了什么

`backward_step()` 是 local backward 的统一入口。

它的签名是：

```python
def backward_step(input_tensor, output_tensor, output_tensor_grad, config):
```

核心逻辑可以读成：

```text
1. 对 input_tensor retain_grad()
2. 对 output_tensor 调 autograd backward
3. 取出 input_tensor.grad
4. 返回 input_tensor_grad
```

为什么要返回 `input_tensor_grad`？

因为这个梯度要发回前一个 PP rank。

在当前 rank 看来：

```text
input_tensor        是上一个 rank forward 发来的 hidden states
output_tensor       是本 rank forward 算出来、发给下一个 rank 的 hidden states
output_tensor_grad  是下一个 rank backward 发回来的梯度
input_tensor_grad   是本 rank 算完 backward 后，要发回上一个 rank 的梯度
```

用图表示：

```text
forward:
prev rank -- input_tensor --> current rank -- output_tensor --> next rank

backward:
prev rank <-- input_tensor_grad -- current rank <-- output_tensor_grad -- next rank
```

如果当前 rank 是 last stage，`output_tensor_grad` 可以是 `None`。

因为最后一段可以从 loss 开始反传。

如果当前 rank 是 first stage，`input_tensor_grad` 不需要再发给前一个 PP rank。

因为没有前一个 PP rank。

---

### B10. P2PCommunicator 的方向感

`P2PCommunicator` 封装了 PP rank 之间的点对点通信。

先记住两个方向：

```text
forward 方向：previous rank -> current rank -> next rank
backward 方向：next rank -> current rank -> previous rank
```

对应到方法：

| 方法 | 含义 |
| --- | --- |
| `recv_forward()` | 从 previous rank 收 forward input |
| `send_forward()` | 向 next rank 发 forward output |
| `recv_backward()` | 从 next rank 收 backward grad |
| `send_backward()` | 向 previous rank 发 backward grad |
| `send_forward_recv_backward()` | 向 next 发 forward output，同时从 next 收 backward grad |
| `send_backward_recv_forward()` | 向 previous 发 backward grad，同时从 previous 收 forward input |

基础方法里有边界判断。

例如：

```text
first stage:
    recv_forward() 返回 None
    send_backward() 不发送

last stage:
    send_forward() 不发送
    recv_backward() 返回 None
```

这就是为什么 schedule 可以在所有 PP ranks 上跑同一份代码。

每个 rank 根据自己是不是 first/last stage，自动跳过无意义的通信。

---

### B11. 进入 1F1B 前的第一步

warmup 结束后，代码有一段：

```python
if num_microbatches_remaining > 0:
    input_tensor = p2p_communicator.recv_forward(...)
```

这一步是在给 steady 1F1B 准备第一个 forward input。

在后面的 1F1B 循环里，每轮一开始会直接调用 `forward_step()`。

所以进入循环前，需要先让 `input_tensor` 准备好。

如果当前 rank 是 first stage，这个 input 仍然是 `None`。

如果是中间或最后 rank，它会从 previous rank 收到 hidden states。

---

### B12. steady 1F1B 主循环

1F1B 主循环是：

```python
for i in range(num_microbatches_remaining):
    output_tensor, num_tokens = forward_step(...)

    output_tensor_grad = p2p_communicator.send_forward_recv_backward(
        output_tensor,
        send_tensor_shapes,
        p2p_communicator.is_pp_last_stage,
    )

    input_tensors.append(input_tensor)
    output_tensors.append(output_tensor)

    input_tensor = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)

    input_tensor_grad = backward_step(
        input_tensor,
        output_tensor,
        output_tensor_grad,
        config,
    )

    if last_iteration:
        p2p_communicator.send_backward(...)
    else:
        input_tensor = p2p_communicator.send_backward_recv_forward(...)
```

压缩成一句话：

> 先 forward 当前 microbatch，再 backward 队列里最早还没 backward 的 microbatch。

这里的 `input_tensors` 和 `output_tensors` 是队列。

warmup 和 steady forward 会把 tensor 放进去：

```text
append(input_tensor)
append(output_tensor)
```

需要 backward 时，从队列头部拿最早的那一组：

```text
pop(0)
```

所以它不是 backward 刚刚 forward 的那个 microbatch。

它 backward 的是更早 forward 完、现在已经从后续 rank 收到梯度的 microbatch。

---

### B13. 为什么组合通信很重要

steady 1F1B 阶段有两个组合通信：

```python
send_forward_recv_backward()
send_backward_recv_forward()
```

第一个出现在 forward 后：

```text
把当前 forward output 发给 next rank
同时从 next rank 收一个 backward grad
```

第二个出现在 backward 后：

```text
把当前 backward input grad 发给 previous rank
同时从 previous rank 收下一个 forward input
```

这两个方法让当前 rank 的两侧通信衔接起来：

```text
forward side:
current -> next
current <- next

backward side:
current -> previous
current <- previous
```

对中间 rank 来说，一轮 1F1B 的方向感是：

```text
1. 用 previous 发来的 input 做 forward
2. 把 output 发给 next
3. 从 next 收 backward grad
4. 对更早的 microbatch 做 backward
5. 把 input grad 发给 previous
6. 从 previous 收下一个 forward input
```

---

### B14. 一个 3-rank 例子

假设：

```text
PP ranks = 3
microbatches = m0, m1, m2, m3
```

每个 rank 的 warmup 数量：

```text
rank 0: 2
rank 1: 1
rank 2: 0
```

这里不要把 1F1B 画成严格对齐的时间线，否则很容易误解成多个 rank 同时 backward 同一个 microbatch。

先只记住几个顺序关系：

- rank 0 先连续 forward 两个 microbatch。
- rank 2 最早开始 backward。
- 稳定后，每个 rank 尽量 F 和 B 交替。
- 最后 rank 0 还需要补完后面几个 backward。

更重要的是每个 microbatch 的依赖方向：

```text
m0 forward:
rank 0 -> rank 1 -> rank 2

m0 backward:
rank 2 -> rank 1 -> rank 0
```

1F1B 只是把多个 microbatch 交错起来，让这条路径更少空转。

---

### B15. cooldown 阶段

steady 1F1B 结束后，代码进入：

```python
for i in range(num_warmup_microbatches):
    input_tensor = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)

    output_tensor_grad = p2p_communicator.recv_backward(...)

    input_tensor_grad = backward_step(...)

    p2p_communicator.send_backward(...)
```

cooldown 只做 backward，不再做 forward。

因为所有 microbatch 的 forward 都已经发出去了。

但 warmup 阶段产生的一部分 activation 还没有 backward。

所以 cooldown 做的是：

```text
从 next rank 收 backward grad
对队列中剩下的 microbatch 做 backward
把 input grad 发给 previous rank
```

它的作用是把 pipeline 排空。

---

### B16. forward_only 分支

`forward_backward_pipelining_without_interleaving()` 也支持 `forward_only=True`。

这种模式下：

```text
只 forward
不保存 input/output tensors 给 backward
不调用 backward_step
不进入 cooldown backward
```

在 1F1B 主循环中，forward-only 分支只做：

```python
p2p_communicator.send_forward(output_tensor, ...)
if not last_iteration:
    input_tensor = p2p_communicator.recv_forward(...)
```

这通常对应 evaluation 或只需要 forward 输出的场景。

本节主线仍然按训练模式理解，也就是：

```text
forward_only = False
```

---

### B17. tensor shape 从哪里来

schedule 在通信前会计算：

```python
recv_tensor_shapes = get_tensor_shapes(...)
send_tensor_shapes = get_tensor_shapes(...)
```

P2P 通信接收 tensor 时，需要提前知道 shape。

常见 dense GPT 主线里，pipeline 之间传的是 hidden states：

```text
[sequence, micro_batch, hidden]
```

也就是：

```text
[S, B, H]
```

如果开启 variable sequence length，`P2PCommunicator` 还会先通信 shape，再通信真正 tensor。

这部分在 `_communicate_shapes()` 里。

当前阶段只需要记住：

> PP rank 之间传 tensor 时，接收方必须知道要分配多大的 buffer。

---

### B18. deallocate_output_tensor 是什么

1F1B 里 forward 之后会看到：

```python
deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
```

它不是把 autograd 图删掉。

注释里说得很明确：

```text
At this point, the output tensor is only useful for its .grad_fn field,
and not its .data.
```

也就是说：

- output 已经发给下一个 PP rank。
- 当前 rank 后面 backward 还需要这个 tensor 的 autograd 连接。
- 但不一定需要完整 `.data`。

所以它可以把 `.data` 换成很小的 tensor，降低 pipeline activation 占用。

这也是为什么 `backward_step()` 里有：

```python
custom_backward(output_tensor[0], output_tensor_grad[0])
```

因为 PyTorch 普通 backward 会检查 output 和 grad shape 是否一致。

伪释放以后，output `.data` 的 shape 已经不是原来的 hidden state shape，所以需要走 custom backward。

这里先理解目的即可：

> output 发出去后，本 rank 尽量只保留 backward 必要的信息，减少显存。

---

### B19. 把函数调用串起来

训练模式下，中间 PP rank 的主线可以记成：

```text
warmup:
    recv_forward
    forward_step
    send_forward
    save input/output

steady 1F1B:
    forward_step
    send_forward_recv_backward
    save current input/output
    pop oldest input/output
    backward_step
    send_backward_recv_forward

cooldown:
    pop oldest input/output
    recv_backward
    backward_step
    send_backward
```

first stage 的区别：

```text
recv_forward -> None
send_backward -> no-op
```

last stage 的区别：

```text
send_forward -> no-op
recv_backward -> None
forward_step 会负责 loss 相关逻辑
```

所以同一个 schedule 函数可以同时服务所有 PP ranks。

---

### B20. 第四轮总图

```text
global batch
  -> split into microbatches: m0, m1, m2, ...


Pipeline ranks:

rank 0             rank 1             rank 2
layers 1..k   ->   layers k+1..m  ->   layers m+1..n


Forward tensor flow:

rank 0 -- hidden --> rank 1 -- hidden --> rank 2


Backward grad flow:

rank 0 <-- grad ---- rank 1 <-- grad ---- rank 2


Schedule:

warmup:
  fill pipeline with forward-only work

steady 1F1B:
  forward current microbatch
  backward older microbatch

cooldown:
  finish remaining backward work
```

---

### B21. 建议带着这些问题复读

1. Pipeline parallel 切的是 batch、sequence，还是 layer 序列？
2. 非 interleaved PP 下，每个 rank 的 local layers 是连续的吗？
3. first / middle / last pipeline stage 的输入输出有什么不同？
4. 为什么 rank 越靠前，warmup microbatch 数越多？
5. 为什么 1F1B 里 backward 的不是刚刚 forward 的 microbatch？
6. `input_tensors` / `output_tensors` 为什么像队列一样 append 和 pop？
7. first stage 的 `recv_forward()` 为什么返回 `None`？
8. last stage 的 `recv_backward()` 为什么返回 `None`？
9. `send_forward_recv_backward()` 同时做了哪两个方向的通信？
10. `send_backward_recv_forward()` 同时做了哪两个方向的通信？
11. `deallocate_output_tensor()` 为什么不能破坏 autograd 图？
12. forward-only 模式和训练模式在 schedule 上有什么不同？
13. 如果只有一个 microbatch，pipeline bubble 会发生在哪里？

---

### B22. 下一步带读建议

下一节建议进入：

```text
Optimizer 和 Distributed DDP
```

这时再读：

- `megatron/core/distributed/`
- `megatron/core/optimizer/`

目标是理解：

```text
PP/TP 负责把模型计算切开
DDP/optimizer 负责把梯度和参数更新组织起来
```

建议下一篇只先看：

- model parameters 如何被 DDP 包装
- gradients 如何进入 buckets
- gradient finalization 在 forward-backward 之后如何触发

暂时不要直接跳 distributed optimizer state sharding。

先把普通 DDP 梯度同步读顺，再进入 distributed optimizer。

