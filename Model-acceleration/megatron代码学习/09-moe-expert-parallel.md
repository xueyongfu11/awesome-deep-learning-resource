# 09. MoE / Expert Parallel 带读

这一节开始读 MoE，也就是 Mixture of Experts。

前面 dense GPT 的 MLP 是：

```text
hidden_states
  -> dense MLP(fc1 -> activation -> fc2)
  -> output
```

MoE 把这个 MLP 换成：

```text
hidden_states
  -> router 选择专家
  -> token dispatcher 把 token 送到对应专家所在 rank
  -> local experts 计算
  -> token dispatcher 把结果合回来
  -> output
```

所以 MoE 的核心不是 attention，而是替换 transformer layer 里的 MLP block。

## A. 不看代码版：MoE / Expert Parallel 运行过程

MoE 解决的问题是：想增大模型参数量，但不希望每个 token 都经过所有参数计算。

普通 dense MLP 是每个 token 都走同一个 MLP：

```text
token -> MLP -> output
```

MoE 把一个 MLP 换成多个 expert，并让 router 为每个 token 选择少数几个 expert：

```text
token hidden
  -> router 打分
  -> 选择 top-k experts
  -> token 被发送到对应 expert
  -> expert 计算
  -> 结果按权重合并
```

Expert Parallel 解决的是 expert 太多时怎么放到多张 GPU 上：

```text
rank0: expert 0, 1
rank1: expert 2, 3
rank2: expert 4, 5
rank3: expert 6, 7
```

由于不同 token 会选不同 expert，MoE 中间会有一次明显的 token 重排和通信：

```text
本地 tokens
  -> router 决定每个 token 去哪里
  -> dispatcher 把 token 发到 expert 所在 rank
  -> local experts 计算
  -> combine 把结果送回原 token 顺序
```

MoE 的难点不是单个 expert 的 MLP，而是 router 是否均衡、token dispatcher 怎么通信、expert 参数和普通参数怎么同步。

不看代码时记住：MoE 是“每个 token 只激活少数 expert”；EP 是“expert 列表分布到多张卡”。

## B. 代码带读版：MoE / Expert Parallel 实现路径

### B1. 先看入口：GPT layer spec 如何切到 MoE

入口在：

```text
megatron/core/models/gpt/gpt_layer_specs.py
megatron/core/models/gpt/moe_module_specs.py
megatron/core/transformer/transformer_layer.py
```

`gpt_layer_specs.py` 里，如果 `num_experts is None`，构建普通 dense MLP：

```python
if num_experts is None:
    return partial(module, submodules=MLPSubmodules(...))
```

如果 `num_experts` 不为空，就构建 MoE：

```python
return get_moe_module_spec_for_backend(
    backend=backend,
    num_experts=num_experts,
    moe_grouped_gemm=moe_grouped_gemm,
)
```

`moe_module_specs.py` 里最终返回：

```python
return partial(
    MoELayer,
    submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
)
```

也就是说：

```text
num_moe_experts is None -> MLP
num_moe_experts != None -> MoELayer
```

再看 `transformer_layer.py`，`TransformerLayer` 构建 MLP 位置没有特殊分叉，只是把 `submodules.mlp` 实例化：

```python
self.mlp = submodules.mlp(
    config=self.config,
    pg_collection=pg_collection,
    is_mtp_layer=self.is_mtp_layer,
    name=(name + ".mlp") if name is not None else None,
)
```

然后判断：

```python
self.is_moe_layer = isinstance(self.mlp, MoELayer)
```

第一条结论：

> MoE 在 Megatron 里主要是“MLP 子模块替换”，TransformerLayer 主结构仍然是 attention -> MLP -> residual。

### B2. Expert Parallel 切的是什么

EP，也就是 Expert Parallel，切的是专家集合。

在 `moe_layer.py` 的 `BaseMoELayer.__init__()`：

```python
self.ep_group = pg_collection.ep
ep_size = utils.get_pg_size(self.ep_group)
ep_rank = utils.get_pg_rank(self.ep_group)

assert self.config.num_moe_experts % ep_size == 0
self.num_local_experts = self.config.num_moe_experts // ep_size
local_expert_indices_offset = ep_rank * self.num_local_experts
self.local_expert_indices = [
    local_expert_indices_offset + i for i in range(self.num_local_experts)
]
```

例子：

```text
num_moe_experts = 8
expert_model_parallel_size = 4

EP rank 0: experts 0,1
EP rank 1: experts 2,3
EP rank 2: experts 4,5
EP rank 3: experts 6,7
```

所以 EP 不是把一个专家内部矩阵切开，而是把“专家列表”分给不同 rank。

如果还启用 expert tensor parallel，专家内部线性层也可以再被 TP 切。代码里可以看到：

```python
self.tp_group = pg_collection.expt_tp
self.tp_ep_group = pg_collection.tp_ep
```

第一遍先分清：

- `ep_group`：专家分布在哪些 rank。
- `expt_tp`：单个 expert 的 Linear 是否还做 tensor parallel。
- `tp_ep_group`：token dispatch 时可能同时跨 TP 和 EP 聚合 token。

### B3. MoELayer 的主线

`MoELayer.forward()` 里最重要的注释已经把流程列出来了：

```text
1. Routing & Preprocessing
2. Dispatch
3. Expert Computation
4. Combine
```

对应代码主线：

```python
shared_expert_output = self.shared_experts_compute(hidden_states)
probs, routing_map = self.route(hidden_states, padding_mask)
hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

dispatched_input, probs = self.dispatch(hidden_states, probs)
output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
output = self.combine(output)

output = self.postprocess(output, shared_expert_output)
```

可以压成一句话：

```text
route -> preprocess -> dispatch -> expert compute -> combine -> postprocess
```

后面读代码时不要迷路，就盯住这 6 个动词。

### B4. Router：每个 token 选哪些 expert

Router 在：

```text
megatron/core/transformer/moe/router.py
```

`Router.__init__()` 里有一个 gate 权重：

```python
self.weight = torch.nn.Parameter(
    torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
)
```

它本质是一个线性分类器：

```text
[H] -> [num_experts]
```

`gating()` 做：

```python
logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
```

`TopKRouter.routing()` 做：

```python
probs, routing_map = topk_routing_with_score_function(
    logits,
    self.topk,
    ...
)
```

输出两个关键张量：

- `probs`：token 被分到专家后的权重。
- `routing_map`：token 到 expert 的布尔映射，形状近似 `[num_tokens, num_experts]`。

如果 `moe_router_topk=2`，每个 token 会选两个 expert。此时一个 token 会被复制或拆分成两份专家输入，最后按 `probs` 加权合并。

### B5. 为什么需要 load balancing loss

MoE 的天然风险是 router 偏向少数 expert。

如果所有 token 都去 expert 0，那么：

- expert 0 负载爆炸；
- 其他 expert 学不到东西；
- EP 通信和计算严重不均衡。

所以 `TopKRouter` 里有几类平衡策略：

```python
self.routing_type = self.config.moe_router_load_balancing_type
```

常见路径包括：

- `aux_loss`
- `seq_aux_loss`
- `global_aux_loss`
- `sinkhorn`
- expert bias

第一遍不用把每种 loss 公式都背下来，只要知道：

> router 不只是选 top-k，还要用额外约束鼓励 token 分布更均衡。

代码里 aux loss 会通过 `MoEAuxLossAutoScaler.apply(...)` 挂到 `probs` 或 `logits` 的 autograd 路径上，这样主 loss 反传时也会带上 router 的平衡损失。

### B6. Token Dispatcher：MoE 最容易迷路的地方

Token dispatcher 在：

```text
megatron/core/transformer/moe/token_dispatcher.py
```

它的职责是：

```text
根据 routing_map，把 token 送到拥有目标 expert 的 rank；
专家算完后，再把 token 输出送回原来的顺序和 rank。
```

抽象接口是：

```python
dispatch_preprocess()
token_dispatch()
dispatch_postprocess()
combine_preprocess()
token_combine()
combine_postprocess()
```

这和 MoELayer 的流程刚好对应：

```text
preprocess         本地整理 token / 统计每个 expert 要多少 token
token_dispatch     跨 rank 通信，把 token 发给 expert 所在 rank
dispatch_postprocess 通信后再次排序，整理成本地 experts 可吃的连续块
combine_preprocess experts 输出后，先本地反向整理
token_combine      跨 rank 通信，把输出送回原 token 所在 rank
combine_postprocess 恢复原 shape 和原 token 顺序
```

第一遍只读两个 dispatcher：

- `MoEAllGatherTokenDispatcher`
- `MoEAlltoAllTokenDispatcher`

`flex`、DeepEP、HybridEP、inference dispatcher 可以先跳过。

### B7. AllGather dispatcher：简单但通信重

`MoEAllGatherTokenDispatcher` 的思路比较直接。

dispatch 时：

```python
self.routing_map = gather_from_sequence_parallel_region(
    self.routing_map, group=self.tp_ep_group
)
probs = gather_from_sequence_parallel_region(probs, group=self.tp_ep_group)
hidden_states = gather_from_sequence_parallel_region(
    hidden_states, group=self.tp_ep_group, use_global_buffer=True
)
```

也就是把 TP x EP 范围内 token 信息 gather 起来，然后每个 rank 从全局 token 里挑自己本地 expert 需要的 token。

本地挑 token 的代码：

```python
self.local_map = self.routing_map[
    :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
].contiguous()

tokens_per_expert = self.local_map.sum(dim=0).long().cpu()

permuted_local_hidden_states, ... = permute(
    hidden_states,
    self.local_map,
    num_out_tokens=tokens_per_expert.sum().item(),
)
```

combine 时：

```python
hidden_states = reduce_scatter_to_sequence_parallel_region(
    hidden_states.to(self.local_probs.dtype), group=self.tp_ep_group
)
```

所以 allgather dispatcher 的心智模型是：

```text
先全收集 -> 本地筛选 expert token -> 本地专家计算 -> reduce-scatter 合回去
```

它容易理解，但通信量比较大。

### B8. AlltoAll dispatcher：更像真实大规模 MoE 主线

`MoEAlltoAllTokenDispatcher` 的注释列了完整流程：

```text
1. preprocess: calculate metadata and permute
2. token dispatch: A2A(EP)
3. dispatch postprocess: AG(TP) -> sort by local expert
4. combine preprocess: unsort -> RS(TP)
5. token combine: A2A(EP)
6. combine postprocess: unpermute
```

先看 `preprocess()`。

它根据 `routing_map` 统计：

```python
num_local_tokens_per_expert = routing_map.sum(dim=0).long()
```

然后得到：

- `input_splits`：当前 rank 要发给每个 EP rank 多少 token。
- `output_splits`：当前 rank 会从每个 EP rank 收多少 token。
- `output_splits_tp`：TP 维度上要 gather / reduce-scatter 的 token 数。
- `tokens_per_expert`：本 rank 的 local experts 各自收到多少 token。

然后 `dispatch_preprocess()` 先本地 permute：

```python
permutated_local_input_tokens, permuted_probs, ... = permute(
    hidden_states,
    self.routing_map,
    probs=probs,
    num_out_tokens=self.num_out_tokens,
)
```

`token_dispatch()` 再做 EP all-to-all：

```python
global_input_tokens = all_to_all(
    self.ep_group,
    permutated_local_input_tokens,
    self.output_splits,
    self.input_splits,
)
```

如果 expert TP 大于 1，`dispatch_postprocess()` 还会在 TP 维 all-gather：

```python
global_input_tokens = gather_from_sequence_parallel_region(
    global_input_tokens, group=self.tp_group, output_split_sizes=output_split_sizes
)
```

专家算完后，combine 方向反过来：

```python
hidden_states = reduce_scatter_to_sequence_parallel_region(..., group=self.tp_group)
permutated_local_input_tokens = all_to_all(
    self.ep_group,
    hidden_states,
    self.input_splits,
    self.output_splits,
)
output = unpermute(...)
```

所以 all-to-all dispatcher 的心智模型是：

```text
本地按目标 expert/rank 排好
  -> EP all-to-all 送到专家所在 rank
  -> 本地专家算
  -> EP all-to-all 送回原 rank
  -> unpermute 恢复 token 顺序
```

### B9. Experts：收到按 expert 分组后的 token

experts 在：

```text
megatron/core/transformer/moe/experts.py
```

MoELayer 调用 experts 的位置：

```python
expert_output, mlp_bias = apply_module(self.experts)(
    dispatched_input, tokens_per_expert, permuted_probs
)
```

这里的关键输入是：

- `dispatched_input`：已经被 dispatcher 送到当前 rank、并按 local expert 排好的 token。
- `tokens_per_expert`：每个 local expert 应该处理多少 token。
- `permuted_probs`：这些 token 对应的 router 权重。

如果本 rank 有 2 个 local experts，`tokens_per_expert=[3, 5]`，那么输入 token 的排列大致是：

```text
前 3 个 token -> local expert 0
后 5 个 token -> local expert 1
```

`TEGroupedMLP` 用 grouped GEMM 同时跑多个 expert，比 Python 循环一个个 expert 更适合大规模训练。

第一遍只需要知道：

> dispatcher 已经把 token 排成 experts 容易消费的连续块；experts 只管按 `tokens_per_expert` 跑 MLP。

### B10. MoE 和 TP / SP / CP / DP 的关系

这块容易混，我建议这样记：

| 并行 | 切什么 | MoE 里对应含义 |
| --- | --- | --- |
| TP | 单层矩阵 hidden/ffn 维度 | expert 内部 Linear 也可以 TP |
| SP | TP group 内切 sequence activation | MoE + TP 训练时通常建议开 SP |
| CP | 长上下文 token 分到 CP ranks | router/aux loss 统计要考虑 TP/CP group |
| DP | 数据副本 | router 和非 expert 参数仍要按 DP 同步 |
| EP | expert 列表 | 不同 EP rank 拥有不同 expert |

MoELayer 里有一个重要检查：

```python
if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
    raise ValueError(...)
```

意思是：训练时 MoE 和 TP 一起开，但不开 SP，性能可能很差。

### B11. expert 参数和普通参数有什么不同

普通 dense 参数每个 DP rank 都有一份，所以梯度在 DP group 内同步。

expert 参数只属于某些 EP rank；不同 EP rank 上可能是不同 experts 的参数。它不能简单在普通 DP group 内同步所有 expert 参数。

你在前面 DDP 章节见过 `param.allreduce`。MoE expert 参数通常会被标记成不同同步语义：

```text
非 expert 参数：按普通 DP / DPxCP 同步
expert 参数：按 expert data parallel 语义同步
```

这也是为什么 `param_and_grad_buffer.py` 和 `distributed_data_parallel.py` 里会有 `expert_parallel_bucket_groups`。

第一遍不用深入 bucket 构造，只要知道：

> MoE 不只改变 forward 的 token 流，也改变了参数/梯度应该在哪个 group 同步。

### B12. 一个小例子

假设：

```text
S*B local tokens = 4
num_moe_experts = 4
moe_router_topk = 1
expert_model_parallel_size = 2
```

专家分布：

```text
EP rank 0: expert 0, expert 1
EP rank 1: expert 2, expert 3
```

router 输出：

```text
token 0 -> expert 2
token 1 -> expert 0
token 2 -> expert 3
token 3 -> expert 1
```

当前 rank 如果是 EP rank 0：

```text
需要把 token 0、2 发给 EP rank 1
保留或接收 token 1、3 给本地 expert 0、1
```

EP rank 1 算完 expert 2、3 后，还要把 token 0、2 的结果发回原 token 所在 rank。

所以 MoE forward 里的通信不是为了同步参数，而是为了：

```text
把 token 送到拥有目标 expert 的 rank，再把结果送回来。
```

### B13. 本节先不要展开的分支

第一遍读 MoE，建议先跳过：

- `MoEFlexTokenDispatcher`
- DeepEP / HybridEP fused kernels
- inference MoE dispatcher
- CUDA graph partial capture
- shared expert overlap
- latent MoE
- expert bias 的完整更新细节
- MoE distributed checkpointing

这些都是重要优化，但会打断主线。先把：

```text
router -> dispatch -> experts -> combine
```

读顺。

### B14. 本节结论

MoE 的核心变化有三点：

1. TransformerLayer 里的 dense MLP 被 `MoELayer` 替换。
2. Router 为每个 token 选择 top-k experts，并产生 `probs` 和 `routing_map`。
3. Expert Parallel 把 experts 分布到不同 ranks，token dispatcher 负责跨 rank 送 token 和收回结果。

一句话记：

```text
TP/SP/CP/DP 是在切模型、activation、上下文或数据；EP 是在切专家列表。
```

下一节建议读 `Transformer Engine / FP8`，因为 MoE 里的 `TEGroupedMLP`、TP comm overlap、FP8 padding 都会回到 TE 后端。

