# 07. Context Parallel 带读

这一节接在 Sequence Parallel 后面读。

Sequence Parallel 已经让你看到：Megatron 可以把 `[S, B, H]` 的 activation 在 TP group 内按 sequence 维切开，主要目的是省 transformer 层之间的 activation 显存。

Context Parallel 也和 sequence/token 维度有关，但它解决的是另一件事：

```text
当上下文很长时，
每张 GPU 不再持有完整输入序列，
而是只持有本 CP rank 负责的 token 片段。

但是 self-attention 仍然需要跨片段看到别的 token 的 K/V，
所以 attention 内部必须在 CP group 中通信。
```

先记住一句话：

```text
SP 切的是层间 activation，通信主要围绕 TP linear。
CP 切的是输入上下文，通信主要围绕 attention 的 Q/K/V。
```

---

## A. 不看代码版：Context Parallel 运行过程

Context Parallel 解决的问题是：上下文太长时，单张 GPU 放不下完整序列的 attention 计算和 KV 信息。

它把长 sequence 按 token 维切到多个 CP rank：

```text
完整序列: [token0 ... token8191]
CP=4:
  rank0: token0    ... token2047
  rank1: token2048 ... token4095
  rank2: token4096 ... token6143
  rank3: token6144 ... token8191
```

每个 rank 先只持有自己那段 token 的 hidden states、position 信息和 mask。问题是 attention 不能只看本地 token，它还需要历史上下文里的 K/V。

所以 CP 的核心运行过程是：

```text
本地 token shard
  -> 本地生成 Q/K/V
  -> attention 期间跨 CP ranks 交换 K/V 或上下文信息
  -> 每个 rank 得到自己 token shard 的 attention output
```

RoPE、position ids、attention mask 也必须和 CP shard 对齐，否则不同 rank 对同一个 token 的位置理解会错。

和 Sequence Parallel 的区别：

```text
SP:
  主要在 TP group 内切中间 activation，省显存。

CP:
  直接把长上下文 token 分到 CP ranks，attention 内部要跨 rank 交换上下文。
```

不看代码时记住：CP 是“长上下文切片”，重点在 attention 如何跨 rank 拿到足够的 K/V。

## B. 代码带读版：Context Parallel 实现路径

### B1. 本节先读哪些文件

建议按这个顺序读：

```text
megatron/core/model_parallel_config.py
megatron/core/parallel_state.py
megatron/core/utils.py
megatron/core/models/common/embeddings/rope_utils.py
megatron/core/models/gpt/gpt_model.py
megatron/core/transformer/attention.py
megatron/core/extensions/transformer_engine.py
megatron/core/distributed/distributed_data_parallel.py
megatron/core/distributed/finalize_model_grads.py
```

第一遍不用追完所有 inference、MoE、Mamba、多模态路径。

本节只盯 dense GPT 训练主线里的 CP：

1. CP 配置如何进入并行组。
2. batch 如何按 CP rank 切 sequence。
3. RoPE 位置如何和 CP 切分对齐。
4. attention 如何把 CP group 交给 Transformer Engine。
5. 梯度同步为什么用 `dp_cp_group`。

---

### B2. CP 的配置入口

先看 `megatron/core/model_parallel_config.py`：

```python
context_parallel_size: int = 1
```

注释很直接：它把网络输入沿 sequence 维切到多个 GPU rank 上。

同一个配置附近还有几个相关字段：

```python
hierarchical_context_parallel_sizes
max_seqlen_per_dp_cp_rank
hybrid_context_parallel
```

第一遍只需要理解：

```text
context_parallel_size > 1
  => 开启普通 CP

hierarchical_context_parallel_sizes
  => 给分层 CP 通信用，比如 a2a+p2p

hybrid_context_parallel
  => 面向 packed / variable length 样本的负载均衡
```

这一节重点读普通 CP，hybrid CP 先知道它存在即可。

---

### B3. CP group 是怎么建出来的

进入 `megatron/core/parallel_state.py`。

初始化模型并行组时，`initialize_model_parallel` 有：

```python
context_parallel_size: int = 1
```

后面会计算：

```python
model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
data_parallel_size = world_size // model_size
```

这说明 CP 和 TP、PP 一样，属于 model-parallel 维度的一部分。

直观上：

```text
world_size
  = TP * CP * PP * DP
```

当 `context_parallel_size` 变大时，如果总 GPU 数不变，`data_parallel_size` 会变小。

再看函数：

```python
get_context_parallel_group()
get_tensor_and_context_parallel_group()
get_data_parallel_group(with_context_parallel=True)
```

这里有一个很重要的边界：

```text
cp_group
  attention 内交换上下文用

dp_cp_group
  梯度同步、token 数归一化、参数 broadcast 等训练同步用
```

CP 不是 DP，但 CP rank 持有的是同一个样本的不同 token 片段。
所以做梯度同步和 loss/token 归一化时，经常要把 DP 和 CP 放在一起看。

---

### B4. CP 怎么切 batch

核心在 `megatron/core/utils.py`：

```python
get_batch_on_this_cp_rank(...)
get_pretrain_batch_on_this_cp_rank(...)
get_sft_batch_on_this_cp_rank(...)
```

普通预训练路径看 `get_pretrain_batch_on_this_cp_rank`。

它不是简单地把 sequence 平均切成连续的 `cp_size` 段，而是切成：

```text
2 * cp_size 个 chunk
```

然后每个 CP rank 拿两个 chunk：

```python
index[0] = cp_rank
index[1] = 2 * cp_size - cp_rank - 1
```

如果 `cp_size = 2`，sequence 会先切成 4 段：

```text
chunk: 0  1  2  3

CP rank 0 拿: 0 和 3
CP rank 1 拿: 1 和 2
```

如果 `cp_size = 4`，sequence 会先切成 8 段：

```text
rank 0: chunk 0 和 7
rank 1: chunk 1 和 6
rank 2: chunk 2 和 5
rank 3: chunk 3 和 4
```

为什么要这么绕？

因为 causal attention 的计算量不是每个位置都一样。
越靠后的 token 能看见越多历史 token，attention 工作量更重。
如果简单连续切分，后面的 rank 会更忙。

这种前后配对的 zigzag 切法，是为了负载更均衡。

---

### B5. 被切的是哪些 batch 字段

`get_pretrain_batch_on_this_cp_rank` 会遍历 batch 里的 tensor。

普通字段默认沿 sequence 维切：

```text
tokens
labels
loss_mask
position_ids
```

这里的 sequence 维通常是 dim 1：

```text
[micro_batch, sequence]
```

attention mask 特殊一点：

```python
seq_dim = 2 if key == 'attention_mask' else 1
```

也就是说 mask 的 sequence 维位置不同，需要单独处理。

另外这些 metadata 不切：

```python
cu_seqlens
cu_seqlens_padded
max_seqlen
local_cp_size
hybrid_cp_group
```

第一遍读到这里要建立一个坐标：

```text
进入模型前:
  每个 CP rank 只拿本 rank 的 token 片段

不是进入模型后才临时 scatter。
```

这是 CP 和 SP 的一个重要差别。

---

### B6. packed sequence / SFT 路径

如果 batch 里有：

```python
cu_seqlens
```

`get_batch_on_this_cp_rank` 会认为这是 packed sequence 相关路径。

非 hybrid CP 会走：

```python
get_sft_batch_on_this_cp_rank
```

它调用 Transformer Engine 的：

```python
tex.thd_get_partitioned_indices(...)
```

然后对这些字段做 `index_select`：

```python
tokens
labels
loss_mask
position_ids
```

这里的关键不是平均切连续片段，而是让 TE 根据 THD packed sequence 的 `cu_seqlens` 计算当前 CP rank 应该拿哪些 token。

第一遍可以先记：

```text
pretrain fixed sequence:
  Megatron 自己用 zigzag chunk 切

SFT / packed THD:
  TE 根据 cu_seqlens 算 index
```

---

### B7. RoPE 也必须按 CP rank 对齐

batch token 被 CP 切开后，位置编码不能还用完整序列的连续前半段。

看 `megatron/core/models/common/embeddings/rope_utils.py`：

```python
get_pos_emb_on_this_cp_rank(pos_emb, seq_dim, cp_group)
```

它和 batch 切法一样，也是先 reshape 成：

```text
2 * cp_size 个 chunk
```

然后当前 CP rank 取：

```python
[cp_rank, 2 * cp_size - cp_rank - 1]
```

这保证：

```text
rank 0 拿 token chunk 0 和 7
rank 0 也拿 position chunk 0 和 7
```

如果 RoPE 位置和 token 片段错位，attention 语义就错了。

所以 CP 不只是切 token，还必须同步处理：

```text
tokens / labels / loss_mask / position_ids
attention_mask
rotary position embedding
```

---

### B8. GPTModel 里 CP 怎么传下去

看 `megatron/core/models/gpt/gpt_model.py`。

构造 RoPE 时会把 CP group 传进去：

```python
self.rotary_pos_emb = RotaryEmbedding(..., cp_group=self.pg_collection.cp)
```

forward 过程中，生成 RoPE 时也会考虑 packed sequence 的动态 CP group：

```python
cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None
```

后面调用 decoder：

```python
self.decoder(
    hidden_states=decoder_input,
    attention_mask=attention_mask,
    rotary_pos_emb=rotary_pos_emb,
    packed_seq_params=packed_seq_params,
    ...
)
```

这条线说明：

```text
batch 切分:
  utils.py 里完成

position / RoPE 对齐:
  rope_utils.py + gpt_model.py

attention 内 CP 通信:
  后面交给 attention 模块和 TE
```

---

### B9. attention 模块本身做什么

进入 `megatron/core/transformer/attention.py`。

这里的 `Attention` 初始化会接收：

```python
cp_comm_type: str | None = None
pg_collection: ProcessGroupCollection | None = None
```

然后构造 core attention：

```python
self.core_attention = build_module(
    ...,
    cp_comm_type=cp_comm_type,
    pg_collection=self.pg_collection,
)
```

真正 CP 通信通常不在这个 Python 文件里手写 ring/p2p 细节。
Megatron 把 CP group、通信类型和 packed sequence metadata 传给 Transformer Engine 的 DotProductAttention。

所以读 `attention.py` 时，不要期待在这里看到完整 CP 算法。
它更像是桥：

```text
Megatron Q/K/V
  |
  v
TEDotProductAttention
  |
  v
TE 内部根据 cp_group / cp_comm_type 做跨 CP 通信
```

---

### B10. Transformer Engine 入口

看 `megatron/core/extensions/transformer_engine.py` 里的 `TEDotProductAttention`。

当：

```python
self.config.context_parallel_size > 1
```

它会设置：

```python
extra_kwargs["cp_group"] = pg_collection.cp
extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(pg_collection.cp)
extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
```

如果 TE 版本支持，还会设置：

```python
extra_kwargs["cp_comm_type"] = cp_comm_type
```

默认通信类型可以是：

```text
p2p
```

也可能是分层通信：

```text
a2a+p2p
```

当 `cp_comm_type == "a2a+p2p"` 时，它会取：

```python
get_hierarchical_context_parallel_groups(...)
```

也就是前面配置里的 `hierarchical_context_parallel_sizes` 开始发挥作用。

第一遍不用深入 TE 内核。
只要知道 Megatron 这边传了四类东西：

```text
cp_group
cp_global_ranks
cp_stream
cp_comm_type
```

TE 拿这些信息完成 attention 内部的跨 CP K/V 交换。

---

### B11. packed_seq_params 里的动态 CP

看 `megatron/core/packed_seq_params.py`。

里面有：

```python
local_cp_size
cp_group
cu_seqlens_q
cu_seqlens_kv
cu_seqlens_q_padded
cu_seqlens_kv_padded
```

这说明 packed sequence 场景下，CP 不一定只是全局固定 group。
某些 batch 可能携带自己的动态 CP 信息。

`TEDotProductAttention.forward` 里会处理：

```python
if packed_seq_params.cp_group is not None:
    super().set_context_parallel_group(...)
elif packed_seq_params.local_cp_size is not None:
    super().set_context_parallel_group(None, None, None, self.cp_comm_type)
```

含义是：

```text
packed_seq_params.cp_group 有值:
  本次 attention 用动态 CP group

packed_seq_params.local_cp_size == 1 且 cp_group 为空:
  本次 attention 动态关闭 CP
```

第一遍可以不深挖 dynamic CP，只要知道 packed sequence 会让 CP group 变得更动态。

---

### B12. CP 对梯度同步的影响

CP rank 之间持有的是同一个训练样本的不同 token 片段。
因此，训练同步时不能只把传统 DP rank 放一起。

看 `megatron/core/distributed/distributed_data_parallel.py`。

初始化时会取：

```python
self.dp_group
self.dp_cp_group
self.intra_dp_cp_group
```

普通非 expert 参数的 buffer 使用：

```python
data_parallel_group = self.intra_dp_cp_group
```

参数 broadcast 时也会对非 expert 参数使用：

```python
data_parallel_group = self.dp_cp_group
```

这说明：

```text
有 CP 时，梯度同步的“数据并行等价集合”
不是单纯 DP group，
而是 DP x CP 组合后的 group。
```

再看 `megatron/core/distributed/finalize_model_grads.py`。

里面取：

```python
dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
```

做 token 数归一化时：

```python
torch.distributed.all_reduce(num_tokens, group=dp_cp_group)
```

因为每个 CP rank 只看了部分 token，本地 `num_tokens` 也只是局部 token 数。
要按全局有效 token 数缩放梯度，就必须跨 `dp_cp_group` 汇总。

---

### B13. CP 和 SP 的边界

这是本节最容易混的地方。

Sequence Parallel：

```text
依赖 TP。
切的是 transformer 层间 activation 的 sequence 维。
典型形状从 [S, B, H] 变成 [S/TP, B, H]。
通信在 TP group 内发生。
关键函数在 tensor_parallel/mappings.py 和 layers.py。
```

Context Parallel：

```text
不等同于 TP。
切的是输入上下文 token。
每个 CP rank 持有本样本的一部分 sequence。
attention 需要跨 CP group 交换上下文信息。
梯度同步、token 归一化常用 dp_cp_group。
关键入口在 utils.py、rope_utils.py、TE attention。
```

一句话区分：

```text
SP 是“层间 activation 怎么省”。
CP 是“长上下文 attention 怎么分摊”。
```

---

### B14. 一条完整数据流

把 CP 主线串起来：

```text
config.context_parallel_size > 1
  |
  v
parallel_state 建 cp_group / dp_cp_group / tp_cp_group
  |
  v
dataloader batch
  |
  v
get_batch_on_this_cp_rank
  |
  +--> pretrain: zigzag chunk
  |
  +--> packed SFT: TE thd_get_partitioned_indices
  |
  v
tokens / labels / loss_mask / position_ids / attention_mask 被切到当前 CP rank
  |
  v
GPTModel 生成 RoPE
  |
  v
rope_utils 按同样 CP 规则切 position embedding
  |
  v
Attention 生成 Q/K/V
  |
  v
TEDotProductAttention 拿 cp_group / cp_comm_type 做跨 CP attention 通信
  |
  v
loss / backward
  |
  v
DDP / finalize_model_grads 使用 dp_cp_group 同步梯度和 num_tokens
```

---

### B15. 读源码时的最短路径

第一遍建议只读这些点：

```text
1. model_parallel_config.py
   - context_parallel_size
   - hierarchical_context_parallel_sizes
   - hybrid_context_parallel

2. parallel_state.py
   - context_parallel_size 如何进入 model_size
   - get_context_parallel_group
   - get_data_parallel_group(with_context_parallel=True)

3. utils.py
   - get_batch_on_this_cp_rank
   - get_pretrain_batch_on_this_cp_rank
   - get_sft_batch_on_this_cp_rank

4. rope_utils.py
   - get_pos_emb_on_this_cp_rank

5. gpt_model.py
   - cp_group 如何传给 RotaryEmbedding
   - packed_seq_params.cp_group 如何传下去

6. transformer_engine.py
   - TEDotProductAttention 中 context_parallel_size > 1 的分支
   - cp_group / cp_comm_type / cp_stream

7. distributed_data_parallel.py / finalize_model_grads.py
   - dp_cp_group
   - num_tokens all-reduce
```

注意：`attention.py` 里更多是模块桥接，不是 CP 通信算法本体。

---

### B16. 自检问题

读完后用这些问题检查：

1. CP 和 SP 都切 sequence 相关维度，它们切的位置和目的分别是什么？
2. 为什么 CP 的预训练 batch 要切成 `2 * cp_size` 个 chunk，而不是简单切成 `cp_size` 段？
3. `cp_size = 4` 时，rank 2 会拿哪两个 chunk？
4. 为什么 RoPE position embedding 必须按 CP rank 同步切？
5. `cp_group` 和 `dp_cp_group` 分别服务什么通信？
6. 为什么 `num_tokens` 归一化要在 `dp_cp_group` 上 all-reduce？
7. Megatron 里 CP attention 的具体通信细节主要交给谁实现？

---

### B17. 下一步带读建议

下一节建议回到训练系统完整性，读：

```text
Distributed Checkpointing
```

原因是前面已经读完：

```text
DDP
Distributed Optimizer
SP
CP
```

此时参数、梯度、optimizer state、activation、sequence/context 都可能被切。
接下来最自然的问题就是：

```text
这些被切开的训练状态如何保存、加载、跨并行度 reshard？
```

这正是 Distributed Checkpointing 要解决的问题。

