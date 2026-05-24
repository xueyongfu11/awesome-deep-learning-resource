# Megatron Core 学习笔记

这个目录用于沉淀 Megatron Core 的分阶段学习计划、代码带读笔记和后续问题记录。

## 学习顺序

1. GPT 模型主线
   - 重点读 `ModuleSpec` 如何把 GPTModel 的固定骨架和 Attention、MLP、Norm、Linear 等具体实现解耦。
   - 目标是理解同一个 `GPTModel` 如何通过不同 spec/config 适配 local、TE、MoE、inference optimized、LLaMA/Qwen/Mixtral 等形态。
   - 笔记：[01-gpt-mainline.md](01-gpt-mainline.md)，以 ModuleSpec 设计思想为主线。

2. Tensor Parallel
   - 重点看 `ColumnParallelLinear`、`RowParallelLinear` 和 TP 通信映射。
   - 建议从 `megatron/core/tensor_parallel/layers.py` 和 `mappings.py` 入手。
   - 笔记：[02-tensor-parallel-linear.md](02-tensor-parallel-linear.md)，含“不看代码版”和代码带读版。

3. TransformerLayer
   - 重点看 `TransformerLayer` 如何用 residual、bias-dropout-add 和 layer norm 串起 attention/MLP。
   - 建议从 `megatron/core/transformer/transformer_layer.py` 和 `megatron/core/fusions/fused_bias_dropout.py` 入手。
   - 笔记：[03-transformer-layer.md](03-transformer-layer.md)，含“不看代码版”和代码带读版。

4. Pipeline Parallel 与 1F1B Schedule
   - 重点看 pipeline layer 切分，以及 non-interleaved pipeline schedule 中 warmup、1F1B、cooldown 如何组织 microbatch。
   - 建议从 `megatron/core/transformer/transformer_block.py`、`megatron/core/pipeline_parallel/schedules.py` 和 `p2p_communication.py` 入手。
   - 笔记：[04-pipeline-parallel.md](04-pipeline-parallel.md)，含“不看代码版”和代码带读版。

5. Optimizer、Distributed DDP 和 Distributed Optimizer
   - 重点看参数、梯度 bucket、普通 DDP 梯度同步，以及 distributed optimizer state 的组织方式。
   - 建议读 `megatron/core/distributed/` 和 `megatron/core/optimizer/`。
   - 笔记：[05-optimizer-and-distributed-ddp.md](05-optimizer-and-distributed-ddp.md)，含普通 DDP 和 Distributed Optimizer 带读。

6. Sequence Parallel
   - 重点看 TP group 内如何把 `[S, B, H]` 的 sequence 维切成 `[S/TP, B, H]`，以及 gather / reduce-scatter 如何配合线性层。
   - 建议从 `megatron/core/tensor_parallel/mappings.py`、`megatron/core/tensor_parallel/layers.py` 和 `megatron/core/distributed/finalize_model_grads.py` 入手。
   - 笔记：[06-sequence-parallel.md](06-sequence-parallel.md)，含“不看代码版”和代码带读版。

7. Context Parallel
   - 重点看长上下文下如何把输入 token 切到 CP rank，以及 attention 内如何通过 CP group 交换上下文信息。
   - 建议从 `megatron/core/utils.py`、`megatron/core/models/common/embeddings/rope_utils.py` 和 `megatron/core/extensions/transformer_engine.py` 入手。
   - 笔记：[07-context-parallel.md](07-context-parallel.md)，含“不看代码版”和代码带读版。

8. Communication Overlap
   - 重点看通信如何从“算完再同步”变成“边算边发”：DP 梯度 reduce、Distributed Optimizer 参数 all-gather、PP P2P、TP Linear 通信。
   - 建议从 `megatron/core/distributed/param_and_grad_buffer.py`、`megatron/core/distributed/distributed_data_parallel.py`、`megatron/core/pipeline_parallel/schedules.py` 和 `megatron/core/extensions/transformer_engine.py` 入手。
   - 笔记：[08-communication-overlap.md](08-communication-overlap.md)，含“不看代码版”和代码带读版。

9. MoE / Expert Parallel
   - 重点看 token routing、expert parallel group、expert 参数/梯度同步，以及 MoE 和 TP/PP/DP/CP 的组合关系。
   - 建议等 dense GPT 的并行和 overlap 主线清楚后再读。
   - 笔记：[09-moe-expert-parallel.md](09-moe-expert-parallel.md)，含“不看代码版”和代码带读版。

10. Transformer Engine / FP8
   - 重点看 Megatron Core 如何把 Linear、Attention、LayerNorm 等模块切到 TE 实现，以及 FP8/FP4 的量化、缩放和通信约束。
   - 建议在 TP/SP/CP/overlap 都清楚后再展开，否则容易被后端实现细节打散。
   - 笔记：[10-transformer-engine-and-fp8.md](10-transformer-engine-and-fp8.md)，含“不看代码版”和代码带读版。

11. 其他高级能力
   - Distributed checkpointing、activation recompute、inference、FSDP、MTP、CUDA graph。
   - 这些可以按训练需求和部署需求再分支阅读。

## 当前默认学习范围

- 已完成主线：GPT dense 本地实现、TP、PP、DDP/Distributed Optimizer、SP、CP、communication overlap、MoE/EP、Transformer Engine/FP8。
- 后续可继续展开：Distributed checkpointing、activation recompute、inference、FSDP、MTP、CUDA graph、interleaved pipeline schedule。
- 主要张量布局：Megatron Core 内部常用 `[sequence, batch, hidden]`，即 `[S, B, H]`。
