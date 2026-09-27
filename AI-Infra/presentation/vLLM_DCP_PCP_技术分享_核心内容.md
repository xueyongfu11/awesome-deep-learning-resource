# vLLM DCP & PCP 技术分享：核心内容

## 一句话结论

CP（Context Parallelism，上下文并行）将长序列相关的计算和 KV Cache 分散到多卡：**PCP 主要并行 Prefill 输入序列以降低 TTFT，DCP 主要切分 Decode 阶段的 KV Cache，以消除冗余存储、提高可承载的上下文长度和整体吞吐。**

## 1. CP、PCP 与 DCP 的定位

- 长上下文推理的主要瓶颈是 Prefill 计算量和 Decode 阶段不断增长的 KV Cache；若每张卡保留完整 Cache，会产生显著的冗余显存占用。
- **PCP（Prefill Context Parallel）**：在 Prefill 阶段按序列维度切分请求，将不同 chunk 分给不同 PCP rank 处理，降低单卡计算负载和首 token 时延（TTFT）。
- **DCP（Decode Context Parallel）**：在 Decode 阶段按序列维度将 KV Cache 分片保存到不同 DCP rank。这样每卡只存部分 Cache，释放显存，支持更长上下文和更多并发请求。
- 两种并行可组合使用：PCP 处理输入序列的并行计算，DCP 负责 KV Cache 的分布与 Decode 时的协同注意力计算。

## 2. DCP 的关键实现

- DCP 的 Prefill 路径会把 Q/K/V 中的 KV 按 DCP rank 写入不同设备；KV Cache 的逻辑布局可理解为 `(seq / dcp, h)`，而不是在每张卡完整复制。
- 因为同一个逻辑 block 的 token 会跨设备保存，vLLM 引入 **virtual block（虚拟块）** 来统一管理跨卡 block；它影响 block 分配及 prefix caching 的命中判断。
- `slot mapping` 也必须随分布式存储改变：本卡持有的 token 映射到本地 slot，不在本卡的 token 用无效位置标识。后端据此把 KV 写到正确的物理 Cache 位置。
- `interleave_size` 决定 token 在 DCP rank 间轮转的粒度：
  - 值为 1 时逐 token round-robin，负载最均匀，但同一 block 的 token 最分散；
  - 值较大时连续多个 token 才换 rank，单个 block 内 token 更集中；
  - 在 PD 分离等需要按 block 传输 Cache 的场景，通常倾向令其接近 block size，以避免把一个完整 block 拆到多卡而增加通信与浪费。
- DCP 对普通 Prefill 的主流程影响较小，核心改动集中在 BlockTable/slot mapping 与 KV Cache 的写入布局。

## 3. DCP Decode 与 Attention 的通信

- Decode 时，当前卡的 Q 需要与分布在各 DCP rank 的完整 KV 信息协作计算 Attention；因此不能只读取本地 KV。
- 对 GQA，Q head 与 KV head 的切分需保持一致；GQA 的 KV head 少于 Q head，是 DCP 降低冗余存储的重要适用点。MLA 的 KV 压缩/共享特征也使该方案尤有价值。
- 标准流程会先做 Q 相关的 AllGather，再分别计算各卡持有的局部 KV Attention，得到局部输出与 LSE（online softmax 的中间状态）。
- 随后在 DCP group 内聚合结果，并通过 `correct_attn` / `merge_attn_states` 按 LSE 校正合并，恢复与原始 Attention 等价的输出。
- 实现上将原先“序列维度 AllGather + head 维度 ReduceScatter”等多步通信融合为更紧凑的 collective（图中为 All2All 等），减少通信次数和同步开销。
- 对 Chunked Prefill，需要额外处理当前 chunk 与历史 KV：可选择聚合 Q 或聚合 KV 的路径，并逐 chunk 合并 Attention 状态，确保因果注意力结果正确。

## 4. PCP 的核心机制

- PCP 通信组由相同 TP rank 的不同 PCP rank 组成；rank 布局同时考虑 TP、PCP 和 DCP 的层次关系。
- Prefill 请求在 `PCPManager` 中切分，除 token 外还需同步维护 `query_lens`、`kv_lens`、`positions`、以及 `q_head_idx/q_tail_idx` 等边界信息和 attention mask。
- 请求切分后，每个 PCP rank 承担部分序列。PCP size 增大时，单 rank 计算 token 减少；对于不等长请求，需要 padding 与边界 mask，避免无效 token 参与计算。
- PCP 的 Prefill 路径中，部分 Q/KV 通过 PCP group AllGather 补齐；结合模型类型（MLA、GQA/MHA）做 Q/KV 的重排、扩展或通信，再执行 Attention。
- PCP Decode 同样需要在 PCP/DCP/TP 的组合通信组中进行局部计算、聚合和校正，最终恢复标准 TP 维度的输出。

## 5. 部署收益与适用范围

- **DCP 的收益**：KV Cache 被切分后，每卡可用 Cache 容量明显增加；显存紧张时可承载更多并发请求，吞吐显著提升。长上下文场景可支持超过百万 token 的推理（PPT 示例为单机 A3）。
- **时延取舍**：短序列会因额外通信而有一定劣化；长序列（通常 128K 以上）中，独立的 KV Cache 计算可有效降低 TPOT。
- **PCP 的适用区间**：中长序列（约 32K–256K）下，时延通常优于 DP、略逊于 TP；吞吐优于 TP、略逊于 DP，处于 DP/TP 的折中位置。当并发量接近系统最佳吞吐时，PCP 有助于继续降低时延并兼顾吞吐。
- 部署应按模型、序列长度和并发量选择 PCP/DCP size，而不是固定采用最大并行度。

## 6. 适配、测试与限制

- 分享中给出 Ascend 侧的适配矩阵：相比原始 vLLM，Ascend 的 DCP/PCP 已覆盖或正在覆盖 Chunked Prefill、APC、MTP/EAGLE、Piecewise Graph、Full Graph、PD 分离、Qwen3（GQA）、DeepSeek（MLA）、Qwen3.5、DSV3.2 等更多组合；DSV4 仍未支持。
- 验证不能只跑常规短序列 benchmark。针对长序列、Chunked Prefill、PD 分离、特定模型结构与功能开关，都应构造能稳定触发目标路径的用例。
- 可通过类似 `long_sequence_threshold_tokens` 的阈值/测试开关，让请求进入长序列路径，以暴露通信、图编译或边界处理问题。
- PCP/DCP 可让引擎在资源层面支持超长序列；但若推理长度超过模型训练时的 context window，模型能力与效果仍可能下降。这是模型能力限制，不是并行引擎能完全解决的问题。

## 7. 后续优化方向

- **TPA（tensor-parallel-size-attention）**：将 `q_proj` 的切分维度从 TP 调整为 `TP/DCP`，减少 KV 计算前的一次 AllGather，降低同步开销。
- **动态 CP**：根据请求长度选择 CP size；短请求走 CP=1 的标准 DP 路径，长请求切换到更大的 CP，并通过跨 DP 域调度、统一 KV Block Pool 与后端协作完成执行。
- 分享中的实测结论是：在中短序列（4K–128K）和变长请求场景，动态 CP 的吞吐与 TTFT 均优于 DP，TPOT 没有明显劣化。

## 8. 问答中的补充结论

- 引擎层面可以通过 DCP/PCP 推理很长的序列；分享中提到早期已做过百万级长序列实验。
- 但“能推理”不等于模型在超出自身上下文训练范围后仍保持原有效果，实际模型质量可能下降。
- 团队建议沉淀不同模型、参数、部署组合的最佳适配信息，帮助使用者理解计算与通信之间的权衡，并降低后续开发和调优成本。

