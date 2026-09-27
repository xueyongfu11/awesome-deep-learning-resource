# vLLM 中的 DeepSeek V4：架构、KV Cache 与 Kernel 优化

## 分享概览

本次分享围绕 vLLM 如何支持 DeepSeek V4 展开，主要包括四部分：

1. 回顾模型架构，重点介绍 mHC、Compressed Sparse Attention（CSA）、Highly Compressed Attention（HCA）以及 MoE 的变化。
2. 解释压缩注意力带来的 KV Cache 结构，以及 vLLM 如何对复杂、异构的缓存做统一调度与动态内存规划。
3. 介绍 C4A decode path 上的 kernel fusion 与 multi-stream 优化。
4. 讨论 KV Cache offloading、分布式部署、kernel 组织方式和后续社区路线。

分享者强调，DeepSeek V4 的支持是团队协作成果。分享的目的不只是列出已完成的实现，也包括解释实现背后的取舍、当前限制和后续可能的改进方向。

## 一、模型架构回顾

从整体上看，DeepSeek V4 仍然是 Transformer 架构，与 DeepSeek V3/V3.2 在大结构上大同小异，但有几个值得重点关注的变化。

### 1. mHC：增强残差流

mHC 主要作用于残差流，将过去的单流/单维残差表示扩展为多维残差表示。Attention 和 MoE 等主体模块仍然使用单流 hidden state，因此在进入这些模块前后会经过 mixing，将多维残差流转换为模块所需的表示，再混合回残差流。

从 vLLM 的模型支持角度看，这一变化主要影响 block 前后的 mixing 和 residual mixing，并不要求 Attention、MoE 的所有内部计算都随之重写。

### 2. 两类压缩注意力

DeepSeek V4 交错使用两类压缩注意力：

- **C4A / CSA（Compressed Sparse Attention）**：以 4:1 压缩 token，并通过 indexer 从压缩后的 KV entries 中选择相关项，执行稀疏注意力。
- **C128A / HCA（Highly Compressed Attention）**：以 128:1 高比例压缩 token，不做稀疏筛选，而是对全部高度压缩后的历史信息执行注意力。

两类机制共同服务于超长上下文：CSA 通过压缩和稀疏选择降低实际计算量，HCA 通过更高压缩比保留全局历史，同时显著减少 KV Cache 占用。

### 3. MoE：Hash Routing 与 MegaMoE

DeepSeek V4 的 MoE 有两点变化：

- **Hash routing**：模型前几层不完全依赖 gate 动态选择专家，而是通过固定 hash table 为 token 指定专家。讲者引用模型报告中的解释，认为这种设计可能让前几层更易训练。
- **MegaMoE**：把 MoE 中原先分散的计算、通信及周边小 kernel 融合到一个更大的 kernel 中。这样既减少 kernel launch 的 CPU 开销，也让接口更紧凑，更便于模型集成。

## 二、CSA：压缩、索引与滑动窗口

### 1. Token-level compressor

CSA 从原始 KV token 的 hidden states 出发，通过 token-level compressor 将若干连续 token 压缩成一个 compressed KV entry。以 C4A 为例，每 4 个原始 token 产生一个压缩 entry。

压缩不是简单的无状态分块。为了保留相邻压缩块之间的关联，C4A 的 compressor 使用重叠上下文：生成后一个压缩 entry 时，还会带上前一组 token 的状态。因此，compressor 本质上具有 sliding-window 式的数据依赖。

讲者给出的直观观察是：如果忽略压缩计算内部的细节，每个新压缩 entry 只依赖附近有限数量的 token。基于这一点，vLLM 可以把 compressor state 当成一种 Sliding Window Attention（SWA）状态来管理，而不必引入一套完全独立的环形缓冲区管理系统。

### 2. Lightning indexer 与 Top-K 选择

CSA 不会让主 Attention 直接访问所有压缩 entry，而是先通过 indexer 计算相关性：

1. KV token 的 hidden states 经另一套 token-level compressor，生成 compressed indexer keys。
2. 当前 query 生成 indexer query。
3. Multi-Query Attention 计算 index scores。
4. Top-K selector 选择最相关的 compressed KV entries。
5. 主 Attention 将选中的 compressed entries 与 sliding-window KV entries 拼接，执行 Shared Key-Value Multi-Query Attention。

主 Attention 的序列长度已经被压缩，因此 indexer 也必须在对应的压缩粒度上工作，否则 index score 无法与主 Attention 的 compressed entries 对齐。主 Attention 和 indexer 都有 compressor，但两者参数、压缩结果和缓存内容并不相同。

### 3. 为什么还需要 sliding window

压缩 entry 只有在积累到足够数量的 token 后才能生成。在一个压缩块尚未完成时，最新的局部 token 如果只依赖 compressed entries，就会丢失信息。因此，CSA 在 compressed KV 之外并联一条 sliding-window KV 路径：

- 尚未形成 compressed entry 的最近 token 由 sliding window 覆盖。
- 一旦形成 compressed entry，query 就可以同时访问局部窗口和历史压缩信息。

最终，局部精确信息和长程压缩信息被拼接后送入同一个注意力计算。

## 三、HCA：更高压缩比的全局注意力

HCA 的结构比 CSA 简单：

- 每 128 个原始 token 压缩成一个 heavily compressed KV entry。
- 不使用 indexer，也不做 Top-K 稀疏筛选。
- 当前 query 对所有 heavily compressed entries 做注意力。
- 同样并联 sliding-window KV，以覆盖尚未完成压缩的局部 token。

这里的 128 是模型训练阶段确定的压缩比，不等同于 vLLM 的逻辑 block size。vLLM 当前实现使用统一的 `block_size=256`；对于 C128A，一个逻辑 block 中只有 2 个存储 entry。

## 四、KV Cache 的容量计算

### 1. 统一逻辑 block 与实际存储 block

vLLM 对 DeepSeek V4 使用统一逻辑 block：

- `block_size=256`，按原始 token 定义。
- 调度、prefix cache hit 和逻辑 block 划分都以 256 个原始 token 为基本单位。
- 压缩后真正存入缓存的 entry 数称为 `storage_block_size`。

因此：

- C4A：每个逻辑 block 存 64 个压缩 entries。
- C128A：每个逻辑 block 存 2 个压缩 entries。

这种区分让调度层维持统一的 token 粒度，同时允许每类 Attention 使用不同的物理存储密度。

### 2. 幻灯片中的容量示例

在 `block_size=256` 的前提下：

| 缓存类型 | 条目维度与精度 | 每 block 大小 | 折算到原始 token |
| --- | --- | ---: | ---: |
| C4 indexer | FP16，`64 × 128 × 2B` | 16,384 B | 64 B/token |
| C4 indexer | FP8，`64 × 132B` | 8,448 B | 33 B/token |
| C4 attention | FP16，`64 × 512 × 2B` | 65,536 B | 256 B/token |
| C4 attention | FP8，`64 × 584B` | 37,376 B | 146 B/token |
| C128 attention | FP16，`2 × 512 × 2B` | 2,048 B | 8 B/token |
| C128 attention | FP8，`2 × 584B` | 1,168 B | 约 4.6 B/token |

C4 indexer 还支持 FP4 KV Cache，但默认不一定启用。FP8/FP4 的 entry 大小不是单纯的 `head_dim × 1B` 或 `×0.5B`，因为还包含量化 scale、布局和对齐开销。

对于 C4A 的 shared KV，DeepSeek V4 的量化布局与 V3.2 有细微差别：部分 positional information 与 shared KV 的存储方式变化，使 FP8 entry 的最终布局为幻灯片所示的 584 B。讲者建议以实际代码中的布局为准，不要只按抽象 head dimension 估算。

### 3. 一百万 token 的量级

幻灯片给出的 DeepSeek V4 Pro 示例中：

- 约 30 个 C4A 层，每层约 320 MiB/序列。
- 约 31 个 C128A 层，每层约 8 MiB/序列。
- FP16 下合计约 `30 × 320 MiB + 31 × 8 MiB ≈ 9.62 GiB/序列`。
- 在 1M context 下，约比 DSV3.2 小 8.7 倍。

分享中进一步指出，实际部署更推荐 FP8 KV Cache，容量可再接近减半；按讲者的粗略估计，1M token 的单序列 KV Cache 可能在约 5 GiB 的量级。该值是部署估算，不应替代特定硬件和配置下的实测。

## 五、混合 KV Cache 与动态内存共享

### 1. DeepSeek V4 Pro 的缓存组成

幻灯片把缓存分为两大类：

- **Full-attention caches**：这里更准确地说，是大小随 sequence length 线性增长的缓存，包括 C4A indexer、C4A main attention 和 C128A main attention。
- **Sliding-window caches**：每个 request 大体保持常数大小，包括普通 SWA、C4 index compressor state、C4 main compressor state 和 C128 main compressor state。

DeepSeek V4 Pro 示例包含 61 层。C4A 与 C128A 在模型层间交错分布；C4A 同时具有 indexer cache 和 main attention cache，C128A 只有高压缩的 main attention cache。

### 2. 为什么把 compressor state 当作 SWA

Compressor 在凑齐 4 个或 128 个 token 前，需要暂存前序状态。vLLM 把这些有限窗口状态统一建模为 SWA：

- C4 compressor：`window_size=8`。
- C128 compressor：`window_size=128`。

C4 使用 8-token window，是因为相邻 4-token 压缩块存在 overlap，需要保留前后两组 token 的状态。C128 的单块已经足够大，不采用同样的跨块 overlap，因此窗口就是 128。

这样做的工程收益是明显的：一旦 memory planning 完成，缓存分配、prefix caching、P/D 分离时的 KV transfer 以及后续 offloading，都可以复用既有 SWA/KV Cache 基础设施。

### 3. Flexible Memory Sharing

Hybrid KV Cache Manager 将不同类型缓存按物理 block 大小分组，大小接近的缓存共享同一块 GPU tensor pool。幻灯片把主要类型分成几组：

1. 随序列增长的 C128A、C4 indexer 和 C4A main attention blocks。
2. 普通 SWA blocks。
3. C128 compressor state（按 SWA 管理）。
4. C4 compressor state（按 SWA 管理）。

模型运行时，随 sequence length 增长的缓存和按 request 固定的状态可以动态争用显存：长序列多时，把更多空间给 full-attention 部分；短请求多时，给 per-request 的 sliding-window states 留出更多空间。

这种拼装无法完全避免 padding 和内部碎片。幻灯片以对齐单位 `P=576B` 展示了不同组中的浪费，但后几组大多是 per-request 常量，因此总体浪费通常仍可控。讲者也承认，这套 planning 已较复杂，未来希望进一步模块化并减少 padding。

### 4. 一处现场公式澄清

观众指出幻灯片中 compressor state 的公式看起来不一致。讲者先认为可能写错，随后重新核对并解释：

- C128A `kv_score` state：形状按 `(8, 2 × 512 × 4B)` 理解；两个 state 各 512 维，并以 FP32 保存。
- C4A `kv_score` state：形状按 `(4, 2 × 512 × 2 × 4B)` 理解；额外的 `×2` 来自相邻压缩块 overlap 时分别维护的两组权重/状态。

以上与最终幻灯片的视觉标注一致。现场先前的口头疑问不应被当作最终结论。

## 六、Kernel 优化：Fusion 与 Multi-stream

### 1. C4A decode 的两条主路径

C4A decode path 可以粗分为两条相对独立的流水线：

- **Main attention / default stream**：计算 compressor、主 Attention 的 Q/KV、写入 KV Cache，最后进入 Flash MLA。
- **Indexer stream**：计算 indexer compressor、indexer query/key、Indexer MQA 和 Top-K logits，产生 Top-K page indices/lengths，再交给 Flash MLA 做稀疏访问。

两条路径在 Top-K 结果被主 Attention 消费前基本没有数据依赖，因此可以放在不同 CUDA streams 上并行执行，提高 SM 利用率。

### 2. 纵向 fusion

对于连续的 element-wise 或 memory-bound 操作，可以沿数据流纵向融合。例如 compressor 路径中的：

`Compressor → RMSNorm → RoPE → FP8 quantization → KV Cache write`

融合后，一个 GPU work item 可以从输入一路处理到写回缓存，减少 kernel launch 次数，也减少 HBM 与寄存器之间的反复搬运。讲者提到，这类融合在适合的局部路径上可能带来约 2～4 倍甚至更高的局部加速，但不是整个模型端到端性能的等比例提升。

### 3. 横向 fusion

如果若干小 kernel 彼此没有依赖，而且单个 kernel 无法占满 GPU，也可以横向合并到同一次 launch：让不同 work items 分别处理 KV RMSNorm、Q RMSNorm 等任务。这样既降低 launch 开销，也提高一次调度中的 SM 覆盖率。

矩阵乘等重型 tensor-core 操作通常不与这些轻量操作强行融合；它们本身计算密集，保留独立 kernel 更合适。

### 4. RoPE 与 inverse RoPE

DeepSeek V4 的 shared KV 设计使 Q/K 的 positional encoding 与最终 value 路径之间存在特殊关系。decode 图中在 Flash MLA 后加入 inverse RoPE，再执行 FP8 quantization、batched matrix multiplication 和输出投影。讲者概括为：通过在输出侧抵消额外的 RoPE 影响，在数学上得到等价于只对 K 应用相应位置编码的效果。

### 5. APE

问答中提到图里的 APE。讲者将其解释为 **Absolute Positional Encoding**，并说明它与模型训练设计有关；现场没有继续展开公式细节。

## 七、关键源码入口

幻灯片给出的代码索引如下：

- 模型入口与定义：`models/deepseek_v4.py`
- 高层 Attention：`layers/deepseek_v4_attention.py`
  - 包含 indexer 与 main attention module
- CSA/HCA compressor：`layers/deepseek_compressor.py`
  - 包含 compressor state cache 与 metadata builder
- SWA KV Cache：`mla/sparse_mla.py`
  - 包含 SWA cache 与 metadata builder
- MegaMoE：内联在 `models/deepseek_v4.py`
- KV Cache layout：`core/kv_cache_utils.py:_get_kv_cache_config_deepseek_v4()`

讲者原本计划现场过代码，但由于分享和问答已持续约一个半小时，最后改为给出源码地图，方便大家按图索骥，或使用代码 Agent 辅助阅读。

## 八、重点问答整理

### Q1：CSA 与 SSM 有什么关系和区别？

两者都可以维护随序列更新的有限状态，因此 compressor state 在管理方式上可类比 SSM state。但它们并不相同：

- SSM state 往往表示整个序列当前的聚合状态，并不严格对应单个 token。
- CSA 的 compressed entries 仍按 token 分组产生，并持续随 sequence length 增长；只是在压缩时使用有限窗口的 residual/compressor state。
- 因此，CSA 的主 KV Cache 总量仍随序列长度增长，只是增长速度被压缩比显著降低。

### Q2：与线性注意力或其他稀疏注意力有什么区别？

讲者以线性注意力作对比：线性注意力通常把历史压入固定大小状态，而稀疏注意力的 KV Cache 总量仍随 sequence length 增长，只在每次计算时选择固定数量的块。CSA 同时采用两种手段：先压缩以减少缓存容量，再通过 indexer 的 Top-K 选择减少实际计算。

分享者还提到，vLLM 当前对某些 linear attention cache 的管理仍不够理想。状态块过大时，为了便于分配会使用较大的 block size，反而使 prefix cache 的命中粒度变粗。DeepSeek V4 的 compressor state 相对更小、更可控，因此更容易纳入统一 KV Cache 管理。

### Q3：128 压缩比是否因为 block size 是 128？

不是。128 是训练阶段确定的模型压缩比；推理框架的逻辑 `block_size` 是独立概念。当前 vLLM 实现固定使用 256 个原始 token 的逻辑 block，因此 C128A 每 block 恰好存 2 个 compressed entries。

### Q4：为什么 C4 compressor 的窗口是 8，而 C128 是 128？

C4A 每 4 token 产生一个压缩 entry，但相邻压缩块有重叠。为了保存前后两组状态，窗口需要覆盖 8 token。C128A 的压缩块已经足够大，训练设计中没有相同的 overlap，因此窗口为 128。

### Q5：Compressor state 能否重计算，而不是缓存？

原则上可以。如果上一层完整 hidden states 仍可取得，就能选择性重算 compressor state。DeepSeek 的报告也讨论过一种类似策略：只缓存随 sequence length 增长的主要 attention blocks，遇到 SWA 层时重算局部状态。

vLLM 当前选择把 compressor state 当作 SWA 缓存，因为 planning 完成后可以直接获得 prefix caching、P/D 分离和 KV transfer 的支持，整体工程复杂度更低。未来也可结合 checkpoint：每隔若干 blocks 保存一次局部状态，在缓存容量与重计算成本之间折中。

### Q6：两台 H100 是否适合 `TP=16`？

不推荐。即使权重分片整除问题可以在代码上处理，跨两台机器做 16 路 TP 会产生频繁、数据量较大的全张量通信，性能通常不理想。现场建议考虑单机内 `TP=8`，跨机使用 `DP=2`，并结合 EP/外部 EP 处理 MoE 专家。

### Q7：Prefix cache 命中粒度是多少？

当前仍按 256 个原始 token 命中。如果未来采用每 1024 token 保存一个 checkpoint 的方案，那么相应 checkpoint/prefix 命中粒度也会变为 1024。

### Q8：社区如何看通信与计算融合？

MegaMoE 就是一个重要例子：它把 MoE 内的计算、all-to-all 通信和周边小 kernel 融入一个 mega kernel。讲者认为 MoE 是当前主要通信瓶颈之一；其他更激进的通算融合还需结合社区 roadmap 和最新实现继续推进。

### Q9：KV Cache offloading 的计划是什么？

分享时社区正在推进 offloading，重点包括：

- 原生接入 Mooncake，使用其分布式 CPU memory pool 和一定的 disk offloading 能力。
- 在 vLLM 的 simple CPU offloading backend 上继续加入 DeepSeek V4 等 hybrid models。
- 后续可能提供简单的 disk-offloading backend，方便单机演示和使用。
- 长期重新审视 connector API，把公共逻辑抽象出来，使外部库更容易接入 NFS 或其他存储后端。

由于 compressor state 已统一建模为 SWA，hybrid model 的多层缓存可以尽量复用同一套 transfer/offload 流程。

### Q10：Mooncake 与 LMCache 会更倾向哪个？

社区目标不是只保留一个后端，而是把它们都作为 first-class connector，在相同场景下比较性能和易用性。分享者近期与 Mooncake 团队合作较多，因此该方向迭代更快；LMCache 团队也在推进独立进程与缓存服务等能力。

### Q11：是否会把 kernel 独立成单独算子库？

现场没有给出确定结论。拆库能隔离代码，但会增加版本同步和包管理成本；合在主仓库内则容易使代码庞大、kernel 选择逻辑隐蔽。一个可能方向是按模型组织 kernel，使每个模型的选择逻辑更清晰，减少不同模型和 KV layout 之间的隐性影响。

讲者观察到，追求极致性能往往需要模型特定 fusion，因此即使建立独立算子库，也不能完全消除模型级定制。

### Q12：后续会一直跟随 DeepSeek 的技术路线吗？

vLLM 的原则是跟随模型架构，为模型提供合适的支持，而不是预先绑定某一家模型路线。DeepSeek 报告本身也承认当前架构很复杂，后续代际可能会做简化；推理框架需要保留适应不同模型设计的能力。

### Q13：如何理解“scale your model”？

讲者把这个问题拆成训练和推理两方面。训练侧是模型参数继续增大；推理侧则是模型、负载、输入输出长度和集群规模同时增大。对于已经无法装入单机的大模型，系统设计需要转向分布式视角：

- 组合 TP、EP、DP，而不是只扩大单一并行维度。
- 在多组实例前增加 router/front end。
- 把 KV Cache offloading 从可选优化变成必要能力。
- 建立跨机器共享的分布式、多层级 KV Cache 池，避免请求换机后缓存全部 miss。
- 根据模型、硬件和 workload 给出可直接使用的部署 recipe。

主持人补充，不同负载、不同 GPU（如 H 系列和 B 系列）的最佳部署策略可能完全不同。普通用户往往没有精力穷举测试，也未必熟悉模型执行原理，因此社区应提供“足够好且开箱即用”的推荐配置，而不只是要求用户自己寻找理论最优解。

## 九、当前限制与后续方向

分享中反复提到的改进方向可以归纳为：

1. **Memory planning 模块化**：允许模型自定义 KV Cache planning，把 DeepSeek V4 的复杂逻辑放回模型相关代码，减少对其他模型的干扰。
2. **降低 padding 与碎片**：进一步改进不同 cache block 的分组与共享方式。
3. **更灵活的 prefix caching**：探索 checkpoint 与重计算的折中，提升可缓存上下文规模和命中率。
4. **Hybrid model offloading**：打通 Mooncake、LMCache、simple CPU offloading 与未来磁盘/网络存储后端。
5. **Kernel 持续融合**：继续推进模型特定的纵向、横向和通信-计算融合。
6. **Kernel 选择显式化**：减少由 KV layout、量化方式等隐性条件导致的算子选择困惑。
7. **部署 recipe**：针对硬件、模型和 workload 提供更可靠的参考配置。

## 十、核心结论

DeepSeek V4 对推理框架的挑战并不只是“增加一种 Attention”。它同时改变了模型层结构、KV Cache 的物理密度、每请求状态、Attention 数据流、kernel 组织方式以及多机缓存策略。

vLLM 当前实现的核心思路是尽量把新机制映射到已有抽象：以 256 个原始 token 维持统一逻辑 block，把 compressor state 视作 SWA，用 Hybrid KV Cache Manager 对异构缓存做分组和动态共享，再用 multi-stream 与 kernel fusion 消化复杂 decode path 的性能开销。

这套方案已经能复用 prefix caching、P/D 分离、KV transfer 和 offloading 基础设施，但 memory planning、padding、模型特定 kernel 选择和分布式 KV Cache 仍是需要继续演进的部分。分享中最重要的工程观点是：随着模型和部署规模扩大，模型架构与系统架构必须共同设计；“更多”不仅意味着线性扩容，也会带来新的缓存、路由和调度问题。
