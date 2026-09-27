# Kimi K3 in vLLM 技术分享整理

来源：

- 语音识别：`语音识别结果.json` 中的 `text` 字段
- 参考资料：`Kimi K3 vLLM Tech Share.pdf`

说明：本文按直播讲述顺序整理，去除了口头停顿和明显重复；术语、数值和图示结构已结合 PDF 页面校正。

## 1. 分享主题与背景

这次分享的主题是 vLLM 如何支持 Kimi K3，并把性能优化到较好的水平。Kimi K3 在 vLLM 中的支持主要由 Jiangyun Zhu 和 Yongye Zhu 参与完成，也有 Inferact 团队其他同学共同贡献。

分享重点包括：

- Kimi K3 的架构变化；
- Kimi Delta Attention（KDA）对推理系统的挑战；
- vLLM 中 Hybrid Memory Allocator / Hybrid KV Cache Manager 的设计；
- KDA / Mamba 类状态模型下的 prefix caching；
- partial cache hit、selective retention 等缓存优化；
- DSpark speculative decoding；
- 小并发、低延迟场景下的 kernel 优化。

## 2. Kimi K3 架构概览

Kimi K3 相比 Kimi K2 有明显变化。Attention 从 K2 的 MLA 变成 Hybrid KDA-MLA，并在 MLA 上也做了一些修改；MoE 部分引入了 Latent MoE 设计。

PDF 中给出的 Kimi K2 与 Kimi K3 对比要点：

| 项目 | Kimi K2 | Kimi K3 |
| --- | ---: | ---: |
| 层数 | 61 | 93 |
| 总参数量 | 1.04T | 2.78T |
| 激活参数量 | 32.6B | 104.2B |
| Hidden dimension | 7168 | 7168 |
| Latent MoE dimension | - | 3584（0.5x） |
| MoE hidden dimension per expert | 2048 | 3072 |
| Routed experts | 384 | 896 |
| Experts active per token | 8 | 16 |
| Shared experts | 1 | 2 |
| Attention heads | 64 | 96 |
| Vocabulary size | 160K | 160K |
| Training context length | 128K | 1M |
| Attention mechanism | MLA | Hybrid KDA-MLA |
| Attention layer composition | 61 MLA | 69 KDA + 24 MLA |
| ViT | - | 401M 参数、27 层、patch size 14、12 heads |

K3 原生支持多模态输入。分享中提到，开源版本关于视频能力的表达需要谨慎理解：常见视频理解通常是抽帧后作为连续图像输入；DSpark 主要优化 decode 阶段，对 prefill 阶段的视频/图像编码收益有限。

## 3. 性能结果

分享中重点关注低并发、batch size = 1 的场景，因为这是很多 agent / reasoning workload 里更看重的单请求延迟和解码速度。

在 GB300 上，K3 decode throughput（batch = 1）表现如下：

| 配置 | Non-spec | DSpark |
| --- | ---: | ---: |
| TP8 | 111 tok/s | 410 tok/s |
| TP16 | 118 tok/s | 464 tok/s |

DSpark speculative decoding 带来约 3.7x 到 3.9x 的提升。分享中口头提到“TP16 可以达到 460+ tok/s”，对应 PDF 中的 464 tok/s。

## 4. Latent MoE：降低通信和计算

普通 MoE 的流程是 hidden state 经过 router 分发到不同 expert，再将 expert 输出汇总。问题在于 MoE 的通信量很大。

Latent MoE 的核心做法是在入口处先对 hidden state 降维，再在出口处升维回原维度。这样每个 token 在专家间传输的数据更小，计算量和通信量都会下降。

分享中解释了一个容易混淆的点：Latent MoE 外层的 down/up projection 和 MoE expert 内部原本的 up/down projection 不是同一个东西。Expert 内部的 up/down 作用在中间 FFN 维度；Latent MoE 外层的投影作用在 hidden state 维度，目的是减少进入 routed experts 的表示维度，从而降低通信和矩阵乘开销。

关于“降维是否影响质量”，分享者建议参考 Kimi K3 技术报告和 Latent MoE 相关论文，因为这是模型设计本身的问题，不是 vLLM 推理实现单独决定的。

## 5. Kimi Delta Attention 的系统挑战

Kimi K3 中重要创新之一是 Kimi Delta Attention（KDA）。KDA 来自 Kimi Linear 相关工作，其思想是维护一个状态 `S_t`，每来一个新 token 就更新状态，而不是像 full attention 那样持续追加 KV cache。

直观上，KDA / Mamba / linear attention 类模型有明显好处：

- 不需要维护随序列长度线性增长的完整 KV cache；
- 可以缓解 HBM 压力；
- decode 时复杂度不再像 full attention 那样直接受序列长度影响。

但它也打破了 vLLM 原有的核心假设：KV cache 是随序列连续增长、append-only 的。KDA 的状态会被下一步 in-place 覆写，因此不能随意回滚到任意前缀状态。

这带来两个主要系统问题：

- memory management：如何统一管理 full attention 的 KV cache 和 KDA/Mamba 的状态；
- prefix caching：如何在状态会被覆盖的模型里缓存并复用前缀状态。

分享中强调：SSM / linear attention 的建模优势，会复杂化系统层面的推理实现。

## 6. Hybrid Memory Allocator

vLLM 的做法是使用同一个 `KVCacheTensor`，在不同视图下解释成不同语义：

- Attention View：解释为 Key / Value pages；
- Mamba/KDA View：解释为 Conv / SSM 或 KDA states。

也就是说，底层物理分配是一套统一 tensor；上层 manager 根据模型和 attention 类型，把它解释成不同 cache/state。

具体分配策略使用 LCM page 思路：

- 将内存切成大小为所有可能 allocation size 的最小公倍数（LCM）的 page；
- 每种 cache 类型再把一个 LCM page 拆成多个对应大小的小 page；
- 例如 1 KB、2 KB、3 KB 三种类型，可以统一分配 6 KB 的 LCM page，再分别拆成 6 个 1 KB、3 个 2 KB 或 2 个 3 KB。

KDA / Mamba state 通常比 MLA 的 KV cache 大很多。因此实际实现中会把 attention page 不断扩大，直到 attention page 能覆盖 Mamba/KDA state size；如果仍有差距，则给 Mamba/KDA page 增加少量 padding，使 `page_size_bytes` 对齐。

这种对齐会浪费少量内存，但换来统一 allocator 和统一 cache manager 的管理便利。

## 7. Hybrid KV Cache Manager

PDF 中的结构是：

- 上层 `KVCacheManager`；
- 下面可以是 `UnitaryKVCacheCoordinator`、`HybridKVCacheCoordinator` 或 `KVCacheCoordinatorNoPrefixCache`；
- 再往下由不同 `SingleTypeKVCacheManager` 负责，例如 `FullAttentionManager`、`MambaManager`、`SlidingWindow` 等。

每个 manager 负责自身语义下的 cache 命中、block 生命周期、free 等问题，并向 coordinator 汇报可命中的 token 数。Coordinator 取多个 manager 的共同可命中范围，作为整个 request 真正可以复用的前缀。

这种设计的好处是扩展性：如果未来有新的 cache / state 类型，只需要新增对应 manager，并在 coordinator 里接入对应逻辑。

## 8. Mamba/KDA 状态能否复用

直播中有观众问：Mamba 这种状态更新模型，也能像自回归模型一样复用 KV cache 吗？

回答是可以，但代价不同。最朴素的做法是把每个位置的状态都存下来，命中到哪个位置就取出哪个状态。但问题是 state 通常比 KV cache 大很多，可能达到几百倍，如果每个 token 都存快照，内存压力会非常大。

更合理的方式是按一定粒度保存状态快照，例如每隔若干个 token 或在 block 边界保存。区别在于：

- full attention 的 KV cache 很容易回滚，因为历史 KV 都是 append-only；
- KDA / Mamba state 会 in-place 更新，不具备随意回滚能力；
- 缓存 KDA / Mamba state 的代价远高于缓存普通 KV。

## 9. Block-aligned Scheduling

为了解决 KDA / Mamba 类模型的 prefix caching，vLLM 使用 block-aligned scheduling。

两个前提：

- 状态必须绑定到 block hash，才能在 cache hit 时复用；
- linear attention 只自然产出最终状态，不能按任意 token 回滚。

解决方法是把每个 prefill chunk 对齐到 `block_size` 的整数倍，只有最后一个 chunk 可以不受约束。这样，每次 prefill chunk 结束时，state 正好落在 block boundary 上，可以缓存下来。

这个方案的优点是由 scheduler 驱动，不需要改模型层代码；同时可以在 GPU memory footprint 和 cache-hit rate 之间做权衡。

分享中补充：并不是“快照数量等于 block 数量”。例如一次 forward 可以处理 4 个 block size 的 token，但只在最后一个 block 边界保存一次 state，前面的 block 对应 state 可以为空。因为如果每个 block 都切开保存，就会变成多次 forward，效率不好。

## 10. 现有设计的问题：K3 的 block size 太大

Hybrid page size 对齐有一个副作用：不同 KV cache group 的 page size 必须对齐。如果一种 cache 远大于另一种 cache，小 cache 可能必须用很大的 block size 才能匹配。

K3 中 KDA state 约为 MLA cache 的 600 倍。在 DP 情况下，vLLM 里的 K3 block size 会达到 6000+ token。

这会让 prefix cache 变得低效。例如共享前缀有 5000 token，但还不到一个 6000+ token 的 block，就无法命中，只能重算。对于多轮对话或系统提示词复用场景，这种重算代价很高。

分享者强调，block size 不是越大越好。如果完全不考虑 cache hit，确实可以减少快照数量；但 prefix caching 的目的就是复用，所以 block 太大会降低命中率。

## 11. 最有价值的缓存位置

重新思考 prefix cache 的目标后，分享者认为真实业务中最有价值的缓存位置主要有两个：

- system prompt；
- 多轮对话中每一轮 turn 的边界。

在 ChatGPT 类多轮对话场景中，每轮请求会携带之前的对话历史。只要能缓存 system prompt 和每轮边界，就能覆盖很多真实复用机会，同时避免无谓缓存所有位置。

## 12. Partial Cache Hit：解耦命中粒度与 block size

Partial cache hit 的目标是把 cache-hit granularity 和 block size 解耦。该设计来自 vLLM RFC #45702，由 Moonshot 提出，并集成到 vLLM 框架中。

PDF 示例中，`block_size = 6`，`hash_block_size = 2`。

原设计：

- Request A 的 prompt 为 `a b c d e f g h`；
- full block key 只覆盖 `[a b c d e f]`；
- Request B 为 `a b c d e f g h x y`；
- 因为第二个 full block 没有填满并注册为 full-block key，`g h x y` 都要从 token 6 开始重算；
- 只能复用 6 tokens，重算 4 tokens。

新设计：

- 对最后未填满的尾部也注册 partial-tail key；
- Request A 中 `[g h]` 可以作为 partial tail 被缓存；
- Request B 到来时，在 full-block miss 后继续 probe partial-tail keys；
- 命中 `g h` 后，只需要重算 `x y`；
- 可以复用 8 tokens，重算 2 tokens。

实现上，partial 和 full 并不改变底层物理 block 的存储方式。底层仍然是一段 tensor；变化在于 cache manager 如何解释和标记它。没有填满的 block 会带上 tag，后续命中时需要注意它是 partial block。

因为 KDA / Mamba state 是 in-place 更新，partial block 命中后需要 copy-on-write：先把已命中的部分拷贝到新的 block，再继续写入更新，避免破坏原缓存状态。

关于 `hash_block_size` 的选择，分享者认为这是需要按 workload 调参的参数。粒度更小会带来更多复用机会，但也会增加管理和内存压力。当前没有自动自适应机制。

分享中提到，在此前 Qwen 3 05 的实验中，命中时 TTFT 约有 1.x 倍提升；K3 场景下由于 block size 特别大，partial cache hit 的收益预计更明显。该功能需要手动打开，具体用法见 RFC 和 PR。

## 13. Marconi-style Selective Retention

前面的 partial cache hit 主要解决多轮对话 turn boundary 的命中问题。另一个问题是 system prompt 如何命中。

Selective retention 的思路是：当下一个请求到来时，如果 MLA 部分命中、KDA 部分没有命中，就说明这段共享前缀有缓存价值。Scheduler 可以把 chunk 再切细，在这个边界保存 KDA state。

例如原本 aligned chunk 一次 forward 可能覆盖 8 份；如果第 6 份边界正好是 system prompt 的边界，原设计没法单独保存该状态。第二次请求暴露出 MLA 命中但 KDA 未命中后，scheduler 可以在后续切分时只切到第 6 份，并把这个边界的 state 保存下来。

分享中还提到两个工程问题：

- tensor reuse 可能导致 NaN，所以复用 block 前需要清零；
- 使用 PD 时，清零操作和 RDMA write 之间可能存在 data race，需要额外处理。

## 14. DSpark：Inferact/Kimi-K3-DSpark

Inferact 训练并开源了 `Inferact/Kimi-K3-DSpark`。分享中提到可以在 Hugging Face 上搜索 Inferact 相关仓库，也可以通过 vLLM blog 和 recipe 页面找到链接。

DSpark 是一种 speculative decoding 范式。分享中说它来自 DeepSeek 开源工作，使用 `deferring` 方式做 draft model，并给 deferring 引入线性依赖，以提高接受率。

Kimi K3 开源模型里没有原生可用的 MTP，因此需要 DSpark 这样的 speculative decoding 方案。使用 DSpark 后，batch = 1 的 decode throughput 接近 4 倍提升。

关于训练，分享中提到使用了 verl / roll-out 相关框架，把 training 和 inference 放在不同机器上，并传 hidden states；具体细节建议参考对应实现和上一期 DSpark 直播。

DSpark 的收益主要发生在 decode 阶段。对于图像/视频类输入，主要开销往往在 prefill，因此 DSpark 对这部分不直接加速。

## 15. 低延迟 Kernel 优化

低延迟优化主要针对小并发场景，尤其是 agent / reasoning 中希望单个请求越快越好的场景。

分享中列出的主要优化：

- 激进使用 Programmatic Dependent Launch（PDL）；
- 小并发场景下使用特化矩阵乘 kernel，而不是完全依赖通用矩阵库；
- 针对 Latent MoE tail 做专门优化；
- 还有一些未展开内容，例如 KDA kernel、CPU overhead profiling、draft batch 推理优化等。

## 16. Programmatic Dependent Launch（PDL）

没有 PDL 时，两个 kernel 顺序启动：第一个 kernel 结束后，第二个 kernel 才能启动。

PDL 的思想是在第一个 kernel 完全结束前就启动第二个 kernel。第二个 kernel 可以先执行与前一个 kernel 输出无关的工作，例如计算 index、thread/block 对应的数据位置等；等真正需要读取前一个 kernel 的输出时，再同步，确保可以看到写入 global memory 的数据。

正确性原则：

- 如果当前 kernel 后续要读取前一个 kernel 的输出，就在读取依赖数据之前插入 PDL / 同步；
- 与依赖数据无关的 index 计算等工作可以提前 overlap；
- 这样正确性可以保证。

性能上需要实际测量。若计算资源不足，overlap 可能拖慢前一个 kernel 的 good state，导致总时间不一定下降。

分享中估算，PDL 相关优化整体约有 3% 到 4% 提升。

## 17. Latent MoE Tail 优化

Latent MoE tail 的优化重点是：

- fuse routed experts 和 shared experts 的 reduction；
- 将 latent up projection 按列切分，使每个 rank 不再重复计算同一份 up projection。

原始问题出现在 Tensor Parallel 下。Routed experts 和 shared experts 都需要通信；小并发时 all-reduce 不友好，因为每个 token 数据量小，NVLink 带宽打不满，同时还增加 kernel launch。

优化思路：

1. Routed expert 部分必须先 all-reduce，再做 RMSNorm。因为 RMSNorm 是非线性计算，不能把各 rank 的部分结果独立算完后简单合成。
2. Shared expert 部分的目标只是升维后与对应部分相加，不必所有 rank 重复做完整 latent up projection。
3. 对 shared expert 使用 reduce-scatter，使每个 rank 拿到不同位置上的完整数据。
4. 将 latent up projection 按列切分，每个 rank 只计算必要列。
5. 每个 rank 把自己那部分 routed/up projection 结果和 shared expert 对应部分相加。
6. 最后再用一次集合通信，让所有 GPU 得到完整结果。

PDF 图中给出的 measured at TP8：

- before：13.200 us；
- after：10.620 us；
- 节省约 2.58 us，约 19.5%。

口头分享中估算，MoE tail 相关优化端到端在小并发场景下贡献约 9% 到 10%；特化小并发矩阵乘 kernel 约有 7% 到 8% 提升。

## 18. 其他工程与社区信息

分享中提到 vLLM 对 K3 的支持 PR 编号是 #50000。这个 PR 覆盖了本次直播的大多数内容，也包含一些直播中没有详细展开的改动。

vLLM recipe 网站支持选择不同硬件和并行策略，例如 GB300、B300、TP、DP+EP、spec decoding 等，并给出对应启动参数。绿色标识表示已经验证过的配置。

当前部署方面，分享时因为依赖较复杂，尤其有些 FlashInfer 相关 PR 尚未完全合入，所以更推荐使用已经 build 好的 Docker image。源码构建可以做，但安装会比较复杂；后续可能提供新的 image 或 nightly wheel。

## 19. 答疑整理

### DSpark 权重在哪里？

可以在 Hugging Face 上找 `Inferact/Kimi-K3-DSpark`。vLLM 的 K3 支持 blog 和 recipe 页面中也有链接。

### 为什么需要 DSpark？不是有原生 MTP 吗？

开源 Kimi K3 模型里没有可用的原生 MTP，所以这里训练并使用 DSpark。

### KDA 相关代码是否做了重构？

分享者说不能算完整重构，主要是把 KDA 相关定义和文件放到 K3 特定文件夹中，便于做更激进的优化，同时避免影响其他模型。

### 降维和不降维的收益如何？

这是 K3 模型设计层面的选择，建议看 K3 技术报告和 Latent MoE 相关论文。推理实现层面需要处理它带来的额外通信，例如 TP 下新增的 all-reduce，需要融合或 overlap。

### Mamba/KDA 的 cache 粒度是否绑定 chunk size？

是。vLLM 会保证 chunk 粒度是 block size 的整数倍，最后一个 chunk 除外。这样 state 可以落在 block boundary 上并缓存。

### token 数量不满一个 block 是否会被丢弃？

原设计中，不满一个完整 block 的尾部不会作为 full block 存到 KV cache，因此后续可能重算。Partial cache hit 的目的就是缓存这类未填满 block 的尾部。

### 为什么不直接减小 block size？

在当前 hybrid 设计中，attention page size 需要向上对齐到 KDA/Mamba state size。K3 中 KDA state 与 MLA cache 大小差距极大，所以不能简单把 block size 降下来。

### Hash block size 避免重算的部分是动态的吗？

可以理解为动态。系统会按 `hash_block_size` 粒度在 prompt 结尾保存 partial tail；例如粒度为 2 时，尾部 5 个 token 会在第 4 个 token 边界缓存，尾部 3 个 token 会在第 2 个 token 边界缓存。

### 输入很长时，计算哈希是否很慢？

vLLM 之前已经在哈希算法上做过优化，当前 overhead 应该较小。

### Partial cache hit 的 block 物理存储变了吗？

没有。物理上仍是同一段连续 tensor。Partial / full 只是 cache manager 对 tensor 的解释方式不同，会给未填满的 block 打 tag，后续命中时按 partial block 处理。

### 是否有 prefix block size 与实际 block size 的自适应机制？

分享者表示当前应该没有，需要根据场景手动调参。

### 开 DSpark 时，block 内每个 speculative token 是否都要临时存一份 KDA state？

是。因为 KDA state 无法像 full attention KV cache 那样轻松 rollback，所以每个 speculative token 都需要保存 state。

### PDL 的插入位置如何确定？

正确性上，在当前 kernel 读取前一个 kernel 输出之前插入即可。无依赖的 index 计算可以提前执行；性能上要 benchmark，因为 overlap 不一定总是收益。

### Index 是否在 kernel 内计算？

是。但这些 index 通常不依赖前一个 kernel 的实际输出，只依赖 thread/block id 和读写位置，因此可以提前算。

### K3 与 K2.5 多模态适配差异大吗？

分享者认为 ViT 部分和之前差异不大，代码层面可能有少量改动，但适配层面没有很大变化。

### Reduce-scatter 在多并发下是否仍有效？

分享者认为思路仍然成立。优化目标是避免 latent up projection 在每个 rank 上重复计算；reduce-scatter 本身需要根据数据量和算法选择，数据量小和大都有不同实现。

### 维度不整除怎么办？

会有 padding。不过当前面对的大多数维度，至少在 TP8 等常见配置下可以整除。

### H 系列 GPU 能否使用这些优化？

分享者表示上述优化基本没有使用 GB/B 系列特有 feature，H 系列卡应该也能用。Kimi K3 也可以在 H200 上跑，分享中提到跑过 16 卡 H200。

### Kimi K3 的 DSpark 是否针对视频加速？

没有直接针对视频。视频理解通常是抽帧后做图像理解，主要发生在 prefill 阶段；DSpark 主要优化 decode 阶段。

### vLLM 新 feature 上线如何保证一致性？

这个问题直播中没有展开，分享者也表示没完全理解提问。

### 推理 MoE 都走 TP，不走 EP 吗？

需要具体情况具体分析。小并发、追求低延迟时 TP 可能更合适；中并发或大并发时可能需要 EP 来降低通信。K3 还涉及 DCP 支持，可减少 MLA 在 TP 下每个 rank 重复存 KV cache 的问题，从而提升并发能力。

### B300 多并发情况下总吞吐最高多少？

分享者表示没有测过完整数据，粗略猜测每 GPU 至少 1000 token/s 级别，但需要实测后再给结果。

### B300 成本这么高，服务能回本吗？

分享者认为本次优化主要是探索模型在 vLLM 中能跑多快，很多优化不是只针对 K3，类似模型也可以复用。是否回本取决于实际服务场景和更细致的线上优化。

### DSpark 随 batch 增大加速效果会下降吗？

原理上会下降，但具体阈值取决于机器和负载。DSpark 接受率也受数据集影响，例如代码生成可能更容易预测，写作类任务更难预测。Batch size 变大时 draft token 数量也需要相应调整。

### 小并发指 batch 小，还是 batch × sequence length 小？

这里主要指 batch 比较小。

### 单请求长上下文下速度会衰减多少？

分享者没有测到 100 万序列这么长。推测不会衰减太多，因为 K3 大部分层是 KDA，KDA decode 不会像 MLA/full attention 那样强依赖序列长度；只有 MLA 部分会受长上下文影响更明显。

### DSpark 和 MTP 能否一起开？

分享者没有尝试过，认为需要较大代码改动。

### 如何入门 inference / vLLM？

建议从修 bug、读 PR、自己实现简化版功能开始。vLLM 社区也有 nano 类框架可用于理解核心机制。对于 KV cache、MoE、scheduler 等核心方向，需要逐步积累经验。

## 20. 参考链接

PDF 最后一页列出的参考资料：

- https://vllm.ai/blog/2026-07-27-k3
- https://recipes.vllm.ai/moonshotai/Kimi-K3
- https://mlsys.org/virtual/2025/poster/3260
- https://www.kimi.com/blog/kimi-k3
- https://arxiv.org/abs/2601.18089
- https://github.com/vllm-project/vllm/pull/50000
- https://zhiyuan1i.github.io/posts/kda-mathematics
- https://github.com/MoonshotAI/Kimi-Linear
- https://github.com/MoonshotAI/Kimi-K3
- https://huggingface.co/Inferact/Kimi-K3-DSpark
