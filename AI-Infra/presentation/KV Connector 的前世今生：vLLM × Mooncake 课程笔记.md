# KV Connector 的前世今生：vLLM × Mooncake 课程笔记

## 一句话概览

KV Connector 是 vLLM 面向“引擎外部”的 KV Cache 传输抽象：它让推理引擎可以把 KV Cache 从本地 GPU 的 paged cache 中取出，交给外部系统存储、压缩、共享或传输；需要时再加载回来。其意义不只是 prefix cache，而是把 KV Cache 逐步发展成可独立管理、可跨实例共享的数据层。

本次分享分两部分：易华介绍 KV Connector 从 CacheGen、vLLM V0 到 V1 的设计演进，以及 LMCache 的落地；嘉豪介绍 Mooncake 的 P/D（Prefill/Decode）分离、分布式 KV Cache Store 与后续 SSD/GDS 方向。

## 1. 起点：为什么要把 KV Cache 拿出推理引擎？

2023 年中，RAG 等应用刚开始流行。长请求产生的 KV Cache 体积很大；若要从磁盘或另一台机器读回它，传输耗时就会变成问题。CacheGen（SIGCOMM 2024）的思路是：先压缩 KV Cache，将压缩结果放到外部存储；同一 context prefix 再次到来时，取回、解压，再回填到推理过程。

最初原型基于 Hugging Face Transformers。它会直接暴露扁平的 PyTorch KV tensors，因此比较容易插入压缩、保存和恢复逻辑。但 vLLM 很快成为 LLM serving 的关键基础设施，团队希望让这套能力进入真正的推理引擎。

这里抽象出的通用问题是：

- 如何把 KV Cache 从推理引擎取出，交由外部系统做存储或变换？
- 如何在需要时将其放回，并让引擎跳过已命中的 prefix 计算？
- 如何让 vLLM 核心不必了解后端是 LMCache、Mooncake、磁盘、对象存储，还是另一 vLLM 实例？

## 2. 最初设计：最小改动与 chunk 管理

早期贡献开源社区的原则是尽可能缩小对 vLLM 的侵入：接口要简单、通用，且不假设 KV Cache 的具体 shape。外部 KV Cache 按 chunk 管理。

最早的接口可概括为：

- `store`：根据 token 或 request，将对应的 KV tensors 从当前 vLLM 卸载到外部介质；介质可以是内存、磁盘、远端机器等。
- `retrieve`：根据相同 token 找回外部 KV Cache，并交回 vLLM。

这套设计与 vLLM 的内部表示之间存在张力：vLLM 以 paged attention 著称，GPU HBM 中的 KV Cache 是许多离散 page，而外部系统通常以连续的 chunk 保存。因此：加载时要把连续外部数据 **scatter** 到离散 page；保存时要把 page 中的数据 **gather** 成连续 chunk。早期实现借助 attention metadata 和 slot mapping 完成这一搬运。

## 3. vLLM V0：通过重建输入实现缓存命中

V0 的调度器只知道本地 GPU 上的 KV Cache，不知道外部缓存。对于一个实际已有外部 prefix 命中的请求，调度器仍可能把它当作全新请求，并要求 worker 从头计算。

V0 Connector 的关键技巧是在即将执行 GPU 计算时“修正”模型输入：例如外部命中前 8,000 tokens、总请求为 10,000 tokens，Connector 重建 attention metadata，使 vLLM 认为前 8,000 tokens 已命中，只计算余下 2,000 tokens。课件中对应的核心接口为：

- `send_kv_caches_and_hidden_states`：prefill 后将新生成的 KV Cache（以及 P/D 分离需要的 hidden states）发出。
- `receive_kv_caches_and_hidden_states`：接收外部 KV Cache / hidden states，并补算未命中部分。
- `rebuild_model_input`：重建模型输入，使实际执行计划不再完全服从原调度结果。

因此，V0 中的 Connector 对 model runner 看起来是 send/receive；其背后可以连接 LMCache、Mooncake、磁盘、对象存储或其他 vLLM 实例。抽象层隔离了 KV 的来源与去向。

### V0 的局限

“重建 attention metadata”早期很有效，但也带来维护负担。随着 FlashAttention、FlashInfer、MLA、稀疏 MLA、index cache 等 attention backend 和 KV layout 不断增加，Connector 若持续直接构造各种 metadata，就会与模型/attention 抽象紧耦合，难以维护。

## 4. 从 V0 到 V1：三项架构变化

vLLM V1 不是简单的接口改名。课件总结了三座“大山”：

1. **Attention metadata 日益复杂**：不能再让 Connector 负责伪造或维护全部 backend 的 metadata。
2. **Scheduler 与 worker 进程分离**：V0 中 scheduler 跑在 worker 0，同进程传递调度数据比较直接；V1 将 scheduler 独立为进程，通过 IPC 共享 `SchedulerOutput`。
3. **Chunked prefill**：V0 倾向逐请求 prefill；V1 将多个请求 chunk 化、批处理执行，Connector 必须适配批量调度与加载。

V1 的设计原则是：Connector 不再替换原生流程，而是在原生 scheduler/worker 流程的明确位置注入逻辑；由 vLLM 原生分配 paged KV blocks，Connector 获得“外部缓存应该装入哪个 block”的信息。

## 5. vLLM V1 Connector 的端到端工作流

下图在课件中明确将控制路径分为 scheduler 进程和 worker 进程：scheduler 负责查询命中与 block 分配，worker 负责实际模型计算和数据读写。

1. Scheduler 对新请求询问 Connector：`get_num_new_matched_tokens()`，得到外部 KV Cache 可命中的 token 数。
2. Scheduler 仅为未命中部分分配或规划 KV blocks，并通过 `update_state_after_alloc()` 将已分配的 block 状态通知 Connector。
3. Scheduler 用 `build_connector_meta()` 打包 Connector 所需元数据，连同 `SchedulerOutput` 经 IPC 传给 worker。
4. Worker 调用 `start_load_kv()`，把外部已命中 KV 加载到 vLLM 分配的 paged cache 指定位置。
5. Worker 执行 `model.forward()`，只计算未命中的部分。
6. 新产生的 KV 通过保存路径写回外部系统；`wait_for_save()` 用于在需要时等待保存完成。

这个划分解决了 V0 的核心问题：调度器在调度前就能知道外部命中，worker 不必再靠篡改 attention metadata 来“纠正”调度结果。

## 6. 传输与计算的三种重叠范式

V1 Connector 不仅承担功能正确性，也将 I/O 隐藏在计算后面。

1. **逐层传输（layer-wise transfer）**：按 transformer layer 传输 KV；前一层数据到达即可开始计算，同时后续层继续传输。V1 首版已内置此机制。
2. **请求级异步（request-level async）**：请求 A 的 KV 仍在加载时，GPU 先运行其他已就绪请求，避免等待 I/O 而空转。这一机制由 Red Hat 工程师后续贡献到 V1 Connector。
3. **L2 预取（L2 prefetching）**：KV 尚远在远端时先预取到较近层级，但暂不占 GPU 显存；思路来自 Dynamo 团队，首先在 LMCache 落地。

三者分别利用 layer 间、request 间和存储层级间的并行性，目标都是缩短端到端时间而非只缩短单次 copy。

## 7. LMCache：Connector 另一端的 KV 管理层

LMCache 被描述为“驱动高性能 LLM 推理的 KV Cache 管理层”。其架构特点是将繁重的缓存管理交给独立 daemon：vLLM 进程发指令，daemon 执行跨层查找、预取、存储和数据移动。

- Scheduler 调用 `get_num_new_matched_tokens()` 时，会以 RPC 询问 LMCache daemon；daemon 可触发分布式、多级查找与预取。
- `update_state_after_alloc()` 同步 engine 侧的 block 分配，并上报给 daemon。
- Worker 的 load/store RPC 主要携带 token ids 和 block ids；实际内存拷贝由 LMCache 进程完成。

由此可以把 KV Cache 看作新的基础设施层：推理引擎可以横向复制计算实例以提高算力利用率；KV Cache Stack 则分布式保存数据、管理多级 I/O，并提供请求级的查询、删除、固定（pin）及可观测性。

课件列出的近期方向包括 KV-aware routing（将请求路由至缓存最热的实例）、跨副本/跨节点的全局 KV 池化、Mamba 等混合模型支持，以及 GPU/CPU/磁盘/远端间的智能分层放置。

## 8. Mooncake：从 P/D 直传到分布式 KV Cache Store

Mooncake 是一个以 KV Cache 为中心的分布式存储系统，包含两个重要子系统：

- **Transfer Engine（TE）**：基于 RDMA 的零拷贝传输。用于 P/D 分离时，可将 Prefill 节点 HBM 的数据直接传到 Decode 节点 HBM；也可将远程节点 DRAM 的数据传到本地 DRAM。性能主要受 RDMA 网络带宽限制，并支持多网卡池化和拓扑感知路径选择。
- **Mooncake Store**：由内存池及可扩展 SSD 池组成。它让 KV Cache 一次写入、全局可用，支持存储节点弹性增删，并可在源/目的内存之间直接发起 RDMA 读写。

课程区分了两个仍在演进的 Connector：

- `MooncakeConnector`：面向 P2P KV 直传，服务 P/D 分离。
- `MooncakeStoreConnector`：面向 KV Cache offloading 与跨实例共享，通过集群级 Mooncake Store 进行读写。

Transfer Engine 的下一代方向（TENT）包括从静态绑定转向动态加载、按实时 workload 与设备健康状态自适应路由，以及秒级故障检测和容错。

## 9. MooncakeStoreConnector 的查询、加载与键设计

### Scheduler 路径

对新请求，vLLM 将 prompt 分块并计算 `block_hashes`。随后：

1. 创建 `LookupKeyClient`，调用 `lookup(token_len, block_hashes)`。
2. Mooncake 侧把 hash 转成 pool keys，并查询 `MooncakeDistributedStore.batch_is_exist(keys)`。
3. Store 返回命中 token 数（例如命中 4 个 tokens）；结果回到 scheduler。
4. Scheduler 只为未命中 blocks 分配空间，并标记命中的 tokens 需要加载。
5. 各 GPU worker 的 `KVCacheStoreRecvingThread` 根据命中 hash 生成 keys，从 Mooncake Store 加载数据到 GPU；随后计算未命中 tokens。

### Worker 路径

每个 GPU worker 嵌入 Mooncake client，并由后台线程进行异步传输。KV Cache 注册为 RDMA buffer，通过 GPUDirect RDMA 直接读写，避免 GPU SM 参与和 CPU 中转。课件列出供 Connector 使用的端到端零拷贝 API：

- `batch_put_from` / `batch_get_into`
- `batch_put_from_multi_buffers` / `batch_get_into_multi_buffers`

键需要编码模型与并行布局，避免不同 tensor-parallel/pipeline-parallel rank 的数据冲突。例：一个 chunk hash 在 `tp_size=2, pp_size=2` 时会生成四个 key，包含 `model`、`tp_rank`、`pp_rank`、group 及 chunk hash 等字段。

这带来三项直接能力：CPU/磁盘 offloading、跨实例的哈希 prefix caching，以及 `kv_both` / XpYd 等灵活部署。

## 10. 存储层演进：SSD、GDS 与工作负载感知

Mooncake 近期重点之一是 GPU Direct Storage（GDS）和分布式文件系统支持。传统 SSD 路径通常是：写入先落 host DRAM，再异步刷到 SSD；读取也先经 DRAM buffer。引入 GDS 后，可复用现有 API，并行发起 GDS 与 DRAM 的写入，减少必须经过 DRAM 的数据路径。这是改动量较小的集成方式。

后续更深入的方案可能引入专门的 GDS replica，让 vLLM 可选择某些 KV 直接走 GDS，减少中间控制与拷贝。另一个方向是 workload-aware KV cache hints：上层通过 session ID、优先级等信息经 KV Connector 透传，Mooncake 据此决定哪些请求应常驻内存、哪些更适合淘汰。

## 11. 直播问答要点

### KV Cache 是否分布式部署？

GPU HBM 内部的 KV Cache 由每个 vLLM instance 自己管理；一旦 KV Cache 被 offload 到 GPU 外，才由 Connector 背后的系统负责管理、维护、存取与共享。

### LMCache 与 Mooncake Store 分别做什么？

Mooncake Store 提供高性能存取与分布式存储能力；LMCache 还可提供请求级 KV Cache 管理与更完整的管理层功能。两者可以在系统中协作，而不是简单互斥。

### RDMA、PCIe、GDR 分别对应什么路径？

- Prefill HBM 到 Decode HBM 的 P/D 直传使用 GPUDirect RDMA。
- GPU HBM 与远端节点 DRAM 的读写走 RDMA。
- SSD 相关路径涉及 PCIe；GDS 的目标是让 GPU 更直接访问存储。

分享时 Mooncake 的主要优化目标仍是 RDMA；TCP 没有特别多的专项优化。计算与通信 overlap 能否弥补 TCP/RDMA 差距，强依赖 workload 和实际带宽：带宽足够时差距可能缩小，带宽差异很大时 overlap 仍难完全覆盖。

### 能否只用 SSD，不用内存？

社区中已有使用 SPDK 与 NVMe-oF 池化 SSD 的方向，也已有相关集成，但分享时尚未完全覆盖“只用 SSD”的场景。全量落盘本质是在成本与性能之间取舍。当前常规路径先写 host memory，再异步落盘。

### 写 SSD 时 KV Cache 是否不可用？host OOM 如何避免？

若某份 KV 恰在异步卸载队列中，可能暂时不能直接访问那一份；但当前实现通常会先在 host memory 保留对应数据，因而多数情况下仍可从内存命中。host memory 通过近似 LRU 和高水位线触发淘汰，降低 OOM 风险。

### P/D 分离和异构/国产卡的难点？

部分国产卡不支持或难以稳定实现 CPU Direct RDMA，与 NVIDIA GPU 的互通性能也存在挑战。Mooncake 的思路是让国产卡与 NVIDIA 卡之间的 P/D 传输经过 Mooncake Store 做优化；这是仍在持续探索的方向。

### 什么是请求级 KV 管理？

它以“一个 request 的整段 KV Cache”为管理单位，而非把单个 KV block 作为用户可见对象。用户或上层可以表达“该 request 很重要，需要 pin”“该 request 要删除/查询”等需求，类似早期 prompt caching API 中按时长缓存一个请求。由此系统能做按 session、优先级或租户的策略。

### 为什么暂不做分布式显存池？

GPU HBM 通常应由推理引擎本地管理，且显存池化的经济成本很高；在算力紧缺环境中，存在大量闲置 GPU HBM 可供集中池化的场景并不常见。因此 Mooncake 分享时不支持分布式显存池，尽管大显存硬件是否适合该方向仍值得讨论。

## 12. 结论

KV Connector 的演进可概括为：从“把 KV 拿出去再拿回来”的最小接口，发展到与 vLLM scheduler/worker 原生流程协作的异步数据通道；再进一步成为连接分布式 KV 数据层、P/D 分离、多级存储和请求级策略的基础接口。

对使用者而言，最重要的边界是：vLLM 负责本地 paged KV cache、调度和计算；Connector 负责告诉引擎外部命中与数据如何进出；LMCache 或 Mooncake 等后端则负责数据的持久化、共享、传输、预取和生命周期管理。
