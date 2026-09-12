# NVIDIA Dynamo Frontend 深度解析：从请求入口到推理集群调度内核

在 NVIDIA Dynamo 1.0.1 中，Frontend 已经超出传统 API Gateway 的职责边界。它既处理请求接入和流式回包，也负责 KV Cache 感知路由、Prefill/Decode（PD）链路编排、Worker 选择以及多推理引擎协议适配。

因此，更准确的理解是：**Frontend 是位于请求关键路径上的推理集群调度内核。** Worker 决定 token 如何计算，Frontend 则决定计算是否值得发生、应该在哪里发生，以及失败时如何降级。

本文以 1.0.1 的 `frontend-embedded` 架构为基线，重点分析它的职责边界、核心机制和工程代价。

## 一、先建立版本与对照基线

### 1.1 为什么必须锁定版本

Dynamo 不同阶段的差异不只是实现细节，而是组件边界本身发生了变化：

- 0.7 及以前：Frontend、KV Router、Prefill Router 相对分离
- 0.8～0.9：能力逐步收敛，部分引擎仍保留外置链路
- 1.0.1：进入 `frontend-embedded`，路由、PD 决策与协议适配进一步内聚

如果混用不同版本的文档，很容易得到一套实际上并不存在的组合架构。因此，讨论 Frontend 前必须先明确版本，本文只分析 1.0.1。

### 1.2 llm-d 是什么

llm-d 是一个面向 Kubernetes 的分布式大模型推理服务栈。它不替代 vLLM、SGLang、TensorRT-LLM 等推理引擎，而是在这些 Model Server 之上提供集群编排与智能调度能力。

其核心链路可以概括为：

```text
客户端 -> Proxy/Gateway -> EPP -> InferencePool -> Model Server
```

- **Proxy/Gateway**：负责连接管理、TLS 终止和请求转发
- **EPP（Endpoint Picker）**：根据 KV Cache 亲和性、队列深度、实例负载和请求优先级选择后端
- **InferencePool**：通过 Kubernetes API 描述一组服务同一模型的推理实例
- **Model Server**：执行 Prefill、Decode 和 token 生成

在 PD 分离场景中，EPP 还负责判断请求是否需要拆分，并选择 Prefill 与 Decode Worker。llm-d 和 Dynamo 都在解决推理集群的全局最优问题，但系统边界不同：**llm-d 强调 Proxy、EPP 与计算层的模块化组合；Dynamo 1.0.1 则把相近的状态感知和调度决策收拢进 Frontend。**

这也是在多编排框架统一抽象时，需要对齐 KV-aware、PD 编排与 SLA/SLO 表达，而不能简单对齐组件名称的原因。

## 二、Frontend 的真实职责边界

传统 Gateway 主要处理认证、限流、连接管理、负载均衡和协议转发。Dynamo Frontend 还要理解一次推理请求的计算成本与执行状态。

| 层面 | Frontend 的主要职责 |
| --- | --- |
| 数据面 | 请求接入、预处理、协议编解码、流式回包 |
| 控制面 | 订阅 Worker 与 KV 状态，维护本地索引 |
| 调度面 | KV-aware 路由、PD 编排、P/D Worker 配对与降级 |
| 适配面 | 将统一调度结果翻译成 vLLM、SGLang、TRT-LLM 所需协议 |

一条 PD 请求链路可简化为：

```text
业务客户端
  -> Frontend
  -> Prefill Worker
  -> KV 传输
  -> Decode Worker
  -> Frontend 流式回包
  -> 客户端
```

Frontend 的价值不在于把请求“转发出去”，而在于回答以下问题：

- 请求应该落到哪个 GPU 节点
- 当前上下文能否复用已有 KV Cache
- 在既定 PD 拓扑中如何组织 Prefill 与 Decode
- P/D Worker 如何配对
- 不同引擎如何表达同一份调度结果
- 节点、缓存或传输链路异常时如何回退

这些问题共同依赖请求内容、缓存分布、节点负载和引擎类型。它们如果分散在多个进程中，请求链路会增加 RPC、状态同步和排障成本；如果交给 Worker，又只能得到局部视角。1.0.1 将其收敛到 Frontend，本质上是在为时延敏感的决策路径减少协调开销。

## 三、KV-aware 路由：调度历史计算，而非平均流量

### 3.1 为什么传统负载均衡不够

传统负载均衡通常假设请求成本相近，因此追求连接数或请求数的平均分布。但推理请求并不等价：

- 输入和输出 token 数量差异很大
- 长上下文 Prefill 的计算成本很高
- 前缀缓存是否命中会减少 Prefill 计算，但其对 TTFT 的实际收益还取决于 KV 传输与同步成本
- Decode 队列长度会影响持续生成速度和尾延迟

因此，请求最少的 Worker 未必是成本最低的 Worker。KV-aware 路由需要把已经计算出的 KV Cache 视为集群资产，优先选择能够复用历史计算、同时 Decode 压力可接受的节点。

### 3.2 Frontend 如何完成决策

Frontend 在本地维护两类互补状态：

- `TrieIndex`：记录 KV Block 前缀与 Worker 的映射，用于快速计算前缀重合度
- `worker_state`：记录各 Worker 的队列、负载等动态状态

请求到达时，Frontend 合并“静态前缀匹配”和“动态运行负载”，在本地完成选择。这样既避免请求路径上的远程查询，也可以解释每次选择的依据。

[NVIDIA 路由概念文档](https://docs.nvidia.com/dynamo/dev/components/router/routing-concepts)给出的核心代价模型可以简化为：

```text
cost = overlap_score_weight × prefill_blocks + decode_blocks
```

它同时考虑两类主要成本：

- 新增 Prefill 计算量
- 当前 Decode 压力

公式本身并不复杂，难点在参数如何匹配真实 workload：偏重 TTFT 和长上下文复用时，应提高缓存命中的影响；偏重整体吞吐时，则要更积极地避免 Decode 热点。

### 3.3 Router 背压：Pending 不等于 Worker Waiting

Frontend 不是直接根据 GPU 利用率判断 Prefill Worker 是否还能接收请求。在启用 Router Queue 和 Prefill token 负载跟踪时，Router 会为每个 Worker 跟踪已接收或正在处理的 `active_prefill_tokens`，其排队判断线为：

```text
active_prefill_tokens > router_queue_threshold × max_num_batched_tokens
```

只要至少有一个候选 Prefill Worker 未超过阈值，Router 仍可继续分发；只有当所有候选 Worker 都超过判断线时，新请求才会留在 Frontend Pending Queue。这是基于 token 数的负载模型，不是对 GPU 剩余算力的精确测量。KV-aware 评分决定“在可分发的 Worker 中选谁”，Queue Threshold 决定“是否先排队”，两者不应混为一个决策。

请求在关键路径上会经过三种不同状态：

```text
Frontend Router Pending（尚未分发）
  -> Prefill vLLM Waiting（已进入 Worker，等待调度）
  -> Prefill vLLM Running（正在执行）
```

`dynamo_frontend_router_queue_pending_requests{worker_type="prefill"}` 是当前队列长度的 Gauge，不是累计请求数，也不包含正在执行的请求。由于 Router 可以在 Worker 前施加背压，线上完全可能出现 Frontend `pending > 0`、但 vLLM `waiting = 0` 的情况：后续请求尚未被发给 Worker，因此不会出现在 Worker 内部队列中。

还要区分两类同名但用途不同的阈值：`--router-queue-threshold` 决定请求何时进入 Router 队列，而监控系统中的 `prefill_queue_pending_requests_threshold` 只决定 Pending 积累到什么程度才告警。

### 3.4 真正的难点是状态新鲜度

索引显示“命中”并不代表缓存此刻仍然可用。线上环境中，Worker 上下线、显存淘汰、热点挤出和缓存引用变化都会使本地视图过期。

因此，KV-aware 路由的可靠性主要取决于：

- KV 状态事件能否及时到达
- 乱序和重复事件能否正确处理
- Worker 摘除是否足够快
- 错误命中后是否存在重试或重算路径

相比设计一个更复杂的评分公式，维持一份足够新且可诊断的本地状态通常更困难，也更影响生产效果。

## 四、PD 分离：何时拆、拆给谁、失败怎么办

PD 分离不是简单地把两个阶段部署在不同节点。完整编排至少包括：

1. 部署是否采用 PD 拓扑
2. 选择哪个 Prefill Worker
3. 选择哪个 Decode Worker
4. 如何建立 KV 传输会话
5. 任一环节失败后如何回退

单个 Worker 只能看到自身负载和缓存，无法判断整个 P/D 池的资源分布。因此，Worker 选择、P/D 配对和传输元数据交接需要由掌握请求上下文与集群状态的 Frontend 统一完成。

### 4.1 并不逐请求判断是否启用 PD

在固定 PD 部署中，只要发现 Prefill Worker 并激活 `PrefillRouter`，1.0.1 就会先执行 Prefill，再进入 Decode。`decode_fallback` 只在 Prefill Worker 不可用或 Prefill 失败时回退到 Decode，并不根据上下文长度、缓存命中率或传输成本动态跳过 P。

所以，“是否值得 PD”在该版本主要是部署和容量规划问题，需要根据以下因素预先评估：

- 上下文长度与 Prefill 计算量
- 输出长度和 Decode 并发压力
- P→D 网络带宽与拓扑
- P/D 独立扩缩容带来的收益

PD 并非默认更优。它增加了跨节点传输、会话编排、超时和故障恢复成本，只有预期收益高于这些额外成本时才适合启用。

### 4.2 传输元数据与跨节点会话

当 Prefill 和 Decode 位于不同节点时，系统需要确认“这份 KV 属于哪个请求、应该传给哪个对端”。不同后端使用不同的传输元数据：

- SGLang 使用包含 host、port 和 `bootstrap_room` 的 `bootstrap_info`
- vLLM 使用包含 Block ID 与远端连接信息的 `kv_transfer_params`
- TRT-LLM 使用序列化的 `opaque_state`

其中，`bootstrap_room` 是 SGLang P/D 两端的请求级会话标识；Frontend 的 `PrefillRouter` 可以生成该标识，并将同一份 Bootstrap 信息交给两端。

### 4.3 P→D KV 传输：成本、增量与 KV 感知

KV Cache 体积随上下文长度线性增长，其大小近似为：

```text
KV Bytes ≈ token 数 × 层数 × 2（K/V）× KV Heads × Head Dimension × 数据类型字节数
```

例如，对 80 层、8 个 KV Head、Head Dimension 为 128 的 BF16 GQA 模型，KV 约为 320 KiB/token：8K 上下文约产生 2.5 GiB KV。即使链路带宽为 400 Gb/s，2.5 GiB 的纯理论传输下限也在 50 ms 左右，实际还要叠加握手、内存注册、协议效率和网络拥塞。因此，P→D 传输完全可能成为 PD 的主要成本。

Dynamo 1.0.1 的官方 vLLM 和 SGLang 集成示例都选择 NIXL，在显存之间直接传输 KV，并优先利用同机 NVLink 或跨机 InfiniBand/UCX、GPUDirect RDMA，避免 GPU→CPU→网络→CPU→GPU 的绕行。官方所说的 non-blocking，表示传输期间 GPU 仍能处理其他工作，并不表示当前请求无需等待所需 KV 到达。

KV 感知实际存在于三个不同层面：

| 环节 | 1.0.1 默认行为 | 目的 |
| --- | --- | --- |
| Prefill 路由 | KV-aware，同时考虑负载 | 复用 P 上已有前缀，减少 Prefill 计算 |
| Decode 路由 | 不使用 KV 前缀评分，主要按 `potential_decode_blocks` 选低负载 D | 均衡显存与长期生成压力 |
| vLLM P→D 传输 | D 检查本地前缀，只拉取缺失 Block | 减少实际网络传输量 |

进入 Decode 路由前，1.0.1 的 `PrefillRouter` 会设置：

```text
overlap_score_weight = 0
assume_kv_reuse = false
```

因此 Decode 侧的通用评分公式退化为：

```text
decode_cost = potential_decode_blocks
```

也就是说，Dynamo 默认选择预计 KV Block 占用更低的 Decode Worker，而不会主动选择“已有最长前缀”的 D。这是路由层面的负载优先策略。

但在该版本配套的 vLLM 0.16.0 中，NIXL Connector 会在 D 内部检查本地 Prefix Cache，并按 Block 增量拉取：

```text
D 无命中：传输整个 Prompt 的 KV
D 部分命中：只传缺失的后缀 KV Block
D 完全命中：不传 KV 数据，只通知 P 释放相关 Block
```

这里的“增量”是**相对于 D 已有前缀只传缺失 Block**，不是每生成一个 token 就从 P 向 D 传一次。由于 Decode 路由默认不感知 D 的前缀分布，这种命中是引擎收到请求后的局部优化，而不是路由器主动保证的结果：如果选中的低负载 D 没有该前缀，仍需传输完整 Prompt KV。

这套行为也依赖具体后端。vLLM、SGLang 和 TRT-LLM 的同步方式与传输协议不同，不能把 vLLM 的 Block 级增量能力直接视为所有后端的统一保证。可参考 [Dynamo 1.0.1 PD 设计](https://docs.nvidia.com/dynamo/v1.0.1/design-docs/disaggregated-serving)、[Dynamo 1.0.1 Decode 路由覆盖逻辑](https://github.com/ai-dynamo/dynamo/blob/v1.0.1/lib/llm/src/kv_router/prefill_router.rs#L709-L720)与 [vLLM 0.16.0 NIXL Connector](https://github.com/vllm-project/vllm/blob/v0.16.0/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py#L2216-L2290)。

### 4.4 实测验证：缓存命中不等于 TTFT 同比例下降

前缀命中率只描述避免了多少重复计算，并不描述跨节点还要搬运多少数据。PD 场景中的首 Token 延迟可以近似拆成：

```text
TTFT ≈ 排队 + 路由 + Prefill 计算 + P→D KV 传输 + Decode 侧准备与首 Token 计算
```

因此，Prefix Cache 命中后只是 `Prefill 计算` 显著缩短；如果 P 仍向 D 发送完整上下文 KV，传输、同步及相关等待就会接替计算成为主导瓶颈。实测表明，全量传输 KV 时，即使 Prefill 几乎完全命中缓存，TTFT 也可能只有有限改善；而当 D 能复用已有 KV，或者链路只传输缺失 Block 时，缓存命中才能更充分地转化为端到端时延收益。

核心结论是：KV-aware 不能只优化 P 侧的历史计算复用，还必须与 D 侧的 KV 复用和增量传输协同，否则系统只是将瓶颈从 Prefill 计算转移到通信和同步。正式评测仍应固定采样参数、请求顺序和初始缓存状态，并通过多轮交叉实验采集链路吞吐、GPU 利用率及各阶段时间戳。

这也解释了为什么 PD 不能只用单请求 TTFT 评价。PD 更稳定的价值通常体现在高并发、长短请求混部时：Prefill 与 Decode 不再争用同一份算力和显存，因而可能改善吞吐、TPOT、ITL 及其 P95/P99，减少长 Prompt 对其他请求 Decode 的干扰，并通过独立扩缩容或 KV Offload 提升容量。这些收益在低并发、短输出测试中未必充分显现。

调度参数同样需要结合 workload 验证。不同请求长度和缓存状态下，Overlap Schedule 等开关可能呈现不同结果；样本量有限时，不应据个别 Case 断言某个调度选项一定有利，应以多轮 A/B 交叉结果为准。

### 4.5 从监控指标定位 P/D 瓶颈

PD 拆分后，“请求慢”必须进一步区分是 Prefill 算力、P/D 任一侧的 KV Cache 容量，还是 KV 传输链路。一个可操作的判定矩阵是：

| 同时出现的现象 | 更可能的瓶颈 | 优先动作 |
| --- | --- | --- |
| Decode `kv_cache_usage_perc` 接近上限，且 Decode `num_requests_waiting > 0` | Decode KV Cache 容量 | 增加 Decode 实例或可用显存 |
| Prefill `kv_cache_usage_perc` 接近上限，且 Prefill `num_requests_waiting > 0` | Prefill KV Cache 容量 | 增加 Prefill 实例或可用显存 |
| Prefill KV Cache 未满，但 Frontend Prefill `pending > 0` 并持续积累 | Prefill 算力或 Worker 接纳能力 | 先核对 Worker 健康与路由，再增加 Prefill 实例 |

每行中的两个条件是“且”关系。例如，仅有 KV Cache 使用率高，但没有 Waiting，不足以证明已经发生容量瓶颈；仅有 Pending 也可能是路由配置错误、Worker 不健康或状态过期。将其归因为 Prefill 算力不足前，还应同时观察 GPU 计算利用率、Prefill token throughput、TTFT 和排队持续时间。如果 KV 已经到达 D，P/D 缓存也未满，但传输阶段耗时随上下文长度增长，则应回到前文的传输字节数、链路带宽和同步时间继续排查。

TTFT 与 TPOT/ITL 的变化能帮助验证定位：TTFT 对 Prefill、排队以及 Decode 接纳能力更敏感，TPOT/ITL 则更接近持续 Decode 效率。扩容后 TTFT 大幅下降、而 TPOT 基本不变，通常说明被消除的是排队或 KV 容量瓶颈，不是单请求逐 Token 解码速度变快。

定向限制资源的对照实验验证了这套方法：Decode KV 容量受限时扩容 D，Prefill KV 容量受限时扩容 P，都能显著缩短 TTFT，而 TPOT 基本保持稳定；Prefill KV 未满但 Router Pending 持续积累时，扩容 P 同样能明显改善 TTFT，支持 Prefill 算力瓶颈的判断。其中部分实验存在中途扩容和生效参数记录不一致的问题，因此只用于确认趋势，严格定量结论仍需固定拓扑后复验。

这类“根据监控定位、扩容对应阶段、再观察 TTFT/TPOT 是否按预期分化”的闭环，比仅根据 GPU 利用率猜测瓶颈更可靠。

### 4.6 扩容收益与长输入容量拐点

PD 的扩容收益主要出现在中高并发，低并发下额外的路由、通信和状态协调开销可能抵消多实例收益。对照实验表明，长输入、较高并发的负载下，增加 D 对 TTFT、ITL 和并发承载能力的改善更明显；只增加 P 的收益在高并发时才逐渐显现，并且会受 D 侧能力限制。在 P 已扩容的基础上继续增加 D 仍有明显收益，说明对应负载的主要压力位于 Decode 侧。

长输入小样本实验还表明，增加 D 能明显推迟 TTFT 长尾拐点，提高稳定并发上限，但不会消除过载后的排队；一旦越过容量拐点，P90/P99 TTFT 仍会快速恶化。由于结果会受前缀缓存、请求到达节奏和样本量影响，具体容量边界必须通过大样本、统一流量模型的多轮压测确认。

因此，当前长输入、较高并发场景下可优先扩容 D；只有当监控证明 Prefill KV 容量或算力成为瓶颈时，再增加 P 或同步扩容 P/D。稳定容量不能用客户端 `max-concurrency`、单个 Mean TTFT 或 vLLM 按 1 秒时间桶统计的 `Peak concurrent requests` 直接代替，应结合实际请求吞吐、P90/P99 TTFT、TPOT/ITL 以及服务端 Running/Waiting/Pending，并将生产并发留在长尾拐点之前。

### 4.7 NIXL 是传输实现，不是 PD 的唯一标准

PD 分离与 NIXL 不应画等号。vLLM 和 SGLang 都提供 PD 执行能力，但把 KV 搬运做成了可替换的传输后端：

| 场景 | KV 传输方式 |
| --- | --- |
| vLLM 独立部署 | 由 `KVConnector` 决定，可使用 NixlConnector、LMCacheConnector、MooncakeConnector 等 |
| SGLang 独立部署 | 由 `--disaggregation-transfer-backend` 决定，默认 Mooncake，也可选择 NIXL |
| Dynamo 1.0.1 + vLLM | 官方示例显式配置 `NixlConnector` |
| Dynamo 1.0.1 + SGLang | 官方示例显式配置 `--disaggregation-transfer-backend nixl` |

vLLM 的结构是：

```text
PD 执行 -> KVConnector -> NIXL / Mooncake / LMCache / 其他实现
```

例如，使用 NIXL 需要配置：

```text
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

SGLang 则直接选择传输后端：

```text
--disaggregation-transfer-backend mooncake
--disaggregation-transfer-backend nixl
```

这里还要区分**会话协调协议**与**数据传输后端**。SGLang 的 `bootstrap_info` 负责告诉 P/D 对端地址、端口和 `bootstrap_room`，实际 KV 数据仍由选定的 Mooncake 或 NIXL 后端搬运；`bootstrap_info` 本身不是传输协议。

因此，端到端 PD 能力由三层共同组成：

- **Dynamo**：负责 P/D 服务发现、Worker 选择、请求编排、传输元数据交接和故障处理
- **vLLM/SGLang/TRT-LLM**：负责 Prefill/Decode 执行、KV Block 管理以及导入导出接口
- **NIXL/Mooncake 等后端**：负责实际的数据搬运

没有推理引擎提供的 KV Connector 或传输接口，Dynamo 无法独立完成 PD；反过来，只使用推理引擎自身的 PD 能力，也需要额外解决服务发现、请求配对、路由、扩缩容和故障恢复。NIXL 是 Dynamo 1.0.1 官方集成中的共同选择，而不是 vLLM 或 SGLang 唯一允许的实现。可参考 [vLLM Disaggregated Prefilling](https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md)、[SGLang PD 参数](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)、[Dynamo vLLM PD 示例](https://github.com/ai-dynamo/dynamo/blob/v1.0.1/examples/backends/vllm/deploy/disagg.yaml)和 [Dynamo SGLang PD 示例](https://github.com/ai-dynamo/dynamo/blob/v1.0.1/examples/backends/sglang/deploy/disagg.yaml)。

### 4.8 降级路径

PD 链路可能因 Prefill 节点抖动、Decode 对端不可用、会话建立失败、KV 传输超时或控制面状态延迟而失效。成熟实现必须允许请求回退到 `DecodeOnly` 等单节点模式。

这类降级的目标不是维持最优性能，而是在分布式增强能力不可用时仍然完成请求。没有回退路径，PD 带来的复杂度就会直接转化为可用性风险。

## 五、多引擎适配：统一的是调度语义

vLLM、SGLang 和 TRT-LLM 都能执行推理，但对 PD、KV 传输和 Bootstrap 信息的表达不同。如果这些差异直接暴露给上层，平台会逐渐形成多套入口、调度策略和排障链路。

Frontend 需要先形成一套引擎无关的调度结果：

- 当前请求处于 Prefill、Decode 还是故障回退路径
- Prefill 与 Decode 节点分别是谁
- 会话标识是什么
- KV 通过什么方式传输
- 失败时采取哪种回退策略
- 如何统一对外响应

随后，Engine Adapter 再把结果翻译成具体引擎协议，例如 vLLM 的 NIXL 参数或 SGLang 的 Bootstrap 信息。这样可以复用路由、降级、监控和诊断逻辑，新引擎接入时主要补充语义映射。

1.0.1 将部分原本位于独立 Router 中的 SGLang 翻译逻辑收进 Frontend，也是这一方向的体现：调度决策与协议落地在同一处闭环，减少额外 hop 和跨组件排障成本。

不过，“统一调度语义”不等于抹平所有引擎差异。例如 vLLM 与 SGLang 的 LoRA 原生接口并不一致，Frontend 仍需要明确哪些能力可以抽象、哪些能力必须保留引擎特性。

## 六、控制面异步更新，数据面同步决策

如果每个请求都实时查询外部存储、Worker 心跳和 KV 状态，远程调用的抖动与尾延迟会进入推理关键路径。Dynamo Frontend 采用的是另一种执行模型：

```text
控制面：异步订阅事件 -> 更新本地状态与索引
数据面：请求到达 -> 读取本地快照 -> 同步完成调度
```

这种设计把复杂性放在状态维护阶段，把低延迟和确定性留给请求决策。它避免的不只是一次远程查询，还包括错误选路可能造成的 KV 失效、重复 Prefill 和 Decode 热点。

代价是本地快照只能“接近实时”。系统必须正视状态陈旧窗口，并依靠版本控制、重试、降级和可观测性吸收误差。真正难的不是写出 `select_worker()`，而是持续证明它所依据的状态足够可信。

## 七、一体化 Frontend 的收益与代价

### 7.1 主要收益

- **减少重复 Prefill**：通过前缀感知路由提高 KV Cache 复用率
- **改善时延与生成稳定性**：联合考虑缓存命中和 Decode 压力，并通过资源隔离改善 TTFT、TPOT、ITL 及其尾延迟
- **统一 PD 编排**：集中完成拆分判断、节点配对、会话建立与降级
- **降低多引擎割裂**：在统一调度语义下适配不同执行后端
- **缩短关键路径**：减少独立 Router 和跨进程状态协调
- **形成平台扩展点**：为租户 QoS、会话粘性、优先级和成本感知调度提供统一入口

### 7.2 系统代价

- **逻辑单核心**：Frontend 即使可以多副本部署，错误状态或错误策略仍可能影响整个集群
- **状态一致性压力**：异步事件不可避免地带来短暂陈旧、乱序和丢失风险
- **可观测性要求高**：需要记录路由评分、缓存命中、PD 阶段与回退、`bootstrap_room` 链路和协议适配结果
- **测试组合增多**：请求类型、缓存状态、P/D 拓扑和引擎差异会形成大量组合场景
- **收益依赖 workload**：短上下文、低前缀复用业务未必能覆盖复杂调度和 KV 传输成本

评估这套架构不能只看 GPU 利用率或 Prefix Cache 命中率。高利用率可能来自有效计算，也可能来自重复 Prefill；接近 100% 的缓存命中也可能被全量 KV 传输抵消。更有意义的指标包括 TTFT、TPOT、ITL 及其 P95/P99、请求与 Token 吞吐、P/D 两侧缓存命中、实际 KV 传输字节数、链路带宽利用率、PD 成功率、降级率以及单位 Token 成本。测试还应覆盖单请求与高并发、长短 Prompt 混部、冷/热缓存和不同输出长度，避免用单一场景代替整体收益。

## 八、结论

Dynamo 1.0.1 Frontend 的核心变化，不是多承担了几个网关插件，而是拥有了影响推理成本的全局决策权：

- 将 KV Cache 从 Worker 内部优化提升为集群调度资产
- 让 Prefix Cache 的计算复用与 Decode 侧的 KV 复用、增量传输协同生效，避免瓶颈从计算转移到通信
- 在既定 PD 拓扑中选择 P/D 节点，并在 Prefill 失败时决定是否回退
- 把统一调度结果转换为不同推理引擎的执行协议
- 通过异步状态维护与本地同步决策控制关键路径延迟
- 在增强能力失败时提供可控降级

这种一体化设计减少了关键路径上的协调成本，却也把状态一致性、故障隔离和可观测性压力集中到了 Frontend。它是否值得，最终取决于真实 workload 能否从 KV 复用、PD 分离和全局调度中获得足够收益。

因此，对 Frontend 最准确的定位不是 Gateway，而是：**连接请求入口与 GPU 资源池的实时推理调度内核。**
