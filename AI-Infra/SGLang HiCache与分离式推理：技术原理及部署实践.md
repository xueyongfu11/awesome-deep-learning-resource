# SGLang HiCache 与分离式推理：技术原理及部署实践

## 1. HiCache 解决什么问题

大语言模型推理通常分为 Prefill 和 Decode 两个阶段。在 Prefill 阶段，模型需要处理输入序列，并为每一层生成 Key-Value Cache（KV Cache）。当多个请求具有相同前缀时，例如共享同一份 System Prompt、长文档或历史对话，这部分 KV Cache 实际上完全相同。若每次都重新计算，会产生大量重复计算并增加首 Token 延迟（TTFT）。

SGLang 原有的 RadixAttention 会利用 GPU 空闲显存保存前缀 KV Cache，但显存容量有限，缓存命中率很快遇到上限。HiCache 将这一机制扩展为三级缓存：

| 层级 | 存储介质 | 作用域 | 特点 |
|---|---|---|---|
| L1 | GPU 显存 | 单个推理实例 | 容量小、速度最快，可直接参与计算 |
| L2 | CPU Host Memory | 单个推理实例 | 容量较大，需要传输到 GPU |
| L3 | 分布式或外部存储 | 集群共享 | 容量最大，可跨实例复用 |

这种设计尤其适合多轮对话、共享 System Prompt、长文档问答、RAG、Prefill/Decode 分离部署，以及多实例之间共享 KV Cache 的场景。

HiCache 的核心价值不是单纯“增加缓存”，而是用容量更大的低速存储扩大 KV Cache 的生命周期和共享范围，同时通过预取、异步回写和零拷贝尽量隐藏数据移动开销。

## 2. 整体架构

```text
                    ┌───────────────────────────────┐
                    │       集群共享 L3 Cache        │
                    │ Mooncake / HF3FS / NIXL 等     │
                    └───────────────┬───────────────┘
                         预取 ↓      │      ↑ 回写
              ┌─────────────────────┴─────────────────────┐
              │                                           │
    ┌─────────▼─────────┐                       ┌─────────▼─────────┐
    │ 推理实例 A         │                       │ 推理实例 B         │
    │ L2：CPU Host Cache │                       │ L2：CPU Host Cache │
    │        ↕           │                       │        ↕           │
    │ L1：GPU KV Cache   │                       │ L1：GPU KV Cache   │
    └────────────────────┘                       └────────────────────┘
```

L1 和 L2 属于单个 SGLang 实例，L3 由多个实例共享。新实例因此可以复用其他实例写入的 KV Cache，而不必在本地重新完成 Prefill。

HiCache 支持的主要 L3 后端包括 Mooncake、DeepSeek 3FS/HF3FS、NIXL、AIBrix KVCache、HiCacheFile，以及通过 Dynamic 机制加载的自定义后端。

## 3. HiRadixTree：缓存元数据组织

HiCache 在 RadixAttention 的 RadixTree 基础上构建了 HiRadixTree。

树中的每个节点表示一段连续 Token 对应的 KV Cache；从根节点到叶节点的一条路径表示一个请求前缀。具有相同前缀的请求可以共享相同节点，避免重复存储和计算。

与普通 RadixTree 不同，HiRadixTree 还会记录节点对应的 KV Cache 位于 L1 GPU、L2 CPU、L3 外部存储，或同时存在于多个层级。

对于本地 L1/L2，HiRadixTree 会维护准确的缓存地址。对于 L3，它不会持续同步整个分布式缓存的元数据，以免引入过高的同步成本；当需要访问 L3 时，系统会实时查询后端，判断目标 KV Cache 是否存在及其位置。

## 4. 请求处理流程

HiCache 的运行过程可以概括为三个动作：本地匹配、L3 预取和数据回写。
### 4.1 本地匹配

系统首先沿 HiRadixTree 匹配请求 Token，得到一段连续的可复用前缀：前半部分可能位于 L1，后半部分可能位于 L2，再往后则需要查询 L3 或重新计算。

当 `page_size > 1` 时，匹配按照 Page 粒度进行。如果匹配边界落在树节点内部，系统会拆分节点，使后续请求可以在准确边界上复用缓存。本地匹配只查询元数据，不复制实际 KV 数据，因此速度很快。

### 4.2 从 L3 预取

对于 L1/L2 未命中的部分，HiCache 会查询 L3 是否存在连续的前缀缓存。默认情况下，当 L3 命中长度超过 256 Token 时才触发预取，该阈值可通过 `prefetch_threshold` 调整。

L3 数据先进入 L2，随后再传输到 GPU。预取提供三种终止策略：

| 策略 | 行为 | 适用场景 |
|---|---|---|
| `best_effort` | GPU 可以开始计算后立即停止等待 | 对延迟极度敏感 |
| `wait_complete` | 等待所有目标数据预取完成 | 优先提高缓存复用率 |
| `timeout` | 完成或超时后继续执行 | 在命中收益和 SLO 之间平衡 |

官方文档建议生产环境优先考虑 `timeout`。其超时时间为：

```text
timeout = min(
    prefetch_timeout_max,
    prefetch_timeout_base
      + prefetch_timeout_per_ki_token × num_token_to_fetch / 1024
)
```

默认值：

- `prefetch_timeout_base = 2` 秒；
- `prefetch_timeout_per_ki_token = 0.1` 秒/1024 Token；
- `prefetch_timeout_max = 30` 秒。

### 4.3 Prefill 与回写

L1、L2 和已经完成预取的 L3 缓存被加载到 GPU 后，模型只需要为剩余未命中的 Token 执行 Prefill。新生成的 KV Cache 随后可以按照配置写入更低层级。

| 策略 | 行为 | 特点 |
|---|---|---|
| `write_through` | 数据生成或访问后立即写入下一层 | 复用收益最大，但 I/O 压力较高 |
| `write_through_selective` | 达到访问次数阈值后才写入 | 只保存热点数据 |
| `write_back` | 上层缓存淘汰时才写入下一层 | I/O 较低，但缓存建立较慢 |

L2 向 L3 回写前会先判断数据是否已经存在，避免重复传输。写入 L3 后，缓存便可以被集群中的其他 SGLang 实例复用。

#### Write-through 的数据驻留与内存占用

使用 `write_through` 时，新生成或访问的 KV Cache 会立即从 GPU L1 写入 CPU L2。因此，同一份热点 KV Cache 可能同时存在于 GPU 和 CPU：

| 阶段 | KV Cache 驻留状态 | 逻辑副本数 |
|---|---|---|
| 刚完成计算并写入 L2 | GPU L1 和 CPU L2 各有一份 | 2 份 |
| GPU 空间不足，冷 Block 被淘汰 | GPU 副本释放，只保留 CPU L2 | 1 份 |
| 后续请求再次命中 | 从 CPU 加载回 GPU，L1/L2 再次各有一份 | 2 份 |

这种“双写”看起来增加了内存占用，但它是分层缓存用空间换时间的核心设计：

1. GPU 保存正在使用或近期频繁访问的热数据，使模型可以直接计算；
2. CPU 内存通常比 GPU 显存容量更大，用来保留更多暂时不活跃的冷数据；
3. GPU 淘汰某个 Block 后，CPU 中仍有副本，无需重新执行 Prefill；
4. 后续命中时，只需完成 CPU→GPU 传输，通常比重新计算整段前缀更快，从而降低 TTFT。

需要注意，这里的“双份”描述的是同一 KV 数据在两个层级的驻留状态，并不一定意味着进程会在每次写入时临时增加两倍内存。GPU KV Pool 和 Host KV Pool 通常在启动阶段按照 `--mem-fraction-static`、`--hicache-ratio` 或 `--hicache-size` 预先规划，运行过程中主要改变的是缓存页在各层级的占用和有效状态。

CPU 副本主要用于应对 GPU 缓存淘汰和后续复用，并不能直接等同于 GPU 故障容灾。如果进程、节点或设备发生故障，CPU L2 数据是否仍然可用取决于故障范围和恢复机制。只有继续写入独立的 L3 后端，才可能获得跨实例复用或一定程度的进程重启后持久性；具体能力仍取决于后端实现。

如果同时启用了 L3 且采用贯穿式回写，同一份逻辑 KV Cache 还可能同时存在于 GPU、CPU 和 L3。其目标不是减少总副本数，而是在访问速度、缓存容量、复用范围和 I/O 成本之间取得平衡。

## 5. 数据布局与传输优化

### 5.1 Host Memory 布局

| 布局 | 特点 | 使用建议 |
|---|---|---|
| `layer_first` | 按模型层组织，与 GPU 计算方式一致 | 通用兼容布局 |
| `page_first` | 同一 Page 的 KV 数据连续存放 | 有利于批量 L3 I/O 和零拷贝 |
| `page_first_direct` | 在 Page 内按 Layer 聚合 Token | 同时优化 L3 I/O 与 CPU→GPU 传输 |

需要注意：

- `page_first` 仅兼容 `kernel` I/O 后端；
- 如果搭配 `direct`，系统会自动切换为 `layer_first`；
- `page_first_direct` 面向 `direct` 后端设计，也兼容 FA3；
- Mooncake 等后端对 Host Memory 布局存在约束，运行时挂载时也会检查。

### 5.2 CPU 与 GPU 之间的 I/O 后端

`--hicache-io-backend direct` 使用标准 CUDA 内存复制；`--hicache-io-backend kernel` 使用针对 KV Cache 优化的 GPU-assisted I/O Kernel。官方文档报告后者相较基线最高可实现约 3 倍的传输速度，因此通常优先推荐 `kernel`。

### 5.3 计算与传输重叠

在 Prefill 过程中，HiCache 可以在 GPU 计算第 N 层时，同时加载第 N+1 层的 KV Cache，从而隐藏部分 CPU→GPU 传输延迟。

### 5.4 Page 粒度权衡

`--page-size` 表示每个缓存页包含多少 Token。

- 较大的 Page：元数据更少、I/O 批量更大，适合具有长公共前缀的请求，但部分匹配时可能降低命中率；
- 较小的 Page：匹配粒度更细，适合前缀差异较大的请求，但元数据和 I/O 调度开销更高。

官方实践示例通常从 `--page-size 64` 开始调优，但它并不是适合所有业务的固定最优值。

## 6. 多 GPU 与异构 TP

在 Tensor Parallel 场景下，各 Rank 必须对缓存命中和预取结果达成一致。HiCache 会在关键阶段使用 `all_reduce(op=min)`，确认所有 Rank 的 L3 命中量及成功获取的公共前缀长度，避免状态不一致。

如果多个集群采用不同 TP 大小，但共享同一个 L3 Namespace，可以配置：

```bash
--hicache-storage-backend-extra-config '{"tp_lcm_size": 8}'
```


多个模型实例共享缓存、且各实例 TP Size 不同时，`tp_lcm_size` 应设为这些 TP Size 的最小公倍数，例如 TP=4 和 TP=8 时设为 8。不同 TP 下，各 Rank 负责的 KV heads 范围不
  同；该参数统一存储分片粒度，让各实例通过拆分或组合分片复用缓存。主要用于 GQA/MHA 模型，不改变实际 TP Size；各实例 TP Size 相同时无需设置。

MLA 先将一个 token 的信息压缩成共享向量 c，不同 heads 通过各自的投影矩阵，从同一个 c 得到所需的 K/V。对于 MLA 模型，每个 TP Rank 可能保存相同的完整 Token KV 数据。为避免重复存储，HiCache 会只让一个 Rank 发起回写。

## 7. 核心配置参数

一个基础配置如下：

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/model \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through
```

| 参数 | 作用 |
|---|---|
| `--enable-hierarchical-cache` | 启用 HiCache |
| `--hicache-ratio` | L2 Host KV Pool 与 GPU KV Pool 的容量比例，必须大于 1 |
| `--hicache-size` | 每个 Rank 的 L2 容量，单位 GB；设置后覆盖 `hicache-ratio` |
| `--page-size` | 每个缓存 Page 的 Token 数 |
| `--hicache-io-backend` | CPU↔GPU 数据传输后端：`direct` 或 `kernel` |
| `--hicache-mem-layout` | L2 内存布局 |
| `--hicache-write-policy` | KV Cache 回写策略 |
| `--hicache-storage-backend` | L3 存储后端 |
| `--hicache-storage-prefetch-policy` | L3 预取终止策略 |
| `--hicache-storage-backend-extra-config` | 后端及预取扩展配置 |

### 7.1 L2 容量配置

`--hicache-ratio 2` 表示每个 Rank 的 Host KV Pool 是 GPU KV Pool 的两倍。

### 7.2 扩展配置输入

扩展参数可以直接使用 JSON：

```bash
--hicache-storage-backend-extra-config \
  '{"prefetch_threshold":512,"prefetch_timeout_base":0.5}'
```

复杂配置也可以放入 TOML、JSON 或 YAML 文件，并以 `@` 开头：

```bash
--hicache-storage-backend-extra-config "@config.toml"
```

## 8. L3 存储后端部署

### 8.1 HF3FS 示例

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/model \
  --tp 8 \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete
```

### 8.2 Mooncake 示例

```bash
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=816043786240
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="$DEVICE_LIST"
export MOONCAKE_MASTER="127.0.0.1:50051"

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 8 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout
```

后端选型通常应同时考虑网络与存储带宽、RDMA 或零拷贝支持、KV 数据规模、跨节点共享需求、部署环境、运维复杂度和 Host Memory Layout 兼容性。

## 9. PD 分离及其与 HiCache 的结合

### 9.1 为什么需要 PD 分离

大模型推理由 Prefill 和 Decode 两个资源特征不同的阶段组成：

传统统一引擎将两种任务混合调度，容易产生两个问题：新到达的 Prefill Batch 会打断正在运行的 Decode，增大 Token 输出间隔；在 DP Attention 场景中，不同 Worker 同时处理 Prefill 和 Decode 还可能造成负载不均。

PD 分离（Prefill-Decode Disaggregation）将两个阶段部署到不同的计算实例，使它们可以独立调度、优化和扩缩容。

### 9.2 请求处理流程

```text
用户请求
   │
   ▼
SGLang Router
   │
   ▼
Prefill 实例
处理输入并生成 KV Cache
   │
   │ 通过 Mooncake / NIXL 等传输
   ▼
Decode 实例
接收 KV Cache，逐 Token 生成结果
   │
   ▼
返回用户
```

Router 负责选择 Prefill 和 Decode 实例，并提供负载均衡和故障处理。Prefill 完成后，KV Cache 通过高速传输后端发送给 Decode 实例，Decode 无需重复计算输入序列。

SGLang 主要支持 Mooncake、NIXL 和 Ascend 等传输后端。实际部署通常依赖 RDMA、IB、RoCE 或 NVLink 等高速互联，否则 KV Cache 传输可能成为新的性能瓶颈。

### 9.3 基础部署示例

```bash
# Prefill 实例
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0

# Decode 实例
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000
```

#### Bootstrap 端口的含义

上面的示例展示了 PD 分离的基本启动方式。下面结合一组实际部署配置，说明 Router 如何关联 Prefill 的 HTTP 端口和 Bootstrap 端口。这组配置在同一台机器上使用 GPU 4 运行 Prefill（P）、GPU 5 运行 Decode（D），客户端通过 Router 的 `7711` 端口访问服务。

原始脚本中，P 侧启用了 HiCache，并使用 file 后端作为 L3 存储；D 侧未启用 HiCache。这里先摘出与 PD 通信有关的参数，省略模型路径、缓存策略等配置（以下 P/D 片段用于对照参数，不是完整启动命令）：

```bash
# P：CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server 的关键参数
--disaggregation-mode prefill \
--host 0.0.0.0 \
--port 8100 \
--disaggregation-bootstrap-port 8190 \
--disaggregation-transfer-backend nixl

# D：CUDA_VISIBLE_DEVICES=5 python3 -m sglang.launch_server 的关键参数
--disaggregation-mode decode \
--host 0.0.0.0 \
--port 8400 \
--disaggregation-transfer-backend nixl
```

Router 将上述 P、D 实例注册为两个阶段的服务入口：

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:8100 8190 \
  --decode http://127.0.0.1:8400 \
  --host 0.0.0.0 \
  --port 7711 \
  --policy cache_aware
```

其中，`--decode` 指向 D 的 HTTP 端口 `8400`；`--prefill` 除了指向 P 的 HTTP 端口，还额外提供 P 的 Bootstrap 端口：

```bash
--prefill http://127.0.0.1:8100 8190
```

`8100` 和 `8190` 是两个用途不同的端口：

| 端口 | 对应参数 | 用途 |
|---|---|---|
| `8100` | Prefill 的 `--port 8100` | Prefill HTTP API，接收 Router 转发的请求 |
| `8190` | Prefill 的 `--disaggregation-bootstrap-port 8190` | PD 内部控制端口，初始化和协调 KV Cache 传输 |

Router 的参数格式为：

```text
--prefill <PREFILL_HTTP_URL> [BOOTSTRAP_PORT]
```

因此 `8190` 不是 HTTP URL 的一部分，而是与该 Prefill 实例绑定的可选 Bootstrap Port。它必须与对应 Prefill 服务的 `--disaggregation-bootstrap-port` 保持一致。

一次请求涉及的端口和通道可以简化为：

```text
7711：客户端访问 Router 的统一入口
8100：Prefill HTTP 业务入口
8190：Prefill KV 传输控制与握手入口
NIXL/Mooncake：实际 KV Cache 数据传输通道
8400：Decode HTTP 业务入口
```

Bootstrap 通道主要负责本次传输的请求标识、目标 KV Cache 位置、连接元数据、初始化握手和传输生命周期管理；真正的大块 KV Cache 数据由 `--disaggregation-transfer-backend` 指定的 NIXL、Mooncake 等后端传输。

多个 Prefill 实例需要分别注册各自的 HTTP 地址和 Bootstrap Port：

```bash
--prefill http://10.0.0.1:8100 8190 \
--prefill http://10.0.0.2:8100 8191
```

`--disaggregation-bootstrap-port` 在官方参数中定义为 Prefill 侧的 Bootstrap Server Port。基础 Mooncake/NIXL 配置通常不需要在 Decode 命令中设置另一个不同的 Bootstrap Port；Router 的 `--decode` 参数也只接收 Decode URL，不接收附加端口。若特定 SGLang 版本对 Decode 侧另有要求，应以该版本的启动日志和实现为准。

编写多行 Shell 命令时，还要保证续行符 `\` 是该行最后一个字符。反斜杠后不能存在普通空格或不可见的特殊空格，否则下一行参数可能被 Shell 当作新的命令。

### 9.4 异构 TP

Prefill 和 Decode 可以采用不同的 TP 配置，但两侧 KV Cache 的内存布局可能不同。对于非 MLA 的 GQA/MHA 模型，Mooncake 可以启用 GPU Staging Buffer：

```bash
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096
```

它会在 Prefill 侧聚合 KV Head，批量传输后再在 Decode 侧分散到对应缓存页。官方文档称，高并发异构 TP 场景下，相比逐 Token Slice 方式可获得约 2～5 倍吞吐提升；同构 TP 下会自动绕过。该功能不适用于 DeepSeek-V2/V3 等 MLA 模型。

### 9.5 与 HiCache 结合

HiCache 在 PD 分离部署中有两种使用方式：

1. 仅在 Prefill 节点启用：多个 Prefill 实例通过 L3 共享 KV Cache，适合公共 System Prompt 或固定知识上下文；
2. Prefill 和 Decode 都接入 L3：Decode 节点异步将多轮对话产生的 KV Cache 写回 L3，后续请求可以由 Prefill 节点直接复用。

Decode 节点需要启用：

```bash
--disaggregation-decode-enable-offload-kvcache
```

第二种模式更适合多轮对话，但会增加 Decode 侧回写流量，需要评估网络带宽和存储压力。

### 9.6 优势与代价

PD 分离减少了 Prefill 对 Decode 的干扰，并允许两个阶段独立优化与扩缩容，更容易分别控制 TTFT 和每 Token 延迟。其代价是增加了 KV Cache 传输、Router、传输后端和故障恢复等系统复杂度，同时需要根据实际流量持续调整 Prefill 与 Decode 实例比例。

### 9.7 EPD：面向多模态模型的三阶段分离

对于视觉语言模型（VLM），一次请求可以进一步拆成三个阶段：

| 阶段 | 工作内容 | 资源特征 |
|---|---|---|
| Encoder | 图像预处理与 ViT 编码，生成视觉 Embedding | 计算密集，只在请求初始化时执行 |
| Prefill | 处理完整的多模态输入，建立语言模型 KV Cache | 计算密集 |
| Decode | 读取 KV Cache，逐 Token 生成结果 | 显存带宽与容量密集 |

普通 PD 分离仍将视觉 Encoder 和语言 Prefill 放在一起。图片较多或视觉编码开销较高时，这种耦合会限制资源利用率和横向扩展能力。EPD（Encoder–Prefill–Decode Disaggregation）把 Encoder 进一步拆成独立服务，形成三阶段架构：

这样 Encoder、Prefill 和 Decode 可以分别扩缩容。例如图片数量和分辨率较高时，可以单独增加 Encoder 实例，而不必同步扩容语言模型实例。

#### EPD 核心参数

- `--encoder-only`：启动纯 Encoder 服务；
- `--language-only`：启动不包含视觉编码器的语言模型服务；
- `--encoder-urls`：为 Language/Prefill 服务配置一个或多个 Encoder 地址；
- `--encoder-transfer-backend`：选择视觉 Embedding 传输方式，支持 `zmq_to_scheduler`、`zmq_to_tokenizer` 和 `mooncake`，默认为 `zmq_to_scheduler`；
- `--disaggregation-mode prefill/decode`：在 EPD 中继续拆分语言 Prefill 与 Decode。

#### EPD 基础部署示例

```bash
# Encoder 实例
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000

# Prefill 实例：只加载语言部分，并连接 Encoder
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002

# Decode 实例
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode decode \
  --port 30003

# Router 仍按 PD 方式连接 Prefill 和 Decode
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30002 \
  --decode http://127.0.0.1:30003 \
  --port 8000
```

Encoder 也可以作为 gRPC 服务运行。此时 Encoder 使用 `--grpc-mode`，Prefill 侧设置 `SGLANG_ENCODER_MM_RECEIVER_MODE=grpc`，并通过 `grpc://` 地址连接。

#### Mooncake 传输与多模态缓存

设置 `--encoder-transfer-backend mooncake` 可以使用 Mooncake 在 Encoder 和 Language/Prefill 服务之间传输视觉 Embedding。该选项只控制传输方式，与全局多模态缓存相互独立。

Encoder 还可以启用：

```bash
--enable-mm-global-cache
```

启用后，Encoder 会先在 Mooncake 中查询图像对应的视觉 Embedding：命中时直接预取，未命中时正常运行视觉编码，并在后台写入全局缓存。它适合重复或重叠图片较多、Encoder 计算成为瓶颈且集群已经部署 Mooncake 的场景。

需要区分两类缓存：

- `--enable-mm-global-cache` 缓存的是视觉 Encoder Embedding；
- HiCache 缓存的是语言模型 KV Cache。

在完整的 EPD + HiCache 架构中，前者减少重复的视觉编码计算，后者减少重复的语言 Prefill 计算，两者可以同时使用。

参考资料：

- [SGLang PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- [SGLang EPD Disaggregation](https://docs.sglang.io/docs/advanced_features/epd_disaggregation)

## 10. 自定义 L3 后端

最简单的自定义后端需要提供三个核心操作：

```python
get(key)
exists(key)
set(key, value)
```

随后将后端注册到 `BackendFactory`。如果不希望修改 SGLang 仓库，可以使用动态加载：

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/model \
  --enable-hierarchical-cache \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config \
  '{
    "backend_name": "custom_backend",
    "module_path": "my_package.my_backend",
    "class_name": "CustomHiCacheStorage"
  }'
```

可通过 `interface_v1` 控制是否启用 `batch_get_v1` 和 `batch_set_v1` 批量接口。

## 11. 运行时挂载与卸载 L3

SGLang 支持在不重启服务的情况下，通过 HTTP Admin API 动态启用或停用 L3 后端。

```text
HTTP Server
    ↓
TokenizerManager
    ↓ FanOutCommunicator
Scheduler：检查服务是否完全空闲
    ↓
HiRadixCache：解析后端和预取配置
    ↓
HiCacheController：创建/销毁后端，启动/停止后台线程
```

### 11.1 查询状态

```bash
curl -s http://127.0.0.1:30000/hicache/storage-backend
```

### 11.2 挂载后端

```bash
curl -s -X PUT \
  http://127.0.0.1:30000/hicache/storage-backend \
  -H 'Content-Type: application/json' \
  -d '{
    "hicache_storage_backend": "mooncake",
    "hicache_storage_backend_extra_config_json":
      "{\"master_server_address\":\"127.0.0.1:50051\",\"protocol\":\"tcp\",\"global_segment_size\":\"4gb\",\"prefetch_threshold\":256}",
    "hicache_storage_prefetch_policy": "timeout"
  }'
```

### 11.3 卸载后端

```bash
curl -s -X DELETE \
  http://127.0.0.1:30000/hicache/storage-backend
```

卸载操作只会停止使用 L3、停止 Prefetch 和 Backup 后台线程，并销毁当前后端实例；它不会删除 Mooncake、HF3FS 等外部存储中已经保存的数据。

## 12. 动态切换的限制与安全流程

运行时 Attach/Detach 虽然不要求重启进程，但严格要求 Scheduler 完全空闲：

- 没有正在运行的 Batch；
- 没有等待或排队请求；
- 没有 Chunked Prefill、Overlap、Pipeline Parallel 任务；
- 没有 Disaggregation Bootstrap、Transfer 或 Inflight 请求；
- 没有 DLLM Staging 请求。

如果不满足条件，接口返回 HTTP 400，并保持现有状态不变。推荐操作流程：

### 12.1 DP 场景的额外风险

当 `dp_size > 1` 时，请求会发送给所有 DP Scheduler：只有所有 Rank 成功，最终结果才是成功；当前没有跨 DP Rank 的自动部分回滚，因而可能出现整体报告失败，但部分 Rank 已完成挂载的情况。

建议所有 Rank 使用一致的后端配置。Attach 失败后，立即调用一次 Detach，修复配置后重新 Attach，并再次查询所有实例状态。

此外，运行时 Attach 不会绕过布局限制。如果当前服务的 L2 内存布局不满足 Mooncake 等后端要求，Attach 仍会失败。

## 13. 生产环境调优建议

### 13.1 延迟优先

```bash
--hicache-storage-prefetch-policy best_effort
--hicache-write-policy write_back
```

这种组合减少等待和前台 I/O，但 L3 缓存复用收益可能下降。

### 13.2 命中率优先

```bash
--hicache-storage-prefetch-policy wait_complete
--hicache-write-policy write_through
```

适合高重复度、长公共前缀场景，但必须确保 L3 带宽充足。

### 13.3 SLO 与复用率平衡

```bash
--hicache-storage-prefetch-policy timeout
--hicache-write-policy write_through_selective
```

该组合通常更适合生产环境。应结合请求长度分布和 TTFT SLO 调整 `prefetch_threshold`、`prefetch_timeout_base`、`prefetch_timeout_per_ki_token` 和 `prefetch_timeout_max`。

推荐的调优顺序：

1. 测量公共前缀长度和重复率；
2. 选择 `page-size`；
3. 确定每 Rank 的 L2 容量；
4. 验证 CPU→GPU 实际带宽；
5. 接入 L3 并观察命中率；
6. 调整 Prefetch Policy 和 Timeout；
7. 最后调整 Write Policy，控制 L3 写入压力。

重点监控指标包括 L1/L2/L3 命中率、Prefill 延迟与 TTFT、跳过计算的 Token 数、L3 预取成功率与超时率、CPU→GPU 传输带宽、L3 读写带宽、后台回写队列长度，以及不同请求长度区间的尾延迟。

## 14. 总结

HiCache 本质上是建立在 RadixAttention 之上的分层、跨实例 KV Cache 系统：HiRadixTree 负责组织前缀及缓存位置，L1/L2 提供实例内快速复用，L3 提供大容量和集群级共享；预取、回写、Page 布局、零拷贝和计算传输重叠则负责控制分层存储带来的 I/O 成本。

它最适合“前缀重复度高、Prefill 成本高、显存缓存容量不足”的负载。实际部署时，决定收益的关键不是简单开启 HiCache，而是让缓存容量、Page 粒度、传输布局、预取等待时间和回写流量与真实工作负载相匹配。

## 参考资料

- [SGLang HiCache Best Practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices#core-hicache-parameters)
- [HiCache System Design and Optimization](https://docs.sglang.io/docs/advanced_features/hicache_design)
- [Runtime Attach/Detach HiCache Storage Backend](https://docs.sglang.io/docs/advanced_features/hicache_storage_runtime_attach_detach)
