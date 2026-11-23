# NIXL KV Cache 传输路径与相关概念

在 Prefill–Decode（PD）分离部署中，Prefill Worker（下文简称 P）生成或复用 Prompt 对应的 KV Cache，Decode Worker（简称 D）接收这些 KV 后继续生成 Token。KV 从 P 的显存传到 D 的显存时，具体路径取决于 GPU 互连、网卡的内存访问能力和部署配置。同机传输和跨机 RDMA 都可能经过主机内存中转。

以 Dynamo + SGLang 为例，SGLang 选用 NIXL 传输 KV 时，默认使用 UCX 后端。SGLang 提供待传输的 KV 页和目标缓冲区，NIXL 管理内存注册与传输请求，UCX 根据两端的配置选择传输协议和通道。 [SGLang PD 分离文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx) · [NIXL 后端指南](https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md)

## 1. KV Cache 传输路径总览

下表按 KV 从 P 显存到 D 显存的顺序列出各条路径。排名仅供比较大块 KV 传输的吞吐量，**1 表示吞吐最高**。

| 参考吞吐排名 | 路径 | KV 经过的位置 | 使用条件 |
| --- | --- | --- | --- |
| 1 | 同机 GPU 直传 | P 显存 → PCIe 或 NVLink → D 显存 | 同节点，CUDA IPC 可用，且两张 GPU 支持所需方向的 P2P。 |
| 2 | 同机主机内存中转 | P 显存 → 主机内存 → D 显存 | 同节点，两端可通过主机缓冲区搬运数据。 |
| 3 | 跨机 GPUDirect RDMA | P 显存 → P 网卡 → RDMA 网络 → D 网卡 → D 显存 | 两端 RDMA 可达，且网卡均能直接访问对应 GPU 的显存。 |
| 4 | 跨机 RDMA，经主机内存 | P 显存 → P 主机内存 → P 网卡 → RDMA 网络 → D 网卡 → D 主机内存 → D 显存 | 两端 RDMA 可达，并使用已注册的主机内存缓冲区。 |
| 5 | 跨机 TCP | P 显存 → P 主机内存 → P 网卡 → IP 网络 → D 网卡 → D 主机内存 → D 显存 | 两端 TCP/IP 可达，且传输后端启用 TCP 通道。 |
| — | 同机 RDMA（网卡回环） | 支持 GPUDirect RDMA 时，P 显存 → 本机 RDMA 网卡回环 → D 显存。使用主机缓冲区时，P 显存 → 主机内存 → 本机 RDMA 网卡回环 → 主机内存 → D 显存。 | 同节点，两端可使用 RDMA，且网卡和配置支持本地回环。 |

这一排序假设同机 GPU 具备高带宽 P2P、主机内存中转没有明显的共享链路或 NUMA 瓶颈，且三种跨机方案使用同速的单端口链路（例如 100 Gb/s）。网络带宽足够高时，跨机 GPUDirect RDMA 的吞吐可超过同机主机内存中转。如果几种跨机方案都受限于网口带宽，吞吐也可能接近。 [OpenUCX GPU 传输说明](https://openucx.readthedocs.io/en/master/faq.html#working-with-gpu) · [NVIDIA GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)

同机 RDMA 的吞吐取决于网卡的回环能力、GPU 与网卡之间的 PCIe 拓扑，以及是否经过主机内存，表中未给它指定排名。Kubernetes 中，同节点的 P、D Pod 在 CUDA IPC 不可用且 RDMA 可用时，可以走这条路径。 [NVIDIA Dynamo RDMA 指南](https://docs.nvidia.com/dynamo/dev/kubernetes/installation/rdma-setup/overview)

## 2. 同机传输与 GPU 访问条件

### GPU 直传的 CUDA IPC 与 P2P 条件

P 和 D 通常运行在不同进程里。**CUDA IPC** 允许一个进程导出显存句柄，另一个进程导入句柄并映射对应的显存。要实现跨 GPU 直传，还需要实际访问方向上的 **GPU P2P** 能力。满足条件时，UCX 可以使用 `cuda_ipc`，数据经 GPU 之间可用的 PCIe 或 NVLink 路径传输。

PCIe 也是 GPU、网卡和主机之间常见的互连。它的实际带宽受协商速率、通道数和共享上行链路影响。NVLink 是 NVIDIA 的高速互连，能否使用取决于 GPU 和部署拓扑。同在一台服务器上的两对 GPU，可能有不同的 P2P 能力。 [CUDA 多 GPU 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html) · [CUDA 进程间通信文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)

### 主机内存中转的拷贝与流水线

走主机内存时，P 先把 KV 从显存拷贝到主机缓冲区，D 再把 KV 拷贝到自己的显存。如果两端使用独立的主机缓冲区，还可能需要在缓冲区之间拷贝一次。分块传输可以让两端并行工作。D 将当前块拷入显存时，P 可以把下一块拷到主机缓冲区。两段拷贝串行时，耗时相加。充分重叠时，整体吞吐主要取决于较慢的一段。缓冲区所在的 NUMA 节点也会影响速度。 [UCX 同机两阶段搬运](https://github.com/openucx/ucx/wiki/UCF2021_slides/UCF2021_paper_15.pdf)

同机 TCP 也可通过本机回环传输，数据经过内核 TCP/IP 栈。在不同容器网络命名空间之间，经过哪些接口取决于容器网络和路由。部分 RDMA 网卡也支持同机回环，能否使用取决于网卡和后端配置。 [OpenUCX 通道说明](https://openucx.readthedocs.io/en/master/faq.html) · [NVIDIA mlx5 本地回环说明](https://docs.nvidia.com/doca/archive/2-9-0/ethernet%2Bnetwork/index.html)

### Kubernetes 部署对同机传输的影响

在 Kubernetes 中，即使 P 和 D 已调度到同一节点，能否通过 CUDA IPC 直传 KV 仍取决于进程的 PID 命名空间、对端 GPU 是否可见，以及 GPU P2P 是否可用。默认每个 Pod 只暴露分配给自己的 GPU，两个 Pod 也通常各自拥有独立的 PID 命名空间。从 UCX 1.23 起，跨 PID 命名空间在缺少可用的 IMEX fabric handle 时会被判定为 CUDA IPC 不可达。此时，UCX 仍可根据设备和配置选择网卡 RDMA 或 TCP。 [NVIDIA Dynamo 同机传输说明](https://docs.nvidia.com/dynamo/dev/kubernetes/disaggregated-serving/overview)

| 部署场景 | 配置与取舍 |
| --- | --- |
| 固定同机 1P1D | 在一个申请多张 GPU 的容器中运行 P、D 两个进程，各自绑定计算用卡，并为 NIXL side-channel 配置不同端口。让两端具备访问对端 GPU 所需的设备可见性。CUDA IPC 可用时，可经 NVLink 或 PCIe 搬运 KV。 |
| 同机独立 Pod，需要使用 NVLink | 用 `nodeSelector` 将两端放到同一节点。NVIDIA 示例使用 `hostPID: true` 共享主机 PID 命名空间、用 `privileged: true` 暴露对端 GPU，再将各推理进程绑定到调度器分配的 GPU。这个方案会放宽 Pod 隔离，适合专用节点。`shareProcessNamespace` 只在单个 Pod 内生效。 |
| P/D 独立扩缩容的 N×M 部署 | 将 P、D 部署为独立 Pod 池，通过设备插件向两端分配 RDMA 设备，并配置 InfiniBand、RoCE 或云厂商的 RDMA 网络。这种部署允许跨机调度，同机的独立 Pod 也可使用网卡传输。 |

[NVIDIA Dynamo Kubernetes 部署指南](https://docs.nvidia.com/dynamo/dev/kubernetes/disaggregated-serving/overview) · [NVIDIA Dynamo RDMA 指南](https://docs.nvidia.com/dynamo/dev/kubernetes/installation/rdma-setup/overview)

## 3. 跨机传输的内存访问与网络协议

下图展示 P 和 D 分别运行在两台服务器上时，使用 **GPUDirect RDMA + RoCEv2** 传输 KV 的路径。两端网卡通过 PCIe 访问本机 GPU 显存，并通过以太网交换数据。

```text
P 服务器              服务器之间                     D 服务器
P 显存 ── PCIe ── P 网卡 ══ 以太网（RoCEv2）══ D 网卡 ── PCIe ── D 显存
```

### 网卡对显存和主机内存的访问

以一次 RDMA WRITE 为例，P 网卡从源缓冲区读取 KV，D 网卡写入目标缓冲区，两端使用已注册的内存。**经主机内存的 RDMA** 把这两个缓冲区放在主机内存里，因此发送前要做显存到主机内存的拷贝，接收后还要拷回显存。**GPUDirect RDMA** 让支持这一能力的网卡经 PCIe 直接访问已注册的 GPU 显存，省去两端的中转拷贝。能否使用它取决于 GPU、网卡、驱动和 PCIe 拓扑。 [NVIDIA GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)

### 跨机网络与传输协议

**RDMA** 提供直接读写远端内存的能力。**RoCE** 把 RDMA 报文放到以太网上传输。RoCEv1 直接封装在以太网帧中，RoCEv2 通过 UDP/IP 封装，可以经过 IP 路由。采用可靠 RDMA 传输服务时，可靠交付由该服务处理。**InfiniBand** 则使用自己的网络和协议，原生支持 RDMA。使用 InfiniBand 原生 RDMA 时，上图中两端网卡之间的数据改由 InfiniBand 网络传输。 [NVIDIA RoCE 协议文档](https://docs.nvidia.com/doca/sdk/rdma-over-converged-ethernet.pdf) · [OpenUCX 网络传输说明](https://openucx.readthedocs.io/en/master/faq.html)

同一套以太网上也可以跑普通 **TCP/IP**。IP 负责寻址和转发，TCP 提供可靠、有序的字节流。使用 UCX 的 TCP 通道搬 KV 时，GPU 数据通常要经主机内存。网络协议中还有 **iWARP**，它在 TCP/IP 之上提供 RDMA 的内存访问语义。 [OpenUCX 通道说明](https://openucx.readthedocs.io/en/master/faq.html) · [IETF RDMA 协议规范](https://www.rfc-editor.org/rfc/rfc5040.html)

### 网卡与交换机之间的物理链路

网卡和交换机之间可以用兼容的 DAC 直连铜缆，也可以用光模块加光纤。日常所说的 RJ45 网线通常是铜双绞线。可用的传输介质取决于两端接口和链路速率。两台服务器可以通过同一套以太网光纤链路先跑 TCP，再跑 RoCEv2。若在 RoCEv2 上进一步启用 GPUDirect RDMA，改变的是服务器内网卡访问的内存。 [NVIDIA 线缆与光模块规格](https://docs.nvidia.com/networking/display/400gbe-and-200g-50g-pam4-and-100g-25g-nrz-cables-and-transceivers-parts-list-using-qsfp-dd-qsfp56-qsfp28-sfp28.pdf)

## 4. NIXL 传输流程与实际路径确认

在 SGLang 的 NIXL 实现里，两端先注册内存、交换连接和内存描述信息。D 提供目标缓冲区，P 为待搬运的 KV 页提交异步 `WRITE`。传输期间，源页和目标缓冲区都需要保持有效。D 确认所需的 KV 页全部到达后，再读取目标缓冲区。NIXL API 也提供由接收方发起的 `READ` 操作。 [SGLang NIXL 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/nixl/conn.py) · [NIXL Python API](https://github.com/ai-dynamo/nixl/blob/main/docs/python_api.md)

UCX 能使用的通道取决于构建、运行时配置和端点是否可达。以 `cuda_ipc` 为例，即使 GPU 对支持 P2P，KV 所用的显存分配也要支持相应的句柄导出和导入。UCX 会结合操作、内存类型和传输大小选择协议。在张量并行部署中，一个请求可能对应多笔 rank 间传输，实际 GPU 对和端点配置不同，选出的路径也可能不同。 [OpenUCX 协议选择说明](https://openucx.readthedocs.io/en/master/faq.html#protocol-selection) · [NIXL 显存分配与 IPC 兼容性案例](https://github.com/ai-dynamo/nixl/issues/1754)

按以下步骤确认一次 KV 传输使用的协议和通道。

1. **定位 KV 操作。** 记录要核对的操作、两端 rank、各自使用的 GPU、内存类型和传输大小。
2. **检查通道限制。** `UCX_TLS` 限制可用通道，其中有些还用于建连或访问 GPU 内存。只填 `cuda_ipc` 可能因缺少所需 CUDA 内存支持被部分 NIXL 版本拒绝。保留 `cuda_copy` 提供的是 GPU 与主机内存间的搬运能力，是否用它中转还要看所选协议。
3. **读取对应的协议条目。** 使用 UCX protocols v2 时，`UCX_PROTO_INFO=y` 会显示各端点配置下按操作、内存类型和大小区间选出的协议。用第一步的信息匹配 KV 数据操作。仅看到 RDMA 网卡名或 `cuda_ipc` 出现在端点配置里，无法确认这笔 KV 数据实际走的路径。

[OpenUCX 通道和协议日志示例](https://openucx.readthedocs.io/en/master/faq.html#how-can-i-tell-which-protocols-and-transports-are-being-used-for-communication) · [NIXL 版本说明](https://github.com/ai-dynamo/nixl/releases)
