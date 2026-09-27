# vLLM DCP & PCP 技术分享
分享人：邱纯硕
vLLM PCP 特性核心贡献者 @华为昇腾

---

## CP 特性总览

tl;dr PCP对prefill的输入序列进行切分，降低TTFT；DCP 对 KVCache 进行序列切分，消除冗余存储，提升最大长度与吞吐。

/image static/images/CP/overview.svg

---

## vLLM DCP 特性
PR https://github.com/vllm-project/vllm/pull/23734
@youzhedian(Chao Hong from Moonshot)

---

### DCP Prefill 方案

/image static/images/CP/DCP-prefill-nochunkedprefill.svg

-> 支线

### BlockTable 变化

引入 virtual block 概念处理跨设备 block 管理，主要影响 block 分配、prefix caching 命中。

/image docs/html/blocktable.html

---

### BlockTable 变化

引入 virtual block 概念处理跨设备 block 管理，主要影响 block 分配、prefix caching 命中。

/dynamic token分配情况随DCP size和 interleave size变化而变化

---

### Slot mapping 分配变化

/dynamic token长度变化、DCP size 和 interleave size 变化导致 不同 DCP rank上slotmapping取值变化。

<- 返回主线
---

### DCP Decode 方案

/image static/images/CP/DCP-decode.svg

---

### DCP ChunkedPrefill 方案

/image static/images/CP/DCP-prefill-MLA.svg
/image static/images/CP/DCP-prefill-GQA.svg

---

## vLLM PCP 特性
RFC https://github.com/vllm-project/vllm/issues/25749
@pisceskkk(QiuChunshuo from Huawei)

---

### PCP 通信组

/dynamic 

---

### PCP 序列切分

/image static/images/CP/PCP-dualchunkswap.svg

该部分切分逻辑在cp_utils.py:PCPManager中处理，包括处理query_lens, kv_lens, positions等，也包括计算q_head_idx, q_tail_idx等。

---

### PCP prefill 方案

/image static/images/CP/PCP-prefill-nochunkedprefill.svg

---
### PCP Decode 方案

/image static/images/CP/PCP-decode.svg

---
### PCP Chunked Prefill 方案

/image static/images/CP/PCP-chunkedprefill.svg

---

## 部署场景

---

### DCP 部署

#### 时延
通常来说，DCP在短序列情况下会因为额外通信导致劣化;长序列情况下（通常为128k以上）则因为kvcache切分独立计算而降低TOPT。

#### 吞吐
在资源紧张的情况下通过DCP提升kvcache容量，可以处理更多请求，吞吐会有一定提升。

#### 请求长度
开启DCP后kvcache可用量大幅提升，常见模型可以在单机 A3 上推理超过 1M 的超长序列。

### PCP 部署

#### 性能
中长序列（32k～256k）场景下时延优于DP,略逊于TP，吞吐优于TP，略逊于DP。

#### 场景
适合在请求并发量与系统最优吞吐接近时进一步降低时延。


## CP 特性优化与拓展

---

### TPA 特性（tensor-parallel-size-attention）
RFC https://github.com/vllm-project/vllm/issues/34444
@SungsooHa from NVIDIA

/mermaid 流程图1 hidden_states -> q_proj(slice by TP) -> Q(1, h/tp) -> DCP.all_gather(dim=1) -> attn -> cp_lse_ag_out_rs(DCP) -> output(1, h/tp) -> o_proj(slice by TP) -> out
/mermaid 流程图2 hidden_states -> q_proj(slice by TPA=tp/dcp) -> Q(1, h*dcp/tp) -> attn -> cp_lse_ag_out_rs(DCP) -> output(1, h/tp) -> o_proj(slice by TP) -> out

### 动态 CP 特性
RFC https://github.com/vllm-project/vllm/issues/29295

/mermaid
```
flowchart LR
    REQ["请求到达"]
    REQ --> JUDGE{"CrossDPScheduler<br/>按序列长度判断"}

    JUDGE -->|"短序列"| CP1["CP=1 标准 DP 通路"]
    JUDGE -->|"长序列"| CPSEL["选择最优 CP size<br/>(预建立通信组)"]
    JUDGE -->|"长尾请求"| MIG["从 CP=1 迁移到 CP>1"]

    CP1 --> DOMAIN["Domain<br/>聚合多个 DP replica<br/>为统一调度实体"]
    CPSEL --> DOMAIN
    MIG --> DOMAIN

    DOMAIN --> SCHED["list SchedulerOutput<br/>跨 DP domain 的异构调度决策"]
    SCHED --> RUNNER["ModelRunner<br/>CP 请求重排到 batch 最前<br/>2D CUDA Graph"]

    RUNNER --> KVCACHE["CrossDPKVCacheManager<br/>统一管理跨 DP 的 Block Pool"]
    KVCACHE --> ATTN["Attention Backend<br/>Prefill CP + Decode CP"]
```

实测中短序列（4k-128k）与变长场景的吞吐和TTFT均优于DP,TPOT也没有明显劣化。

---

感谢页