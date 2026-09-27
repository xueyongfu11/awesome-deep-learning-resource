# 让长序列 MoE RL 训练更好调

## 分享目标与案例

分享围绕一个实际的长上下文 MoE 强化学习（RL）训练案例，目标不是把机器堆满，而是沉淀一套可复用的 recipe：在 **32 张 H100** 上，让 **Qwen3.5-35B-A3B MoE** 模型稳定运行 **128K** 全局序列长度，服务 RL 的 prompt、rollout 与 tool-use 场景。

与预训练不同，RL 通常资源受限，且应将资源切成多组并行实验，快速探索 reward、rollout、数据配方和超参数。因此关键问题是：能否用较少卡数，快速、稳定地跑通 128K/256K 这类长序列训练。

## 1. 为什么长序列 MoE RL 难调

训练配置不是独立旋钮。PP、TP、EP、CP 与 recompute 会共同影响静态显存、动态显存、kernel 效率、关键路径通信暴露和 CPU launch 开销；`local seq` 是这些选择共同作用的结果。

| 选项 | 主要作用 | 主要代价 / 耦合 |
| --- | --- | --- |
| PP（Pipeline Parallel） | 降低静态显存 | pipeline bubble、stage 切分；不能有效降低动态 activation 显存（层数减少了，但同时驻留的 micro-batch activation 往往增多了） |
| TP（Tensor Parallel，含 sequence parallel） | 同时降低静态和动态显存 | GEMM 形状变差、collective 通信及 CPU overhead 上升；还受 KV head / vocab 切分约束 |
| EP（Expert Parallel） | 降低 MoE 静态显存，改善每卡 expert GEMM | all-to-all 通信暴露；expert 不均衡可能反而推高动态显存 |
| CP（Context Parallel） | 切序列、降低动态显存 | attention 通信、额外 kernel 与 CPU overhead；local seq 变小会使效率变差 |
| recompute | 释放 activation，降低动态显存 | full recompute 有约 30% 额外计算 |

讲者特别强调：开并行本质上是为了避免 OOM，但怎样开是系统性取舍。比如 PP 虽切分模型，却因 1F1B 调度需要保存多个 microbatch 的 activation，动态显存不一定降低；TP 会切薄 GEMM 的中间维度，通信和 kernel/CPU 开销则更明显。

### 两条路线

**路线 A：少重算、缩小单卡序列。** 保留 activation（或 offload），需要更大的 CP 来压低单卡 local sequence length。代价是 kernel 变碎、CPU launch 与通信压力上升，因此 CUDA Graph 往往成为必要能力。

**路线 B：接受 full recompute、拉大单卡序列。** full recompute 先释放动态显存，因此可以降低 CP、增大 local seq，进而改善 kernel 形状，并更好掩盖 CPU launch 与通信开销。它的固定代价约为 30% 额外计算。当前长序列 RL 的主流实践更偏向这一路线；本分享主要展开路线 B。

## 2. 从 baseline 到 recipe 的优化路径

### 2.1 建立可运行 baseline

baseline 搜索可按两组维度拆分：`PP × EP` 主要处理静态显存，`TP × CP` 主要处理动态显存。对该模型，KV head 数为 2；在未做 replication 的前提下，TP 最大只能取 2，剩余拆分主要交给 CP。

在 35B、128K、32×H100 的搜索中，PPT 给出的代表性可运行点为：

- `TP=2, PP=4, EP=8, CP=2, full recompute`：127.53 TFLOPS/GPU，42.91 GB peak。
- `TP=2, PP=2, EP=16, CP=4, full recompute`：81.1 TFLOPS/GPU，38.2 GB peak。
- 更激进的 `EP=16`、`TP×CP=8` 虽可跑，但只有 76.9 TFLOPS/GPU、57.6 GB peak。

这说明仅通过 5D 并行找 OOM 边界，往往得到“可用但不便宜”的点；后续要优先拆显存，再优化通信。

### 2.2 先拆 logits peak：linear CE

长序列下，最后一层完整 fp32 logits 会形成不可忽略的 loss-side peak。其每卡估算为：

`local_tokens × vocab_partition × 4 bytes`

`TP=2, CP=2` 与 `TP=1, CP=4` 在该案例中均会产生约 **32.55 GB** fp32 logits peak。采用 linear cross entropy，不生成并保留完整 logits，可以释放该峰值空间，使并行布局可进一步向 `TP=1, PP=2, EP=8, CP=4, full recompute` 移动；对应 **162.07 TFLOPS/GPU、55.91 GB peak**。

### 2.3 FSDP2：将静态状态与 PP/EP 解耦

传统 distributed optimizer 近似 ZeRO-1：optimizer state 已切分，但 parameter 与 grad 仍绑定 PP/EP。FSDP2 更接近 ZeRO-3，将 parameter、grad 与 optimizer state 做全局 shard。

在不改并行配置的比较中，FSDP2 将 peak 从 **55.91 GB** 降至 **47.03 GB**，吞吐为 **163.06 TFLOPS/GPU**。此后各旋钮职责更清楚：

- CP 负责把 local seq 调到显存和计算效率都合适的位置；
- EP 负责 MoE 计算和通信效率，避免跨机 all-to-all 成为瓶颈；
- PP 只在极限性能或 peak 时再考虑，不再默认承担省静态显存职责；
- TP 默认不进入 recipe，除非模型结构或硬件约束要求。

### 2.4 减少 PP bubble

在 FSDP2 释放静态状态后，PP 可从 2 降至 1，减少 pipeline bubble。代表配置为 `TP=1, PP=1, EP=8, CP=4`，达到 **180.18 TFLOPS/GPU、60.54 GB peak**。

### 2.5 chunked EP overlap：处理 MoE all-to-all 暴露

MoE 的 dispatch / combine all-to-all 仍会暴露在关键路径上；旧的 1F1B overlap 不适合 full recompute 路径，退回少重算又会把动态显存、local seq 与 CPU 开销问题带回来。

解决方法是按 token 维度把工作切为 chunk，让通信流与计算流错位并行：

- **no-chunk**：一次 `dispatch → grouped GEMM → combine`；
- **chunk2**：`c0 dispatch → c0 GEMM → c0 combine` 与 `c1` 对应工作交错，使 dispatch/combine 尽可能落到 expert compute 窗口内；
- backward 中还需正确处理 recompute forward、combine backward、dgrad、dispatch backward 与 delayed wgrad 的依赖和 buffer 生命周期；理想情形可将 recompute forward 的若干通信/计算操作与 backward 融合调度。

单层稀疏 MoE（仅 fwd+bwd，不含 attention 和完整 train step）的测量显示，优化后峰值显存随序列长度增长更缓；相对 baseline 的 step 加速从 4K 的 **7.8%**、8K 的 **8.5%**、16K 的 **12.9%**、32K 的 **18.4%**，到 64K 的 **24.0%**。

### 2.6 结果与可复用 recipe

整体路径为：

| 阶段 | 主配置 | 关键变化 | 代表结果 |
| --- | --- | --- | --- |
| baseline | TP=2, PP=4, EP=8, CP=2 | 常规并行 + full recompute | 127.53 TFLOPS/GPU，42.91 GB |
| linear CE | TP=1, PP=2, EP=8, CP=4 | 不生成/保存完整 logits | 162.07 TFLOPS/GPU，55.91 GB |
| FSDP2 | TP=1, PP=2, EP=8, CP=4 | 静态状态全局 shard | 163.06 TFLOPS/GPU，47.03 GB |
| PP=1 | TP=1, PP=1, EP=8, CP=4 | 减少 pipeline bubble | 180.18 TFLOPS/GPU，60.54 GB |
| chunked EP | TP=1, PP=1, EP=8, CP=4 | 覆盖 EP all-to-all | 约 190 TFLOPS/GPU，37–38 GB |

最终小空间 recipe 是：**full recompute + linear CE + FSDP2 + chunked EP**。实践建议是长序列默认 `EP=8`，再按每 GPU 的目标 local seq 反推 CP；PP 仅按需调，TP 默认收起。若 EP 继续增大，要警惕 all-to-all、expert imbalance 和计算不均衡。

## 3. 工程复盘：为什么适合在 Megatron-Lite 中推进

Megatron-Core 的能力完整，以上优化也都能实现；难点在于接入成本：5D 并行、model/parallel state/optimizer/recompute/MoE dispatch 等改动链路长，agent 也较难拆解和验证。

Megatron-Lite 不替换底层 kernel，而是重组上层形式：

`Runtime（训练协议） → Model（选择与组合） → Primitive（可替换能力）`

它与 Megatron-Core 复用相同底层 kernel，因此性能和精度一致；改变的是组织形态与改动边界。此次三个优化对应三个独立 primitive：

- linear CE：loss primitive，处理 logits peak；
- FSDP2：optimizer / state-sharding primitive，处理 static state；
- chunked EP：MoE communication primitive，处理 A2A overlap。

每个 primitive 有清楚的输入、输出与验证路径。推荐的 agent 协作闭环为：**阅读 skill/primitive 说明 → 修改一个局部实现 → 跑 paired numerical test → 比对 loss、grad、peak、time → 组合进模型**。这样 agent 无需一次吃下全栈框架，优化能独立开发、验证和回归。相关能力计划 upstream 至 Megatron 的 dev 分支。

## 4. 问答与补充

### full 与 selective recompute 的区别？

full 是每层计算后丢弃 activation，反向时完整重算；selective 只重算性价比高的部分，例如 core attention 与 MoE activation。分享的主路径采用 full recompute。

### Megatron-LM / Megatron-Lite / AutoModel 如何选择？

Megatron-LM 是 NVIDIA 的正式训练产品；Megatron-Lite 是在 Megatron 开发分支 `experimental/lite` 中探索“用 agent 把训练和框架优化做好”的实验性工作。AutoModel 是另一个基于 Hugging Face 原生 modeling 做优化的训练产品。若追求极致性能，优先 Megatron；若希望开箱即用并获得合理性能，AutoModel 是合适选择。

### 如何学习 Megatron 这类代码库？

讲者建议直接让 AI 结合代码回答问题，并让 AI 指出相关实现位置；即使不能完整读懂代码，也可以先问清各部分实现了什么。Megatron-Lite 的 README 和按 primitive 拆分的组织方式，也能作为更容易进入 Megatron 的入口。

### 训练与推理优化是否相同？

GPU 加速的基本原理相通；但训练要处理 backward，并保存更多 activation，因此约束不同。对于 RL 训练—推理一致性，语音中提到的常见实践包括 QAT、RoPE 相关处理，以及将 kernel 替换为 batch-invariant kernel。

### QAT 是什么？

QAT（Quantization-Aware Training）用于低精度训练一致性。例如 rollout 使用 4-bit、训练使用 16-bit 时，可在训练中先量化至 4-bit、再反量化（fake quant）；计算精度模拟为低比特，从而减小训练与 rollout 的数值差异。若两侧均为 BF16，则需要进一步深入 kernel 层处理差异。

## 后续方向

- Primitive：用 Megatron-FSDP 上位替代 FSDP2；继续推进 chunked EP 的训练用 mega kernel；支持 multi-LoRA、QAT。
- Model：提供更多热门模型实现，提及 GLM 5.2、Kimi K3 等。
- Runtime：对接更多 RL 框架（已打通 veRL），完善异步 rollout、weight resharing 等训练协议。

PPT 还给出可直接运行的参考脚本：`Megatron-LM` 仓库 `lite` 分支下的 `experimental/lite/examples/verl/scripts/run_deepseek_v4_dapo.sh`。
