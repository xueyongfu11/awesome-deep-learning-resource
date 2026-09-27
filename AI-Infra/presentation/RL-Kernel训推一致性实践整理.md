# RL-Kernel：大模型 RL 后训练中的训推一致性实践

## 1. 问题背景：同一模型、同一 token，为什么 LogP 会不同

本次分享介绍 RL-Align 的 RL-Kernel v0.1.0，主题是大模型 RL 后训练中的“训推一致性”。核心问题是：同一个模型、同一个 token，为什么 rollout 阶段和 training 阶段计算出来的 LogP 可能不同？

在典型 RL 流程中，rollout 使用 vLLM，根据当前权重和 prompt 生成 response tokens，并记录每个 token 的 LogP；training 使用 Megatron-LM，在相应权重和 prompt、response tokens 上重新计算 LogP。通常需要比较的是同一 selected token 的 `rollout_logp` 与 `train_logp`。

如果两者存在偏差，importance ratio 就会偏离 1，并可能触发 PPO 等流程中的 clipping。这样即使模型参数和输入相同，训练信号也会受到额外数值误差影响。

## 2. 偏差来源：底层执行路径与浮点加法顺序

rollout 与 training 往往使用不同的框架、kernel、并行方式和显存布局。即使数学表达式相同，底层计算路径也可能不同，从而产生低位浮点差异。

一个典型例子是浮点加法的非结合性。对 `a、b、c、d` 而言，`((a+b)+c)+d` 与 `(a+b)+(c+d)` 在数学上等价，但实际浮点结果可能不同。GPU 为提升吞吐会把大规模计算切分到多个线程、block 或 rank；局部结果再合并时，归约顺序不同就会改变末位结果。

这一问题会出现在：

- GEMM 的 K 维归约；
- Attention 中对可见 key 的归约与 softmax；
- TP/CP 等并行切分后的 rank 间合并；
- vocabulary 维度上的 LogSumExp 与 log-softmax；
- 不同 kernel 对 split-K、tile、chunk 的选择。

LogP 的微小偏差会继续影响 importance ratio、KL 和 clipping，因此不能只比较最终 loss 是否接近，而要定位到 selected-token LogP 的差异。

## 3. 在比较 LogP 前必须锁定的条件

在判断“训练和推理是否一致”前，需要先排除输入和状态不一致：

1. 使用相同的权重版本，并确认 rollout、reference、actor、training 的执行时序。
2. 固定 prompt、response token、token 数量和有效 mask。
3. 固定 position、RoPE、causal mask，以及 Attention 的 KV 可见范围。
4. 固定 token 到 TP/CP rank 的分片映射和逻辑顺序。
5. 固定随机种子、运行参数和必要的 CUDA/ROCm 状态。
6. 确认比较的是同一 selected token，并记录首次出现差异的位置。

因此，排查不能只看“结果不一样”，还要判断是不是输入、权重、mask、位置编码、分片或随机状态已经不同。

## 4. RL-Kernel 的项目定位与工作流

v0.1.0 的定位是为 vLLM、slime、verl 等 RL 框架提供训推一致性的底层 kernel 和约束。它位于 rollout 与 training 执行路径之间，记录关键计算信息，并在 CUDA、ROCm 等硬件平台上复现和约束关键归约过程。

一个 RL step 的基本流程是：

`Prompt → Rollout（生成 token + LogP）→ Reward → Training（重新计算 LogP + 计算loss + 反向传播）→ Weight sync`

RL-Kernel 关注的是权重更新前的同一 RL step：rollout 记录的 LogP 和 training 重新计算的 LogP 应该对应同一权重、同一 token 和同一计算约束，并通过自动化比较确认一致性。

vime 在流程层面组织 rollout、reference、actor 和 train 的先后顺序，记录每一步使用的权重版本与输入，并在参数更新后进行权重同步；RL-Kernel 则负责更底层的数值约束，包括固定 token 顺序、固定归约顺序、固定精度边界和 selected-token LogP 的记录与比较。

## 5. 关键约束：把局部计算变成可复现的全局结果

### RMSNorm

需要固定 hidden dimension 上的归约范围、归约顺序、均值与方差的计算方式，并确认训练与 rollout 使用相同的 RMSNorm 路径。

### GEMM 与 SwiGLU

需要固定 GEMM 的 K 维归约和 tile 的局部结果合并方式。SwiGLU 的 gate/up 与 down projection 要按照约束后的顺序执行，避免 packed 与非 packed 路径带来不同的累计顺序。

### Attention

每个 query 只能在可见 key 上计算 softmax。必须锁定可见 key 范围、split-K 行为、block/chunk 的组织方式及 rescale 的合并顺序。对于严格一致路径，需要关闭会改变归约顺序的 split-K，并使用确定性 backward 路径。

### Linear LogP

每个 token 要在完整 vocabulary 上进行 log-softmax。TP shard 的局部 logits 需要按固定 vocabulary tile 顺序合并，并在同一个 LogSumExp 统计量上计算 selected-token LogP。不能因为 rollout 和 training 采用不同的 local logits 或 LM-head GEMM，就改变归一化结果。

### Collectives

跨 rank 的结果需要使用固定的 collective 拓扑和合并顺序。CUDA 路径使用固定 balanced tree；ROCm 路径使用 HIP IPC 与 RCCL，并固定拓扑约束和 HIP Graph cache 行为。

## 6. 固定 replay：定位第一个 mismatch

验证时要固定 replay 的 token、mask 和 position，保证每次执行拿到完全相同的输入。之后进行逐层、逐 token 的比较，定位第一个出现差异的 LogP；再回到对应 kernel 和归约范围检查。

验证流程包括：

1. 固定权重、token、mask、position、seed 和运行参数。
2. 固定同一条 rollout 输入及 response。
3. 分别得到 `rollout_logp` 与 `old_logp/train_logp`。
4. 对比 selected-token LogP，并记录首次 mismatch 的 token、层和算子。
5. 检查实际 kernel、参数配置与比较结果。

只有在输入、权重和执行路径都固定后，`0 mismatch` 才能说明数值路径达到一致。

## 7. v0.1.0 的验证结果

### CUDA

验证环境为 Qwen3-8B、1 台 8 卡 H100 80GB，TP4/CP2，global batch 128，seed 1234。对比 vime 原始路径与 RL-Kernel + vime，连续运行 200 个 RL steps：

- 200/200 个训练 step；
- RL-Kernel mismatch：0；
- 对比 token 数和 LogP 差异统计均为 0。

CUDA 路径的主要约束包括：

- GEMM 自动选择 cuBLASLt no-split-K，固定 SM90 上的 K 维累计顺序；
- RMSNorm 使用统一且受约束的 torch RMSNorm 路径；
- Attention 使用 FA4 strict core，关闭 split-K，训练 backward 使用 deterministic 路径；
- FFN 使用 packed gate/up，并单独执行 SwiGLU down GEMM；
- LM head / LogP 复用 local logits，避免重新执行产生不同结果的 LM-head GEMM；
- CUDA IPC 使用固定 balanced-tree collective，覆盖 TP1/2/4/8。

复现实验还提供了 native 与 consistency 两条路径。用户只需要替换 workspace 和模型路径，不应修改 VIME 与 RL-Kernel 的实验脚本。

### ROCm

验证环境为 Qwen3-8B、1 台 8 卡 MI300X 192GB，TP4/CP2，global batch 8，seed 1234。连续运行 200 个训练 step：

- 200/200 个训练 step；
- mismatch：0；
- 共比较 9,400,614 个元素，位级结果全部一致。

ROCm 路径的主要约束包括：

- GEMM 使用 MFMA kernel，固定 chunk 顺序，以 FP32 合并后写回 BF16；
- Attention 默认使用 AITER/CK non-split strict core，paged decode 固定 CK 路径；
- 可选 Triton chunked attention，固定 block、chunk 和 rescale 合并顺序；
- FFN 使用 Triton/MFMA 严格 GEMM，固定 packed gate/up 与 down projection 顺序；
- LogP 在 TP shard 上先进行局部统计，再按固定 vocabulary tile 顺序合并；
- 通信使用 HIP IPC 与 RCCL，固定拓扑和 HIP Graph cache。

## 8. 命令行复现要点

基础运行参数示例：

```bash
./rlk run --tp 4 --rollout-tp 4 --temperature 0.7 --top-p 0.95 \
  --lr 5e-7 --kl-coef 0.01 --max-response-len 6912 \
  --max-tokens-per-gpu 4096 --steps 200
```

不同并行配置可以用于检查训练与 rollout 的 TP 不同、CP 切分不同等情况，例如 TP1/CP8 对 rollout TP1、TP2/CP4 对 rollout TP4、TP4/CP2 对 rollout TP8，以及 TP8/CP1 对 rollout TP2。

ROCm profile 复现时，先设置 `RLK_REPRO_PROFILE`，再用 `rlk run` 执行单步检查；`rlk plan` 只负责展示计划，不会执行训练。

## 9. 当前边界与下一阶段

v0.1.0 已在 Dense Qwen3-8B 上完成 CUDA 与 ROCm 的 200-step 一致性验证，覆盖 vLLM 与 Megatron-LM 的 selected-token LogP 对比。当前边界是 Dense 模型和既定硬件、kernel 配置；MoE、动态路由和更多模型结构仍需进一步验证。

下一阶段计划包括：

1. 继续完善 Dense CUDA/ROCm 路径。
2. 支持 DeepSeek-V4 Flash MoE 等 MoE 场景。
3. 为 Gemma、Qwen3-Next 等模型建立适配方案和 RFC。
4. 与 verl、AReaL 等 RL 框架协作。
5. 扩展到 MUSA、Ascend 等硬件平台。

最终目标是让每个 RL step 的 rollout 与 training 在相同 token、权重、位置、mask、归约顺序和精度边界下得到一致的 selected-token LogP，从而降低 importance ratio、KL 和 clipping 中由训推数值偏差带来的不确定性。

## 10. 讨论要点

分享中反复强调：同一模型和同一 token 并不足以保证 LogP 一致；还必须同时锁定执行时机、权重版本、输入 mask、位置状态、rank 分片、kernel 路径、归约顺序和精度转换。出现 mismatch 时，应先固定 replay 并定位第一个差异，再判断是数据、状态还是数值路径问题。

当前验证结论限定在已覆盖的 Dense CUDA/ROCm 配置和 Qwen3-8B 场景。对于 MoE 的动态路由、不同模型架构、新硬件或未覆盖的 kernel 路径，不能直接推断已经达到同样的一致性。
