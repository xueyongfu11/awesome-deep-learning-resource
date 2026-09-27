# 从投机解码、Speculators 到 DSpark

## 分享主线

赵杉佳雯（Helen Zhao，Red Hat Machine Learning Engineer、vLLM 生态中 Speculators 的 maintainer）介绍了四个层次：投机解码的基本原理、Speculators 训练与部署框架、DSpark 算法，以及借助 vLLM 和 Mooncake 的集群规模训练。

核心结论是：投机解码在保持 Target 模型精确输出分布不变的前提下加速 decode；DSpark 以 DFlash 的并行块草稿为基础，再用轻量 Markov Head 恢复块内 token 依赖，兼得吞吐和接近自回归的草稿质量。大规模训练的关键工程问题则是：如何把 Target 的 hidden states 在线、跨节点地可靠送至 Drafter 训练器。

## 1. 投机解码：小模型提议，大模型一次验证

大模型越来越大。投机解码（speculative decoding）的思路是训练一个远小于 Target 的 Drafter：大部分计算先由 Drafter 产生若干 token 草稿，Target 只需要一次前向传播并行验证这些草稿。

以 `jumps / over / the / lazy` 为例，Drafter 一次提出 4 个 token。Target 一次前向为全部位置打分，依次接受最长的正确前缀；若前三个被接受、第四个 `lazy` 被拒绝，则从 Target 分布中重采样该位置，例如输出 `dog`。因此这一轮实际提交 4 个 token：3 个被接受的草稿加 1 个 Target 的 bonus token。

这里的“接受”不能简单理解为 Target 的概率是否高于 Drafter；实际使用的是基于 logprob 的拒绝采样。它保证最终输出与单独运行 Target 时的精确分布一致，因此是**无损加速**。和量化等可能改变模型输出的优化不同，投机解码只要不划算即可关闭，风险很低。

两个重要指标：

- **接受率（acceptance rate）**：每个草稿位置被接受的概率。
- **接受长度（acceptance length）**：每次 Target 步骤平均提交的 token 数，包含 bonus token；它直接决定实际加速效果。接受率越高，平均接受长度通常越长，Target 前向与相关的数据搬运就被摊薄得越多。

早期的方案直接把自然语言 token 输入小模型，让较小的同系列模型为大模型打草稿，效果并不突出。EAGLE-3 之后，Drafter 改为读取 Target 在 prefill 中产生的多层 hidden states。这样小模型能够利用 Target 的内部表示，近似“看到”其思考过程，提出更接近 Target 的 token，从而提高接受率和接受长度。

传统 EAGLE-3 的 Drafter 通常是很小的一层 Transformer，并自回归地连续前向多次来提议多个 token。较新的 DFlash、DSpark 与 P-EAGLE 则可在一次 Drafter 前向中并行产生约 8–16 个 token，因此草稿阶段更快。

### 关于 P-EAGLE 的讨论

直播中有提问认为 P-EAGLE 是否通过双向注意力一次推理多个 token。回答是：其注意力仍是 causal，并非改成双向注意力。它通过 mask token / mask hidden states 等机制，弥补并行预测时缺失的前序 token 信息；其结构与 EAGLE-3 接近，但通过并行多 token 预测和 COD sampling 的内存优化获得更快草稿速度。

对“Drafter 给 Target 的 logprobs 的 shape”这一问题，讲者说明 shape 会随模型而变；实践中还可能截取较高置信度的候选，或做 token embedding 的降维/裁剪，因为 Drafter 不一定需要覆盖全部罕见词表。

## 2. Speculators：训练、转换、部署的一体化库

Speculators 覆盖 Drafter 的训练、已有模型转换和微调、以及 vLLM 中的直接部署。训练产物采用 Hugging Face 兼容格式，vLLM 能从 `config.json` 自动识别草稿算法和对应 verifier，例如：

```bash
pip install speculators
vllm serve RedHatAI/Qwen3-8B-speculator.dflash
```

它支持在线、离线及混合式 hidden-state 提取；单层、多层 Drafter 训练；MoE、非 MoE 与视觉语言模型。训练阶段提供 TensorBoard、Weights & Biases（W&B）曲线，便于发现过拟合并及时停止，避免继续消耗 GPU。

数据处理的一个重点是用 **Target 模型重新生成训练回复**。如果给 Qwen 训练 Drafter，却采用 ChatGPT 或 Claude 等其他模型生成的回复，Drafter 即使能生成合理文本，也没有学会 Qwen 的输出风格与分布，难以被 Qwen 高接受率地验证。训练时通常只对 Target 产生的 assistant token 计算损失，用户 prompt 则用 loss mask 排除；Speculators 也努力自动适配不同聊天模板以构造这个 mask。

训练、算法配置和 verifier 会一起写入 checkpoint 的 `config.json`。其中 `speculators_model_type` 决定 vLLM 的草稿路径，`verifier` 锁定训练时的确切 Target，`block_size`、辅助 hidden-state 层及 `markov_rank` 等字段同训练命令一一对应。也就是说，配方与 checkpoint 是同一对象，不需要另外的胶水代码。

## 3. Drafter 算法谱系

| 方法 | 草稿方式 | 分享中的定位 |
| --- | --- | --- |
| Draft Model | 传统小模型逐 token 草稿 | 相对过时，效果有限；Speculators 不支持这一路线 |
| MTP / FastMTP | 微调 Target 自带的多 token 预测头，但仍自回归产生 token | 许多模型提供商随模型提供 |
| EAGLE-3 | 融合多层 Target hidden states 的小 Drafter 层，自回归 | 最成熟、vLLM 支持最稳定 |
| P-EAGLE | EAGLE-3 基础上的并行多 token 预测 | 使用 mask 状态和 COD sampling，某些场景加速可接近 DFlash |
| DFlash | 非因果 mask token 的锚定并行块、多层辅助状态 | 并行预测，快，但块内依赖/后部质量可能衰减 |
| DSpark | DFlash 并行骨干 + Markov Head | 兼顾并行吞吐和自回归式块内依赖 |

Red Hat AI 已发布 28 个 Drafter checkpoint（当时包括 EAGLE-3 19 个、DFlash 7 个、DSpark 1 个、P-EAGLE 1 个），覆盖 Llama、Qwen3、gpt-oss、Gemma、GLM、DeepSeek、Nemotron 等模型系列，并会持续发布。

### 与 RL 的关系

直播问到 Speculators 是否已集成 RL 框架。回答是目前没有，但未来可能集成。原因是：若在 RL/后训练中更新 Target 权重，原有 Drafter 会与 Target 发生偏移；若希望保持高接受率，理想状态是两者协同训练。因此模型提供商自行联合训练的 Drafter 往往具有优势。讲者认为可考虑与 vLLM 生态的 RL 框架合作。

## 4. DSpark：并行块与 Markov 修正

DSpark 的出发点是解决纯并行块草稿的矛盾：DFlash 一次前向即可为所有位置产生草稿，吞吐高，但块中各 token 的依赖不如自回归生成充分，接受率可能随块内位置下降；自回归方法更准确，却需要多次前向。

其一次解码循环如下：

1. Target 处理当前上下文，得到锚点 `D`。
2. DSpark 的 DFlash 式并行骨干以锚点和多个 mask position 为输入，一次非因果前向同时得到块内位置的表示 `U1…U4`。
3. 轻量 **Markov Head** 按从左到右的顺序，为下一位置加入低秩偏置 `B_k`，形成 `E、F、G、H` 的依赖式预测；它不需要重跑重型骨干。
4. 置信度头与前缀调度器根据 `c1…c4` 仅保留高置信前缀，例如保留 `E、F、G`，丢弃 `H`。模型越不确定，草稿越短，避免浪费 Target 的批次容量。
5. Target 并行验证保留的草稿；例如 `E、F` 被接受、`G` 被拒绝，则输出 Target 重采样的 `G*`，并用它作为下一轮锚点。

因此 DSpark 保留 DFlash 的“一次前向拿到整块”的吞吐，又以近乎零成本的 Markov 修正恢复接近自回归的块内一致性。

PPT 所示、以 Qwen3-14B 为 Target、温度 1.0、链式草稿且关闭置信度调度器的离线接受长度如下：

| 基线/方法 | GSM8K | MATH500 | MBPP | HumanEval | MT-Bench |
| --- | ---: | ---: | ---: | ---: | ---: |
| EAGLE-3 | 5.24 | 4.60 | 3.81 | 4.14 | 2.62 |
| DFlash | 5.41 | 4.84 | 4.44 | 4.59 | 3.10 |
| DSpark | **6.21** | **5.74** | **5.26** | **5.43** | **3.70** |

讲者引述的结果显示：DSpark 相对 EAGLE-3 的宏平均接受长度提升约 30.0%（4B/8B 的结果为 +30.9% / +26.7%），相对 DFlash 提升 18.3%（4B/8B 为 +16.3% / +18.4%）。在 Qwen3-4B 的块内位置接受率中，DFlash 虽在第一个位置领先，但会在块内衰减；DSpark 从约 0.93 起步且整块更平稳。Markov 序列头额外单轮延迟只有约 0.2%–1.3%（块长 4–16、batch 128）。

## 5. 训练一个 Drafter：三条命令

训练流程是“准备数据 → 启动 hidden-state 提取服务器 → 训练草稿模型”：

```bash
# 1. 分词并构建 loss mask
python scripts/prepare_data.py --model zai-org/GLM-5.2-FP8 --data data.jsonl \
  --output ./out --seq-length 8192 --assistant-pattern '<|assistant|>…'

# 2. 由 vLLM 流式提取 Target 的多个 hidden-state 层
python scripts/launch_vllm.py zai-org/GLM-5.2-FP8 --target-layer-ids 8 23 39 55 70 \
  --hidden-states-backend mooncake --mooncake-protocol rdma -- --tensor-parallel-size 4

# 3. 在线消费 hidden states 训练 DSpark
torchrun --standalone --nproc_per_node 4 scripts/train.py --speculator-type dspark \
  --verifier-name-or-path GLM-5.2-FP8 --num-layers 5 --block-size 8 --markov-rank 256
```

提取后端可在 `example`（共享存储）和 `mooncake`（RDMA、多节点）之间切换。这里 `markov_rank` 是低秩分解的秩，示例为 256。

讲者提到，命令表面上仍有一定复杂度，因此项目准备了 tutorial，并计划增加便于 Claude Code 等工具理解参数语义的上下文/技能说明。

## 6. 运行时与大规模 hidden-state 工程

### vLLM Model Runner V2

运行时中，Target 前向、GPU 验证和整个 Drafter 都在一次 worker 调用内完成，并和 CPU 的下一批调度/输入准备重叠。草稿 token id 在 worker 内更新，不再在调度器与 worker 间往返；整个草稿步骤可捕获为 CUDA Graph，减少小 Drafter 的 kernel 启动开销。整块 GPU 验证采用 shared-Gumbel 随机性拒绝采样，降低 TP 通信，并与 CUDA Graph 兼容，稳态下接近零气泡。

### 为什么需要在线流式 hidden states

Drafter 要在大量 token 上蒸馏 Target 的内部状态。离线方式先由 vLLM Target prefill 写盘、再由 DDP Drafter 回读训练，存在两层问题：Target 或精度一变化就必须重做全部数据；写入与回读都形成带宽墙。示例中，GLM-5.2 在 OpenPerfectBlend 上重生成 32 亿 token 的 hidden states，按约 72 KB/token 计算，dump 规模约为 **239 TB**。

在线方式则是 `produce → consume → free`：Target 边 prefill 边提供 hidden states，trainer 按批请求并消费，用完即释放，传输路径不经过磁盘，因而没有陈旧数据与磁盘回读压力。

### 复用 KV Connector

实现没有另建 hidden-state 提取通路，而是复用 vLLM 已经成熟的 KV Connector：从 Target 前向中抽取辅助层（如 `[8, 23, 39, 55, 70]`）的激活，将其“伪装”为 KV，直接写入 KV cache 槽位，再交由 connector 送至训练器。

其形状为 `[T, L, H]`：把层轴 `L` 视作 `num_heads`、hidden size `H` 视作 `head_size`，就能与 KV 槽位严丝合缝地对齐，无需 reshape、无需额外注意力计算，也可顺带复用前缀缓存能力。

最初的实现要求 trainer 和 hidden-state extractor 位于同一节点，通过共享内存和文件/log 机制协调读写。这不适合更大的模型：一个节点往往既放不下 Target，也放不下训练中的 Drafter；同时，生产端若在消费者掉线后持续产生数据，磁盘可能被写满。因此所需系统应当支持跨节点、生产与消费解耦、明确背压和过期数据驱逐。

### Mooncake connector 的取舍

该路径的硬要求是：

1. 跨节点且不假设集群有高速共享文件系统；
2. 按 key 寻址，生产者无需知道哪个消费者、何时消费；
3. 单边读取，不能占用仍在在线 prefill 的生产者 CPU；
4. 有界、可驱逐，以控制背压并清理长期未消费的数据。

共享文件系统会把磁盘放在关键路径，且依赖集群已有快 FS；NCCL 等集合通信要求固定通信域和锁步参与，不匹配异步、动态、多对多的生产消费；直接使用 RDMA/UCX 虽可传输，但寻址、生命周期、驱逐和背压都需要重新实现。Mooncake Store 同时满足这四项，并已被 vLLM 的 PD 分离使用，因此成为适合的 connector。讲者提到 Speculators 已有相关 PR，用它支持跨节点训练，从而可为更大的 Target 训练 Drafter。

### 关于并行切分的问答

- 生产者和消费者采用相同切分、hidden parallel 方式时，理论上可使用 P2P 传输；具体仍取决于 Mooncake Connector 实现。
- 训练场景可选 P2P、all-gather、broadcast 或混合后端。由于最终传输和还原的是 hidden states，其切分方式本身未必敏感；但在训练 group 内可能仍需 broadcast，或逐个 rank 传输。后者会提高通信负载。
- 对“复用 hidden states 是否与 PD 分离类似”的理解，讲者认同两者有相似性。
- 训练通常不需要 PD 分离。该训练过程对 Target 只需 prefill 的 hidden states，不需要 decode；decode 仅用于之后在线推理中的投机解码。

## 7. 其他直播问答与实践边界

### 部署兼容性

关于哪些 vLLM 优化不能与投机解码同时开启，讲者暂未遇到明确的不兼容项，建议查看 vLLM 的 speculative decoding recipe。投机解码主要加速 decode，不加速 prefill；因此某些仅与 prefill 有关的设置不会与它形成直接配合。团队也希望积累更多 runtime 配方，给出更优化的部署建议。

### 数据规模和继续训练

从头训练时，分享中常用约 **50 万** 条调优数据；也在试验约 **100 万** 条的 OpenPerfectBlend。现在已支持多轮对话数据，而不再只支持单轮样本。

若从已有 DFlash/DSpark 模型继续训练，数据需求可以显著下降。讲者听到过用户以约 3 万到 7 万条生产环境数据微调并获得较好加速的案例。Speculators 可通过 `--from-pretrained` 加载已训练的 DFlash 或 DSpark，在自己的生产数据上微调；DFlash 模型也可以较容易转换为 DSpark 模型。

### 多模态与长上下文

库支持图像数据和视觉语言模型训练，已有用户在使用，但团队尚未完成深入的多模态研究。投机解码主要加速文本部分，图像部分通常不承担 token 草稿预测；欢迎社区反馈需要支持的数据格式和改进点。

过去若训练时 `max_model_len` 很小（如 2048），推理时超过该长度可能发生明显性能退化。团队加入了 sliding-window attention 优化后，训练长度可适度短于推理上下文，长上下文的回归不再明显，部分场景甚至可以持平；更长上下文的训练优化仍在进行。

### Diffusion 模型是否需要投机解码

提问中讨论了 diffusion language model。讲者认为现实落地中此类模型较少，且纯并行生成往往存在精确度不足的问题；强行再加投机解码未必有意义，也缺少专用推理引擎。vLLM 对 Gemma Diffusion 的支持涉及 DFlash；社区若有具体需求可进一步研究。

## 8. 收束

投机解码的价值在于：它把 Target 的精确分布作为最终裁决，Drafter 只负责尽可能多地提交可被接受的高质量草稿，所以能够在无损条件下带来实际 decode 加速。Speculators 将数据准备、hidden-state 提取、训练、转换和 vLLM 部署串成一条路径；DSpark 则用并行块加 Markov 修正，把草稿吞吐与块内一致性结合起来。

分享最后邀请用户试用、反馈短板和加速效果，并欢迎将使用 Speculators 训练的 Drafter 贡献到 Hugging Face。相关入口：Speculators 代码库 `github.com/vllm-project/speculators`、文档 `docs.vllm.ai/projects/speculators`、vLLM Community Slack 的 `#speculators` 与 `#feat-spec-decode`。
