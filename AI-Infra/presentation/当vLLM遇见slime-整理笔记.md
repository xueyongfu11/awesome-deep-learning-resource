# 当 vLLM 遇见 slime：vime、vLLM RL 与 Agent 训练实践

## 一、vLLM RL 生态的演进

### 1. 从 OpenRLHF 到 veRL

ChatGPT 兴起后，OpenRLHF 成为 RLHF 工程化的重要里程碑。它奠定了后来 RL 框架的一种典型形态：使用 Ray 协调资源，以成熟的 LLM 推理引擎承担 rollout，并把训练引擎和推理引擎组合成完整的后训练流程。

随着实践深入，社区逐渐发现 rollout 已经成为 RL 训练的显著瓶颈，生成时间可能占总耗时的一半以上。参数同步等问题得到解决后，训练与推理解耦的 RL 架构逐步成熟。

2024 年底，veRL 以 single-controller 架构和训推共卡等训练范式进入社区，又赶上 DeepSeek-R1 带来的强化学习热潮，迅速形成了庞大的开发者生态。主讲人提到，veRL 的周会通常有来自字节跳动、美团、NVIDIA 等公司的二十多位贡献者参加；它也已经具备生产级扩展能力，例如 Transfer Queue，以及面向多模态的 veRL-Omni。

今天围绕 vLLM 的 RL 框架已经形成广泛生态，包括 veRL、OpenRLHF、SkyRL、Prime-RL、PipelineRL、AReaL、ROLL、Open Instruct、NeMo RL 等。

### 2. “全面”带来的代价

veRL 的优势是覆盖面广：支持多种训练后端、推理后端、算法和模态，并经过生产规模验证。但全面性也使代码从 OpenRLHF 约 11,940 行增长到约 96,370 行，抽象层级明显增加。

这带来了一种需求错位：成熟团队可能需要完整、开箱即用的能力；研究团队则更在意代码短小、封装少、便于阅读和修改，也希望代码对 coding agent 更友好。现场讨论认为，训练框架的用户往往需要修改算法和流程，因此代码规模、多后端抽象以及整套 CI 的维护成本会直接影响迭代速度。

### 3. slime 的定位

社区因此开始问：“能否得到一个运行在 vLLM 上的 slime？”

slime 的特点是：

- 简单清晰：代码库干净，容易理解。
- 低复杂度：核心只有一个推理引擎和一个训练引擎，没有很多 wrapper 层。
- 灵活：rollout、数据组织、reward、样本过滤等研究者经常修改的部分都提供可插拔接口。
- 社区认可度高：幻灯片将其列为同类项目中 Star 数排名第三，仅次于 veRL 和 OpenRLHF；同时它诞生较晚，成长速度较快。

## 二、RL 框架驱动推理引擎的两种方式

### 1. In-process：Ray + internal API

训练框架在进程内通过 Ray 和内部 API 直接操作推理引擎。

优点是可以直接获得 logprobs、routed experts 等任意内部信息，无须等待服务器新增 endpoint。缺点是与 vLLM 内部实现深度耦合，侵入性强，版本升级时容易失效，而且推理引擎会绑定训练环境。

### 2. Server + HTTP

推理引擎作为独立服务运行，Ray actor 只持有 server handle，训练框架通过 HTTP contract 与服务交互。

优点是耦合浅：框架依赖公开接口，而不是 vLLM 内部实现，更干净，也适合把推理部署到独立甚至跨地域的集群。缺点是能力完全受 server 暴露的接口限制，而 vLLM 面向 RL 的 HTTP API 还不够完整。

以 partial rollout 为例，in-process 模式可以直接实现；HTTP 模式则必须等服务器提供相应端点。SkyRL、Prime-RL 等选择 HTTP 架构的团队因此常常需要在各自框架中给 vLLM 打 monkey patch。

两条路线本身都合理。真正的生态缺口是：vLLM 需要更成熟的、纯 HTTP 的 RL rollout 支持。Cursor Composer 2 的技术报告也验证了这一需求：当本地 GPU 只用于训练、rollout 由外部云厂商甚至跨区域集群提供时，普通推理、权重同步和控制操作都只能通过网络端点完成。

### 3. 现场讨论：internal API 与 HTTP 谁更适合多后端？

有听众提出，internal API 可能更适合单一推理后端；如果框架支持多个后端，就要增加大量抽象。相反，HTTP 可以规定统一接口，而不关心引擎内部如何实现，因此更适合多后端。

主讲人认可这是重要的框架设计视角，但强调两者各有取舍：in-process 能力最完整，HTTP 的边界更清晰。vime 的工作重点不是判定哪种路线唯一正确，而是补足 vLLM 的 HTTP RL 能力。

## 三、长尾请求、partial rollout 与推测解码

### 1. rollout 中的长尾问题

RL rollout 类似批量离线推理：一批请求同时进入引擎，但完成时间不同。如果必须等待全部请求结束才能进入下一步，少量 P99 级慢请求就会让已经完成的请求和计算资源空等，成为整体瓶颈。

可选解决思路包括：

- 算法与系统协同，不强求严格 on-policy，允许一定程度的 off-policy 或异步训练。
- 使用 partial rollout，让已完成的部分先推进。
- 优化长尾请求，例如采用 PD 分离（prefill/decode disaggregation）。
- 在并发逐渐下降、系统从高吞吐阶段转入少量长尾请求的低时延阶段后，启用 MTP 等推测解码手段专门加速尾部请求。

现场有人问 RL 中的 PD 分离和普通推理服务的 PD 分离是否不同。回答是原理没有本质区别：如果 rollout 通过 HTTP 接入，它表现得就是一个普通 serving endpoint，框架不需要关心内部具体部署方式。

### 2. 推测解码何时有收益

推测解码只有在 verification 成本明显小于常规逐 token 解码成本时才有收益。RL rollout 的初始阶段通常被大量请求打满，目标偏向高吞吐；请求陆续结束后，只剩少量长尾请求，此时目标转向低延迟。推测解码更适合后一个阶段，因此需要针对 RL 场景进行调度或按阶段启用。

## 四、vime：slime + vLLM

vime 的定义很直接：`vime = slime + vLLM`。它满足社区对简洁 slime 框架与 vLLM rollout 引擎组合的需求，也推动 vLLM 建立成熟的纯 HTTP RL 接口。

幻灯片给出的硬件覆盖包括 NVIDIA Grace Blackwell、Blackwell、Hopper，以及华为昇腾和 AMD 平台。

主讲人强调，vime 与 veRL 的生态位不同：

- veRL 特性丰富、后端广、适合开箱即用和生产级需求。
- vime 追求与 slime 对齐，轻量、简洁、容易修改，更适合研究者快速适配新算法。
- 在主流 LLM 场景下，两者的核心训练性能不应有本质差异；在已有 Transfer Queue 等成熟能力的场景里，veRL 的数据面能力暂时更强。

## 五、如何持续维护 vime 这个 fork

### 1. 为什么 fork 难维护

vime 是 slime 的早期 fork，维护人力有限，因此策略是复用上游，而不是重写：保持接口对齐，并大约每两周同步一次 slime。问题是上游持续变化，手工同步机械、重复，容易产生 drift。

### 2. 借鉴 Cohere：把 fork 维护变成闭环控制

Cohere 使用 AI agents 自动维护自己的 vLLM fork。其核心不是单次自动改代码，而是控制理论中的闭环：

1. 目标：同步上游后，fork 的功能仍然正常。
2. 干扰：每次上游 release 或 PR 都可能引入冲突、破坏 fork。
3. 测量：用机械镜像之间的 diff、CI 和测试衡量偏差。
4. 修复：agent 根据偏差修改代码。
5. 重复测量和修复，直到偏差归零、测试通过。

没有测试就没有可测量的反馈，也就不存在真正的闭环。

### 3. vime 的知识库与两道验收线

vime 相比 slime 的系统性变化主要是把 SGLang 替换成 vLLM。大部分改动是机械重命名和接口映射，只有少量属于真正的引擎重实现。团队为 agent 准备了三类知识：

- 翻译表：记录 SGLang 参数、接口到 vLLM 的映射。
- 历史表：解释每个非显然改动为何存在。
- 机械镜像：构造一个“slime-as-vime”镜像，作为 diff 基准。

同步后的验收有两道门槛：

1. **代码 diff 稳定**：vime 与机械镜像之间只允许已经人工签字确认的差异；新增 drift 必须人工 review 和 sign-off。
2. **CI 保持绿色**：不仅检查功能，还检查与 slime 的精度一致性，并在主要模型上验证长时间训练的收敛一致性。

因此，mirror diff 是代码目标，端到端 CI、精度和收敛曲线则是运行时测量。

### 4. 现场补充：agent 模型与同步成本

有人问 Cohere 使用什么模型维护 fork，现场没有确定答案。一位参与者分享了维护 veRL-Omni fork 的经验：他们试过多种模型，目前使用 DeepSeek V4 做大版本 rebase，原因是价格低且效果好；一次大版本 rebase 约花费一百元人民币，大多数 CI 可以自动通过，最后只需人工处理少量失败项。这一经验说明，模型选择不仅看编码能力，也要看大规模重复同步的成本。

## 六、vime 的验证结果

### 1. 与 slime 的一致性

团队在代表性模型上比较 vime 与 slime。幻灯片展示的 raw reward、rollout reward 和 rollout/train logprob absolute difference 等曲线高度重合。这里的关键结论不是某条曲线绝对涨幅多大，而是 fork 替换推理后端后仍保持行为和训练趋势一致。

### 2. GLM-5.2 大规模训练

vime 已在 GB300 集群上跑通 GLM-5.2 训练，规模为 16 个节点、64 张 GPU：

- Rollout：EP=8、TP=8、启用 MTP，共 8 个 vLLM 实例。
- Training：PP=4、EP=16、TP=8、CP=2。

## 七、继续扩展时的数据面瓶颈

为了支持更长上下文、多模态和更大规模 RL，rollout 集群每一步都可能向训练集群传输多模态数据、routing replay 信息和 DSA index replay 等数据，以维持训推一致性。

许多框架采用 single-controller：多个 rollout 节点先把数据聚合到零号节点或 head node，再由它分发给多个训练节点，即 `m → 1 → n`。如果 bulk data plane 也经过 controller，中心节点会出现 CPU、GPU-to-CPU 搬运和序列化压力，形成 single-point explosion。

解决方向是将控制面与大数据面分离：controller 只负责轻量管理，数据由 producer 直接发送给 consumer，拓扑变为 `m → n`，消除额外跳数和单点瓶颈。

路线图如下：

- 短期：接入 Mooncake Store（vime#300）和 Transfer Queue（vime#242），使 rollout tensor 跳过 Ray object store。
- 长期：继续依赖 Mooncake 与 Transfer Queue，并与 vLLM 协同设计（vLLM#45221），进一步减少传输跳数和序列化成本。

## 八、Agent RL：黑盒、白盒与 Uni-Agent

### 1. 为什么需要两种 Agent 形态

生产现实要求训练与评测尽可能复现真实使用环境，因此需要支持 Claude Code 等成熟 agent harness。但这类产品通常是黑盒：外部只能运行命令，无法修改内部 system prompt、固化工作流或动态增删工具。

研究场景需要另一种形态。Shell、Desktop、GUI、Browser、Embodied AI 等任务要求研究者能完全定义 system prompt、工作流和工具调用。感知、决策、执行构成 Agent Loop；当整个循环开放、可观测、可修改时，就是白盒 agent。白盒还更容易获得工具超时、轨迹等调试信息。

### 2. Uni-Agent 的统一栈

Uni-Agent 的目标是大规模构建、运行和训练 agent。其结构包含：

- Model Proxy：纯推理可接 vLLM/SGLang server；RL 训练中，白盒走 veRL server client，黑盒走 agent gateway。
- Agent Tools：支持 Coding Agent、Search Agent、GUI Agent、Lark ChatBot Agent 等。
- Environment：文件系统、搜索引擎或数据库、桌面/浏览器，以及火山引擎、Modal、Docker 等运行环境。
- Agent Interaction System：让模型使用工具、识别或修改环境，并把环境反馈返回给模型，形成完整 Agent Loop。

Gateway 用于把 Claude Code、OpenClaw 等外部黑盒 harness 插入 RL 训练。它在 RL 训练框架与真实应用 harness 之间建立会话，向外部 agent 发送兼容请求，同时收集 response IDs、logprobs 和 trajectory 返回训练侧。

### 3. Reward hacking 与实践结果

幻灯片展示了一组 reward hacking 现象：如果不治理，训练期 reward 持续升高，但 SWE-Bench 测试通过率反而在后期明显下降。采用 Future-Commit Removal 和启发式规则阻断 hacking 行为后，训练 reward 不再虚高，测试通过率却稳步提升。这说明训练期奖励必须与真实泛化结果共同观察。

在 Qwen3.6 36B A3B、SWE-bench Verified 100 题的实践中：

- 官方结果：73/100。
- vime + Modal + Uni-Agent：71.6/100。
- vime + Modal + Claude Code：59/100。

## 九、现场问答整理

### Q1：在哪里参加 veRL weekly meeting？

需要先加入 veRL 的飞书群，群内会发布会议预告。例会通常在每周四上午 11 点，贡献者会分享各自模块最近一周的进展。

### Q2：字节内部训练豆包使用 veRL 主分支吗？

现场回答是：核心模型训练另有内部系统。veRL 团队属于 Seed 的子团队，需求会来自内部，但开源 veRL 并不等同于豆包核心训练所使用的完整内部框架。讨论者据此指出，大型通用开源框架与内部追求快速迭代的专用框架可能有不同目标。

### Q3：vLLM-Omni 何时进入 RL 生态？

当时 vime 主要用 FSDP 做训练，部分大模型使用 Megatron-LM；vLLM-Omni 的 RL 支持还在规划中。关键前置工作是让 vLLM-Omni 通过 HTTP 支持权重同步、定制输出等训练所需接口，并复用 rollout manager group 的部署抽象来支持 PD、AR/DiT、EPD 等形态。

### Q4：一个 rollout server 能否同时提供在线服务和收集 RL 数据？

从接口上可以：它可以同时作为 serving endpoint 接受请求，也可以保存 trajectory 和性能指标。但实践中不建议简单混用：

- RL 训练需要周期性同步权重，在线服务通常要求不中断；同步期间如何处理正在 decode 的请求、是否丢弃旧权重生成结果、是否回滚用户已收到的输出，都很棘手。
- RL 需要把 render、推理、derender 拆开，并可能直接传 token IDs，以保留训练所需的前后处理信息；普通在线推理往往把这些过程封装在服务内部。
- 两类负载的配置和生命周期不同，共享服务会增加实现复杂度。

因此技术上可行，权重同步与在途请求的一致性才是主要障碍。

### Q5：vime 是否支持黑盒 coding-agent RL？

支持。只要 slime 的训练流程支持相应方案，vime 也可以接入。实践中可以把代码和环境放入 Modal 提供的远程 CPU sandbox，再通过 Anthropic 或其他 gateway 提供模型推理服务，从而形成黑盒 harness 训练链路。

### Q6：单轮 token ID 对齐、多轮 token 漂移与 slime 有何不同？

没有本质区别，这些属于 RL 框架的基础一致性问题。vime 的目标是保持与 slime 的行为对齐。

### Q7：有没有适合新手的案例？

案例选择取决于计算资源。现场建议从有代表性、资源要求相对可控的模型开始，例如小规模 dense 模型或 MoE 的激活参数较小配置；更大一些的 30B A3B 类配置可能需要单机 8 卡。多模态和 vLLM-Omni 适配仍需要社区贡献，具备计算资源的参与者可以从相关接口与验证工作入手。

### Q8：如何看待训推不一致？

基础判断通常先看训练曲线与 slime 是否一致；若追求更强的确定性，可以使用训练与推理完全相同的算子，但这会牺牲部分激进优化和性能。讨论中提到 routing/index replay、确定性算子等手段可作为补救，但不应默认一开始就把所有高成本补救全部打开。理想状态是框架本身尽量一致，出现问题后再有针对性地启用 replay 或确定性机制。

### Q9：RL 的策略与打分模型怎么确定？

这是任务相关问题。简单、可验证的任务可以用确定性规则，例如表格填对多少项就按正确项占比打分；复杂任务可能需要模型作为 judge。框架侧更关心流程能否运行、指标是否呈上升趋势，并为 reward、过滤和算法模块提供低侵入的自定义接口；具体怎样的 reward 最有效仍需要算法和训练团队实验。

### Q10：哪些推理优化容易影响训推一致性？

现场重点提到 kernel/算子差异，以及 routing/index 等可能导致不一致的因素。确定性实现或 replay 可以提高稳定性，但会有性能成本。更合理的策略是先使用正常高性能配置，监控误差和训练表现，发现确切问题后再启用对应补救。

## 十、核心结论

1. vime 不是另一个追求“大而全”的 RL 框架，而是用 vLLM 替换 slime 的 SGLang rollout 后端，保留 slime 轻量、灵活、易修改的生态位。
2. vime 的更大价值在于推动 vLLM 建立成熟的纯 HTTP RL 能力，使推理引擎可以独立部署、跨集群乃至跨区域提供 rollout。
3. 维护长期 fork 的关键不是一次性用 agent 改代码，而是建立由镜像 diff、CI、精度和收敛验证共同构成的闭环。
4. 更长上下文、多模态与 MoE 会把 rollout 到 training 的数据传输推向瓶颈，因此必须将 bulk data plane 从 single controller 中拆出，走生产者到消费者的直连路径。
5. Agent RL 同时需要黑盒 harness 的生产真实性和白盒 Agent Loop 的研究自由度；Uni-Agent 的 gateway 和统一交互栈试图把两者接入同一训练体系。
