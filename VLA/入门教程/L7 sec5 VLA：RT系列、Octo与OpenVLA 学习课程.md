# L7 sec5 VLA：RT系列、Octo 与 OpenVLA 学习课程

---

## 0. 这节课在讲什么

本节课的主题是 **VLA：Vision-Language-Action Model**，也就是把视觉、语言和机器人动作统一到一个模型或策略系统里。

可以先用一句话理解：

```text
VLA = 让机器人看见场景、理解语言指令，并直接输出可执行动作的模型。
```

前面已经讲过：

```text
Visuomotor Policy：视觉到动作的策略
VLM / LLM Planning：用视觉语言模型或大语言模型做任务理解和规划
```

本节课把它们连接起来：

```text
Visuomotor Policy + VLM&LLM Planning ≈ VLA
```

也就是说，VLA 试图把“会看、会听懂指令、会动作”放进一个更统一的框架中。

本节重点包括：

1. VLA 和传统 visuomotor policy 的区别
2. RT-1：语言条件模仿学习策略
3. RT-2：把动作接入预训练 VLM
4. RT-X、RT-H、AutoRT 等 RT 系列扩展
5. Octo：较小但通用的机器人 Transformer 策略
6. OpenVLA：开源 VLM + Action 模型
7. VLA 的动作表示、训练数据、模型结构和设计发现

---

## 1. VLA 的核心概念

VLA 的输入通常包括：

```text
图像 / 视频观测
自然语言任务指令
机器人历史状态或历史动作
```

输出通常是：

```text
机器人动作
```

例如：

```text
Input:
  图像：桌面上有杯子和盘子
  语言：把杯子放到盘子旁边

Output:
  机械臂末端位姿变化、夹爪开合、移动底盘控制等动作
```

VLA 想解决的问题不是单个技能，而是更一般的机器人任务：

```text
新任务
新场景
新物体
新机器人平台
复杂语言指令
```

这也是它和传统机器人策略最大的区别。

---

## 2. Visuomotor Policy 与 VLA 的区别

### 2.1 Visuomotor Policy

Visuomotor policy 通常学习从视觉到动作的映射：

```text
image / state -> action
```

如果加入语言条件，可以变成：

```text
image + language instruction -> action
```

它的特点是：

- 主要面向一个或少数几个任务
- 可以通过修改支持语言输入
- 通常不擅长复杂语言推理
- 更关注 skill-level learning，即学习某个具体技能
- 泛化通常发生在同类任务或相似场景中
- 数据规模可以是百级、千级或更高的示范数据

典型例子：

```text
ACT
Diffusion Policy
Behavior Transformer
```

### 2.2 VLA

VLA 的目标更大：

```text
vision + language + action -> general-purpose robotic policy
```

它的特点是：

- 面向通用机器人任务
- 原生支持语言指令
- 希望具备复杂指令理解能力
- 希望泛化到新任务、新场景、新机器人
- 通常依赖大规模机器人数据预训练
- 可以再针对新任务做 finetuning

典型例子：

```text
RT-1
RT-2
Octo
OpenVLA
RDT
Pi0
GR-2
```

需要注意：有些模型严格来说更像语言条件的模仿学习策略，但因为它们连接了视觉、语言和动作，也经常被放在 VLA 讨论范围内。

---

## 3. 背景：从 VLM 到 VLA

VLA 的基础来自两个方向。

第一个方向是 **LLM / VLM**：

```text
Transformer
Tokenization
Large Language Model
Vision-Language Model
```

第二个方向是 **language conditioned policy**：

```text
vision feature + language feature -> action
```

从结构上看，VLA 常见做法有两类：

### 3.1 预训练 VLM + 机器人数据微调

这种路线先拿一个已经具备视觉语言能力的大模型，然后加入动作预测能力：

```text
pretrained VLM
-> finetune with robot action data
-> output robot actions
```

RT-2 和 OpenVLA 更接近这条路线。

### 3.2 简化版 VLM 结构 + 机器人数据训练

这种路线不一定使用完整大规模预训练 VLM，而是设计一个较小的 Transformer 或视觉语言策略模型，用机器人数据训练：

```text
vision encoder + language encoder + transformer + action head
```

RT-1 和 Octo 更接近这条路线。

需要特别注意：

```text
Robotics people usually use VLMs rather than designing them.
```

也就是说，在机器人方向，很多工作不是重新发明 VLM，而是把已有 VLM 接入机器人动作学习。

---

## 4. RT-1：Robot Transformer 1

### 4.1 RT-1 的定位

RT-1 是 Robot Transformer 1。

它是一个相对较小的模型，参数量约 35M。核心特点是：

```text
RT-1 更像 language conditioned policy，而不是真正意义上的大规模 VLA。
```

原因是它没有直接从大型预训练 VLM 迁移而来，而是一个带语言条件的机器人模仿学习模型。

### 4.2 RT-1 的整体结构

RT-1 的输入包括：

```text
图像序列
语言指令
历史动作或上下文
```

输出是离散化后的机器人动作 token。

主要模块如下：

```text
Language instruction
-> Universal Sentence Encoder

Images
-> EfficientNet-B3
-> FiLM 视觉语言融合
-> TokenLearner 压缩 token
-> Transformer
-> discrete action tokens
```

### 4.3 Vision Encoder：EfficientNet-B3

RT-1 使用预训练的 EfficientNet-B3 作为视觉编码器。

图像会被编码成视觉特征，然后通过后续模块变成 token。

### 4.4 Language Encoder：Universal Sentence Encoder

语言指令使用 Universal Sentence Encoder 编码。

这个语言向量不会单独做规划，而是作为条件注入视觉编码过程。

### 4.5 FiLM：视觉语言融合

FiLM 的全称是：

```text
Feature-wise Linear Modulation
```

可以粗略理解为：

```text
用语言特征调制视觉特征
```

形式上可以写成：

```text
FiLM(x, z)
```

其中：

- `x` 是视觉特征
- `z` 是语言条件

直觉上，如果指令是“拿起红色杯子”，语言特征会让视觉编码器更关注与“红色杯子”相关的区域。

### 4.6 TokenLearner：压缩视觉 token

RT-1 中 TokenLearner 的作用是把大量视觉 token 压缩成少量 token。

关键数字是：

```text
91 tokens -> 8 tokens
```

这样做的原因是 Transformer 的计算量和 token 数相关。减少 token 可以降低计算成本，同时保留关键视觉信息。

### 4.7 Action Output：动作离散化

RT-1 把每个动作维度离散成 256 个 bin。

动作维度一共有 11 维：

```text
mode
arm:  x, y, z, roll, pitch, yaw, gripper
base: x, y, yaw
```

所以 RT-1 的动作预测更像分类问题：

```text
每个动作维度 -> 预测一个 0 到 255 的类别
```

训练损失是：

```text
Cross Entropy for discrete actions
```

### 4.8 RT-1 的训练数据

RT-1 使用：

```text
30k tele-operation demonstrations
13 robots
744 tasks
```

任务包括：

```text
pick-and-place
move object
open object
```

此外，相比 ACT 和 Diffusion Policy 关注的一些复杂操作任务，RT-1 的复杂 manipulation task 并不算很多。

### 4.9 RT-1 的代码理解

RT-1 sample code 主要对应三个核心实现点。

第一个是 `RT1` 类：

```text
class RT1(nn.Module)
```

其中可以看到一些关键超参数：

```text
num_layers = 8
layer_size = 128
num_heads = 8
feed_forward_hidden_size = 512
vocab_size = 256
num_image_tokens = 8
num_action_tokens = 11
image_num_features = 512
```

这些参数对应：

- 8 层 Transformer
- 每个动作维度 256 类
- 图像最后压缩成 8 个 token
- 输出 11 个动作 token

第二个是 `ImageTokenizer`：

```text
class ImageTokenizer(nn.Module)
```

它的流程是：

```text
image
-> EfficientNet + FiLM
-> 1x1 convolution
-> FiLM conditioning
-> TokenLearner
-> image tokens
```

第三个是动作 token 化：

```text
tokenize_action
detokenize_action
```

`tokenize_action` 把连续动作压到离散类别：

```text
continuous action -> 0 到 vocab_size - 1 的 token
```

`detokenize_action` 再把模型输出的离散 token 还原成连续动作区间里的数值。

这说明 RT-1 的关键工程技巧不是直接回归动作，而是把动作预测转成 token 分类。

---

## 5. RT-2：把动作接入预训练 VLM

### 5.1 RT-2 的定位

RT-2 是较早把机器人动作集成到预训练 VLM foundation model 中的工作。

它和 RT-1 的关键区别是：

```text
RT-1：从机器人数据训练语言条件策略
RT-2：在预训练 VLM 上加入动作模态并做微调
```

因此 RT-2 更接近严格意义上的 VLA。

### 5.2 RT-2 使用的 VLM Backbone

RT-2 使用的 backbone 包括：

```text
PaLM-E
PaLI-X
```

模型规模包括：

```text
5B PaLI-X
55B PaLI-X
12B PaLM-E
```

这比 RT-1 的 35M 大很多。

### 5.3 RT-2 的动作输出

RT-2 也把动作离散成 256 个 bin。

但它把动作表示成类似语言 token 的格式：

```text
terminate Δpos_x Δpos_y Δpos_z Δrot_x Δrot_y Δrot_z gripper_extension
```

也就是说，机器人动作被写成模型可以生成的 token 序列。

RT-2 只预测一步动作：

```text
one-step action
```

这和 ACT、Diffusion Policy 不同。ACT 和 Diffusion Policy 往往预测 action chunk 或轨迹片段。

### 5.4 RT-2 的训练方式

RT-2 使用 co-finetuning：

```text
robotic action data + VLM data
```

也就是在预训练 VLM 上，同时保留视觉语言能力并加入机器人动作能力。

训练步数大致是：

```text
270k 到 1M gradient steps
```

具体取决于 batch size。

### 5.5 RT-2 的运行速度

推理速度是：

```text
base model: 1-3 Hz
small model: 5 Hz
```

这说明大 VLM 接入机器人动作后，推理速度会成为实际部署的重要瓶颈。

---

## 6. RT 系列扩展

### 6.1 RT-X

RT-X 指的是把 RT-1 和 RT-2 扩展到 Open-X Embodiment 数据集。

Open-X Embodiment 数据集规模：

```text
超过 1M robot trajectories
超过 22 种 robots
```

对应模型：

```text
RT-1-X：35M，JAX 实现
RT-2-X：5B 和 55B
```

RT-X 的重点是跨机器人、跨数据集训练，让模型从更多 embodiment 中学习。

### 6.2 RT-H

RT-H 中的 H 表示 Hierarchy。

核心思想是加入层级结构：

```text
VLM 生成 motion instruction
VLA 根据 motion instruction 输出动作
```

它基于 RT-2 使用一个 VLM 来处理查询和运动指令生成。

可以理解为：

```text
高层语言/视觉推理
-> 中层 motion instruction
-> 低层动作控制
```

### 6.3 AutoRT

AutoRT 是面向真实生活场景部署通用机器人的系统框架。

它关注的不是单个模型结构，而是完整 pipeline：

```text
真实场景部署
任务自动生成
机器人执行
远程操作或模型控制
持续学习循环
```

该系统部署了 20+ Google robots。

这个方向强调：VLA 不只是训练一个模型，还要形成数据收集、执行、反馈、再训练的闭环。

### 6.4 其他 RT 系列

补充要点：

```text
SARA-RT
RT Trajectory
```

这些可以作为后续阅读扩展。

---

## 7. Octo：较小但通用的机器人模型

### 7.1 Octo 的定位

Octo 是一个通用机器人模型，但相比 RT-2 规模更小。

它不是从大型预训练 VLM 直接微调而来，而是基于 Transformer 的机器人策略模型。

需要强调：

```text
Transformer based VLA model
Not from a pretrained foundation VLM model
```

### 7.2 Octo 的编码器

语言编码器：

```text
T5-base，约 111M 参数
```

视觉编码器：

```text
shallow conv + 16x16 patch
```

视觉部分不是预训练的，但作者认为应该使用 vision pretraining data。

### 7.3 Octo 的 Transformer Backbone

Octo 有两个规模：

```text
Octo-Small：类似 ViT-S
Octo-Base：类似 ViT-B
```

参数量：

```text
Octo-small：27M
Octo-base：93M
```

它比 RT-2 小很多，但比 RT-1 设计得更通用。

### 7.4 Octo 的动作输出

Octo 的动作输出是：

```text
Transformer 输出 action token sequence
再由 Diffusion Policy decoder 解码动作
```

关键点：

- 使用 Diffusion Policy decoder
- 支持 action chunking
- action head 可以在新任务微调时替换

这里和 RT-2 有明显差异：

```text
RT-2：把动作当作语言 token 生成，一次预测一步
Octo：Transformer 表示后接 diffusion action decoder，可预测动作序列
```

### 7.5 Octo 的泛化目标

Octo 被设计为适配：

```text
不同机器人
不同视觉输入
不同任务
不同部署环境
```

它支持：

```text
zero-shot
finetuning
```

### 7.6 Octo 的训练成本

训练信息：

```text
ViT-B 训练 300k steps
batch size 2048
TPU v4-128 pod
约 14 小时
```

微调信息：

```text
单张 NVIDIA A5000 24GB
约 5 小时
```

这说明 Octo 相比大型 VLM 路线更轻量，更适合研究和迁移。

---

## 8. OpenVLA：开源 VLM + Action 模型

### 8.1 OpenVLA 的定位

OpenVLA 是开源的 VLA 模型。

它的路线更接近：

```text
VLM + Action
```

也就是把预训练视觉语言模型改造成机器人动作预测模型。

### 8.2 Prismatic VLM Backbone

OpenVLA 使用 Prismatic-7B VLM。

组成包括：

```text
600M visual encoder
small 2-layer MLP projector
7B Llama 2 language model backbone
```

视觉编码器是两部分：

```text
SigLIP
DINOv2
```

这些组件本身是预训练模型。需要注意，SigLIP、DINOv2 和 Llama 2 并没有完全公开训练数据细节。

Prismatic VLM 在这些组件上用 LLaVA 1.5 data mixture 进行 fine-tuning。

### 8.3 OpenVLA 如何把动作变成 VLM 任务

OpenVLA 把动作预测表述成视觉语言任务：

```text
Input:
  observation image
  natural language task instruction

Output:
  string of predicted robot actions
```

换句话说：

```text
机器人动作 -> 被编码成语言模型可以输出的 token
```

### 8.4 OpenVLA 的动作离散化

OpenVLA 的动作解码方式类似 RT-2。

每个动作维度单独离散成：

```text
256 bins
```

bin 的范围不是简单取最大最小值，而是使用训练数据中的：

```text
1st quantile 到 99th quantile
```

这样可以减少极端异常值对离散区间的影响。

OpenVLA 还做了一个工程处理：

```text
覆盖 Llama tokenizer 词表中最少使用的 256 个 token
```

用这些 token 表示动作 bin。

### 8.5 OpenVLA 的训练方式

OpenVLA 使用：

```text
Open-X Embodiment dataset
```

训练方式是：

```text
action finetune
```

需要特别指出：

```text
no co-finetuning with VLM data
```

这和 RT-2 不同：

```text
RT-2：robot action data + VLM data co-finetuning
OpenVLA：主要用机器人动作数据做 action finetune
```

训练成本：

```text
21500 A100 training hours
batch size 1024
```

### 8.6 OpenVLA 的微调

OpenVLA 支持 LoRA 微调。

关键信息：

```text
使用 LoRA 时，可以在单张 A100 上 10-15 小时内微调新任务
```

代码页中展示的 `finetune.py` 配置包括：

```text
vla_path = "openvla/openvla-7b"
dataset_name = "droid_wipe"
batch_size = 16
max_steps = 200000
save_steps = 5000
learning_rate = 5e-4
image_aug = True
use_lora = True
lora_rank = 32
lora_dropout = 0.0
use_quantization = False
```

这些参数说明，OpenVLA 的新任务适配主要靠：

```text
加载 openvla-7b
指定新任务数据集
用 LoRA 更新少量参数
保存 adapter 或合并权重
```

### 8.7 OpenVLA 的推理速度

推理速度：

```text
bfloat16
单张 NVIDIA RTX 4090
约 6 Hz
```

这比 RT-2 的 base model 快一些，但仍然不是传统控制器那种高频控制。

### 8.8 OpenVLA 的 VLM Backbone 对比

OpenVLA 除了 Prismatic，也测试过：

```text
IDEFICS-1
LLaVA
```

观察结论：

- 单物体场景中，LLaVA 和 IDEFICS-1 表现接近
- 多物体场景中，LLaVA 的语言 grounding 更强

这说明 VLA 的能力不只取决于机器人数据，也强烈依赖底层 VLM 的视觉语言理解能力。

### 8.9 OpenVLA 代码结构

OpenVLA repo 的重点包括：

```text
prismatic/conf/vla.py
vla-scripts/finetune.py
```

`vla.py` 里定义 VLA 配置，例如：

```text
base_vlm
freeze_vision_backbone
freeze_llm_backbone
data_mix
shuffle_buffer_size
epochs / max_steps
global_batch_size
learning_rate
train_strategy
mixed precision
gradient checkpointing
```

`finetune.py` 里定义新任务微调配置，例如数据路径、训练步数、LoRA 参数、日志配置等。

如果读 OpenVLA 代码，建议顺序是：

```text
1. 看配置：prismatic/conf/vla.py
2. 看模型：prismatic/models/vlas
3. 看数据和 action tokenizer：prismatic/vla
4. 看训练脚本：vla-scripts/finetune.py
5. 看推理脚本：vla-scripts/deploy.py
```

---

## 9. 四个模型的对比

| 模型 | 核心路线 | Backbone | 动作表示 | 数据 | 特点 |
|---|---|---|---|---|---|
| RT-1 | 语言条件模仿学习 | EfficientNet-B3 + USE + Transformer | 11 维动作，每维 256 bins | 30k demos, 13 robots, 744 tasks | 小模型，约 35M，更像 language conditioned policy |
| RT-2 | 预训练 VLM + action token | PaLM-E / PaLI-X | 类语言 token 的一步动作 | RT-1 robot data + VLM data | 早期把动作接入大型 VLM 的代表 |
| Octo | 通用机器人 Transformer | T5 + ViT-like Transformer | diffusion decoder + action chunking | 多机器人数据 | 较小，开源，支持 zero-shot 和 finetuning |
| OpenVLA | 开源 VLM + Action | Prismatic-7B | 每维 256 bins，覆盖 Llama 低频 token | Open-X Embodiment | 开源 7B VLA，支持 LoRA 微调 |

---

## 10. 动作表示：VLA 设计的关键问题

VLA 中动作表示非常重要，因为模型最终必须控制真实机器人。

常见方式有三类。

### 10.1 离散动作 token

代表：

```text
RT-1
RT-2
OpenVLA
```

做法：

```text
连续动作维度 -> 分桶 -> token 分类
```

优点：

- 可以复用语言模型的 token 生成机制
- 训练可用 cross entropy
- 工程上和 VLM 兼容

缺点：

- 动作精度受 bin 数限制
- 高频连续控制不自然
- 复杂动作轨迹表达能力有限

### 10.2 Policy Head

做法：

```text
VLM / Transformer hidden state -> policy head -> continuous action
```

优点：

- 更接近传统机器人连续控制
- 可以直接输出连续动作
- 适合和历史状态、动作序列结合

缺点：

- 和语言模型 token 生成框架结合不如离散 token 简单

### 10.3 Diffusion Action Decoder

代表：

```text
Octo
Diffusion Policy
```

做法：

```text
conditioned representation -> diffusion decoder -> action chunk
```

优点：

- 适合生成多步动作序列
- 对多模态动作分布更友好
- 在机器人 manipulation 中表现强

缺点：

- 推理可能更慢
- 系统复杂度更高

---

## 11. VLA 相关工作的历史脉络

VLA 可以放在多个技术分支的交汇处理解。

计算机视觉方向：

```text
ResNet -> ViT -> SAM
```

自然语言处理方向：

```text
GRU -> Transformer -> BERT -> ChatGPT
```

强化学习方向：

```text
DQN -> AlphaGo -> PPO -> Dactyl -> Decision Transformer
```

视觉语言方向：

```text
ViLBERT -> CLIP -> LLaVA
```

机器人 VLA / policy 方向：

```text
CLIPort
BC-Z
RT-1
VIMA
SayCan
PaLM-E
RT-2
OpenVLA
```

这说明 VLA 不是单独冒出来的技术，而是由视觉、语言、强化学习、模仿学习、机器人控制共同推动的结果。

---

## 12. VLA 结构分类

总结了几类 VLA / RoboVLM 结构：

### 12.1 One-Step Continuous-Action Models

模型一次输出一步连续动作。

```text
VLM -> action decoder -> one-step continuous action
```

### 12.2 One-Step Discrete-Action Models

模型一次输出一步离散动作 token。

```text
VLM -> discrete action tokens -> detokenizer -> action
```

RT-2 和 OpenVLA 更接近这一类。

### 12.3 Interleaved Continuous-Action Models

把视觉 token、文本 token、历史 token、动作相关 token 交错建模。

```text
text token + vision token + history token + action token
```

这类方法更强调序列历史直接进入 VLM。

### 12.4 Policy-Head Continuous-Action Models

VLM 负责表征，policy head 负责输出连续动作。

```text
VLM representation -> policy head -> continuous action
```

一个常见总结是：

```text
Interleaved method deals with sequence in VLM,
while policy head deals with sequence in itself.
```

意思是：

- interleaved 方法把历史和序列建模交给 VLM
- policy head 方法把动作序列建模交给策略头或动作解码器

---

## 13. VLA Design 的研究发现

关于 VLA 设计，有几个重要问题和发现。

### 13.1 为什么选择 VLA

研究问题：

```text
VLAs 是否适合构建 generalist robot policies？
```

结论：

```text
VLA 是通向通用机器人策略的 promising path。
```

也就是说，它不是完美答案，但方向很有潜力。

### 13.2 哪种 Backbone 更合适

研究问题：

```text
哪类 VLM 更适合构建 VLA？
```

结论：

```text
在大规模视觉语言数据上充分预训练的 VLM 对 VLA 有帮助。
```

这解释了为什么 RT-2、OpenVLA 都重视 foundation VLM。

### 13.3 如何表示动作和历史

研究问题：

```text
什么 VLA structure 表现最好？
不同 formulation 如何影响泛化和数据效率？
```

主要结论包括：

```text
continuous action space + policy head + history integration
```

是很强的结构选择。

另一个结论是：

```text
带 separate policy head 的 backbone 在泛化和数据效率上表现更好。
```

这说明“直接把动作全部当作语言 token”不一定总是最优。对于真实机器人，连续动作和历史建模仍然很重要。

### 13.4 什么时候使用额外数据

研究问题：

```text
大规模 cross-embodiment datasets 对 VLA 有什么帮助？
```

结论：

```text
额外的 in-domain data 有帮助
post-training 可以进一步提升整体和 few-shot 表现
```

这和 Open-X Embodiment、RT-X、OpenVLA 的路线一致。

---

## 14. 学习本节课时最容易混淆的点

### 14.1 RT-1 是不是 VLA

广义上可以算，因为它使用：

```text
vision + language -> action
```

但严格来说，它更像：

```text
language conditioned imitation learning policy
```

因为它不是从大型预训练 VLM 微调而来。

### 14.2 RT-2 和 OpenVLA 的共同点

共同点：

```text
都把动作预测变成 VLM 可以处理的 token 生成问题
都使用动作离散化
都接近 pretrained VLM + robot action finetuning 路线
```

不同点：

```text
RT-2 使用 co-finetuning with VLM data and robot data
OpenVLA 主要 action finetune with Open-X data
OpenVLA 是开源模型和代码体系
```

### 14.3 Octo 为什么重要

Octo 说明 VLA 不一定必须是 7B、55B 这种大模型。

它选择较小模型，并通过：

```text
Transformer backbone
Diffusion policy decoder
Action chunking
Cross-robot data
```

来获得通用能力。

### 14.4 离散动作 token 是否一定最好

不一定。

离散 token 方便接入语言模型，但真实机器人动作是连续的。相关设计发现更支持：

```text
continuous action space + policy head + history
```

所以未来 VLA 可能会更多采用混合结构：

```text
VLM 负责理解
policy head / diffusion decoder 负责动作
```

---

## 15. 本节课的知识地图

可以把本节课整理成如下结构：

```text
VLA
├── 背景
│   ├── LLM / VLM
│   ├── Transformer / tokenization
│   └── language conditioned policy
│
├── RT 系列
│   ├── RT-1：小模型，语言条件模仿学习
│   ├── RT-2：预训练 VLM + action token
│   ├── RT-X：Open-X Embodiment 多机器人数据
│   ├── RT-H：层级 motion instruction
│   └── AutoRT：真实部署和持续学习系统
│
├── Octo
│   ├── T5 language encoder
│   ├── ViT-like Transformer
│   ├── diffusion action decoder
│   └── zero-shot / finetuning
│
├── OpenVLA
│   ├── Prismatic-7B
│   ├── SigLIP + DINOv2 + Llama 2
│   ├── action tokenization
│   ├── Open-X action finetune
│   └── LoRA 新任务微调
│
└── 设计问题
    ├── 动作离散化 vs 连续控制
    ├── action token vs policy head
    ├── one-step action vs action chunk
    ├── small model vs foundation VLM
    └── single robot data vs cross-embodiment data
```

