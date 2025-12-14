[TOC]



# 视觉与多模态基础模型分类体系

- ✅ 覆盖 **CV → VLM → Video → 3D → World Model → Agent**
- ✅ 每一类都有**清晰研究目标**
- ✅ 每个子类都包含**该方向最重要的方法**

# A. 表示学习（Representation Learning）

> **目标**：学习**通用、可迁移、任务无关**的视觉或时空表征
> （不直接输出语义、不生成内容）

------

## A1. 监督 / 半监督视觉表示

**代表方法**

- ResNet / ConvNeXt
- ViT / DeiT
- Scaling ViT
- EVA / EVA-02
- InternImage
- MaxViT

**特征**

- 强 backbone
- 依赖大规模标注或弱监督
- 作为几乎所有下游任务的 encoder

------

## A2. 自监督视觉表示（Image-level）

**代表方法**

- MAE / iMAE
- SimCLR / MoCo
- BYOL
- DINO / DINOv2
- iBOT

**特征**

- 不依赖人工标签
- 学习语义 + 结构感知 embedding
- 是 **SAM / OVD / Dense Task 的能力来源**

------

## A3. 时空 / 视频表示学习（Video Representation）

**代表方法**

- VideoMAE / VideoMAE-v2
- Masked Video Transformer
- TimeSformer
- InternVideo
- VideoCLIP

**特征**

- 学习时间一致性与动态结构
- 是 **视频理解、世界模型、行为建模** 的基础

------

## A4. 密集 / 局部表征学习（Dense Representation）

**代表方法**

- DenseCL
- PixPro
- Dense DINO / iBOT-Dense
- Self-Training ViT

**特征**

- 面向 detection / segmentation / tracking
- 强调局部一致性与像素级结构

# B. 结构理解（Parsing / Perception）

> **目标**：将视觉世界**结构化解析**
> （是什么、在哪里、边界、几何关系）

------

## B1. 经典 2D 结构理解任务

**任务 / 方法**

- Object Detection（Faster R-CNN, DETR）
- Semantic / Instance Segmentation
- Pose Estimation
- Tracking

📌 非 foundation，但仍是重要应用层

------

## B2. Foundation-level 结构理解（Promptable Parsing）

**代表方法**

- SAM
- SAM2 / Video-SAM
- MedSAM
- Segment Anything in Videos

**特征**

- Promptable
- Task-agnostic
- Zero-shot / Few-shot

------

## B3. 3D / 几何 / 场景结构理解（升格为一级）

**代表方法**

- NeRF / Instant-NGP
- 3D Gaussian Splatting
- Occupancy Network
- OpenScene / OpenOccupancy
- SAM-3D / 3D-SAM

**特征**

- 建模空间、深度、几何一致性
- 是 **机器人 / 世界模型 / 仿真** 的关键

------

## B4. Open-Vocabulary & Attention-based Parsing

**代表方法**

- DINO Attention Probing
- Open-Vocabulary Detection (OVD)
- ViT Attention Analysis

**特征**

- 介于 A ↔ B
- 提供结构感知能力来源

# C. 生成建模（Generation）

> **目标**：建模视觉数据分布并合成内容
> （从无到有生成视觉世界）

------

## C1. 基础生成模型

- GAN（StyleGAN）
- VAE
- Diffusion Models

------

## C2. 文本到图像（Text-to-Image）

**代表方法**

- Stable Diffusion
- Imagen
- DALL·E
- PixArt
- DiT

**特征**

- p(image | text)
- 强视觉先验
- 语义一致性不完全保证

------

## C3. 视频生成

**代表方法**

- Video Diffusion Models
- Latent Video Diffusion
- VideoCrafter
- Phenaki
- Latte
- Open-Sora（工程）

------

## C4. 可控 / 条件 / 编辑生成（关键补充）

**代表方法**

- ControlNet
- InstructPix2Pix
- Layout-to-Image
- Image / Video Editing Diffusion

**特征**

- 生成 → 工具化 的分水岭
- 强依赖 B 类结构信息

# D. 多模态对齐（Vision–Language）

> **目标**：对齐视觉与语言语义空间
> （看懂、描述、推理、交互）

------

## D1. 对比式视觉-语言对齐

- CLIP
- ALIGN
- SigLIP

📌 Open-vocabulary 能力的根基

------

## D2. Encoder–Decoder VLM

- BLIP / BLIP-2
- OFA
- Flamingo

------

## D3. LLM-centric VLM（主流）

- LLaVA
- Qwen-VL / Qwen-VL2.5
- InternVL
- Kimi-VL
- MiniGPT-4

**特征**

- LLM 为核心
- Vision encoder 为感知模块
- 支持 reasoning / agent

------

## D4. 视频-语言 / 行为-语言理解

**代表方法**

- Video-LLaMA
- InternVideo-Chat
- Ego4D Models
- V-JEPA

**特征**

- 理解事件、动作、因果
- 直接通向 Agent 与 World Model

# E. 世界建模（World Model / Simulation）

> **目标**：建模世界的**时空演化、因果结构与可交互性**

------

## E1. 视频世界模型

- Sora
- Genie
- DeepMind Video World Models

------

## E2. 强化学习世界模型

- PlaNet
- Dreamer
- DreamerV3

------

## E3. 生成 + 交互式世界

- Neural Scene Simulation
- Game Engine + Diffusion

------

## E4. Planning / Control / Policy Interface

- Latent Action World Models
- Planning via World Model
- World Model + Agent Loop

# F. 方法论 / 评测 / Scaling

> 非模型，但决定 foundation 是否成立

- Linear probing / Zero-shot eval
- Open-vocabulary benchmarks
- Scaling laws（ViT / Diffusion / VLM）
- Data curation（DINOv2, InternImage）

------

# 跨类方法

| 方法       | 主类 | 次类  |
| ---------- | ---- | ----- |
| DINOv2     | A    | B     |
| CLIP       | D    | A     |
| SAM        | B    | A     |
| NeRF       | B    | C     |
| ControlNet | C    | B     |
| DiT        | C    | A     |
| VideoMAE   | A    | E     |
| Sora       | E    | C + D |

