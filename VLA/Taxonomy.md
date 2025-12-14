[TOC]



# A：**Control & Reasoning Paradigm（控制范式）**

> **谁在“想”，谁在“做”？**

## A1. Policy-Centric VLA（端到端）

* RT-1 / RT-2
* BC-Z
* PerAct
* OpenVLA

📌 特点：
端到端 imitation / RL，弱组合泛化

## A2. Planner–Executor VLA（任务分解）

* SayCan
* Code-as-Policies
* ProgPrompt
* Inner Monologue

📌 特点：
LLM 负责 high-level reasoning，低层 policy 执行

## A3. Agentic VLA（长期交互）

* Voyager
* Eureka
* AutoGPT-style embodied agents

📌 核心区分点：
**在线 skill discovery / memory / curriculum**

> Voyager ≠ SayCan 的关键在于「能否自我进化」

---

# B：**World Modeling Assumption（世界建模）**

> **模型是否显式学习环境动力学？**

## B1. No World Model（Reactive）

* RT-1
* BC-Z
* SayCan（本质）

---

## B2. Explicit World Model（显式动力学）

* PlaNet
* Dreamer / DreamerV3
* TD-MPC / TD-MPC2
* MuZero（VLA 扩展）

📌 强 planning / imagination，但难 scale 到复杂视觉

---

## B3. Implicit World Model（隐式）

* **π0 / π0.5**
* Gato
* Decision Transformer
* VIMA
* RT-2（部分）

📌 **Transformer 本身即世界模型**（sequence = trajectory）

---

# C：**Representation Form（表示范式）**

> **Action / State / Language 如何进入模型？**

## C1. Symbolic / Programmatic

* SayCan
* Code-as-Policies
* Behavior Trees + LLM

---

## C2. Latent Continuous

**这是当前工业和学术的主流**

* PerAct（latent voxel action）
* CLIPort
* R3M-based policies
* VLM + latent policy heads

📌 特点：

* 非 token
* Action 是向量 / pose / heatmap
* 泛化依赖 representation learning，而非序列建模

---

## C3. Tokenized Generalist Representation

> **Vision / Language / Action / State → tokens**

* **π0 / π0.5**
* **Gato**
* Decision Transformer
* VIMA（token action）
* (未来) Qwen-Embodied / Gemini Robotics

📌 这是你 V4 的正确抽象，但它是 **表示范式，不是单独一代**

