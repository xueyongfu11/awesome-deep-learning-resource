[TOC]



## llm

- https://github.com/OpenBMB/BMPrinciples
  - A collection of phenomenons observed during the scaling of big foundation models, which may be developed into consensus, principles, or laws in the future

- https://github.com/modelscope/data-juicer
  - A one-stop data processing system to make data higher-quality, juicier, and more digestible for (multimodal) LLMs

  - 为大模型提供更高质量、更丰富、更易”消化“的数据

- https://github.com/wdndev/llm_interview_note
  - 主要记录大语言大模型（LLMs） 算法（应用）工程师相关的知识及面试题

- https://github.com/open-compass/opencompass
  - OpenCompass is an LLM evaluation platform, supporting a wide range of models

- https://github.com/modelscope/evalscope
  - A streamlined and customizable framework for efficient large model evaluation and performance benchmarking

- [从头预训练大模型实践经验](https://mp.weixin.qq.com/s/LvOsS25mSRFWAHYv9IqTyQ)

- https://github.com/yanqiangmiffy/how-to-train-tokenizer

- [大模型的涌现能力介绍](https://mp.weixin.qq.com/s/Jymq2ho3VQx7-o6y-JHxeg)

- https://github.com/huggingface/datatrove
  - 大规模处理、过滤和删除重复文本
- An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning


## leaderboard

- https://flageval.baai.ac.cn/#/trending

- https://opencompass.org.cn/leaderboard-llm

- https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

- https://cluebenchmarks.com/superclue.html

- https://cevalbenchmark.com/static/leaderboard.html
  - 中文大模型评估排行榜

- http://115.182.62.166:18000/public
    - 大模型安全性评估

- https://github.com/google/BIG-bench
- https://github.com/suzgunmirac/BIG-Bench-Hard

- LongBench
  - eval datasets: https://kaiokendev.github.io/context
  - https://github.com/THUDM/LongBench#leaderboard
  - 清华 中英文大模型评估
- 偏好数据标注工具
  - https://github.com/SupritYoung/RLHF-Label-Tool



## MoE

- https://github.com/TUDB-Labs/MoE-PEFT
- MIXLORA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts
  - 2024.04
  - 提出了MixLoRA，不同于其它方法，MixLoRA的每个专家都由原始模型层结构和LoRA组成，与传统的MoE模型更加相似。
  - MixLoRA使用top-k router策略，训练时加入了expert balance loss，MixLoRA也探索了对self-attention使用MoE-LoRA，结果是进一步提升了整体效果。整体结构与PESC非常相似。
  - ![](../assets/mixlora.png)

- Higher Layers Need More LoRA Experts
  - 2024.02
  - 论文验证了模型的较低层的expert之间相似性更大，存在参数冗余
  - 论文提出了MoLA，从对比实验中可以得出，固定expert的总数，模型较高层相比较低层设置更多的expert，效果表现最好
  
- Parameter-Efficient Sparsity Crafting from Dense to Mixture-of-Experts for  Instruction Tuning on General Tasks
  - 2024.01，PESC
  
  - 提出了一种MoE-LoRA架构模型PESC，该方法使用了串行结构的adapter-MoE，并在损失中添加了expert balance loss
  
- LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via MoE-Style Plugin
  - 2023.12
  - 实验发现在SFT阶段显著增加指令数据量会损害LLMs中的世界知识
  - 引入了LoRAMoE框架，通过集成LoRAs和路由器来增强模型处理下游任务的能力，同时减轻世界知识遗忘。具体是提出局部平衡约束，以鼓励专家之间的合作，并在不同类型的任务上实现专家的专业化。
- When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications
  - 2023.10
  - 主要为了解决任务多样性问题和高昂的调优成本，提出的方法主要应用的医疗领域
  - MOELoRA：结合了MoE和LoRA，核心是门控的设计，具体是不同层共享门控，构建任务类型的embedding，通过一个线性层输出专家权重，这里可以使用稀疏和非稀疏的加权方式。
  - 推理时，可以恢复出每个任务类型的微调权重，并将微调权重与原始模型合并，避免的MoE带来的额外推理成本
- Pushing Mixture of Experts to the Limit:  Extremely Parameter Efficient MoE for  Instruction Tuning
  - 2023.09
  - 提出了结合peft和MoE的两种方法MoV和MoLoRA
  - 结论1：上一层网络输出x作为门控网络的输入相比sentence embedding效果更好
  - 结论2：专家数量达到10时，效果开始区域稳定
  - 结论3：soft merging策略相比离散路由策略效果更好


## LLM和知识图谱

- [万字长文讲述大模型与知识图谱的关系](https://zhuanlan.zhihu.com/p/626433991)
- [LLM 时代的金融知识图谱实践](https://zhuanlan.zhihu.com/p/623104680)
  - 比较基础

## 大模型幻觉

- https://eecs.berkeley.edu/research/colloquium/230419-2/
  - Reinforcement Learning from Human Feedback: Progress and Challenges

- R-Tuning: Instructing Large Language Models to Say ‘I Don’t Know’
  - 2023.11，NAACL 2024
  - 提出了一种解决大模型幻觉现象的方法，即通过训练的方式让大模型拒答或者表示不确定性
  - 拒答数据生成：首先用native模型生成指令集的预测结果，与ground truth对比，确定哪些指令集是native模型可回答的，哪些是不能回答的。使用一个prompt将指令集组合起来，并根据指令集的分类结果，以“I am sure” or “I am not sure”作为回复
  - 使用拒答数据来微调模型，注意，只计算answer和“I am sure” or “I am not sure”的loss
  - 推理时，先用native模型生成answer，然后用prompt组合起来，送入微调好的模型，得到模型的置信输出
  - 不太适合用在在线对话模型上

## RolePlay

- RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models

  - https://github.com/InteractiveNLP-Team/RoleLLM-public
- https://github.com/LC1332/Chat-Haruhi-Suzumiya

  - trick: 动作、表情、心理以及对应的场景描写，通常中文放在（）里
- [角色扮演大模型技术分享](https://zhuanlan.zhihu.com/p/685823865)
- [论文阅读_训练大模型用于角色扮演](https://zhuanlan.zhihu.com/p/678889518)

- leaderboard
  - https://github.com/boson-ai/RPBench-Auto
  - https://github.com/morecry/CharacterEval
  - https://github.com/InteractiveNLP-Team/RoleLLM-public
  - https://github.com/CLUEbenchmark/SuperCLUE-Role
- Open-source LLM
  - https://huggingface.co/bosonai/Higgs-Llama-3-70B
  - https://github.com/thu-coai/CharacterGLM-6B

## 大模型遗忘

### 主流方法

- EWC（Elastic Weight Consolidation），基于正则化的方法
- 经验回放
  - LAMOL，生成式回放，让模型生成虚拟的旧任务数据
    - 经验回放可以防止大模型遗忘，但是很多时候无法获取旧数据，LAMOL方法是用大模型本身来生成数据，作为“旧数据”，然后再把“旧数据”和新任务数据混合训练。
  - Self-Synthesize Rehearsal (Magpie)：让基础模型自问自答，生成数据，然后将这些“自己合成”的数据用于后续训练。
  - 改写 (Paraphrase)：不用人工标注的答案，而是让基础模型将标准答案“换句话说”，再用改写后的答案去训练。因为改写后的句子更符合模型自身的语言风格，学习起来更不容易引起冲突。
  - 自输出 (Self-Output)：将新任务的问题输入基础模型，让它直接生成答案，通过某种机制验证答案是否正确，如果答案正确，就用模型自己的输出作为标准答案来训练它自己，它本质上是在强化模型已有的、正确的知识路径。
  - 过滤困难样本：一篇研究发现，人工标注的答案中，有些词元（token）对于基础模型来说是极难生成的（即低机率）。如果在训练时，直接忽略掉这些对模型来说过于困难的学习目标（不计算这些 token 的损失），反而能让模型学得更好，同时更好地保留原有能力。
- Task Vector，模型合并，通过向量叠加或抵消来合并新旧任务参数
- Gradient Projection，梯度投影，将新任务梯度投影到与旧任务冲突最小的方向上

### Paper

- Retaining By Doing: The Role of On-policy Data in Mitigating Forgetting
  - 2025.10
- A Comprehensive Survey of Continual Learning:  Theory, Method and Application
  - 2024.08, survey

### Blog

- [缓解LLM fine-tune模型遗忘](https://zhuanlan.zhihu.com/p/700156271)
- [ICLR 2025 | 大模型“遗忘”竟是错觉？首次揭示 LLM 训练中的“虚假遗忘”](https://zhuanlan.zhihu.com/p/23021161842)
  - 通过冻结模型底层若干层（如6层或者10层，以及输入嵌入层），可以显著防止对旧任务对齐的过度破坏
  - 但是冻结层数过多，会降低对新任务的适应能力，因此需要进行权衡。
- [UIUC、Amazon团队最新研究指出SFT灾难性遗忘问题或被误解](https://mp.weixin.qq.com/s/29pDMmX821Z10yFPFZ-5Ng)
  - 只需是要较小的学习率，即可缓解SFT的灾难性遗忘问题
  - 发现造成灾难性遗忘的问题主要是训练数据中的难token，提出了TALR方法，即token预测概率低自适应的分配更小的loss权重。
