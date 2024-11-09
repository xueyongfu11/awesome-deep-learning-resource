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

## 大模型个性化

### 个性化记忆

- langchain
  - https://github.com/langchain-ai/langchain
  - 支持检索和摘要两种记忆方式

- mem0
  - https://github.com/mem0ai/mem0
  - 通过LLM+制定的metadata，抽取记忆信息，重点是定制一个合适的prompt来抽取有效信息
  - 相关记忆信息通过向量化存储，因此可以支持记忆信息检索
  - 记忆支持更新
- MemGPT
  - https://github.com/cpacker/MemGPT

### paper

- Understanding the Role of User Profile in the Personalization of Large Language Models
  - 2024.06，
  - 验证了user profile对LLM个性化的影响是个性化信息而非语义内容信息
  - 文章中的profile是历史对话信息，不包含其他信息
  - 检索到的user profile放在开始位置对效果有更大的影响
- HYDRA: Model Factorization Framework for Black-Box LLM Personalization
  - 2024.06
  - 方法：不仅用到了历史用户行为模式数据，而且也用到了所有用户共享通用知识
- Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation
  - 2024.04
  - 基于检索的方法
- A Survey on the Memory Mechanism of Large Language Model based Agents
  - 2024.04
- Personalized Large Language Models
  - 2024.02
- Integrating summarization and retrieval for enhanced personalization via large language models
  - 2023.10，Amazon Alexa AI
  - 将对话检索和对话摘要结合在了一起
  - employ instruction-tuned LLMs to generate abstract summaries of user history data, integrating summarization for enhanced personalization
- Once: Boosting content-based recommendation with both open-and closed-source large language models
  - creates user profiles by summarizing topics and regions of interest from their browsing history
- When Large Language Models Meet Personalization: Perspectives of Challenges and Opportunities
  - 2023.07，survey
  - 11.3节提及了长对话历史建模的三种方法：检索、摘要、使用层次模型或者记忆增强模型
- LaMP: When Large Language Models Meet Personalization
  - 2023.04，google
  - 提出了个性化语言模型评估的benchmark，以及用户的多条profile相关数据集（对话历史数据）