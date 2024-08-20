[TOC]



## llm

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