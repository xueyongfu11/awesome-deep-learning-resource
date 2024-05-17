<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [llm](#llm)
- [Training/Inference Framework](#traininginference-framework)
- [leaderboard](#leaderboard)
- [LLM和知识图谱](#llm%E5%92%8C%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# llm

- [从头预训练大模型实践经验](https://mp.weixin.qq.com/s/LvOsS25mSRFWAHYv9IqTyQ)

- https://github.com/yanqiangmiffy/how-to-train-tokenizer

- [大模型的涌现能力介绍](https://mp.weixin.qq.com/s/Jymq2ho3VQx7-o6y-JHxeg)

- https://github.com/huggingface/datatrove
  - 大规模处理、过滤和删除重复文本

# Training/Inference Framework

- https://github.com/hpcaitech/ColossalAI


# leaderboard

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


# LLM和知识图谱
- [万字长文讲述大模型与知识图谱的关系](https://zhuanlan.zhihu.com/p/626433991)
- [LLM 时代的金融知识图谱实践](https://zhuanlan.zhihu.com/p/623104680)
  - 比较基础

# 大模型幻觉

- https://eecs.berkeley.edu/research/colloquium/230419-2/
  - Reinforcement Learning from Human Feedback: Progress and Challenges

- R-Tuning: Instructing Large Language Models to Say ‘I Don’t Know’
  - 2023.11，NAACL 2024
  - 提出了一种解决大模型幻觉现象的方法，即通过训练的方式让大模型拒答或者表示不确定性
  - 拒答数据生成：首先用native模型生成指令集的预测结果，与ground truth对比，确定哪些指令集是native模型可回答的，哪些是不能回答的。使用一个prompt将指令集组合起来，并根据指令集的分类结果，以“I am sure” or “I am not sure”作为回复
  - 使用拒答数据来微调模型，注意，只计算answer和“I am sure” or “I am not sure”的loss
  - 推理时，先用native模型生成answer，然后用prompt组合起来，送入微调好的模型，得到模型的置信输出
  - 不太适合用在在线对话模型上