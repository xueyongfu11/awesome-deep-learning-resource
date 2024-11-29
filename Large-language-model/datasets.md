[TOC]




# LLM Datasets

## 各类数据集收集repo
- https://github.com/yaodongC/awesome-instruction-dataset
  - 多模态数据集、文本指令集、RLHF数据集

- https://github.com/FreedomIntelligence/InstructionZoo

- 


## Preference Data

- openbmb/UltraFeedback
  - https://huggingface.co/datasets/openbmb/UltraFeedback
  - 使用不同的模型生成多回复，然后使用细粒度的偏好标注指令进行gpt-4自动标注
- HuggingFaceH4/ultrafeedback_binarized

  - https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
  - UltraFeedback数据处理好的版本
- Anthropic/hh-rlhf
  - https://huggingface.co/datasets/Anthropic/hh-rlhf
  - 人工标注，可能含有噪声
- fnlp/hh-rlhf-strength-cleaned
  - https://huggingface.co/datasets/fnlp/hh-rlhf-strength-cleaned
- stanfordnlp/SHP
  - https://huggingface.co/datasets/stanfordnlp/SHP
- orca_dpo_pairs
  - https://huggingface.co/datasets/Intel/orca_dpo_pairs
  - 无数据集相关介绍
- HannahRoseKirk/prism-alignment
  - https://huggingface.co/datasets/HannahRoseKirk/prism-alignment
- orca_dpo_pairs
  - https://huggingface.co/datasets/Intel/orca_dpo_pairs
  
- summarize_from_feedback
  - 摘要偏好数据，from openai
  - https://huggingface.co/datasets/openai/summarize_from_feedback


- PKU-SafeRLHF
  - 安全对齐数据集
  - https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
- CValues
  - https://huggingface.co/datasets/xiaodongguaAIGC/CValues
  - 中文safe偏好数据
- zhihu_rlhf_3k
  - 中文偏好数据
  - https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k
- oasst1_pairwise_rlhf_reward
  - https://huggingface.co/datasets/tasksource/oasst1_pairwise_rlhf_reward

- openai-prm800k-stepwise-critic
  - 过程监督数据集
  - https://huggingface.co/datasets/Birchlabs/openai-prm800k-stepwise-critic
  - 多步推理：数据集中每个示例通常包含需要逐步分解的复杂任务或问题；通过将问题分解为子问题，模型可以逐步构建最终的答案
  - 批判性评估：包含对不同推理步骤的“评估”机制；训练过程中，模型可以学习在每个推理步骤中如何自我评估和调整

- lmsys/chatbot_arena_conversations
  - 多轮偏好数据，基于从两个模型采样出来的多轮对话，对话上下文可能不同
  - https://huggingface.co/datasets/lmsys/chatbot_arena_conversations

- lmsys/mt_bench_human_judgments
  - https://huggingface.co/datasets/lmsys/mt_bench_human_judgments
  - 多轮偏好数据，基于从两个模型采样出来的多轮对话，对话上下文可能不同，使用大模型/人工判断哪一个多轮对话效果更好

## chinese dataset

- https://huggingface.co/datasets/Suprit/CMtMedQA
  - 医疗多轮对话数据集

- alpaca_chinese_dataset
  - 52K，将Alpaca数据集进行机器翻译+人工校验，并补充一些对话数据
  - https://github.com/hikariming/alpaca_chinese_dataset

- HC3-Chinese
  - 12.9k，human-chatgpt对比数据集
  - https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese

- firefly-train-1.1M
  - 1.65M，基于23个常见的中文数据集，人工编写指令模板
  - https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M

- COIG
  - v1
    - 178k， 北京智源人工智能研究院发布的中文指令数据集，包括翻译指令，考试指令，人类价值对齐指令，反事实多轮对话指令，leecode指令等
    - https://huggingface.co/datasets/BAAI/COIG
  - v2
    - https://huggingface.co/datasets/BAAI/COIG-PC
    - 312M
    - 轻量级数据集，每个任务采样200条
      https://huggingface.co/datasets/BAAI/COIG-PC-Lite

- BBT-FinCUGE-Applications
  - 通用金融语料，以及金融相关的qa、分类、ner、re等数据
  - https://github.com/ssymmetry/BBT-FinCUGE-Applications

- pCLUE
  - 1.2M，通过原有的NLP任务数据集，结合特定的prompt模板生成
  - https://github.com/CLUEbenchmark/pCLUE

- OpenLabel-Chinese Conversations Dataset (OL-CC) 
  - 包含 10k+ “指令-回答”数据对和 1.6k+ 人工指令数据。指令类型丰富，包括问答任务、文本写作、文本抽取、编辑改写、分类选择、头脑风暴、 闲聊对话、逻辑&数学等任务

- BelleGroup/multiturn_chat_0.8M
  - 用户与助手的多轮对话，由ChatGPT产生
  - https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M

- BelleGroup/school_math_0.25M
  - 中文数学题数据，包含解题过程，由ChatGPT产生
  - https://huggingface.co/datasets/BelleGroup/school_math_0.25M

- BelleGroup/generated_chat_0.4M
  - https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M
  - 个性化角色对话数据，包含角色介绍，由ChatGPT产生

- BelleGroup/train_2M_CN
  - https://huggingface.co/datasets/BelleGroup/train_2M_CN
  - 中文指令数据，由ChatGPT产生


## multi-language datasets

- MOSS 
  - https://huggingface.co/datasets/fnlp/moss-002-sft-data
    - 1.16M，类似self-instruct，人工编写一些seed
  - https://huggingface.co/datasets/fnlp/moss-003-sft-data
    - 多轮对话数据集，包含写作、代码、角色扮演、无害等类型数据集
    - https://huggingface.co/datasets/YeungNLP/moss-003-sft-data  
      官方数据的简化版本

- Guanaco
  - 1.17M，对alpaca self-instruct随机种子进行扩充
  - https://huggingface.co/datasets/JosephusCheung/GuanacoDataset

## english datasets

- InstructionWild
  - 110k，基于chatgpt用户共享出的指令构建，不同与self-instruct，多样性、真实性更高
  - https://github.com/XueFuzhao/InstructionWild
  
- HC3
  - 24k，human-chatgpt对比数据集
  - https://huggingface.co/datasets/Hello-SimpleAI/HC3

- stanford_alpaca
  - 52k，基于text-davinci-003模型以self-instruct方式生成
  - https://github.com/tatsu-lab/stanford_alpaca

- prosocial-dialog
  - 120k，多轮对话数据集，对有问题的内容做出正确的反应
  - https://huggingface.co/datasets/allenai/prosocial-dialog

- AlpacaDataCleaned
  - 50k，基于gpt4的模型以self-instuct方式生成，质量更高
  - https://github.com/gururise/AlpacaDataCleaned

- ultrachat
  - https://huggingface.co/datasets/YeungNLP/ultrachat