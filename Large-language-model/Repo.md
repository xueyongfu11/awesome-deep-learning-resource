<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Paper](#paper)
  - [subdomain-1](#subdomain-1)
  - [subdomain-1](#subdomain-1-1)
- [Dataset](#dataset)
 
<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Repo

- https://github.com/MLNLP-World/Awesome-LLM
- https://github.com/formulahendry/awesome-gpt

- https://github.com/dandelionsllm/pandallm/
- https://github.com/stability-AI/stableLM/

- https://huggingface.co/YeungNLP/firefly-1b4
- https://huggingface.co/YeungNLP/firefly-2b6

- https://github.com/FreedomIntelligence/LLMZoo
- https://github.com/project-baize/baize-chatbot/blob/main/README.md

- 超对称大模型：https://bbt.ssymmetry.com/model.html

- https://github.com/LC1332/Luotuo-Chinese-LLM

- MOSS
  - https://github.com/OpenLMLab/MOSS

- OpenAssistant
  - https://github.com/LAION-AI/Open-Assistant 
  - https://huggingface.co/datasets/OpenAssistant/oasst1
  - https://open-assistant.io 

- Vicuna
  - https://vicuna.lmsys.org/
  - https://github.com/lm-sys/FastChat
  - https://chat.lmsys.org/

- LAMMA
  - https://github.com/ymcui/Chinese-LLaMA-Alpaca
  - https://huggingface.co/blog/trl-peft
  - https://github.com/tatsu-lab/stanford_alpaca
    - 基于自生成的instructions来微雕LLaMA模型
  - https://github.com/tloen/alpaca-lora
    - 基于low-rank adaption训练的alpaca
    - 即使用了自生成的instruction和low-rank adaption微调/训练的LLaMA
  - https://github.com/Neutralzz/BiLLa
  - https://github.com/starmpcc/CAMEL
  - https://github.com/zjunlp/CaMA
  - https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
    - IDEA研究眼

- GLM
  - https://github.com/THUDM/ChatGLM2-6B
  - https://github.com/THUDM/ChatGLM-6B
  - https://github.com/lich99/ChatGLM-finetune-LoRA
    - 基于alpaca数据集，使用Lora技术微调ChatGLM-6B
  - https://github.com/mymusise/ChatGLM-Tuning
    - 基于alpaca数据集，使用Lora技术微调ChatGLM-6B
  - https://github.com/liangwq/Chatglm_lora_multi-gpu
    - 支持多机多卡训练
  - https://huggingface.co/silver/chatglm-6b-slim
    - 是在ChatGLM-6B的基础上通过裁剪词表构建的。因为ChatGLM-6B使用了icetk，在其词表中，前20000个token是预留给图片的
  - https://github.com/THUDM/GLM-130B/
  - https://github.com/THUDM/GLM
    - [ChatGLM内测](https://chatglm.cn/login)
    - [ChatGLM：千亿基座的对话模型开启内测⸺对应单卡版本开源](https://chatglm.cn/blog)
  - https://github.com/MediaBrain-SJTU/MedicalGPT-zh
    - 一个基于ChatGLM的在高质量指令数据集微调的中文医疗对话语言模型
  - https://github.com/hiyouga/ChatGLM-Efficient-Tuning
  - glm10B: https://huggingface.co/THUDM/glm-10b

- BLOOM
  - https://github.com/bigscience-workshop/xmtf
  - https://github.com/LianjiaTech/BELLE

- https://github.com/Muennighoff/sgpt#asymmetric-semantic-search-be
  - GPT sentence embedding for semantic search

- https://github.com/dair-ai/Prompt-Engineering-Guide
  - https://www.promptingguide.ai/
- https://huggingface.co/google/flan-t5-xxl
- https://github.com/nebuly-ai/nebullvm 
  - LLaMA+RLHF
- https://github.com/lucidrains/PaLM-rlhf-pytorch
  - PaLM+RLHF
- https://github.com/togethercomputer/OpenChatKit
  - 指令微调，对话时使用检索结果增强上下文，20b、6b
- https://github.com/bigscience-workshop/xmtf
- https://github.com/LAION-AI/Open-Assistant
  - Open Assistant 是 LAION 机构开源的，旨在训练一个 ChatGPT 的小规模替代版
  - Open-Assistant全流程训练指南:https://zhuanlan.zhihu.com/p/609003237
- https://github.com/CarperAI/trlx
  - trlX is a distributed training framework designed from the ground up to focus on fine-tuning large language models with reinforcement learning using either a provided reward function or a reward-labeled dataset.
- https://github.com/BlinkDL/RWKV-LM
  - 用 RWKV 架构（不是transformer结构）训练的chatgpt
- https://github.com/allenai/RL4LMs
  - 这个库包括了比较多 RL 算法（PPO，NLPO，A2C和TRPO），它论文说在 2000 个实验里做了 RL 在 LLM 上面的训练。RL4LMs当前的计划包括分布式训练更大的模型和新的RL算法
- https://github.com/hpcaitech/ColossalAI/tree/main/applications/ChatGPT
- https://github.com/LianjiaTech/BELLE
  - https://huggingface.co/BelleGroup

- https://github.com/lvwerra/trl
  - 基于GPT2的instructGPT
- https://github.com/HarderThenHarder/transformers_tasks/tree/main/RLHF
  - 基于上面的TRL做的；增加了基于人工打分的Reward模型训练，还提供了Reward数据的标注平台；RM 模型训练：基于 ernie-3.0-base-zh 继续训练的


## leaderboard

- https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- https://cluebenchmarks.com/superclue.html
- https://cevalbenchmark.com/static/leaderboard.html
  - 中文大模型评估排行榜
- http://115.182.62.166:18000/public
    - 大模型安全性评估

## RLHF&RL

- [RLHF几大常用框架实践对比（trlx、deepspeedchat、colossalaichat）](https://zhuanlan.zhihu.com/p/626046758?utm_campaign=shareopn&utm_medium=social&utm_oi=615941546193850368&utm_psn=1646992448919416832&utm_source=wechat_session)
- https://github.com/GanjinZero/RRHF
- https://github.com/CarperAI/trlx
- https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
- https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat


## 领域大模型

- https://github.com/scir-hi/huatuo-llama-med-chinese

- https://github.com/BioFM/OpenBioMed
  - 多模态生物医疗领域大模型

- https://github.com/pengxiao-song/LawGPT
  - 基于中文法律知识的大语言模型
  - 预训练+指令微调


## 并行训练框架

- https://github.com/hpcaitech/ColossalAI

