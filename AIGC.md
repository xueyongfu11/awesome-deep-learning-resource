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

- https://github.com/THUDM/ChatGLM-6B
- https://github.com/lich99/ChatGLM-finetune-LoRA
- https://github.com/mymusise/ChatGLM-Tuning
- https://huggingface.co/silver/chatglm-6b-slim
- https://github.com/THUDM/GLM-130B/
- https://github.com/THUDM/GLM
  - [ChatGLM内测](https://chatglm.cn/login)
  - [ChatGLM：千亿基座的对话模型开启内测⸺对应单卡版本开源](https://chatglm.cn/blog)

- https://github.com/tatsu-lab/stanford_alpaca
  - 基于自生成的instructions来微雕LLaMA模型
- https://github.com/tloen/alpaca-lora
  - 基于low-rank adaption训练的alpaca
  - 即使用了自生成的instruction和low-rank adaption微调/训练的LLaMA


# Paper

### 2023
- GLM-130B: AN OPEN BILINGUAL PRE-TRAINED MODEL
  - ICLM  [[code]](https://github.com/THUDM/GLM-130B/)
  - <details>
    <summary>阅读笔记: </summary>
    1. GLM使用了自编码和自回归的方式进行预训练  <br>
    2. 自编码使用blank infilling的方式训练，即提取spans，并替换成mask，使用自回归的方式预测mask，为了使得双向可见，将spans的顺序随机排列  <br>
    3. 每个样本对应唯一一个预训练任务，两种预训练任务使用不同mask token <br>
    4. 为了使得LLM的训练稳定，使用了DeepNorm；使用了三种训练策略；使用旋转位置编码，理论上支持无限长；FFN使用Gelu激活函数  <br>
    5. 训练集的95%使用自回归、自编码任务训练，分别占30%、70%，训练集的另外5%来自于各个NLP任务数据集，通过转化成prompt的方式来构建  <br>
    6. GLM的int4量化模型performance未明显下降，可以用4*3090运行
    <img src="" align="middle" />
    </details>

### 2022
- Training language models to follow instructions with human feedback
  - <details>
    <summary>阅读笔记: </summary>
    1. 使用人工编写的prompt数据，基于GPT3模型进行再训练，得到一个增强的预训练语言模型  <br>
    2. 基于1中训练好的预训练语言模型以及构建好的prompt集，使用beam-search等生成prompt的多个回复结果，然后人工对生成的多个结果排序，然后基于该数据训练了一个6B的打分模型，使用的是pair-wise的ranking loss  <br>
    3. 基于PPO算法训练强化学习模型，模型使用1中预训练好的模型进行初始化，给定prompt生成输出，然后用2中打分模型计算得分，然后优化强化学习模型。然后再使用强化学习模型生成，再打分。优化目标函数中添加了自回归语言模型的loss  <br>
    <img src="assets\instructGPT1.png" align="middle" />
    <img src="assets\instructGPT2.png" align="middle" />
    </details>



# Dataset
