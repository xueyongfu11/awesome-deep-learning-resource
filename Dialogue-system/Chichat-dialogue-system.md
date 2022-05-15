
# Repo
- chatterBot闲聊机器人   https://github.com/gunthercox/ChatterBot
  - 基于文档知识库的问答系统
  
- https://github.com/Morizeyao/GPT2-Chinese

## 基于预训练语言模型的对话机器人

- 中文GPT模型 https://github.com/thu-coai/CDial-GPT  
  - 提供了中文对话数据集
  
- 基于GPT+NEZHA的预训练语言模型 https://github.com/bojone/nezha_gpt_dialog

- 引入外部文档的对话生成 https://github.com/dreasysnail/RetGen

- 中文GPT2 https://github.com/yangjianxin1/GPT2-chitchat
  - 在对话生成方向，GPT1和GPT2的模型是相同的
  - 该模型的发布的使用效果一般

- https://github.com/microsoft/ProphetNet
  - 包含中英文
  
- 中文对话系统 https://github.com/thu-coai/EVA

- 悟道 https://github.com/TsinghuaAI/CPM-1-Generate
  

# Paper

## Pre-trained-model-for-generation

- Paper: PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning
  - year: 2021
  - 阅读笔记: 
    1.two stage：第一阶段使用基本的unilm进行one-to-one模型的训练，目的是学习通用的文本生成能力
    2.第二阶段：第一阶段的模型分别copy到两个模型：多样回复生成模型和回复连贯性评估模型
    3.多样性回复生成模型类似去掉了回复选择任务的Plato-v1
    4.回复连贯性评估模型：与plato-v1中的回复选择模型不同，response和context分别加上隐变量z，句首加CLS进行分类。推理时，预测多次采样生成的z预测出的回复的得分
  - code: https://github.com/PaddlePaddle/ Knover/tree/develop/projects/PLATO-2

- Paper: PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable
  - year: 2020
  - 阅读笔记: 
    1.模型输入：tokens，role embedding， relative turn embedding（current turn为0），pos embedding
    2.隐变量z：训练时先用context，response计算z，推理时采样z，生成response，采样不同的z得到不同的response
    3.LOSS：NLL loss，BOW loss（通过z和context，无顺序的预测response中的token），response select loss（真实response，sample response，使用隐变量token接全连接进行分类）
  - code: https://github.com/PaddlePaddle/ Research/tree/master/NLP/Dialogue-PLATO
  
- Paper: DIALOGPT : Large-Scale Generative Pre-training for Conversational Response Generation
  - year: 2020 微软
  - 阅读笔记: 
    1.使用GPT2
    2.使用MMI，是一个问题，回复的逆序拼接数据训练出来的GPT2模型
    3.给生成的topk使用MMI模型打分，loss替代
    4.选择loss最小的response作为最终的结果进行回复
  - code: https://github.com/microsoft/DialoGPT


# Datasets

- 训练中文对话系统的语料库 https://github.com/candlewill/Dialog_Corpus

- 中文聊天语料库 https://github.com/codemayq/chinese_chatbot_corpus