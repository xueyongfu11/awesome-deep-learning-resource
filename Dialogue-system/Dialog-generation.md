<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Other](#other)
  - [基于预训练语言模型的对话机器人](#%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%AF%B9%E8%AF%9D%E6%9C%BA%E5%99%A8%E4%BA%BA)
- [Paper](#paper)
  - [Pre-trained-model-for-generation](#pre-trained-model-for-generation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

[[TOC]]

# Other
  
- https://github.com/Morizeyao/GPT2-Chinese

## 基于预训练语言模型的对话机器人

- 中文GPT模型 https://github.com/thu-coai/CDial-GPT  
  - 提供了中文对话数据集
  
- 基于GPT+NEZHA的预训练语言模型 https://github.com/bojone/nezha_gpt_dialog

- 中文GPT2 https://github.com/yangjianxin1/GPT2-chitchat
  - 在对话生成方向，GPT1和GPT2的模型是相同的
  - 该模型的发布的使用效果一般

- https://github.com/microsoft/ProphetNet
  - 包含中英文
  
  
# Paper

- Paper: EVA2.0: Investigating Open-Domain Chinese Dialogue Systems with Large-Scale Pre-Training
  - year: 
  - 阅读笔记: 
    1. 使用unilm模型结构
    2. 更好的数据清洗策略：句对相关性得分，句子流畅性，句子娱乐倾向性
    3. 在多个方面探索模型性能：模型层数，解码方式，解码重复问题处理，预训练方式（从头训练，在训练好模型上再训练）
    4. 结果相对CDial-GPT有很大提高
  - code: https://github.com/thu-coai/EVA

- Paper: RetGen: A Joint framework for Retrieval and Grounded Text Generation Modeling
  - year: 
  - 阅读笔记: 
    1. 将文本生成和外部文档结合进行回复生成  
    2. 将query和doc进行点积计算，找回少量文档  
    3. 将召回的每个文档和query,y[1:t-1]concate,预测y[t]  
    4. 比如找回4个文档，得到4个y[t]分布，综合考虑
    5. RetGen can generate more relevant, interesting and human-like text comparing to vanilla DialoGPT or GPT-2.
  - code: https://github.com/dreasysnail/RetGen

## Pre-trained-model-for-generation

- Paper: PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning
  - year: 2021
  - 阅读笔记: 
    1. two stage：第一阶段使用基本的unilm进行one-to-one模型的训练，目的是学习通用的文本生成能力
    2. 第二阶段：第一阶段的模型分别copy到两个模型：多样回复生成模型和回复连贯性评估模型
    3. 多样性回复生成模型类似去掉了回复选择任务的Plato-v1
    4. 回复连贯性评估模型：与plato-v1中的回复选择模型不同，response和context分别加上隐变量z，句首加CLS进行分类。推理时，预测多次采样生成的z预测出的回复的得分
  - code: https://github.com/PaddlePaddle/ Knover/tree/develop/projects/PLATO-2

- Paper: PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable
  - year: 2020
  - 阅读笔记: 
    1. 模型输入：tokens，role embedding， relative turn embedding（current turn为0），pos embedding
    2. 隐变量z：训练时先用context，response计算z，推理时采样z，生成response，采样不同的z得到不同的response
    3. LOSS：NLL loss，BOW loss（通过z和context，无顺序的预测response中的token），response select loss（真实response，sample response，使用隐变量token接全连接进行分类）
  - code: https://github.com/PaddlePaddle/ Research/tree/master/NLP/Dialogue-PLATO
  
- Paper: DIALOGPT : Large-Scale Generative Pre-training for Conversational Response Generation
  - year: 2020 微软
  - 阅读笔记: 
    1. 使用GPT2
    2. 使用MMI，是一个问题，回复的逆序拼接数据训练出来的GPT2模型
    3. 给生成的topk使用MMI模型打分，loss替代
    4. 选择loss最小的response作为最终的结果进行回复
  - code: https://github.com/microsoft/DialoGPT
