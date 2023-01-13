<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Repo

- https://github.com/CLUEbenchmark/pCLUE
  - 中文提示学习排行榜，将多个不同NLP任务转化成基于prompt格式，使用生成式模型进行预测
- https://github.com/clue-ai/PromptCLUE
  
- https://github.com/bigscience-workshop/promptsource/ 
  - PromptSource is a toolkit for creating, sharing and using natural language prompts.
- zero shot model https://huggingface.co/bigscience/T0pp/tree/main
- thunlp prompt paper https://github.com/thunlp/PromptPapers
- thunlp开源 https://github.com/thunlp/OpenPrompt
- 刘知远 https://github.com/thu-coai/PPT


# Paper

- It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners
  - 2021 NAACL
  - 阅读笔记：
    1. 对PET的扩展，PET只支持一个mask，该文章支持多个mask
    2. 推理时获取最大置信度的token来替换相应位置的mask，然后再经过一个前向来计算其他位置的token，多次迭代完成
    3. 获取最大的mask长度，训练时使用一次前向，忽略冗余的mask token，计算交叉熵损失（比如最大mask长度时5，当前verbalizer长度时3，冗余的mask长度是2）
  - code: https://github.com/timoschick/pet

- Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference
  - 2021 EACL
  - 阅读笔记：
    1. 提出了一个基于prompt learning的少样本文本分类模型
    2. 构建patten，将任务转化为完形填空问题，将mask的输出结果映射到标签
    3. 损失有交叉熵损失和MLM loss组成。计算交叉熵损失时，需要先获取所有标签对应的verbalizer的logits，使用softmax计算得分，真实label用one-hot表示，然后计算交叉熵损失
    4. iPET：使用不同的patten训练多个模型，然后对部分无标注数据进行预测，将多个模型预测结果进行ensemble，作为soft-label训练集，然后和原始数据集合并再训练多个模型，上述过程迭代多次
  - code：https://github.com/timoschick/pet



