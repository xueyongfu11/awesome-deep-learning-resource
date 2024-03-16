<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [2023](#2023)
- [2022](#2022)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Information Extraction

**2023**

- Improving Open Information Extraction with Large Language Models: A Study on Demonstration Uncertainty
  - 一种基于大模型chatgpt或者llama的多轮的开放信息抽取方法。
  - 提出错误纠正的trick：通过提问大模型抽取并进行纠正
  - 使用样例展示的few-shot方法：使用基于指令大模型embedding方法，来检索文本结构组成相似的样本作为展示样本
  - 提出一种大模型生成不确定性评估方法：基于3的方法采样多组的展示样本，然后统计三元组的概率并使用阈值过滤掉低置信度的样本

- SeqGPT: An Out-of-the-box Large Language Model for Open Domain Sequence Understanding
  - 通过提示chatgpt来生成预训练数据，收集了大量中英文的NLU任务数据进一步微调。模型主要使用了BLOOMZ
  - 探究了数据scale，模型scale，域外数据泛化性、跨语言泛化性、跨任务泛化性

- UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition
  - https://universal-ner.github.io/
  - https://huggingface.co/Universal-NER

- ZeroPrompt: Scaling Prompt-Based Pretraining to 1,000 Tasks Improves
Zero-shot Generalization
  - [相关blog](https://zhuanlan.zhihu.com/p/461075783)
  - 使用1000个nlp任务构建prompt数据，提升了zero-shot能力，在部分任务上超过了finetune，相比其他zero-shot方法提升了30个点


- Easy-to-Hard Learning for Information Extraction
  - 使用了easy-to-hard来学习信息抽取任务，按顺序学习三个任务：easy stage, hard stage, main stage
  - easy stage: 把每个任务分解成多个简单的skill来学习
  - hard stage: 把text中包含多个target或者拼接两个text和相应的target作为学习样本
  - main stage: 学习一个样本的所有的skill

- InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction
  - <details>
    <summary>阅读笔记: </summary>
    - 将NER，RE，EE等数据集转化为instruction格式，基于FlanT5-10B模型训练  <br>
    - 针对NER，RE，EE等任务都加入辅助任务使得模型更好的学习  <br>
    - 相对于Bert-finetune，InstructUIE去的更好的效果；在少样本数据集上，超过了chatgpt3.5模型  <br>
    </details>

- Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors
  - ACL  [[code]](https://github.com/OSU-NLP-Group/QA4RE)
  - <details>
    <summary>阅读笔记: </summary>
    - 认为指令微调并不能提高大模型在RE任务上的效果，原因是因为RE数据的占比很小  <br>
    - 为了解决上诉问题，将RE任务和数据占比非常多的QA任务对齐起来，从而提高RE的任务效果   <br>
    - QA任务:提供候选答案的选项，直接让模型输出选项号  <br>
    </details>

- Revisiting Relation Extraction in the era of Large Language Models
  - <details>
    <summary>阅读笔记: </summary>
    - 基于大模型的关系抽取，使用GPT-3在few-shot的prompt下取得了解决sota的效果；在此基础上使用CoT，能够进一步提升效果  <br>
    - 基于flan-T5在few-shot prompt情况下并未取得sota效果，但是在基于GPT-3生成的CoT数据微调，能够取得sota的效果  <br>
    </details>

- Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!
  - <details>
    <summary>阅读笔记: </summary>
    - 将大模型和小模型结合起来做关系抽取任务  <br>
    - 使用小模型过滤出困难样本（根据关系的置信度），然后转化成多项选择问题，输出使用了analysis（CoT）  <br>
    </details>

- InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis
  - <details>
    <summary>阅读笔记: </summary>
    - 基于GPT2模型微调，将多方面情感分析人物转化为instruction方式  <br>
    </details>

**2022**

- InstructionNER: A Multi-Task Instruction-Based Generative Framework for Few-shot NER
  - <details>
    <summary>阅读笔记: </summary>
    -  基于T5模型的指令ner信息抽取  <br>
    </details>


# Table understanding

- https://github.com/SpursGoZmy/Paper-List-about-LLM-and-Table

- TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT
  - [浙大团队推出TableGPT：可分析数据、作图和建模](https://zhuanlan.zhihu.com/p/643198387)

- [综述：大型模型在表格数据上的应用](https://zhuanlan.zhihu.com/p/684981680)
  - 包括传统的小模型和大模型在表格数据上的应用

- [大模型技术在表格数据任务上的应用进展](https://mp.weixin.qq.com/s/VLOqVXoOO81arJuBkIO6Jw)
  - 部分paper并非是LLM相关的

- [面向表格数据的大模型推理综述](https://mp.weixin.qq.com/s/sXeyH2Ob8-CbGwHHBYOYfQ)
  - 微调、ICL、使用工具、鲁棒性

