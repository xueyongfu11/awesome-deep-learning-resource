

- SeqGPT: An Out-of-the-box Large Language Model for Open Domain Sequence Understanding
  - 通过提示chatgpt来生成预训练数据，收集了大量中英文的NLU任务数据进一步微调。模型主要使用了BLOOMZ
  - 探究了数据scale，模型scale，域外数据泛化性、跨语言泛化性、跨任务泛化性

- UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition
  - https://universal-ner.github.io/

- ZeroPrompt: Scaling Prompt-Based Pretraining to 1,000 Tasks Improves
Zero-shot Generalization
  - [相关blog](https://zhuanlan.zhihu.com/p/461075783)
  - 使用1000个nlp任务构建prompt数据，提升了zero-shot能力，在部分任务上超过了finetune，相比其他zero-shot方法提升了30个点


### 2023

- Easy-to-Hard Learning for Information Extraction
  1. 使用了easy-to-hard来学习信息抽取任务，按顺序学习三个任务：easy stage, hard stage, main stage
  2. easy stage: 把每个任务分解成多个简单的skill来学习
  3. hard stage: 把text中包含多个target或者拼接两个text和相应的target作为学习样本
  4. main stage: 学习一个样本的所有的skill

- InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction
  - <details>
    <summary>阅读笔记: </summary>
    1. 将NER，RE，EE等数据集转化为instruction格式，基于FlanT5-10B模型训练  <br>
    2. 针对NER，RE，EE等任务都加入辅助任务使得模型更好的学习  <br>
    3. 相对于Bert-finetune，InstructUIE去的更好的效果；在少样本数据集上，超过了chatgpt3.5模型  <br>
    </details>

- Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors
  - ACL  [[code]](https://github.com/OSU-NLP-Group/QA4RE)
  - <details>
    <summary>阅读笔记: </summary>
    1. 认为指令微调并不能提高大模型在RE任务上的效果，原因是因为RE数据的占比很小  <br>
    2. 为了解决上诉问题，将RE任务和数据占比非常多的QA任务对齐起来，从而提高RE的任务效果   <br>
    3. QA任务:提供候选答案的选项，直接让模型输出选项号  <br>
    </details>

- Revisiting Relation Extraction in the era of Large Language Models
  - <details>
    <summary>阅读笔记: </summary>
    1. 基于大模型的关系抽取，使用GPT-3在few-shot的prompt下取得了解决sota的效果；在此基础上使用CoT，能够进一步提升效果  <br>
    2. 基于flan-T5在few-shot prompt情况下并未取得sota效果，但是在基于GPT-3生成的CoT数据微调，能够取得sota的效果  <br>
    </details>

- Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!
  - <details>
    <summary>阅读笔记: </summary>
    1. 将大模型和小模型结合起来做关系抽取任务  <br>
    2. 使用小模型过滤出困难样本（根据关系的置信度），然后转化成多项选择问题，输出使用了analysis（CoT）  <br>
    </details>

- InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis
  - <details>
    <summary>阅读笔记: </summary>
    1. 基于GPT2模型微调，将多方面情感分析人物转化为instruction方式  <br>
    </details>

### 2022

- InstructionNER: A Multi-Task Instruction-Based Generative Framework for Few-shot NER
  - <details>
    <summary>阅读笔记: </summary>
    1.  基于T5模型的指令ner信息抽取  <br>
    </details>



