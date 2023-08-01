
# Repo

# Paper

## information extraction
### 2023

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
    <img src="" align="middle" />
    </details>



# Dataset
