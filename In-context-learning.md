

- https://github.com/Shark-NLP/OpenICL


### 2023

- Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation
  - 探究了在同等模型参数条件下few-shot ICL和few-shot微调的跨领域泛化效果对比，二者取得了相近的性能
  - 随着模型参数的增加，in-domain和OOD上的评估效果都进一步提升，并且微调相比ICL在更多情况下取得了更好的效果

- In-Context Instruction Learning
  - 相比ICL，ICIL使用了由多个任务数据组成的prompt作为in-context样例
  - 本质是加入了固定样例数据组成的prompt的指令学习，好处是不需要为不同的任务添加相应的in-context样本

- Learning to Retrieve In-Context Examples for Large Language Models
  - 提出了一个新颖的框架去迭代性的训练向量检索工具，目的是为了识别高质量的in-context样本


### 2022

- Improving In-Context Few-Shot Learning via Self-Supervised Training
  - Meta Ai; NAACL-HLT 2022
  - 为了提高in-context few-shot能力，提出了在预训练和下游few-shot学习这两个阶段之间添加自监督训练，目的是为了减少预训练模型预训练目标和in-context
few-shot学习目标的差异

- What Makes Good In-Context Examples for GPT-3?
  - Duke & microsoft; DeeLIO 2022
  - 提出基于展示样例检索的in-context学习