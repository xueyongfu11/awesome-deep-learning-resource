



- MoDS: Model-oriented Data Selection for Instruction Tuning
  - year: 2023
  - 


- From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning
  - year: 2023
  - 指令数据筛选：首先基于聚类的方法筛选出多样性比较高的少量样本，然后对模型进行微调
  - 基于微调的大模型计算样本的指令跟随难度，即模型预测答案的概率越小，对模型来说难度越高，使用这样的样本继续训练能够带来更好的效果
  - code：https://github.com/MingLiiii/Cherry_LLM
  - 

- LIMA: Less Is More for Alignment
  - year: 2023
  - 仅仅使用高质量的少量数据，便得到一个效果很好的大模型
  - 在多轮对话上，添加30个高质量的多轮对话链，使得模型的多轮对话能力显著提升
  - [Meta AI 重磅推出LIMA！媲美GPT-4、无需RLHF就能对齐](https://mp.weixin.qq.com/s/sbIa-fIHvMlp-2aYtCtVLQ)