


# 数据筛选

- LESS: Selecting Influential Data for Targeted Instruction Tuning
  - year: 2024
  - 利用梯度信息来筛选少量训练集
  - codes: https://github.com/princeton-nlp/LESS
  - [Less is More](https://mp.weixin.qq.com/s/8KYNYvKCWhRJ3BWJxe0-Qw)

- What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
  - year: 2023
  - 先对数据进行复杂性和质量评分，再通过多样性进行数据筛选
  - code: https://github.com/hkust-nlp/deita
  - [DEITA-大模型指令微调的数据高效筛选方法](https://zhuanlan.zhihu.com/p/675928711)

- MoDS: Model-oriented Data Selection for Instruction Tuning
  - year: 2023
  - 首先使用deberta模型对数据质量进行打分，得到质量较高的数据集
  - 基于K-center-greedy方法，从得到的数据中获取最大多样化的数据子集
  - 基于种子子集微调一个大模型，基于该大模型对1中高质量数据集进行预测，使用奖励模型获取不能很好预测结果的数据，这样的数据对大模型难度更高，最后将难例数据和数据子集混合起来，训练最终效果更好的模型
  - code: https://github.com/CASIA-LM/MoDS
  - [高质量指令数据筛选方法-MoDS](https://zhuanlan.zhihu.com/p/671183709)

- Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
  - 总共两步：生成（E-step）：语言模型为每个输入上下文生成多个输出样本，然后使用二元奖励过滤这些样本以收集训练数据集.
  - 改进（M-step）：原始语言模型在来自前一个 E-step 的训练数据集上进行监督微调，然后在下一个 E-step 中使用。

- From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning
  - year: 2023
  - 指令数据筛选：首先基于聚类的方法筛选出多样性比较高的少量样本，然后对模型进行微调
  - 基于微调的大模型计算样本的指令跟随难度，即模型预测答案的概率越小，对模型来说难度越高，使用这样的样本继续训练能够带来更好的效果
  - code：https://github.com/MingLiiii/Cherry_LLM

- Lion: Adversarial Distillation of Closed-Source Large Language Model
  - 先是根据chatgpt生成的数据进行模仿学习；基于小模型的生成结果来判别难样本；再生成数据来学习这种难样本
  - [blog](https://mp.weixin.qq.com/s/_LQVHMJqPzMzIuM4wsO2Dw)

- LIMA: Less Is More for Alignment
  - year: 2023
  - 仅仅使用高质量的少量数据，便得到一个效果很好的大模型
  - 在多轮对话上，添加30个高质量的多轮对话链，使得模型的多轮对话能力显著提升
  - [Meta AI 重磅推出LIMA！媲美GPT-4、无需RLHF就能对齐](https://mp.weixin.qq.com/s/sbIa-fIHvMlp-2aYtCtVLQ)