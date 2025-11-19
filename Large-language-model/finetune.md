[TOC]

# 模型微调

- Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models

  - 2025.06，RLSC方法
  - 这是一种无需人类标注、外部奖励模型或手动奖励设计的强化学习框架，其核心是利用语言模型自身的置信度作为奖励信号，通过 “模态锐化” 优化模型输出分布以提升置信度。
  - 模态锐化（Mode Sharpening）：多数投票本质是选择输出分布的众数，隐含优化目标是提升众数的概率质量，RLSC 将此目标转化为可微分的自监督目标，直接利用模型自身置信度作为奖励信号，无需外部监督。
  
- TTRL: Test-Time Reinforcement Learning

  - 2025.04，TTRL方法
  - LLM先处理提示x，从策略πθ(y|x)中采样生成N个候选输出{y₁, y₂, ..., yₙ}。通过多数投票确定共识答案y*，再按与y*的匹配情况（匹配得1分、不匹配得0分）为各候选输出计算奖励，最后用PPO或GRPO等RL算法更新模型参数θ，以最大化预期奖励。
  
- Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate

  - 2025.01
  - 数据构建：以 WebInstruct 为基础，先获取错误的原始回答对，然后使用gpt-4o对错误回答进行结构化批判，给出错误位置、修正建议和正确推到过程。
  - 然后基于大模型进行训练，使用了LoRA微调，在多个测试基准中，相比sft等方法效果显著。
  - 局限性：训练用的批判数据也存在20%比例的噪声；CFT只能批判外部响应，对自身的响应无法进行批判。

- ORPO: Monolithic Preference Optimization without Reference Model

  - 2024.03, ORPO, 无需参考模型
  - ORPO的核心方法在于它不需要参考模型，并且可以在单一步骤中通过赋予不希望生成的风格一个小的惩罚，来高效地进行偏好对齐的监督式微调
  - ORPO通过在传统的负对数似然损失函数中加入一个基于赔率比（Odds Ratio）的惩罚项来区分优选和非优选的生成风格

- Contrastive Instruction Tuning

  - 2024.02
  - 

- Reasons to Reject? Aligning Language Models with Judgments

  - 2024

  - 该研究首次将 “拒绝理由监督” 融入对比训练框架：当模型生成低质量输出时，不仅提供优质回答作为正样本，还附加具体的批评性评语（如“逻辑矛盾”“信息缺失”）作为负样本标注。这种设计使模型同时学习“应该生成什么”和“应该避免什么”，解决了传统SFT仅优化正样本导致的“好坏不分”问题。

  - 论文提出的双向对比损失函数：通过传统SFT损失最大化优质回答的生成概率

    正样本赔率：$\text{odds}(y_w|x) = \frac{P(y_w|x)}{1-P(y_w|x)}$，负样本赔率：$\text{odds}(y_l|x) = \frac{P(y_l|x)}{1-P(y_l|x)}$ ，最大化两者的差值$\log \text{odds}(y_w|x) - \log \text{odds}(y_l|x)$ 

    $$L_{OR} = -\log \sigma\left( \log \text{odds}(y_w|x) - \log \text{odds}(y_l|x) \right)$$ 

    总损失：$L_{SFT} + \lambda \cdot L_{OR}$

- Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases

  - year: 2023
  - 主要探索了指令数据集的数据量对模型效果的影响，总的来说，仅仅通过增加指令数据的数量，可以连续的提升模型的性能
  - 随着数据量的增加，在不同数据集上的表现有所不同，在Extract, Classification, Closed QA, 和Summarization任务上，增加数据带来的效果提升并未到达天花板；在Math, Code, 和COT任务上，继续增加数据，效果出现下降；在Translation, Rewrite, 和Brainstorming任务上，少量的数据就可以获得不错的效果，继续增加数据，模型提升非常有限。

- RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
  - 2023.09
  - 方法旨在解决传统通过人类反馈进行强化学习中的一个关键瓶颈问题：获取高质量的人类偏好标签
  - 偏好标记：使用现成的LLM为一对候选摘要打上偏好标签。然后，使用对比损失训练一个奖励模型，最后使用RM提供的奖励进行策略模型的强化学习微调


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