

# Paper

### 2023

- Language Is Not All You Need: Aligning Perception with Language Models
  - [[code]](https://github.com/microsoft/unilm)
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出了一个多模态因果大模型，方法是将图片用clip模型编码成离散的code，当作额外的token的加入到词表中。针对text和image，使用special token了区别模态的类型
    2. 使用Magneto来初始化模型，使用xPOS作为位置编码的方法，该方法的好处是解决了传统注意力消散的情况  <br>
    <img src="" align="middle" />
    </details>

- GLM-130B: AN OPEN BILINGUAL PRE-TRAINED MODEL
  - ICLM  [[code]](https://github.com/THUDM/GLM-130B/)
  - <details>
    <summary>阅读笔记: </summary>
    1. GLM使用了自编码和自回归的方式进行预训练  <br>
    2. 预训练方式1：自编码使用blank infilling的方式训练，即提取spans，并替换成mask，使用自回归的方式预测mask，spans的双向可见性取决于spans的随机排列顺序。每个样本对应唯一一个预训练任务，两种预训练任务使用不同mask token；预训练任务2：使用了多任务指令预训练  <br>
    3. 每个样本对应唯一一个预训练任务，两种预训练任务使用不同mask token <br>
    4. 为了使得LLM的训练稳定，使用了DeepNorm；使用了三种训练策略；使用旋转位置编码，理论上支持无限长；FFN使用Gelu激活函数  <br>
    5. 训练集的95%使用blank filling任务训练，分别占30%、70%，训练集的另外5%来自于各个NLP任务指令数据集，通过转化成prompt的方式构建  <br>
    6. GLM的int4量化模型performance未明显下降，可以用4*3090运行
    <img src="" align="middle" />
    </details>

### 2022
- Training language models to follow instructions with human feedback
  - <details>
    <summary>阅读笔记: </summary>
    1. 使用人工编写的prompt数据，基于GPT3模型进行再训练，得到一个增强的预训练语言模型  <br>
    2. 基于1中训练好的预训练语言模型以及构建好的prompt集，使用beam-search等生成prompt的多个回复结果，然后人工对生成的多个结果排序，然后基于该数据训练了一个6B的打分模型，使用的是pair-wise的ranking loss  <br>
    3. 基于PPO算法训练强化学习模型，模型使用1中预训练好的模型进行初始化，给定prompt生成输出，然后用2中打分模型计算得分，然后优化强化学习模型。然后再使用强化学习模型生成，再打分。优化目标函数中添加了自回归语言模型的loss  <br>
    <img src="assets\instructGPT1.png" align="middle" />
    <img src="assets\instructGPT2.png" align="middle" />
    </details>

- GLM: General Language Model Pretraining with Autoregressive Blank Infilling
  - ACL  [[code]](https://github.com/THUDM/GLM)
  - <details>
    <summary>阅读笔记: </summary>
    1. 使用了blank filling的自回归方式来统一所有任务目标。其通过mask spans来自回归的预测被mask的span，非span区域是互见的，span之间的可见性取决于span的随机排列顺序  <br>
    2. 为了获得更好的生成性能，通过mask更长span，以及对整句进行mask  <br>
    3. 使用的2D位置编码：被mask的序列绝对位置，非span区域位置为0，span内部位置从1开始编码  <br>
    <img src="" align="middle" />
    </details>

### 2021
- A General Language Assistant as a Laboratory for Alignment
  - <details>
    <summary>阅读笔记: </summary>
    1. 偏好模型是在序列的最后一个token上加value head，value head负责预测一个标量值来表示得分；模仿学习是只用good example来微调模型  <br>
    2. 排序偏好模型相比二进制偏好模型有更好的效果  <br>
    3. context distillation: prompt会减少输入的长度等缺点，使用了一种基于KL的loss来对prompt微调  <br>
    4. 偏好模型预训练的第二个阶段，使用二进制判别的预训练方法相比排序偏好方法有更好的收益
    </details>


# Blog

- [新加坡科技设计大学提出FLACUNA：提高LLMs问题解决能力!](https://mp.weixin.qq.com/s/QwLs9mdfyIwdTtCOwptkbA)
- [LLM综述全新出炉：51页论文带你盘点LLM领域专业化技术](https://mp.weixin.qq.com/s/bRQ7xjs4ewNJoE2Gug3lSA)