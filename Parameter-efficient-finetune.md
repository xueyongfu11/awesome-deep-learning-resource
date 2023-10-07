

# Repo

- https://github.com/huggingface/peft
- https://github.com/thunlp/OpenDelta
- https://github.com/adapter-hub/adapter-transformers



# survey

### 2023
- Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning



# peft-llm

### 2023

- ConPET: Continual Parameter-Efficient Tuning for Large Language Models
  - 提出基于PET的memory-based大模型持续学习方法
  - Static ConPET：使用一个PET模块，历史数据的获取使用了动态的历史数据重现的方法
  - Dynamic ConPET：使用多个PET模块，基于一个选择模块选取topk的PET模块，然后合并预测结果，为了减少计算负担，将pet模块的预测结果cache起来
  - 选择模块从支持k-1个schema到支持k个schema，涉及到分类类别维度的扩展

- LONGLORA: EFFICIENT FINE-TUNING OF LONGCONTEXT LARGE LANGUAGE MODELS
  - 使用了shift short attention来对大模型上下文长度进行扩展
  - shift short attention：一半注意力头只在组内进行注意力的计算，另外一半注意力头使用带重叠的滑动，使得不同组可以进行信息交互
  - 为了解决长上下文lora微调相比full finetune效果差的问题，对embedding和LN层进行了微调

- Bayesian low-rank adaptation for large language models

- LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language
  - 提出大模型微调用的adapter训练系统框架，包含常用的adapter，lora
  - https://github.com/AGI-Edgerunners/LLM-Adapters

- LORA-FA: MEMORY-EFFICIENT LOW-RANK ADAPTATION FOR LARGE LANGUAGE MODELS FINE-TUNING
  - 对lora改进，通过freeze lora中的A的权重，只对B的权重进行微调，无需存储A输入的激活，存储B输入的激活，从而通过减小激活的方式来减少显存占用


# peft

### 2023

- LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
  - [blog](https://mp.weixin.qq.com/s/QlTxcwdtOt8NizqqBzo6yg)
  - 在不同的任务上训练相应的lora
  - 将训练好的lora加权起来，权重的计算使用了无梯度的组合优化算法，优化目标是评估集的loss，以及lora权重的正则项
  - 数据集：https://adapterhub.ml/explore/text_task/

- CPET: Effective Parameter-Efficient Tuning for Compressed Large Language Models
  - 为了减少模型压缩带来的知识损失，提出了知识继承和知识恢复模块
  - 首先使用未压缩的模型基于lora训练，得到lora的权重。然后使用压缩的模型，并使用训练好的lora进行初始化。
  - 为了恢复知识损失，对每个线形层添加恢复模块，具体是使用bottleneck结构，并使用sigmoid函数进行信息激活

- Composing Parameter-Efficient Modules with Arithmetic Operations
  - 通过线性组合不同peft模块来研究分布泛化：不同分布数据训练的peft，多任务能力：不同nlp任务训练的peft，去学习能力：减少某种能力，领域泛化：泛化其他领域能力

### 2022

- AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning

- P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks
  - <details>
    <summary>阅读笔记: </summary>
    - 相比P-tuning v1只在输入部分添加连续的prompt token，v2在每一层添加prompt token  <br>
    - 做抽取任务时，类似与bert做token分类，直接对每个位置的输出做分类  <br>
    - 探索：
        1）分类任务使用更少的prompt token，抽取任务使用更多的prompt
        2）先使用多任务的方式微调参数，再单个任务微调能有一定的提升
        3）在靠近输出的模型层上添加prompt token，能获得更好的效果  <br>
        4）Deep Prompt Tuning:在每一层添加prompt token,方法是先设定输入的prompt token，使用全连接层为每个token生成每层的k和v向量
    <img src="" align="middle" />
    </details>

### 2021

- AdapterFusion: Non-Destructive Task Composition for Transfer Learning
  - 每个任务训练相应的adapter，使用adapterFusion融合训练好的adapter
  - adapterFusion：使用FFN的输出作为query，各个adapter的输出作为key和value，然后计算注意力

- LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
  - [[code]](https://github.com/microsoft/LoRA)
  - <details>
    <summary>阅读笔记: </summary>
    - 提出了一种低秩自适应的模型微调方式：freeze整个模型的参数，在每个transformer层注入可训练的秩分解矩阵来适应下游任务  <br>
    - 对self-attention中的q k v以及输出投射层进行了测试，发现在q v上添加lora层与在所有权重上添加效果相同，都取得了最好的效果  <br>
    - 随着rank的提高，模型并没有取得更好的效果，原因是low-rank已经捕获了足够的信息  <br>
    - 低秩矩阵与相应的模型层的权重呈现很强的相关性 <br>
    <img src="./assets/lora.jpg" align="middle" />
    </details>

### 2019

- Parameter-Efficient Transfer Learning for NLP
  - adapter

# task-related peft

### 2023
- Knowledgeable Parameter Efficient Tuning Network for Commonsense Question Answering
  - 在freeze的bert的每层添加参数共享的adapter模块来吸收知识
  - 知识抽取：根据query中的实体获取三元组，然后将三元组组成句子从图数据库中检索相似的句子片段；直接使用query从图数据库中检索相似的句子片段
  - 模型架构：将PLM的第l层的输出和上一层adapter的输出拼接起来，注意PLM的第l层输出要使用一个门函数
  - ACL

- Infusing Hierarchical Guidance into Prompt Tuning: A Parameter-Efficient Framework for Multi-level Implicit Discourse Relation Recognition
  - ACL