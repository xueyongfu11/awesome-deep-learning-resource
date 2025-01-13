[TOC]



# Resource

- https://github.com/EricLBuehler/xlora
- https://github.com/huggingface/peft
- https://github.com/thunlp/OpenDelta
- https://github.com/adapter-hub/adapter-transformers
- https://github.com/Chaos96/fourierft

# survey

- Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning


# peft

- Learning to Route Among Specialized Experts for Zero-Shot Generalization	
  
  - 2024.02
  
  - PHATGOOSE旨在通过参数高效微调后的专门化模块中进行路由，以改善对未见任务的零样本泛化能力。
  
  - 与以往学习在专门化模型间路由的方法同，PHATGOOSE探索了一种可能性，即如果能够针对模型中的每一层和每个token自适应地选择不同的专家，可能会提高零样本泛化能力。
  - 这种方法是后处理的（post-hoc），它不需要同时访问用于创建专门化模型的数据集，且在每个专家模型训练后只需要适度的额外计算。
- LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition

  - 2023.07, 
  - [blog](https://mp.weixin.qq.com/s/QlTxcwdtOt8NizqqBzo6yg)
  - 在不同的任务上训练相应的lora
  - 将训练好的lora加权起来，权重的计算使用了无梯度的组合优化算法，优化目标是评估集的loss，以及lora权重的正则项
  - 数据集：https://adapterhub.ml/explore/text_task/
- CPET: Effective Parameter-Efficient Tuning for Compressed Large Language Models
  - 2023.07
  - 为了减少模型压缩带来的知识损失，提出了知识继承和知识恢复模块
  - 首先使用未压缩的模型基于lora训练，得到lora的权重。然后使用压缩的模型，并使用训练好的lora进行初始化。
  - 为了恢复知识损失，对每个线形层添加恢复模块，具体是使用bottleneck结构，并使用sigmoid函数进行信息激活
- Composing Parameter-Efficient Modules with Arithmetic Operations
  - 2023.06
  - 通过线性组合不同peft模块来研究分布泛化：不同分布数据训练的peft，多任务能力：不同nlp任务训练的peft，去学习能力：减少某种能力，领域泛化：泛化其他领域能力
- π-Tuning: Transferring Multimodal Foundation Models with Optimal Multi-task Interpolation
  - 2023.04，ICML2023
  - 提出了一种名为π-Tuning的参数高效迁移学习方法，用于处理视觉、语言和视觉-语言任务。
  - π-Tuning的核心思想是通过预测任务之间的相似性，并将相似任务的轻量级专家参数进行聚合，以辅助目标下游任务的学习
- LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Model
  - 2023
  - 提出大模型微调用的adapter训练系统框架，包含常用的adapter，lora
  - https://github.com/AGI-Edgerunners/LLM-Adapters
- ConPET: Continual Parameter-Efficient Tuning for Large Language Models
  - 2023
  - 提出基于PET的memory-based大模型持续学习方法
  - Static ConPET：使用一个PET模块，历史数据的获取使用了动态的历史数据重现的方法
  - Dynamic ConPET：使用多个PET模块，基于一个选择模块选取topk的PET模块，然后合并预测结果，为了减少计算负担，将pet模块的预测结果cache起来
  - 选择模块从支持k-1个schema到支持k个schema，涉及到分类类别维度的扩展
- LONGLORA: EFFICIENT FINE-TUNING OF LONGCONTEXT LARGE LANGUAGE MODELS
  - 2023
  - 使用了shift short attention来对大模型上下文长度进行扩展
  - shift short attention：一半注意力头只在组内进行注意力的计算，另外一半注意力头使用带重叠的滑动，使得不同组可以进行信息交互
  - 为了解决长上下文lora微调相比full finetune效果差的问题，对embedding和LN层进行了微调
- Multi-Head Adapter Routing for Cross-Task Generalization
  - 2022.11，NeurlPS2023
  - 问题背景：研究者们关注于如何使用少量样本在多任务预训练后对未见任务进行泛化
  - 论文中提到了Polytropon（Poly）模型，它联合学习了一系列适配器和一个路由函数，该路由函数在预训练和少量样本适应期间为每个任务选择一组适配器的子集
  - 基于Polytropon模型，论文提出了MHR，这是一种新的路由函数，它通过将适配器的参数块组合起来，以更细粒度的方式进行路由，从而提高了模型的表达能力
- Multi-Head Adapter Routing for Data-Efficient Fine-Tuning
  - 2022.11
  - 对前继工作Poly的讨论
  - 相比Poly对adapter的线性组合，引入类似multi-head attention机制，将adapter中的A/B进行分片，在分片内线性组合，然后再把结果进行concatenate
- AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning
  - 2022.10
- Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning
  - 2022.05, (IA)3
  - 为了使微调更有效，IA3（通过抑制和放大内部激活注入适配器）使用学习向量重新调整内部激活
  - IA3 权重被添加到 Transformer 模型的 key， value 和 feedforward 层
  - IA3 不会增加任何推理延迟，因为适配器（adapter）权重可以与基础模型合并
- Combining Modular Skills in Multitask Learning
  - 2022.02
  - 提出一种多任务学习的peft模块组合方法
  - 预训练阶段：每个task由S个skill组成，每个skill对应一个adapter，同时每个任务对应一个skill selection binary matrix。预训练阶段学习的参数是adapter和skill selection矩阵
  - 测试阶段：对每个测试任务初始化相应的skill selection binary matrix，同时微调所有skill对应的adapter
  - code：https://github.com/microsoft/mttl
- P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks
  - 2021.10, ACL2022
  - 相比P-tuning v1只在输入部分添加连续的prompt token，v2在每一层添加prompt token
  - 做抽取任务时，类似与bert做token分类，直接对每个位置的输出做分类
  - 探索：
      1）分类任务使用更少的prompt token，抽取任务使用更多的prompt
      2）先使用多任务的方式微调参数，再单个任务微调能有一定的提升
      3）在靠近输出的模型层上添加prompt token，能获得更好的效果  
      4）Deep Prompt Tuning:在每一层添加prompt token,方法是先设定输入的prompt token，使用全连接层为每个token生成每层的k和v向量
- CrossFit: A Few-shot Learning Challenge for Cross-task Generalization in NLP
  - 2021.04, EMNLP2021
- The power of scale for parameter-efficient prompt tuning
  - 2021.04
  - Prompt Tuning方法
  - 在模型的输入层即embedding层concat上prompt token embedding
  - 训练时直接对prompt token embedding进行优化
- GPT Understands, Too
  - 2021.03
  - P-tuning方法
  - 将prompt转化成可学习的embedding，并使用mlp+lstm对prompt embedding进行处理
  - 仅限在模型的输入层
- Prefix-Tuning: Optimizing Continuous Prompts for Generation
  - 2021.01
  - 不同与Prompt Tuning，Prefix-Tuning时在每层添加prompt token，且每层的token embedding是共享同一个参数矩阵P
  - 直接更新P矩阵的参数会导致优化时不太稳定，因此使用了一个小矩阵P’，并通过一个大的前馈神经网络得到P矩阵
- AdapterFusion: Non-Destructive Task Composition for Transfer Learning
  - 2020.05
  - 每个任务训练相应的adapter，使用adapterFusion融合训练好的adapter
  - adapterFusion：使用FFN的输出作为query，各个adapter的输出作为key和value，然后计算注意力
- Parameter-Efficient Transfer Learning for NLP
  - 2019.02
  - adapter


# task-related peft

- Infusing Hierarchical Guidance into Prompt Tuning: A Parameter-Efficient Framework for Multi-level Implicit Discourse Relation Recognition
  - 2024.02, ACL

- Knowledgeable Parameter Efficient Tuning Network for Commonsense Question Answering
  - 2023, ACL
  - 在freeze的bert的每层添加参数共享的adapter模块来吸收知识
  - 知识抽取：根据query中的实体获取三元组，然后将三元组组成句子从图数据库中检索相似的句子片段；直接使用query从图数据库中检索相似的句子片段
  - 模型架构：将PLM的第l层的输出和上一层adapter的输出拼接起来，注意PLM的第l层输出要使用一个门函数
  

