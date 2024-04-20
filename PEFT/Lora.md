

# Lora-related

- PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large
  - 2024

- PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA
  - 2024

- RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation
  - 2024
  - RoSA同时训练两种类型的组件，即低秩组件和稀疏组件。低秩组件旨在捕获微调更新的主要结构，而稀疏组件则用于捕获更新中的重要但不那么显著的方向
  - RoSA借鉴了鲁棒PCA的思想，通过将微调更新表示为低秩矩阵和稀疏矩阵的和，从而在保持参数和计算效率的同时，提供更准确的模型更新
  - 生成稀疏掩码：RoSA使用一种特别设计的算法来生成稀疏掩码（masks），这些掩码用于确定哪些权重在微调过程中应该被更新

- Learning to Route Among Specialized Experts for Zero-Shot Generalization
  - 2024
  - 它在每个专家模型训练完成后，通过一个额外的计算步骤来训练一个门控机制，而不需要对模型的其他部分进行进一步的训练
  - 探索了在模型的每个层级为每个令牌选择不同的专家模块的可能性，这样可以在不同的阶段或对不同的令牌使用不同的专家能力，从而可能提高对新任务的泛化能力
  - 在每个PEFT模块前引入一个sigmoid门控层，并训练一个门控向量，该向量决定了是否将给定序列位置的激活传递到模块中。这个门控向量是在模型的所有序列位置共享的
  - 在推理阶段，PHATGOOSE使用标准的“top-k”路由策略，根据门控向量与给定激活的点积最高来选择模块

- AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models
  - 2024
  - AFLoRA的核心思想是在微调过程中适应性地冻结低秩适应路径中的投影矩阵，以减少计算成本和缓解过拟合
  - 自适应冻结：AFLoRA引入了一个新颖的冻结得分（freezing score）机制，根据这个得分在微调过程中逐步冻结投影矩阵。这个得分基于权重在训练过程中的变化程度，当权重的变化变得可以忽略时，就可以将它们冻结

- LoTR: Low Tensor Rank Weight Adaptation
  - 2024
  - LoTR是LoRA的扩展和泛化，旨在通过张量分解来改进微调过程，特别是在深度模型中
  - LoTR通过将每个层的低秩适配器构建为三个矩阵的乘积，从而在Transformer块中共享左右乘数，从而实现了更好的参数效率

- BiLoRA: A Bi-level Optimization Framework for Overfitting-Resilient Low-Rank Adaptation of Large Pre-trained Models
  - 2024
  - 这篇论文提出了一个名为BiLoRA的新方法，旨在解决大型预训练模型在下游任务中微调时的过拟合问题，高模型在测试数据上的泛化能力
  - 双层次优化（BLO）：BiLoRA采用BLO框架，将参数分为两个层次进行优化。在较低层次，优化伪奇异向量矩阵（P和Q），而在较高层次，优化伪奇异值矩阵（Λ）
  - 正则化：为了保持P和Q的正交性，BiLoRA应用了正则化项R1。此外，还可以使用R2来鼓励Λ中的伪奇异值接近二值（0或1），进一步约束模型的复杂度

- Navigating Text-To-Image Customization:From LyCORIS Fine-Tuning to Model Evaluation
  - 2024
  - LoHa: 这是LoRA的扩展，它使用Hadamard乘积（逐元素乘积）来进一步增加权重更新的秩，同时保持了与原始LoRA方法相同的可训练参数数量。
  - LoKr: 这是另一种扩展方法，使用Kronecker乘积来增加权重更新的秩。这种方法允许更大的矩阵秩，从而可能提高微调的性能

- LoDA: Low-Dimensional Adaptation of Large Language Models
  - 2024
  - 通过将传统的低秩线性适应（LoRA）推广为低维非线性适应（LoDA），提出了一种在参数效率方面具有竞争力的微调方法，本质是在A/B之间添加具有残差结构的非线性层。
  - LoDA和LoDA+方法可以提高非线性适应的表达能力，并且与LoRA相比，所需的可调参数数量几乎相同。
  - 此外，为了提高推理的计算效率，作者还提出了R-LoDA(+)和S-LoDA(+)方法，通过使用低秩或稀疏逼近来替换预训练权重矩阵，从而减少了微调过程中的计算开销

- SuperLoRA: Parameter-Efficient Unified Adaptation of Multi-Layer Attention Modules
  - 2024
  - 提出了lora系列变体的统一框架，通过控制不同的超参，可以得到不同的lora变体
  - 基于统一框架，得到了一些新的lora变体

- GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection
  - 2024
  - 从理论上证明了梯度是低秩的，可以使用类似lora的方法进行低秩分解
  - 梯度使用lora之后，优化器状态的参数量急剧减少，提高显存有效性

- Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models
  - 2024 COLING
  - 提出了大模型多任务学习的MoA架构
  - 该架构类似MoE架构，首先使用有监督数据为每个任务训练一个Lora，然后使用路由策略将所有Lora结合起来
  - 引入了领域标签来防止任务之间的互相干扰，提高多任务学习能力
  - 每个Lora模型支持新领域任务的持续迭代

- DoRA: Weight-Decomposed Low-Rank Adaptation
  - 2024
  - DoRA的目标是通过模仿全微调的学习能力，来缩小PEFT方法（如LoRA）与FT之间的准确性差距，同时避免增加推理成本
  - DoRA将预训练权重分解为两个组成部分——幅度（magnitude）和方向（direction）。这种分解灵感来源于权重归一化，它通过改善梯度的条件数来加速收敛
  - DoRA特别使用LoRA来高效地更新方向组件，因为方向组件在参数数量上占据较大比例。通过这种方式，DoRA能够在保持LoRA的参数效率的同时，提高微调的学习能力和训练稳定性
  - 通过对LoRA和FT的权重更新模式进行分析，DoRA揭示了两者在幅度和方向更新上的显著差异。DoRA的设计旨在使学习行为在经验上和数学上都类似于FT，从而提高其学习能力

- LoRA+: Efficient Low Rank Adaptation of Large Models
  - 2024
  - https://github.com/nikhil-ghosh-berkeley/loraplus
  - lora_A和lora_B使用不同的学习率

- Orthogonal Subspace Learning for Language Model Continual Learning
  - 2023
  - O-LoRA是一种用于语言模型持续学习的方法，旨在解决大型语言模型在顺序学习多个任务时遇到的灾难性遗忘
  - 正交低秩适应：O-LoRA通过在不同的（低秩）向量子空间中学习任务，并将这些子空间保持正交以最小化任务间的干扰，从而有效缓解灾难性遗忘
  - 在训练过程中，为了减轻对过去任务知识的遗忘，O-LoRA固定了先前任务的LoRA参数，并且对新任务的LoRA参数进行正交化学习

- LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models
  - 2023
  - 提出了一种更合适的lora的初始化方法，该方法可以缓解量化模型和全精度模型的差异（相比QLora）
  - 使用一种轮换优化的方法来计算lora的初始化权重