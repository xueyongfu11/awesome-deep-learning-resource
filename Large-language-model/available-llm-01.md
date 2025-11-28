[TOC]




# Open source LLM

- Qwen3 Technical Report
  - 2025.05
  - 相比如Qwen2.5，Qwen3不同点：
    - MoE模型：Qwen3取消了不同层之间的专家共享机制；为防止路由不均导致某些专家过载或“闲置”，引入全局批量负载均衡损失。即在计算门控损失时，将所有层、整个batch的token分配情况综合考虑，鼓励模型将不同token分散给尽可能多的专家。
    - 预训练预料达到了36T tokens
    - 预训练：使用多阶段训练，第一个阶段使用30T tokens、上下文4k训练，第二阶段进行知识密集型训练，第三阶段将模型上下文长度扩展到32k，然后使用高质量长上下文进行微调
    - 后训练：使用四个阶段的后训练，首先使用长链路的思维链数据进行微调，然后使用基于可验证的规则式奖励模型进行强化学习训练，第三个阶段是融合思考模式和直答模式，将思维链数据和SFT数据混合之后微调，最后一个阶段是通用性的强化学习训练。

- Qwen2.5 Technical Report
  - 2024.12
  - 相比如Qwen2，Qwen2.5不同点：
    - Qwen2.5 扩充了控制符号集：控制token从原有的3个增至22个，其中新增了2个特殊符号用于工具功能（如函数调用接口等）。其余新增的控制token用于实现角色指令、格式控制等功能。
    - 预训练预料达到了18T tokens
    - 对齐：先使用DPO训练，接着使用GRPO进行强化学习训练，使用训练好的奖励模型对一组响应进行打分。

- Qwen2 Technical Repor
  - 2024.07
  - 基础模型：
    - 使用 SwiGLU 激活函数替代传统的ReLU
    - 在注意力计算中引入 QKV 偏置
    - 采用 RMSNorm 层归一化及 Pre-Norm 架构以保证训练稳定性
    - 所有模型均应用分组查询注意力机制
    - 输入层embedding和输出层embedding权重矩阵共用，但是小模型不共用

  - 使用7T tokens，MoE模型现在7T tokens训练 Dense 模型，然后使用upcycling机制构建MoE模型，继续用4.5T tokens训练
  - 长上下文扩展：
    - 先4k长度上下文训练，后阶段使用32k上下文训练，具体操作是：
      - 调整ROPE位置编码参数，将频率基础从1e4调到1e6，以便编码32k长度时保持良好表现
      - 使用DCA（dual chunk attention）注意力机制，是一种全局注意力和局部注意相结合的方法，在长上下文下能够节省显存

    - 进一步使用YARN扩展模型上下文，该方法通过重新缩放ROPR并进行少量训练，可以支持更长上下文。

  - 混合专家模型：使用共享专家+细粒度专家策略
  - 指令微调和对齐
    - 指令微调：使用50w指令数据训练2epoch，使用余弦退火和指令衰减
    - 对齐：先使用DPO进行离线训练；训练一个奖励模型，从模型中采样输入并用奖励模型判别，构建偏好数据，继续用DPO训练，循环多次。为了减轻对齐过程中引起的对齐代价，在DPO损失的基础上增加语言模型损失。

- Qwen/Qwen1.5：Qwen Technical Report
  - 2023.09
  - https://github.com/QwenLM/Qwen
  - 基础模型主要基于llama改进：
    - embedding和output project不在共享权重
    - 使用FP32的Rope
    - 移去所有的bias，但是QKV添加了bias
    - 使用Pre-Norm & RMSNorm
    - SwiGLU激活函数
    - 使用分组查询注意GQA
    - 将嵌入矩阵和输出投影矩阵分开，不共享参数
  - 推理长度扩展：
    - NTK-Rope；LogN-Scaling；
    - window attention（lower layer对window size更加敏感，因此lower layer使用更小的window size）
  - 使用了3T tokens，使用了BPE tokenizer（基于tiktoken），定制了新tokenizer，152k 个词元，使用常用chinese token增强；数字细粒度切分
  - 训练策略：
    - SFT，为了提高泛化性，去掉了固定prompt格式的数据；使用了OpenAI的chatML-style format
    - RLHF，评估结果发现，使用了RLHF之后输出质量相比 SFT 大幅度提升
    - 在Qwen1.5中，尝试了将DPO和PPO进行结合训练
  - 语言判别工具；去重工具；低质量数据过滤（rule-based & machine-learning-based）；上采样部分数据；在预训练阶段添加部分指令数据
- Yuan
  - https://github.com/IEIT-Yuan/Yuan-2.0
- Telechat
  - https://github.com/Tele-AI/Telechat
  - 开源的预训练数据，多轮对话的数据构建方式

- Yi
  - https://github.com/01-ai/Yi
  - [Yi-34b模型训练和微调对齐介绍](https://mp.weixin.qq.com/s/cPjioApwlx711pNIPpB24w)
- MistralAI
- https://github.com/FlagAlpha/Llama2-Chinese
  - 基于llama-2做中文预训练，词表扩充，推理加速
- https://github.com/InternLM/InternLM/blob/main/README-zh-Hans.md
- https://github.com/OpenBuddy/OpenBuddy/blob/main/README.zh.md
  - 基于开源大模型微调，如llama、chatglm、baichuan，多轮对话，包含3b,7b,34b,70b
- https://github.com/FlagAI-Open/Aquila2
- MindLLM: Pre-training Lightweight Large Language Model from Scratch, Evaluations and Domain Applications
  - [blog](https://mp.weixin.qq.com/s/nPdll5L4PsPYcjguBD4_rQ)
- https://github.com/SkyworkAI/Skywork
  - benchmark评估集效果好的很大一部分原因很可能是stage-2的in-domain training

- https://github.com/mosaicml/llm-foundry
- https://github.com/RUCAIBox/LLMSurvey
- https://github.com/FlagAI-Open/FlagAI
- https://github.com/jerryjliu/llama_index
- https://github.com/lonePatient/awesome-pretrained-chinese-nlp-models
- https://github.com/ssymmetry/BBT-FinCUGE-Applications
- https://github.com/CVI-SZU/Linly
- [新加坡科技设计大学提出FLACUNA：提高LLMs问题解决能力!](https://mp.weixin.qq.com/s/QwLs9mdfyIwdTtCOwptkbA)
- [LLM综述全新出炉：51页论文带你盘点LLM领域专业化技术](https://mp.weixin.qq.com/s/bRQ7xjs4ewNJoE2Gug3lSA)
- https://github.com/MLNLP-World/Awesome-LLM
