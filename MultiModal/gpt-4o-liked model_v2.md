[TOC]



## gpt-4o-liked models

- Mini-Omni
  - [Code](https://github.com/gpt-omni/mini-omni), [Paper: Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming](https://arxiv.org/pdf/2408.16725)
- LLaMA-Omni
  - [Code](https://github.com/ictnlp/LLaMA-Omni), [Paper: LLaMA-Omni: Seamless Speech Interaction with Large Language Models](https://arxiv.org/pdf/2409.06666)

### VITA

- 2024.08，[Code](https://github.com/VITA-MLLM/VITA), [Paper: VITA: Towards Open-Source Interactive Omni Multimodal LLM](https://arxiv.org/pdf/2408.05211)
- 大模型指令微调
  - 模型使用了Mixtral 8x7B，为了对中文有更好的支持，扩充了词表，并使用了5M的合成双语数据进行指令微调
- 多模态对齐
  - 该阶段主要是为了对齐文本模态和其他模态
  - 视觉模态对齐
    - Visual Encoder使用InternViT-300M-448px，来对image、video进行encoding
    - 对齐过程中只训练连接器部分
    - 该阶段使用了大量的多模态数据集，query中不包含语音模态数据
  - 音频模态对齐
    - Audio Encoder使用了梅尔普特征，然后使用4×CNN进行下采样，接着使用24层的transformer，最后使用两层mlp最为连接器
    - 对齐过程使用了ASR任务和音频描述任务
- 多模态指令微调过程
  - 数据源与对齐阶段的数据相同，做了一些改变。将一半的问题用GPT-SoVITS工具转成语音作为输入，另外一半的问题作为纯文本直接作为输入
  - 为了避免不同类型的数据引起的冲突，针对不同模态类型的输入数据设置了不同的system prompt
  - Noisy Audio Construction: 从现有的多模态或者多模态QA数据中采样474K的答案，这些答案的长度分布与有效的问题近似，然后将答案用TTS工具转成语音，构建出拒答语音数据，也就是噪声语音数据
  - 训练过程：为了对不同模态类型的输入进行更好的交互，在训练时，针对不同的输入模态类型，在答案的开始位置添加相应的特殊token，语音query添加<1>，噪声语音query添加<2>，纯文本query<3>
  - 模型无法输出语音，需要TTS转成语音
- 双工交互
  - 使用了VAD模型SileroVAD，确定音频内容是否构成人类的声音
  - 噪声过滤：使用了状态token <2> 来判断输入音频是否时有效的query
  - 声音打断交互：同时部署两个模型，生成模型用来回复用户的query，而监测模型同时监测环境声音，当监测到有效的query时，生成模型终止回复，并将上下文提供给监测模型，监测模型开始对新query进行回复，而生成模型对环境生成进行监测，二者完成了身份的互换。

### Moshi

- [Code](https://github.com/kyutai-labs/moshi), [Paper: Moshi: a speech-text foundation model for real-time dialogue](https://kyutai.org/Moshi.pdf)
- 整体概述
  - Moshi是一个多流语音到语音Transformer模型，能够实现与用户的全双工语音对话。它基于Helium构建，是一个从零开始创建的文本大型语言模型（LLM）
  - 引入了“Inner Monologue”，使得在训练和推断过程，该过程中同时处理文本和音频标记，使模型可以利用文本模态的知识，但仍保持为一个语音到语音系统
  - 设计为多流架构，允许模型同时进行说话和聆听，不需要明确地控制讲话轮次，以支持实时对话
  - 提出了Mimi，一种神经音频编码器，通过残差向量量化（RVQ）和知识蒸馏技术将语义和声学信息整合进单一tokenizer中，以高效且高质量地捕捉用户输入音频并输出声音
- Helium大语言模型
  - Helium是基于Transformer的自回归语言模型，做了如下改进：在attention block，feed-forward block以及output layer之前使用了RMS Norm；使用RoPE旋转位置编码；在feed-forward block中，使用门控线性单元GLU；tokenizer使用了unigram model（from sentencepiece）
  - 预训练数据只保留了英文数据，做了数据去重和质量过滤
- Audio Tokenization
  - 背景：
    - 声学token表征了音频细节信息，可以用来重建高质量的音频。声学token一般有下面几个用处：与text结合生成语音（即ASR），与text结合生成音乐，与语义token结合进行无条件的音频生成。
    - 声学token无法重建高质量的音频，而是与人类语言内容相关，因此与语言模型的token更加相似。语义token主流是使用双向transfomer模型建模，无法支持流式任务。
  - 基于背景介绍中存在的问题，文章提出了Mimi音频编解码器，核心是将支持语义token编码的双向模型蒸馏到单向因果模型中。
  - Mimi编码器使用了时序卷积网络对输入音频进行encoding，得到隐式表征。接着使用残差向量量化RVQ将隐式表征离散化为codebook中的向量（RVQ具体方法可以学习相关论文）
  - 在量化前和量化后添加了8层的casual transformer层，为了稳定训练，使用了LayerScalue初始化方法
  - 如何蒸馏？论文将WavLM语义信息蒸馏到了RVQ的第一层中，方法是计算RVQ首次量化输出的embedding和WavLM的embedding的cosine举例，作为训练loss的一部分。因此第一个量化层得到的是语义token，而声学token则在RVQ的其他量化层进行了保留。
  - ![image-20240928174746137](../assets/Mini.png)























