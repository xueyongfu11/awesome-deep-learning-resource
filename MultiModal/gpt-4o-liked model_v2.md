[TOC]



## gpt-4o-liked models

- Moshi
  - [Code](https://github.com/kyutai-labs/moshi), [Paper: Moshi: a speech-text foundation model for real-time dialogue](https://kyutai.org/Moshi.pdf)
- Mini-Omni
  - [Code](https://github.com/gpt-omni/mini-omni), [Paper: Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming](https://arxiv.org/pdf/2408.16725)
- VITA
  - 
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

