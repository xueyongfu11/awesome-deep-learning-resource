[TOC]



# Audio tokenizer

- SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models

  - 2023.08，ICML2024，Fudan

  - 为了评估语音令牌对于构建语音语言模型的适用性，作者建立了第一个基准测试SLMTokBench

  - 提出了SpeechTokenizer，这是一个为语音大型语言模型设计的统一语音分词，基于RVQ-VAE，它采用编码器-解码器架构，并结合残差向量量化（RVQ）技术

  - 基于SpeechTokenizer，作者构建了一个统一的语音语言模型（USLM），它结合了自回归和非自回归模型

- High Fidelity Neural Audio Compression

  - 2022.10，Meta，EnCodec

  - https://github.com/facebookresearch/encodec
  - ![](../../assets/encodec.png)