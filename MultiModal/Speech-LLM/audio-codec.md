[TOC]



## Resource

- SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models
  - 2023.08，ICML2024，Fudan
  - 为了评估语音令牌对于构建语音语言模型的适用性，作者建立了第一个基准测试SLMTokBench
  - 提出了SpeechTokenizer，这是一个为语音大型语言模型设计的统一语音分词，基于RVQ-VAE，它采用编码器-解码器架构，并结合残差向量量化（RVQ）技术
  - 基于SpeechTokenizer，作者构建了一个统一的语音语言模型（USLM），它结合了自回归和非自回归模型
- https://github.com/LAION-AI/CLAP
  - 通过text-audio对比学习的方式进行audio的表示学习
- High Fidelity Neural Audio Compression
  - 2022.10，Meta，EnCodec
  - https://github.com/facebookresearch/encodec
- SoundStream: An End-to-End Neural Audio Codec
  - 2021.07，
  - VQ面临一个问题，如果要更加准确的表征音频片段，那就是它需要一个庞大的码本(codebook)来进行工作
  - 本工作提出了RVQ，RVQ是VQ的一个变种，它在多级量化过程中被使用。
  - 在第一级，使用标准的VQ过程来量化信号，然后计算出原始信号与第一级量化后的信号之间的残差，对这个残差再进行一次或多次量化，以进一步减小量化误差，每一级都会产生一个新的残差，然后对新的残差继续量化，这样做可以逐步细化量化结果，提高最终的重建质量。
- Neural Discrete Representation Learning
  - 2017，VQ-VAE，
  - 将输入x编码为离散的向量，计算离散向量，映射到离散潜在嵌入空间e中的最近向量，映射结果输入到decoder解码出x'
  - 模型训练的损失：
    - 向量量化损失：使用l2范数来计算编码器输出和最近嵌入向量之间的距离，并通过梯度下降来最小化这个距离，在反向传播中更新离散潜在嵌入空间e；
    - 重建损失，即输入和输出的均方误差损失；
    - 为了确保编码器的输出不会无限制地增长，并且嵌入空间的体积保持稳定，引入了承诺损失（commitment loss），这有助于模型更坚定地选择特定的嵌入向量，类似正则项
  - 参数更新：编码器参数更新依赖于重建损失和承诺损失，解码器参数仅依赖于重建损失进行更新，离散潜在嵌入空间的更新主要依赖向量量化损失