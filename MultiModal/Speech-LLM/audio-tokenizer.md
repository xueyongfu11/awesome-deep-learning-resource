[TOC]



# Audio tokenizer

- SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models

  - 2023.08，ICML2024，Fudan

  - 为了评估语音令牌对于构建语音语言模型的适用性，作者建立了第一个基准测试SLMTokBench

  - 提出了SpeechTokenizer，这是一个为语音大型语言模型设计的统一语音分词，基于RVQ-VAE，它采用编码器-解码器架构，并结合残差向量量化（RVQ）技术

  - 基于SpeechTokenizer，作者构建了一个统一的语音语言模型（USLM），它结合了自回归和非自回归模型
- High-Fidelity Audio Compression with Improved RVQGAN

  - 2023.06，Improved RVQGAN，
  - 对比EnCodec、SoundStream等基于GVQGAN框架的音频压缩算法，论文提出的Improved RVQGAN 具有更低的码率（8kbps），并缓解了音调伪影，音高，周期性伪影以及高频建模不完善的问题。
  - 音频存在明显的周期性，传统的激活函数如Leaky ReLUs无法有效外推周期性信号，使用了BigVGAN中的Snake激活函数：$\text{snake}(x) = x + \frac{1}{\alpha} \sin^2(\alpha x)$
  - 原始的向量量化的码本利用率不高，该论文尝试了基于K-Means的码本初始化方法以及随机重启机制，虽然一定程度缓解但仍然存在部分码本未被利用的问题。论文引入了两个tricks：第一个是因子分解解耦lookup和embedding，使用低维的码本lookup，embedding则使用正常维度的码本。第二个是使用L2归一化的码本，可以提高稳定性和质量。
  - 不同于SoundStream为了动态比率对每个样本都采样不同的量化器数量，论文只对以0.5概率采样到的样本执行动态比率量化
  - MS-STFT判别器通过在多个时间-频率尺度上对音频的复数 STFT 进行判别，联合刻画音频的细节纹理与长程结构。 它由多个结构相同的子判别器组成，分别处理不同窗口长度的 STFT 特征，利用带时间维空洞卷积的 2D CNN 提取多尺度时频模式。该方法能显著增强对高频细节和瞬态结构的感知能力，从而提升生成音频的真实度与清晰度。如下图：

    ![](../../assets/MS-STFT.png)
  - 损失设置

    - 联合使用Mel重构损失和多尺度STFT频谱损失，基于L1计算Loss
    - 基于 HingeGAN 的对抗损失，结合多周期波形判别器与多频带多尺度 STFT 判别器，从时域与频域同时约束生成音频的真实性；同时引入 L1 特征匹配损失，对齐真实与生成样本在判别器中间特征层的分布
    - 码本学习的loss：码本损失+commitment损失
    - 启发式的损失加权
- High Fidelity Neural Audio Compression

  - 2022.10，Meta，EnCodec
  - https://github.com/facebookresearch/encodec
  - 该方法的encoder使用了卷积+LSTM网络+1D卷积，decoder使用了1D卷积+LSTM网络+时序卷积
  - ![](../../assets/encodec.png)