[TOC]



## 全模态大模型（gpt-4o-like model）

- AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling
  - 2024.02, Fudan
  -  将所有模态转化为离散的token，然后进行模型训练，支持各种模态的输入和输出。
  - 各个模态的离散token生成使用了现成的tokenizer和de-tokenizer
  - 提出了AnyInstruct-108k多模态数据集，包含了各种模态复杂交织的多轮对话数据集
  - https://junzhan2000.github.io/AnyGPT.github.io/
  - https://ai-scholar.tech/zh/articles/large-language-models/anygpt
- SpeechGPT-Gen: Scaling Chain-of-Information Speech Generation
  - 2024.01
- OneLLM: One Framework to Align All Modalities with Language
  - 2023.12, CVPR2024
  - https://onellm.csuhan.com/
- Generative Multimodal Models are In-Context Learners
  - 2023.12, CVPR2024
  - https://baaivision.github.io/emu2/
- Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action
  - 2023.12，Allen AI
  - 可以同时理解和生成图片、文本、声音和动作的自回归的多模态模型
  - 为了克服训练中的稳定性和可扩展性问题，做了关键的架构更改，包括在perceiver resampler上使用2D旋转嵌入、QK归一化和缩放余弦注意力机制
  - https://zhuanlan.zhihu.com/p/682141007
- AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model
  - 2023.09,  FAIR & Meta
  - 使用多模态数据进行预训练，用轻量级adapter将各个模态投射到语言模型的表示空间（每个模态单独训练，然后进行模态之间的对齐）
  - 训练时freeze语言模型，从而达到多模态训练更快的收敛
  - 用到的不同模态的encoder：Our modality-specific encoder zoo includes: CLIP ViT-L, ViT-G, DinoV2 (image), CLAP (audio), IMU2CLIP (IMU motion sensor), and Intervideo (video)
- NExT-GPT: Any-to-Any Multimodal Large Language Model
  - 2023.09,
  - https://next-gpt.github.io/
- Meta-Transformer: A Unified Framework for Multimodal Learning
  - 2023.07
- IMAGEBIND: One Embedding Space To Bind Them All
  - 2023.05，Meta
  - 将其他模态如video或者audio对齐到image模态，就可以得到一个全模态对齐到同一个空间的目标
  - 使用CLIP进行初始化时，在将image对齐到其他模态时，也隐含了将text对齐到其他模态
  - 主要应用在图片检索，音频分类等
- SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities
  - 2023.05，Fudan
  - https://github.com/0nutation/SpeechGPT/tree/main/speechgpt
  - https://0nutation.github.io/SpeechGPT.github.io/
  - stage1：模态适应预训练：只使用语音进行next token prediction训练（加入文本混合训练可能更好吧）
  - stage2：只是用了audio-text pair数据，输入是语音，输出是文本，训练时只计算text的loss（prefix部分除外）
  - stage3：使用lora训练，训练数据包含多种类型，text2speech、text2text、speech2text、speech2speech（感觉可以重点优化这个阶段）
- CoDi: Any-to-Any Generation via Composable Diffusion
  - 2023.05,
  - https://codi-gen.github.io/
- VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset
  - 2023.04
- 李宏毅gpt-4o模型解读
  - https://www.youtube.com/watch?v=CgQ3lUOpXgc
    - PPT：https://drive.google.com/file/d/19bx4c60v5xudRvLiK0RnBQBAXMm0Dx70/view?usp=sharing
    - 混合编码器+语者自动分辨标记，混合编码器可以看
    - decoder是基于speech unit转化成声音波形
    - 纯speech2speech训练的可能性比较低，因为纯语音训练很难学习到足够的知识，可以参考两个paper，Toward Joint Language Modeling for Speech Units and Text；SPIRIT-LM: Interleaved Spoken and Written Language Model
    - 推理时可能是纯speech2speech
    - 如何验证？测试文字版gpt-4o和语音版的推理能力的差异，如果是纯speech2speech的方式，推理能力按理说比较弱
    - 如何同时做到听说看，参考Dialogue GSLM，将各个模态信息以类似的方式进行融合
- 知乎问答
  - https://www.zhihu.com/question/655916579/answer/3497812976
  - https://zhihu.com/question/655951646/answer/3499150933
  - https://zhuanlan.zhihu.com/p/700092179
  - https://zhuanlan.zhihu.com/p/697582130
    - 提及了gpt-4o可能的数据生产方式
  - https://zhuanlan.zhihu.com/p/698552507



[**点击查看我的更多AI学习笔记github**](https://github.com/xueyongfu11/awesome-deep-learning-resource)