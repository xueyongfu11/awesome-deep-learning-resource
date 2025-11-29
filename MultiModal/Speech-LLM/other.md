[TOC]

# Benchmark

- Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities

  - 2025.03；李宏毅团队

  - codes: https://github.com/DanielLin94144/Full-Duplex-Bench

  - 提供了全双工语音大模型对话交互能力评估的benchmark，从以下4个维度进行评估：pause handling（用户停顿时保持沉默）、backchanneling（用户说话时模型给与及时的肯定信号）、smooth turn taking（模型及时接管对话）、user interruption

  - Full-Duplex-Bench 的自动评估流程是：首先构造包含停顿、打断和可回声时机的用户语音流作为输入，将其送入被测的 full-duplex 语音对话模型进行实时交互，得到系统输出语音；随后使用 ASR 将系统语音转写为带时间戳的文本，与用户语音在同一时间轴上做精确对齐；在此基础上自动检测系统是否发生抢话（takeover）、是否产生 backchannel、接话延迟是否合理、以及在用户打断时的响应行为；最后根据预定义的量化指标（如 Takeover Rate、Backchannel 频率与时间分布 JSD、Turn-taking 延迟、打断响应质量等）进行自动统计，从而实现对 full-duplex 对话行为的标准化、可复现评估。

# 级联式speech2speech

- HuggingGPT：整合了各个任务的pipline
- AudioGPT
- https://github.com/wwbin2017/bailing?tab=readme-ov-file
  - 百聆 是一个类似GPT-4o的语音对话机器人，通过ASR+LLM+TTS实现，集成DeepSeek R1等优秀大模型，时延低至800ms，Mac等低配置也可运行，支持打断

# 音频多模态

- Qwen-Audio
  - https://github.com/QwenLM/Qwen-Audio
- LLaSM: Large Language and Speech Model
  - 2023.08，Yi01
- https://github.com/ga642381/speech-trident
  - ⭐⭐⭐
- https://github.com/QwenLM/Qwen2-Audio

# 音频理解

Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities

- 2025.03

# Other

- fastrtc: https://github.com/freddyaboulton/fastrtc
  - 原来的gradio-webrtc
  - 它是一个用于实现实时通信（Real-Time Communication）的工具，支持音频和视频流的实时处理