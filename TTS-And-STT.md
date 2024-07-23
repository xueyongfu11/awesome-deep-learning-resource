[TOC]



## TTS

- https://github.com/2noise/ChatTTS
  - 文本输出太长，会出现解码问题，约支持200字左右
  - 还不支持流式，需要开发
  - 只开源了推理代码
- https://github.com/suno-ai/bark
  - 与ChatTTS原理相似
- https://github.com/lucidrains/audiolm-pytorch
- https://github.com/coqui-ai/TTS
- https://github.com/gemelo-ai/vocos
- https://github.com/fishaudio/fish-speech
  - 目前只支持动漫风格，开源了微调相关代码
- 神经网络音频编解码器
  - HuBERT：Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units
    - 2021.07
  - EnCodec：High Fidelity Neural Audio Compression
    - 2022.10
- TTS工具
  - https://github.com/coqui-ai/TTS
  - https://github.com/netease-youdao/EmotiVoice
  - https://github.com/open-mmlab/Amphion
  - https://github.com/metavoiceio/metavoice-src
  - https://github.com/DigitalPhonetics/IMS-Toucan
  - https://github.com/lucidrains/naturalspeech2-pytorch
  - https://github.com/AIGC-Audio/AudioGPT
- Seed-TTS
  - https://bytedancespeech.github.io/seedtts_tech_report/
  - https://arxiv.org/abs/2406.02430
  - https://www.jiqizhixin.com/articles/2024-06-26-8
- NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models
- https://github.com/facebookresearch/audiocraft
  - 音频生成框架：EnCodec audio compressor / tokenizer，controllable music generation LM with textual and melodic conditioning
- MAGNet：Masked Audio Generation using a Single Non-Autoregressive Transformer
  - 2024.01，meta
  - 非自回归的音频生成方法
- Massively Multilingual Speech (MMS)：Scaling Speech Technology to 1,000+ Languages
  - 2023.05，meta
- CosyVoice：https://github.com/FunAudioLLM/CosyVoice
  - 2024.07，alibaba
  - https://fun-audio-llm.github.io/
  - CosyVoice consists of an autoregressive transformer to generate corresponding speech tokens for input text, an ODE-based diffusion model, flow matching, to reconstruct Mel spectrum from the generated speech tokens, and a HiFTNet based vocoder to synthesize waveforms.

### end2end

- VALL-E/VALL-E X/VALL-E R/VALL-E 2/MELLE
  - https://www.microsoft.com/en-us/research/project/vall-e-x/overview/
  - VALL-E X非官方复现模型：https://github.com/Plachtaa/VALL-E-X/blob/master/README-ZH.md
  - VALL-E：输入：文本序列转化为音素序列，3s的音频使用Encodec编码离散的token，然后生成合成的音频token，再用Encodec解码为声波
- Vits2
  - Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design 
- Bert-VITS2 2.2
  - https://github.com/PlayVoice/vits_chinese/tree/bert_vits
  - https://github.com/fishaudio/Bert-VITS2
- Vits-Fast
  - https://github.com/Plachtaa/VITS-fast-fine-tuning
- https://github.com/myshell-ai/OpenVoice
- MeloTTS 
  - https://github.com/myshell-ai/MeloTTS

### two-stage

two-stage是从文本生成音频需要经过两个阶段，第一个是使用声学模型（Acoustic model）将文本转化为梅尔频谱，第二个阶段是使用声码器（Vocoder）将梅尔频谱转化为波形。

- **Acoustic model**
  - tacotron
  - tacotron2
  - fastspeech
  - fastspeech2
- **vocoder**
  - WaveNet: A Generative Model for Raw Audio
    - 一种基于dilated casual convolution的auto-regression方法
  - FFTNet：A REAL-TIME SPEAKER-DEPENDENT NEURAL VOCODER
    - 该图是生成单步value的模型架构图，实际使用时，循环该模型架构的计算，进行auto-regression的生成
    - <img src="./assets/FFTNet" alt="img" style="zoom: 67%;" />
  - WaveRNN
  - WaveGlow：是一种Flow-based model（不同于auto-regression方法）的方法
  - SeamlessM4T：SeamlessM4T: Massively Multilingual & Multimodal Machine Translation
    - a single model that supports speech-to-speech translation, speech-to-text translation, text-to-speech translation, text-to-text translation, and automatic speech recognition for up to 100 languages
    - 1M hours的自监督学习的语音语料，40w小时的多模态有监督微调语料
    - 在next token prediction loss的基础上添加了一个辅助任务loss
    - <img src="./assets/seamlessM4T" alt="img" style="zoom: 33%;" />
  - hifigan
  - melgan
  - waveglow

## STT

- SenseVoice

  - https://github.com/FunAudioLLM/SenseVoice
  - 2024.07，alibaba
  - 非自回归，效果优于whisper
  - high-precision multilingual speech recognition, emotion recognition, and audio event detection

- Whisper：Robust Speech Recognition via Large-Scale Weak Supervision

  - 2022.12，openai

  - 模型：使用的是基于transformer的encoder-decoder结构

  - 为了支持多任务，使用不同的prompt或者说special token来表示不同的任务类型

  - trick：通过special token或者prompt来将多个任务一起建模

  - https://github.com/openai/whisper

  - <img src="./assets/whisper" alt="img" style="zoom: 33%;" />

    