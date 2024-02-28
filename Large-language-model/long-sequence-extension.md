

- Efficient Streaming Language Models with Attention Sinks

- Scaling Laws of RoPE-based Extrapolation

- Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading

- Randomized Positional Encodings Boost Length Generalization of Transformers
  - 训练时从[0,L]中随机获取递增的位置编码，而训练的长度N远小于L，推理时使用正常的位置编码

- logn attention scale
  - 使用logn attention scale，目的时解决当预测长度远远大于训练时的最大长度，attention的值变得比较平缓，有助于解决外推性问题
  - 窗口注意力和logn attention scale的结合

- [无限外推的ReRope](https://kexue.fm/archives/9708)

- Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation
  - [ALiBi介绍](https://zhuanlan.zhihu.com/p/632780188)
    - 在原始注意力矩阵上加上相对距离矩阵，q和k近，相对距离小，否则相对距离大
    - 相对距离矩阵分配一个偏执项系数，平均分布在0-1/(2^8)
  - [ALiBi位置编码的两种外推方法：内插法和NTK-ALiBi](https://zhuanlan.zhihu.com/p/657161287)
    - 内插法：直接用偏置系数除以长度扩展倍数
    - 使用NTK的思想：高频外推低频内插，改动的是偏执系数，第一个注意力头的偏执系数与原始偏执系数相同，最后一个注意力头的偏置系数等于内插的偏置项系统
    - ALiBi的第一个注意力头的偏置系数大，高频情况，视野小，最后一个注意力头的偏置系数小，低频情况，视野大
  
  - https://www.mosaicml.com/blog/mpt-7b 
    - 在相对短的文本上预训练，然后基于ALiBi方法在长文本上微调

- [NBCE：使用朴素贝叶斯扩展LLM的Context处理长度](https://kexue.fm/archives/9617)
  - 基于朴素贝叶斯的独立性假设，各个片段预测token的概率是独立的
  - 通过推导得出生成token的概率由无context的概率与各个context的概率的average pooling的加权
  - 文章提出可以使用参数学习的方式来得到权重
  - 去掉无context的概率可以理解为让模型更加倾向于结合context内容而不是模型本身的知识来回答问题

- Dynamic Scaled RoPE &&  Dynamic NTK-Aware Scaled RoPE
  - 相比静态的RoPE的线性内插和NTK-Aware Scaled RoPE，动态的方法是scale随着序列长度的增加而增加

- NTK-Aware Scaled RoPE
  - https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/?rdt=61154
  - 基于NTK的思想改进的RoPE
  - 实现时是将base乘上一个因子a=scale*(d/d-2)，当i较小时，近似等于未插值情况，当i=d/2-1时，近似等于插值情况
  - 实现：https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=fd650d79

- RoPE的线性内插
  - https://kaiokendev.github.io/context
  - 直接对位置index除上一个长度扩展倍数
  - LongChat模型：基于该方法的基础上用长文本数据进一步微调：https://lmsys.org/blog/2023-06-29-longchat/

- [RoPE位置编码](https://zhuanlan.zhihu.com/p/647109286)
  - 通过绝对位置表示相对位置，attention计算公式可以表示为q和k之间距离的函数，从而推导出位置编码是一个旋转矩阵
  - 具体是q和k分别乘上一个旋转矩阵（可以进行简化为两个向量的内积），再进行注意力的计算，这种attention的计算方法可以简化为q和k的乘积中间插入一个矩阵
  - RoPE的总流程：首先对每个token进行embedding，然后根据embedding计算q和k向量，q向量和k向量分别乘上一个旋转矩阵（本质是两两一组应用旋转变换），然后再计算内积得到self-attention结果

- [浅谈LLM的长度外推](https://zhuanlan.zhihu.com/p/645770522)

- [LLM长度外推研究1——外推结果及原因分析](https://blog.csdn.net/maxsen_jn/article/details/132517811)
  - 通过分析attention score的值在训练长度以内以及超过训练长度的分布的不同，提出在训练长度以内，attention score时被bound住的
  - 解决办法时压制住attention score，核心思想是使得权重矩阵跟位置信息配合起来

- [Transformer升级之路：7、长度外推性与局部注意力](https://spaces.ac.cn/archives/9431)
  - 函数式位置编码外推行不好的原因是sin和cos不具有光滑性质，属于震荡型函数；另外一方面是因为更长的长度分散了注意力
  - 推理时使用窗口注意力，即token只和最近窗口的token计算注意力，窗口大小一般使用训练时的长度