


# Paper

### 2023

- Alibi
  - [ALiBi介绍](https://zhuanlan.zhihu.com/p/632780188)
    - 在原始注意力矩阵上加上相对距离矩阵，q和k近，相对距离小，否则相对距离大
    - 相对距离矩阵分配一个偏执项系数，平均分布在0-1/(2^8)
  - [ALiBi位置编码的两种外推方法：内插法和NTK-ALiBi](https://zhuanlan.zhihu.com/p/657161287)
    - 使用NTK的思想：高频外推低频内插，改动的是偏执系数，第一个注意力头的偏执系数与原始偏执系数相同，最后一个注意力头的偏置系数等于内插的偏置项系统
    - ALiBi的第一个注意力头的偏置系数大，高频情况，视野小，最后一个注意力头的偏置系数小，低频情况，视野大
  
  - https://www.mosaicml.com/blog/mpt-7b 
    - 在相对短的文本上预训练，然后基于ALiBi方法在长文本上微调


- https://kaiokendev.github.io/til#extending-context-to-8k
  - 基于RoPE，scaling down frequency window

- [NBCE：使用朴素贝叶斯扩展LLM的Context处理长度](https://kexue.fm/archives/9617)
  - 基于朴素贝叶斯的独立性假设，各个片段预测token的概率是独立的
  - 通过推导得出生成token的概率由无context的概率与各个context的概率的average pooling的加权
  - 文章提出可以使用参数学习的方式来得到权重
  - 去掉无context的概率可以理解为让模型更加倾向于结合context内容而不是模型本身的知识来回答问题

- [RoPE位置编码](https://zhuanlan.zhihu.com/p/647109286)
  - 通过绝对位置表示相对位置，attention计算公式可以表示为q和k之间距离的函数，从而推导出位置编码是一个旋转矩阵
  - 具体是q和k分别乘上一个旋转矩阵（可以进行简化为两个向量的内积），再进行注意力的计算，这种attention的计算方法可以简化为q和k的乘积中间插入一个矩阵
  - RoPE的总流程：首先对每个token进行embedding，然后根据embedding计算q和k向量，q向量和k向量分别乘上一个旋转矩阵（本质是两两一组应用旋转变换），然后再计算内积得到self-attention结果


# Blog

- [浅谈LLM的长度外推](https://zhuanlan.zhihu.com/p/645770522)

- [LLM长度外推研究1——外推结果及原因分析](https://blog.csdn.net/maxsen_jn/article/details/132517811)

- [Transformer升级之路：7、长度外推性与局部注意力](https://spaces.ac.cn/archives/9431)

- https://kaiokendev.github.io/context
  - 综述性blog

- [聊聊拉长LLaMA的一些经验](https://zhuanlan.zhihu.com/p/647145964)

- [大模型位置编码及其外推性](https://mp.weixin.qq.com/s/OGP49dzhXfIudHEGHOVPcw)