

- ACL2023关于基于检索来增强大模型的讲习班ppt
  - https://acl2023-retrieval-lm.github.io/
  - 陈丹奇组工作

- Lost in the Middle: How Language Models Use Long Contexts
  - [相关blog](https://zhuanlan.zhihu.com/p/643723202)
  - 1. 语言模型在长输入上下文的中间使用信息时会带来性能下降，并且随着输入上下文的增长，性能会进一步恶化，其更偏向于头尾两个位置
    2. 对检索到的文档进行有效的重新排序，将相关信息推向输入上下文的开头或排序列表截断，必要时返回更少的文档，这个是值得深入的方向

- Copy is All You Need
  - ICLR2023  
  - [聊聊我的AI大黄蜂：Copy is All You Need背后的故事](https://zhuanlan.zhihu.com/p/647457020)
  - 方法：通过文档片段检索的方式来进行文本生成

- In-Context Retrieval-Augmented Language Models
  - [相关blog](https://zhuanlan.zhihu.com/p/647112059)
  - 将已经生成的一些token通过检索的方式获取一些文档，并把这些文档作为prompt加在已经生成文本的前面，然后继续生成后续的token


# retrieval-augmented llm 

- [Ziya-Reader：注意力增强的训练方法](https://mp.weixin.qq.com/s/ekAyYT-Fxj5fw8GNk6Rg0g)
  