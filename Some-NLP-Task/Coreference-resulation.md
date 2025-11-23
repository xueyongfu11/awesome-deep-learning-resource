[TOC]



# Repo

- https://github.com/huggingface/neuralcoref


# dataset

- CoNLL 2012
  - http://conll.cemantix.org/2012/data.html

# Blog

[一文详解自然语言处理任务之共指消解](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247490008&idx=1&sn=ef679bd95788c8a46c0a3cc2ad314330&chksm=eb500d4bdc27845dbc122533a0ae4ea3475f4d350dc1ab49efbb13938cae00a10052f16c2ec8&mpshare=1&scene=1&srcid=11245KayKJzZ89wdRmQVr7hF&sharer_sharetime=1637748146177&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A9mcDkL4Vq6vob%2FV%2Ft6c4I0%3D&pass_ticket=FVXzVd6yWxG%2B0cVb1fBXuMn3sRqbaPHr1VXt2A%2BQ1R%2FpI%2Fpfv01eV0arVDwW0wda&wx_header=0#rd)

- 介绍共指消解任务；mention出去，聚成各个簇，每个簇中的mention代表相同的指代
- 一种建模方法：token embedding, 根据embedding生成n(n-1)/2个mention embedding,方法可以是avg embedding，头尾embedding concat上avg embedding
  或者使用自注意力，对没对mention pair，判断是否是candidate mention，以及是否是相同指代，三个分数最后相加
  推理时，可以先识别出所有的mention，再分类，来减少冗余计算
- 另外一种建模方法：sequence to sequence