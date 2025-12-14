[TOC]



# mix

- End-To-End Memory Networks
  - year: 2015 NeuralIPS
  - 阅读笔记: 
    1. 给定一个memory set，通过一个矩阵A映射到一个向量空间，与query embedding进行点积计算，使用softmax计算weights
    2. 将memory set通过一个矩阵C映射到一个向量空间，使用weights加权后得到输出
    3. 将输出向量和query向量相加后，通过一个线性层softmax计算概率分布
  - code: 


# Normalization

- Layer Norm

  - 在单个样本的特征维度上进行归一化

  - 对每个样本 N 归一化

  - 具体如何做：假设输入张量形状为：(N, C, H, W)（例如图像数据，N 是 batch size，C 是通道数，H/W 是高/宽），针对每个样本，计算所有特征的均值和方差，然后进行归一化。

  - 代码实现

    ```python
    def manual_ln(x):
        # x: (..., D)
        # x.mean(dim)其中dim表示在哪个维度上进行聚合
        mean = x.mean(dim=-1, keepdim=True)  # shape: (..., 1)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5)
    ```

    

- Batch Norm

  - 在 batch 维度上进行归一化

  - 对每个通道 C 归一化

  - 具体如何做：同样假设输入为：(N, C, H, W) ，batchNorm是针对每个特征维度，即通道Chanel，计算该维度在整个batch上的均值和方差，然后进行归一化。

  - 代码实现

    ```python
    def manual_bn(x):
        # x: (N, C, H, W)
        mean = x.mean(dim=(0, 2, 3), keepdim=True)  # shape: (1, C, 1, 1)
        var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5)
    ```

    

- RMSNorm（Root Mean Square Layer Normalization）

  - 相比Layer Norm，分子去掉了减去均值部分，分母的计算使用了平方和的均值再开平方

- DeepNorm
  - 对Post-LN的改进
  - 以alpha参数来扩大残差连接，LN(alpha * x + f(x))
  - 在Xavier初始化过程中以Bata减小部分参数的初始化范围


# activation func

- Gaussian Error Linear Units（GELUs）
  - GELU，Relu的平滑版本
  - 处处可微，使用了标准正态分布的累计分布函数来近似计算

- Swish: a Self-Gated Activation Function
  - Swish
  - 处处可微，使用了beta来控制函数曲线形状
  - 函数为f(x) = x * sigmoid(betan * x)

- SwiGLU
  - 是Swish激活函数和GLU（门控线性单元）的结合
  - GLU使用sigmoid函数来控制信息的通过，GLU = sigmoid(xW+b) 矩阵点积操作 
    (xV + c)
  - SwiGLU: swish(xW+b) 矩阵点积操作 (xV + c)
  
  

- [Gaussian Error Linerar Units(GELUS)激活函数详细解读](https://mp.weixin.qq.com/s/I0fjxnNRPOkQN3wbZA0csA)
- [激活函数综述](https://www.cnblogs.com/YoungF/p/13424038.html)


# Loss

- [PolyLoss超越Focal Loss](https://mp.weixin.qq.com/s/4Zig1wXNDHEjmK1afnBw4A)
- [[损失函数]——负对数似然](https://www.jianshu.com/p/61cf7f2ac53f)
- [Contrastive Loss](https://zhuanlan.zhihu.com/p/93917636)
- [医学影像分割---Dice Loss](https://zhuanlan.zhihu.com/p/86704421)
- [从NCE loss到InfoNCE loss](https://blog.csdn.net/m0_37876745/article/details/110933812)

# 优化器

- [Adam和AdamW的区别](https://blog.csdn.net/weixin_45743001/article/details/120472616)
- [Adam,AdamW,LAMB优化器原理与代码](https://blog.csdn.net/weixin_41089007/article/details/107007221)
  - Adam使用了一阶动量矩和二阶动量矩，为每个参与赋予不同的学习率，梯度较大的参数获取的学习率较小，梯度较小的参数获取的学习率大
  - Adam收敛速度快但是存在过拟合问题，直接在loss中添加L2正则，但是会因为adam中存在自适应学习率而对使用adam优化器的模型失效，AdamW在参数更新时引入参数自身，达到同样使得参数接近于0的目的
  - LAMB是是模型在大批量数据训练时，能够维持梯度更新的精度

# 对比学习

- [张俊林：对比学习在微博内容表示的应用](https://mp.weixin.qq.com/s?__biz=MjM5MzY4NzE3MA==&mid=2247497128&idx=1&sn=0997501622c152c56fad9dd7f0add095&chksm=a691864591e60f53f594a4d39215e22a3f3adc5f9754b6dc985dd83567f595139e6eeb91c3fa&mpshare=1&scene=24&srcid=1022ReJs04EkY7dtCq20WcE6&sharer_sharetime=1634900024954&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A%2F13kBqrU8qVfi4GsLfVA6k%3D&pass_ticket=SHGOUtseKTQDhBbQUkxPd534tLY%2B6lmiRxoDIEirNdgCF3uij%2FoHBbS1BpQARUsW&wx_header=0#rd)
- [利用Contrastive Learning对抗数据噪声：对比学习在微博场景的实践](https://zhuanlan.zhihu.com/p/370782081)
- [对比学习（Contrastive Learning）:研究进展精要](https://zhuanlan.zhihu.com/p/367290573)
- [2021最新对比学习（Contrastive Learning）在各大顶会上的经典必读论文解读](https://mp.weixin.qq.com/s/9iHZqWGjJLz7Sw7JSnpmWQ)
- [对比学习（Contrastive Learning）在CV与NLP领域中的研究进展](https://mp.weixin.qq.com/s/UlV-6wBZSGIH7y2uWaAAtQ)
- [[论文随读]ACL2021对比学习论文一句话总结](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247546768&idx=3&sn=2b901db23b200f3c1be1f92270ae157e&chksm=ebb70f44dcc086521b4acc4a8e34870083a849d3b256acf776fe26bf6db5ba78a7bb384b017b&mpshare=1&scene=24&srcid=1122ge21TiW4HuHCIsJ8BaQl&sharer_sharetime=1637544192248&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A5%2F9Y%2FC0ksa2ffPCD0buO%2FM%3D&pass_ticket=FVXzVd6yWxG%2B0cVb1fBXuMn3sRqbaPHr1VXt2A%2BQ1R%2FpI%2Fpfv01eV0arVDwW0wda&wx_header=0#rd)
- [图灵奖大佬 Lecun 发表对比学习新作，比 SimCLR 更好用！](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650427247&idx=4&sn=afe5889660c7758358e25d2df78775d2&chksm=becdc93589ba40230568db838c315d86a4f9e0565199fcbb4add834770c779a924e7da42a2e3&mpshare=1&scene=24&srcid=11184RuiYMo0ZB4FFIflceCR&sharer_sharetime=1637165934870&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=AzuUGV7dgs2oDllVfn7%2BRoQ%3D&pass_ticket=3YSLQZ0%2BFGkSbSLIxeI5ld3daRcSE5x5m%2FqFag47PCWFTeogIXft8nu1uI5rJumG&wx_header=0#rd)
- [陋室快报-对比学习热文-20211115](https://mp.weixin.qq.com/s?__biz=MzIzMzYwNzY2NQ==&mid=2247487721&idx=1&sn=391155bf41b3169a9c949ed0424c6afc&chksm=e8824877dff5c161066ddbb9b96f60aebad0439b67c156889b7e16ee64621f55aa45d6003a60&mpshare=1&scene=24&srcid=11150N8snMWHjjthAnSYfkG3&sharer_sharetime=1636982174652&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A3PkxpffV5pLL9MZQQzlXT4%3D&pass_ticket=3YSLQZ0%2BFGkSbSLIxeI5ld3daRcSE5x5m%2FqFag47PCWFTeogIXft8nu1uI5rJumG&wx_header=0#rd)
- [自监督对比学习（Contrastive Learning）综述+代码](https://zhuanlan.zhihu.com/p/334732028)
- [对比学习（Contrastive Learning）综述](https://zhuanlan.zhihu.com/p/346686467)
- [对比学习（Contrastive Learning）相关进展梳理](https://zhuanlan.zhihu.com/p/141141365)



**NLP领域：**

- [CoSENT（一）：比Sentence-BERT更有效的句向量方案](https://kexue.fm/archives/8847)
- [又是Dropout两次！这次它做到了有监督任务的SOTA](https://spaces.ac.cn/archives/8496)
- [ACL 2021｜美团提出基于对比学习的文本表示模型，效果提升8%](https://mp.weixin.qq.com/s/C4KaIXO9Lp8tlqhS3b0VCw)
- [你可能不需要BERT-flow：一个线性变换媲美BERT-flow](https://kexue.fm/archives/8069)
- [丹琦女神新作：对比学习，简单到只需要Dropout两下](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247533410&idx=4&sn=30b4ebac4e5a53f7d4ab26e2830d5ce9&chksm=ebb77bb6dcc0f2a0d557fe7515741912a8503b556835b37a5c6ea7ac5ad1da4fcc981c0ab5fc&mpshare=1&scene=1&srcid=04283B09OrgnfhsQpAHFDrNv&sharer_sharetime=1619603216960&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A%2FhTSAed3D5EZ8yiQ3jReaE%3D&pass_ticket=ByIBSOIYAHACqz3WJN1dcPN%2B9hph%2BWklKYhLMYomHQ%2FGnhOMle2hsSltuKWZesaz&wx_header=0#rd)
- [基于对比学习(Contrastive Learning)的文本表示模型为什么能学到语义相似度？](https://mp.weixin.qq.com/s/mX12zl5KTmcZDHPlVl8NZg)
- [Open-AI：基于对比学习的无监督预训练](https://mp.weixin.qq.com/s/7S06f0WoXEqvvLNXm4tYjg)
- [ACL2022 | 反向预测更好？基于反向提示的小样本槽位标注方法](https://mp.weixin.qq.com/s/pONO9Ta-pW7p7x9O605t1g)

# 类别不平衡问题

- [通过互信息思想来缓解类别不平衡问题](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247509465&idx=2&sn=1a637068c358970af1ba1bfd1e4b0536&chksm=96ea7a59a19df34f0734d645ba02e33f3df5ac7be442dac471b4799569140c0d0d7d10f27fb9&mpshare=1&scene=24&srcid=0731eGYCh4BPeAvF1wsGhaBR&sharer_sharetime=1596188396962&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A%2FIx29Nq%2Fv5DJvAY21sw%2Fe4%3D&pass_ticket=IL%2BeHRprAt5yAlLjjC250jaLkeHDOYyDyV4vRbYX%2F0r7c3KJ%2FwPqrBhOiTesV9Z9&wx_header=0#rd)
- [10分钟理解Focal loss数学原理与Pytorch代码](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247505638&idx=3&sn=bf9ff2fe30a1212a67246f1e05464505&chksm=ebb7ee32dcc06724e4832a585f1a9162b1b53cd8f4c309f8aa47e5f744347d8668672b7b6371&mpshare=1&scene=24&srcid=0731PXzoa1oCrdWzwJl0O6wr&sharer_sharetime=1596188820788&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A2M6ShKYApr0upqtrXiTJGk%3D&pass_ticket=IL%2BeHRprAt5yAlLjjC250jaLkeHDOYyDyV4vRbYX%2F0r7c3KJ%2FwPqrBhOiTesV9Z9&wx_header=0#rd)

- [长尾（不均衡）分布下图像分类问题最新研究综述](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247500731&idx=3&sn=0fa43b9aab2ad749c6f70b2fa17b8cb8&chksm=ec1c2e42db6ba7548f64394b79c7884b05e36eab230e313a656ef71c3cef8dd8451b1b0b59f3&mpshare=1&scene=24&srcid=07316Bz3Yg3rvvzCHDKlb2mX&sharer_sharetime=1596190892137&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=AwAKM7VqpKYTLZWxlX0BMnI%3D&pass_ticket=IL%2BeHRprAt5yAlLjjC250jaLkeHDOYyDyV4vRbYX%2F0r7c3KJ%2FwPqrBhOiTesV9Z9&wx_header=0#rd)
- [Long-Tailed Classification (2) 长尾分布下分类问题的最新研究](https://zhuanlan.zhihu.com/p/158638078)
- [NUS颜水成等发布首篇《深度长尾学习》综述，20页pdf172篇文献阐述长尾类别深度学习进展](https://mp.weixin.qq.com/s?__biz=MzU2OTA0NzE2NA==&mid=2247571158&idx=1&sn=0f1813128cc8b4e89b3ba4148575ae19&chksm=fc8737c5cbf0bed3d0ba694a67608b151b2299f39154080d76076a4ee40495d5f53aee3b0453&mpshare=1&scene=24&srcid=10144yEywrDy5LBexJ7Rqp81&sharer_sharetime=1634202597255&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A0%2Fu79C9c4chhaVsYPhEWlg%3D&pass_ticket=X1hVh%2FzYha2Fa9G%2FZWK0bpCofPY07lt8BPBNyjf1xUWYljT%2Bk%2F9q5rZ%2F%2B4bWWFme&wx_header=0#rd)
- [再谈类别不平衡问题：调节权重与魔改Loss的对比联系](https://kexue.fm/archives/7708)
- [CB Loss：基于有效样本的类别不平衡损失](https://mp.weixin.qq.com/s?__biz=Mzg5ODAzMTkyMg==&mid=2247494771&idx=1&sn=f605465cd49ccf808e5dcd89833d06d6&chksm=c06a642ef71ded3872b5d79629b7f91b44e47ea8126e6cd4d40ab25806bebd7a55f8e36e9e48&mpshare=1&scene=24&srcid=0410uIDzSiQmZcw7Ze4KLvHX&sharer_sharetime=1618024133785&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A3%2BfhPu4WaPYypNvLLOSfCs%3D&pass_ticket=X1hVh%2FzYha2Fa9G%2FZWK0bpCofPY07lt8BPBNyjf1xUWYljT%2Bk%2F9q5rZ%2F%2B4bWWFme&wx_header=0#rd)
- [Google Brain最新论文：标签平滑何时才是有用的？](https://www.jiqizhixin.com/articles/2019-07-09-7)
- [使用一个特别设计的损失来处理类别不均衡的数据集](https://www.toutiao.com/article/6764709766112477699/)
- [5分钟理解Focal Loss与GHM——解决样本不平衡利器](https://zhuanlan.zhihu.com/p/80594704)

- [深入研究不平衡回归问题](https://mp.weixin.qq.com/s/2LzMGvS3aN2HSg2jhoDREA)

# 卷积

- [conv1d与conv2d的区别](https://renzibei.com/2020/06/30/conv1d%E4%B8%8Econv2d%E7%9A%84%E5%8C%BA%E5%88%AB/#:~:text=conv2d%E7%9A%84%E5%8D%B7%E7%A7%AF%E8%BF%90%E7%AE%97%E6%98%AF%E5%9C%A8%20%E4%BA%8C%E7%BB%B4%E7%9F%A9%E9%98%B5%20%E4%B8%AD%E6%BB%91%E5%8A%A8%EF%BC%8C%E8%80%8Cconv1d%E7%9A%84%E5%8D%B7%E7%A7%AF%E8%BF%90%E7%AE%97%E6%98%AF%E5%9C%A8,%E4%B8%80%E7%BB%B4%E5%90%91%E9%87%8F%20%E4%B8%AD%E6%BB%91%E5%8A%A8%E3%80%82%20%E5%BD%93%E6%88%91%E4%BB%AC%E4%BD%BF%E7%94%A8conv2d%E5%A4%84%E7%90%86%E5%9B%BE%E7%89%87%E6%97%B6%EF%BC%8C%E4%BA%8C%E7%BB%B4%E5%8D%B7%E7%A7%AF%E6%A0%B8%E5%9C%A8%E5%9B%BE%E7%89%87%E6%AF%8F%E4%B8%80%E9%80%9A%E9%81%93%E7%9A%84%E7%9F%A9%E9%98%B5%E4%B8%AD%E6%BB%91%E5%8A%A8%E5%8D%B7%E7%A7%AF%EF%BC%8C%E5%BD%93%E6%88%91%E4%BB%AC%E7%94%A8conv1d%E5%A4%84%E7%90%86%E5%90%91%E9%87%8F%E6%97%B6%EF%BC%8C%E4%B8%80%E7%BB%B4%E5%8D%B7%E7%A7%AF%E6%A0%B8%E5%9C%A8%E6%AF%8F%E4%B8%80%E9%80%9A%E9%81%93%E7%9A%84%E5%90%91%E9%87%8F%E4%B8%AD%E6%BB%91%E5%8A%A8%E5%8D%B7%E7%A7%AF%E3%80%82)
  - conv2d的卷积运算是在二维矩阵中滑动，而conv1d的卷积运算是在一维向量中滑动。
  - textCNN中使用的conv2d
- [为什么要用空洞卷积？](https://mp.weixin.qq.com/s?__biz=MzAxMjMwODMyMQ==&mid=2456342851&idx=4&sn=6556c82aaf414df0ee60774a0da98c25&chksm=8c2fab4dbb58225b6892f242ba5a881fd003d1fcc10c2d081a82fad38beb6a084c292cd36cf6&scene=0&xtrack=1&exportkey=Ay8QN8s%2FMncipMjJc0MBypo%3D&pass_ticket=LlL6Ad5uohnLAlqJrzan%2BA5dDM3m9%2Bnl4L%2FaTWpnfTNnifRhbExGygOrgXBzVB7b&wx_header=0#rd)
- [时间卷积网络（TCN）将取代RNN成为NLP预测领域王者](https://www.toutiao.com/article/6753489961078489612/)
- [因果卷积（causal）与扩展卷积（dilated）](https://blog.csdn.net/tonygsw/article/details/81280364)
- [一文读懂 12种卷积方法（含1x1卷积、转置卷积和深度可分离卷积等）](https://mp.weixin.qq.com/s?__biz=MzI0NDUwNzYzMg==&mid=2247485405&idx=1&sn=61077d5709b0361f57bd86e3eb2ba580&chksm=e95df142de2a78548aa27a082bd511b54bfb9f76f9f430f17079c184f37006c111917874afe2&mpshare=1&scene=24&srcid=0924obMS666axIi3wDfaGvQh&sharer_sharetime=1569337834116&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&pass_ticket=5l2GTJoNs3UnPjzRsDzXqTZBP6%2Btylp4BwIFxk3aFUwONC5l8MJz3gdjYHCbXS%2FH#rd)
- [关于 Network-in-network理解和实现](https://blog.csdn.net/m0_37561765/article/details/78874699)
- [可变形卷积从概念到实现过程](https://blog.csdn.net/LEEANG121/article/details/104234927)

# LSTM

- [人工智能 CNN VS LSTM 选择及优化](https://www.dazhuanlan.com/dbray/topics/1519290)
- [pack_padded_sequence 和 pad_packed_sequence](https://zhuanlan.zhihu.com/p/342685890)
- [人人都能看懂的GRU](https://zhuanlan.zhihu.com/p/32481747)
- [人人都能看懂的LSTM](https://zhuanlan.zhihu.com/p/32085405)
- [[深度学习] RNN对于变长序列的处理方法, 为什么RNN需要mask](https://blog.csdn.net/zwqjoy/article/details/95050794)
- [Tree-Structured LSTM介绍](https://zhuanlan.zhihu.com/p/36608614)
- [LSTM网络参数计算](https://zhuanlan.zhihu.com/p/52618361)

# softmax中e指数的作用

- 将任意实数映射为正值，这是概率的必要条件
- 放大差异：小的logits差异，概率比呈现指数级变化
- e指数梯度稳定、数值友好，与对数似然/最大熵原理的内在一致性
- 具有平移不变性
- 可微、光滑、梯度处处存在

# MLP回归网络的前向和后向传播

当然，以下是带有LaTeX格式公式的两层MLP回归网络的前向传播和反向传播过程。

### 1. 前向传播

假设输入 ( X ) 的维度是 ( (n, d) )，隐藏层的单元数是 ( h )，输出层的单元数是1（回归任务）。前向传播包括两个主要步骤：

#### 第一层（输入到隐藏层）

输入通过权重 ( W_1 ) 和偏置 ( b_1 ) 进行线性变换：

$$
 Z_1 = X W_1 + b_1
$$

然后应用激活函数（例如ReLU）：

$$
 A_1 = \text{ReLU}(Z_1)
$$

#### 第二层（隐藏层到输出层）

隐藏层的输出通过权重 ( W_2 ) 和偏置 ( b_2 ) 进行线性变换：

$$
 Z_2 = A_1 W_2 + b_2
$$

最终输出：

$$
 \hat{Y} = Z_2
$$

这里，输出 ( \hat{Y} ) 是回归任务的预测值。

### 2. 反向传播

反向传播计算每一层的梯度，以下是每一层的梯度计算公式。

#### 损失函数（均方误差）

损失函数使用均方误差（MSE）：

$$
 \mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

其中 ( y_i ) 是真实值，( \hat{y}_i ) 是预测值。

#### 计算梯度

##### 1. 输出层到隐藏层（第二层到第一层）

计算输出层的梯度：

$$
 \frac{\partial \mathcal{L}}{\partial \hat{Y}} = \frac{2}{n} (\hat{Y} - Y)
$$

然后计算输出层的权重和偏置的梯度：

- 对 ( W_2 ) 的梯度：

$$
 \frac{\partial \mathcal{L}}{\partial W_2} = A_1^T \frac{\partial \mathcal{L}}{\partial \hat{Y}}
$$

- 对 ( b_2 ) 的梯度：

$$
 \frac{\partial \mathcal{L}}{\partial b_2} = \sum \frac{\partial \mathcal{L}}{\partial \hat{Y}}
$$

##### 2. 隐藏层到输入层（第一层到输入层）

接下来，计算隐藏层的梯度：

$$
 \frac{\partial \mathcal{L}}{\partial A_1} = \frac{\partial \mathcal{L}}{\partial \hat{Y}} W_2^T
$$

然后对隐藏层的输入 ( Z_1 ) 应用激活函数的导数：

$$
 \frac{\partial \mathcal{L}}{\partial Z_1} = \frac{\partial \mathcal{L}}{\partial A_1} \cdot \text{ReLU}'(Z_1)
$$

ReLU的导数为：

$$
 \text{ReLU}'(Z_1) =
 \begin{cases}
 1 & \text{if } Z_1 > 0 \
 0 & \text{if } Z_1 \leq 0
 \end{cases}
$$

然后计算隐藏层的权重和偏置的梯度：

- 对 ( W_1 ) 的梯度：

$$
 \frac{\partial \mathcal{L}}{\partial W_1} = X^T \frac{\partial \mathcal{L}}{\partial Z_1}
$$

- 对 ( b_1 ) 的梯度：

$$
 \frac{\partial \mathcal{L}}{\partial b_1} = \sum \frac{\partial \mathcal{L}}{\partial Z_1}
$$

### 3. 代码实现

```python
import numpy as np

class MLPRegressor:
    def __init__(self, input_dim, hidden_dim):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros(1)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.X = X
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.Y_hat = self.Z2
        return self.Y_hat

    def backward(self, X, Y):
        # 计算输出层的梯度
        m = X.shape[0]
        dZ2 = (2/m) * (self.Y_hat - Y)
        dW2 = self.A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # 计算隐藏层的梯度
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0)
        
        # 更新权重和偏置
        self.W1 -= 0.01 * dW1
        self.b1 -= 0.01 * db1
        self.W2 -= 0.01 * dW2
        self.b2 -= 0.01 * db2

# 测试 MLP 回归网络
X = np.random.randn(5, 3)  # 5个样本，3个特征
Y = np.random.randn(5, 1)  # 5个样本，1个输出

# 创建模型
model = MLPRegressor(input_dim=3, hidden_dim=4)

# 前向传播
Y_hat = model.forward(X)
print("预测值 Y_hat:\n", Y_hat)

# 反向传播
model.backward(X, Y)

```



### 4. 总结

在前向传播中，使用线性变换和激活函数来计算每一层的输出。在反向传播中，通过链式法则计算每一层的梯度，并更新网络的权重和偏置。反向传播的核心步骤包括：

- 计算输出层的梯度；
- 计算隐藏层的梯度；
- 更新权重和偏置。

这些梯度计算步骤对于训练神经网络至关重要，尤其是在回归任务中。



# 算法基础

- [Noise Contrastive Estimation 前世今生——从 NCE 到 InfoNCE](https://mp.weixin.qq.com/s/QlrxIZ8wNjmcFVoB78l9ag)
- [标签平滑Label Smoothing](https://blog.csdn.net/qq_43211132/article/details/100510113)
- [一个小问题：深度学习模型如何处理大小可变的输入](https://mp.weixin.qq.com/s/jV_cqwZix6OPVhr2UxajYA)
- [损失函数为什么使用交叉熵多而不是MSE（均方差）？](https://blog.csdn.net/soga235/article/details/122094044)
- [文本分类入门（十一）特征选择方法之信息增益](http://www.blogjava.net/zhenandaci/archive/2009/03/24/261701.html)
- [互信息（Mutual Information）的介绍](https://blog.csdn.net/qq_15111861/article/details/80724278)
- [径向基函数（RBF）神经网络](https://blog.csdn.net/lin_angel/article/details/50725600)
- [预测时一定要记得model.eval()!](https://zhuanlan.zhihu.com/p/356500543)
- [bert家族中的mask机制](https://zhuanlan.zhihu.com/p/360982134)
- [谈谈由异常输入导致的 ReLU 神经元死亡的问题](https://liam.page/2018/11/30/vanishing-gradient-of-ReLU-due-to-unusual-input/)
- [PyTorch || 优化神经网络训练的17种方法](https://mp.weixin.qq.com/s?__biz=MzU1MjYzNjQwOQ==&mid=2247495246&idx=1&sn=db2fdce3a5a58db29174ab163d10778a&chksm=fbfdb4d8cc8a3dcef208aaad943cdd42616476159d8abd2d7f5a045b9db5436fb7ed151ab932&mpshare=1&scene=1&srcid=0506S7D0VBmvhlM3M83tKTi6&sharer_sharetime=1620274095789&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A%2FmmbI4pcXfMW6PviF8A5CE%3D&pass_ticket=zQDDIUhIADOvRcLFnDfeb1%2FQJUysanjrtRnVNxo8e6uhRDnY1TW%2B8mgGkSdPrrW6&wx_header=0#rd)
- [Multi-Sample Dropout](https://blog.csdn.net/weixin_37947156/article/details/95936865)
- [F-散度(F-divergence)](https://blog.csdn.net/UESTC_C2_403/article/details/75208644)
- [从Softmax到AMSoftmax(附可视化代码和实现代码)](https://zhuanlan.zhihu.com/p/97475133)
- [准确率Accuracy与损失函数Loss的关系](https://blog.csdn.net/u014421797/article/details/104689384)
- [关于LogSumExp](https://zhuanlan.zhihu.com/p/153535799)
- [稀疏矩阵存储格式总结+存储效率对比:COO,CSR,DIA,ELL,HYB](https://www.cnblogs.com/xbinworld/p/4273506.html)
- [BiLSTM-CRF学习笔记（原理和理解）](https://www.cnblogs.com/Nobody0426/p/10712835.html)
- [微调也重要：探究参数初始化、训练顺序和提前终止对结果的影响](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247503917&idx=2&sn=fe84c23bd8c42df8181042bcb715ab47&chksm=96ea0fada19d86bbc0196c5fcb4c769aec5422f85625ba7d501c34a62618a03ce24d5352ea8e&scene=0&xtrack=1&exportkey=A0vQw2ARuljx5%2BSKTALb7zc%3D&pass_ticket=2nNdCGl4e4sq9wAo0Jz1c8Wmcz0v2Ul5F4CrBxcFYeAouMQJDtkRpzhq8COdlQLP#rd)
- [看完这篇，别说你还不懂Hinton大神的胶囊网络 ](https://www.sohu.com/a/226611009_633698)
- [漫谈autoencoder：降噪自编码器/稀疏自编码器/栈式自编码器](https://blog.csdn.net/wblgers1234/article/details/81545079)
- [胶囊网络：更强的可解释性](https://zhuanlan.zhihu.com/p/264910554)
- [【深度学习笔记】熵 KL散度与交叉熵](http://www.sniper97.cn/index.php/note/deep-learning/note-deep-learning/3886/)
- [一文看懂深度学习发展史和常见26个模型](https://zhuanlan.zhihu.com/p/50967380)
- [欧氏距离与余弦距离的关系](https://blog.csdn.net/liuweiyuxiang/article/details/88736615)
- [仿射VS线性全连接 双仿射VS双线性](https://zhuanlan.zhihu.com/p/358079428)
- [综述：深度学习中的池化技术](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247539902&idx=2&sn=f56914d88067d5e4e4918498625df2f3&chksm=ec1cb147db6b3851bc9e65eaabd53bf011d7474f877689c66f64cbff4251fe64266147d0eb2f&mpshare=1&scene=24&srcid=0222KwjDsYBqvALlE7zhKRnL&sharer_sharetime=1613991579109&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A%2FSuuEp93Vy8FeU%2BiqNQrRQ%3D&pass_ticket=ahSCjZBnxTVe3IcKWMxBQVeAXXap9Se8HXejNWF3PIlQHiDsRH5Yr1%2FzLdG%2FTkZA&wx_header=0#rd)
- [硬核Softmax！yyds! (面试真题，慎点！)](https://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247488004&idx=1&sn=e323c72c5e066def9acddaa3fdc9fdac&chksm=c241f148f536785e3283b2e554107c23e6cdffb2ca74c74c861265a0a8501f55515411fa80e1&mpshare=1&scene=24&srcid=0628rbE3Rcad0WIIky6Q0Fhm&sharer_sharetime=1624884516348&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=Aw5u9UZYxqjvVUdGR3BsI10%3D&pass_ticket=ahSCjZBnxTVe3IcKWMxBQVeAXXap9Se8HXejNWF3PIlQHiDsRH5Yr1%2FzLdG%2FTkZA&wx_header=0#rd)
- [Spatial Dropout](https://blog.csdn.net/weixin_43896398/article/details/84762943)
- [【让模型更加谦虚】Adaptive Label Smoothing方法让模型结果更加鲁棒](https://mp.weixin.qq.com/s?__biz=MzA4MDExMDEyMw==&mid=2247489910&idx=2&sn=83f64bd846aaf8b0e2dbdeba5008d472&chksm=9fa86e32a8dfe7244453a3e8226265dcd543fa4f5020a687008600fe8e3e86f6fecdb2f0eaac&mpshare=1&scene=24&srcid=09287VzsiRZ1PwzBLs74bDyB&sharer_sharetime=1601253692847&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A3n6EZSkPRlGGg%2FRD08LKKw%3D&pass_ticket=FVXzVd6yWxG%2B0cVb1fBXuMn3sRqbaPHr1VXt2A%2BQ1R%2FpI%2Fpfv01eV0arVDwW0wda&wx_header=0#rd)