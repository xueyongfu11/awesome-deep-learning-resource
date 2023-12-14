<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [类别不平衡问题](#%E7%B1%BB%E5%88%AB%E4%B8%8D%E5%B9%B3%E8%A1%A1%E9%97%AE%E9%A2%98)
- [LSTM](#lstm)
- [卷积](#%E5%8D%B7%E7%A7%AF)
- [梯度消失和梯度爆炸](#%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E5%92%8C%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8)
- [BN LN](#bn-ln)
- [鲁棒性](#%E9%B2%81%E6%A3%92%E6%80%A7)
- [Attention](#attention)
- [损失函数](#%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0)
- [优化器](#%E4%BC%98%E5%8C%96%E5%99%A8)
- [激活函数](#%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0)
- [评价指标](#%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87)
- [算法基础](#%E7%AE%97%E6%B3%95%E5%9F%BA%E7%A1%80)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



## 类别不平衡问题

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

## LSTM

- [人工智能 CNN VS LSTM 选择及优化](https://www.dazhuanlan.com/dbray/topics/1519290)
- [pack_padded_sequence 和 pad_packed_sequence](https://zhuanlan.zhihu.com/p/342685890)
- [人人都能看懂的GRU](https://zhuanlan.zhihu.com/p/32481747)
- [人人都能看懂的LSTM](https://zhuanlan.zhihu.com/p/32085405)
- [[深度学习] RNN对于变长序列的处理方法, 为什么RNN需要mask](https://blog.csdn.net/zwqjoy/article/details/95050794)
- [Tree-Structured LSTM介绍](https://zhuanlan.zhihu.com/p/36608614)
- [LSTM网络参数计算](https://zhuanlan.zhihu.com/p/52618361)
- 

## 卷积

- [conv1d与conv2d的区别](https://renzibei.com/2020/06/30/conv1d%E4%B8%8Econv2d%E7%9A%84%E5%8C%BA%E5%88%AB/#:~:text=conv2d%E7%9A%84%E5%8D%B7%E7%A7%AF%E8%BF%90%E7%AE%97%E6%98%AF%E5%9C%A8%20%E4%BA%8C%E7%BB%B4%E7%9F%A9%E9%98%B5%20%E4%B8%AD%E6%BB%91%E5%8A%A8%EF%BC%8C%E8%80%8Cconv1d%E7%9A%84%E5%8D%B7%E7%A7%AF%E8%BF%90%E7%AE%97%E6%98%AF%E5%9C%A8,%E4%B8%80%E7%BB%B4%E5%90%91%E9%87%8F%20%E4%B8%AD%E6%BB%91%E5%8A%A8%E3%80%82%20%E5%BD%93%E6%88%91%E4%BB%AC%E4%BD%BF%E7%94%A8conv2d%E5%A4%84%E7%90%86%E5%9B%BE%E7%89%87%E6%97%B6%EF%BC%8C%E4%BA%8C%E7%BB%B4%E5%8D%B7%E7%A7%AF%E6%A0%B8%E5%9C%A8%E5%9B%BE%E7%89%87%E6%AF%8F%E4%B8%80%E9%80%9A%E9%81%93%E7%9A%84%E7%9F%A9%E9%98%B5%E4%B8%AD%E6%BB%91%E5%8A%A8%E5%8D%B7%E7%A7%AF%EF%BC%8C%E5%BD%93%E6%88%91%E4%BB%AC%E7%94%A8conv1d%E5%A4%84%E7%90%86%E5%90%91%E9%87%8F%E6%97%B6%EF%BC%8C%E4%B8%80%E7%BB%B4%E5%8D%B7%E7%A7%AF%E6%A0%B8%E5%9C%A8%E6%AF%8F%E4%B8%80%E9%80%9A%E9%81%93%E7%9A%84%E5%90%91%E9%87%8F%E4%B8%AD%E6%BB%91%E5%8A%A8%E5%8D%B7%E7%A7%AF%E3%80%82)
  - conv2d的卷积运算是在二维矩阵中滑动，而conv1d的卷积运算是在一维向量中滑动。
  - textCNN中使用的conv2d
- [为什么要用空洞卷积？](https://mp.weixin.qq.com/s?__biz=MzAxMjMwODMyMQ==&mid=2456342851&idx=4&sn=6556c82aaf414df0ee60774a0da98c25&chksm=8c2fab4dbb58225b6892f242ba5a881fd003d1fcc10c2d081a82fad38beb6a084c292cd36cf6&scene=0&xtrack=1&exportkey=Ay8QN8s%2FMncipMjJc0MBypo%3D&pass_ticket=LlL6Ad5uohnLAlqJrzan%2BA5dDM3m9%2Bnl4L%2FaTWpnfTNnifRhbExGygOrgXBzVB7b&wx_header=0#rd)
- [时间卷积网络（TCN）将取代RNN成为NLP预测领域王者](https://www.toutiao.com/article/6753489961078489612/)
- [因果卷积（causal）与扩展卷积（dilated）](https://blog.csdn.net/tonygsw/article/details/81280364)
- [一文读懂 12种卷积方法（含1x1卷积、转置卷积和深度可分离卷积等）](https://mp.weixin.qq.com/s?__biz=MzI0NDUwNzYzMg==&mid=2247485405&idx=1&sn=61077d5709b0361f57bd86e3eb2ba580&chksm=e95df142de2a78548aa27a082bd511b54bfb9f76f9f430f17079c184f37006c111917874afe2&mpshare=1&scene=24&srcid=0924obMS666axIi3wDfaGvQh&sharer_sharetime=1569337834116&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&pass_ticket=5l2GTJoNs3UnPjzRsDzXqTZBP6%2Btylp4BwIFxk3aFUwONC5l8MJz3gdjYHCbXS%2FH#rd)
- [关于 Network-in-network理解和实现](https://blog.csdn.net/m0_37561765/article/details/78874699)
- [可变形卷积从概念到实现过程](https://blog.csdn.net/LEEANG121/article/details/104234927)
- 

## 梯度消失和梯度爆炸
- [sigmoid函数解决溢出_梯度消失和梯度爆炸及解决方法](https://blog.csdn.net/weixin_39612726/article/details/111391713)

## BN LN
- [nn.LayerNorm的实现及原理](https://blog.csdn.net/weixin_41978699/article/details/122778085)
- [常用的 Normalization 方法：BN、LN、IN、GN](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247495854&idx=1&sn=e2d967621307dd2c728cc3559937e6cb&source=41#wechat_redirect)
  - 理解BN、LN时，从一个模型的任何一层的特征出发，而不是从一个模型出发，所以它针对的是N*C*H*W这样的特征图
  - BN保留C通道，对N个H*W进行归一化
  - LN与N无关，对一个样本里面的C*H*W进行归一化

## 鲁棒性
- [复旦张奇：如何解决NLP中的鲁棒性问题？](https://mp.weixin.qq.com/s?__biz=MzU5ODg0MTAwMw==&mid=2247508080&idx=1&sn=3fe6c9920d93fd73c9645405ea6e95f3&chksm=febce3b4c9cb6aa2927a67bec04ca87d8cda660dbae3d25d8e381cbdbe786ef33fdf388a2973&mpshare=1&scene=24&srcid=1116RsD3z7FbUBuIFRO7emC7&sharer_sharetime=1637057596381&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=Azky0WfOZTyCjp%2BewyvxTrM%3D&pass_ticket=X1hVh%2FzYha2Fa9G%2FZWK0bpCofPY07lt8BPBNyjf1xUWYljT%2Bk%2F9q5rZ%2F%2B4bWWFme&wx_header=0#rd)


## Attention
- [这是一篇关于Attention的综述](https://zhuanlan.zhihu.com/p/148800609)
- [深度学习中Attention Mechanism详细介绍：原理、分类及应用](https://zhuanlan.zhihu.com/p/31547842)


## 损失函数
- [[损失函数]——负对数似然](https://www.jianshu.com/p/61cf7f2ac53f)
- [Contrastive Loss](https://zhuanlan.zhihu.com/p/93917636)
- [医学影像分割---Dice Loss](https://zhuanlan.zhihu.com/p/86704421)
- [从NCE loss到InfoNCE loss](https://blog.csdn.net/m0_37876745/article/details/110933812)


## 优化器
- [Adam和AdamW的区别](https://blog.csdn.net/weixin_45743001/article/details/120472616)
- [Adam,AdamW,LAMB优化器原理与代码](https://blog.csdn.net/weixin_41089007/article/details/107007221)
  - Adam使用了一阶动量矩和二阶动量矩，为每个参与赋予不同的学习率，梯度较大的参数获取的学习率较小，梯度较小的参数获取的学习率大
  - Adam收敛速度快但是存在过拟合问题，直接在loss中添加L2正则，但是会因为adam中存在自适应学习率而对使用adam优化器的模型失效，AdamW在参数更新时引入参数自身，达到同样使得参数接近于0的目的
  - LAMB是是模型在大批量数据训练时，能够维持梯度更新的精度


## 激活函数
- [Gaussian Error Linerar Units(GELUS)激活函数详细解读](https://mp.weixin.qq.com/s/I0fjxnNRPOkQN3wbZA0csA)
- [激活函数综述](https://www.cnblogs.com/YoungF/p/13424038.html)


## 评价指标
- AUC & ROC
  - [笔记︱统计评估指标AUC 详解](https://mp.weixin.qq.com/s/6PLGH3MjpQBvkxqfGP5M4A)
  - [如何计算AUC](https://mp.weixin.qq.com/s/SDGl1C4fCVrVYe7ZHNlAMw)
- [谈谈评价指标中的宏平均和微平均](https://www.cnblogs.com/robert-dlut/p/5276927.html)


## 算法基础
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
- 