

- [PyTorch nn.Module中的self.register_buffer()解析](https://www.jianshu.com/p/12a8207149b0)
- [pytorch中的expand（）和expand_as（）函数](https://blog.csdn.net/weixin_39504171/article/details/106090626)
  - expand()沿着指定维度扩展张量，只能对维度值为1的进行扩展
- [pytorch中repeat方法](https://blog.csdn.net/weixin_42060572/article/details/114254532)
  - 参数是复制的倍数，而不是维度
- [torch.stack()的官方解释，详解以及例子](torch.stack()的官方解释，详解以及例子)
  - 对给定的序列按照指定维度进行拼接，序列中的张量维度都必须相同，默认dim=0
- [torch.ne()参数解释及用法](https://blog.csdn.net/m0_37962192/article/details/105308012)
  - 逐个元素进行比较，不等返回1，相等返回0
- [剖析 | torch.cumsum维度详解](https://blog.csdn.net/songxiaolingbaobao/article/details/114580364)
  - 根据指定维度累计求和，返回张量维度和原始的相同
- [Pytorch阅读文档之flatten函数](https://blog.csdn.net/GhostintheCode/article/details/102530451)
  - 根据给定的起始维度，将张量进行展开
  - 如果只给一个值，则是将该维度至最后的维度全部展开
- [pytorch入门：unsqueeze](https://blog.csdn.net/ygys1234/article/details/109685299)
  - 根据指定维度，再增加一个为1的维度，相当于tensor的升维
  - 也可以使用None的方式
- [使用torch.full()、torch.full_like()创造全value的矩阵](https://blog.csdn.net/anshiquanshu/article/details/112508958)
- [torch.min or torch.max](https://blog.51cto.com/u_13977270/3395913)
  - 可以时根据指定维度查找最小值
  - 也可以输入两个相同维度的tensor，比较之后输出相对小的所有元素，shape保持不变
- [torch.where()的用法以及例子](https://blog.csdn.net/pearl8899/article/details/112408714)
  - torch.where(condition，a，b)其中
    输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出。

- [实操教程｜PyTorch实现断点继续训练](https://mp.weixin.qq.com/s/HobZsq2Mz5PGK8TWaPifsw)