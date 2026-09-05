[TOC]

# inference tools

- tensorRT https://github.com/shouxieai/tensorRT_Pro

- https://github.com/NVIDIA/FasterTransformer

- 神经网络可视化工具 https://github.com/lutzroeder/Netron

- https://github.com/htqin/awesome-model-quantization
  - 模型量化paper整理

- https://github.com/airaria/TextPruner

- Asset eXchange models
  - https://developer.ibm.com/articles/introduction-to-the-model-asset-exchange-on-ibm-developer/

# 模型并发推理

- https://github.com/ShannonAI/service-streamer

- https://github.com/thuwyh/InferLight

- [Flask+Gunicorn的web服务](https://zhuanlan.zhihu.com/p/460235764)
  - Gunicorn采用多进程的方式处理请求，通过辅助工作进程的方式提高应用的并发能力


# PTLM显存分析

- [分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)
- [PyTorch显存分配原理——以BERT为例](https://zhuanlan.zhihu.com/p/527143823)
- [Self Attention 固定激活值显存分析与优化及PyTorch实现](https://zhuanlan.zhihu.com/p/445016136)
- [BertLarge 中间激活值分析](https://zhuanlan.zhihu.com/p/424180513)
- [[实践] Sequence Parallel](https://zhuanlan.zhihu.com/p/626553071)
- https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
- torchinfo
  - 使用 torchinfo 可以查看模型共有多少参数，比直接print(model)的信息更全，可视化效果更好
  - 如果指定输入向量的shape，可以得到每一层输出的shape，总共占用多少显存

# knowledge distillation

- https://github.com/HoyTta0/KnowledgeDistillation

- https://github.com/airaria/TextBrewer

# Blog

## 汇总

- [轻量级模型设计与部署总结](https://mp.weixin.qq.com/s/a9yNr6hFVocPFe-DEmRiXg)
- [模型部署工具箱 MMDeploy使用介绍](https://mp.weixin.qq.com/s/l494lru5RYEovkq16E1Rpg)


### pytorch转onnx

- [在 C++ 平台上部署 PyTorch 模型流程+踩坑实录](https://mp.weixin.qq.com/s/0QVS71W68qAxqi3GbBQc8w)
- [Pytorch转onnx详解](https://mp.weixin.qq.com/s/LKBCAuxJDEJ6Rs3Kzx3zQw)
- [从pytorch转换到onnx](https://zhuanlan.zhihu.com/p/422290231)
- [学懂 ONNX，PyTorch 模型部署再也不怕！](https://developer.aliyun.com/article/914229)
- [pytorch导出torchscript记录](https://blog.csdn.net/zysss_/article/details/128222610)


## 模型加速

- [【CUDA编程】Faster Transformer v1.0 源码详解](https://mp.weixin.qq.com/s/VzDCfKUMZBB_cO7uLjOUig)

- [推荐几个不错的CUDA入门教程](https://godweiyang.com/2021/01/25/cuda-reading/)




## 官网

- https://pytorch.org/docs/master/onnx.html
  - pytorch转onnx
- https://onnxruntime.ai/
  - onnxruntime推理引擎
- https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/advanced/model_to_onnx_cn.html
  - paddle2onnx

# 基础内容

## fp32训练和fp16训练

这里的计算展示了在混合精度训练中显存的存储开销情况。我们可以分别解释FP32和FP16的显存使用。

**FP32（单精度浮动）存储开销：**

FP32使用4字节（32位）来表示每个数值。计算公式如下：

- **参数**：每个参数（例如模型的权重）占用4字节。假设网络的每个参数都是用FP32表示的，那么参数的总显存开销为4字节。
- **梯度**：在反向传播时，我们需要存储每个参数的梯度。梯度通常也使用FP32表示，所以梯度的显存开销为4字节。
- **优化器**：例如在Adam优化器中，需要存储一阶和二阶矩估计。通常这两个值也用FP32表示，所以每个优化器状态的开销为8字节（因为Adam会存储两个浮动值：一阶矩和二阶矩）。

因此，FP32训练中每个参数、梯度和优化器的总显存使用为：
$$
 \text{总显存} = 4 (\text{参数}) + 4 (\text{梯度}) + 8 (\text{优化器}) = 16 \text{字节}
$$
**FP16（半精度浮动）存储开销：**

FP16使用2字节（16位）来表示每个数值。我们来看混合精度训练中每部分的存储：

- **参数**：在FP16训练中，参数通常使用FP16表示，因此每个参数的存储开销是2字节。
- **梯度**：梯度也会被转换为FP16表示，因此每个梯度的存储开销是2字节。
- **参数的FP32备份**：为了避免由于FP16精度不足导致的数值不稳定，训练时通常会保留参数的FP32备份。这意味着每个参数在训练过程中需要额外的4字节来保存其FP32备份。因此，参数的存储开销是2字节（FP16）+ 4字节（FP32备份）= 6字节。
- **优化器**：与FP32相同，优化器的状态（如一阶和二阶矩估计）仍然使用FP32表示，因此每个优化器状态占用8字节。

因此，FP16训练中每个参数、梯度和优化器的总显存使用为：
$$
 \text{总显存} = 2 (\text{参数}) + 2 (\text{梯度}) + 4 (\text{参数的FP32备份}) + 8 (\text{优化器}) = 16 \text{字节}
$$
**总结**：

- 在FP32模式下，显存开销为16字节（参数4字节 + 梯度4字节 + 优化器8字节）。
- 在FP16模式下，显存开销仍然是16字节，但参数的存储开销被减半（2字节），同时由于需要保留参数的FP32备份，导致参数部分的开销为6字节，而优化器部分仍然是8字节。















