

# inference tools

- tensorRT https://github.com/shouxieai/tensorRT_Pro

- https://github.com/NVIDIA/FasterTransformer

- 神经网络可视化工具 https://github.com/lutzroeder/Netron

- https://github.com/htqin/awesome-model-quantization
  - 模型量化paper整理

- https://github.com/airaria/TextPruner

- Asset eXchange models
  - https://developer.ibm.com/articles/introduction-to-the-model-asset-exchange-on-ibm-developer/



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