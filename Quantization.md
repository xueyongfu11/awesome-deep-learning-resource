

- https://github.com/HuangOwen/Awesome-LLM-Compression

- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models
   - [LLM-QAT相关blog](https://mp.weixin.qq.com/s/zKndNym9Q7QJWlmn60HmyQ)
   - https://github.com/facebookresearch/LLM-QAT

- https://github.com/intel/neural-compressor
  - Intel开源的模型量化、稀疏、剪枝、蒸馏等技术框架

- SmoothQuant和增强型SmoothQuant
  - 增强的SmoothQuant使用了自动化确定alpha值的方法，而原始的SmoothQuant则是固定了alpha值
  - [相关blog](https://zhuanlan.zhihu.com/p/648016909)

- [pytorch profiler 性能分析 demo](https://zhuanlan.zhihu.com/p/403957917)

- https://github.com/666DZY666/micronet
  - 剪枝、量化

- [大模型量化概述](https://mp.weixin.qq.com/s/_bF6nQ6jVoj-_fAY8L5RvQ)
  - 分为量化感知训练、量化感知微调、训练后量化

## llm quantization

- FPTQ: Fine-grained Post-Training Quantization for Large Language Models
  1. 相比smoothquant，使用了指数函数把激活量化的难度转移到权重量化上
  2. 相比通道量化，使用了分组量化

- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
  1. 使用了alpha增强的将激活量化难度转移到权重量化上，同时保证矩阵乘积不变
  2. 实现时只对计算密集型算法进行了smooth量化，而对LN，relu，softmax等访存密集型算子使用fp16计算

- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
  1. 对OBQ的改进，使用了任意顺序方法进行量化，因此支持并行，提高量化速度
  2. 使用了批次更新海森矩阵的逆，引入group_size的分组超参
  3. 使用了Cholesky信息重组的方法，提高了稳定性

- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
  - W8A8量化，根据激活参数量级大小，从激活中选取outliers，使用fp16*fp16的矩阵乘法，对于激活中的其他行，使用int8*int8的量化矩阵乘法
  - 选取激活中的outliers，同时需要将权重矩阵中相应的列取出，与outliners进行矩阵相乘
- https://github.com/openppl-public/ppq

- https://github.com/NVIDIA-AI-IOT/torch2trt

## Post-training quantization

- GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS
  1. 对OBQ的改进，使用了任意顺序方法进行量化，因此支持并行，提高量化速度
  2. 使用了批次更新海森矩阵的逆，引入group_size的分组超参
  3. 使用了Cholesky信息重组的方法，提高了稳定性
  

- Up or Down? Adaptive Rounding for Post-Training Quantization
  - [blog](https://zhuanlan.zhihu.com/p/363941822)
  - 核心：对weights进行量化时，不再是round to nearest，而是自适应的量化到最近右定点值还是左定点值
  - 核心：对weights进行量化时，不再是round to nearest，而是自适应的量化到最近右定点值还是左定点值
