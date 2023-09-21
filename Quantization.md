

- https://github.com/HuangOwen/Awesome-LLM-Compression

- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models
   - [LLM-QAT相关blog](https://mp.weixin.qq.com/s/zKndNym9Q7QJWlmn60HmyQ)
   - https://github.com/facebookresearch/LLM-QAT

- https://github.com/intel/neural-compressor
  - Intel开源的模型量化、稀疏、剪枝、蒸馏等技术框架

- SmoothQuant和增强型SmoothQuant
  - 增强的SmoothQuant使用了自动化确定alpha值的方法，而原始的SmoothQuant则是固定了alpha值
  - [相关blog](https://zhuanlan.zhihu.com/p/648016909)

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
