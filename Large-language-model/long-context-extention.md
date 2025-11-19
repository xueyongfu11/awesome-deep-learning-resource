[TOC]



# 长度外推

- Efficient Streaming Language Models with Attention Sinks
- Scaling Laws of RoPE-based Extrapolation
- Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading
- Randomized Positional Encodings Boost Length Generalization of Transformers
  - 训练时从[0,L]中随机获取递增的位置编码，而训练的长度N远小于L，推理时使用正常的位置编码
- logn attention scale
  - 使用logn attention scale，目的时解决当预测长度远远大于训练时的最大长度，attention的值变得比较平缓，有助于解决外推性问题
  - 窗口注意力和logn attention scale的结合
- [无限外推的ReRope](https://kexue.fm/archives/9708)
- Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation
  - [ALiBi介绍](https://zhuanlan.zhihu.com/p/632780188)
    - 在原始注意力矩阵上加上相对距离矩阵，q和k近，相对距离小，否则相对距离大
    - 相对距离矩阵分配一个偏执项系数，平均分布在0-1/(2^8)
  - [ALiBi位置编码的两种外推方法：内插法和NTK-ALiBi](https://zhuanlan.zhihu.com/p/657161287)
    - 内插法：直接用偏置系数除以长度扩展倍数
    - 使用NTK的思想：高频外推低频内插，改动的是偏执系数，第一个注意力头的偏执系数与原始偏执系数相同，最后一个注意力头的偏置系数等于内插的偏置项系统
    - ALiBi的第一个注意力头的偏置系数大，高频情况，视野小，最后一个注意力头的偏置系数小，低频情况，视野大
  
  - https://www.mosaicml.com/blog/mpt-7b 
    - 在相对短的文本上预训练，然后基于ALiBi方法在长文本上微调
- [NBCE：使用朴素贝叶斯扩展LLM的Context处理长度](https://kexue.fm/archives/9617)
  - 基于朴素贝叶斯的独立性假设，各个片段预测token的概率是独立的
  - 通过推导得出生成token的概率由无context的概率与各个context的概率的average pooling的加权
  - 文章提出可以使用参数学习的方式来得到权重
  - 去掉无context的概率可以理解为让模型更加倾向于结合context内容而不是模型本身的知识来回答问题
- Dynamic Scaled RoPE &&  Dynamic NTK-Aware Scaled RoPE
  - 相比静态的RoPE的线性内插和NTK-Aware Scaled RoPE，动态的方法是scale随着序列长度的增加而增加
- NTK-Aware Scaled RoPE
  - https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/?rdt=61154
  - 基于NTK的思想改进的RoPE
  - 实现时是将base长度（比如10000）乘上一个因子a=scale*(d/d-2)，d是向量维度，当位置 i 较小时，近似等于未插值情况，当i=d/2-1时，近似等于插值情况
  
    scale是缩放因子，$\alpha = \max\left(1, \frac{L_{\text{inference}}}{L_{\text{train}}} \right)$，每次前向传播时都要重新计算base
  - 实现：https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=fd650d79
- RoPE的线性内插
  - https://kaiokendev.github.io/context
  - 直接对位置index除上一个长度扩展倍数
  - LongChat模型：基于该方法的基础上用长文本数据进一步微调：https://lmsys.org/blog/2023-06-29-longchat/
- [浅谈LLM的长度外推](https://zhuanlan.zhihu.com/p/645770522)
- [LLM长度外推研究1——外推结果及原因分析](https://blog.csdn.net/maxsen_jn/article/details/132517811)
  - 通过分析attention score的值在训练长度以内以及超过训练长度的分布的不同，提出在训练长度以内，attention score时被bound住的
  - 解决办法时压制住attention score，核心思想是使得权重矩阵跟位置信息配合起来
- [Transformer升级之路：7、长度外推性与局部注意力](https://spaces.ac.cn/archives/9431)
  - 函数式位置编码外推行不好的原因是sin和cos不具有光滑性质，属于震荡型函数；另外一方面是因为更长的长度分散了注意力
  - 推理时使用窗口注意力，即token只和最近窗口的token计算注意力，窗口大小一般使用训练时的长度

# 位置编码

### 正弦位置编码（Sinusoidal Positional Encoding）

**特点:**

- **固定、不可学习**：公式预先定义，训练过程中不更新。
- **显式包含绝对位置信息**。
- 具有**相对位置可推导性**（通过三角恒等式，两个位置的编码之差可表达相对距离）。

**公式：**

对于位置 ( pos ) 和维度 ( i )（从 0 开始），编码为：

$$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right) \\ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right) $$

其中 ( d ) 是嵌入维度。每个位置是一个d维的向量。

**优点**

- 无需训练参数；
- 可处理比训练时更长的序列（外推能力）；
- 数学上具有良好的性质（如平滑性和周期性）。

**缺点**

- 表达能力受限于固定函数形式；
- 对长序列的区分度可能不足。

------

### 可学习位置编码（Learnable Positional Embedding）

**特点**

- **可训练参数**：为每个位置分配一个可学习的向量。
- 通常与词嵌入相加后输入模型。

**实现方式**

- 初始化一个形状为 $ (L_{\text{max}}, d) $ 的嵌入矩阵$( L_{\text{max}} $ 是最大序列长度）；
- 输入序列时，根据 token 的位置索引取出对应的位置向量。

**优点**

- 灵活，能适应任务特定的位置模式；
- 训练充分时效果通常优于正弦编码。

**缺点**

- **无法泛化到超过训练长度的序列**（因为没有对应的位置向量）；
- 需要额外参数，对内存和训练数据有一定要求。

------

### ROPE（Rotary Position Embedding，旋转位置编码）

**核心思想**

- 将**位置信息融入注意力计算过程**，通过**旋转变换**实现；
- 不是加在 token 嵌入上，而是在计算 query 和 key 时动态引入位置信息；
- **天然支持相对位置建模**。

**数学原理（简化版）**

将 query 向量和 key 向量按照偶数/奇数维度分组，视为复数： $$ q_i = q_{2i} + i q_{2i+1}, \quad k_j = k_{2j} + i k_{2j+1} $$ 然后乘以旋转因子 $ e^{i \theta_{|i-j|}} $，等价于对向量进行二维旋转： $$ \text{Re}(q_i \cdot k_j^* \cdot e^{i \theta_{i-j}}) = \text{内积考虑了相对位置} $$

[RoPE位置编码](https://zhuanlan.zhihu.com/p/647109286)

- 在计算qk注意力时，为了能够利用token之间的相对位置信息，关键是找到一个函数，函数输入是q向量、k向量以及m-n（两个token之间的差值），作者发现这个函数就是q向量先乘以一个旋转矩阵，再乘以k向量。这种方法通过绝对位置表示了相对位置。
- 扩展到多维，可以看作是二维的拼接。由于旋转矩阵的稀疏性，为了进行高效计算，先计算每个位置的旋转位置编码，q和k分别乘上旋转位置编码（可以进行简化为两个向量的内积），然后再进行注意力的计算。
- RoPE的总流程：首先对每个token进行embedding，然后根据embedding计算q和k向量，q向量和k向量分别乘上一个旋转矩阵（本质是向量每个维度两两一组应用旋转变换），然后再计算内积得到self-attention结果
- 实际实现中，通过预计算旋转矩阵（或使用 sin/cos 查表）高效完成。

**优点**

- **显式建模相对位置**，且保持绝对位置信息；
- **理论上支持任意长度外推**（虽然实际中仍有限制）；
- 与注意力机制天然融合，不增加嵌入层负担；
- 在长上下文任务中表现优异。

**缺点**

- 外推性能依赖于旋转频率的设计（如 LLaMA 使用了 NTK-aware 插值等改进）。

**代码：**

```python
import torch
import torch.nn.functional as F

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    Returns:
        freqs_cis: [end, dim // 2] (complex numbers represented as real tensors of shape [end, dim])
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [end, dim//2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return torch.view_as_real(freqs_cis)  # [end, dim//2, 2]

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    """
    Apply rotary embeddings to query and key.
    xq, xk: [batch, seq_len, n_heads, head_dim]  q向量和k向量
    freqs_cis: [seq_len, head_dim//2, 2] -> interpreted as complex numbers
    """
    # Reshape to complex
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)  # [..., d/2, 2]  两两一组进行旋转
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Convert to complex
    xq_complex = torch.view_as_complex(xq_)
    xk_complex = torch.view_as_complex(xk_)

    # freqs_cis: [seq_len, d/2] -> broadcast to [..., seq_len, d/2]
    freqs_cis = freqs_cis[None, :, None, :]  # [1, seq_len, 1, d/2]
    freqs_cis = freqs_cis.expand_as(xq_complex)

    # Rotate
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Precompute rotary frequencies
        # 为每个位置和每个特征维度提供旋转所需的 cos/sin 值
        # 形状[max_seq_len, head_dim // 2, 2]（最后维是 [cos, sin]）
        # 不可以进行学习，固定，基于公式计算，但可通过插值/NTK等方法扩展
        # 使用方式，在 attention 计算前，对 Q 和 K 按位置分别应用旋转变换
        # 支持长上下文时，会使用 动态 NTK 插值 或 YaRN 等方法重新缩放 freqs_cis 的频率，以提升外推能力；
        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)

        # Apply ROPE to q and k
        freqs_cis = self.freqs_cis[:L].to(x.device)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # [B, n_h, L, d_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, n_h, L, d_h]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)
```

