[TOC]



## mix

- End-To-End Memory Networks
  - year: 2015 NeuralIPS
  - 阅读笔记: 
    1. 给定一个memory set，通过一个矩阵A映射到一个向量空间，与query embedding进行点积计算，使用softmax计算weights
    2. 将memory set通过一个矩阵C映射到一个向量空间，使用weights加权后得到输出
    3. 将输出向量和query向量相加后，通过一个线性层softmax计算概率分布
  - code: 


## Normalization

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


## activation func

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


## Loss

- [PolyLoss超越Focal Loss](https://mp.weixin.qq.com/s/4Zig1wXNDHEjmK1afnBw4A)

## MLP回归网络的前向和后向传播

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