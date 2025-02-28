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


# CUDA

- cuda在线文档：https://developer.nvidia.com/cuda-toolkit-archive

- CUDA C++ Programming Guide
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

- [《CUDA C 编程指南》导读](https://mp.weixin.qq.com/s/0wFD5Q_U0TT32NIxy45y0g)

- [超算习堂](https://easyhpc.net/course)

- [中科大周斌-CUDA基础实践-B站视频](https://www.bilibili.com/video/BV1kx411m7Fk/?spm_id_from=333.1387.homepage.video_card.click&vd_source=74dccf8bd3bdb248a665ea9aa40edf58)

- 相关书籍：CUDA C编程权威指南、

## cuda基础入门

- [显卡，显卡驱动,nvcc, cuda driver,cudatoolkit,cudnn介绍](https://zhuanlan.zhihu.com/p/91334380)
  - 多版本cuda管理
- [详解PyTorch编译并调用自定义CUDA算子的三种方式](https://mp.weixin.qq.com/s/rG43pnWY8fBjyIX-mFWTqQ)
- [教你用 Cuda 实现 PyTorch 算子](https://mp.weixin.qq.com/s/YwxhvjhqxYy_ejBOt0bkrQ)
- [Cuda编程：基础知识](https://blog.csdn.net/qq_44753080/article/details/136528574?spm=1001.2014.3001.5506)

## 线程块、线程网格和索引的计算

CUDA 中的线程块（block）和线程网格（grid）是多维的。默认情况下，`blockIdx` 和 `threadIdx` 都是三维的，分别包含 x、y、z 维度的索引。所以，在 CUDA 中每个线程块（block）可以有多个维度（如：`blockIdx.x`, `blockIdx.y`, `blockIdx.z`），每个线程也可以在多个维度上进行索引（如：`threadIdx.x`, `threadIdx.y`, `threadIdx.z`）。

然而，实际上你不一定需要使用所有的维度。如果你的问题是二维或一维的，你可以只使用 `x` 维度，省略 `y` 和 `z` 维度。**因此具体使用哪些维度，主要是根据所要计算的数据决定的，当然与计算方法也有一定关系。**

具体来说：

- **blockIdx**: 用于表示线程块在网格中的位置。`blockIdx.x` 表示块在 x 维度上的位置，`blockIdx.y` 和 `blockIdx.z` 分别表示 y 和 z 维度上的位置。
- **threadIdx**: 用于表示线程在当前线程块中的位置。`threadIdx.x` 表示线程在 x 维度上的位置，`threadIdx.y` 和 `threadIdx.z` 分别表示 y 和 z 维度上的位置。



**一维列表计算例子**

```c++
__global__ void vecAdd(const double *x, const double *y, double *z, const int N)
{
    // 计算当前线程在一维数据中的全局索引
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // 判断索引是否超出数组的边界
    if (index < N)
    {
        // 进行数组元素的加法
        z[index] = x[index] + y[index];
    }
}

```

有似于对数据进行分段。



**二维网格计算例子**

```c++
__global__ void vecAdd(const double *x, const double *y, double *z, const int N)
{
    const int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int index = index_y * gridDim.x * blockDim.x + index_x;
    if (index < N)
    {
        z[index] = x[index] + y[index];
    }
}

```

类似对二维网格，划分成多个小的二维网格。

其中`gridDim.x * blockDim.x`其实就是`x`方向的行的数量，比如对于一个`M*N`(x方向, y方向)的网格，那么`gridDim.x * blockDim.x`其实就是M。



**通用的全局索引计算**

```c++
__global__ void print_thread_idx_per_grid_kernel()
{
	// 一个block中含有的线程数量
    int bSize = blockDim.z * blockDim.y * blockDim.x;

	// block的全局index号
    int bIndex = blockIdx.z * gridDim.x * gridDim.y +
                 blockIdx.y * gridDim.x +
                 blockIdx.x;
	
	// block内的线程索引
    int tIndex = threadIdx.z * blockDim.x * blockDim.y +
                 threadIdx.y * blockDim.x +
                 threadIdx.x;
	
	// 全局唯一索引
    int index = bIndex * bSize + tIndex;

    printf("block idx: %3d, thread idx in block: %3d, thread idx: %3d\n",
           bIndex, tIndex, index);
}


```

## cudaMalloc的参数解释

```c++
// 调用 cudaMalloc 分配内存
cudaMalloc((void **)&d_z, M);
```

cudaMalloc 函数的第一个参数：(void \*\*)&d_z，cudaMalloc 函数需要一个类型为 void \*\* 的参数。这个参数的作用是传递一个指针的地址，以便 cudaMalloc 在执行时能够修改它，让它指向分配的设备内存。

cudaMalloc 函数会修改传入的指针，使它指向 GPU 上分配的内存。

当调用 cudaMalloc 时，你给它传入的是 &d_z（即 double ** 类型），所以 cudaMalloc 会将 d_z 指向 GPU 上分配的内存块。

cudaMalloc 会在内部修改 d_z 的值，使得它指向实际的设备内存。为了能让 d_z 在函数外部也能反映这个修改，必须通过 &d_z 传递它的地址，也就是传递指向指针的指针（void **）。

## CUDA核心与线程的关系

**GPU线程的实际含义：**

- GPU线程（Thread）在GPU编程模型（例如CUDA或OpenCL）中，是一个逻辑上的概念。
- GPU同时管理数以万计的线程，但并不是所有线程同时运行，而是分批调度执行。

**CUDA核心的实际含义：**

- CUDA核心本质上是GPU中负责执行**单个简单指令**（比如一次浮点运算）的基本执行单元，类似于CPU的算术逻辑单元(ALU)。
- 一个CUDA核心每个时钟周期只能执行一个简单操作（例如一次浮点加法或乘法）。

**实际执行情况：**

- GPU线程以**Warp**（在NVIDIA中，Warp通常包含32个线程）为单位调度。一个Warp内的32个线程会同步执行同一条指令（SIMT模型）。
- 当GPU执行一个Warp（32个线程）时，如果这个Warp调度到Streaming Multiprocessor (SM) 上，而SM中有足够的CUDA核心（例如32个CUDA核心），则该Warp的全部32个线程会同时并行执行，每个CUDA核心执行一个线程的同一条指令。
- 如果Warp大小超过CUDA核心的数量（或者可用核心不足），那么GPU则需要多次执行（多个时钟周期）才能完成Warp中所有线程的运算。

**关键的理解：**

- CUDA线程是逻辑抽象的概念，用于管理并行性和任务。
- CUDA核心是物理单元，是线程任务在硬件上实际执行的地方。
- CUDA核心可以在不同时间点执行不同的线程指令，因此**线程数远远大于CUDA核心数**是很正常的情况（GPU往往同时管理上万个线程，但物理CUDA核心数量却只有几百或几千个）。





















