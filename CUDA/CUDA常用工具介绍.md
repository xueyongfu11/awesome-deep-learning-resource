[TOC]



# nsight system

## 介绍

NVIDIA Nsight Systems是一款全系统性能分析工具，旨在可视化应用程序的算法，找出最大的优化机会，并进行调整以在任何数量或规模的CPU和GPU（从大型服务器到最小的片上系统 (SoC)）上实现高效扩展。

主要的功能如下：

1. 可视化CPU和GPU之间的交互

   可视化CPU与GPU的交互 Nsight Systems会接入目标应用程序，以时间线的形式展示GPU和CPU的活动、事件、标注、吞吐量以及性能指标。该工具开销低，能准确且并行地可视化这些数据，便于理解。此外，它还会将GPU工作负载与应用程序内的CPU事件进一步关联，从而让性能瓶颈能够被轻松识别并解决。

2. 追踪GPU活动

   若要进一步探究GPU，开启GPU指标采样功能后，将绘制出底层输入/输出（IO）活动的图表，例如PCIe吞吐量、NVIDIA NVLink以及动态随机存取存储器（DRAM）活动。 此外，GPU指标采样还能显示SM利用率、张量核心活动、指令吞吐量和线程束占用率。每一项工作负载及其对应的CPU源头都可被轻松追踪，从而为性能调优提供支持。

3. 跟踪GPU工作负载

   Nsight Systems在计算任务方面，支持对CUDA API以及cuBLAS、cuDNN、NVIDIA TensorRT等CUDA库进行研究和追踪

4. 加速多节点性能

   Nsight Systems支持多节点性能分析，可解决数据中心和集群规模的性能限制问题。其多节点分析能同时自动诊断多个节点的性能限制因素，且结合网络指标与Python回溯采样，可全面呈现GPU、CPU、DPU以及节点间通信的情况。

## 具体使用

### 执行性能分析

Nsight通常展现给我们的是一个时序图，为了将每个时间段与代码对应起来，通常使用NVTX来实现。下面以一段python demo为例进行性能分析。

```python
import nvtx  # 引入nvtx包，便于在nsight system中观察各个函数的逻辑关系。
import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

device = torch.device("cuda:0")
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda(device)
criterion = torch.nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()


def train(data, batch_idx):
    # 数据传输
    nvtx.push_range("copy data " + str(batch_idx), color="rapids")
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    nvtx.pop_range()
    # 前向传播
    nvtx.push_range("forward " + str(batch_idx), color="yellow")
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    nvtx.pop_range()
    # 后向传播
    nvtx.push_range("backward " + str(batch_idx), color="green")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    nvtx.pop_range()


# 由于enumerate(train_loader)无法通过插入nvtx统计数据加载时间，将for循环代码替换为如下等价代码。
dl = iter(train_loader)
batch_idx = 0
while True:
    try:
        # 统计最后3个batch总共持续的时间。
        if batch_idx == 5:
            tet = nvtx.start_range(message="Total Elapsed Time(3 batchs)", color="orange")
        # 只观察前面8个batch，在这8个batch中前面5个用于wait和warmup，目标观察后面3个
        if batch_idx >= 8:
            nvtx.end_range(tet)
            break
        # 数据加载。
        nvtx.push_range("__next__ " + str(batch_idx), color="orange")
        batch_data = next(dl)
        nvtx.pop_range()
        # batch处理，包括数据传输和GPU计算
        nvtx.push_range("batch " + str(batch_idx), color="cyan")
        train(batch_data, batch_idx)
        nvtx.pop_range()
        batch_idx += 1
    except StopIteration:
        nvtx.pop_range()
        break
```

执行以下命令，并在当前目录获取`baseline.nsys-rep`文件

```bash
nsys profile \
    -w true \                  # 启用 warnings 警告信息输出
    --cuda-memory-usage=true \ # 记录 CUDA 内存分配和释放事件
    --python-backtrace=cuda \  # 在 CUDA 事件发生时捕获 Python 调用栈
    -s cpu \                   # 指定分析的采样类型为 CPU
    -f true \                  # 强制覆盖已存在的输出文件
    -x true \                  # 生成额外的详细分析报告
    -o baseline \              # 指定输出文件的前缀名为 "baseline"
    python ./main.py           # 要分析的目标程序及参数

# 更多常用参数：
    -d 60 \                    # 设置分析持续时间（秒），超时后自动停止
    -c cuda \                  # 指定要分析的 CUDA 活动
    --nvtx \                   # 启用 NVTX 标记追踪
    --mpi \                    # 启用 MPI 通信分析
    --trace=cuda,osrt \        # 指定要追踪的事件类型（cuda：CUDA操作；osrt：操作系统运行时）
    --sample=none \            # 禁用采样（仅追踪事件）
    --export=sqlite \          # 指定导出格式（sqlite/json等）
    --force-overwrite=true \   # 强制覆盖已有文件（同 -f）
    --output-dir=./reports \   # 指定输出文件的保存目录
```





























参考：

- [利用Nsight Systems对AI应用进行性能分析与优化](https://help.aliyun.com/zh/ack/cloud-native-ai-suite/use-cases/using-nsight-system-to-realize-performance-analysis)