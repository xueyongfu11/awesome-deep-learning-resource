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
## Nsight Systems 与 Nsight Compute 的分工和优化流程

Nsight Systems 和 Nsight Compute 是 NVIDIA 新一代性能分析工具。早期的 Visual Profiler 和 `nvprof` 已逐步被它们替代：

- **Nsight Systems**：从系统和应用视角观察 CPU、GPU、线程、库调用、CUDA API、kernel、内存传输及其时间关系。
- **Nsight Compute**：针对单个 CUDA kernel，深入分析计算、内存、调度和硬件指标。

性能优化通常采用 top-down 的迭代流程：先 profiling，识别空闲、等待、同步、数据传输和计算热点；再定位到代码或应用阶段，优化后重新构建并 profiling。典型流程是先用 Nsight Systems 定位系统级问题，再用 Nsight Compute 分析代表性 kernel，最后回到 Nsight Systems 验证整体收益。该工具链支持 Linux、Windows、x86-64 和 Tegra；Mac 主要作为 host 使用。

### Nsight Systems 的补充分析能力

Nsight Systems 可追踪进程与线程、线程所在 CPU core、CPU 利用率、线程状态及状态转换、线程迁移、pthread、文件 I/O 和其他 OS runtime 调用；也能追踪 CUDA API、kernel launch、内存操作、cuBLAS、cuDNN、TensorRT、CUDA Runtime、CUDA stream、多 GPU 活动，以及 kernel 与 memory 的时间重叠。时间线可区分线程运行和等待状态，查看线程所在 CPU core 及利用率。对于较长的 OS runtime 调用，还可以查看 backtrace，追溯调用来源。线程等待时，应先判断原因，常见原因包括信号量或互斥锁、文件 I/O、CPU 等待 GPU、GPU 等待 CPU，以及内存操作或 kernel launch 排队。

它还可以把 CPU 上发起的 CUDA API 与 GPU 上对应的 kernel 或 memory 活动连接起来。kernel launch 在 CPU 上发生、kernel 执行在 GPU 上发生，观察二者之间的距离和先后关系，有助于判断 kernel launch bound 或调度空洞。GPU workload 轨道中的 coverage 表示一段时间内 GPU 的活动覆盖程度，不能与 SM occupancy 混淆。

## Nsight Compute：kernel 级分析

Nsight Compute 是交互式 CUDA kernel profiler，提供图形化 profiling report、SOL（Speed of Light）、Memory Chart、API Stream、baseline 前后对比，以及较快的数据采集能力。工具包括 `ncu-ui`（旧资料中也称 `nv-nsight-cu`）和 `ncu`/`nv-nsight-cu-cli`；资料中的支持范围为 Pascal、Volta、Turing、Ampere，具体能力以当前版本官方文档为准。

### API Stream、触发器与指标

API Stream 可以跳转到下一个 CUDA kernel、CUDA API 调用或 NVTX range，也可以用正则表达式过滤 kernel/API 名称。`Next Trigger` 可输入 kernel 名称的一部分，快速定位并 profile 匹配的 kernel。

- **Event**：设备上可计数的活动或发生次数，例如 cache miss、读请求。
- **Metric**：由一个或多个 event 计算得到的特征指标。
- **Section**：按主题组织的一组 metrics，用于判断瓶颈类型，例如 Memory Workload、Compute Workload 和 Scheduler Statistics。

Nsight Compute 的 metric 更贴近硬件行为，与 `nvprof` 的 event/metric 定义不能简单一一对应。迁移旧工具时，应根据当前 GPU 和版本查询可用指标，并参考官方映射关系。

### SOL 与 stall reason

SOL 以理论峰值为参照，帮助区分：

- **Compute bound**：SM/计算资源利用率高，内存利用率相对低；
- **Memory bound**：内存子系统利用率高；
- **Latency bound**：SM 和内存利用率都不高，常见原因是 warp 数量不足、依赖链或等待。

常见 stall reason 包括 LG throttle/scoreboard（全局或局部内存请求过多、依赖未完成）、Short scoreboard（等待 shared memory 或特殊函数）、MIO throttle（数学或特殊函数管线使用过度）和 Barrier（线程块同步等待）。应结合具体 pipeline、吞吐量和等待原因下钻分析，不能只根据单一百分比下结论。

## 典型优化案例

### Transformer 的 kernel launch bound

如果 Nsight Systems 时间线显示大量很小的 kernel、kernel 之间空隙明显且 GPU 长时间空闲，通常可能是 kernel launch bound。Transformer encoder 包括 embedding、multi-head attention（Q/K/V、softmax）、输出变换、bias、layer normalization、feed-forward、残差和 normalization 等阶段；一个 layer normalization 可能被拆成多个 kernel。讲座案例中 kernel 数量约从 50 降到 24 后，进一步通过融合继续减少空洞。常见优化路径是使用 TensorFlow XLA 或手工融合，将 bias、transpose、softmax、layer normalization、activation 等操作与 GEMM 融合，并使用 cuBLAS/cuBLASLt 完成主要矩阵运算，从而减少 kernel 数量和启动间隙。优化后应重新查看时间线验证空洞是否缩短。

### Global memory 合并访问与矩阵转置\n\n讲座案例的矩阵规模为 8192×4096，数据类型按 4 字节计算，输入读取和输出写回各约 128 MiB；global memory 与 L2 之间以 32 字节 sector 传输。

矩阵转置的直接实现可能导致 global store 按列写入，warp 中线程访问分散到多个 sector，有效请求字节与实际传输字节的比例很低，因而表现为 memory bound。常用优化方法是：先合并读取到 shared memory，在 shared memory 中完成 tile 转置，再合并写回 global memory。典型的 32×32 tile 可使用 32×33 padding，减少 shared memory bank conflict。

无论是 global memory 还是 shared memory，都要结合 warp 的访问方向、访问粒度、sector 数量和 bank 映射分析。首先保证 global memory 合并访问；在问题规模允许时，提高每线程访问粒度可以减少访存指令数。讲座案例中，最差访问模式约耗时 834 微秒；每线程访问 4、8、16 字节时分别约为 220、143、121 微秒。最差情况下，一个 warp 的 32 个线程访问分散到许多 sector，实际传输字节远大于有效请求字节，效率可能只有 12.5%。

## 实践建议

- Systems 用于发现 CPU/GPU 利用率、线程状态、同步、API、kernel launch、memory copy 和多 GPU 交互问题。
- Compute 用于深入分析单个 kernel 的 SOL、访存模式、调度、stall reason 和硬件指标。
- 使用 NVTX 将算法阶段映射到时间线。C/C++ 可使用 `nvtxRangePushA()` 和 `nvtxRangePop()` 包围初始化、训练迭代、batch、forward、backward 等阶段，并在编译时链接 NVTX 库（例如 `-lnvToolsExt`）；Python、CuPy、TensorFlow 也提供相应支持。
- 先解决系统级空闲、同步、启动和传输问题，再分析 kernel 内部的合并访问、bank conflict、占用率和指令级问题。
- 每次修改后重新 profile，确认整体性能变化，并继续寻找下一个主要热点。



