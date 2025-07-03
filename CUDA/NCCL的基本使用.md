[TOC]



## Single Process, Single Thread, Multiple Devices

这个例子对应的就是PyTorch Data Parallel里面的AllReduce操作

```C
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"

// CUDA错误检查
#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// NCCL错误检查
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[])
{
  ncclComm_t comms[4];


  //managing 4 devices
  int nDev = 4;
  int size = 32*1024*1024;
  int devs[4] = { 0, 1, 2, 3 };


  //allocating and initializing device buffers
  //为每个设备定义定义独立的输入/输出缓冲区
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  //s是指向cudaStream_t类型的指针，用于存储多个流的句柄
  //分配nDev个cudaStream_t大小的内存
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  //给每个GPU设置当前设备，然后分配发送和接收缓冲区的GPU内存，初始化发送缓冲区为1，接收缓冲区为0，最后为每个设备创建CUDA流
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }


  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));


  //calling NCCL communication API. Group API is required when using multiple devices per thread
  //ncclGroupStart() 开启一个 NCCL 操作组，允许批量执行多个 NCCL 操作。这些操作会被优化为一个整体执行，减少通信开销。
  NCCLCHECK(ncclGroupStart());
  //循环执行 all-reduce
  for (int i = 0; i < nDev; ++i)
    //对所有 GPU 上相同位置的元素进行求和，最终每个 GPU 的recvbuff[i]包含完整的求和结果
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
  //ncclGroupEnd() 结束组操作，触发所有排队的 NCCL 操作实际执行
  NCCLCHECK(ncclGroupEnd());


  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }


  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }


  //终止NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
```

## One Device per Process or Thread

这个例子对应的就是PyTorch Distributed Data Parallel里面的AllReduce操作

```C
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

static uint64_t getHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash(const char* hostname) {
  char hostHash[1024];

  // Fall back is the hostname if something fails
  (void) strncpy(hostHash, hostname, sizeof(hostHash));
  int offset = strlen(hostHash);

  FILE *file = fopen(HOSTID_FILE, "r");
  if (file != NULL) {
    char *p;
    if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        free(p);
    }
  }
  fclose(file);

  // Make sure the string is terminated
  hostHash[sizeof(hostHash)-1]='\0';

  return getHash(hostHash, strlen(hostHash));
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


int main(int argc, char* argv[])
{
  int size = 32*1024*1024;

  //进程相关变量：myRank为本进程全局排名，nRanks为总进程数，localRank为本节点内进程排名
  int myRank, nRanks, localRank = 0;

  //初始化MPI
  MPICHECK(MPI_Init(&argc, &argv));
  // 获取当前进程的全局唯一标识
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  // 获取参与并行计算的进程总数
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  //基于hostname计算localrank，localrank用来选择一个gpu
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);  //获取当前主机名
  hostHashs[myRank] = getHostHash(hostname);   // 计算当前主机名的哈希值
  // 收集所有进程的主机哈希值到每个进程的hostHashs数组中
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    
  // 统计当前节点上排名在自己之前的进程数量，确定localRank
  // 遍历所有进程的主机哈希值，遇到相同哈希值(同一节点)且排名比自己小的进程时计数
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  //获取rank0的id，并广播到其他进程
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  //为gpu申请输入/输出缓存
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));

  //初始化nccl
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  //使用nccl进行all reduce操作
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
        comm, s));

  //同步所有的cuda流来完成nccl操作
  CUDACHECK(cudaStreamSynchronize(s));

  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  ncclCommDestroy(comm);
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
```

## 重要概念

### 通道

理解 NCCL 中的“通道”概念是掌握其高性能通信设计的关键。它本质上是实现空间并行性的一种机制，用于更充分地利用网络硬件资源（链路带宽、网络接口卡、交换机端口），从而加速大型集合通信操作（尤其是像 AllReduce 这样需要传输大量数据的操作）。

#### 怎么理解通道

我们可以从以下几个方面来理解通道：

1. **核心目的**：增加并行度，榨干网络带宽

- **问题：** 一个复杂的通信操作（如一个大的 AllReduce）需要传输海量数据。如果只使用一条逻辑路径（例如，一个简单的树结构），即使这条路径被优化得很好，它也只能利用网络总带宽的一部分。
- **解决方案：通道 (Channels)：** NCCL 会将参与通信的一组 GPU（一个通信组）和它们要传输的数据逻辑上划分成多个独立的子组和子数据流。每个这样的子组和子数据流就对应一个通道。
- **并行工作**：多个通道可以同时、独立地进行数据传输。
  - 不同的通道可以使用不同的物理网络路径（链路、端口）。
  - 不同的通道处理原始数据的不同部分（数据块）。
- **效果：** 这就像把一条单车道的高速公路变成了多条并行车道。只要网络硬件资源（物理链路带宽、NIC、交换机处理能力）允许，多个通道就能叠加带宽，逼近网络的物理带宽上限。

2. **通道如何工作？**

- 数据分块 (Chunking):
  1. 对于一个大的 AllReduce 操作，需要传输的总数据量假设是 `totalSize`。
  2. NCCL 会将 `totalSize` 的数据切割成多个大小固定或最优的块 (Chunks)。每个 Chunk 是一个独立的数据传输单元。
- 通道分配 (Channel Assignment):
  1. NCCL 内部会创建 `numChannels` 个逻辑通道（数量通常根据拓扑、算法类型等确定）。
  2. 数据 Chunk 会被轮询 (Round-Robin)或根据某种策略分配给不同的通道
- 独立通信树 (Per-Channel Tree):
  1. 最关键的： 每个通道都独立构造一个完整的通信树（或环形拓扑等）！
  2. CUDA 内核是按通道启动的（更准确地说，内核会被调用来处理一个特定通道的数据）。

#### 为什么需要为每个通道构造独立的树

- **避免资源冲突：** 假设多个通道的数据块都通过同一个物理 GPU 节点在树中走完全相同的路径，那么这个节点（包括它的 CPU、内存、网卡、PCIe 带宽）就会成为瓶颈。不同通道的树结构在物理GPU拓扑上相互错开或交叉覆盖，才能最大化利用不同物理链路的带宽。
- **维持顺序和一致性：** AllReduce 操作的最终结果需要保证一致性。通过为每个数据块 (Chunk) 在它自己的通道树内进行独立的 Reduce-Scatter 和 AllGather，可以保证属于同一个 Chunk 的数据在逻辑上经过了正确的处理。最终，所有通道处理完所有分配给它们的 Chunk 后，组合起来就是完整的、全局一致的 AllReduce 结果。
- **负载均衡：** 将邻居连接分配到不同的树中，有助于平衡通信负载。

#### 通道与其他概念的关系

- **与拓扑 (Topology)：** 通道的数量 (`numChannels`) 通常是 NCCL 在初始化阶段根据检测到的硬件拓扑结构（有多少个独立的 NVLink Switch、多少个 NIC、机器内部的层级）自动计算出来的，目的是最大化带宽利用。物理资源丰富的节点/集群通常会启用更多通道。
- **与算法 (Algorithm)：** 不同的算法类型（如 `TREE`, `RING`, `COLLNET`）使用通道的方式基本相同，都依赖于数据块划分和独立路径传输。树形算法中每个通道有自己的树，环形算法中每个通道有自己的环。
- **与线程 (Threads)：** CUDA 内核里的 `tid`, `nthreads` 是负责执行单个 GPU 上单个通道内的通信操作的线程。它们处理的是这个通道分配到的数据块 (`chunkCount`, `channelCount`, `gridOffset`) 在这个特定通道的树 (`&ncclShmem.channel.tree`) 上的通信（Send/Recv/Reduce 等）。一个物理GPU上可能同时有多个 CUDA 内核在运行（每个负责一个不同的通道！），或者一个复杂的内核处理多个通道（需要同步机制）。

## 几种NCCL的通信算法

### Ring Allreduce算法源码

Ring Allreduce算法比较简单，主要由两个阶段组成：

1. n-1次的ReduceScatter操作：每个节点将数据分片，通过规约操作逐步聚合到其他节点
2. n-1次的Allreduce操作：每个节点将聚合后的结果逐步扩散到其他所有节点，最终每个节点都拥有完整的全局结果

```c
// Ring AllReduce算法实现 (结合了ReduceScatter和AllGather操作)
template<typename T, typename RedOp = Sum<T>, NcclProto Proto = NCCL_PROTO_SIMPLE>
__global__ void ringAllReduceKernel(ncclWorkElem* args) {
    const int tid = threadIdx.x;      // 获取当前线程ID
    const int nthreads = args->nWarps * WARP_SIZE;  // 计算总线程数
    const int bid = args->bid;        // 获取块ID
    const int nChannels = args->nChannels;  // 获取总的通道数
    ncclRing* ring = &ncclShmem::channel.ring;  // 获取环形通信结构的指针
    int ringIx = ring->index;         // 获取环形索引，获取当前GPU在环里的位置信息
    
    // 计算每步处理的数据块大小
    // 计算数据块的大小。整个AllReduce操作不是一次性完成的，而是把总数据（比如一个巨大的神经网络权重张量）切成很多片（loopSize），一片一片地执行完整的Ring AllReduce流程。chunkSize 就是比喻中每一轮传递的“一小份”数据的大小。
    const size_t chunkSize = (int(ProtocolTraits<Proto>::calcBytePerStep() / sizeof(T)) * 
                             (Proto == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const int nranks = ncclShmem::comm.nRanks;  // 获取总进程数
    const size_t loopSize = nChannels * nranks * chunkSize;  // 计算循环大小
    const size_t size = args->count;  // 获取需要处理的总数据量

    int minChunkSize;  // 最小数据块大小
    if (Proto == NCCL_PROTO_LL) {
        // LL协议下计算最小数据块大小
        minChunkSize = nthreads * (ProtocolTraits<Proto>::calcBytePerGrain() / sizeof(T));
    }
    if (Proto == NCCL_PROTO_LL128) {
        // LL128协议下的特殊处理
        // 注释说明这里的除2可能是个bug，但能提高性能
        minChunkSize = nthreads * (ProtocolTraits<Proto>::calcBytePerGrain() / sizeof(T)) / 2;
    }

    // 使用Primitives模板类处理规约操作
    Primitives<T, RedOp, FanSymmetric<1>, ProtocolTraits<Proto>, 0> prims(
        tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->redOpArg);

    //这个循环就是把整个大任务分成小任务来处理。gridOffset 代表当前正在处理的是第几“片”数据。
    for (size_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        size_t realChunkSize;
        
        // 处理NCCL协议简单模式
        if (Proto == NCCL_PROTO_SIMPLE) {
            // 计算实际的chunk大小，考虑网格偏移和通道数
            realChunkSize = min(chunkSize, divide(size - gridOffset, nChannels * nranks));
            // 根据线程数和数据类型大小调整chunk大小
            realChunkSize = roundUp(realChunkSize, (nthreads * WARP_SIZE) * sizeof(uint64_t) / sizeof(T));
        } else {
            // 非简单模式下的chunk大小计算
            realChunkSize = min(chunkSize, divide(size - gridOffset, nChannels * nranks * minChunkSize));
            realChunkSize = static_cast<size_t>(realChunkSize);
        }

        // 计算每个chunk的偏移量
        auto calcOffset = [&]__device__(int chunk) -> size_t {
            if (Proto == NCCL_PROTO_SIMPLE)
                return gridOffset + bid * nranks * realChunkSize + chunk * realChunkSize;
            else
                return gridOffset + (chunk * nChannels + bid) * realChunkSize;
        };

        // 计算每个rank的修改位置
        auto modRanks = [&]__device__(int r) -> int {
            return r >= nranks ? r - nranks : r;
        };

        // 声明变量
        size_t offset;
        int nelem;
        int chunk;

        // ReduceScatter阶段
        // step 0: 将数据推送到下一个GPU。这一步并非完整的第一步，而是第一步的数据发送部分
        
        //这一行在计算当前这一步应该操作哪个数据块。环形算法的精髓就在于，每个GPU在同一步操作的数据块索引是不同的，但相对于自己在环中的位置是有规律的。
        chunk = modRanks(ringIx + nranks - 1); 
        offset = calcOffset(chunk);           // 计算偏移量
        nelem = min(realChunkSize, size - offset); // 计算元素数量
        // 每个GPU把自己负责的第一个数据块发送给下一个GPU，这一步不同于后续步骤，不需要规约加和操作
        prims.send(offset, nelem);           

        // 共nranks-2步: 执行规约操作并将结果复制到下一个GPU
        for (int j = 2; j < nranks; ++j) {
            // 计算当前需要处理的数据块索引
            chunk = modRanks(ringIx + nranks - j);
            
            // 根据chunk计算在缓冲区中的偏移量
            offset = calcOffset(chunk);
            
            // 计算本次需要传输的实际元素数量
            nelem = min(realChunkSize, size - offset);
            
            // 执行接收-规约-发送操作
            prims.recvReduceSend(offset, nelem);
        }

        // last step，总的步数是(nranks-2+1=nranks-1)
        chunk = ringIx + 0;
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size - offset);
        //执行接受-规约，并把数据发送到目标缓冲区。
        //当 postOp 为 true 时,意味着把最后一步的规约结果写入最终的目标缓冲区 recvbuff
        prims.directRecvReduceCopySend(offset, offset, nelem, /*postOp=*/true);

        // AllGather阶段
        // 实现AllGather操作，将规约结果广播到所有GPU
        for (int j = 0; j < nranks - 1; ++j) {
            __syncthreads(); // 同步确保数据准备好
            
            int current_rank = modRanks(ringIx + j);
            chunk = current_rank;
            offset = calcOffset(chunk);
            nelem = min(realChunkSize, size - offset);
            
            if (j == 0) {
                // 当前rank发送数据
                prims.send(offset, nelem);
            } else {
                // 其他rank接收并复制数据
                prims.recvReduceSend(offset, nelem);
            }
        }
    }
}
```

### Tree Allreduce算法源码

#### TreeUpDown算法

- UP阶段：首先将节点两两分组，每组有两个节点，一个节点把数据规约到另外一个节点。第一轮规约之后，只保留规约组内数据的节点，然后再把节点两两分组，重复执行操作，最后总数据规约到跟节点。
- Down阶段：将根节点总数据逐层往叶子节点广播，这样所有的节点都获得了总数据。



```c
template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeUpDown(int tid, int nthreads, struct ncclDevWorkColl* work) {
    //获取当前通道的tree通信的指针
    ncclTree *tree = &ncclShmem.channel.tree;
    // gridOffset: 这次Kernel调用处理的数据块的起始偏移
    size_t gridOffset;
    // channelCount: 这个通信通道总共要处理的数据量
    size_t channelCount;
    // chunkCount: 为了效率，数据被分成一小块一小块(chunk)处理，这是每一小块的大小
    size_t chunkCount;
    // 这个函数很重要，它根据总工作量和当前GPU的信息，计算出这个Kernel要处理哪一部分数据
    //Proto 代表通信协议（Protocol），Proto::Id 就是这个协议的唯一标识符
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    
    //在整个数据中的绝对位置偏移
    size_t offset;
    int nelem;

    { // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      //初始化一个“原语”对象，配置为“Up”模式
      // FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>: 这是关键配置。它告诉工具箱，在Reduce阶段，一个节点最多会从 NCCL_MAX_TREE_ARITY 个子节点接收数据（比如二叉树就是2个），并且最多向 1 个父节点发送数据。
      //tree->down: 子节点的rank列表；&tree->up: 父节点的rank；
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0> prims
        (tid, nthreads, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      //根据当前GPU在树中的角色，执行不同操作
      if (tree->up == -1) {  // 角色：树根 (Root)，只接收和计算，不发送
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {  //角色：叶子节点 (Leaf)，只发送，不接收
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSend(offset, offset, nelem);
        }
      }
      else {  //角色：中间节点 (Intermediate)，既接收、计算，又发送
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
      }
    }
    //Reduce阶段完成后，最终的归约结果已经存在于树根GPU的 recvbuff 中

    // 广播阶段
    { // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      //FanAsymmetric<1, NCCL_MAX_TREE_ARITY>:一个节点最多从 1 个父节点接收数据，然后最多向 NCCL_MAX_TREE_ARITY 个子节点发送数据
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0> prims
        (tid, nthreads, &tree->up, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
        
      if (tree->up == -1) {  // 角色：树根 (Root)，只发送，不接收
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSendFromOutput(offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {  // 角色：叶子节点 (Leaf)，只接收
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecv(offset, nelem);
        }
      }
      else {       // 角色：中间节点 (Intermediate)，即接收又发送
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    }
    //Broadcast阶段完成后，所有参与通信的GPU的 recvbuff 中都拥有了相同的、最终的归约结果
  }
```

#### TreeSplit算法

```c
template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeSplit(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclTree *tree = &ncclShmem.channel.tree;
    size_t gridOffset;
    size_t channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;
    int nthreadsSplit;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      nthreadsSplit = nthreads/2;
      if (nthreadsSplit >= 256) nthreadsSplit += 64;
    } else { // LL & LL128
      //为什么分裂比例不同？ 注释里写得很清楚。对于 LL/LL128 协议，接收数据并计算（Reduce）比单纯发送（Broadcast）更耗费计算资源，所以给“上报组”分配了大约 70% 的线程，剩下的 30% 给“下达组”。这是为了让两组任务的完成时间差不多，避免出现“短板效应”。
      nthreadsSplit = (nthreads*7/(10*WARP_SIZE))*WARP_SIZE;
    }

    if (tree->up == -1) {  // 根节点，只发送数据
      // Reduce and broadcast. Max number of recv is 2, max number of send is 2
      Primitives<T, RedOp, FanSymmetric<NCCL_MAX_TREE_ARITY_TOP>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreads, tree->down, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*doPost=*/true);
      }
    }
    else if (tid < nthreadsSplit) {  //Up组线程，把数据上报
      /* Reduce up. Max number of recv is 3, max number of send is 1 (binary tree + local).
       * Why Direct=1????
       * Answer: Because despite not performing any direct operations, the ctor
       * must assume Direct so that it can exchange direct pointers with remote ctors
       * that are Direct, otherwise it hangs. A cleaner solution would be to seperate
       * into DirectRecv and DirectSend capabilities, this ctor would have both=0,
       * but the ctor above for tree roots would be DirectRecv=0 DirectSend=1.
       */
      // Coverity reports that the callee treats &tree->up as an array.  However, due to the use of
      // FanAsymmetric<n, 1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreadsSplit, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, work);
        
      if (tree->down[0] == -1) {  // 叶子节点
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSend(offset, offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
      }
    }
    else {   //Down组线程
      // Broadcast down. Max number of recv is 1, max number of send is 3 (binary tree + local)
      // Coverity reports that the callee treats &tree->up as an array.  However, due to the use of
      // FanAsymmetric<1, n>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0>
        prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, work->sendbuff, work->recvbuff,
            work->redOpArg, 1*Proto::MaxGroupWidth, 0, 0, work);
        
      if (tree->down[0] == -1) {  // 叶子节点，只接收数据
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecv(offset, nelem);
        }
      }
      else {   // 中间节点
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    }
  }
}
```

































