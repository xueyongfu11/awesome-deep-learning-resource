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

## Ring Allreduce

Ring Allreduce算法比较简单，主要由两个阶段组成：

1. n-1次的ReduceScatter操作：每个节点将数据分片，通过规约操作逐步聚合到其他节点
2. n-1次的Allreduce操作：每个节点将聚合后的结果扩散到其他所有节点，最终每个节点都拥有完整的全局结果

```c
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstddef>
#include <algorithm>

// 定义基本常量
#define WARP_SIZE 32
#define ALLREDUCE_CHUNKSTEPS 1

// NCCL协议类型枚举
enum NcclProto {
    NCCL_PROTO_SIMPLE,
    NCCL_PROTO_LL,
    NCCL_PROTO_LL128
};

// 通信结构定义
struct ncclComm {
    int nRanks;         // 总进程数
    int myRank;         // 当前进程ID
};

// 环形通信结构
struct ncclRing {
    int index;          // 环形索引
    int prev;           // 前一个节点
    int next;           // 下一个节点
};

// 通道结构
struct ncclChannel {
    ncclRing ring;      // 环形通信结构
};

// 共享内存结构
struct ncclShmem {
    static __device__ ncclComm comm;  // 通信信息
    static __device__ ncclChannel channel;  // 通道信息
};

// 工作元素结构
struct ncclWorkElem {
    int nWarps;         // warp数量
    int bid;            // 块ID
    int nChannels;      // 通道数
    size_t count;       // 数据量
    void* sendbuff;     // 发送缓冲区
    void* recvbuff;     // 接收缓冲区
    void* redOpArg;     // 规约操作参数
};

// 规约操作模板
template<typename T>
struct Sum {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

// 协议特性模板
template<NcclProto Proto>
struct ProtocolTraits;

template<>
struct ProtocolTraits<NCCL_PROTO_SIMPLE> {
    static constexpr NcclProto Id = NCCL_PROTO_SIMPLE;
    static __device__ __forceinline__ size_t calcBytePerStep() { return 1024; }
    static __device__ __forceinline__ size_t calcBytePerGrain() { return 128; }
};

template<>
struct ProtocolTraits<NCCL_PROTO_LL> {
    static constexpr NcclProto Id = NCCL_PROTO_LL;
    static __device__ __forceinline__ size_t calcBytePerStep() { return 2048; }
    static __device__ __forceinline__ size_t calcBytePerGrain() { return 256; }
};

template<>
struct ProtocolTraits<NCCL_PROTO_LL128> {
    static constexpr NcclProto Id = NCCL_PROTO_LL128;
    static __device__ __forceinline__ size_t calcBytePerStep() { return 4096; }
    static __device__ __forceinline__ size_t calcBytePerGrain() { return 512; }
};

// 扇出模式模板
template<int Fanout>
struct FanSymmetric {};

// 基础操作原语
template<typename T, typename RedOp, typename Fanout, typename ProtoTraits, int Flags>
class Primitives {
private:
    const int tid;
    const int nthreads;
    int* prev;
    int* next;
    void* sendbuff;
    void* recvbuff;
    void* redOpArg;
    RedOp redOp;

public:
    __device__ __forceinline__ Primitives(
        int tid, int nthreads, int* prev, int* next, 
        void* sendbuff, void* recvbuff, void* redOpArg)
    : tid(tid), nthreads(nthreads), prev(prev), next(next),
      sendbuff(sendbuff), recvbuff(recvbuff), redOpArg(redOpArg) {}

    // 发送数据
    __device__ __forceinline__ void send(size_t offset, size_t nelem) {
        // 实际实现中会包含GPU间通信代码
        // 这里简化为打印操作
        printf("Rank %d sending %zu elements from offset %zu\n", 
               ncclShmem::comm.myRank, nelem, offset);
    }

    // 接收-规约-发送操作
    __device__ __forceinline__ void recvReduceSend(size_t offset, size_t nelem) {
        // 实际实现中会包含GPU间通信和规约操作
        // 这里简化为打印操作
        printf("Rank %d receiving, reducing and sending %zu elements at offset %zu\n", 
               ncclShmem::comm.myRank, nelem, offset);
    }

    // 直接接收-规约-复制-发送操作
    __device__ __forceinline__ void directRecvReduceCopySend(
        size_t srcOffset, size_t dstOffset, size_t nelem, bool postOp) {
        // 实际实现中会包含GPU间通信、规约和内存复制操作
        // 这里简化为打印操作
        printf("Rank %d direct recv-reduce-copy-send %zu elements from %zu to %zu\n", 
               ncclShmem::comm.myRank, nelem, srcOffset, dstOffset);
    }
};

// 辅助函数：向上取整
__device__ __forceinline__ size_t roundUp(size_t x, size_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

// 辅助函数：除法，向上取整
__device__ __forceinline__ size_t divide(size_t numerator, size_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

// Ring AllReduce算法实现 (结合了ReduceScatter和AllGather操作)
template<typename T, typename RedOp = Sum<T>, NcclProto Proto = NCCL_PROTO_SIMPLE>
__global__ void ringAllReduceKernel(ncclWorkElem* args) {
    const int tid = threadIdx.x;      // 获取当前线程ID
    const int nthreads = args->nWarps * WARP_SIZE;  // 计算总线程数
    const int bid = args->bid;        // 获取块ID
    const int nChannels = args->nChannels;  // 获取通道数
    ncclRing* ring = &ncclShmem::channel.ring;  // 获取环形通信结构的指针
    int ringIx = ring->index;         // 获取环形索引
    
    // 计算每步处理的数据块大小
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

    // 主循环处理所有数据块
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
        // step 0: 将数据推送到下一个GPU
        chunk = modRanks(ringIx + nranks - 1);  // 计算chunk索引
        offset = calcOffset(chunk);           // 计算偏移量
        nelem = min(realChunkSize, size - offset); // 计算元素数量
        prims.send(offset, nelem);           // 发送数据

        // k-2步: 执行规约操作并将结果复制到下一个GPU
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

        // step k-1: 在当前GPU上规约缓冲区和数据
        // 规约结果将存储在当前数据中并传送到下一个GPU
        chunk = ringIx + 0;
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size - offset);
        
        // 执行接收-规约-复制-发送操作
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

// 初始化函数示例
void initRingAllReduce(ncclComm& comm, ncclChannel& channel, int rank, int nRanks) {
    comm.nRanks = nRanks;
    comm.myRank = rank;
    
    channel.ring.index = rank;
    channel.ring.prev = (rank - 1 + nRanks) % nRanks;
    channel.ring.next = (rank + 1) % nRanks;
}

// 启动核函数的包装函数
template<typename T, typename RedOp = Sum<T>, NcclProto Proto = NCCL_PROTO_SIMPLE>
void launchRingAllReduce(ncclWorkElem* args, dim3 gridDim, dim3 blockDim) {
    ringAllReduceKernel<T, RedOp, Proto><<<gridDim, blockDim>>>(args);
    cudaDeviceSynchronize();
}
```

































