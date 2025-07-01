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





































