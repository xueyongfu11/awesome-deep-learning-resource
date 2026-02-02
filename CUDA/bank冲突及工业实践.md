[TOC]



## 基础概念

### 存储体

现代内存（如 DRAM、GPU 共享内存、SRAM 等）为了提高并行访问带宽，会将物理存储空间分割成多个独立的 “存储体”。每个存储体可以独立地进行读写操作，就像多个并行的 “通道”—— 当多个访问请求指向不同的存储体时，它们可以同时被处理，从而显著提升内存带宽。

例如：

- GPU 的共享内存（Shared Memory）通常被划分为 32 个或 16 个存储体（不同架构可能不同）；
- DRAM 通过 “Bank Group” 和 “Bank” 的层级结构支持并行访问；
- 多核处理器的 L1/L2 缓存也可能采用多 bank 设计以支持多核并行访问。



在 CUDA 的 shared memory 里，数据按地址被切成以 bank 宽度（常见 4B）为单位的连续“word”，这些 word 会以交错轮转的方式分配到多个 bank：第 0 个 word 落在 bank0、第 1 个落在 bank1……到 bankN-1 后再回到 bank0，如此循环；

因此某个地址属于哪个 bank 主要由 bank_id = (addr / bank_width) % num_banks 决定，而不是“整块数据顺序装进同一个 bank”。

### 存储体冲突的定义与产生机制

当多个并行访问请求（如 GPU 线程束中的多个线程、多核处理器的多个核心）试图在同一周期内访问同一个存储体时，就会发生 “存储体冲突”。

例如：GPU 线程束中的 32 个线程通常会同时访问共享内存。如果多个线程访问的地址通过计算后，得到相同的存储体编号，就会发生冲突。

## 如何判断是否存在bank冲突

计算存储体（bank）冲突的核心是确定线程访问的存储体编号，并判断是否存在多个线程访问同一存储体的情况。以下是关键公式及步骤总结：


### 核心公式
#### 1. 线程i访问的字节地址
假设：
- 数组起始字节地址为 `base`（基地址）；
- 线程编号为 `i`（0 ≤ i < 线程束大小，通常为32）；
- 访问步长为 `step`（每个线程访问的元素间隔字节数）。

则线程i的**字节地址**为：  
```
字节地址 = base + i × step
```


#### 2. 转换为字地址
GPU共享内存中，1个“字（word）”通常为4字节（32位），因此**字地址**（即该字节地址属于第几个4字节字）为：  
```
字地址 = 字节地址 ÷ 4 = (base + i × step) ÷ 4
```


#### 3. 计算存储体编号
共享内存通常有 `N` 个存储体（N=32 是常见配置，与线程束大小匹配），存储体编号由字地址对 `N` 取模得到：  
```
存储体编号 = 字地址 % N = [(base + i × step) ÷ 4] % N
```


### 判断存储体冲突的规则
- 若**多个线程的存储体编号相同**，则发生冲突；
- 冲突程度：同一存储体被 `k` 个线程访问，称为 `k-路冲突`（k越大，冲突越严重）；
- 无冲突：所有线程的存储体编号均不重复（每个存储体仅被1个线程访问）。


### 示例（N=32，step=4）
- 字节地址 = base + i×4  
- 字地址 = (base + i×4)÷4 = base/4 + i  
- 存储体编号 = (base/4 + i) % 32  

由于i从0到31，32个线程的存储体编号为0~31（无重复），故**无冲突**。


通过以上公式可快速计算存储体编号，进而判断是否存在冲突及冲突程度。

## swizzling机制

swizzling是在不额外分配内存的情况下，通过将shared memory的数据进行重排来避免bank conflict。该方法通过逻辑坐标和物理坐标之间的映射来实现。

### 矩阵转置的swizzling机制

**native矩阵转置：非合并全局内存写入**

```cpp
/**
* 全局内存中的元素坐标是(row, col)
* 在2D线程块中，threadIdx.x对应的是列索引，threadIdx.y对应的是行索引
* 
* 下面两种方式中，从全局内存读取和向全局内存写入时，每个线程操作的元素是相同的
*/

// 方法1：
__global__ void matrix_trans_shm_direct(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ int s_data[32][32];

  if (row < M && col < N) {
    s_data[threadIdx.y][threadIdx.x] = dev_A[row * N + col];
    __syncthreads();
    
    // 直接使用转置坐标
    int n_row = col;  // 转置后行坐标 = 原始列坐标
    int n_col = row;  // 转置后列坐标 = 原始行坐标
    if (n_row < N && n_col < M) {
      // 直接从共享内存中读取当前线程对应的值
      // 此处是非合并的全局内存写入：线程在threadIdx.x方向连续，此时n_col（row）不变，n_row（col）递增，因此dev_B[n_row * M + n_col]写入的全局内存地址间隔为M*sizeof(int)
      dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];
    }
  }
}

// 方法2：
__global__ void matrix_trans_shm_direct(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ int s_data[32][32];

  if (row < M && col < N) {
    // 从全局内存中读取并转置存储在共享内存中
    s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];
    __syncthreads();
    
    // 直接使用转置坐标
    int n_row = col;  // 转置后行坐标 = 原始列坐标
    int n_col = row;  // 转置后列坐标 = 原始行坐标
    if (n_row < N && n_col < M) {
      // 直接从共享内存中读取当前线程对应的值
      // 此处是非合并的全局内存写入：线程在threadIdx.x方向连续，此时n_col（row）不变，n_row（col）递增，因此dev_B[n_row * M + n_col]写入的全局内存地址间隔为M*sizeof(int)
      dev_B[n_row * M + n_col] = s_data[threadIdx.x][threadIdx.y];
    }
  }
}
```

**native矩阵转置：合并全局内存写入**

```cpp
/** 
* 从全局内存读取和向全局内存写入时，每个线程操作的元素是不同的
*/

// 方法1：写入共享内存无bank冲突，读取共享内存有bank冲突
__global__ void matrix_trans_shm(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ int s_data[32][32];

  if (row < M && col < N) {
    s_data[threadIdx.y][threadIdx.x] = dev_A[row * N + col];
    __syncthreads();
    // 只对块进行转置，不对块内的元素进行转置
    int n_row = blockIdx.x * blockDim.x + threadIdx.y;
    int n_col = blockIdx.y * blockDim.y + threadIdx.x;
      
    if (n_col < M && n_row < N) {
      // 因为块内元素未进行转置，此时需要从共享内存中获取块内转置的元素，从而完成转置
      dev_B[n_row * M + n_col] = s_data[threadIdx.x][threadIdx.y];
    }
  }
}

// 方法2：写入共享内存有bank冲突，读取共享内存无bank冲突
__global__ void matrix_trans_shm(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ int s_data[32][32];

  if (row < M && col < N) {
    // 从全局内存中读取并转置存储在共享内存中
    s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];
    __syncthreads();
    // 只对块进行转置，不对块内的元素进行转置
    int n_row = blockIdx.x * blockDim.x + threadIdx.y;
    int n_col = blockIdx.y * blockDim.y + threadIdx.x;
      
    if (n_col < M && n_row < N) {
      // 因为块内元素未进行转置，此时需要从共享内存中获取块内转置的元素，从而完成转置
      dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];
    }
  }
}
```

**基于swizzling的矩阵转置：**

```cpp
__global__ void matrix_trans_swizzling(int* dev_A, int M, int N, int* dev_B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int s_data[32][32];

  if (row < M && col < N) {
    // 从全局内存读取数据，转置写入共享内存的逻辑坐标(row=x,col=y)
    // 其映射的物理存储位置位置(row=x,col=x^y)，x^y可保证写入共享内存不会发生bank冲突
    s_data[threadIdx.x][threadIdx.x ^ threadIdx.y] = dev_A[row * N + col];
    __syncthreads();
     
    // 只对块进行转置，不对块内的元素进行转置
    int n_col = blockIdx.y * blockDim.y + threadIdx.x;
    int n_row = blockIdx.x * blockDim.x + threadIdx.y;
      
    if (n_row < N && n_col < M) {
      // 其映射的物理存储位置(row=y,col=x^y)，x^y可保证写入共享内存不会发生bank冲突
      /** 因为块内元素未进行转置，此时需要从共享内存中获取块内转置的元素，从而完成转置，那么如何获取转置元素在共享内存中的物理坐标？
      * 获取转置元素在共享内存中的物理坐标，需要从逻辑坐标推理：对于逻辑坐标(row=x,col=y)，其转置的逻辑坐标是(row=y, col=x)
      * 逻辑坐标(row=y, col=x)相应的物理坐标是(row=y, col=x^y)，
      */
      dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];
    }
  }
}
```

### TMA的swizzling机制





