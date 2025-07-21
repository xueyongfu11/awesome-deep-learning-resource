[TOC]



## Forward过程中的CUDA Kernel

每一层transformer的前向代码如下：

```c
layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
residual_forward(l_residual2, residual, l_attproj, B*T*C);
layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
```



### 嵌入encoder

```c
void encoder_forward(float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C) {
    assert(C % 4 == 0);
    // 512 是一个在多种 GPU 架构中都能稳定工作的块大小。虽然新的 GPU 可能支持更大的块（像 1024），但 512 能在新旧硬件之间取得较好的平衡。
    const int block_size = 512;
    const int N = B * T * C;
    // 注意N / 4；wte和wpe均被强转成float4*类型
    const int grid_size = CEIL_DIV(N / 4, block_size);
    encoder_forward_kernel3<<<grid_size, block_size>>>((float4*) out, inp, (float4*) wte, (float4*) wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// use of float4 leads to using 128-bit LDG / STG instructions in SASS,
// very helpful in memory-bound kernels like encoder_forward
/**
 * float4类型占用16字节（4个float × 4字节）
 */
__global__ void encoder_forward_kernel3(float4* out,
                               const int* inp, const float4* wte, const float4* wpe,
                               int B, int T, int C) {
    int C4 = C / 4;    // 向量化后的通道维度
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 由kernel启动参数决定，即idx [0, B * T * C4)
    int N = B * T * C4;  // 输出总元素数（float4计数）
    if (idx < N) {
        int bt = idx / C4;   // 展平的(b,t)索引
        int b = bt / T;   // Batch索引 [0, B-1]
        int t = bt % T;   // 时间步索引 [0, T-1]
        int c4 = idx % C4;    // 向量化通道索引 [0, C4-1]
        int ix = inp[b * T + t];  // token ID (来自输入序列)

        // wte和wpe都是float4类型
        out[b * T * C4 + t * C4 + c4] = add_float4(wte[ix * C4 + c4], wpe[t * C4 + c4]);
    }
}
```

### LayerNorm

```c
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    // 共使用N * 32个线程，32个线程为一个warp，共使用了N个warp
    const int grid_size = CEIL_DIV(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// __restrict__：它主要是向编译器传达一个信息：被该修饰符修饰的指针是访问对应内存区域的唯一方式。
// 借助这一提示，编译器能够开展更多的优化工作，进而提升代码的执行效率
// cg 是 cooperative_groups 命名空间的别名，借助它能让线程协作变得更为简便。支持在不同的线程集合内进行同步操作，比如单个线程束（warp）、多个线程块，甚至整个网格。
__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    cg::thread_block block = cg::this_thread_block();
    // 将当前线程块划分为多个大小为 32 的子组（称为tile），实际上是将线程块按 warp 划分
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // warp.meta_group_size() 每个线程块被划分后的总子组数（即 warp 数量）
    // warp.meta_group_rank() 返回当前 warp 在其所在线程块内的索引（从 0 开始）
    // idx 表示当前warp在整个网络中的全局位置
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    // 每个warp负责一行数据，共C个元素
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    // warp.thread_rank() 用于标识线程在其所在线程束（warp）内的相对位置
    // 每个线程执行C/warp_size次运算
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    // Warp 级归约：cg::reduce 对 32 个线程的 sum 求和。
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    // 存储均值：Warp 的 0 号线程将 m 写入 mean[idx]（__stcs 流存储优化缓存）
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        // __ldcs：从指定内存地址加载数据到寄存器，并提示硬件该数据不会被重复使用；绕过一级缓存（L1 Cache），直接访问二级缓存（L2 Cache）或全局内存；适用只访问一次的数据
        // __stcs：将寄存器中的数据存储到指定内存地址，并提示硬件该数据不会被立即读取；绕过一级缓存，直接将数据写入二级缓存或全局内存；适用计算结果只需写入一次，后续不会立即读取
        float n = s * (__ldcs(x+c) - m);  // 归一化
        __stcs(o+c, n * weight[c] + bias[c]);   
    }
}
```

### matmul_forward

```c
// kernel 1 is the most naive matmul kernel
void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    int sqrt_block_size = 16;   // 16是通过精心设计的平衡各方面性能之后选取的常用值

    // 重要：在 CUDA 内核设计里，矩阵乘法的gridDim和blockDim的设计思路一般是基于输出矩阵的维度来计算的
    // gridDim决定了将输出矩阵划分为多大的子块
    // blockDim决定了分块的线程数
    // 下面的例子中，16*16的线程块要处理 8*16 *  8*16 的数据块大小，意味着每个线程处理8*8个数据
    dim3 gridDim(CEIL_DIV(B * T, 8*sqrt_block_size), CEIL_DIV(OC, 8*sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);   // 16*16大小的线程块
    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
}

/**
 * _launch_bounds__是 CUDA C/C++ 中的一个内核函数属性，用于向编译器提供线程块 (thread block) 的调度约束信息
 * 它有两个参数：
 * 最大线程块大小：指定内核函数允许的最大线程数量 (示例中为16*16=256)
 * 最小块数：指定每个 SM (Streaming Multiprocessor) 必须能够同时调度的最小线程块数量 (示例中为 2)
 */
__global__ void __launch_bounds__(16*16, 2) matmul_forward_kernel4(float* out,
                                                                   const float* inp, const float* weight, const float* bias,
                                                                   int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // each thread handles 8x8 elements; each block 128 by 128 elements.

    /**
     * 整体流程：
     * 1. 每个线程块block处理输出矩阵的128*128数据块
     * 2. 每个线程计算输出矩阵中的 8×8 子块
     * 3. 输出矩阵的128*128数据块，需要输入矩阵的128*C数据块和权重矩阵的128*C数据块的乘积来计算
     * 4. 输入矩阵的128*C数据块和权重矩阵的128*C数据块，同样需要再次分块，分块大小都是为128*32
     * 5. 每个线程需要遍历所有的128*C的子块，并处理子块(128*C)中属于当前线程的子子块，然后把计算结果相加，得到当前线程的输出子块(8*8)
     */                                                                

    // oc
    int oc = 8*(blockIdx.y * blockDim.y + threadIdx.y);

    // 缓存输入矩阵
    __shared__ float lhs_s[128][32];
    // 缓存权重矩阵
    __shared__ float rhs_s[128][32];

    // adjust our pointers for the current block
    // 计算输出矩阵的128*128分块所需要的输入矩阵的地址偏移
    inp += 128 * blockIdx.x * C;
    // 计算输出矩阵的128*128分块所需要的权重矩阵的地址偏移
    weight += 128 * blockIdx.y * C;
    // 指向输出矩阵的128*128分块的左上角，行偏移+列偏移
    out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    // vals是当前线程处理的8*8的输出子块
    float vals[8][8] = {};
    if(bias != NULL) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                float4 b = ld_vec(bias + oc + j);
                vals[i][j+0] = b.x;
                vals[i][j+1] = b.y;
                vals[i][j+2] = b.z;
                vals[i][j+3] = b.w;
            }
        }
    }

    // (16 * threadIdx.y + threadIdx.x)是每个线程在处理矩阵分块时的线程的起始索引（每个数据块的线程数是16*16）
    // 4*() 每次处理 4 列数据（通过float4向量操作），循环 8 次覆盖 32 列
    // si_start的目的是让不同线程访问共享内存的不同列片段，分散共享内存访问​​，避免多个线程同时访问相同的列（减少 Bank Conflicts）
    // si_start不能设置为0，否则会引起严重的bank冲突
    int si_start = 4*(16 * threadIdx.y + threadIdx.x);
    // 每个线程块通过循环遍历所有128×32子块。基于线程索引，获取128×32子块中负责处理的数据
    // so += 32表示每次加载并处理128*32的数据块偏移, 表示将输入矩阵和权重矩阵分成多个 32×128 的子块
    for (int so = 0; so < C; so += 32) {
        // 第一个同步：确保所有线程在开始加载新子块前，上一次迭代的共享内存数据已被使用。
        __syncthreads();

        // 加载数据到共享内存
        // threadIdx.x 0-15  threadIdx.y  0-15
        int xmod8 = threadIdx.x % 8;  // 9-7
        int xby8 = threadIdx.x / 8;  // 0-1 
        int xo = 4 * xmod8;  // xo 生成 0,4,8,...,28 的列偏移量，用于确定在共享内存中的列位置。后续每次根据地址加载4个连续浮点数。

        // y是行偏移量，行方向宽度是128
        /**
         * 256 个线程如何覆盖 128×32 矩阵？
         * 1. threadIdx.y的线程共16个，(2 * threadIdx.y + xby8)对应到连续的两行
         * 2. 相同threadIdx.y的线程每次读取4个数据，刚好可以把2行数据读完
         * 3. 16*16的线程组可以同时读取32行数据
         * 4. for循环让线程处理多行数据
         */
        for(int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
            // so是数据子块(128*32)左上角的列偏移，xo是子块(128*32)内的偏移
            // inp + y * C + so + xo是每次读取连续4个浮点数的首地址偏移
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        // 第二个同步：确保所有线程完成数据加载后，再开始基于共享内存的计算（避免数据未加载完就使用）
        __syncthreads();

        // 使用共享内存执行计算：本质是两个子块（128*32）的乘积，输出128*128的子块，每个线程负责其中8*8的子块
        for (int si = si_start; si < si_start + 32; si += 4) {  // 共循环8次，每次处理4列数据（通过float4）
            float4 rhs[8];
            // u + 8 * threadIdx.y，确保每个线程加载连续的 8 行
            // si % 32，列偏移，循环处理当前 32 列子块中的不同部分
            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);  // si % 32 -> 0,4,8...28
            }

            // ii + 8 * threadIdx.x，确保每个线程加载连续的 8 行
            for (int ii = 0; ii < 8; ++ii) {
                float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
                // 将inp的1个float4向量分别与weights的8个float4向量相乘
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    // 把每个线程计算好的子块(8*8)拷贝到输出矩阵out
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            st_vec(out + (8*threadIdx.x+i) * OC + 8*threadIdx.y + j, result);
        }
    }
}
```

### attention层

```c
void attention_forward(float* out, float* qkvr, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;   // 表示不进行缩放
    const float beta = 0.0f;   // 表示输出矩阵在进行矩阵乘法之前会先进行清零
    float* preatt = inp;

    // 在cublas的cublasSgemmStrideBatched函数中，batch是指多个独立的矩阵乘法运算，每个批次的矩阵乘法运算是相互独立的。
    // 在两个矩阵(B, NH, T, HS)的乘法中，每个batch是指独立的(T, HS)的两个矩阵的乘法，因此总的batch是B*NH
    cublasCheck(cublasSgemmStridedBatched(
        cublas_handle,                  // cuBLAS 库的上下文句柄，用于管理 GPU 资源和状态
        CUBLAS_OP_T,                    // 指定矩阵A(k)需要进行转置操作 (k^T)
        CUBLAS_OP_N,                    // 指定矩阵B(q)不需要转置
        T,                              // 结果矩阵C(preatt)的行数 (target sequence length)
        T,                              // 结果矩阵C(preatt)的列数 (source sequence length)
        HS,                             // 矩阵乘法的中间维度 (hidden size)
        &alpha,                         // 缩放因子，用于 A × B (通常为 1.0f/sqrt(HS))
        k,                              // 输入矩阵A的基地址 (key向量矩阵)
        HS,                             // 矩阵A的leading dimension (列数，转置前为HS)
        T * HS,                         // 相邻批次间矩阵A的内存步长 (每个batch包含T*HS个元素)
        q,                              // 输入矩阵B的基地址 (query向量矩阵)
        HS,                             // 矩阵B的leading dimension (列数为HS)
        T * HS,                         // 相邻批次间矩阵B的内存步长
        &beta,                          // 缩放因子，用于原有矩阵C (通常为0.0f)
        preatt,                         // 输出矩阵C的基地址 (注意力得分矩阵)
        T,                              // 矩阵C的leading dimension (列数为T)
        T * T,                          // 相邻批次间矩阵C的内存步长
        B * NH                          // 批量处理的总数 (batch_size × num_heads)
    ));

    // multiply all elements of preatt elementwise by scale
    /**
     * grid_size取值经验法则：
     * 1. 总线程数 ≥ GPU流处理器(SM)数 × 每个SM支持线程数
     * 2. Block数量​​ ≥ GPU SM数量 × ​​占用率系数​​（通常5-10倍）
     */
    float scale = 1.0 / sqrtf(HS);
    // 32应该是内核调优结果，即32个线程负责一行
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    // new approach: first cuBLAS another batched matmul
    float* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    // warp.meta_group_size() 每个线程块被划分后的总子组数（即 warp 数量）
    // warp.meta_group_rank() 返回当前 warp 在其所在线程块内的索引（从 0 开始）
    // idx是反向的warp索引，正向的warp索引是(blockIdx.x * warp.meta_group_size() + warp.meta_group_rank())
    // 注意：每个warp处理一行数据
    int idx = (gridDim.x - blockIdx.x - 1) * warp.meta_group_size() + warp.meta_group_rank(); // backward order
    if(idx >= N * T) {  // 线程共N*T*32个，warp共N*T个
        return;
    }
    int own_pos = idx % T;  // own_pos的取值范围(0, T), own_pos表示当前处理的数据行在序列中的时间步位置，在执行softmax时，只需计算到own_pos时间步即可，其他位置被mask
    int pos_by_4 = own_pos / 4;  // 按4元素分块的截止位置

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    // ix表示当前warp所指向的行首地址
    const float* x = inp + idx * T;  

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;  // 负的最大值
    float sumval = 0.0f;

    // reinterpret_cast 是C++中的强制类型转换
    // 基于online softmax的计算方式
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        // 更新最值
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        // 更新分段softmax计算公式中的分母sum_exp
        sumval *= expf(inv_temperature * (old_maxval - maxval));  // 减去分段最值，防止溢出
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    // 处理不能被4整除的元素
    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }
    // 计算当前warp的最大值
    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    // 用当前warp的最大值更新当前线程的sum_exp
    sumval *= expf(inv_temperature * (maxval - global_maxval));
    // 将当前warp的所有线程的sum_exp相加，计算处理行的sum_exp
    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    // warp中的线程同时写入数据
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // recalculation is faster than doing the round-trip through memory.
        // ​​直接重算​​：比存中间值更快的策略
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}
```

### residual

```c
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = __ldcs(&inp1[idx]) + __ldcs(&inp2[idx]);
    }
}
```

### gelu

```c
void gelu_forward(float* out, const float* inp, int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_forward_kernel<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

// 编译时常量，在编译阶段就把GELU_SCALING_FACTOR值计算好了
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}
```

### 最后的分类头

```c
// fused classifier: does the forward pass and first part of the backward pass
// we're passing dlosses = NULL, which will default them to 1.0f/(B*T), i.e. uniform loss
// 执行完之后acts.output存储logits的梯度
// 融合的分类头：前向计算和部分反向计算
fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);

// replaces logits with logit gradients
void fused_classifier3(float* logits, float* losses,
                      const float* dlosses, const int* targets,
                      int B, int T, int V, int P) {
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel3<<<grid_size, block_size>>>(logits, losses, NULL, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}


// same as 2 but not using float4 (see dev/cuda/classifier_fused.cu)
// will _update_ logits to logit gradients
__global__ void fused_classifier_kernel3(float* logits, float* losses, float* probs,
                                         const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x;
    // ix是真实类别标签
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        // 只计算真实类别标签所在位置的prob，其他位置均为0
        // sp.Offset/sp.Scale：softmax的数值稳定参数
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        // 计算loss：-logsumexp
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    // dloss是​​损失函数对当前样本的梯度(样本是指每个token的损失)，表示​​整体损失对当前样本损失​​的偏导数
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float v = __ldcs(&logits_vec[i]);
        float prob = expf(v - sp.Offset) * sp.Scale;
        if (probs != NULL) {
            probs[idx * P + i] = prob;
        }
        float indicator = (i == ix) ? 1.0f : 0.0f;
        // (prob - indicator)：计算​​样本级别的梯度​​ (交叉熵损失对 logits 的导数)
        // 乘以 dloss：将样本梯度按比例缩放到​​全局梯度​
        logits[idx * P + i] = (prob - indicator) * dloss;
    }
}
```



## backward过程中的CUDA Kernel

每一层transformer的反向传播代码如下：

```c
// backprop this layer
// dresidual是上游回传的梯度
matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
// dl_bt4c是回传到gelu_backward层的上游梯度
// l_fch是当前层的输入，并把计算到的l_fch梯度保存在dl_bt4c中
gelu_backward(dl_bt4c, l_fch, dl_bt4c, B*T*4*C);
// dl_btc是计算到的输入的梯度
// dl_fcw、dl_fcb是计算的当前层权重和bais的梯度
matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
// layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
// 由于残差结构，layernorm层输入的梯度由两部分组成
layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);
// we more B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
float* buffer_a = l_atty;
float* buffer_b = l_fch;        // this is B x T x 4C, so even larger than what we need

attention_backward(dl_bt4c, buffer_b, dl_preatt, scratch, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH);
matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
// layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
```

### matmul_backward

```c
void matmul_backward(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    /**
     * dout是上游反传下来的梯度
     * inp是当前层的输入
     * weight是当前层的权重（word embedding）
     * dinp是计算得到的输入的梯度 (B,T,C)
     * dweight是计算得到的权重的梯度
     */
    float one = 1.0f;
    float zero = 0.0f;
    // backward to input, uses = in the backward pass (set the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &one, weight, C, dout, OC, &zero, dinp, C));
    // backward to weight, uses += in the backward pass (accumulate the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &one, inp, C, dout, OC, &one, dweight, C));
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        const int block_size = 1024;
        // OC是padded_vocab_size
        const int grid_size = OC / 32; // for now, OC must be divisible by 32 for this kernel to work
        // bias的梯度计算是对dout的每一列求和（最后一个维度求和）
        matmul_backward_bias_kernel4<<<grid_size, block_size, block_size * sizeof(float)>>>(dbias, dout, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}


// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
// the idea is to employ one block to reduce along several columns,
// where each block has a width of 32 columns to ensure coalesced access.
// at the end we accumulate the reductions performed by the warps in each block via shared memory
__global__ void matmul_backward_bias_kernel4(float* dbias, const float* dout, int B, int T, int OC) {
    // this kernel is launched with 1D grid_dim of OC/32
    // for example let's say block_size is 128
    extern __shared__ float smem[]; // of size block_size (128)
    const int warp_id = threadIdx.x / warpSize; // warp index in the block, 0,1,2,3
    const int lane_id = threadIdx.x % warpSize; // thread index in the warp, 0,1,2,...,31
    const int tl = blockIdx.x * warpSize; // pointer to the start column for this block
    const int vstep = blockDim.x / warpSize; // number of warps in a block, e.g. 4

    // pointer to the start of the column for one lane of threads
    // so e.g. 4 threads (of the same lane_id) will reduce this one column
    const float* dout_col = dout + tl + lane_id;

    // column reductions by looping through the rows
    // each of the 4 threads offsets by its warp_id and then skips by vstep
    // together these 4 threads cover all B*T rows of this (lane_id) column
    // importantly, consecutive threads (in threadId) are processing adjacent columns,
    // leading to a coalesced memory access pattern
    float dout_sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep) {
        dout_sum += dout_col[row * OC];
    }
    smem[lane_id + warp_id * warpSize] = dout_sum;
    __syncthreads();

    // warp_id 0 reduces the shared memory column-wise, linearly
    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum += smem[lane_id + j * warpSize];
        }
        dbias[tl + lane_id] += dout_sum;
    }
}
```

### gelu_backward

```c
void gelu_backward(float* dinp, const float* inp, const float* dout, const int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

__global__ void gelu_backward_kernel(float* dinp, const float* inp, const float* dout, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];
    }
}
```

### layernorm_backward

```c
void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const  float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    /**
     * dout是上游传来的梯度
     * inp是ln层的输入
     */
    const int block_size = 512;
    const int N = B * T;
    // 从warp的视角看，一个warp处理一行
    const int grid_size = CEIL_DIV(32*N, block_size);
    size_t shared_mem_size = 2 * C * sizeof(float);
    layernorm_backward_kernel2<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}

// uses shared memory instead for the reduces
__global__ void layernorm_backward_kernel2(float* dinp, float* dweight, float* dbias,
                                           const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                                           int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // idx是warp的索引，idx (0, B*T]
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int N = B * T;
    if(idx >= N) { return; } // thread guards

    int b = idx / T;
    int t = idx % T;

    // 当前线程处理数据的首地址
    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll  //编译器优化手段，用来对循环进行展开
	for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    // first: two reduce operations
    // 首先计算出梯度计算时所需要的中间统计量
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
    dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias_shared[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight_shared[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
    __syncthreads();

    // write to global memory
	for(int i = threadIdx.x; i < C; i+= blockDim.x){
        atomicAdd(&dbias[i], dbias_shared[i]);
        atomicAdd(&dweight[i], dweight_shared[i]);
	}
}
```

### attention_backward

```c
// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* scratch,
                        const float* dout,
                        const float* qkvr, const float* att,
                        int B, int T, int C, int NH) {
    const int block_size = 256;
    int HS = C / NH; // head size
    const float one = 1.0f;
    const float zero = 0.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;
    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    // 改变上游梯度dout的形状，不改变梯度大小
    // scratch是临时存储空间，用于中间结果，数据形状是(B,NH,N,HS)
    unpermute_kernel_backward<<<num_blocks, block_size>>>(scratch, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    // backward into datt
    // 计算注意力矩阵的梯度
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS, scratch, HS, T * HS, &zero, datt, T, T * T, B * NH));
    // backward into dv
    // 计算V矩阵的梯度
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, scratch, HS, T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));
    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    // block_size一般固定设置为32的倍数(128,256,512,1024)
    // grid_size需要根据数据进行设置
    // dim3(T / 4, B * NH)意味着每个block处理4行数据，大小为4*T
    softmax_autoregressive_backward_kernel<<<dim3(T / 4, B * NH), 256>>>(dpreatt, datt, att, B, T, C, scale);
    cudaCheck(cudaGetLastError());
    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS, dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));
    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS, dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    // permute_kernel_backward是unpermute_kernel_backward的逆操作
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

__global__ void softmax_autoregressive_backward_kernel(float* dpreatt, const float* datt, const float* att,
                                                       int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int idx = blockIdx.y;  // idx (0, B*NH)
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx.x;  // blockIdx.x (0, T/4)

    att += idx * T * T;   // 当前线程处理的数据内存地址
    datt += idx * T * T;
    dpreatt += idx * T * T;

    // 共享内存初始化
    if (warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    // 每个线程参与处理4行数据
    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const float* att_bth = att + t * T;
        const float* datt_bth = datt + t * T;
        float* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        // 共256/32=8个warp，共享内存block_acc中前8个非零，其余均为0
        block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
        // block.sync()是 CUDA 协作组（Cooperative Groups）API 中的​​线程块级同步操作
        block.sync();
        // local_sum是全局的加和
        local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});

        for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
            __stcs(dpreatt_bth + t3, scale * acc);
        }
    }
}
```

### encoder_backward

```c
void encoder_backward(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_backward_kernel<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

// really bad naive kernel with atomicAdd
__global__ void encoder_backward_kernel(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        atomicAdd(dwte_ix, *dout_btc);
        atomicAdd(dwpe_tc, *dout_btc);
    }
}
```

## AdamW参数更新

```c
void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        cudaCheck(cudaMalloc((void**)&model->m_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&model->v_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->m_memory, 0, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, model->num_parameters * sizeof(float)));
        printf("allocated %zu MiB for AdamW optimizer state m\n", (model->num_parameters * sizeof(float)) >> 20);
        printf("allocated %zu MiB for AdamW optimizer state v\n", (model->num_parameters * sizeof(float)) >> 20);
    }

    int block_size = 512;
    int num_blocks = CEIL_DIV(model->num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel2<<<num_blocks, block_size>>>(model->params_memory, model->grads_memory, model->m_memory, model->v_memory,
                                              model->num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

__global__ void adamw_kernel2(float* params_memory, float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}
```

































































