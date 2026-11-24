# NCU 指标与瓶颈诊断

## 1. 指标口径

- **SM Active**：SM 有驻留 warp 的活跃情况；活跃不意味着持续发射指令。
- **SM Throughput**：计算资源接近其峰值吞吐的程度，需看具体管线分解，不等于模型 FLOPS 利用率。
- **Occupancy**：驻留 active warp 数占硬件可驻留上限的比例，不代表 warp 已就绪。
- **Eligible Warps**：当前可发射下一条指令的 warp。
- **Issue Active**：调度器有指令发射的周期比例。

指标后缀也很重要：`pct_of_peak_sustained_elapsed` 以 elapsed cycles 为口径，`pct_of_peak_sustained_active` 以 active cycles 为口径，不能直接混用。

## 2. 优先关注的指标与页面

| 指标 / 页面                          | 代表什么                          | 分析用途                        |
| ------------------------------------ | --------------------------------- | ------------------------------- |
| Duration：`gpu__time_duration.sum` | 单次 kernel 时长                  | 判断优化是否实际有效            |
| SpeedOfLight                         | 计算与内存资源吞吐概览            | 初步识别饱和资源                |
| ComputeWorkloadAnalysis              | Tensor、FP32、特殊函数等计算管线  | 找具体计算限制，不能只看 Tensor |
| DRAM Throughput                      | 显存吞吐相对峰值                  | 判断 HBM 带宽是否接近上限       |
| DRAM Read / Write Bytes              | 实际显存读写量                    | 判断数据搬运是否减少            |
| L1 / L2 吞吐与命中率                 | 缓存压力和复用                    | 区分缓存与显存瓶颈              |
| Shared Memory / Bank Conflicts       | 共享内存访问效率                  | 查共享内存吞吐与冲突            |
| Local Memory / Spill                 | 线程私有存储流量与寄存器溢出      | 查寄存器压力引起的额外访存      |
| LaunchStats                          | grid、block、寄存器和共享内存需求 | 查任务数量、资源限制            |
| Occupancy                            | 理论及实际驻留 warp 比例          | 查隐藏延迟能力与负载不均        |
| SchedulerStats                       | active / eligible / issued warp   | 判断调度器是否缺少就绪工作      |
| WarpStateStats / SourceCounters      | 停顿原因及源码位置                | 在发射不足时追踪原因            |

注意：Memory Throughput 是内存层级汇总，不能等同于 HBM 带宽。低缓存命中率对于流式读取不一定有问题；local memory 也不等同于 spill，需要源码和指令证据。

本机可用指标和 section 可通过 `ncu --query-metrics`、`ncu --list-sections` 查询。

## 3. 计算吞吐、带宽和延迟的区别

| 观察组合                         | 初步判断         | 下一步                             |
| -------------------------------- | ---------------- | ---------------------------------- |
| 某计算管线接近上限               | 计算吞吐受限     | 看具体指令路径，减少工作或改善算法 |
| DRAM 接近可达带宽                | 显存带宽受限     | 减少读写、提高数据复用             |
| DRAM 不高，L1/L2/shared 接近上限 | 片上内存层级受限 | 查该层流量和访问效率               |
| 各类吞吐都低、eligible warp 少   | 延迟或并行度问题 | 查依赖、同步、工作量及流水         |

没有通用的“超过某个百分比就一定受限”阈值。应结合可达基准、资源分解、持续时间和对照实验。

## 4. Roofline 判断

设计算量为 F FLOPs，实际 HBM 传输量为 Q Bytes，可达计算吞吐为 P FLOP/s，可达显存带宽为 B Byte/s：

```text
理想计算时间：T_compute = F / P
理想访存时间：T_memory  = Q / B
计算强度：    AI = F / Q
设备转折点：  P / B
理想时间下界：max(T_compute, T_memory)
```

- `F/P` 大很多：潜在计算限制。
- `Q/B` 大很多：潜在显存带宽限制。
- 实测时间远大于二者：查延迟、同步、并行度、其他管线或模型假设。

这是理想下界而非精确时间模型。Tensor Core kernel 要选匹配精度、指令和稀疏性口径的计算上限；Q 使用实际传输量，不能只用输入张量大小。分析 L2/L1 时，要对应使用该层的流量和带宽。

例如假设 P=100 TFLOP/s、B=2 TB/s：

|          F |    Q | 理想计算时间 | 理想访存时间 | 潜在限制 |
| ---------: | ---: | -----------: | -----------: | -------- |
| 100 GFLOPs | 1 GB |         1 ms |       0.5 ms | 计算     |
|  10 GFLOPs | 1 GB |       0.1 ms |       0.5 ms | 带宽     |

## 5. SM 吞吐（SM Throughput）低的排查顺序

1. **先看 NSYS 时间线**：kernel 之间是否有空白？若有，查 CPU 发射、数据加载、通信与同步。
2. **再看内存层级**：若 DRAM、L2 或 shared 已饱和，SM 低可能是相应瓶颈的结果。
3. **查任务量**：grid 的 block 数是否足够，是否有明显拖尾或工作量不均。
4. **查 occupancy 限制**：理论低时看寄存器和共享内存；实际远低于理论时看负载与调度。
5. **查就绪情况**：active warp 多、eligible warp 少，说明多数 warp 在等。
6. **最后定位 stall 和源码**：查等待的生产者、数据依赖及流水。

不要仅为提高 occupancy 而降低寄存器用量：可能引入 spill 并降低性能。低 occupancy 与高吞吐可以同时存在。

## 6. 主要 stall 的含义

| 原因                       | 含义                                              | 优先检查                              |
| -------------------------- | ------------------------------------------------- | ------------------------------------- |
| long_scoreboard            | 等待 L1TEX 路径操作的依赖，例如 global/local load | 加载延迟、缓存、spill、并发访存和流水 |
| short_scoreboard           | 等待其他可变延迟操作，常涉及共享内存              | bank conflict、共享内存依赖           |
| barrier                    | 等待同步                                          | 线程工作不均、同步安排                |
| wait                       | 固定延迟执行依赖                                  | 指令依赖链、独立指令数量              |
| math_pipe_throttle         | 相应数学管线无法接收更多指令                      | 指令混合、具体管线吞吐                |
| lg_throttle / mio_throttle | 相应队列或发射路径压力                            | 访存和共享内存指令密度                |
| not_selected               | warp 已就绪，但选中了别的 warp                    | 通常不能直接视为性能问题              |

只有调度器无法持续发射时，才优先深入 stall。不同页面的 stall 百分比可能有不同分母，不等于 kernel 时间占比，也不等于可消除的时间。`long_scoreboard` 高不证明 HBM 带宽已满；采样位置可能是等待结果的消费者指令。

典型组合：SM 和内存吞吐都低、occupancy 不低、eligible warp 少、long_scoreboard 高，支持“驻留工作存在但供数延迟暴露”的判断，需进一步验证。
