# vLLM 模型编译：从 torch.compile 到分段 CUDA Graph

## 先看整体流程

先说结论：`torch.compile` 不会改变模型在算什么，它改变的是模型的执行方式。PyTorch 原本在 Eager 模式下逐行执行 Python、逐个启动算子；有了 `torch.compile`，这些张量计算会先被整理成图，再交给编译器统一优化。实际收益主要来自两处：

- **调度成本**：大量小算子意味着大量 CPU 发起的 kernel launch；融合和 CUDA Graph replay 可以显著降低该成本。
- **访存成本**：相邻算子若各自读写全局显存，中间结果会反复落到 HBM/Global Memory；融合可以让中间值留在寄存器或片上存储中。

典型链路如下：

```text
Python / nn.Module
  → TorchDynamo：捕获可编译的张量片段，形成 FX Graph
  →（训练时）AOT Autograd：提取前向图与反向图
  → PrimTorch / ATen 分解：化为更基础的原语
  → TorchInductor：图优化、算子选择、代码生成与自动调优
  → Triton / CUDA / C++ / cuBLAS / CUTLASS 等可执行实现
```

vLLM 在 Dynamo 和 Inductor 之间插入了自己的编译后端。它会按规则切分 FX 图，分别编译和缓存各个片段，只对适合的部分录制 CUDA Graph。这样既能保留变长请求需要的灵活性，又能优化吞吐和延迟。

# 一、torch.compile 基础

## 1. 为什么需要计算图

可以把计算图看成一张 DAG：节点是加法、MatMul、归一化等算子，边表示张量之间的数据依赖。单独看一个算子，编译器能做的事情有限；看到整段数据流以后，它才有机会做这些优化：

- 死代码删除、公共子表达式消除与无效操作消除；
- Pointwise、归约、归一化等算子的融合；
- 中间张量生命周期分析与内存复用；
- 合理的执行顺序、布局与底层 kernel 选择。

Eager 模式的优点是灵活、容易调试，代价是编译器事先看不到一张稳定的全局图。`torch.compile` 做的事情，可以概括为：尽量保留动态图的编程体验，同时从运行中的 Python 代码里提取出可优化的图。

## 2. 图捕获的难点与三种思路

动态图模型并不是天然就能被完整捕获。常见障碍有下面几类：


| 障碍            | 例子                              | 后果                                       |
| --------------- | --------------------------------- | ------------------------------------------ |
| 数据相关控制流  | `if x.sum() < 0:`                 | 分支取决于运行时张量值，不能简单预先固定   |
| 动态形状        | 变长序列的 RoPE、不同`num_tokens` | 可能产生新的 guard、重新捕获或需要符号形状 |
| 非张量/外部操作 | NumPy 随机数、I/O、C++/Rust 扩展  | 图无法连续，通常形成 graph break           |

几种常见的图捕获方案，对这些问题的处理能力并不相同：


| 方案                          | 核心做法                                         | 主要局限                                       |
| ----------------------------- | ------------------------------------------------ | ---------------------------------------------- |
| `torch.fx.symbolic_trace`     | 用`Proxy` 符号执行 Python                        | 不能可靠处理依赖真实张量值的 Python 分支       |
| `torch.jit.trace`             | 用示例输入实际执行一次并记录路径                 | 只记录这一次走到的分支；输入变化可能语义不正确 |
| `torch.compile` / TorchDynamo | 在 Python frame 执行时拦截字节码，提取可编译片段 | 仍会受动态值、外部调用和不支持语义限制         |

`torch.fx.symbolic_trace` 使用的输入不是真实 Tensor，而是 `Proxy` 符号对象。运行到 `x + 1` 时，它不会真的做加法，只会在 FX 图里记下一个节点。这种方式适合流程不依赖输入值的模型，但遇到 `if x.sum() > 0:` 就没法继续，因为 Python 无法判断一个 `Proxy` 条件是真是假。

`torch.jit.trace` 走的是另一条路：拿真实样例执行一次，把当时经过的路径记录下来。问题也很明显——换一组输入后，如果程序本应走另一条分支，之前记录的图可能就不再正确。

### TorchDynamo：以 graph break 换取覆盖率

TorchDynamo 基于 CPython 的 frame evaluation 机制，在代码运行时分析字节码、追踪张量运算，最后输出一个或多个 `torch.fx.GraphModule`。它不要求整个函数一次性全部进图。遇到无法安全处理的操作时，Dynamo 会先结束当前图，回到普通 Python 执行，之后再尝试捕获下一段。这个中断点就是 **graph break**。

每份编译结果还会带上一组 **guards（守卫）**，用来检查当初编译时的假设是否仍然成立，比如输入的 shape、dtype、device，或者某个 Python 对象的类型。再次调用时，Dynamo 会先检查缓存版本的 guards；有版本匹配就直接复用，没有版本匹配才重新追踪和编译。

这件事对在线服务很重要。如果生产流量突然出现未覆盖的 shape，那一次请求可能恰好承担重编译成本，P99 延迟就会出现尖刺。动态 shape、尺寸分桶和启动预热，都是在设法把这部分成本移出真实请求链路。

实践中通常要同时关注两件事：一是减少 graph break，不要在热点 `forward` 里混入打印、I/O 或随意的 Python/NumPy 操作；二是控制 shape 的种类，尽量让输入落入有限、可预测的桶（bucket）。

## 3. AOT Autograd、PrimTorch 与 TorchInductor

### AOT Autograd：训练场景的前反向联合优化

Eager Autograd 会在前向过程中逐步记录反向传播需要的信息。AOT Autograd 则提前提取前向图和反向图，让后端有机会一起编译，并权衡中间值保存、重计算和内存占用。

推理一般用不到反向图，这里只需记住三者的分工：Dynamo 负责捕获，AOT Autograd 主要服务于训练中的反向图，Inductor 负责把图变成高效的执行代码。

### PrimTorch：降低后端复杂度

高层 ATen 算子很多，如果后端逐个适配，实现会非常复杂。PrimTorch 通过 decomposition，把它们改写成更小、更稳定的原语组合。例如，`log2(x)` 可以表示成 `log(x) * (1 / log(2))`。不同 API 最终落到一组相对有限的基础操作上，后端就不必为每个复杂算子单独实现 lowering 和代码生成。拆开后的数据依赖也更清楚，方便做公共子表达式消除、类型和广播分析，以及统一的模式匹配。

可以把 PrimTorch 理解成“统一表达”，把 Inductor 理解成“安排执行”。编译过程经常先把高层操作拆开，分析清楚之后再重新融合。不过也不会把所有复杂算子都拆到底。GEMM、attention 和卷积通常保留为整体，这样才能继续调用成熟的库或专用 kernel。

### TorchInductor：优化、选择与生成

TorchInductor 是 `torch.compile` 的默认后端。它先把 FX 算子图降低成循环级 IR，用来描述元素索引、读写和归约；随后由调度器决定哪些操作可以融合、按什么顺序执行，以及中间缓冲区怎样复用。

例如，`relu(x + y)` 在 Python 里是两个操作，降低后可以变成一个遍历全部元素的循环，最后生成一个融合 kernel。这样 `x + y` 的中间结果就不必先写回全局显存。

生成代码时，并不是在 Triton、CUDA 和 C++ 之间简单三选一，而是根据设备和计算特点走不同路径：

- GPU 上规则的 pointwise、broadcast 和 reduction 通常生成 Triton 源码；Inductor 生成指针、索引、mask、load/store 和数学表达式，再由 Triton 编译器产生 GPU 代码。
- CPU 上同样的循环 IR 通常生成 C++/OpenMP 循环，并结合多线程和 SIMD 向量化。
- GEMM、卷积等复杂操作可以在 Triton 模板、专用模板和 cuBLAS/CUTLASS 等外部实现之间选择；还可生成多组 block size、warps、stages 候选，在目标硬件上 autotune 并缓存最优结果。

Inductor 最后还会生成 wrapper，处理动态 shape、缓冲区分配和复用、kernel 启动以及外部库调用。把整条链路连起来，就是：图 → 循环 IR → 融合与调度 → kernel 或库调用 → wrapper。

这里的“编译”不等于“凡事都生成一个新 kernel”。更常见的做法是融合适合融合的小算子，同时把大矩阵乘交给专门的高性能库。最终能提速多少，还要看瓶颈究竟在 attention、GEMM、通信，还是显存带宽。

## 4. CUDA Graph 与 torch.compile 的关系

CUDA Graph 会把一段固定的 GPU 工作录制下来，之后直接 replay，省去 CPU 逐个提交 kernel 的开销。decode 阶段经常是小 batch、短 kernel，而且调用频繁，因此很适合这种方式。

不过，CUDA Graph 的约束也更强。捕获范围里的 CUDA 操作必须支持捕获，控制流要稳定，输入输出的内存地址和执行结构也要能够复用。动态 shape、动态内存分配，以及某些自定义算子或 attention 后端，都可能不满足这些条件。

因此真实系统很少只在“整图捕获”和“完全不用”之间二选一。更实用的方式是找出适合捕获的局部图，为它们管理稳定的缓冲区，其余部分继续走普通执行路径。

## 5. 最小使用与诊断建议

```python
import torch

model = model.eval().cuda()
compiled_model = torch.compile(model, backend="inductor")

with torch.inference_mode():
    output = compiled_model(inputs)
```

评估时要把冷路径和热路径分开。第一次调用可能包含 Dynamo 捕获、代码生成和 autotune，稳定运行后的请求才更接近线上表现。通常应先验证数值正确性，再观察 TTFT、TPOT、吞吐、显存、重编译次数和 P95/P99 延迟。只比较一次冷启动耗时，很容易得出错误结论。

# 二、vLLM 编译架构和优化技术

## 1. 与 torch.compile 的关系及推理矛盾

先厘清几个容易混淆的概念。`torch.compile` 是 PyTorch 的编译入口和整体框架：TorchDynamo 从模型的 Python `forward` 中捕获 FX 图，默认再由 TorchInductor 优化图并生成 GPU/CPU 代码。因此，TorchInductor 是 `torch.compile` 的默认编译后端，而不是另一套并列框架。

vLLM 没有替代这条链路，而是在 Dynamo 和 Inductor 之间加入了面向大模型推理的处理：它先对 FX 图做切分、改写和尺寸专用化，再把子图交给 Inductor，最后由 vLLM 运行时管理普通执行和 CUDA Graph replay。

```text
模型 forward
  → TorchDynamo：捕获并生成 FX 图             ┐
  → VllmBackend：切图、改图、编排             ├─ torch.compile 编译链路
  → TorchInductor：优化并生成底层代码          ┘
  → vLLM 运行时：选择普通执行或 CUDA Graph replay
```

简单说，`torch.compile` 解决的是“怎样从 Python 得到并编译计算图”，vLLM 编译架构解决的是“这张图在 LLM serving 中应该怎样切、怎样优化，以及运行时怎样执行”。之所以需要这层定制，是因为 LLM serving 里有两类差异很大的工作负载：

- **prefill**：处理长 prompt，token 数与序列长度变化大，attention 形状和工作量高度动态；
- **decode**：每轮通常只新增少量 token，但请求会动态合批，CPU launch 开销相对更显著。

如果把整张图完全静态化，执行效率会很高，但很难适应连续批处理和变长序列；如果全部使用 Eager，灵活性够了，又会损失算子融合和 launch 优化。

vLLM 采用了一种折中方案：用动态形状的通用图覆盖变化较大的输入，为少数常见尺寸准备专用图，再对适合的片段使用 CUDA Graph。后面的编译架构基本都围绕这个思路展开。

## 2. vLLM 后端的分层职责

主要组件可以整理成下面这条链路。vLLM 更新较快，具体类名在不同版本中可能会变化。

```text
torch.compile(model, backend=VllmBackend)
  → VllmBackend：接收完整 FX 图，按 splitting_ops 切图
  → PiecewiseCompileInterpreter：遍历并调度各片段
  → PiecewiseBackend：按编译范围保存/分发片段的 runnable
  → CompilerManager / CompilerInterface：缓存与具体编译器适配
  → Inductor adaptor：调用 Inductor 生成或加载执行产物
  → CUDAGraphWrapper：对满足条件的片段 capture / replay
```

- `VllmBackend` 是 PyTorch 调用 vLLM 后端的入口，主要负责切图和编排，并不直接执行模型。
- `PiecewiseBackend` 按 shape 或 token 范围保存子图的编译结果。运行时拿到当前尺寸后，很快就能找到对应的可执行版本。
- `CUDAGraphWrapper` 用批次描述符作为 key。某类批次第一次出现时完成捕获，以后遇到同类批次就直接 replay；条件不满足时，则退回底层的普通执行路径。
- `CompilerInterface` 隔开了上层策略和 Inductor 等具体编译器，缓存机制、编译 pass 或底层实现发生变化时，不必连带修改整个系统。

这样拆层以后，模型只描述“要算什么”，vLLM 后端决定怎样切图和改图，Inductor 负责生成代码，运行时再决定何时 replay。attention、量化和通信相关的优化可以放在统一的编译层里，不需要分别写进 Llama、Qwen 等每一个模型实现。

## 3. 动态通用图与静态尺寸专用图

vLLM 通常按一次 `forward` 处理的**展平 token 数**来划分编译范围。这个数字不一定等于请求数；只有在纯 decode 阶段，每条活跃序列通常各生成一个 token 时，两者才比较接近。

- **动态范围图**可以覆盖多个输入尺寸，编译数量和冷启动成本较低，但编译器掌握的静态信息更少，优化往往相对保守。
- **`compile_sizes` 专用图**针对某些固定 token 数生成静态版本，例如只为 size 4 编译 `[4, 4]`。shape 完全确定后，Inductor 有机会找到更好的 kernel 和 autotune 配置。
- 运行时，`PiecewiseBackend` 会优先匹配固定尺寸，再查找能够覆盖当前值的动态区间。配置和调度必须保证所有可能出现的尺寸都有可用路径。

`compile_sizes` 不宜凭感觉列一长串数字。固定尺寸越多，命中专用优化的机会越大，但首次编译、autotune、缓存和预热的成本也会增加。大多数场景可以先使用默认配置；确实需要调优时，再根据生产流量中的 token 数直方图，挑少量高频小尺寸做基准测试。

## 4. 分段编译（Piecewise Compilation）

### 为什么要在 attention 处切图

attention 的动态性通常比较强，尤其是在变长 prefill、prefill-decode 混合批次，或者使用某些特殊 attention 后端时，很难和周围计算一起放进同一张 CUDA Graph。vLLM 因而常把 attention 当作切分点：

```text
[embedding / norm / MLP 等] → [attention] → [token-wise 计算] → [attention] → ...
        可编译 + 可选 CUDA Graph      保持动态执行          可编译 + 可选 CUDA Graph
```

attention 本身仍然可以使用专用高性能实现，也可能由 Inductor 处理。这里切开的主要是 CUDA Graph 捕获边界：动态性强的 attention 继续使用灵活的执行路径，attention 前后的 token-wise 计算则可以享受编译和 replay 的收益。

### 图分割规则

`splitting_ops` 用来定义这些切点，常见例子是 unified attention。切点算子通常会被隔离出来，它前后的普通计算各自形成子图。

切分粒度需要权衡：太粗时，图中容易混入无法捕获的操作；太细时，调用边界和中间张量流转又会变多，抵消融合带来的收益。所以切点要结合算子兼容性、动态程度和 profiler 数据来定，不能只看模块名称机械拆分。

## 5. 分段 CUDA Graph：捕获与重放

对一个适合 CUDA Graph 的片段，运行时大致会经过以下步骤：

1. 读取本次 forward 的 batch descriptor 与 CUDA Graph 模式；
2. 模式不匹配、没有 forward context 或不满足条件时，直接执行 runnable；
3. 若该 descriptor 尚未捕获，准备稳定输入/输出缓冲并 capture；
4. 若已捕获，复制/更新输入缓冲后 replay，复用既有 launch 序列。

`cudagraph_capture_sizes` 指定要捕获哪些尺寸。尺寸列得多，replay 的覆盖率可能更高，但启动时间和显存占用也会随之上升；列得太少，很多请求又只能回退到普通路径。调优时应以线上 token 数分布为依据，同时观察实际的 replay 命中率。

## 6. 编译缓存与部署策略

冷启动期间会产生 FX 图、变换后的 Python 路径、编译产物，以及可能的 autotune 结果。vLLM 会把这些内容放入编译缓存。根据官方 V1 文档，缓存 key 会考虑相关配置、PyTorch 配置和模型 `forward` 关联代码，避免错误复用不兼容的产物。

部署时可以按下面的方式处理：

- 在与线上**GPU 架构、CUDA、PyTorch、vLLM、驱动与关键配置一致**的环境中预热；
- 将完整的编译缓存作为只读工件分发或挂载给弹性实例；
- 版本、硬件或编译配置改变时让缓存失效并重新预热，不能跨不一致环境盲目复用；
- 排障时可临时禁用 vLLM 编译缓存（`VLLM_DISABLE_COMPILE_CACHE=1`），以区分缓存问题和编译问题。

最好在实例接收流量之前完成所需的编译和预热。否则，第一个命中新 shape 的用户请求可能触发编译，形成难以预测的尾延迟。线上评估不能只看平均吞吐，也要关注 P99 是否稳定。

## 7. 自定义编译 Pass 的意义

编译 Pass 可以理解为一套处理计算图的规则。TorchDynamo 捕获 FX 图以后、Inductor 生成 GPU 代码以前，vLLM 会在图中查找特定的节点模式，并进行融合、替换、简化或删除。例如，把 `RMSNorm → Quant`、`RoPE → KV-cache 更新`、`AllReduce → RMSNorm` 替换成融合实现，或者去掉没有意义的 clone/no-op。这样可以减少 kernel 启动、中间张量和显存读写。

Pass 作用在 FX 图上，不依赖某一个模型的 `forward` 写法。只要 Llama、Qwen 等模型最终产生了相同的图模式，就可以复用同一条优化规则。具体 GPU 实现仍由 Inductor/Triton 生成，模型代码不用掺入这些编译细节。

当然，融合并不保证一定更快。它可能增加寄存器压力、降低 occupancy，也可能破坏原本很高效的库调用路径；同一项优化在不同 GPU 和 shape 上也会有不同表现。因此自定义 Pass 必须配套语义等价测试、代表性 shape 的性能回归，并提供关闭开关。

## 8. 调优决策表


| 现象              | 优先检查                                    | 常见动作                                                  |
| ----------------- | ------------------------------------------- | --------------------------------------------------------- |
| 首次启动很慢      | 编译、autotune、CUDA Graph 预捕获各自耗时   | 使用并验证缓存；减少非必要专用尺寸；离线预热              |
| 热启动仍慢        | 缓存 key 未命中、环境不一致                 | 比对版本/硬件/配置；检查缓存挂载与权限                    |
| 吞吐未提升        | 瓶颈可能在 attention/GEMM/通信，而非 launch | 用 profiler 定位；不要只增加`compile_sizes`               |
| P99 偶发尖刺      | guard 失败、尺寸未覆盖、运行时重新编译      | 收集 shape/graph-break 日志，收敛请求桶并预编译高频尺寸   |
| CUDA Graph 命中低 | capture size 与真实 token 分布不匹配        | 基于生产直方图调整`cudagraph_capture_sizes`               |
| 编译后结果异常    | graph break、动态图假设或自定义 pass 问题   | 先 Eager 对照，再最小化复现；逐步关闭编译/Pass/CUDA Graph |

## 9. 小结

`torch.compile` 从动态 Python 执行中提取 FX 图，再交给编译器优化。vLLM 在这条通用链路上补充了服务场景需要的能力：用动态图适应变长输入，为高频尺寸准备专用版本，在 attention 等动态边界处分段捕获 CUDA Graph，最后通过缓存和预热避免让用户请求承担编译成本。

## 参考资料

- [PyTorch：torch.compile 文档](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [vLLM：torch.compile integration（V1 设计）](https://docs.vllm.ai/en/latest/design/torch_compile/)
- [vLLM Blog：Introduction to torch.compile and How It Works with vLLM](https://vllm.ai/blog/2025-08-20-torch-compile)
