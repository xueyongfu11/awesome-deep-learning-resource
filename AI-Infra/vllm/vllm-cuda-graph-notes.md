# vLLM 里的 CUDA Graph：动态推理如何塞进一张“静态图”

如果只看定义，CUDA Graph 并不难理解：先把一段反复执行的 GPU 工作连同依赖关系录下来，之后用一次 graph launch 整体重放。少掉一连串零碎的 kernel launch，CPU 的提交压力会小很多。

麻烦在于，大模型推理几乎处处都在变：batch 会变，序列长度会变，KV Cache 的映射会变，attention 走哪条路径也可能变；而 CUDA Graph 最喜欢的恰恰是稳定的执行结构和内存地址。一个要动态，一个要静态，这就是 vLLM 需要解决的核心矛盾。

vLLM 的办法并不是让 CUDA Graph 突然支持任意动态 shape，而是在两者之间加一层适配：静态 Buffer、packed token、attention metadata、分桶、padding，再配合多种 graph mode，把千变万化的请求归到有限几种可重放的物理规格里。

这篇文章想把这套机制从头到尾串起来。我会先讲 CUDA Graph 本身，再看 vLLM V1 如何处理动态 batch、变长 prefill、KV Cache 和多卡通信，最后结合公开测试聊聊它到底能快多少、又要付出什么代价。

先说结论：CUDA Graph 最擅长优化的是反复执行、单轮计算不重的 decode；它省掉的是 CPU 提交开销，不是模型 FLOPs。至于能不能用、收益有多大，最终都取决于请求形状、attention backend、捕获规格和回退比例。

> vLLM 迭代很快。文中涉及的类名、默认策略和支持边界，请以实际使用版本的源码与文档为准。

---

## 1. CUDA Graph 到底省掉了什么

先看普通的 eager forward。Python、PyTorch dispatcher 和 CUDA Driver 要把 kernel、通信和内存操作一个个提交出去：

```text
CPU: launch RMSNorm
CPU: launch QKV GEMM
CPU: launch Attention
CPU: launch All-Reduce
CPU: launch MLP
...
GPU: 等待并依次执行这些工作
```

虽然 kernel launch 通常是异步的，但每次调用仍需要 CPU 和驱动完成参数准备、解析、排队与调度。当每个 kernel 很短、kernel 数量很多时，CPU 可能来不及持续“喂饱”GPU，时间线上会出现空隙。

CUDA Graph 把一组 CUDA 操作及其依赖关系固化为一张可执行图：

```text
第一次：warmup → capture → instantiate

后续：CPU 调用 graph.replay()
                    ↓
      GPU 执行整张图中的 kernel、通信和依赖
```

所以 CUDA Graph 带来的收益，主要有三类：

- 减少 Python、框架 dispatcher、Driver 和逐 kernel launch 的 CPU 开销；
- 减少相邻 kernel 之间因提交不及时产生的 GPU 空隙；
- 让重复执行路径更稳定，降低部分时延抖动。

这里有个很容易混淆的点：CUDA Graph 不会减少模型的 FLOPs，也不会让 GEMM 或 attention 本身凭空变快。它优化的是“怎么把工作交给 GPU”，而不是“GPU 要做多少工作”。

这也解释了为什么它通常更适合小 batch、逐 token decode：每一轮计算不算重，却要重复很多轮，CPU launch 的占比自然更显眼。长 prompt 的 prefill 更容易被 GPU 计算本身卡住，相对收益通常就没那么大。

## 2. 一张图从捕获到重放，经历了什么

### 2.1 Warmup

正式 capture 前，一般会先拿相同 shape 跑几轮 eager forward。目的很朴素：先把各种只发生一次的初始化工作做完。

- CUDA context、kernel 和通信库完成初始化；
- 编译、autotune 或算法选择完成；
- 必要的 workspace 和缓存完成分配；
- 后续捕获不再混入“一次性”的初始化工作。

Warmup 更像工程上的保险措施，并不是 CUDA Graph API 规定“必须预热恰好几次”。模型、编译器、attention backend 和通信实现不同，合适的次数也会不同。

### 2.2 Capture

接下来，程序会在 capture stream 上真实执行一次目标路径。CUDA Runtime 一边执行，一边记下设备侧工作和依赖关系，最后得到一张 CUDA Graph。里面常见的节点有：

- kernel launch；
- 可捕获的异步内存操作；
- event 和 stream 依赖；
- 支持 graph capture 的 collective communication。

所以这张图本质上是一张依赖 DAG，不只是把几个 kernel 粗暴地装进同一个包。

### 2.3 Instantiate

刚捕获到的 graph 还不能直接高效运行，它要先实例化成 graph executable。这个阶段会完成校验，并准备好后续重放所需的执行结构。

PyTorch 的 `torch.cuda.CUDAGraph` 和 `torch.cuda.graph(...)` 已经封装了底层 API 的大部分细节，所以应用代码通常只显式看到 capture 和 `replay()`。

### 2.4 Replay

到了真正处理请求时，事情反而简单了：把新值写进捕获时用过的固定输入 Buffer，然后调用 `graph.replay()`。图仍从原来的地址读数据，也仍把结果写到原来的输出地址，只是 Buffer 里的内容换了。

```text
新请求数据
   ↓ copy_ / 原地更新
静态输入 Buffer（地址不变）
   ↓ graph.replay()
静态输出 Buffer（地址不变，内容更新）
```

注意，replay 不会重新跑 Python 控制流，也不会让 PyTorch 根据新输入再构建一次执行路径。它只是忠实地重放捕获时记录下来的 GPU 工作。

## 3. CUDA Graph 为什么这么“难伺候”

CUDA Graph 捕获的是一次具体的 GPU 执行实例，不是一个能随输入变化重新解释的高级程序。后面 vLLM 里的很多设计，其实都是在绕下面四个约束。

### 3.1 地址要稳定：换变量不等于换输入

图中 kernel 使用的是捕获时的设备地址。Python 变量后来指向另一个同 shape 张量，并不会让 graph 自动改读新地址。

```python
# 正确：修改旧 Buffer 的内容，地址不变
static_x.copy_(new_x)
graph.replay()

# 错误：x 指向了新张量，但 graph 仍然读取捕获时的旧地址
static_x = new_x
graph.replay()
```

所以，输入、输出以及要求地址稳定的 workspace/activation，都需要在捕获期间进入稳定的内存池，并在 replay 时继续复用。模型权重也一样，不能在重放前悄悄搬家。

### 3.2 物理 shape 和执行路径也不能乱变

shape 改变可能进一步改变：

- GEMM 的 M/N/K；
- kernel grid/block 配置；
- attention kernel 路径；
- workspace 大小；
- TP collective 的数据规模；
- 某些由 shape 或数据触发的控制分支。

因此，capture 与 replay 之间通常要求相关张量的 shape、stride、layout、dtype 和底层地址保持兼容。张量里的数值可以变，物理执行结构不能随意变。

支持动态业务输入通常有两条路：

1. 为若干常用 shape/bucket 分别捕获多张图；
2. 把较小输入 padding 到某个已捕获的较大规格，并用 mask/metadata 排除填充部分。

vLLM 两种办法都用了：常见规格单独捕获，不完全匹配的请求则尽量向上 padding 到附近的规格。

### 3.3 Capture 里放不下动态的 CPU 判断

`.item()`、`.cpu()`、`torch.cuda.synchronize()` 等操作会要求主机取得或等待设备结果，不适合出现在被捕获的设备执行路径中。依赖这些结果的 Python `if` 也不会在 replay 时重新判断。

更稳妥的做法是把边界划清楚：

- capture 前完成输入准备和主机侧逻辑；
- capture 内只保留可捕获、可重复的设备侧工作；
- 输出先留在 GPU；需要返回 CPU 时，在 replay 完成后再复制或同步；
- 多 stream 场景显式维护稳定、合法的依赖关系。

这里也别把“使用异步接口”机械理解成给任意 `copy_` 加个 `non_blocking=True` 就万事大吉。Host↔Device 拷贝能否异步、能否进入 capture，还取决于 pinned memory、stream、CUDA API 和具体的 capture 规则。vLLM 更常见的做法，是在 graph 外准备好本轮输入，再去更新预分配的设备 Buffer。

### 3.4 Capture 期间别让其他 CUDA 工作“串台”

捕获期间，如果其他线程或 stream 插进了不属于当前 capture 的 CUDA 工作，就可能导致 capture 失败、被 invalidated，甚至留下很难复现的错误。因此生产框架通常要统一管理 capture stream、同步关系和捕获时机。

## 4. 先用一个最小 PyTorch 例子跑通

下面这段代码不复杂，但把“预热—捕获—更新固定输入—重放”这个闭环完整走了一遍：

```python
import torch

assert torch.cuda.is_available()
device = "cuda"

# 固定地址和固定 shape 的静态输入
static_x = torch.randn(1024, 1024, device=device)
static_y = torch.randn(1024, 1024, device=device)

# 在侧流上预热；真实项目可能需要多轮
warmup_stream = torch.cuda.Stream()
warmup_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(warmup_stream):
    for _ in range(3):
        _ = torch.mm(static_x, static_y) + torch.sin(static_x)
torch.cuda.current_stream().wait_stream(warmup_stream)

# 捕获时这次计算也会真实执行
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_out = torch.mm(static_x, static_y) + torch.sin(static_x)

# 新输入只覆盖原 Buffer 的内容
new_x = torch.randn_like(static_x)
new_y = torch.randn_like(static_y)
static_x.copy_(new_x)
static_y.copy_(new_y)

graph.replay()

# static_out 的地址不变，内容已被本次 replay 更新
expected = torch.mm(new_x, new_y) + torch.sin(new_x)
torch.testing.assert_close(static_out, expected)
```

跑这个例子时，有几个细节值得留意：

- capture 中的 Python 语句只在 capture 当下执行，replay 的是其产生的 CUDA 工作；
- `static_out` 需要保持存活，否则应用层可能丢失对固定输出 Buffer 的引用；
- 计时时 CUDA 是异步的，应使用 CUDA Event，或在计时边界显式同步；
- 第一次 eager 和第一次 graph 执行都可能包含初始化开销，不能只各测一次就下结论。

别看这个 runner 很小，它已经有了 vLLM CUDA Graph 的雏形：固定 Buffer、内容更新和 replay。再往前一步，就是维护多种捕获规格并在运行时做选择。

当然，“再往前一步”说起来轻松。到了 vLLM，这套机制还得同时容纳动态调度、PagedAttention、KV Cache、LoRA、speculative decoding 和多卡通信，真正的复杂度从这里才刚开始。

---

## 5. 真正的麻烦，从大模型推理开始

先把大模型推理粗略分成两个阶段：

- **Prefill**：一次处理 prompt 中的一批 token，建立 KV Cache；
- **Decode**：后续每轮通常为每个请求处理一个新 token，并读取历史 KV Cache。

Decode 的 shape 比较规整，重复轮数又多，而且每一轮可能很短，几乎就是为 CUDA Graph 量身定做的。Prefill 则完全是另一种性格：prompt 长短差异很大，mixed batch 里还可能同时出现 prefill 和 decode，请求边界、attention 路径都更动态，想把整个 forward 塞进一张静态图自然更难。

不过在继续看 vLLM 的模式和源码流程之前，得先把几个很容易混在一起的长度概念说清楚。

## 6. 先别被这些长度和 batch 概念绕晕

### 6.1 query_len、context_len 与 seq_len

对某一个真实请求：

```text
query_len   = 本轮新处理的 token 数
context_len = forward 开始前已经计算、可由 attention 使用的 token 数
seq_len     = context_len + query_len
```

对标准 decoder-only 自回归 attention 来说，这是逻辑长度上的等式，不是近似关系。至于 CUDA Graph padding 后的物理 token 容量、补出来的请求行和实现里的长度上界，属于另一层概念，先不要混在一起。

| 场景 | forward 前 KV Cache | 本轮 query_len |
|---|---:|---:|
| 首次 prefill，prompt 长度 100 | 0 | 100 |
| chunked prefill，已经处理 60，本轮处理 40 | 60 | 40 |
| 普通 decode | 任意历史长度 | 1 |
| 静态 speculative decode | 任意历史长度 | 通常为 `1 + num_speculative_tokens` |

也就是说，两个 decode 请求的历史长度即使分别是 100 和 2000，本轮也完全可以都是 `query_len=1`。这里看的是“这轮新算多少”，不是“历史一共有多长”。

### 6.2 num_reqs、num_tokens 与 decode batch size

`num_reqs` 是本轮 batch 中的真实请求数。`num_tokens` 是所有请求本轮实际调度 token 数之和：

```text
num_tokens = sum(num_scheduled_tokens_per_request)
```

普通单 token decode 中：

```text
decode batch size = num_reqs
query_len         = 1
num_tokens        = num_reqs
```

例如 4 个请求的历史长度各不相同，但本轮都生成一个 token，则 `num_reqs=4`、`num_tokens=4`。

静态 speculative decode 中，若配置 3 个 draft token，target model 的 validation forward 通常还包含一个上一轮已采样并作为本轮输入的 token：

```text
num_reqs  = 4
query_len = 1 + 3 = 4
num_tokens = 4 × 4 = 16
```

dynamic/varlen speculative decoding 下，不同请求实际验证的 token 数可能不同，此时不能用一个统一的 `query_len × num_reqs` 表示真实 `num_tokens`，应对各请求的调度 token 数求和。

### 6.3 Uniform 与 non-uniform

这里的 uniform，问的是同一次 forward 里各请求本轮处理的 token 数是否一致，不是在问它们的历史上下文是否一样长。

```text
uniform:
  普通 decode                    [1, 1, 1, 1]
  静态 speculative（3 个 draft） [4, 4, 4, 4]

non-uniform:
  prefill                        [100, 60, 23]
  mixed prefill/decode           [128, 1, 1, 32]
```

所以普通 decode 即使各请求的 `context_len` 不同，依然可以是 uniform decode。动态 speculative batch 就没这么简单了，需要看实际的 `num_scheduled_tokens`、`uniform_token_count` 和 `max_query_len`，不能一概当成固定的 `[K, K, ...]`。

## 7. vLLM 为什么准备了五种 graph mode

现实里的 workload 很难靠一种策略吃遍，所以 vLLM V1 把完整图和分段图统一到了 `CUDAGraphMode` 中：

| 模式 | Decode | Prefill / mixed | 含义 |
|---|---|---|---|
| `NONE` | 非 graph | 非 graph | 禁用 CUDA Graph；是否还有其他 compile 优化取决于配置 |
| `PIECEWISE` | 分段图 | 分段图 | 只捕获编译后适合静态图的片段 |
| `FULL` | 完整图 | 尝试完整图 | 所有兼容 batch 都优先使用整个 forward 的 CUDA Graph |
| `FULL_DECODE_ONLY` | 完整图 | 非 graph | 只让 pure decode 使用完整图 |
| `FULL_AND_PIECEWISE` | 完整图 | 分段图 | decode 用 full，prefill/mixed 用 piecewise |

### 7.1 FULL

`FULL` 很直接：整个模型 forward 都放进同一张 CUDA Graph，attention 和支持 capture 的 collective 也留在图内。它的边界切换最少、潜在收益最高，代价是对 attention backend、batch descriptor、物理 shape、通信和执行路径的要求也最苛刻。

对 non-uniform prefill/mixed batch，只有 backend 支持相应的 full graph 路径时才能使用。捕获时还必须构造正确的 dummy attention metadata，让 backend 进入对应的 prefill/mixed kernel 路径，而不是误走 decode 路径。

### 7.2 PIECEWISE

`PIECEWISE` 走的是折中路线。编译系统先把完整 forward 切成若干片段，只给适合 capture 的部分建立 CUDA Graph；动态性太强或兼容性不好的操作继续留在图外。

```text
Graph 片段：RMSNorm + QKV projection
图外路径： Attention
Graph 片段：Output projection + MLP
```

这只是概念示意，真正的切分点由编译结果和不支持捕获的算子决定。PIECEWISE 的优势是兼容性更好，缺点也很明显：图内、图外来回切换，优化得没有完整图那么彻底。

### 7.3 默认怎么选，不能用时怎么退

在具备 piecewise compilation 条件的常见生成模型上，当前 V1 通常采用 `FULL_AND_PIECEWISE`：

```text
uniform decode  → FULL
prefill / mixed → PIECEWISE
不兼容 batch   → NONE / 普通执行
```

当然，这不是一条无条件规则。pooling model 通常默认 `PIECEWISE`；piecewise compilation 不可用时可能变成 `NONE`；如果 attention backend 不支持用户指定的模式，vLLM 也会转换或降级到最接近的受支持模式。

还要注意，`enforce_eager=True` 的影响比 `CUDAGraphMode.NONE` 更大：前者通常会把 compile 等更广泛的优化路径一起关闭。所以做基准测试时，不能把 `--enforce-eager` 与默认配置之间的全部差值都记到 CUDA Graph 头上。

## 8. vLLM 是怎么把这些东西串起来的

把细节先放到一边，核心对象之间的关系大致可以画成这样：

```text
Scheduler
   │  本轮每请求 token 数
   ▼
GPUModelRunner ──预处理/准备静态输入──► CudagraphDispatcher
   │                                      │
   │ set_forward_context                  │ runtime_mode
   │                                      │ batch_descriptor
   ▼                                      ▼
ForwardContext ───────────────────► CUDAGraphWrapper
                                           │
                             capture / replay / 普通执行
```

- **GPUModelRunner**：组织输入准备、dummy run、graph capture 和模型执行；
- **CudagraphDispatcher**：根据本轮 batch 特征和 backend 能力选择 `FULL`、`PIECEWISE` 或 `NONE`，并找到兼容的捕获规格；
- **BatchDescriptor**：描述一张图需要特殊化和匹配的关键 batch 特征；
- **ForwardContext**：把 runtime mode 和 descriptor 带入模型 forward，避免层层修改函数签名；
- **CUDAGraphWrapper**：包装 runnable，并管理具体 graph entry 的 capture、输入地址校验、输出缓存和 replay。

### 8.1 别把 BatchDescriptor 当成固定协议

分派时常见的维度包括：

```text
num_tokens
num_reqs
uniform / uniform_token_count
max_query_len
是否启用 LoRA
active LoRA 数量
```

具体字段会跟着 vLLM 版本演进，stable 设计文档里的原型与 main 分支代码也可能不同。与其背字段列表，不如抓住它的作用：descriptor 是一组特殊化条件，用来判断两轮执行能不能安全复用同一张图。

FULL 对完整物理执行结构要求更严格，通常需要更完整的 descriptor，例如同时约束 token 和 request 维度。PIECEWISE 的 descriptor 通常更宽松，常见情况下主要按 token capture size 匹配，不一定需要 request padding。

### 8.2 一个模型，背后往往有一组图

服务不会给一个模型只捕获一张万能图，而是针对多个常用规格维护一组候选 graph：

```text
FULL(tokens=1, reqs=1, uniform=True)
FULL(tokens=2, reqs=2, uniform=True)
FULL(tokens=4, reqs=4, uniform=True)
...
PIECEWISE(tokens=64)
PIECEWISE(tokens=128)
PIECEWISE(tokens=256)
...
```

实际 key 还可能包含 LoRA 和其他维度。同一个模型、同一个 token 数，也可能因为 runtime mode 或 batch 特征不同而对应不同的图。

## 9. 图是什么时候捕获的：通常在启动阶段

当前常见的 V1 路径，会在服务启动或初始化阶段通过 `capture_model` 主动捕获配置好的尺寸。这样虽然启动会慢一些，却不会把捕获延迟留给第一个真实请求。

对每个需要捕获的规格，流程可概括为：

```text
生成固定规格的 dummy input / attention metadata
                    ↓
              eager dummy warmup
                    ↓
设置 runtime mode 和 BatchDescriptor
                    ↓
     调用被 CUDAGraphWrapper 包装的 forward
                    ↓
             capture 并保存 graph entry
```

捕获通常按尺寸从大到小进行，方便较小规格复用已经建立的 graph memory pool。至于具体顺序、warmup 次数和捕获入口，仍然属于可能随版本变化的实现细节。

这里顺便澄清一个容易产生的误解。`CUDAGraphWrapper` 从抽象上看，确实像是“entry 已有就 replay，entry 为空就 capture”；但在常见服务流程里，首次 capture 主要由启动阶段的 dummy run 主动触发。真正的用户请求如果找不到兼容图，通常会直接走普通路径，而不是当场跑 dummy forward、临时捕获一张新图。

## 10. 请求真的来了以后，vLLM 做了什么

启动阶段把图准备好之后，一次真实推理大致会经过下面几步。

### 10.1 Scheduler 先决定这一轮算什么

Scheduler 决定每个请求本轮处理多少 token，Model Runner 得到：

```text
num_reqs
num_scheduled_tokens_per_request
num_tokens = sum(num_scheduled_tokens_per_request)
max_query_len
uniform 信息
```

### 10.2 Dispatcher 再决定走哪张图

Dispatcher 结合用户配置、backend 支持能力、batch 特征和已配置的 capture size，选择 `runtime_mode` 与兼容 descriptor。概念上的优先级通常是：

```text
FULL 精确/兼容匹配
        ↓ 未命中
PIECEWISE 较宽松匹配
        ↓ 未命中
NONE / 普通执行
```

如果真实 token 数没有恰好命中某张图，也不意味着马上回退 eager。只要它没超过最大 capture size，其他条件也兼容，Dispatcher 就可以向上找一个装得下它的规格，再用 padding 补齐。

```text
真实 num_tokens = 6
可用 capture size = [1, 2, 4, 8, 16]
选择 graph capacity = 8
padding tokens = 2
```

超过最大捕获尺寸，或 request 数、uniform、LoRA、backend 路径等条件不兼容时，才回退到普通执行。

### 10.3 把本轮数据写进静态 Buffer

Model Runner 把本轮真实内容写入预分配 GPU Buffer，例如：

```text
input_ids
positions
is_padding
query_start_loc
seq_lens
block_tables
slot_mapping
```

这些 Buffer 的值每轮都可以变化，但 graph 使用的地址和物理 shape 必须与捕获规格兼容。随后 runtime mode 和 descriptor 通过 `ForwardContext` 传到 wrapper。

### 10.4 能重放就重放，不能就正常执行

```text
兼容 entry 已存在 → graph.replay() → 读取固定输出 Buffer
没有兼容 entry     → 普通 forward
```

回退到普通 forward 并不代表出错，只是这一轮吃不到 CUDA Graph 的提交优化。它也不会因此自动触发一轮新的 dummy forward。

## 11. 变长 prefill 是怎么塞进静态图的

这里最关键的一句话是：CUDA Graph 要求固定的是 replay 看到的**物理 shape**，不是每个请求的**逻辑长度**。

假设三个请求本轮的 `query_len` 为：

```text
A = 100
B = 60
C = 23
```

vLLM 没必要构造一个 `[3, 100]` 的稠密矩形。它会把有效 token 紧凑地 pack 成一维 batch：

```text
num_tokens      = 100 + 60 + 23 = 183
input_ids.shape = [183]
```

请求边界和 KV Cache 访问由动态 metadata 表达：

```text
query_start_loc = [0, 100, 160, 183]
seq_lens        = 各请求本轮处理后的逻辑长度
block_tables    = 各请求的 KV Cache block 映射
slot_mapping    = 本轮 token 的 KV Cache 写入位置
```

若候选 graph 的 token capacity 为 184，则把 packed token 补到 184：

```text
真实 num_tokens = 183
物理 num_tokens = 184
padding token   = 1
```

这样一来，Graph 始终看到的都是 `[184]` 物理输入；attention kernel 再根据本轮 metadata 找到真实的请求边界、逻辑长度和 KV block。

动态性并没有消失，只是换了个地方：它从 tensor shape 转移到了固定 shape 的 metadata 内容里。这正是 vLLM 适配 CUDA Graph 最关键的一步。

上例只展示 token 维度 padding。FULL graph 的某些 attention backend 还要求 `num_reqs` 与捕获规格兼容，因此可能同时补齐请求维度，以及对应的 `query_start_loc`、`seq_lens`、`block_tables` 行。PIECEWISE 通常只对 token 数 padding，不一定要求 request padding。

因此，听到“prefill 使用 CUDA Graph”时，最好再追问一句它指的是哪一种：

- 默认常见路径：prefill 的稳定计算片段使用 PIECEWISE graph，attention 等动态部分可能在图外；
- 特定 backend/config：整个 non-uniform prefill forward 使用 FULL graph。

## 12. 多卡 Tensor Parallel 也能放进图里吗

可以，但并不是生成一张横跨所有 GPU 的“全局 CUDA Graph”。实际做法是，每个 TP rank 都在自己的 GPU 上捕获本 rank 的本地计算和可捕获通信：

```text
rank 0 graph：本地算子 → collective → 本地算子
rank 1 graph：本地算子 → collective → 本地算子
...
```

要把 all-reduce、all-gather、reduce-scatter 或 custom all-reduce 纳入完整图，通常要求：

- 通信库和具体 collective 实现支持 graph capture；
- TP world size 固定；
- 通信 tensor 的物理 shape 固定；
- collective 调用顺序固定；
- stream 和同步关系稳定；
- 所有 rank 使用兼容且一致的 batch 规格和执行分支。

这里要求的是各 rank 的执行路径和 collective 顺序保持一致，并不等于 replay 前一定存在一个显式的全局 barrier。真正危险的是不同 rank 选中了不同的 graph mode、padding size 或 collective 顺序，最后很可能不是变慢，而是直接通信 hang。

## 13. 图不是越多越好：启动时间和显存都要付账

### 13.1 捕获规格越多，启动为什么越慢

每一个 capture 规格都要准备 dummy metadata、做 warmup、执行 capture，最后再实例化。模型层数越多，图里的 kernel、通信节点和依赖越多，捕获单个规格花的时间也越长。

可以粗略理解为：

```text
捕获时间 ≈ graph 规格数 × 单规格 forward/capture 成本 × warmup 因子
```

Graph 规格不只由 token size 决定，还可能由以下维度共同扩张：

```text
FULL / PIECEWISE
uniform decode / non-uniform prefill-mixed
普通 decode / speculative decode
有 LoRA / 无 LoRA / active LoRA 数量
不同 token 和 request capacity
```

### 13.2 多出来的显存花在哪了

好消息是，模型权重通常不会为每张图复制一份。额外的 GPU 显存主要花在下面这些地方：

- 静态输入、输出 Buffer；
- 捕获期间分配且地址需要稳定的 activation；
- attention、GEMM 和通信 workspace；
- CUDA Graph memory pool。

Graph executable 和驱动管理结构也会消耗资源，但并非全部位于 GPU 显存。多张图还可能共享或复用部分 graph memory pool，因此显存增量不一定与 graph 数量严格线性。

### 13.3 分桶本质上是一道取舍题

```text
捕获规格少：启动更快、图资源较少，但平均 padding 和无效计算更多

捕获规格多：启动更慢、图资源更多，但能选择更接近真实规模的图，padding 更少
```

实际配置时，通常会优先覆盖高频的 decode 规模：它的 shape 更规整、调用轮数更多、CPU launch 占比更高，前期的捕获成本也更容易摊薄。capture size 应该跟着真实并发和 `num_tokens` 分布走，而不是盲目追求“图越多越好”。

## 14. 第一组测试：Eager 和默认优化栈差多少

先看一组 vLLM latency benchmark。测试环境是 NVIDIA A10 24GB、Qwen2.5-1.5B FP16 和 V1 架构，对比方式如下：

```text
Eager：     --enforce-eager
默认配置： 不加 --enforce-eager（包含当前默认 compile/CUDA Graph 路径）
Warmup：    3 iterations
Benchmark： 10 iterations
```

测试结果显示：batch size 1 时约有 `1.47×` 加速，batch size 64 时约为 `1.07×`；同时多个场景的 P99-P50 差值缩小。

| 场景 | Eager P99-P50 | 默认优化栈 P99-P50 | 改善 |
|---|---:|---:|---:|
| BS=1, I128/O128 | 2.0 ms | 1.4 ms | 30% |
| BS=16, I256/O128 | 9.6 ms | 2.6 ms | 73% |
| BS=64, I256/O128 | 30.8 ms | 6.0 ms | 80% |
| BS=32, I512/O256 | 53.6 ms | 14.3 ms | 73% |

这个结果基本符合直觉：batch 小时，launch 开销更容易被放大；batch 上去以后，GPU 计算本身的占比提高，CUDA Graph 的相对收益就会收窄。P99-P50 的差值变小，也说明执行稳定性可能有所改善。

不过，我更愿意把这组结果看成方向性数据，而不是 CUDA Graph 的严格归因，原因有两个：

1. `--enforce-eager` 与默认配置的差异不只有 CUDA Graph，还包括 compile 等优化，因此不能把全部提升都归因于 CUDA Graph；
2. 每个场景只有 10 次 benchmark，尾延迟样本很少，不能据此直接推断生产 SLA。

如果要做严格归因，应该固定 commit、compile 配置和 attention backend，只改变 `cudagraph_mode`，同时增加预热、请求数和重复轮次。

## 15. 再看一组开源数据：不同 graph mode 怎么选

vLLM PR #20059 的作者还公开过一组 serving benchmark，专门比较重构后的几种 CUDA Graph 模式。相比上一组数据，它更适合观察 graph mode 之间的相对差异；但它仍然只是特定 PR、模型、硬件和负载下的短测试，不能直接当成生产环境收益。

### 15.1 环境与命令

```text
GPU：A100 40GB
PyTorch：2.6
CUDA：12.4
模型：Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
数据集：ShareGPT
请求数：50
请求到达率：10 req/s
编译配置：-O3
```

```bash
python vllm/benchmarks/benchmark_serving.py \
  --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 50 \
  --request-rate 10
```

`main piecewise` 表示当时 main 分支上的 piecewise 实现，其他列是 PR 中显式选择的模式。

### 15.2 FlashAttention v2

| 指标 | main piecewise | PIECEWISE | FULL | FULL_DECODE_ONLY | FULL_AND_PIECEWISE |
|---|---:|---:|---:|---:|---:|
| Benchmark duration (s) | 9.04 | 8.90 | 8.75 | 8.36 | **8.32** |
| Request throughput (req/s) | 5.53 | 5.62 | 5.71 | 5.98 | **6.01** |
| Output throughput (tok/s) | 1125.11 | 1141.77 | 1161.84 | 1215.89 | **1221.38** |
| Mean TTFT (ms) | 60.81 | **57.71** | 60.34 | 57.74 | 62.53 |
| Median TPOT (ms) | 10.35 | 10.34 | 9.95 | **8.95** | 9.10 |
| Median ITL (ms) | 8.74 | 8.33 | 8.00 | 7.14 | **7.13** |

相对 `PIECEWISE`，`FULL_AND_PIECEWISE` 的输出吞吐提高约 7.0%，Median TPOT 降低约 12.0%，Median ITL 降低约 14.4%；Mean TTFT 则从 57.71 ms 上升到 62.53 ms。

### 15.3 Triton Attention

| 指标 | main piecewise | PIECEWISE | FULL | FULL_DECODE_ONLY | FULL_AND_PIECEWISE |
|---|---:|---:|---:|---:|---:|
| Benchmark duration (s) | 10.39 | 10.51 | 8.98 | **8.34** | 8.38 |
| Request throughput (req/s) | 4.81 | 4.76 | 5.57 | **6.00** | 5.97 |
| Output throughput (tok/s) | 978.85 | 966.95 | 1132.26 | **1219.67** | 1213.76 |
| Mean TTFT (ms) | 57.54 | 62.33 | 62.46 | 61.08 | **56.07** |
| Median TPOT (ms) | 12.64 | 12.96 | 10.03 | 8.98 | **8.93** |
| Median ITL (ms) | 11.00 | 10.96 | 8.00 | **6.84** | **6.84** |

相对 `PIECEWISE`，`FULL_AND_PIECEWISE` 的输出吞吐提高约 25.5%，Median TPOT 降低约 31.1%，Median ITL 降低约 37.6%。本组数据中 Triton 的改善最大。

### 15.4 FlashInfer

当时 FlashInfer 只支持 pure-decode full CUDA Graph，因此没有独立的通用 `FULL` 结果。

| 指标 | main piecewise | PIECEWISE | FULL_DECODE_ONLY | FULL_AND_PIECEWISE |
|---|---:|---:|---:|---:|
| Benchmark duration (s) | 9.00 | 9.10 | **8.45** | 8.48 |
| Request throughput (req/s) | 5.55 | 5.49 | **5.92** | 5.89 |
| Output throughput (tok/s) | 1126.68 | 1114.08 | **1201.74** | 1195.79 |
| Mean TTFT (ms) | **56.14** | 56.54 | 62.03 | 59.46 |
| Median TPOT (ms) | 10.21 | 10.15 | **9.58** | 9.60 |
| Median ITL (ms) | 8.49 | 8.36 | **7.45** | 7.59 |

相对 `PIECEWISE`，`FULL_DECODE_ONLY` 的输出吞吐提高约 7.9%，Median TPOT 降低约 5.6%，Median ITL 降低约 10.9%；Mean TTFT 从 56.54 ms 上升到 62.03 ms。

### 15.5 我会怎么解读这组结果

- 三个 backend 在启用 decode full graph 后，输出吞吐均提高，Median TPOT 和 ITL 均下降，符合 CUDA Graph 主要改善重复 decode 提交开销的预期；
- 收益有明显 backend 差异，不能只凭一种 attention 实现外推；
- TTFT 没有一致改善，因为它还受到 prefill、排队、调度和短测试随机性的影响；
- 该测试只有 50 个请求、持续约 8～10 秒，适合说明实现方向，不足以证明生产收益。

如果要做生产评测，我会固定 commit、backend、量化方式、并发、输入输出长度分布和随机种子，再分别报告 TTFT、TPOT、ITL、吞吐、GPU 利用率、CPU 占用、启动时间与显存增量。只看一个吞吐数字，通常解释不了真实问题。

## 16. 真正落地时，我会先看这几件事

### 16.1 哪些场景值得优先尝试

- 在线生成以 decode 为主，batch 较小或中等；
- CPU launch/调度已经成为瓶颈，GPU 时间线上存在 kernel 间隙；
- 请求形状集中在少数高频 bucket，padding 浪费可控；
- attention backend、LoRA、speculative decode 和 TP collective 都有明确的 graph 支持边界。

### 16.2 开了却没变快，先查什么

1. 实际是否命中了 FULL/PIECEWISE，还是频繁回退普通执行；
2. `num_tokens` 分布与 capture size 是否匹配，padding 是否过多；
3. 工作负载是否已经 GPU compute-bound；
4. attention backend 是否只支持部分模式；
5. LoRA、动态 speculative 或模型结构是否扩大了 descriptor 种类；
6. benchmark 是否把 compile、CUDA Graph、量化和 backend 变化混在了一起。

### 16.3 启动变慢、显存变多，再查什么

- capture size 是否过密、最大规格是否远超真实负载；
- 是否同时捕获了不需要的 FULL、PIECEWISE、LoRA 或 speculative 组合；
- graph memory pool、workspace 和静态 Buffer 的增量；
- 多 rank 是否每张卡都捕获了相同的大量规格；
- 是否应该用更少的 bucket 接受少量 padding，换取启动时间和内存。

## 17. 最后，再回到那个“动与静”的矛盾

回头看，CUDA Graph 做的事情其实很纯粹：用静态性换更低的 CPU 提交成本。设备侧的执行结构和关键地址固定下来，每轮只在 replay 前更新 Buffer 内容。

vLLM 也没有试图改变这条规则。它做的是在动态请求和静态图之间搭一层映射：

```text
动态请求
  ↓ Scheduler 产生每请求 token 数
packed token + 动态 KV/attention metadata
  ↓ Dispatcher 选择 mode 与 capture bucket
token/request padding + 静态 GPU Buffer
  ↓
FULL / PIECEWISE graph replay
  ↓ 不兼容
普通执行回退
```

把整篇文章压缩成一句话就是：**vLLM 通过 packed token、KV Cache 元数据、静态 Buffer、BatchDescriptor、capture-size 分桶和 padding，把动态的大模型推理 batch 映射成有限个固定的物理执行规格。**

Decode 形状规整、重复次数多，通常最适合完整图；prefill 和 mixed batch 更动态，默认更多依赖分段图；如果 backend 支持 non-uniform full graph，还可以进一步把完整的 prefill forward 纳入 CUDA Graph。

理解了这一层，很多看起来零散的实现细节——为什么要 padding、为什么要维护多张图、为什么会回退、为什么启动更慢、为什么显存会上涨——其实都能顺着同一条逻辑解释清楚。

## 参考资料

- [vLLM CUDA Graphs 设计文档](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [vLLM CudagraphDispatcher 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/cudagraph_dispatcher.py)
- [vLLM CUDA Graph 管理源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/cudagraph_utils.py)
- [vLLM GPU Model Runner 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py)
- [vLLM InputBatch 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/input_batch.py)
- [PR #20059 性能测试评论](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3160858458)
