# L7&8 sec4 LLM&VLM for Planning 学习课程

---

## 0. 这一节课在讲什么？

本节课的核心问题是：

```text
如何把 LLM / VLM 用作机器人任务规划器，让机器人能理解人类语言和视觉场景，并生成可执行的高层计划？
```

这里的重点不是直接训练一个端到端的机器人动作模型，而是把大模型放在**高层规划层**：

```text
人类语言 / 图像 / 视频
        ↓
VLM / LLM 进行理解、分解、推理、代码生成
        ↓
子任务 / 代码 / CoT / PDDL / STL 等中间表示
        ↓
底层策略、控制器、规则系统或传统规划器执行
```

所以，本节课关注的是一个层级式机器人系统：

```text
High-level planner: VLM / LLM
Low-level executor: visuomotor policy / language-conditioned policy / rule-based policy / controller
```

学习时要抓住一句话：

```text
LLM/VLM for planning 主要解决“要做什么、按什么顺序做、用什么接口表达给底层系统”的问题；
底层策略主要解决“具体怎么动、怎么抓、怎么控制”的问题。
```

---

## 1. 总体框架：VLM & LLM for Robotic Planning

### 1.1 层级式规划

机器人任务通常可以分成两层：

```text
高层任务规划：理解目标，拆分步骤
底层动作执行：完成每个具体动作
```

例如用户说：

```text
I am hungry.
```

LLM / VLM 可能会生成：

```text
1. Find apple
2. Pick up apple
3. Bring it back
```

之后，每个子任务可以交给底层模块完成：

```text
Find apple      -> 视觉检测 / 目标搜索
Pick up apple   -> 抓取策略 / 机械臂控制
Bring it back   -> 导航 / 搬运 / 放置
```

这就是本节课的主线：

```text
VLM/LLM 负责规划和接口生成
底层 policy / controller 负责真实执行
```

### 1.2 VLM 和 LLM 的输入输出

输入可以包括：

- 自然语言指令
- 图像或视频
- 当前场景描述
- 机器人可用技能
- 已知物体、状态、位置
- 用户反馈或对话历史

输出可以是多种形式：

- 子任务列表
- Python 代码
- Chain-of-Thought 推理过程
- PDDL 规划问题
- STL 逻辑约束
- 给 VLA 或底层策略的二次输入

用统一形式表示：

```text
Language + Vision + Prompt
        ↓
VLM / LLM
        ↓
Subtasks / Code / CoT / PDDL / STL / Other representation
```

### 1.3 为什么需要中间表示？

机器人不能只靠一句自然语言直接执行复杂任务。中间表示的作用是把模糊的人类意图变成更清晰的执行接口。

例如：

```text
自然语言：我饿了，帮我拿点吃的。
```

可以转换为：

```text
子任务：
1. 找到一个饮料
2. 拿起饮料
3. 送到用户身边
4. 找到一个零食
5. 拿起零食
6. 送到用户身边
7. 完成
```

也可以转换为代码：

```python
drink = find_object("drink")
pick_up(drink)
bring_to_user(drink)

snack = find_object("snack")
pick_up(snack)
bring_to_user(snack)
```

或者转换为传统规划语言、逻辑约束、底层技能调用。

---

## 2. 本节课的主要方向

VLM/LLM for planning 可以分成几类：

1. VLM/LLM for robot code generation
2. VLM/LLM for robot sub-task generation
3. PDDL generation
4. Chain-of-Thought
5. LLM planning
6. Sub-task and affordance evaluation
7. PaLM-E 等多模态高层规划模型

这些方向可以放在同一个框架下理解：

```text
大模型不是直接替代所有机器人模块，
而是把复杂任务转成底层系统更容易执行的形式。
```

需要特别注意：

```text
多数 LLM/VLM planning 论文不会深入讨论底层控制细节。
它们通常假设已经存在可调用的低层技能、控制器或 policy。
```

例如代码生成方法会默认有这些函数：

```python
find_obj("apple")
pick_up("apple")
move_to("table")
place("apple", "table")
```

真正困难的抓取、轨迹规划、避障、控制稳定性，可能被封装在这些函数内部。

---

## 3. Microsoft ChatGPT for Robotics 系列

### 3.1 核心思想

Microsoft ChatGPT for Robotics 系列探索了如何用 ChatGPT 控制机器人。

关键做法是：

```text
用精心设计的 prompt 告诉 LLM：
1. 当前任务是什么
2. 场景中有什么
3. 机器人有什么能力
4. 可调用哪些函数
5. 输出应该是什么格式
```

然后让 LLM 生成：

- Python 代码
- step-by-step plan
- 可执行函数调用序列

底层控制仍然由脚本动作或已有模块完成。

### 3.2 Prompt 的内容

一个机器人 prompt 通常包含：

```text
Scenario description:
  当前场景、物体、约束

Robot description:
  机器人类型、自由度、传感器、限制

Available functions:
  机器人可以调用的 API

Task instruction:
  用户希望机器人完成什么

Output format:
  例如必须输出 Python code
```

这种方法的本质是：

```text
把 LLM 限制在一个可控的动作接口空间里。
```

如果不给定可用函数，LLM 可能会生成无法执行的自然语言计划；给定 API 后，它更容易输出可运行的程序结构。

### 3.3 适用任务

相关任务包括：

- drone / AirSim 控制
- mobile navigation
- manipulation
- long-step robot control
- multimodal task planning

它们共同点是：

```text
任务具有长时序结构，但每个底层动作可以由已有技能或函数执行。
```

### 3.4 迭代式交互

LLM 的计划不一定一次就正确。系统可以通过对话进行迭代：

```text
用户提出任务
LLM 生成计划
用户或系统检查计划
LLM 修正计划
机器人执行
```

这种方式适合开放式任务，因为人类可以持续补充约束。

---

## 4. Microsoft LLM for Robot：长时序任务规划

### 4.1 Long-horizon task

长时序任务的难点是：

```text
一个目标往往需要很多步骤，
并且步骤之间有依赖关系。
```

例如：

```text
把桌面整理好
```

可能包含：

```text
1. 找到杯子
2. 把杯子放到杯垫上
3. 找到书
4. 把书放到书架
5. 清理桌面上的垃圾
6. 检查桌面状态
```

LLM 适合做这种高层拆分，因为它具备较强的常识和语言理解能力。

### 4.2 结构化输入与输出

Structured input and output 是一个关键点。

输入可以组织为：

```text
Environment:
  assets and states
  objects and states

Instruction:
  user command
```

输出可以组织为：

```text
Task cohesion:
  sequence
  instruction
  object

Summary:
  environment before
  environment after
```

这种结构化的好处是：

```text
减少 LLM 输出的随意性，让计划更容易被程序解析和检查。
```

### 4.3 Task cohesion

Task cohesion 可以理解为“任务步骤之间的连贯性”。

一个好的任务计划不仅要列出动作，还要保证：

- 顺序合理
- 对象一致
- 前后状态衔接
- 每一步可执行
- 最终状态满足用户目标

例如：

```text
错误计划：
1. Pour water into cup
2. Find cup

合理计划：
1. Find cup
2. Pick up cup
3. Move cup under dispenser
4. Pour water into cup
```

---

## 5. GPT-4V for Robotics：从人类演示中做多模态任务规划

### 5.1 Two-step approach

GPT-4V 系列方法利用视频输入，从人类演示中提取任务结构。

两步流程是：

```text
1. Ground subtasks from human demonstration
2. Hierarchical task planning
```

也就是先看懂人类在视频里做了什么，再把它转成可复用的机器人任务计划。

### 5.2 从视频中 ground 子任务

视频演示包含丰富信息：

- 人手在哪里
- 操作了什么物体
- 物体状态如何变化
- 动作发生的顺序
- 哪些对象是关键对象

VLM 可以根据这些视觉线索生成子任务，例如：

```text
1. Grasp the bottle
2. Move the bottle to the cup
3. Tilt the bottle
4. Pour liquid into the cup
5. Place the bottle back
```

这里的重点是 grounding：

```text
不是凭空生成计划，而是把计划绑定到视频中的具体物体、手部动作和状态变化。
```

### 5.3 低层 planner 与 affordance analyzer

低层 planner 可以来自 affordance analyzer，主要关注：

- human hands
- objects
- grasp
- movehand
- rotate

这些可以作为底层策略或低层 action primitive。

可以理解为：

```text
VLM 负责看懂演示并抽象成任务步骤；
affordance analyzer 判断哪些动作在当前场景中可能执行；
底层策略执行具体动作。
```

### 5.4 One-shot learning

这种方法希望通过一次人类演示就让机器人学会任务流程。

这里的 one-shot 主要指高层任务结构学习：

```text
从一次演示中获得任务分解、对象关系和步骤顺序。
```

但底层动作执行通常仍然依赖已有技能，而不是从一次视频里完整学会低层控制。

---

## 6. SayCan：LLM + Affordance 的层级规划

### 6.1 SayCan 解决什么问题？

LLM 很会生成合理的语言计划，但它不一定知道机器人当前是否能执行某个动作。

例如用户说：

```text
I just worked out, can you bring me a drink and a snack?
```

LLM 可能生成：

```text
1. Find a drink
2. Pick up the drink
3. Bring it to me
4. Find a snack
5. Pick up the snack
6. Bring it to me
7. Done
```

这些步骤语言上合理，但机器人还需要判断：

- 当前场景有没有 drink？
- 机器人能否到达？
- 物体是否可抓？
- 当前技能成功概率是多少？

SayCan 的核心是：

```text
LLM 判断“什么步骤在语义上合理”
Affordance / value function 判断“什么步骤在物理上可执行”
两者结合选择下一步动作
```

### 6.2 高层：LLM 生成子任务

SayCan 使用大语言模型做高层任务规划。

相关模型包括：

- PaLM 540B
- PaLM 8B / 64B
- FLAN 137B

LLM 的作用是给每个候选技能一个语言合理性评分。

例如：

```text
用户目标：bring me a drink
候选技能：
  find a drink        -> 语言上很合理
  open the drawer     -> 可能合理
  pick up a sponge    -> 语言上不太合理
```

### 6.3 低层：policy 与 value function

低层模块包括：

- language-conditioned policy
- value function
- imitation learning policy，例如 BC-Z
- reinforcement learning policy，例如 MT-OPT

Affordance 可以理解为：

```text
成功执行某个子任务的概率。
```

value function 的训练信号可以是：

```text
success episode -> 1.0
failure episode -> 0
```

### 6.4 SayCan 的选择机制

直观地说，SayCan 会综合两个分数：

```text
LLM score: 这个动作对完成用户目标是否合理？
Value score: 机器人当前是否能成功执行这个动作？
```

最终选择：

```text
既符合语言目标，又具有高执行成功率的子任务。
```

这解决了纯 LLM planning 的一个关键问题：

```text
LLM 可能会说出合理但机器人做不到的步骤。
```

### 6.5 添加新技能

Adding skills 的思路是：

```text
1. 用 BC / RL 训练新技能
2. 训练对应 value function
3. 把技能描述加入 prompt
4. LLM 就可以在规划中调用这个新技能
```

这说明 SayCan 是模块化的：

```text
技能库越丰富，LLM 可选择的动作空间越大。
```

---

## 7. Code as Policies：把 LLM 当作代码生成器

### 7.1 核心思想

Code as Policies 的目标是让 LLM 生成机器人控制代码。

它不是只输出自然语言步骤，而是输出类似：

```python
objects = detect_objects()
target = find_nearest(objects, "red block")
pick(target)
place(target, "green bowl")
```

这种方法把 LLM 用作：

```text
Language Model Programming, LMP
```

也就是用语言模型写程序，让程序调用底层机器人 API。

### 7.2 为什么代码有优势？

相比自然语言子任务，代码有几个优势：

- 更明确
- 可执行
- 可以包含条件判断
- 可以包含循环
- 可以调用工具函数
- 可以组合已有 API

例如自然语言：

```text
把所有蓝色方块放进盒子里。
```

代码可以表达为：

```python
for obj in get_objects():
    if obj.color == "blue" and obj.shape == "block":
        pick(obj)
        place(obj, box)
```

这类逻辑用自然语言也能描述，但代码更容易被系统执行。

### 7.3 Prompt 设计

Code as Policies 的 prompt 通常包含：

- 可调用函数列表
- 函数说明
- 示例代码
- 当前任务
- 约束条件
- 输出格式

要点如下：

```text
Prompt = hints + examples
```

示例很重要，因为 LLM 可以模仿示例的代码风格、函数调用方式和任务分解方式。

### 7.4 Hierarchical code generation

复杂任务可以用层级代码生成：

```text
高层函数：完成一个任务
中层函数：组织多个技能
低层函数：调用具体机器人 API
```

例如：

```python
def make_breakfast():
    prepare_plate()
    get_food("bread")
    get_drink("milk")
    bring_to_user()
```

然后每个函数再细化。

这和层级式规划是一致的，只是中间表示从“子任务列表”变成了“程序”。

---

## 8. Chain-of-Thought：用推理过程改善机器人规划

### 8.1 CoT 是什么？

Chain-of-Thought 指让模型在输出答案前生成中间推理步骤。

在数学题中，CoT 可以改善复杂推理；在机器人任务中，CoT 可以帮助模型分析：

- 任务目标
- 当前状态
- 物体关系
- 前置条件
- 动作顺序
- 潜在失败原因

例如：

```text
目标：把杯子装满水。

推理：
1. 需要先找到杯子。
2. 需要把杯子移动到水源附近。
3. 需要打开水源或倾倒水瓶。
4. 需要避免溢出。
5. 最后把杯子放到目标位置。
```

### 8.2 为什么机器人任务需要 CoT？

要点如下：

```text
Next token prediction for LLM is difficult.
```

对于机器人任务，即使指令看起来简单，下一步动作也可能依赖很多隐含条件。

例如：

```text
把苹果拿给我。
```

可能需要判断：

- 苹果在哪里？
- 是否被遮挡？
- 是否可抓？
- 机器人手上是否已经拿着东西？
- 用户在哪里？
- 路径是否通畅？

CoT 可以把这些隐含判断显式化，减少直接输出错误动作的概率。

### 8.3 Embodied CoT

Robotic Control via Embodied Chain-of-Thought Reasoning 可以与 OpenVLA 结合。

Embodied CoT 可以理解为：

```text
不仅用语言推理任务，还把视觉场景、身体动作和运动约束纳入推理。
```

例如普通 CoT 可能只写：

```text
I should pick up the cup and place it on the table.
```

Embodied CoT 更关注：

```text
杯子在右侧，机械臂需要先靠近杯子；
杯柄朝左，抓取点应避开杯口；
桌面上有障碍物，需要绕开。
```

这类推理更贴近机器人执行。

---

## 9. LLM as Translator and Checker：自然语言到形式化表示

### 9.1 自然语言到 STL

要点如下：

```text
LLM for NL to STL
```

STL 是 Signal Temporal Logic，用来描述连续信号随时间变化的逻辑约束。

可以把它理解成：

```text
用形式化语言描述“在什么时候、什么状态必须成立”。
```

例如自然语言：

```text
机器人必须在 10 秒内到达目标，并且始终避开障碍物。
```

可以转换成类似时间逻辑约束：

```text
eventually within 10s: at_goal
always: not collision
```

LLM 在这里的作用是 translator：

```text
把人类自然语言需求翻译成可被传统控制或验证系统使用的逻辑规范。
```

### 9.2 LLM as checker

LLM 也可以作为 checker，帮助检查：

- 计划是否满足指令
- 形式化表达是否漏掉约束
- 任务步骤是否前后矛盾
- 对象状态是否一致

但要注意：

```text
LLM checker 不是严格形式验证器。
高可靠系统仍需要传统验证、仿真或执行反馈。
```

---

## 10. LLM for PDDL：连接传统任务规划

### 10.1 PDDL 是什么？

PDDL 是 Planning Domain Description Language，是传统任务规划中常用的描述语言。

它通常描述：

- objects
- predicates
- actions
- preconditions
- effects
- goals

例如一个动作可以表示为：

```text
Action: pick_up(obj)
Precondition:
  robot_hand_empty
  obj_on_table
Effect:
  robot_holding(obj)
  not obj_on_table
```

### 10.2 LLM + PDDL 的意义

传统 PDDL planner 擅长搜索和保证逻辑一致性，但人工写 PDDL 很繁琐。

LLM 可以帮助：

```text
自然语言任务 -> PDDL domain/problem -> 传统 planner 求解 -> 可执行计划
```

也就是让 LLM 做语言接口，把规划求解交给传统方法。

相关方向包括：

- LLM+P
- Generalized Planning in PDDL Domains with Pretrained LLMs

### 10.3 为什么不直接让 LLM 规划？

因为传统 planner 有一些 LLM 不具备的优势：

- 明确的状态空间
- 可检查的前置条件和效果
- 可搜索最优或可行计划
- 更容易验证计划合法性

因此合理组合是：

```text
LLM 负责把人类语言翻译成规划问题；
PDDL planner 负责严谨地求解计划。
```

---

## 11. PaLM-E：多模态语言模型做高层规划

### 11.1 模型特点

PaLM-E 的核心是：

```text
562B multimodal language model
```

它可以处理多种输入模态，并输出 action description。

这里的 action description 不是低层控制量，而是高层动作描述，例如：

```text
move to the drawer
open the drawer
pick up the object
place it on the table
```

### 11.2 PaLM-E 的系统假设

PaLM-E 通常假设：

```text
存在一个低层 policy 或 planner，
可以把这些高层决策转成低层动作。
```

所以它仍然符合本节课的层级框架：

```text
Multimodal model -> high-level decision
Low-level policy/planner -> executable action
```

### 11.3 泛化能力

PaLM-E 的意义在于：

- 可以处理多模态输入
- 可以做高层任务泛化
- 可以跨任务理解语言和视觉信息

但对于机器人执行来说，仍然需要可靠的底层技能支持。

---

## 12. 底层策略在整个系统中的地位

本节虽然重点讲 LLM/VLM，但 Summary 也强调了低层策略的重要性。

低层策略可能包括：

- language-conditioned policy
- selected policy
- visuomotor policy
- rule-based policy
- hybrid policy
- controller

它们负责真正执行动作。

### 12.1 有限技能集的高质量完成很重要

一个关键观点是：

```text
High-quality completion of a limited set of low-level tasks
is probably more important than generalizing to unlimited-set tasks.
```

也就是说：

```text
与其让底层策略什么都想做，
不如先让它把一组关键技能稳定、可靠、高成功率地完成。
```

原因很直接：

```text
高层 LLM 可以拆分任务，
但如果底层 pick、place、navigate 都不稳定，
整个系统仍然无法可靠完成任务。
```

### 12.2 LLM/VLM planning 与 VLA 的区别

本节课讲的 LLM/VLM planning 和下一类 VLA 模型有明显区别：

| 方向 | 主要作用 | 输出 | 低层动作 |
|---|---|---|---|
| LLM/VLM for planning | 高层规划、代码生成、任务分解 | 子任务、代码、逻辑表示 | 通常依赖已有 policy/controller |
| VLA | 视觉-语言-动作统一建模 | 直接或半直接输出动作 | 模型内部学习 action policy |

可以用一句话区分：

```text
LLM/VLM planning 更像“大脑里的任务规划器”；
VLA 更像“从看见、听懂到行动的统一策略模型”。
```

---

## 13. 方法对比表

| 方法 | 输入 | 输出 | 核心贡献 | 主要限制 |
|---|---|---|---|---|
| ChatGPT for Robotics | 语言、场景、可用函数 | Python 代码或步骤 | 用 prompt 让 LLM 调用机器人 API | 强依赖 prompt 和底层函数 |
| Microsoft long-horizon planning | 环境状态、对象状态、指令 | 结构化任务序列 | 适合长时序任务分解 | 低层执行仍需额外模块 |
| GPT-4V for Robotics | 视频/图像、人类演示、指令 | grounded subtasks | 从演示中提取任务结构 | 视频理解和动作 grounding 可能不稳定 |
| SayCan | 用户指令、技能描述、value function | 下一步可执行子任务 | 结合语言合理性与可执行性 | 技能库和 value function 很关键 |
| Code as Policies | 指令、API、示例 | 可执行代码 | 用代码表达复杂逻辑和组合 | 代码安全性与 API 假设是问题 |
| CoT / Embodied CoT | 指令、场景、视觉信息 | 推理过程和计划 | 改善复杂任务推理 | 推理过程不一定保证正确 |
| NL to STL / PDDL | 自然语言需求 | 形式化逻辑或规划问题 | 连接 LLM 与传统规划/验证 | 翻译错误会影响后续规划 |
| PaLM-E | 多模态输入 | 高层动作描述 | 多模态高层泛化 | 假设有低层策略执行 |

---

## 14. 学习时容易混淆的点

### 14.1 LLM 会规划，不等于机器人会执行

LLM 输出的计划可能语义正确，但物理不可执行。

例如：

```text
Pick up the refrigerator.
```

语言上是一个动作，但真实机器人可能做不到。

所以需要：

- affordance 判断
- value function
- 物理仿真
- 视觉检测
- 低层策略反馈
- 失败恢复机制

### 14.2 Prompt engineering 不是训练

本节很多方法强调：

```text
Prompt rather than training or finetuning.
```

也就是说，很多系统没有重新训练 LLM，而是通过 prompt 描述机器人接口和任务规则。

这带来两个特点：

- 部署快，改 prompt 就能适配新任务
- 稳定性和可靠性受 prompt 质量影响大

### 14.3 子任务不是动作

子任务：

```text
Pick up the apple
```

动作：

```text
机械臂末端位姿变化、夹爪开合、关节速度、力控信号
```

二者中间还需要底层策略。

### 14.4 高层泛化与低层泛化不同

LLM/VLM 可能能理解新任务：

```text
帮我找一个能解渴的东西。
```

但底层策略是否能抓住新杯子、新瓶子、新零食，是另一个问题。

---

## 15. 一条完整机器人系统链路示例

假设用户说：

```text
我刚运动完，帮我拿一瓶水和一个零食。
```

系统可以这样工作：

### Step 1：LLM 理解目标

```text
用户需要 drink 和 snack。
```

### Step 2：生成候选子任务

```text
1. Find a drink
2. Pick up the drink
3. Bring it to the user
4. Find a snack
5. Pick up the snack
6. Bring it to the user
```

### Step 3：结合 affordance 选择下一步

```text
LLM 认为 find a drink 合理
视觉系统检测到桌上有 water bottle
value function 判断机器人可到达且可抓
```

### Step 4：调用底层策略

```text
language-conditioned policy 执行 "pick up the water bottle"
```

### Step 5：检查结果并继续

```text
如果成功，进入 bring it to user
如果失败，重新规划或请求人类帮助
```

这条链路体现了本节课的核心：

```text
高层语言/视觉推理 + 低层可执行技能 + 执行反馈
```

