# L4 sec2 6DoF Grasping：新手学习课程

---

## 0. 这节课在讲什么

本节课讲的是 **6DoF Grasping**，也就是机器人在三维空间中生成和评估抓取姿态。

如果前一节 2D Grasping 更像是在俯视图里找一个抓取矩形，那么 6DoF Grasping 要解决的是：

```text
夹爪在三维空间中应该放在哪里？
夹爪应该朝哪个方向？
夹爪张开多宽？
这个抓取会不会碰撞？
这个抓取成功概率高不高？
```

本节课的主线是：

```text
2D 平面抓取 -> 6DoF 抓取 -> 典型网络方法 -> 数据集和评价指标 -> 研究方向
```

本节重点：

1. 2D Planar Grasping 和 6DoF Grasping 的区别
2. 6DoF 抓取姿态 `(R, t, w)` 的含义
3. 6DoF 抓取的一般 pipeline
4. 6DoF GraspNet、GraspNet-1Billion Baseline、GSNet、AnyGrasp 等方法
5. ACRONYM、GraspNet-1Billion、Grasp Anything 6D 等数据集
6. AP、success rate、coverage/completion rate 等评价指标
7. 透明物体、杂乱场景、语言引导抓取等发展方向

---

## 1. 先理解什么是 6DoF

### 1.1 DoF 是什么

`DoF` 是 Degree of Freedom，自由度。

在三维空间里，一个刚体的姿态通常有 6 个自由度：

```text
3 个平移自由度：x, y, z
3 个旋转自由度：roll, pitch, yaw
```

所以 6DoF 可以理解成：

```text
物体或夹爪在三维空间中的位置 + 朝向
```

### 1.2 6DoF 抓取姿态怎么表示

6DoF 抓取姿态通常写成：

```text
Grasping pose: (R, t, w)
```

含义如下：

| 符号 | 含义 | 新手理解 |
|---|---|---|
| `R` | rotation，旋转 | 夹爪朝哪个方向 |
| `t` | translation，平移 | 夹爪中心放在哪里 |
| `w` | width，夹爪宽度 | 夹爪张开多大 |

如果用更直观的方式，可以理解成：

```text
x, y, z, roll, pitch, yaw, gripper_width
```

其中 `x, y, z` 决定夹爪位置，`roll, pitch, yaw` 决定夹爪方向，`gripper_width` 决定张开宽度。

---

## 2. 2D Planar Grasping 和 6DoF Grasping 的区别

### 2.1 2D Planar Grasping

2D 平面抓取通常输出一个抓取矩形：

```text
(x, y, w, h, theta)
```

含义：

| 参数 | 含义 |
|---|---|
| `x, y` | 图像中的抓取中心 |
| `w, h` | 抓取矩形宽高 |
| `theta` | 平面内旋转角度 |

它的特点：

```text
主要在像素坐标中学习
常用 2D conv 提取图像特征
输入常是 top-view RGB-D 图像
输出偏向图像平面上的抓取矩形
```

直观理解：

```text
从上往下看桌面，在图像上画一个夹爪应该抓的位置
```

### 2.2 6DoF Grasping

6DoF 抓取输出的是三维空间中的夹爪姿态：

```text
(R, t, w)
```

它的特点：

```text
在 3D 空间中生成抓取
需要考虑夹爪完整朝向
常用点云特征处理，也可以结合 RGB
比 2D 平面抓取更复杂
```

直观理解：

```text
不是只在图片上画矩形，而是决定真实夹爪在空间中怎么接近、怎么闭合、怎么避障
```

### 2.3 二者对比

| 对比项 | 2D Planar Grasping | 6DoF Grasping |
|---|---|---|
| 输出 | `(x, y, w, h, theta)` | `(R, t, w)` |
| 坐标空间 | 主要是像素坐标 | 三维空间坐标 |
| 常见输入 | 俯视 RGB-D 图像 | 点云、RGB-D、多视角图像 |
| 特征处理 | Conv2D | PointNet、PointNet++、NeRF、3DGS 等 |
| 难点 | 找平面抓取矩形 | 三维姿态、碰撞、可达性、遮挡 |

---

## 3. 抓取在机器人系统中的位置

一个完整的 6DoF 抓取链路可以整理为：

```text
位姿估计 -> 抓取位姿 -> 路径规划
```

展开后可以理解成：

```text
1. 位姿估计
   - 相机系位姿
   - 机器人系位姿

2. 抓取位姿
   - 抓取位姿计算
   - 抓取评估和优化

3. 路径规划
   - 笛卡尔空间点
   - IK 求解
   - 关节空间插值
```

新手要注意：抓取网络只解决“抓哪里、怎么抓”的感知决策问题，真正执行还需要运动规划。

```text
抓取检测给出候选抓取 -> 机器人规划接近路径 -> IK 求关节角 -> 执行夹爪动作
```

---

## 4. 2D 抓取 Pipeline

2D Planar Grasping Pipeline：

```text
Feature extraction
Middle level representation
Grasping detection and scoring
Refinement
```

### 4.1 Feature extraction

2D 抓取常用：

```text
Conv2D
```

输入是图像，输出是特征图。

### 4.2 Middle level representation

中间表示可以是：

```text
segmentation
object detection
heatmap
```

也就是先把图像理解成更结构化的表示。

### 4.3 Grasping detection and scoring

2D 抓取检测常被做成：

```text
矩形检测问题
回归问题
```

输出抓取矩形，并给每个候选抓取打分。

### 4.4 Refinement

Refinement 是对候选抓取进一步优化。

例如：

```text
调整中心点
调整角度
去掉碰撞或低分候选
保留最高质量抓取
```

---

## 5. 6DoF 抓取 Pipeline

6DoF Grasping Pipeline：

```text
Scene representation
Middle level representation
Grasping detection and scoring
Refinement
```

### 5.1 Scene representation

6DoF 抓取首先要表示三维场景。

常见方式：

```text
PointNet
NeRF
3DGS
RGB-D point cloud
multi-view reconstruction
```

新手可以先抓住一句话：

```text
2D 方法主要看图像；6DoF 方法通常要理解三维场景。
```

### 5.2 Middle level representation

中间表示可能包括：

```text
graspness
affordance
reconstruction
segmentation
```

含义：

| 词 | 新手理解 |
|---|---|
| graspness | 某个点或区域适不适合抓 |
| affordance | 某个区域能支持什么动作，例如可抓、可推、可拉 |
| reconstruction | 重建三维场景或物体形状 |
| segmentation | 分割物体或可抓区域 |

### 5.3 Grasping detection and scoring

6DoF 抓取要预测：

```text
R, t, w
```

也就是：

```text
旋转、平移、夹爪宽度
```

同时还要评估抓取质量：

```text
这个抓取能不能成功？
会不会碰撞？
力闭合是否稳定？
夹爪能不能到达？
```

### 5.4 Refinement

很多 6DoF 方法不是一步到位，而是：

```text
先生成候选抓取 -> 评估 -> 优化 -> 再评估 -> 输出最终抓取
```

需要强调：

```text
3D pose learning is more complicated.
```

原因是 6DoF 抓取同时涉及三维几何、传感器噪声、碰撞、夹爪结构和机器人运动可达性。

---

## 6. 6DoF GraspNet

### 6.1 方法概览

6DoF GraspNet 包含三部分：

```text
Grasp Sampling
Grasp Evaluation
Grasp Refinement
```

可以理解成：

```text
先生成候选抓取
再判断候选抓取好不好
最后把候选抓取往更好的方向调整
```

### 6.2 Grasp Sampling

Grasp sampler 负责从场景中生成可能的抓取姿态。

它使用：

```text
VAE grasp sampler
```

VAE 可以理解成一种生成模型。这里它的作用不是分类，而是生成候选抓取。

```text
场景输入 -> VAE sampler -> 多个候选 grasp
```

### 6.3 Grasp Evaluation

Grasp evaluator 负责给候选抓取打分。

```text
候选 grasp -> evaluator -> 成功概率或质量分数
```

它回答的问题是：

```text
这个抓取是否可能成功？
```

### 6.4 Grasp Refinement

Refinement 的目标是改进抓取。

核心意思是：

```text
Compute gradient with evaluator
P(s = 1 | g + delta_g) > P(s = 1 | g)
```

意思是：

```text
通过 evaluator 的梯度，调整原始抓取 g
希望调整后的抓取 g + delta_g 成功概率更高
```

### 6.5 Loss

VAE sample 的 loss 包括：

```text
reconstruction loss
KL divergence loss
```

简单理解：

| loss | 作用 |
|---|---|
| reconstruction loss | 生成结果要像真实抓取 |
| KL divergence loss | 约束隐空间分布，方便采样 |

### 6.6 数据和评价

6DoF GraspNet 使用模拟数据训练：

```text
ShapeNet 3D 模型
7M grasps
2M successful grasps
206 objects
```

数据划分：

```text
30% positive grasps
30% negative grasps
40% hard negative grasps
```

评价指标：

```text
success rate
coverage rate
```

实验中也使用真实机器人：

```text
Franka 7DoF
```

---

## 7. GraspNet-1Billion Baseline

### 7.1 方法概览

GraspNet-1Billion Baseline 是一个重要的基线方法。

模型结构：

```text
Backbone: PointNet++
Approach Net
Operation Net
Tolerance Net
```

数据集：

```text
GraspNet-1B
```

### 7.2 Backbone：PointNet++

PointNet++ 用于从点云中提取特征。

```text
点云输入 -> PointNet++ -> 点级或局部特征
```

这里的关键是：6DoF 抓取很多时候不是在 RGB 图像上直接做，而是在点云或三维表示上做。

### 7.3 Approach Net

Approach Net 预测：

```text
approach vector
score
```

也就是夹爪从哪个方向接近物体，以及这个接近方向质量如何。

补充要点：

```text
Cylinder group
```

可以理解为在候选抓取点附近按圆柱邻域聚合局部点云特征。

### 7.4 Operation Net

Operation Net 预测抓取操作参数。

常见包括：

```text
抓取角度
抓取深度
夹爪宽度
抓取分数
```

### 7.5 Tolerance Net

Tolerance Net 预测抓取质量或容忍度。

直观理解：

```text
这个抓取对姿态误差、摩擦条件、局部几何变化有多稳健？
```

### 7.6 GraspNet-1B 的评价指标

要点如下：

```text
AP
AP_mu for different friction
```

这里的 `mu` 表示摩擦系数。

在 GraspNet-1Billion 中，判断 true positive 时会考虑：

```text
friction
force-closure metric
```

新手可以先理解成：

```text
不是只看预测姿态像不像标注，还要看这个抓取在物理上是否稳定。
```

---

## 8. GSNet

### 8.1 方法概览

GSNet 来自 2021 年工作：

```text
Graspness Discovery in Clutters for Fast and Accurate Grasp Detection
```

模型结构：

```text
ResUNet14
Graspness model
Grasp operation model
```

### 8.2 Graspness

Graspness 是 GSNet 的核心概念之一。

它表示：

```text
某个点是否适合产生抓取
```

常见写法：

```text
point-wise graspable landscape
```

意思是为点云中的每个点估计“可抓程度”。

### 8.3 FPS subsample

要点如下：

```text
FPS subsample
```

FPS 是 Farthest Point Sampling，最远点采样。

它的作用是从点云中选出分布较均匀的一部分点。

```text
原始大量点 -> FPS -> 少量代表点
```

这样可以降低计算量，同时保持空间覆盖。

### 8.4 Grasp operation model

GSNet 的抓取操作模型预测：

```text
grasp cylinder
cylinder angle
depth
score
```

可以理解为：

```text
先用局部圆柱区域描述抓取邻域
再在这个邻域里预测角度、深度和质量分数
```

---

## 9. AnyGrasp

### 9.1 方法概览

AnyGrasp 来自 2022 年工作：

```text
AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains
```

模型结构：

```text
Geometric Processing: GSNet
Temporal Association
```

### 9.2 Temporal Association

Temporal Association 是时间关联。

如果机器人连续观察一个场景，不同帧之间的信息可以互相补充。

```text
第 1 帧看到一部分
第 2 帧看到另一部分
时间关联后得到更稳定的抓取感知
```

这对动态抓取、移动物体、相机视角变化很重要。

### 9.3 适用任务

AnyGrasp 的实验包括：

```text
GraspNet-1B+
dataset experiment
real-world experiment
bin-picking
dynamic grasping
```

新手可以这样理解：

```text
AnyGrasp 在 GSNet 的几何抓取基础上加入时间信息，让抓取在真实场景中更稳健。
```

---

## 10. FlexLoG

### 10.1 方法概览

FlexLoG 来自 2024 年工作：

```text
Rethinking 6-Dof Grasp Detection: A Flexible Framework for High-Quality Grasping
```

模型结构：

```text
Flexible Guidance Module
Local Grasp Module
```

### 10.2 Flexible Guidance Module

Flexible Guidance Module 可以接收不同中间表示作为引导。

例如：

```text
detection
segmentation
heatmap
affordance
```

这使得模型可以灵活利用上游视觉模块。

### 10.3 Local Grasp Module

Local Grasp Module 负责在局部区域生成或评估抓取。

抓取表示为：

```text
(theta, gamma, beta, w, delta_x, delta_y, delta_z)
```

可以理解成：

```text
局部坐标系下的旋转、宽度和位置偏移
```

---

## 11. ASGrasp

### 11.1 方法概览

ASGrasp 来自 2024 年工作：

```text
ASGrasp: Generalizable Transparent Object Reconstruction and 6-DoF Grasp Detection from RGB-D Active Stereo Camera
```

它重点关注：

```text
transparent object
specular object
```

也就是透明物体和高反光物体。

### 11.2 模型结构

模型结构：

```text
Scene Reconstruction Module
Grasp Detection Module
```

Scene Reconstruction Module：

```text
IR input
depth completion task
```

Grasp Detection Module：

```text
similar to GSNet
```

抓取表示：

```text
(q, t, R, w)
```

其中 `q` 可理解为抓取质量或分数，`t, R, w` 仍然表示位置、旋转和宽度。

### 11.3 为什么透明物体难

透明物体和反光物体会让深度相机更容易出错。

例如：

```text
深度缺失
深度噪声
物体边界不清楚
点云不完整
```

所以这类方法通常要先做重建或深度补全，再做抓取检测。

---

## 12. GraspNeRF

### 12.1 方法概览

GraspNeRF 使用多视角图像和 NeRF 来帮助 6DoF 抓取。

模型结构：

```text
Scene representation reconstruction
Volumetric grasp detection
```

也就是：

```text
先重建三维场景表示
再在体积空间里检测抓取
```

### 12.2 Loss

Loss 包括：

```text
grasp loss
color loss
geometric loss
```

含义：

| loss | 作用 |
|---|---|
| grasp loss | 学习抓取检测 |
| color loss | 约束渲染颜色和图像一致 |
| geometric loss | 约束三维几何正确 |

### 12.3 优点和代价

优点：

```text
对透明和高反光物体有效
能利用多视角信息重建场景
```

代价：

```text
抓取前需要多次拍摄图像
流程更复杂
实时性可能更难
```

---

## 13. RGBGrasp

### 13.1 方法概览

RGBGrasp 是 GraspNeRF 方向的扩展工作。

模型结构：

```text
Multi-view RGB perception
Online NeRF
Grasping Detection
```

也就是：

```text
机器人移动过程中采集多视角 RGB 图像
在线构建 NeRF
再进行抓取检测
```

### 13.2 适用场景

RGBGrasp 关注：

```text
transparent object
specular object
multi-view RGB grasping
```

当深度信息不可靠时，多视角 RGB 和神经重建可以提供额外几何线索。

---

## 14. R2SGrasp

### 14.1 方法概览

R2SGrasp 来自 2024 年工作：

```text
Real-to-Sim Grasp: Rethinking the Gap between Simulation and Real World in Grasp Detection
```

它关注的是：

```text
vision and point cloud sim-real gap
```

也就是仿真数据和真实数据之间的差距。

### 14.2 模型结构

模型结构：

```text
R2S Repairer
R2S Enhancer
```

可以这样理解：

| 模块 | 新手理解 |
|---|---|
| R2S Repairer | 修复真实输入，让它更接近仿真训练分布 |
| R2S Enhancer | 增强特征，让抓取检测器更好利用仿真经验 |

### 14.3 数据

相关数据：

```text
256 daily household objects
500 cluttered desktop scenes
64,000 RGB-D images
approximately 14.4 million grasp annotations
GraspNet-1B
```

### 14.4 为什么 sim-real gap 重要

很多抓取数据来自仿真，因为真实抓取标注很贵。

但仿真和真实世界不同：

```text
物体材质不同
深度噪声不同
光照不同
遮挡不同
点云分布不同
```

如果模型只在仿真里表现好，到真实机器人上可能失败。

R2SGrasp 的目标就是缩小这个差距。

---

## 15. 6DoF Grasping 数据集

三个重要数据集包括：

```text
ACRONYM
GraspNet-1Billion
Grasp Anything 6D
```

### 15.1 ACRONYM

ACRONYM 是模拟数据集。

关键信息：

```text
17.7M parallel-jaw grasps
8872 objects
262 categories
```

特点：

```text
规模大
来自仿真
适合训练平行夹爪抓取模型
```

### 15.2 GraspNet-1Billion

GraspNet-1Billion 是真实世界数据集。

关键信息：

```text
real world data
RGB-D from Kinect and RealSense
1.2B grasps
```

特点：

```text
真实采集
规模极大
有标准评价指标
常用于通用物体抓取 benchmark
```

### 15.3 Grasp Anything 6D

关键信息：

```text
simulated data generated by 2D image and depth estimation
1M point cloud scenes
comprehensive grasping prompts
200M grasps
```

特点：

```text
结合图像、深度估计和抓取 prompt
更贴近语言或开放场景抓取方向
```

---

## 16. 6DoF Grasping 评价指标

常用指标包括：

```text
AP
Success rate
Completion rate
```

### 16.1 AP

AP 是 Average Precision。

在抓取检测里，它通常评估：

```text
模型预测的一批抓取候选中，排序靠前的抓取有多少是真正可行的
```

GraspNet-1Billion 中还会考虑不同摩擦系数：

```text
AP_mu
```

摩擦越小，抓取越难稳定。

### 16.2 Success rate

Success rate 是真实机器人实验中常见指标。

```text
成功抓取次数 / 总尝试次数
```

例如尝试 100 次，成功 85 次：

```text
success rate = 85%
```

### 16.3 Coverage rate

Coverage rate 更关注模型能覆盖多少可行抓取。

```text
能否为不同物体、不同位置、不同姿态生成足够多的有效抓取
```

### 16.4 Completion rate

Completion rate 常用于杂乱环境。

例如 bin-picking：

```text
一箱物体里，机器人最终清空了多少物体？
```

它比单次 success rate 更关注连续任务完成情况。

---

## 17. 6DoF 抓取中的任务难点

最后需要关注几个任务点：

```text
cluttered scene or object-oriented
detection and segmentation
affordance and graspable area
collision
```

### 17.1 杂乱场景

杂乱场景里物体相互遮挡、堆叠。

难点：

```text
点云不完整
物体边界不清楚
候选抓取容易碰撞
一次抓取可能影响其他物体
```

### 17.2 Object-oriented grasping

Object-oriented grasping 是面向指定物体的抓取。

例如：

```text
只抓白色杯子
只抓指定零件
不要抓旁边的物体
```

这时模型不仅要找“可抓位置”，还要知道目标是谁。

### 17.3 Affordance

Affordance 可以理解成“可操作性”。

例如：

```text
杯子的把手适合抓
瓶身中部适合夹
刀刃区域不适合抓
```

在机器人中，affordance 不只是视觉概念，而是和动作结果有关。

### 17.4 Collision

6DoF 抓取必须考虑碰撞。

碰撞可能来自：

```text
夹爪碰到桌面
夹爪碰到其他物体
机器人手臂碰到环境
接近路径不可达
```

所以一个抓取姿态在视觉上看起来很好，也可能因为碰撞而不能执行。

---

## 18. 这些方法的共同范式

需要注意：

```text
Grasp learning is formulated as a supervised learning problem,
rather than a MDP problem in these works.
```

意思是，本节介绍的大多数方法把抓取看作监督学习问题。

### 18.1 监督学习范式

监督学习做法：

```text
输入：图像、点云或三维场景
标签：可行抓取姿态、抓取分数
目标：学习从输入到抓取输出的映射
```

它依赖：

```text
数据集
标注
评价指标
训练 loss
```

### 18.2 MDP / 强化学习范式

MDP 更像强化学习：

```text
状态 -> 动作 -> 奖励 -> 下一个状态
```

抓取可以被建模成：

```text
机器人观察场景 -> 选择动作 -> 执行抓取 -> 成功或失败奖励
```

但本节课主要讲的不是这种路线，而是基于数据集标注和监督学习的抓取检测。

---

## 19. 研究方向

相关方向包括：

```text
Transparent object
Bin-picking or cluttered environment
Language-guided grasping
```

### 19.1 透明物体抓取

透明物体难在传感器。

常见问题：

```text
深度相机看不准
点云缺失
边界不明显
反光导致噪声
```

相关方法：

```text
ASGrasp
GraspNeRF
RGBGrasp
```

### 19.2 Bin-picking 和杂乱环境

Bin-picking 是从箱子或料框中抓取物体。

难点：

```text
物体堆叠
遮挡严重
抓取空间狭窄
连续抓取需要稳定
```

相关方向：

```text
GSNet
AnyGrasp
GraspNet-1Billion
```

### 19.3 Language-guided grasping

语言引导抓取是让机器人根据文字指令抓取。

例如：

```text
Grasp the white cup
Pick the red block
拿起桌上的透明杯
```

这需要把语言、视觉和抓取结合起来。

```text
语言理解 -> 目标定位 -> 抓取检测 -> 执行
```

---

## 20. 方法总结表

本节可以总结为：

| 方法 | 年份 | 核心思路 | 优点 |
|---|---:|---|---|
| 6DoF GraspNet | 2019 | VAE grasp sampler + grasp evaluation + grasp refinement | 较早把完整 3D 深度学习抓取 pipeline 串起来，也包含数据生成 |
| GraspNet-1B Baseline | 2020 | Approach Net、Tolerance Net、Operation Net 分阶段回归抓取 | 问题定义清晰，多阶段分解 6DoF 抓取 |
| GSNet | 2021 | Graspness + cascade module + 多阶段抓取检测 | 通过 graspness 提高杂乱场景抓取效率 |
| AnyGrasp | 2022 | 基于 GSNet，并加入 temporal information | 更稳健，可用于动态物体和连续场景 |
| GraspNeRF | 2023 | 多图像 NeRF 重建后做抓取检测 | 适合透明和高反光物体 |
| FlexLoG | 2024 | 灵活利用 affordance、detection、segmentation、heatmap 等中间表示 | 方便接入不同中间视觉表示 |
| R2SGrasp | 2024 | 在抓取 pipeline 中加入 real-to-sim 方法 | 改善仿真训练到真实场景的迁移 |

