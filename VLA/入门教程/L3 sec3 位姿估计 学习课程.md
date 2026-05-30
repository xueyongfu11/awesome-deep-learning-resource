# L3 sec3 位姿估计：新手学习课程

---

## 0. 这节课在讲什么

本节讲的是 **6D 物体位姿估计**。

机器人想抓取、放置、插入或操作一个物体，首先要知道这个物体在空间中的位置和朝向。只知道“图像里有一个杯子”还不够，机器人还要知道：

```text
杯子在哪里？
杯子离相机多远？
杯子的朝向是什么？
夹爪应该从哪个方向接近？
```

这就是位姿估计要解决的问题。

6D 位姿通常写成：

```text
R, t
```

其中：

| 符号 | 含义 | 新手理解 |
|---|---|---|
| R | rotation，旋转 | 物体朝哪个方向 |
| t | translation，平移 | 物体在空间中哪里 |

如果用更直观的方式写，也可以理解成：

```text
x, y, z, roll, pitch, yaw
```

其中 `x, y, z` 是位置，`roll, pitch, yaw` 是三维朝向。

---

## 1. 位姿估计和机器人抓取的关系

机器人抓取通常需要一条完整链路：

```text
图像 / 点云输入 -> 物体识别 -> 物体位姿估计 -> 抓取姿态生成 -> 预抓取位姿 -> 路径规划 -> 轨迹执行
```

位姿估计处在感知和操作之间。

如果位姿估计不准，后面的抓取姿态、路径规划和执行都会受到影响。例如：

- 物体位置估计偏了，夹爪可能抓空
- 物体朝向估计错了，夹爪可能夹到不稳定的位置
- 深度估计不准，机器人可能撞到物体或桌面

所以位姿估计不是单独的视觉问题，它直接影响机器人能不能完成真实操作。

---

## 2. 先分清三个问题层级

位姿估计可以按目标物体的已知程度分成三类。

### 2.1 Instance-level：已知实例

Instance-level 指目标物体是已知的具体实例。

例子：

```text
已知要识别的是某一个特定型号的杯子
系统有这个杯子的 3D 模型或训练数据
需要估计这个具体杯子的 6D 位姿
```

这类问题相对明确，因为系统知道目标物体长什么样。

常见方法包括：

- correspondence-based method
- template-based method
- voting-based method
- regression-based method

### 2.2 Category-level：已知类别

Category-level 指目标不是某一个固定物体，而是一类物体。

例子：

```text
系统要识别“杯子”这个类别
测试时出现的杯子可能不是训练集中那个杯子
但它仍然属于杯子这个类别
```

这类问题更难，因为同一类物体内部差异很大。杯子可能有高有矮，有没有把手，形状也不一样。

### 2.3 Unseen Object：未见过物体

Unseen object 指测试时出现的是训练中没见过的新物体。

这类问题最难，因为系统不能依赖“我以前见过这个东西”。它通常需要额外输入，例如：

- CAD 模型
- 参考图像
- 少量物体图片
- foundation model 的泛化能力

新手可以先按难度记：

```text
Instance-level < Category-level < Unseen object
```

---

## 3. 输入数据通常有哪些

位姿估计常见输入有三类。

### 3.1 RGB 图像

RGB 图像就是普通彩色图像。

优点：

- 获取方便
- 传感器便宜
- 纹理、颜色、边缘信息丰富

难点：

- 没有直接深度
- 遮挡时容易出错
- 光照变化影响大
- 仅凭单张图像恢复 3D 位姿比较难

### 3.2 RGB-D 图像

RGB-D 比 RGB 多了 depth，也就是深度信息。

可以理解为：

```text
RGB 告诉你“看起来是什么”
Depth 告诉你“离相机有多远”
```

RGB-D 对机器人很有用，因为机器人最终要在三维空间中行动。

### 3.3 3D 模型

有些方法会使用物体的 CAD 模型或 3D mesh。

如果系统知道物体的几何形状，就可以把模型和图像/点云进行匹配，从而估计位姿。

---

## 4. 数据集：BOP 和 YCB

### 4.1 BOP Benchmark

BOP 是 6D object pose estimation 中常见的 benchmark。

它关注的是：同一个算法在多个数据集、多个物体和多种场景下表现如何。

BOP 数据通常包含：

| 数据 | 含义 |
|---|---|
| RGB | 彩色图像 |
| depth | 深度图 |
| model | 物体 3D 模型 |
| camera info | 相机内参 |
| mask | 物体分割区域 |

BOP 的意义在于提供统一格式和统一评估方式，方便比较不同算法。

### 4.2 YCB 数据集

YCB 是机器人操作里非常常见的物体数据集。

它包含 21 个常见生活物品，并且有视频数据。常见物体包括杯子、盒子、瓶子、工具等。

YCB 的价值在于：这些物体更接近真实机器人抓取和操作任务，而不是只在简单实验环境里出现。

扩展数据集还包括：

- DexYCB
- YCB Affordance
- YCB-Sight

---

## 5. 位姿估计怎么评价

位姿估计不能只说“看起来差不多”。它需要指标来量化估计结果和真实结果之间的差距。

常见指标包括：

- VSD：Visible Surface Discrepancy
- MSSD：Maximum Symmetry-Aware Surface Distance
- MSPD：Maximum Symmetry-Aware Projection Distance
- ADD：average point distance
- ADD-S：average closest point distance
- 3D IoU：三维交并比

新手先重点理解 ADD 和 ADD-S。

### 5.1 ADD

ADD 的直觉是：

```text
把物体模型上的点分别用真实位姿和预测位姿变换到空间中，
然后计算这些点的平均距离。
```

如果预测位姿很准，那么同一个模型点在真实位姿和预测位姿下的位置应该很接近。

可以简化理解为：

```text
ADD 越小，预测位姿越接近真实位姿。
```

### 5.2 ADD-S

ADD-S 主要处理对称物体。

有些物体旋转后看起来差不多，例如圆柱、碗、某些瓶子。如果直接用 ADD，可能会把一个视觉上等价的姿态判成错误。

ADD-S 会计算 closest point distance，也就是找最近点来比较，更适合对称物体。

### 5.3 BOP 的评价思路

BOP 会对 VSD、MSSD、MSPD 等指标设置多个 threshold，然后计算 recall，再对多个数据集和指标取平均。

新手可以先理解成：

```text
BOP 不只看一个误差值，而是看算法在不同误差阈值下有多少结果算成功。
```

---

## 6. 传统方法：PnP

PnP 全称是 Perspective-n-Point。

它解决的问题是：

```text
已知一组 3D 点
已知这些 3D 点在图像中的 2D 像素位置
求相机和物体之间的位姿关系
```

更直白地说：

```text
我知道物体上某些点在真实 3D 模型里的位置，
也知道它们在图片里投影到了哪里，
那么我就可以反推出物体相对相机的姿态。
```

PnP 的关键前提是 correspondence，也就是 2D 图像点和 3D 模型点之间的对应关系。

### 6.1 PnP + RANSAC

真实图像中，对应点可能有错误。RANSAC 可以帮助剔除错误匹配。

基本思想是：

```text
随机取一小部分匹配点
估计一个位姿
检查这个位姿能解释多少匹配点
重复多次
选支持点最多的结果
```

OpenCV 中常用的接口是 `solvePnPRansac()`。

---

## 7. 传统方法：ICP

ICP 全称是 Iterative Closest Point。

它常用于 3D 点云配准。

ICP 要解决的问题是：

```text
给定两个点云
找到一个旋转和平移
让这两个点云尽量对齐
```

常见流程是：

```text
1. 对每个源点，在目标点云中找最近点
2. 根据这些最近点对应关系估计 R 和 t
3. 用新的 R 和 t 变换源点云
4. 重复迭代，直到误差不再明显下降
```

ICP 的优点是直观，适合几何信息明确的点云。缺点是比较依赖初始位姿，如果一开始偏得太远，可能收敛到错误结果。

---

## 8. Instance-level 方法

Instance-level 位姿估计面向已知具体物体。

常见方法可以分成三类：

| 方法 | 核心思路 |
|---|---|
| correspondence-based | 找 2D-3D 或 3D-3D 对应关系 |
| template-based | 把当前观测和已有模板匹配 |
| voting / regression-based | 让网络直接预测中心、旋转、平移等信息 |

下面几个经典方法可以帮助建立整体认识。

---

## 9. PoseCNN

PoseCNN 是一个使用 RGB 图像进行 6D 位姿估计的方法。

它的核心流程包括：

- feature extraction：提取图像特征
- segmentation：分割目标物体
- center point prediction：预测物体中心
- rotation and translation regression：回归旋转和平移

PoseCNN 的任务可以理解为三个子问题：

```text
先知道哪里是目标物体
再知道物体中心大概在哪里
最后估计物体的旋转和平移
```

其中中心点预测会用到 Hough voting 的思想。每个像素不直接给最终中心，而是预测一个指向中心的方向；多个像素投票后，中心位置就更容易被确定。

损失函数方面，PoseCNN 会考虑 segmentation loss、center point prediction loss，以及 pose loss。对于对称物体，还会使用 shape match loss，避免把等价姿态误判成严重错误。

---

## 10. DenseFusion

DenseFusion 使用 RGB 和 depth，也就是同时利用图像和点云。

它的核心想法是：

```text
图像特征提供纹理和外观信息
点云特征提供几何和深度信息
把两者融合起来估计位姿
```

模型会先在分割出的物体区域上，通过 CNN 提取图像特征，通过 PointNet 提取点云特征。然后在像素坐标下把图像特征和点云特征拼接，进一步提取局部特征和全局特征。

DenseFusion 的预测头通常输出：

- translation：平移
- rotation：旋转
- confidence：置信度

它还会使用 pose refinement，也就是再用一个 residual estimator 对初始位姿进行细化。

新手可以这样记：

```text
PoseCNN 更偏 RGB 图像；
DenseFusion 把 RGB 和点云信息融合起来。
```

---

## 11. YOLO6D

YOLO6D 的特点是 simple and fast。

它借鉴检测网络的思路，用 CNN 提取特征，然后预测物体 3D bounding box 的 8 个角点和 1 个中心点在图像中的位置，同时预测类别。

预测出这些 2D 控制点之后，就可以使用 PnP 来恢复 3D 位姿。

它的直觉很清楚：

```text
先在图像中找到物体关键控制点
再用 2D-3D 对应关系求位姿
```

这种方法把深度学习检测和传统 PnP 结合在一起。

---

## 12. Category-level 方法

Category-level 位姿估计不是针对某一个固定物体，而是针对一类物体。

例如自动驾驶里识别 car，不只是识别某一辆具体车，而是要处理 truck、SUV、sedan 等不同外形。

在机器人抓取里也类似。系统可能要估计“杯子”这个类别的位姿，但具体杯子形状可能从没见过。

这类方法的关键难点是：

```text
同一类别内部形状差异很大，但算法仍然要估计一个统一可用的位姿。
```

---

## 13. NOCS

NOCS 全称是 Normalized Object Coordinate Space。

它的核心思想是：为一个类别定义一个归一化的物体坐标空间。

对于同一类物体，例如杯子，即使具体形状不同，也尝试把它们映射到一个统一的标准空间里。这样网络就不只是预测“这个像素属于物体”，还会预测这个像素在归一化物体坐标系中的位置。

NOCS 通常建立在 Mask R-CNN 之上：

- Mask R-CNN 负责检测和分割
- 额外的 head 负责预测 NOCS map
- 训练时会包含 Mask R-CNN loss 和 NOCS 相关 loss

NOCS 的数据生成也很重要，常用 mixed reality 的方式：

```text
真实背景 + 仿真物体
不同光照下渲染
生成带标注的训练数据
```

新手可以这样理解：

```text
NOCS 是在同一类别内部建立一个“标准坐标系”，
让不同形状的同类物体可以被统一描述。
```

---

## 14. Unseen Object Pose Estimation

Unseen object pose estimation 面向训练时没见过的新物体。

这类方法通常会给额外输入，例如：

- CAD model
- reference image
- 少量目标物体图片

它和 category-level 不一样。Category-level 通常需要在某个类别上训练，例如杯子类别；而 unseen object 更强调新物体本身没有参与训练。

传统方法可以用 template-based 或 feature-based 的思路。如果有 CAD 模型，可以渲染不同角度的模板，再和当前图像匹配。如果有参考图像，可以找特征匹配，再估计位姿。

---

## 15. Foundation Model 方向

近年的方向开始使用更强的 foundation model 来处理新物体位姿估计。

相关方法包括：

- FoundationPose
- SAM-6D
- FreeZe

这类方法通常希望同时支持：

- model-based：给定带纹理的 3D CAD 模型
- model-free：只给一组参考图像
- pose estimation：单帧位姿估计
- pose tracking：连续帧位姿跟踪

以 FoundationPose 为例，它可以处理 model-based 和 model-free 两种输入。它的数据生成流程也比较复杂，会结合物体描述、纹理生成、物理引擎渲染、神经渲染、姿态生成和姿态筛选。

新手不需要一开始就掌握这些细节。先记住：

```text
传统方法依赖几何、模板、匹配；
深度学习方法通过网络学习特征和位姿；
foundation model 方向希望增强对新物体的泛化能力。
```

---

## 16. 位姿估计到机器人抓取

位姿估计最终要服务机器人操作。

一个常见抓取流程是：

```text
6D object pose estimation
-> grasping pose generation
-> pre-grasping pose
-> path planning
-> trajectory generation
```

也就是说，先估计物体 6D 位姿，再根据物体位姿生成抓取姿态。机器人通常不会直接从当前位置冲到抓取点，而是先到达 pre-grasping pose，也就是预抓取位姿，再沿较短、较安全的路径接近目标。

最后还要做路径规划和轨迹生成，让机械臂能平滑、安全地执行动作。

所以位姿估计不是终点。它只是机器人操作系统中的一个关键输入。

---

## 17. 方法怎么选

不同场景适合不同方法。

| 场景 | 更适合的方法 |
|---|---|
| 有 CAD 模型，物体固定 | PnP、ICP、模板匹配、instance-level 方法 |
| 有 RGB-D 数据 | DenseFusion、点云融合方法 |
| 只用 RGB 图像 | PoseCNN、YOLO6D、单目 6D 方法 |
| 同类物体很多，具体实例会变 | NOCS、category-level 方法 |
| 新物体，没有训练过 | FoundationPose、SAM-6D、template/reference-based 方法 |

工程上还要考虑：

- 是否有 3D 模型
- 是否有深度相机
- 是否需要实时
- 物体是否对称
- 场景遮挡是否严重
- 估计结果是否直接用于机器人抓取

如果只是做实验，可以先从公开数据集和现成模型入手。如果要做机器人抓取系统，除了算法精度，还要关注稳定性、速度、标定误差和失败恢复。

---

