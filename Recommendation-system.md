<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Paper](#paper)
  - [subdomain-1](#subdomain-1)
- [Dataset](#dataset)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Repo

# Paper

## 航班行程排序

### 2020
- Personalized Flight Itinerary Ranking at Fliggy
  - CIKM  
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出了一个个性化的航班行程预定的rank模型，基于listwise来捕捉特征的上下文信息。对于数值型特征要进行归一化，类别型特征使用embedding空间查找来表示  <br>
    2. 用户偏好表征：特征有历史行为信息、实时用户点击信息和属于相同用户组的组属性信息。使用LEF网络，具体是基于relative position的multi-head transformer，然后基于每层transformer的输出使用dense neural network融合每层的特征。最后将三种类型信息的输出concat。  <br>
    3. 航班行程列表的表示仍然使用LEF网络，将用户表征跟每个行程的表征做注意力计算，损失函数使用交叉熵损失  <br>
    <img src="./assets\PFRN.png" align="middle" />
    </details>


# Dataset
