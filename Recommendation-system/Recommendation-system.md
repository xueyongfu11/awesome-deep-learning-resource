<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Blog](#blog)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Repo

- https://github.com/tensorflow/ranking
- 快速加入多模态信息进行推荐效果比较的框架 https://github.com/PreferredAI/cornac
- 基于评论文本的深度模型推荐系统库 https://github.com/ShomyLiu/Neu-Review-Rec
- https://github.com/jennyzhang0215/STAR-GCN
- https://github.com/yang-zhou-x/GCN_HAN_Rec
- https://github.com/breadbread1984/PinSage-tf2.0
- https://github.com/AlexYangLi/KGCN_Keras
- https://github.com/JockWang/Graph-in-Recommendation-Systems
- https://github.com/Jhy1993/Awesome-GNN-Recommendation


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
    <img src="../assets\PFRN.png" align="middle" />
    </details>
