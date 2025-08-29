[TOC]



## 基本概念



Tensor Core硬件支持限定的数据类型和尺寸大小，具体参考[文档](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/#warp-matrix-functions)。由于仅能处理固定大小的矩阵块（Tile），必须先将输入矩阵（A、B）切割为与 WMMA Tile 尺寸匹配的子矩阵，才能通过wmma API 调度 Tensor Core 计算。

