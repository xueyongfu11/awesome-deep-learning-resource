<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [survey](#survey)
- [KG Completion](#kg-completion)
- [KG Alignment](#kg-alignment)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# survey

- Paper: Multi-Modal Knowledge Graph Construction and Application: A Survey
  - year: 2022
  - 阅读笔记:  
    1.多模态知识图谱综述型文章
    2.介绍多模态知识图谱的构建，两种方法：from symbols to images, from images to symbols
    3.多模态知识图谱应用，挑战


# KG Completion

- Paper: 
  - year: 
  - 阅读笔记:  
     
  - code: https://github.com/zjunlp/MKGformer/
  

# KG Alignment

- Paper: MMEA: Entity Alignment for Multi-modal Knowledge Graph
  - year: 
  - 阅读笔记:  
    1.多模态图谱对齐  
    2.构建正负三元组样本，比如把一个三元组中的实体用对齐的其他图谱的实体替换，从而构建出一个正样本。使用margin loss使得正负样本更远  
    3.实体-image pair distance loss  
    4.attribute value用RBF建模，然后计算attribute， attribute value和实体embedding的distance loss  
    5.计算1，2，3中实体embedding和common 实体的distance loss  
    6.计算来自不通图谱的对齐的实体embedding distance loss 
  - code: 
