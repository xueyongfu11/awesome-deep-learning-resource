<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Paper](#paper)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

[[TOC]]


- https://github.com/yechens/NL2SQL
- https://github.com/wangpinggl/TREQS


# Paper



- SDCUP: Schema Dependency-Enhanced Curriculum Pre-Training for Table Semantic Parsing
  - <details>
    <summary>阅读笔记: </summary>
    1. 通过探索question和schema的关系，提出了table pre-train的两个预训练方法：schema依赖预测任务和实体扰动恢复任务  <br>
    2. schema依赖预测：通过sql以及规则，使用biaffine attention来建立question和schema的关系和label，以交叉熵作为损失  <br>
    3. 实体扰动恢复：通过对question中的存在依赖关系的实体交换，然后用模型预测真实的实体来恢复  <br>
    4. MLM：对question的token进行mask，对schema中的column用相应的value来替代  
    <img src="../assets\SDCUP1.png" align="middle" />
    <img src="../assets\SDCUP2.png" align="middle" />
    </details>

