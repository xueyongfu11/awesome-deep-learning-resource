

# unified NER

- https://github.com/universal-ie/UIE

- Named Entity Recognition via Machine Reading Comprehension: A Multi-Task Learning Approach
  - 使用mrc的方式做ner任务，将每个实体类型构成的question和text拼接，然后使用同一个encoder进行编码
  - 将编码特征直接水平方向concatenate起来，通过一个self-attention层，对标签之间的依赖进行建模，输出的特征维度和输入的特征维度相同
  - 从输入中获取每个任务的输出，通过一个输出层，得到start index, end index, span index

- A Unified MRC Framework for Named Entity Recognition
  - 提出了一个统一的基于MRC的flatten实体和嵌套实体的抽取方法
  - 使用两个多个二分类的header预测相应的start和end index，然后判断任意两个start和end组合是否构成一个实体
  - 具体方法是把start index和end index对应的hidden state concat起来用一个FFC层做二分类，来判断是否可以组成一个实体