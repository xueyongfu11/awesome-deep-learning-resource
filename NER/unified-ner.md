

- https://github.com/universal-ie/UIE

- Named Entity Recognition via Machine Reading Comprehension: A Multi-Task Learning Approach
  1. 使用mrc的方式做ner任务，将每个实体类型构成的question和text拼接，然后使用同一个encoder进行编码
  2. 将编码特征直接水平方向concatenate起来，通过一个self-attention层，对标签之间的依赖进行建模，输出的特征维度和输入的特征维度相同
  3. 从输入中获取每个任务的输出，通过一个输出层，得到start index, end index, span index