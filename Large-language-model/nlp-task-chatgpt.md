[TOC]




# information extraction

- How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?
  - ACL 
  - <details>
    <summary>阅读笔记: </summary>
    1. 基于chatgpt大模型的few-shot关系抽取  <br>
    2. 提出了两种策略：task-related instructions and schema-constrained data generation.  <br>
    3. task-related instructions：不同于text prompt，使用任务指令描述的方式来进行关系抽取任务  <br>
    4. schema-constrained data generation：使用实体类型限定的关系抽取任务
    </details>

- Small Models are Valuable Plug-ins for Large Language Models
  - [[code]](https://aka.ms/SuperICL)
  - <details>
    <summary>阅读笔记: </summary>
    1. 将大模型chatgpt和本地微调的小模型结合起来  <br>
    2. 基于in context learning学习，将小模型的预测的label和置信度加入到context中，基于该上下文使用chatgpt进行label的预测，并给出预测结果的推理过程  <br>
    </details>

- GPT-NER: Named Entity Recognition via Large Language Models
  - [code](https://github.com/ShuheWang1998/GPT-NER)
  - <details>
    <summary>阅读笔记: </summary>
    1. 为了减少生成大模型和抽取任务的gap，将原始文本的实体前后添加special token并作为label。然后使用任务指令+few-shot学习的方式用大模型进行预测  <br>
    2. few-shot样例使用了随机采样和基于kNN相似度检索的方式获取  <br>
    3. 为了防止大模型的过度自信，说了few-shot的自我验证方法，即通过指令描述的方式让大模型回答抽取到的实体是否是相应类别的实体  <br>
    </details>

- Zero-Shot Information Extraction via Chatting with ChatGPT
  - [[code]](https://github.com/cocacola-lab/ChatIE)
  - <details>
    <summary>阅读笔记: </summary>
    1. 基于chatgpt模型，把zero-shot信息抽取分成了两个阶段。第一个阶段：通过prompt的方式得到可能涉及的实体类型  <br>
    2. 第二个阶段：根据第一个阶段的实体类型和任务相关的schema来提示获取其他的相关信息  <br>
    </details>
