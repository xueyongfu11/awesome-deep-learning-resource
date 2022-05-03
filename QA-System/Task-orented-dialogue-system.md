

## Projects
- Rasa  https://github.com/RasaHQ/rasa 
  - 官方文档 https://rasa.com/docs
  - https://github.com/RasaHQ/rasa-demo
  - https://github.com/RasaHQ/financial-demo
  - https://github.com/RasaHQ/helpdesk-assistant
  
- 查询话费消费情况的对话机器人 https://github.com/zqhZY/_rasa_chatbot
  - all easy 
  
- 基于rasa的中文对话系统 https://github.com/GaoQ1/rasa_chatbot_cn
  - https://github.com/GaoQ1/rasa_nlu_gq
  - https://github.com/GaoQ1/rasa-nlp-architect
  - all easy, base on rasa1.10
  
- rasa中文对话系统 easy https://github.com/crownpku/Rasa_NLU_Chi 


## Important

- https://github.com/thu-coai/Convlab-2
  - demo整合了任务型对话系统的pipline
  
- https://github.com/budzianowski/multiwoz
  - 任务型对话系统的相关论文
  
- 任务型对话系统相关内容 https://github.com/AtmaHou/Task-Oriented-Dialogue-Research-Progress-Survey

- https://github.com/yizhen20133868/Awesome-TOD-NLG-Survey
  - 任务型对话系统中的文本回复生成 
  
- https://github.com/thu-coai
  - 清华大学AI对话组 github
  
  
## end-to-end

- Paper：End-to-End Task-Completion Neural Dialogue Systems https://github.com/MiuLab/TC-Bot

- 未知paper https://github.com/shawnwun/NNDIAL

- Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System 2021 https://github.com/awslabs/pptod

- Paper MinTL: Minimalist Transfer Learning for Task-Oriented Dialogue Systems 2020 EMNLP https://github.com/zlinao/MinTL

- Global-to-local Memory Pointer Networks for Task-Oriented Dialogue 2019 ICLR https://github.com/jasonwu0731/GLMP

- Paper: Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures 2018 ACL https://github.com/WING-NUS/sequicity

- An End-to-End Trainable Neural Network Model with Belief Tracking for Task-Oriented Dialog; Bing Liu, et. al.；A Network-based End-to-End Trainable Task-oriented Dialogue System; Wen, Tsung-Hsien, et. al.

- https://github.com/edward-zhu/dialog

- ACCENTOR: Adding Chit-Chat to Enhance Task-Oriented Dialogues 2021 NAACL https://github.com/facebookresearch/accentor

- https://github.com/MiuLab/TC-Bot
  - 端到端的对话系统，同时实现了用户模拟器
- https://github.com/shawnwun/NNDIAL


## DST

- Task-Oriented Dialogue as Dataflow Synthesis 
  - 2020 TACL 
  - https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis
  
- SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking
  - 2019 ACL
  - 阅读笔记
    1.基于匹配的方式获取slot-value，具有可扩展性  
    2.参考阅读理解模型，将slot-type作为query，sys，usr的utterance作为text，通过计算注意，得到表示向量  
    3.2中向量和slot-value向量计算相似度  
    4.DST需要判断每个domain-slot的slot-value是空的还是具体的值。所以做法是每个domain-slot跟utterance进行融合计算，
    跟相应slot-value的所有可能取值进行相似度的计算
    
- Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems
  - 2019 ACL
  - 阅读笔记：
    1.使用生成模型，对每个domain-slot作为decoder的first-token，进行slot-value的预测  
    2.将decoder的first-token的隐向量跟encoder进行slot-gate的三分类，作为生成的slot-value的标识

- 百度unit对话产品 https://github.com/baidu/unit-dmkit
  - DM Kit作为UNIT的开源对话管理模块，可以无缝对接UNIT的理解能力，并赋予开发者多状态的复杂对话流程管理能力，还可以低成本对接外部知识库，迅速丰富话术信息量。
  - 已经不再更新
  
- https://github.com/google-research-datasets/simulated-dialogue

- https://github.com/budzianowski/multiwoz

## NLU
- https://github.com/sz128/slot_filling_and_intent_detection_of_SLU
- rasa对话策略学习模型 https://github.com/RasaHQ/TED-paper
- https://github.com/RasaHQ/conversational-ai-workshop-18
- https://github.com/RasaHQ/DIET-paper
- https://github.com/yizhen20133868/Awesome-SLU-Survey
- Joint Multiple Intent Detection and Slot Filling https://github.com/yizhen20133868/GL-GIN

- Paper:   SlotRefine: A Fast Non-Autoregressive Model for Joint Intent Detection and Slot Filling
  - year: 2020 EMNLP
  - 阅读笔记: 
    1.提出了一个非自回归但是拥有自回归性能的ID-SF联合模型  
    2.bert未使用绝对位置编码，而是使用相对位置编码  
    3.使用two-pass组件替代CRF，解决了非自回归的输出标签独立的问题。方法是将上一次迭代的标签作为下次模型迭代的输入，起始时全O，需要对tag进行embedding。目的是为了学习标签的边界或者依赖问题。
  - code: 

- 医疗领域任务型对话系统 2018 https://github.com/nnbay/MeicalChatbot-HRL
## Retrieval-Base
- 增量训练的任务型对话系统 Incremental Learning from Scratch for Task-Oriented Dialogue Systems 2019 ACL https://github.com/Leechikara/Incremental-Dialogue-System


## DPL

- Guided Dialog Policy Learning: Reward Estimation for Multi-Domain Task-Oriented Dialog
  - 2019 EMNLP-IJCNLP
  - 学习笔记：
    1.使用AIRL对抗逆强化学习方法估计reward  
    2.使用PPO算法进行更新策略函数和值函数
  - code： https://github.com/truthless11/GDPL

## NLG

- Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems
  - 2015, EMNLP
  - 阅读笔记：  
    1.LSTM单元中加入DM门控单元，对action进行编码  
    2.两个基于seq2seq模型生成，前向模型生成候选结果，后向模型对候选再排序（基于损失）
  - code：https://github.com/andy194673/nlg-sclstm-multiwoz

## Tools
- 生成对话数据集 https://github.com/rodrigopivi/Chatito    
  - 移动电信数据集 https://github.com/GaoQ1/chatito_gen_nlu_data

## dataset
- 中文任务型对话系统数据集 https://github.com/thu-coai/CrossWOZ
- 任务型对话数据集 https://github.com/sz128/NLU_datasets_with_task_oriented_dialogue
- 任务型对话数据集 https://github.com/alexa/dialoglue
- end to end dialogue dataset https://github.com/yizhen20133868/Retriever-Dialogue


