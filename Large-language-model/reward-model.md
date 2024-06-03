# RLHF中reward model的模型选型？奖励值如何计算？Loss如何计算？

reward 模型的训练涉及到几个问题：

- 奖励值如何输出？（注意不是reward model loss）
- reward model的常见loss有哪些？
- 针对不同的奖励值的输出方法、不同的loss计算，相应的reward model训练数据样式是怎么样的？
- 可以使用的模型结构有哪些？

## decoder-only模型

如常见的GPT-1/2/3，Llama，Qwen，gemma等模型可以作为reward model的基座模型。

使用decoder-only模型有多种输出奖励值的方式：

1. 获取输入序列的最后一个token的hidden state，然后接一个线性层映射到2分类（accept和reject），然后使用交叉熵损失计算loss。  

   一个基于该方式的实现：https://github.com/CarperAI/autocrit/tree/main

2. 获取输入序列的最后一个token的hidden state，然后接一个线性层映射到一个标量值，然后使用MES等计算loss。要求训练数据的标签是一个标量值（打分值）。  

   存在的问题：不同标注人员的打分标准很难保持一致，如标注员A：0.8 VS 0.7，标注员B：0.6 VS 0.5

3. token-level pairwise reward loss：有两种实现方式

   1. OpenAI的实现方式：

      ```python
      class LogSigLoss(nn.Module):
          """
          Pairwise Loss for Reward Model
          Details: https://arxiv.org/abs/2203.02155
          """
      
          def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
              return -torch.nn.functional.logsigmoid(chosen_reward - reject_reward).mean()
      ```

   2. Anthropic的实现方式

      ```python
      class LogExpLoss(nn.Module):
          """
          Pairwise Loss for Reward Model
          Details: https://arxiv.org/abs/2204.05862
          """
      
          def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
              loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
              return loss
      ```

   其中reject_reward和chosen_reward是一维列表，response中的每个token输出一个reward值

4. token-level pairwise reward loss的具体实现方法

   **本质是accept response和reject response的token-wise的reward的差值的sigmoid（尽可能的去掉pad token loss，并且去掉prompt token loss）**

   具体的实现方式参考：[url](https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py)

   ```python
   def compute_loss(
       self, model: "PreTrainedModel", inputs: Dict[str, torch.Tensor], return_outputs: bool = False
   ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
       r"""
       Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.
       Subclass and override to inject custom behavior.
       Note that the first element will be removed from the output tuple.
       See: https://github.com/huggingface/transformers/blob/v4.39.1/src/transformers/trainer.py#L3777
       """
       # Compute rewards
       _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)
       unwrapped_model: "PreTrainedModel" = self.accelerator.unwrap_model(self.model)
       if getattr(unwrapped_model.config, "model_type", None) == "chatglm":
           values = torch.transpose(values, 0, 1)
       # Split the inputs and rewards into two parts, chosen and rejected
       batch_size = inputs["input_ids"].size(0) // 2
       chosen_input_ids, rejected_input_ids = inputs["input_ids"][:batch_size], inputs["input_ids"][batch_size:]
       chosen_rewards, rejected_rewards = values[:batch_size], values[batch_size:]
       chosen_scores, rejected_scores = [], []
       # Compute pairwise loss. Only backprop on the different tokens before padding
       # Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
       loss = 0
       for i in range(batch_size):
           chosen_length = (chosen_input_ids[i] != self.tokenizer.pad_token_id).nonzero()[-1] + 1
           rejected_length = (rejected_input_ids[i] != self.tokenizer.pad_token_id).nonzero()[-1] + 1
           check_divergence = (chosen_input_ids[i] != rejected_input_ids[i]).nonzero()
           if len(check_divergence) == 0:
               end_index = chosen_length
               div_index = end_index - 1
           else:
               end_index = max(chosen_length, rejected_length)
               div_index = check_divergence[0]
           assert div_index > 0
           chosen_trunc_rewards = chosen_rewards[i, div_index:end_index]
           rejected_trunc_rewards = rejected_rewards[i, div_index:end_index]
           if return_outputs:  # use the score on the last token except pad token for inference
               chosen_scores.append(chosen_rewards[i, chosen_length - 1])
               rejected_scores.append(rejected_rewards[i, rejected_length - 1])
           loss += -torch.nn.functional.logsigmoid(chosen_trunc_rewards - rejected_trunc_rewards).mean()
       loss = loss / batch_size
       if return_outputs:
           chosen_scores, rejected_scores = torch.stack(chosen_scores), torch.stack(rejected_scores)
           return loss, [loss, chosen_scores, rejected_scores]
       return loss
   ```

   ## encoder-only模型

   主要是使用传统的bert、deberta等模型，作为reward model的基座模型。

   奖励值的输出的方法：

   1. 基于传统的文本分类的方法，如使用cls的hidden state，或者对sequence hidden state进行pooling操作，接线形层进行分类
   2. 直接接一个线形层，使用MSE或者RMSE计算loss，要求训练数据的标签是标量值

   

   

[**点击查看我的更多AI学习笔记github**](https://github.com/xueyongfu11/awesome-deep-learning-resource)









