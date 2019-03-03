## fairseq 项目修改 过程

what a huge program = = 

+ 运行过程

fairseq/fairseq_cli/generate.py  __main__

  fairseq/fairseq/process_bar.py 

  tqdm_process_bar (进度条设置)
  
  
+ decoder 部分

```

fairseq/fairseq_cli/generate.py ： hypos = task.inference_step(generator, models, sample, prefix_tokens)

-> fairseq/fairseq/tasks/fairseq_task.py : def inference_step(self, generator, models, sample, prefix_tokens=None):
         return generator.generate(models, sample, prefix_tokens=prefix_tokens)

-> fairseq/fairseq/sequence_generator.py : @torch.no_grad()
                                            def generate(
                                                self,
                                                models,
                                                sample,
                                                prefix_tokens=None,
                                                bos_token=None,
                                                **kwargs
                                            ):
   model = EnsembleModel(models)
   
-> fairseq/fairseq/sequence_generator.py : class EnsembleModel(torch.nn.Module): 

->              @torch.no_grad()
                def forward_decoder(self, tokens, encoder_outs):

->  if len(self.models) == 1:
         return self._decode_one(tokens, self.models[0], encoder_outs[0], self.incremental_states, log_probs=True)

-> def _decode_one(self, tokens, model, encoder_out, incremental_states, log_probs):
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
 
-> fairseq/fairseq/models/fairseq_model.py  def get_normalized_probs(self, net_output, log_probs, sample=None):

-> return self.decoder.get_normalized_probs(net_output, log_probs, sample)

-> fairseq/fairseq/models/fairseq_decoder.py 

            def get_normalized_probs(self, net_output, log_probs, sample):
            
            if log_probs:
                # print("i do not know 5\n")
                return F.log_softmax(logits, dim=-1)  
                
            line 50
```

problems：

+ File "/home/linye/YinFei/fairseq/fairseq/models/fairseq_decoder.py", line 29, in simulation_softmax
  
  if output[now_dim][now_len] > now_dim_max:
  
  RuntimeError: bool value of Tensor with more than one value is ambiguous
    
  语句歧义：未解决  

+ expected type torch.LongTensor but got torch.cuda.LongTensor

  参考文章： https://www.aiuai.cn/aifarm606.html 
  
  参考文章：https://blog.csdn.net/qq_38410428/article/details/82973711
  
  解决：torch.tensor-> torch.tensor.cuda() -> cuda 类型
  
  尝试添加：.cuda().data.cpu().numpy()
  
+ File "/home/linye/YinFei/fairseq/fairseq/sequence_generator.py", line 327, in generate
  
  scores = scores.type_as(lprobs)

  TypeError: type_as(): argument 'other' (position 1) must be Tensor, not numpy.ndarray

  尝试添加：.cuda().data.cpu()
  
  尝试添加： torch.from_numpy(output.cuda().data.cpu().numpy())
  
+ File "/home/linye/YinFei/fairseq/fairseq/models/fairseq_decoder.py", line 45, in simulation_softmax

  return torch.from_numpy(output.cuda().data.cpu().numpy())

  RuntimeError: CUDA error: out of memory

  猜测调用了cuda不能再使用--cpu指令
  
+ 总结解决：

  --cpu 直接return output即可
  
  --gpu 需要使用转化运算
