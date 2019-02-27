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
