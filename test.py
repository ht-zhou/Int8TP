import torch
from transformers import BloomTokenizerFast, BloomForCausalLM, BloomConfig
import torch.distributed as dist
import os
from utils import replace_8bit_linear_tp, get_8bit_tp_model, get_8bit_tp_model_list, replace_8bit_linear, getModelSize, init_empty_weights, Linear8bitTP
import time
import sys
import torch.profiler

def run():
        world_size = torch.cuda.device_count()
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        rank = local_rank
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        cpu_group = dist.new_group(backend='gloo') if dist.get_backend() != 'gloo' else None

        # meta init
        if rank == 0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
                profile_memory=True, 
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
                record_shapes=True,
                with_stack=True
                ) as prof:        
                        model = BloomForCausalLM.from_pretrained("/data2/users/lccsr/bloom3b/data").half()
                        model_list = get_8bit_tp_model_list(model, world_size)
                        for name, param in model_list[0].named_parameters():
                                param_list = [param.data]
                                for i in range(1, world_size):
                                        param_list.append(model_list[i].state_dict()[name])
                                param_tensor = torch.zeros_like(param_list[0],dtype=param_list[0].dtype)
                                dist.scatter(param_tensor, scatter_list=param_list, src=0, group=cpu_group)
                        model = model_list[0]
                        del model_list
                        prof.step()
                        

        else:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
                profile_memory=True, 
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
                record_shapes=True,
                with_stack=True
                ) as prof:
                        with init_empty_weights():
                                model = BloomForCausalLM.from_pretrained("/data2/users/lccsr/bloom3b/data").half()
                        model = get_8bit_tp_model(model, rank, world_size)
                        for name, param in model.named_parameters():
                                param_tensor = torch.zeros(param.data.size(), dtype=param.dtype)
                                dist.scatter(param_tensor, src=0, group=cpu_group)
                                param = torch.nn.Parameter(param_tensor, requires_grad=False)
                                name_list = name.split('.')
                                module = model._modules[name_list[0]]
                                for i in range(1, len(name_list) - 1):
                                        module = module._modules[name_list[i]]
                                module._parameters[name_list[-1]] = param
                                del param_tensor
                        model._modules['lm_head']._parameters['weight']= model._modules['transformer']._modules['word_embeddings'].weight  
                        prof.step()
                
        
        model = model.to(rank)
        
        tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lccsr/bloom3b/data")
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(rank)
        
        outputs = model(**inputs, labels=inputs["input_ids"])
        
        print(torch.cuda.max_memory_allocated()/1024/1024, "MB")
                
        
               
                

if __name__ == '__main__':
        run()
