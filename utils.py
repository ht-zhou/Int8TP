import torch
from torch import nn, Tensor
import torch.distributed as dist
import bitsandbytes as bnb
import torch.nn.functional as F
from typing import Optional
from torch.distributed.distributed_c10d import ReduceOp
import time
import torch.profiler
import copy

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

class Int8Params(torch.nn.Parameter):
    def __new__(
        cls,
        data=None,
        requires_grad=False,
        has_fp16_weights=False,
        SCB=None,
    ):
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self, data, requires_grad=False):
        super(Int8Params, self).__init__
        self.data = data

class Linear8bitTP(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        has_fp16_weights=False,
        memory_efficient_backward=False,
        threshold=6.0,
        weight_data=None,
        index=None,
        bias_data=None
    ):
        super(Linear8bitTP, self).__init__(
            input_features, output_features, bias
        )
        self.state = bnb.MatmulLtState()
        self.index = index
        self.bias = bias_data
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.register_parameter("SCB", nn.Parameter(torch.empty(0), requires_grad=False))
        # print(self.weight.data.type())
        # print(weight_data.data.type())
        self.weight = weight_data
        

    def quant(self):  
        weight = self.weight.data.contiguous().half().to(self.rank)
        CB, _, SCB, _, _ = bnb.functional.double_quant(weight)
        delattr(self, "weight")
        setattr(self, "weight", nn.Parameter(CB, requires_grad=False))
        delattr(self, "SCB")
        setattr(self, "SCB", nn.Parameter(SCB, requires_grad=False))
        self.weight.data = self.weight.data.to("cpu")
        self.SCB.data = self.SCB.data.to("cpu")

    def forward(self, x):
        self.state.is_training = self.training
        
        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()
        
        self.state.CB = self.weight.data
        self.state.SCB = self.SCB.data
        
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
        tensor_list = [torch.zeros_like(out) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, out)
        out = torch.cat(tensor_list, dim=2)
        del tensor_list
        del self.state.CxB
        
        return out

class Linear8bit(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        has_fp16_weights=False,
        memory_efficient_backward=False,
        threshold=6.0,
        weight_data=None,
        index=None,
        bias_data=None
    ):
        super(Linear8bit, self).__init__(
            input_features, output_features, bias
        )
        self.state = bnb.MatmulLtState()
        self.index = index
        self.bias = bias_data
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        weight = weight_data.data.contiguous().half().to(torch.cuda.current_device())

        CB, _, SCB, _, _ = bnb.functional.double_quant(weight)
        delattr(self, "weight")
        setattr(self, "weight", Int8Params(data=CB, SCB=SCB))

    def forward(self, x):
        self.state.is_training = self.training
        
        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()
        
        self.state.CB = self.weight.data
        self.state.SCB = self.weight.SCB
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
        del self.state.CxB
        
        return out

class LinearTP(torch.nn.Linear):
    def __init__(self, input_features, output_features, bias=False, weight_data=None, bias_data=None):
        super(LinearTP, self).__init__(input_features, output_features, bias)
        self.weight = weight_data
        self.bias = bias_data
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
    def forward(self, x):
        x = x.chunk(self.world_size, dim=2)[self.rank]
        out = F.linear(x, self.weight, self.bias)
        dist.all_reduce(out, op=ReduceOp.SUM)
        return out

class EmbeddingTP(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        weight: Optional[Tensor] = None,
    ) -> None:
        super(EmbeddingTP, self).__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            weight,
        )
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.weight = weight

    def forward(self, input: Tensor) -> Tensor:
        emb = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        
        tensor_list = [torch.zeros_like(emb) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, emb)
        emb = torch.cat(tensor_list, dim=2)
        del tensor_list
        return emb


def replace_8bit_linear_tp(model, threshold=6.0, modules_to_not_convert="lm_head"):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear_tp(module, threshold, modules_to_not_convert)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
                model._modules[name] = Linear8bitTP(
                        input_features=module.in_features,
                        output_features=module.out_features,
                        threshold=6.0,
                        weight_data=module.weight,
                        bias_data=module.bias,
                )
        
        if isinstance(module, nn.Embedding):
            model._modules[name] = EmbeddingTP(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                max_norm=module.max_norm,
                norm_type=module.norm_type,
                scale_grad_by_freq=module.scale_grad_by_freq,
                sparse=module.sparse,
                weight=module.weight,
            )
        if name == 'lm_head':
            model._modules[name] = LinearTP(
                input_features=module.in_features,
                output_features=module.out_features,
                weight_data=module.weight,
                bias=False,
            )
    return model

def replace_8bit_linear(model, threshold=6.0, modules_to_not_convert="lm_head"):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold, modules_to_not_convert)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            model._modules[name] = Linear8bit(
                    input_features=module.in_features,
                    output_features=module.out_features,
                    threshold=6.0,
                    weight_data=module.weight,
                    bias_data=module.bias,
                )
    
    return model

def get_8bit_tp_model(model, rank, world_size):
    model = replace_8bit_linear_tp(model)
    for name, module in model.named_modules():
        if isinstance(module, Linear8bitTP):
            bias_list = list(module.bias.data.chunk(world_size, dim=0))
            bias = bias_list[rank]
            
            weight_list = list(module.weight.data.chunk(world_size, dim=0))
            weight = weight_list[rank]
            # try:
            #     SCB_list = list(module.SCB.data.chunk(world_size, dim=0))
            #     SCB = SCB_list[rank]
            #     print(1)
            # except:
            SCB = torch.zeros_like(bias).to("meta")
            
            delattr(module, "weight")
            setattr(module, "weight", nn.Parameter(weight.to(torch.int8), requires_grad=False))
            
            delattr(module, "SCB")
            setattr(module, "SCB", nn.Parameter(SCB.to(torch.float32), requires_grad=False))
            
            delattr(module, "bias")
            setattr(module, "bias", nn.Parameter(bias))
            
            
        if isinstance(module, EmbeddingTP):   
            weight_list = list(module.weight.chunk(world_size, dim=1))
            delattr(module, 'weight')
            weight = nn.Parameter(weight_list[rank])
            setattr(module, 'weight', weight)
            
        if isinstance(module, LinearTP):
            delattr(module, 'weight')
            setattr(module, 'weight', model._modules['transformer']._modules['word_embeddings'].weight)
    
    return model


def get_8bit_tp_model_list(model, world_size):
    model = replace_8bit_linear_tp(model)
    model_list = []
    for rank in range(world_size):
        model_tmp = copy.deepcopy(model)
        for name, module in model_tmp.named_modules():
            if isinstance(module, Linear8bitTP):
                module.quant()
                
                weight_list = list(module.weight.data.chunk(world_size, dim=0))
                weight = weight_list[rank]

                SCB_list = list(module.SCB.data.chunk(world_size, dim=0))
                SCB = SCB_list[rank]
                
                delattr(module, "weight")
                setattr(module, "weight", nn.Parameter(weight, requires_grad=False))
                
                delattr(module, "SCB")
                setattr(module, "SCB", nn.Parameter(SCB, requires_grad=False))

                bias_list = list(module.bias.data.chunk(world_size, dim=0))
                bias = bias_list[rank]
                delattr(module, "bias")
                setattr(module, "bias", nn.Parameter(bias))
                
                    
            if isinstance(module, EmbeddingTP):   
                weight_list = list(module.weight.chunk(world_size, dim=1))
                delattr(module, 'weight')
                weight = nn.Parameter(weight_list[rank])
                setattr(module, 'weight', weight)
            
            if isinstance(module, LinearTP):
                delattr(module, 'weight')
                setattr(module, 'weight', model_tmp._modules['transformer']._modules['word_embeddings'].weight)
        model_list.append(model_tmp)
        del model_tmp    
    return model_list



from contextlib import contextmanager
@contextmanager
def init_empty_weights():
    old_register_parameter = nn.Module.register_parameter
    
    def register_empty_param(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(torch.device("meta")), **kwargs)
            
    try:
        nn.Module.register_parameter = register_empty_param
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        

