
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.verbose = True
torch._inductor.config.debug = True
torch._functorch.config.debug_partitioner = True


isolate_fails_code_str = None



# torch version: 2.2.0a0+gitf9a250c
# torch cuda version: 11.7
# torch git version: f9a250c35bd061e2e6f4c2d92e2b1b16390e8636


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Tue_May__3_18:49:52_PDT_2022 
# Cuda compilation tools, release 11.7, V11.7.64 
# Build cuda_11.7.r11.7/compiler.31294372_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 3090 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2):
        full_default = torch.ops.aten.full.default([1, 3, 224, 224], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        add = torch.ops.aten.add.Tensor(primals_2, full_default);  primals_2 = full_default = None
        convolution = torch.ops.aten.convolution.default(add, primals_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        full_default_1 = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        add_1 = torch.ops.aten.add.Tensor(convolution, full_default_1);  convolution = full_default_1 = None
        return [add_1, primals_1, add]
        
def load_args(reader):
    buf0 = reader.storage(None, 602112, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 3, 224, 224), requires_grad=True, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 602112, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 3, 224, 224), is_leaf=True)  # primals_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
