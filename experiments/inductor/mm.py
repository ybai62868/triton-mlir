import torch

import torch
import time
import csv
import torchvision.models as models
from tabulate import tabulate

import torch._dynamo.config
import torch._inductor.config as config


print(torch._inductor.config.__file__)
# print(dir(torch._inductor.config.triton))
print(dir(torch._inductor.config.triton._config.triton))

import torch._dynamo as dynamo
""" DEBUG
dynamo.config.verbose=True
dynamo.config.suppress_errors=True
"""
 
# dynamo.config.verbose=True
# torch._dynamo.config.suppress_errors = True
torch._inductor.config.triton.autotune_pointwise = True
# torch._inductor.config.max_autotune_gemm_backends

device = torch.device("cuda")

def inductor_triton_mm(A, B):
    return torch.mm(A, B).to(device=device, dtype=torch.float16)

A = torch.rand(1024, 1024, device=device, dtype=torch.float16)
B = torch.rand(1024, 1024, device=device, dtype=torch.float16)

compiled_mm = torch.compile(inductor_triton_mm, backend="inductor")
# compiled_model_reduce = torch.compile(mm, mode="reduce-overhead", backend="inductor")  # 适合小模型
# compiled_model_max = torch.compile(mm, mode="max-autotune", backend="inductor")  # 最大加速比
# result = compiled_softmax(input_tensor)
# result = compiled_model_reduce(A, B)
# result = compiled_model_max(A, B)
result = compiled_mm(A, B)
