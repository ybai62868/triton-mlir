import torch

import torch
import time
import csv
import torchvision.models as models
from tabulate import tabulate
import torch._dynamo as dynamo
""" DEBUG
dynamo.config.verbose=True
dynamo.config.suppress_errors=True
"""
 
# dynamo.config.verbose=True
# torch._dynamo.config.suppress_errors = True

device = torch.device("cuda")

def bmm(A, B):
    return torch.bmm(A, B).to(device=device, dtype=torch.float16)

A = torch.rand(4, 1024, 1024, device=device, dtype=torch.float16)
B = torch.rand(4, 1024, 1024, device=device, dtype=torch.float16)

compiled_bmm = torch.compile(bmm, backend="inductor", mode="reduce-overhead")
# compiled_model_reduce = torch.compile(mm, mode="reduce-overhead", backend="inductor")  # 适合小模型
# compiled_model_max = torch.compile(mm, mode="max-autotune", backend="inductor")  # 最大加速比
# result = compiled_softmax(input_tensor)
result = compiled_bmm(A, B)
# result = compiled_model_reduce(A, B)
# result = compiled_model_max(A, B)
