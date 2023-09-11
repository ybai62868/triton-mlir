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
 
dynamo.config.verbose=True
torch._dynamo.config.suppress_errors = True

def softmax(x):
    return torch.nn.Softmax(dim=1)(x)


# input_tensor = torch.randn(1, 128).cuda()
input_tensor = torch.randn(1, 128)
compiled_softmax = torch.compile(softmax, backend="inductor")
# compiled_model_reduce = torch.compile(softmax, mode="reduce-overhead", backend="inductor")  # 适合小模型
# compiled_model_max = torch.compile(softmax, mode="max-autotune", backend="inductor")  # 最大加速比
result = compiled_softmax(input_tensor)
# result = compiled_model_reduce(input_tensor)
