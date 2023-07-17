import torch
import triton
import triton.language as tl
from triton.compiler import compile
from triton.runtime import JITFunction

def program(b0, b1, b2):
  idx = tl.program_id(0)
  x = tl.load(b1 + idx)
  y = tl.load(b2 + idx)
  tl.store(b0 + idx, x+y)


def program_0(b0, b1, b2):
   idx = tl.program_id(0)
   x = tl.load(b1+ idx)
   y = tl.load(b2+idx)
   tl.store(b0+idx, x*y)


def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)


# program_jit = JITFunction(program)
program_jit = JITFunction(program_0)


# JITFunction(__main__:program) {'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'constants': {}, 'num_warps': 4, 'num_stages': 3, 'extern_libs': None, 'configs': (instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),)}
# ast -> ttir -> ttgir -> llir -> ptx -> cubin
compiled = compile(program_jit, signature={0: '*fp16', 1: '*fp16', 2: '*fp16'})
# print(compiled.asm['ast'])
print(compiled.asm['ttir'])
print(compiled.asm['ttgir'])
# print(eval(compiled.asm['llir']).decode('utf-8'))
#print(compiled.asm['ptx'])

print("running")
size = 4
x = torch.ones(size, device='cuda')
y = torch.ones(size, device='cuda')
output = torch.empty_like(x)
out = compiled[(output.numel(),1,1)](output, x, y)
print(output)