import torch

import triton
import triton.language as tl

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=3, num_warps=8),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    A_ptr = A_ptr + (offs_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptr = B_ptr + (offs_b * stride_bb + offs_k[:, None] * stride_bk  + offs_n[None, :] * stride_bn)
    
    # initialize and iteratively update accumulator
   
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):

        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        
        acc += tl.dot(a, b)
    
        A_ptr += BLOCK_SIZE_K * stride_ak
        B_ptr += BLOCK_SIZE_K * stride_bk
        
    c = acc.to(tl.float16)
    C_ptr = C_ptr + (offs_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr, c, mask=c_mask)
     
    
    

# we can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=None):
    # checks constraints
    assert len(a.shape) == len(b.shape) == 3
    assert a.shape[2] == b.shape[1], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    B, M, K = a.shape
    B, K, N = b.shape
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
       triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B, 1
    )
    matmul_kernel[grid](
        a, b, c,
        B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        ACTIVATION=activation,
    )
    return c


torch.manual_seed(0)
a = torch.randn((4, 512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((4, 512, 512), device='cuda', dtype=torch.float16)
torch_output = torch.bmm(a, b)
triton_output = matmul(a, b, activation=None)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
    
    
# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=['B'],  # argument names to use as an x-axis for the plot
#         x_vals=[
#             i for i in range(2, 100)
#         ],  # different possible values for `x_name`
#         line_arg='provider',  # argument name whose value corresponds to a different line in the plot
#         # possible values for `line_arg``
#         line_vals=['cublas', 'triton'],
#         # label name for the lines
#         line_names=["cuBLAS", "Triton"],
#         # line styles
#         styles=[('green', '-'), ('blue', '-')],
#         ylabel="TFLOPS",  # label name for the y-axis
#         plot_name="matmul-performance",  # name for the plot. Used also as a file name for saving the plot.
#         args={},
#     )
# )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 33)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="batch-matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={"BATCH": 4},
    )
)



# def benchmark(B, provider):
#     M = K = N = 512
#     a = torch.randn((B, M, K), device='cuda', dtype=torch.float16)
#     b = torch.randn((B, K, N), device='cuda', dtype=torch.float16)
#     if provider == 'cublas':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(a, b), rep=100)
#     if provider == 'triton':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), rep=100)
#     perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
#     return perf(ms), perf(max_ms), perf(min_ms)

def benchmark(BATCH, M, N, K, provider):
    BATCH = 16
    a = torch.randn((BATCH, M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((BATCH, K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    # max_tensorcore_tflops = triton.testing.get_max_tensorcore_tflops(dtype=torch.float16)
    # max_simd_tflops = triton.testing.get_max_simd_tflops(dtype=torch.float16)
    # print(max_tensorcore_tflops, max_simd_tflops)
    # import pdb; pdb.set_trace()
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: BATCH * 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=False, print_data=False, save_path='.')