import triton
import triton.language as tl
import ast
import astpretty

code = """
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, N,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr+offsets, output, mask=mask)
"""

# # x, y are torch.Tensor
# grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
# add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

ast_tree = ast.parse(code)
print(dir(ast_tree))
print(ast.dump(ast_tree))
print(astpretty.pprint(ast_tree))