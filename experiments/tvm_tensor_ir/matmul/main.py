import tvm
from typing import Tuple
from tvm import te, tir, topi
# from tvm.script import ir as I, relax as R, tir as T
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.script import ir_module
from tvm.ir.module import IRModule
from tvm.tir import TensorIntrin


# print(dir(tvm.script))
# print(dir(R))
# print(dir(T))

# @ir_module
# class Matmul:
#     @R.function
#     def main(
#         a: R.Tensor((128, 128), "float32"), b: R.Tensor((128, 128), "float32")
#     ) -> R.Tensor((128, 128), "float32"):
#         out: R.Tensor((128, 128), "float32") = R.matmul(a, b)
#         return out
    
# print(type(Matmul))
# print(type(Matmul["main"]))

shape_configs = {"GEMM-1024": [1, 1024, 1024, 1024],
                 "VECADD-2048": [1, 2048]}

def batch_matmul_nkmk_f16(  # pylint: disable=invalid-name,missing-docstring
    B: int,
    N: int,
    M: int,
    K: int,
    out_dtype: str = "float32",
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((B, N, K), name="X", dtype="float16")
    y = te.placeholder((B, M, K), name="Y", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    z = te.compute(
        (B, N, M),
        lambda b, i, j: te.sum(
            tir.Cast(out_dtype, x[b][i][k]) * tir.Cast(out_dtype, y[b][j][k]),
            axis=[k],
        ),
        name="Z",
    )
    return (x, y, z)

def vec_add_n_f16(
    B: int,
    N: int,
    out_dtype: str = "float32",
):
    x = te.placeholder((N, ), name="X", dtype="float16")
    y = te.placeholder((N, ), name="Y", dtype="float16")
    z = te.compute(
        (N, ),
        lambda i: te.sum(
            tir.Cast(out_dtype, x[i]), tir.Cast(out_dtype, y[i]),
        ),
        name="Z",
    )
    return (x, y, z)



def create_te_workload_f16(
    name: str,
    batch_size: int = 1,
    out_dtype="float32",
) -> tir.PrimFunc:
    # workload_func = batch_matmul_nkmk_f16
    workload_func = vec_add_n_f16
    param = [batch_size] + shape_configs[name][1:]
    print(param)
    f = te.create_prim_func(workload_func(*param, out_dtype=out_dtype))  # type: ignore
    ir_module_from_te = IRModule({"main": f})
    with open("ir_module_from_te.txt", "w") as f_write:
        f_write.write(ir_module_from_te.script())
    # print(ir_module_from_te.script())
    return f


def tune(workload: str, batch: int):
    mod = create_te_workload_f16(workload, batch_size=batch, out_dtype="float16")
    # sch = ms.tune_tir(
    #     mod=mod,
    #     target=ARGS.target,
    #     config=get_search_config(ARGS.num_trials, ARGS.num_trials),
    #     work_dir=f"{ARGS.work_dir}/TIR/{workload}-{batch_size}/{ARGS.out_dtype}",
    #     builder=ms.builder.LocalBuilder(f_build=cuda_build),
    #     runner=ARGS.runner,  # type: ignore
    #     sch_rules=sch_rules_tensor_core,
    #     postprocs=postprocs_tensor_core,
    # )
    print(mod)


if __name__ == "__main__":
    # tune("GEMM-1024", batch=1)
    tune("VECADD-2048", batch=1)