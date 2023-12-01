import numpy as np
import os

import tvm 
from tvm import meta_schedule as ms
from tvm import dlight as dl
from tvm import relax

from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.relax.backend.contrib.cublas import partition_for_cublas
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass

import tvm.tir.tensor_intrin.cuda
from mlc_llm.transform import FuseDecodeMatmulEwise

TARGET = tvm.target.Target("nvidia/nvidia-a100")
DEV = tvm.cuda(0)


def make_arg(info):
    if info.dtype in ["float16", "float32"]:
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype(info.dtype)
    elif info.dtype in ["int32", "uint32", "int16", "int8"]:
        arr_np = np.random.randint(0, 16, size=info.shape).astype(info.dtype)
    else:
        assert False, f"Unimplemented, dtype={info.dtype}"

    return tvm.nd.array(arr_np, device=DEV)



@I.ir_module
class linear1_ft_mod:
    @T.prim_func(private=True)
    def decode(A: T.Buffer((T.int64(4096), T.int64(11008)), "int8"), B: T.Buffer((T.int64(22016),), "float16"), decode: T.Buffer((T.int64(4096), T.int64(22016)), "float16")):
        # with T.block("root"):
        for i, j in T.grid(T.int64(4096), T.int64(22016)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i, v_j // T.int64(2)], B[v_j])
                T.writes(decode[v_i, v_j])
                decode[v_i, v_j] = T.Cast("float16", T.shift_right(T.shift_left(T.bitwise_and(T.shift_right(T.Cast("int32", A[v_i, v_j // T.int64(2)]), T.Cast("int32", v_j % T.int64(2)) * 4), 15), 28), 28)) * B[v_j]

    @R.function
    def main(A: R.Tensor((1, "n", 4096), dtype="float16"), 
             B: R.Tensor((T.int64(4096), T.int64(11008)), "int8"),
             B_SCL: R.Tensor((T.int64(22016),), "float16")):
        cls = linear1_ft_mod
        with R.dataflow():
            b_dec = R.call_tir(cls.decode, (B, B_SCL), out_sinfo=R.Tensor((4096, 22016), dtype="float16"))
            x = R.matmul(A, b_dec)
            R.output(x)
        return x

def linear1_ft_arg_info_provider(n):
    return [
        ms.arg_info.TensorInfo("float16", [1, n, 4096]),
        ms.arg_info.TensorInfo("int8", [4096, 11008]),
        ms.arg_info.TensorInfo("float16", [22016]),
    ]

def linear_1_gen(_N, _K): 
    @I.ir_module
    class _mod:
        @R.function
        def main(A: R.Tensor((1, "n", 4096), dtype="float16"), 
                 B: R.Tensor((4096, 22016), dtype="float16")):
            with R.dataflow():
                x = R.matmul(A, B)
                R.output(x)
            return x

    def _arg_info_provider(n):
        return [
            ms.arg_info.TensorInfo("float16", [1, n, 4096]),
            ms.arg_info.TensorInfo("float16", [4096, 22016]),
        ]
    
    return _mod, _arg_info_provider


def compile_relax(mod, with_dlight=False, with_cublas=False, with_cutlass=False, db=None):
    if with_cublas:
        mod = partition_for_cublas(mod)
        mod = relax.transform.RunCodegen()(mod)

    if with_cutlass:
        mod = partition_for_cutlass(mod)
        mod = relax.transform.RunCodegen()(mod)        

    mod = relax.pipeline.get_pipeline()(mod)
    mod = FuseDecodeMatmulEwise()(mod)
    if db is not None:
        with TARGET, db:
            mod = relax.transform.MetaScheduleApplyDatabase()(mod)

    if with_dlight:
        with TARGET:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
            mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)

    with tvm.transform.PassContext(opt_level=3, config={"relax.backend.use_cuda_graph": True}):
        ex = relax.build(mod, TARGET)

    return ex



def main():
    N, K = 4096, 22016
    M = 512

    ex = compile_relax(linear1_ft_mod, with_cutlass=True)

    vm = relax.VirtualMachine(ex, DEV)

    M = 512
    args_info = linear1_ft_arg_info_provider(M)
    args = [make_arg(info) for info in args_info]

    score = vm.time_evaluator("main", dev=DEV, number=100, repeat=1, min_repeat_ms=100)(*args).mean
    score_us = int(float(score)*1e6)
    thrp_tmacps = M * N * K / score_us / 1e6
    print(f"{M} TIME_US {score_us} THRP {thrp_tmacps}")


def main_cublas():
    N, K = 4096, 22016
    M = 512
    mod, arg_info_provider = linear_1_gen(N, K)

    ex = compile_relax(mod, with_cublas=True)

    vm = relax.VirtualMachine(ex, DEV)

    args_info = arg_info_provider(M)
    args = [make_arg(info) for info in args_info]

    score = vm.time_evaluator("main", dev=DEV, number=100, repeat=1, min_repeat_ms=100)(*args).mean
    score_us = int(float(score)*1e6)
    thrp_tmacps = M * N * K / score_us / 1e6
    print(f"{M} TIME_US {score_us} THRP {thrp_tmacps}")


if __name__ == "__main__":
    main()
    # main_cublas()
