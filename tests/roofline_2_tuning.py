import numpy as np
import os

import tvm 
from tvm import meta_schedule as ms

from tvm.script import ir as I
from tvm.script import tir as T

import tvm.tir.tensor_intrin.cuda
from mlc_llm.utils import dtune_space_gen

TARGET = tvm.target.Target("nvidia/nvidia-a10g")
DEV = tvm.cuda(0)


@T.prim_func(private=True)
def fused_decode_matmul(B: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), B_SCL: T.Buffer((T.int64(128), T.int64(22016)), "float16"), p_A: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(p_A, (T.int64(1), n, T.int64(4096)), "float16")
    var_matmul_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(22016)), "float16")
    # with T.block("root"):
    decode_handle_intermediate = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(B[v_i // T.int64(8), v_j], B_SCL[v_i // T.int64(32), v_j])
            T.writes(decode_handle_intermediate[v_i, v_j])
            decode_handle_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(B[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B_SCL[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], decode_handle_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * decode_handle_intermediate[v_k, v_i2]


def main():
    def make_func(mpad, mval): 
        func = fused_decode_matmul
        func = func.with_attr({"metaschedule.hint.m_pad_value": mpad})
        func = func.with_attr({"metaschedule.hint.dyn_var_value": {"n": mval}})
        return f"linear1_mpad{mpad}_mvar{mval}", func

    tasks = [make_func(i, i) for i in [16,48,64]]   # [16,32,48,64,128]
    tasks = {name: func for name, func in tasks}

    ms.tir_integration.tune_tir(
        mod=tvm.IRModule(tasks),
        target=TARGET,
        work_dir="__tmp/tune_roofline_2_16_48_64",
        max_trials_global=100500,
        # max_trials_per_task=4096,
        max_trials_per_task=256,
        num_trials_per_iter=32,
        cost_model="random", 
        space=dtune_space_gen()
    )


if __name__ == "__main__":
    main()
