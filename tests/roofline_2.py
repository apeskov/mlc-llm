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
from mlc_llm.utils import dtune_load_db


TARGET = tvm.target.Target("nvidia/nvidia-a10g")
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


@I.ir_module
class linear1_q4_mod:
    @T.prim_func(private=True)
    def decode(A: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), B: T.Buffer((T.int64(128), T.int64(22016)), "float16"), decode: T.Buffer((T.int64(4096), T.int64(22016)), "float16")):
        # with T.block("root"):
        for i, j in T.grid(T.int64(4096), T.int64(22016)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i // T.int64(8), v_j], B[v_i // T.int64(32), v_j])
                T.writes(decode[v_i, v_j])
                decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B[v_i // T.int64(32), v_j]

    @R.function
    def main(A: R.Tensor((1, "n", 4096), dtype="float16"), 
             B: R.Tensor((T.int64(512), T.int64(22016)), "uint32"),
             B_SCL: R.Tensor((T.int64(128), T.int64(22016)), "float16")):
        cls = linear1_q4_mod
        with R.dataflow():
            b_dec = R.call_tir(cls.decode, (B, B_SCL), out_sinfo=R.Tensor((4096, 22016), dtype="float16"))
            x = R.matmul(A, b_dec)
            R.output(x)
        return x

def linear1_q4_arg_info_provider(n):
    return [
        ms.arg_info.TensorInfo("float16", [1, n, 4096]),
        ms.arg_info.TensorInfo("uint32", [512, 22016]),
        ms.arg_info.TensorInfo("float16", [128, 22016]),
    ]

@I.ir_module
class linear1_q4_static_mod:
    @T.prim_func(private=True)
    def fused_fused_decode4_NT_matmul2(lv16: T.Buffer((T.int64(22016), T.int64(512)), "uint32"), lv17: T.Buffer((T.int64(22016), T.int64(128)), "float16"), lv1: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_NT_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
        for i, j in T.grid(T.int64(22016), T.int64(4096)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv16[v_i, v_j // T.int64(8)], lv17[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate[v_i, v_j])
                p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv16[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv17[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

    @R.function
    def main(A: R.Tensor((1, 1, 4096), dtype="float16"),
             B: R.Tensor((T.int64(22016), T.int64(512)), "uint32"),
             B_SCL: R.Tensor((T.int64(22016), T.int64(128)), "float16")):
        cls = linear1_q4_static_mod
        with R.dataflow():
            x = R.call_tir(cls.fused_fused_decode4_NT_matmul2, (B, B_SCL, A), out_sinfo=R.Tensor((1, 1, 22016), dtype="float16"))
            R.output(x)
        return x


def linear1_q4_static_arg_info_provider(n):
    return [
        ms.arg_info.TensorInfo("float16", [1, 1, 4096]),
        ms.arg_info.TensorInfo("uint32", [22016, 512]),
        ms.arg_info.TensorInfo("float16", [22016, 128]),
    ]


@I.ir_module
class linear1_q4_static_2_mod:
    @T.prim_func(private=True)
    def fused_fused_decode4_NT_matmul2(lv16: T.Buffer((T.int64(22016), T.int64(512)), "uint32"), lv17: T.Buffer((T.int64(22016), T.int64(128)), "float16"), lv1: T.Buffer((T.int64(1), T.int64(2), T.int64(4096)), "float16"), var_NT_matmul_intermediate: T.Buffer((T.int64(1), T.int64(2), T.int64(22016)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
        for i, j in T.grid(T.int64(22016), T.int64(4096)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv16[v_i, v_j // T.int64(8)], lv17[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate[v_i, v_j])
                p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv16[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv17[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(2), T.int64(22016), T.int64(4096)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

    @R.function
    def main(A: R.Tensor((1, 2, 4096), dtype="float16"),
             B: R.Tensor((T.int64(22016), T.int64(512)), "uint32"),
             B_SCL: R.Tensor((T.int64(22016), T.int64(128)), "float16")):
        cls = linear1_q4_static_2_mod
        with R.dataflow():
            x = R.call_tir(cls.fused_fused_decode4_NT_matmul2, (B, B_SCL, A), out_sinfo=R.Tensor((1, 2, 22016), dtype="float16"))
            R.output(x)
        return x


def linear1_q4_static_2_arg_info_provider(n):
    return [
        ms.arg_info.TensorInfo("float16", [1, 2, 4096]),
        ms.arg_info.TensorInfo("uint32", [22016, 512]),
        ms.arg_info.TensorInfo("float16", [22016, 128]),
    ]


@I.ir_module
class linear1_mod:
    @R.function
    def main(A: R.Tensor((1, "n", 4096), dtype="float16"), 
             B: R.Tensor((4096, 22016), dtype="float16")):
        with R.dataflow():
            x = R.matmul(A, B)
            R.output(x)
        return x


def linear1_arg_info_provider(n):
    return [
        ms.arg_info.TensorInfo("float16", [1, n, 4096]),
        ms.arg_info.TensorInfo("float16", [4096, 22016]),
    ]


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


def benchmark(ex, arg_info_provider):
    vm = relax.VirtualMachine(ex, DEV)
    
    print(f"NB TIME_US")
    for m in range(1, 512):
        args_info = arg_info_provider(m)
        args = [make_arg(info) for info in args_info]

        score = vm.time_evaluator("main", dev=DEV, number=100, repeat=1, min_repeat_ms=100)(*args).mean
        score_us = int(float(score)*1e6)
        print(f"{m} {score_us}")


def main():
    mod = linear1_mod
    arg_info_provider = linear1_arg_info_provider

    # ex = compile_relax(mod, with_cublas=True)
    # benchmark(ex, arg_info_provider)

    # ex = compile_relax(mod, with_dlight=True)
    # benchmark(ex, arg_info_provider)

    mod = linear1_ft_mod
    arg_info_provider = linear1_ft_arg_info_provider

    # ex = compile_relax(mod, with_cutlass=True)
    # benchmark(ex, arg_info_provider)

    # ex = compile_relax(mod, with_dlight=True)
    # benchmark(ex, arg_info_provider)

    mod = linear1_q4_mod
    arg_info_provider = linear1_q4_arg_info_provider

    # ex = compile_relax(mod, with_dlight=True)
    # benchmark(ex, arg_info_provider)

    # args = lambda: None
    # args.tune_db_path = "__tmp/tune_roofline_2_16_48_64"
    # mpad = 64
    # def filter(attrs):
    #     return attrs["metaschedule.hint.dyn_var_value"]["n"] == mpad and attrs["metaschedule.hint.m_pad_value"] == mpad
    
    # db = dtune_load_db(args, filter=filter)
    # ex = compile_relax(mod, db=db)
    # benchmark(ex, arg_info_provider)

    # mod = linear1_q4_static_mod
    # arg_info_provider = linear1_q4_static_arg_info_provider
    # ex = compile_relax(mod, with_dlight=True)
    # benchmark(ex, arg_info_provider)

    # mod = linear1_q4_static_2_mod
    # arg_info_provider = linear1_q4_static_2_arg_info_provider
    # ex = compile_relax(mod, with_dlight=True)
    # benchmark(ex, arg_info_provider)


if __name__ == "__main__":
    main()
