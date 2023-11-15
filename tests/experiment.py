import numpy as np
import tvm.meta_schedule as ms
from tvm import IRModule
from tvm.contrib import utils
import subprocess

import tvm.tir.tensor_intrin.cuda
from mlc_llm.utils import MDS1ScheduleRule
from tvm.script import tir as T

from tvm.contrib import nvcc

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



def get_sass(cubin):
    temp = utils.tempdir()
    temp_cubin = temp.relpath("my_kernel.cubin")
    with open(temp_cubin, "wb") as out_file:
        out_file.write(cubin)
    
    cmd = [ "nvdisasm", "-c", temp_cubin]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg += "\nCompilation error:\n"
        msg += out.decode("utf-8")
        raise RuntimeError(msg)

    return out.decode("utf-8")


def cuda_dump(lib, dump_path="."):
    src = lib.imported_modules[0].get_source()
    with open(f"{dump_path}/shaders.cu", "w") as f:
        print(src, file=f)

    ptx = nvcc.compile_cuda(src, target_format="ptx")
    with open(f"{dump_path}/shaders.ptx", "wb") as f:
        f.write(ptx)

    cubin = nvcc.compile_cuda(src, target_format="cubin")
    # with open(f"{dump_path}/shaders.cubin", "wb") as f:
        # f.write(cubin)

    sass = get_sass(cubin)
    with open(f"{dump_path}/shaders.sass", "w") as f:
        f.write(sass)




def generate_dec_matmul(N, K): 
    @T.prim_func()
    def _func(B: T.Buffer((T.int64(K // 8), T.int64(N)), "uint32"), B_SCL: T.Buffer((T.int64(K // 32), T.int64(N)), "float16"), p_A: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(p_A, (T.int64(1), n, T.int64(K)), "float16")
        var_matmul_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(N)), "float16")
        # with T.block("root"):
        decode_handle_intermediate = T.alloc_buffer((T.int64(K), T.int64(N)), "float16")
        for i, j in T.grid(T.int64(K), T.int64(N)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(B[v_i // T.int64(8), v_j], B_SCL[v_i // T.int64(32), v_j])
                T.writes(decode_handle_intermediate[v_i, v_j])
                decode_handle_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(B[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * B_SCL[v_i // T.int64(32), v_j]
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(N), T.int64(K)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], decode_handle_intermediate[v_k, v_i2])
                T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * decode_handle_intermediate[v_k, v_i2]
    
    return _func



def main():
    # M, N, K = 192*16, 16128, 4032
    M, N, K = 3840, 15360, 3840
    

    func = generate_dec_matmul(N=N, K=K)
    sch = tvm.tir.Schedule(func) 
    
    mds_rule = MDS1ScheduleRule(decisions={
        "m_pad": 192,
        "m_factors":   [1,3,4],
        "n_factors": [1,2,3,4],
        "k_factors":    [1,12],
    })

    sch = mds_rule.apply(sch, sch.get_block("matmul"))[0]
    sch.compute_inline(sch.get_block("decode"))

    with TARGET:
        lib = tvm.build(sch.mod)

    cuda_dump(lib)

    args_info = ms.arg_info.ArgInfo.from_prim_func(func, sym_var_hint={"n": M})
    args = [make_arg(info) for info in args_info]

    score_us = int(lib.time_evaluator(lib.entry_name, DEV)(*args).mean * 1e6)
    print("[EXE TIME US] : ", score_us)
    print("[THRP TMACPS] : ", M*N*K / score_us / 1e6)


if __name__ == "__main__":
    main()