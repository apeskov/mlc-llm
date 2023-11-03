import numpy as np

import tvm 
from tvm import meta_schedule as ms
from tvm import dlight as dl
from tvm import relax

from tvm.script import tir as T

import tvm.tir.tensor_intrin.cuda

# from tvm.relax.backend import get_patterns_with_prefix
# from tvm.relax.backend.contrib.cutlass import annotate_workspace
# from tvm.contrib.nvcc import parse_compute_version

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


# def compile_ft(func):
#     mod = tvm.IRModule({"main": func})

#     patterns = get_patterns_with_prefix("cutlass.decode_matmul")

#     major, minor = parse_compute_version(tvm.cuda(0).compute_version)

#     if major == 8:
#         sm = 80
#     else:
#         sm = 10 * major + minor

#     options = {"cutlass": {"sm": sm, "find_first_valid": False}}

#     mod = tvm.transform.Sequential(
#         [
#             relax.transform.FuseOpsByPattern(
#                 patterns, bind_constants=False, annotate_codegen=True
#             ),
#             annotate_workspace,
#             relax.transform.AllocateWorkspace(),
#             relax.transform.RunCodegen(options, entry_functions=["main"]),
#         ]
#     )(mod)

#     with TARGET:
#         lib = tvm.build(mod)
    
#     return lib


def benchmark(func, db):
    is_dynamic = "n = T.int64()" in func.__str__()
 
    with TARGET:
        dl_mod = tvm.IRModule({"main": func})    
        dl_mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(dl_mod)
        dl_lib = tvm.build(dl_mod)

        with db:
            dt_mod = tvm.IRModule({"main": func})    
            dt_mod = relax.transform.MetaScheduleApplyDatabase()(dt_mod)
            dt_lib = tvm.build(dt_mod)    

    print("="*20)
    print(func)

    n_range = range(8, 513, 8) if is_dynamic else range(1, 3)
    for n in n_range:
        args_info = ms.arg_info.ArgInfo.from_prim_func(func, sym_var_hint={"n": n})
        args = [make_arg(info) for info in args_info]

        score_dl_s = dl_lib.time_evaluator(dl_lib.entry_name, dev=DEV, number=100, repeat=1, min_repeat_ms=1000)(*args).mean
        score_dt_s = dt_lib.time_evaluator(dt_lib.entry_name, dev=DEV, number=100, repeat=1, min_repeat_ms=1000)(*args).mean

        print(f"N: {n}  DL: {score_dl_s}  DT: {score_dt_s}")


def main():
    db = ms.database.JSONDatabase(work_dir="dist/vicuna-v1-7b-q4f16_0/dtune-a10g")

    wklds = []
    for rec in db.get_all_tuning_records():
        if rec.workload not in wklds:
            wklds.append(rec.workload)

    for i, wkld in enumerate(wklds):
        func = wkld.mod["main"]
        benchmark(func, db)


if __name__ == "__main__":
    main()
