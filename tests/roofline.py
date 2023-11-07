import numpy as np
import os

import tvm 
from tvm import meta_schedule as ms
from tvm import dlight as dl
from tvm import relax

from tvm.script import tir as T
from tvm.ir import structural_hash

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


def benchmark(func: tvm.tir.PrimFunc, attrs, db):
    is_dynamic = "n = T.int64()" in func.__str__()

    if not is_dynamic:
        return
 
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

        dt_ker_names = []
        dt_libs = {}
        for m_pad, m_val in attrs:
            dt_func = func
            dt_func = dt_func.with_attr("metaschedule.hint.m_pad_value", m_pad)
            dt_func = dt_func.with_attr("metaschedule.hint.dyn_var_value", {"n": m_val})

            with db:
                dt_mod = tvm.IRModule({"main": dt_func})    
                dt_mod = relax.transform.MetaScheduleApplyDatabase()(dt_mod)
                dt_lib = tvm.build(dt_mod)
                dt_libs[f"dt_mpad{m_pad}_mval{m_val}"] = dt_lib
                dt_ker_names.append(f"dt_mpad{m_pad}_mval{m_val}")

    print("="*20)
    print("N DL", " ".join(dt_ker_names))

    n_range = range(8, 2049, 8) if is_dynamic else range(1, 3)
    for n in n_range:
        args_info = ms.arg_info.ArgInfo.from_prim_func(func, sym_var_hint={"n": n})
        args = [make_arg(info) for info in args_info]

        score = dl_lib.time_evaluator(dl_lib.entry_name, dev=DEV, number=100, repeat=1, min_repeat_ms=100)(*args).mean
        score = str(int(score*1e6))
        score_dl_s = score

        score_dt_s = []
        for ker_name in dt_ker_names:
            dt_lib = dt_libs[ker_name]
            score = dt_lib.time_evaluator(dt_lib.entry_name, dev=DEV, number=100, repeat=1, min_repeat_ms=100)(*args).mean
            score = str(int(score*1e6))
            score_dt_s.append(score)

        print(f"{n} {score_dl_s}", " ".join(score_dt_s))


def strip_db(db, stripped_db_path):
    os.makedirs(stripped_db_path, exist_ok=True)
    stripepd_db = ms.database.JSONDatabase(work_dir=stripped_db_path)

    wklds = []
    for rec in db.get_all_tuning_records():
        if rec.workload not in wklds:
            wklds.append(rec.workload)
    
    for wkld in wklds:
        rec = db.get_top_k(wkld, top_k=1)[0]
        func = wkld.mod["main"]
        mod = tvm.IRModule({"main": func})
        new_rec = ms.database.TuningRecord(
            trace=rec.trace,
            workload=stripepd_db.commit_workload(mod),
            run_secs=rec.run_secs,
            target=rec.target,
            args_info=rec.args_info,
        )
        stripepd_db.commit_tuning_record(new_rec)


def main():
    db_path = "dist/vicuna-v1-7b-q4f16_0/dtune_varios_mpad"
    just_strip = False
    
    if just_strip:
        db = ms.database.JSONDatabase(work_dir=db_path)
        strip_db(db, f"{db_path}_stripped")
        exit()
    else:
        db = ms.database.JSONDatabase(work_dir=f"{db_path}_stripped")
    
    wklds = []
    for rec in db.get_all_tuning_records():
        if rec.workload not in wklds:
            wklds.append(rec.workload)

    # find similar
    func_combined = {}
    for wkld in wklds:
        func = wkld.mod["main"]
        mpad = func.attrs["metaschedule.hint.m_pad_value"]
        mval = func.attrs["metaschedule.hint.dyn_var_value"]["n"]
        func = func.without_attr("metaschedule.hint.m_pad_value")
        func = func.without_attr("metaschedule.hint.dyn_var_value")
        
        struct_hash = structural_hash(func)
        if struct_hash not in func_combined:
            func_combined[struct_hash] = (func, [(mpad, mval)])
        else:
            func_combined[struct_hash][1].append((mpad, mval))


    for i, (_, (func, attrs)) in enumerate(func_combined.items()):
        print(f"Func [{i}]")
        benchmark(func, attrs, db)


if __name__ == "__main__":
    main()
