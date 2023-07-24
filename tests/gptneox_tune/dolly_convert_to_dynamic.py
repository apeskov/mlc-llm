from pathlib import Path
from typing import List

import numpy as np
import tvm
from tvm import meta_schedule as ms

import tvm.tir.tensor_intrin.cuda

from dynm_mutator import DynMMutator, AsIs
from prepare_tune import GPTNEOX_TARGET, get_mod_to_tune

def get_all_funcs(mod: tvm.ir.module.IRModule):
    return {gv.name_hint:mod[gv] for gv in mod.get_global_vars() if gv.name_hint}


def change_m(args_info: List[ms.arg_info.ArgInfo], *, m=None):
    """ 
    Convert args info to cover different M value
    Use some heuristics. Like M dim used only for data input and data output tensors, shich has first and last
    position in args_info list
    """
    res = []
    for info in args_info:
        shape = list(info.shape)
        if len(shape) == 3:
            shape[1] = m
        res.append(ms.arg_info.TensorInfo(info.dtype, shape))

    return res


def generate_arg(info: ms.arg_info.ArgInfo, dev: tvm.runtime.Device):
    if info.dtype in ["float16", "float32"]:
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype(info.dtype)
    elif info.dtype in ["int32", "int16", "int8"]:
        arr_np = np.random.randint(0, 16, size=info.shape).astype(info.dtype)
    else:
        assert False, "Unimplemented"

    return tvm.nd.array(arr_np, device=dev)


def bench_mod(mod: tvm.ir.module.IRModule, 
              trace: tvm.tir.schedule.Trace, 
              args_info: ms.arg_info.ArgInfo,
              target: tvm.target.Target 
              ) -> float:
    dev = tvm.cuda(0)
    
    sch = tvm.tir.Schedule(mod)
    trace.apply_to_schedule(sch, remove_postproc=False)

    with target:
        lib = tvm.build(sch.mod["main"])
    
    args = [generate_arg(info, dev) for info in args_info]
    score_s = lib.time_evaluator(lib.entry_name, dev=dev, number=2000, repeat=1)(*args).mean
    
    return score_s


def combine_db(from_db_name, to_db_name):
    Path(to_db_name).mkdir(parents=True, exist_ok=True)

    in_db = ms.database.JSONDatabase(work_dir=from_db_name, allow_missing=False)
    out_db = ms.database.JSONDatabase(work_dir=to_db_name, allow_missing=True)
    
    target = GPTNEOX_TARGET

    # get all workloads
    visited_workloads = {}
    for rec in in_db.get_all_tuning_records():
        if rec.workload in visited_workloads:
            continue
        visited_workloads[rec.workload] = 1
        
        mod = rec.workload.mod
        in_rec = in_db.query(mod, target, kind="record")
        
        workload = out_db.commit_workload(mod)
        rec = ms.database.TuningRecord(in_rec.trace, workload, run_secs=in_rec.run_secs, target=in_rec.target)
        out_db.commit_tuning_record(rec)


def convert_db(
        in_db: ms.database.Database, 
        out_db: ms.database.Database,
        cfg: list,
        target: tvm.target.Target):
    """ convert schedules from in_db to dynamic version and store to out_db """

    for name, s_mod, d_mod, dynm_trace_mutator in cfg:        
        # Convert sttaic trace to dynamic version
        s_rec: ms.database.TuningRecord = in_db.query(ms.tir_integration._normalize_mod(s_mod), target=target, kind="record")
        if s_rec is None:
            print(f"Func {name} not found in DB. Skip it.")
            continue
        
        dyn_trace = dynm_trace_mutator(s_rec.trace)
        
        # Benchmark score
        arg_info = ms.arg_info.ArgInfo.from_prim_func(s_mod)
        d_score_m32 = bench_mod(d_mod, dyn_trace, arg_info, target=target)
        # d_score_m1 = bench_mod(d_mod, d_trace, change_m(arg_info, m=1), dev=dev, target=target)

        # just print result
        print(f"{name} DynM==32: {d_score_m32}  db_StaticM=32: {s_rec.run_secs[0]}")

        d_workload = out_db.commit_workload(ms.tir_integration._normalize_mod(d_mod))
        d_rec = ms.database.TuningRecord(dyn_trace, d_workload, run_secs=[d_score_m32])
        out_db.commit_tuning_record(d_rec)


def main():
    in_db_name = "__tmp/tune_dynamic_m32"
    out_db_name = "__tmp/tune_dynamic"
    target = GPTNEOX_TARGET

    # Create output folder if not exists
    Path(out_db_name).mkdir(parents=True, exist_ok=True)

    acceptable_names = [
        "fused_q_matmul1_add1_add1",
        "fused_q_matmul2_gelu1",
        "q_matmul",
        "q_matmul3",
        # "fused_NT_matmul1_divide2_maximum1_minimum1_cast3",     # To long....
        # "fused_NT_matmul_divide1_maximum_minimum_cast",         # Previously it was OK. Now fail during compilation: "entry function 'default_function_kernel0' uses too much shared data"   
        "fused_softmax1_cast1",
        "fused_softmax2_cast4",
        "fused_squeeze1",
        "layer_norm1",
        "matmul3",    # ???
        # "matmul4",     # This lead to wrong out
        "reshape3",
        "reshape5",
        "reshape6",
        "reshape7",
        "reshape8",
        "slice",
        "split1",
        "squeeze1",       
        # "take1",         # doesn't tuned. No record inside DB  
        "transpose2",
        "transpose4",
    ]
    
    mod_dyn_m32 = get_mod_to_tune("dynamic_m32")
    mod_dyn = get_mod_to_tune("dynamic")

    # Open input and output db
    in_db = ms.database.JSONDatabase(work_dir=in_db_name, allow_missing=False)
    out_db = ms.database.JSONDatabase(work_dir=out_db_name, allow_missing=True)

    def make_converter(func_name):
        if "q_matmul" in func_name:
            return DynMMutator(base_block_name="matmul", pad_factors=[32, 1, 1])
        if "fused_NT_matmul1_divide2_maximum1_minimum1_cast3" == func_name:
            return DynMMutator(base_block_name="NT_matmul", pad_factors=[1, 1, 32, 32, 1], padded_mask_in=[1, 1])
        if "fused_NT_matmul_divide1_maximum_minimum_cast" == func_name:
            return DynMMutator(base_block_name="NT_matmul", pad_factors=[1, 1, 1, 32, 1], padded_mask_in=[1])
        if "matmul3" == func_name:
            return DynMMutator(base_block_name="matmul", pad_factors=[1, 1, 1, 1, 32], padded_mask_in=[1, 1], padded_mask_out=[])
        if "matmul4" == func_name:
            return DynMMutator(base_block_name="matmul", pad_factors=[1, 1, 32, 1, 32], padded_mask_in=[1, 1], padded_mask_out=[1])

        return AsIs()  # "softmax"

    # Convert scedules for dynm
    all_func_dyn_m32 = get_all_funcs(mod_dyn_m32)
    all_func_dyn = get_all_funcs(mod_dyn)
    config =[(name, all_func_dyn_m32[name], all_func_dyn[name], make_converter(name)) for name in all_func_dyn_m32.keys() if name in acceptable_names]

    convert_db(in_db, out_db, cfg=config, target=target)


if __name__ == "__main__":
    main()