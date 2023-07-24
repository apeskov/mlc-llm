
from pathlib import Path
from typing import List

import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm.script import tir as T

import tvm.tir.tensor_intrin.cuda

from dolly_convert_to_dynamic import generate_arg
from prepare_tune import get_mod_to_tune, GPTNEOX_TARGET
from dynm_mutator import DynMMutator, AsIs

"""
A10G 
Memory bandwidth : 600 GBps
Compute TCores   :
"""

def _generate_arg_info(m):
    HS = 5120
    GS = 128
    return [
        ms.arg_info.TensorInfo("float16", [1, m, HS]),        #A
        ms.arg_info.TensorInfo("int32",   [HS//8, HS*4]),     #WGH
        ms.arg_info.TensorInfo("int32",   [HS//GS, HS*4//8]), #ZP
        ms.arg_info.TensorInfo("float16", [HS//GS, HS*4]),    #SCL
        ms.arg_info.TensorInfo("float16", [HS*4]),            #BIAS
        ms.arg_info.TensorInfo("float16", [1, m, HS*4]),      #OUT
    ]   
    

def benchmark():
    mxx = "m128"
    db_name = f"__tmp/tune_dynamic_{mxx}"
    target = GPTNEOX_TARGET

    mod_dyn_mxx = get_mod_to_tune(f"dynamic_{mxx}")
    mod_dyn = get_mod_to_tune("dynamic")

    db = ms.database.JSONDatabase(work_dir=db_name, allow_missing=False)

    fucn_name = "fused_q_matmul2_gelu1"
    converter = DynMMutator(base_block_name="matmul", pad_factors=[32, 1, 1])

    # Convert scedules for dynm
    func_dyn_mxx = mod_dyn_mxx[fucn_name]
    func_dyn = mod_dyn[fucn_name]
    
    rec = db.query(ms.tir_integration._normalize_mod(func_dyn_mxx), target=target, kind="record")
    
    dyn_trace = converter(rec.trace)
    dyn_sch = tvm.tir.Schedule(func_dyn)
    try:
        dyn_trace.apply_to_schedule(dyn_sch, remove_postproc=False)
    except:
        print("Failed to apply trace")
        exit()
    
    with target:
        lib = tvm.build(dyn_sch.mod["main"])
    
    print(f"Benchmark func {fucn_name}")
    dev = tvm.cuda(0)
    for m in range(1, 256):
        args_info = _generate_arg_info(m)
        args = [generate_arg(info, dev) for info in args_info]
        
        score_s = lib.time_evaluator(lib.entry_name, dev=dev, number=2000, repeat=1)(*args).mean
        print(f"[{m}]  dur {score_s * 1e6} us")


if __name__ == "__main__":
    benchmark()
