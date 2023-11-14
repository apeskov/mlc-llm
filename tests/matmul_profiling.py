import numpy as np
import argparse
import os

import tvm
from tvm import meta_schedule as ms
from tvm import dlight as dl
from tvm import relax

from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.target import Target
from tvm.relax.backend.contrib.cublas import partition_for_cublas
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass

import tvm.tir.tensor_intrin.cuda

from collections import namedtuple
from typing import List


DEV = tvm.cuda(0)
M_PAD = 256


def make_arg(info):
    if info.dtype in ["float16", "float32"]:
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype(info.dtype)
    elif info.dtype in ["int32", "uint32", "int16", "int8"]:
        arr_np = np.random.randint(0, 16, size=info.shape).astype(info.dtype)
    else:
        assert False, f"Unimplemented, dtype={info.dtype}"

    return tvm.nd.array(arr_np, device=DEV)


@I.ir_module
class matmul_mod:
    @R.function
    def main(
        A: R.Tensor((1, "n", 4096), dtype="float16"), B: R.Tensor((4096, 22016), dtype="float16")
    ):
        with R.dataflow():
            x = R.matmul(A, B)
            R.output(x)
        return x


def matmul_arg_info_provider(n):
    return [
        ms.arg_info.TensorInfo("float16", [1, n, 4096]),
        ms.arg_info.TensorInfo("float16", [4096, 22016]),
    ]


def transform_mod(mod, target, with_dlight=False, with_cublas=False, with_cutlass=False, db=None):
    if with_cublas:
        mod = partition_for_cublas(mod)
        mod = relax.transform.RunCodegen()(mod)

    if with_cutlass:
        mod = partition_for_cutlass(mod)
        mod = relax.transform.RunCodegen()(mod)

    mod = relax.pipeline.get_pipeline()(mod)
    #from mlc_llm.transform import FuseDecodeMatmulEwise
    #mod = FuseDecodeMatmulEwise()(mod)
    if db is not None:
        with target, db:
            mod = relax.transform.MetaScheduleApplyDatabase()(mod)

    if with_dlight:
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
            mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)

    return mod


def load_ms_db(tune_work_dir):
    if tune_work_dir is None:
        return None
    db = ms.database.JSONDatabase(work_dir=tune_work_dir, allow_missing=False)
    # Read all workloads
    wklds = []
    for rec in db.get_all_tuning_records():
        if rec.workload not in wklds:
            wklds.append(rec.workload)

    # Clean workloads from dtune specific hints
    mem_db = ms.database.MemoryDatabase()
    for wkld in wklds:
        rec = db.get_top_k(wkld, top_k=1)[0]
        func = wkld.mod["main"]
        func = func.without_attr("metaschedule.hint.dyn_var_value")
        func = func.without_attr("metaschedule.hint.m_pad_value")
        mod = tvm.IRModule({"main": func})
        new_rec = ms.database.TuningRecord(
            trace=rec.trace,
            workload=mem_db.commit_workload(mod),
            run_secs=rec.run_secs,
            target=rec.target,
            args_info=rec.args_info,
        )

        mem_db.commit_tuning_record(new_rec)
    return mem_db


def main():
    seq_len = 2000
    mod = matmul_mod
    target_c = "nvidia/nvidia-a100"
    target_h = "llvm -mcpu=tigerlake"
    schedule_types = ["tvm", "cublas", "dlight"]
    parser = argparse.ArgumentParser(description="Measure matmul exec time")
    parser.add_argument("--target", default=target_c, help=f"Specify target (Default: {target_c})")
    parser.add_argument(
        "--target_host", default=target_h, help=f"Specify target host (Default: {target_h})"
    )
    parser.add_argument("--tune_only", action="store_true", help="Only tune this model")
    parser.add_argument(
        "--tune_work_dir", default=None, help="Path to dir with meta-scheduler database"
    )
    parser.add_argument(
        "--schedule_type",
        default=schedule_types[0],
        help=f"Select schedule type (Default: {schedule_types[0]})",
        choices=schedule_types,
    )
    parser.add_argument(
        "--seq_len",
        default=seq_len,
        type=int,
        help=f"Select seq_len (Default: {seq_len})",
    )
    args = parser.parse_args()
    target_c = args.target
    target_h = args.target_host
    target = Target(target_c, host=target_h)
    seq_len = args.seq_len
    if args.tune_only:
        if args.tune_work_dir is None:
            raise Exception("--tune_work_dir must be specified with --tune_only")
        mod = transform_mod(mod, target)
        for gv, func in mod.functions.items():  # pylint: disable=invalid-name
            if gv.name_hint == "matmul":
                func = func.with_attr({"metaschedule.hint.m_pad_value": M_PAD})
                func = func.with_attr({"metaschedule.hint.dyn_var_value": {"n": seq_len}})
                mod.update_func(gv, func)
        from mlc_llm.utils import dtune_space_gen
        ms.tir_integration.tune_tir(
            mod=mod,
            target=target,
            work_dir=args.tune_work_dir,
            max_trials_global=100500,
            # max_trials_per_task=4096,
            max_trials_per_task=256,
            num_trials_per_iter=32,
            cost_model="random", 
            space=dtune_space_gen()
        )
        exit(0)

    with_dlight = False if args.schedule_type != "dlight" else True
    with_cublas = False if args.schedule_type != "cublas" else True
    db = load_ms_db(args.tune_work_dir)
    mod = transform_mod(mod, target, with_dlight=with_dlight, with_cublas=with_cublas, db=db)

    with tvm.transform.PassContext(opt_level=3, config={"relax.backend.use_cuda_graph": True, "cuda.kernels_output_dir": "my_cuda_kernels"}):
        ex = relax.build(mod, target)

    vm = relax.VirtualMachine(ex, DEV)

    args_info = matmul_arg_info_provider(seq_len)
    args = [make_arg(info) for info in args_info]
    score = vm.time_evaluator("main", dev=DEV, number=100, repeat=1, min_repeat_ms=100)(*args).mean
    score_us = int(float(score) * 1e6)
    print("{}: {} us".format(seq_len, score_us))


if __name__ == "__main__":
    main()
