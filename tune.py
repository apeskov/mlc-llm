import tvm
import tvm.meta_schedule as ms

import tvm.tir.tensor_intrin.cuda

def tune(mod: tvm.IRModule):
    func_to_exclude = [
        # ===== Decode body =====
        "rotary_embedding1",
    ]

    # ms_rule_type = "cuda"
    ms_rule_type = "cuda-tensorcore"
    target = tvm.target.Target("nvidia/nvidia-t4")
    work_dir = "dolly_tune_2"
    
    funcs = {gv.name_hint:mod[gv] for gv in mod.get_global_vars() if gv.name_hint not in func_to_exclude}
    print(funcs)

    database = ms.tir_integration.tune_tir(
        mod=tvm.ir.IRModule(funcs),
        target=target,
        work_dir=work_dir,
        max_trials_global=100500,
        max_trials_per_task=32,
        num_trials_per_iter=8,
        space=ms.space_generator.PostOrderApply(                
                sch_rules=ms_rule_type,
                postprocs=ms_rule_type,
                mutator_probs=ms_rule_type,
            ),
    )


if __name__ == "__main__":
    from dist.to_tune import Module as dolly_stat_tir
    tune(dolly_stat_tir)
