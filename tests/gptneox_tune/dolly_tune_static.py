import tvm
from tvm import meta_schedule as ms

from dolly_tune import tune
from prepare_tune import get_mod_to_tune, GPTNEOX_TARGET


def tune(mod: tvm.IRModule, work_dir, *, max_trials_per_task=2048, num_trials_per_iter=32, ms_rule_type="cuda-tensorcore"):    
    ms.tir_integration.tune_tir(
        mod=mod,
        target=GPTNEOX_TARGET,
        work_dir=work_dir,
        max_trials_global=100500,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        space=ms.space_generator.PostOrderApply(                
                sch_rules=ms_rule_type,
                postprocs=ms_rule_type,
                mutator_probs=ms_rule_type,
            ),
    )


def main():
    def filter(name):
        if "rotary_embedding" in name:
            # TODO: unacceptable func. Have to find reason
            return False
        if "q_matmul" not in name:
            return False
        return True

    mod = get_mod_to_tune("static", filter=filter)
    tune(mod, work_dir="__tmp/tune_static", max_trials_per_task=2048)


if __name__ == "__main__":
    main()