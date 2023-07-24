from dolly_tune import tune

from prepare_tune import get_mod_to_tune

if __name__ == "__main__":
    # ms_rule_type="cuda"
    ms_rule_type="cuda-tensorcore"
    def filter(name):
        if "full" in name:
            # It breacks tuning. Only one trial.
            # Error: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'
            return False
        if "triu_te" in name:
            # TODO: n var is argument. Why?? 
            return False
        if "rotary_embedding" in name:
            # TODO: n var is argument. again
            return False
        if "q_matmul" not in name:
            return False
        if ms_rule_type == "cuda-tensorcore":
            if "NT_matmul" in name:  # FP16 batched matmul, Only ms_rule="cuda"
                return False
            if "matmul3" == name:
                return False         # FP16 batched matmul, Only ms_rule="cuda"
            if "matmul4" == name:
                return False         # FP16 batched matmul, 2 dyn var, only ms_rule="cuda"
        return True
    
    mod = get_mod_to_tune("dynamic_m128", filter=filter)
    tune(mod, work_dir="__tmp/tune_dynamic_m128", max_trials_per_task=2048, ms_rule_type=ms_rule_type)
