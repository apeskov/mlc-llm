import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
from typing import List, Tuple, Dict, Optional

import tvm 
from tvm import relax
from tvm.relax.expr_functor import PyExprVisitor, visitor


TARGET = tvm.target.Target("nvidia/nvidia-a100")
DEV = tvm.cuda(0)


"""
Benchmark each linear kernel inside prefill of compiled model
And plot performance results in single chart.
"""

def make_arg(shape, dtype, hint: Dict[str, int]={}):
    shape = [d if isinstance(d, int) else hint[d] for d in shape]

    if dtype in ["float16", "float32"]:
        arr_np = np.random.uniform(-1, 1, size=shape).astype(dtype)
    elif dtype in ["int32", "uint32", "int16", "int8"]:
        arr_np = np.random.randint(0, 16, size=shape).astype(dtype)
    else:
        assert False, f"Unimplemented, dtype={dtype}"

    return tvm.nd.array(arr_np, device=DEV)


@visitor
class LinearFuncCollector(PyExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.collection = []

    def visit_call_(self, op) -> None:
        if not isinstance(op.op, tvm.ir.op.Op):
            return

        def parse_shape(arg):
             return [int(d) if not isinstance(d, tvm.tir.expr.Var) else d.name for d in arg.struct_info.shape]

        if op.op.name == "relax.call_tir":
            name = op.args[0].name_hint
        elif op.op.name == "relax.call_dps_packed":
            name = op.args[0].global_symbol
        else:
            return
        
        if "matmul" not in name:
            return

        args_info = [(parse_shape(arg), arg.struct_info.dtype) for arg in op.args[1]]
        args_info.append((parse_shape(op), op.struct_info.dtype))  # return value is last arg in case of DPS call 

        if name not in [name for name, _ in self.collection]:
            self.collection.append((name, args_info))


def plot_folder(log_dir):
    def read_csv(csv_file_path):
        m, score = [], []
        with open(csv_file_path, "r") as f:
            for line in f:
                m_, s_  = line.split(" ")
                m.append(int(m_))
                score.append(int(s_))
        return m, score
    
    def read_wgh_shape(desc_txt_file):
        with open(desc_txt_file, "r") as f:
            content = f.read()
            content = json.loads(content)
            n = content["N"]
            k = content["K"]
        return n, k
    
    list_of_wkld = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]

    fig, axs = plt.subplots(len(list_of_wkld), 2, layout='constrained', figsize=(15, 20))
    fig.get_layout_engine().set(hspace=0.1, wspace=0.1)

    for wkld_name, (axs_time, axs_thrp) in zip(list_of_wkld, axs):
        list_of_impl = [f for f in os.listdir(f"{log_dir}/{wkld_name}") if os.path.isdir(os.path.join(f"{log_dir}/{wkld_name}", f))]

        for impl_name in list_of_impl:
            m, score = read_csv(f"{log_dir}/{wkld_name}/{impl_name}/score.csv")
            n, k = read_wgh_shape(f"{log_dir}/{wkld_name}/{impl_name}/desc.json")

            thrp = [int(m_*n*k/scr/1e6) for m_, scr in zip(m, score)]

            axs_thrp.plot(m, thrp, label=impl_name)
            axs_time.plot(m, score, label=impl_name)
        
        axs_time.set_title(f"{wkld_name} Mx{n}x{k}, time")
        axs_time.set_xlabel("M")
        axs_time.set_ylabel('Time, us')
        axs_time.grid(True)
        axs_time.legend(loc="upper left")

        axs_thrp.set_title(f"{wkld_name} Mx{n}x{k}, throuphput")
        axs_thrp.set_xlabel("M")
        axs_thrp.set_ylabel('Throughput, TMACps')
        axs_thrp.grid(True)

    plt.savefig(f"{log_dir}/graph.png")
    plt.savefig(f"{log_dir}/graph.svg")


def store_score(log_path,
                workload_name: str,
                impl_name: str,
                func_name: str,
                args_info: Tuple[List, str],
                func: Optional[tvm.tir.PrimFunc], 
                score: List[Tuple[int, int]]):
    store_folder = f"{log_path}/{workload_name}/{impl_name}" 
    os.makedirs(store_folder, exist_ok=True)

    with open(f"{store_folder}/desc.py", "w", encoding="utf-8") as f:
        print(func, file=f)

    with open(f"{store_folder}/desc.json", "w", encoding="utf-8") as f:
        shapes = [shape for shape, _ in args_info]
        shapes = [shape for shape in shapes if any([isinstance(d, str) for d in shape])]
        assert len(shapes) >= 2
        k = shapes[0][-1]  # First dinamic tensor is Data tensor [1, dyn_m, k]
        n = shapes[-1][-1] # Last dinamic tensor is output tensor [1, dyn_m, n]

        content = json.dumps({
            "func_name": func_name,
            "impl_type": impl_name,
            "args_info": args_info,
            "K": k,
            "N": n,
        }, indent=4)
        print(content, file=f)
    
    with open(f"{store_folder}/score.csv", "w", encoding="utf-8") as f:
        for m, s in score:
            print(f"{m} {s}", file=f)


def benchmark(func_name, args_info, vm):
    results = []
    m_limit = 128
    for m in range(8, m_limit + 1, 8):
        hint = {"n": m}
        args = [make_arg(*info, hint) for info in args_info]

        score = vm.time_evaluator(func_name, dev=DEV, number=100, repeat=1, min_repeat_ms=100)(*args).mean
        score_us = int(float(score)*1e6)
        
        results.append((m, score_us))
        print(f"{m} {score_us}")

    return results


def main():
    log_folder = "__tmp/roofline_log"

    impls = [
        ("cublas-f16", "Llama2-7b-chat-hf-q0f16"),
        ("cutlass-q4", "Llama2-7b-chat-hf-q4f16_ft"),
        ("dtune-q4-256-512", "Llama2-7b-chat-hf-q4f16_0"),
        ("dlight-q4-1", "Llama2-7b-chat-hf-q4f16_1"),
    ]

    for impl_name, model_id in impls:
        cache_path = f"dist/{model_id}/mod_cache_before_build.pkl"
        so_file = f"dist/{model_id}/{model_id}-cuda.so"
        assert os.path.isfile(cache_path)
        assert os.path.isfile(so_file)

        with open(cache_path, "rb") as pkl:
            mod = pickle.load(pkl)
        
        funcs_of_mod = [gv.name_hint for gv, _ in mod.functions_items()]

        ex = tvm.runtime.load_module(so_file)
        vm = relax.VirtualMachine(ex, DEV)

        collector = LinearFuncCollector()
        collector.visit_expr(mod["prefill"])
        linears = collector.collection[:-1]

        for i, (name, args_info) in enumerate(linears):
            desc = mod[name] if name in funcs_of_mod else None
            print(f"linear [{i}] {name}")
            score = benchmark(name, args_info, vm)
            
            store_score(log_path=log_folder,
                        workload_name=f"linear{i}",
                        impl_name=impl_name,
                        func_name=name,
                        args_info=args_info,
                        func=desc,
                        score=score)

    plot_folder(log_folder)


if __name__ == "__main__":
    main()
