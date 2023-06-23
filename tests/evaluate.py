# Used as reference

import argparse
import os
import time
import random
from typing import List, Tuple

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer  # type: ignore[import]
from tvm import relax
from tvm.relax.testing.lib_comparator import LibCompareVMInstrument
from tvm.runtime import ShapeTuple

from mlc_llm import utils


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--prompt", type=str, default="The capital of Canada is")
    args.add_argument("--profile", action="store_true", default=False)
    parsed = args.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )
    return parsed


class LibCompare(LibCompareVMInstrument):
    def __init__(self, mod, device):
        super().__init__(mod, device, verbose=False)
        self.time_eval_results = {}

    def compare(
        self,
        name: str,
        ref_args: List[tvm.nd.NDArray],
        new_args: List[tvm.nd.NDArray],
        ret_indices: List[int],
    ):
        if name.startswith("shape_func"):
            return
        if name not in self.time_eval_results:
            super().compare(name, ref_args, new_args, ret_indices)
            res = self.mod.time_evaluator(name, dev=self.device)(*new_args).mean
            self.time_eval_results[name] = (res, 1)
        else:
            record = self.time_eval_results[name]
            self.time_eval_results[name] = (record[0], record[1] + 1)


def print_as_table(sorted_list: List[Tuple[str, Tuple[float, int]]]):
    print(
        "Name".ljust(50)
        + "Time (ms)".ljust(12)
        + "Count".ljust(8)
        + "Total time (ms)".ljust(18)
        + "Percentage (%)"
    )
    total_time = sum([record[1][0] * record[1][1] for record in sorted_list]) * 1000
    for record in sorted_list:
        time = record[1][0] * 1000
        weighted_time = time * record[1][1]
        percentage = weighted_time / total_time * 100
        print(
            record[0].ljust(50)
            + "{:.4f}".format(time).ljust(12)
            + str(record[1][1]).ljust(8)
            + "{:.4f}".format(weighted_time).ljust(18)
            + "{:.2f}".format(percentage)
        )
    print("Total time: {:.4f} ms".format(total_time))
    print()


def answer_prompt(args) -> None:
    device = tvm.device(args.device_name)
    const_params = utils.load_params(args.artifact_path, device)
    ex = tvm.runtime.load_module(
        os.path.join(
            args.artifact_path,
            f"{args.model}-{args.quantization.name}-{args.device_name}.so",
        )
    )
    vm = relax.VirtualMachine(ex, device)
    
    top_p = 0.95
    temperature = 0.8
    
    # prompt = args.prompt
    # prompt = "2 plus 2"
    # prompt = "Write the introduction to a steamy romance novel"
    # prompt = "Lovely poema with flowers"
    prompt = "What is the capital of Russia?"

    full_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n\n\n" \
                  f"### Instruction:\n{prompt}\n\n" \
                  f"### Response:\n"

    temperature_arr = tvm.nd.empty({}, dtype="float32", device=device)
    temperature_arr.copyfrom(np.array(temperature, dtype="float32"))

    sample_top_p_from_prob = tvm.get_global_func("vm.builtin.sample_top_p_from_prob")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.artifact_path, "params"), trust_remote_code=True
    )

    print("Tokenizing...")
    inputs_np = tokenizer(full_prompt, return_tensors="pt").input_ids.to(torch.int32).numpy()
    inputs = tvm.nd.array(inputs_np, device)

    first_sampled_token = tvm.nd.array(np.array([[6234]]).astype("int32"), device)
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    
    # Warmup
    kv_caches = vm["create_kv_cache"]()
    logits, kv_caches = vm["prefill"](inputs, seq_len_shape, kv_caches, const_params)
    logits, kv_caches = vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )

    # Inference
    next_token_nd = tvm.nd.array(np.array([[6234]]).astype("int32"), device)
    kv_caches = vm["create_kv_cache"]()
    cur_seq_len = inputs.shape[1]
    answer_tokens = []
    
    print("Running inference...")
    
    # from torch.profiler import profile, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CUDA], with_modules=False, with_stack=True) as prof:

    logits, kv_caches = vm["prefill"](inputs, seq_len_shape, kv_caches, const_params)
    prob = vm["softmax_with_temperature"](logits, temperature_arr)
    next_token = sample_top_p_from_prob(prob, top_p, random.uniform(0, 1))
    
    answer_tokens.append(next_token)
    
    for _ in range(100):
        next_token_nd.copyfrom(np.array([[next_token]], dtype="float32"))
        cur_seq_len += 1
        cur_seq_len_shape = tvm.runtime.ShapeTuple([cur_seq_len])
        
        second_seq_len_shape
        logits, kv_caches = vm["decode"](
            next_token_nd, cur_seq_len_shape, kv_caches, const_params
        )
        prob = vm["softmax_with_temperature"](logits, temperature_arr)
        next_token = sample_top_p_from_prob(prob, top_p, random.uniform(0, 1))
        next_token_text = tokenizer.decode(next_token) 

        if next_token == 2 or next_token_text == "### End":
            break
        else: 
            answer_tokens.append(next_token)
    
    # prof.export_chrome_trace("./trace_llm_tvm.json")

    answer_text = tokenizer.decode(answer_tokens) 
    print("Question: ", prompt)
    print("Answer: ", answer_text)


def deploy_to_pipeline(args) -> None:
    device = tvm.device(args.device_name)
    const_params = utils.load_params(args.artifact_path, device)
    ex = tvm.runtime.load_module(
        os.path.join(
            args.artifact_path,
            f"{args.model}-{args.quantization.name}-{args.device_name}.so",
        )
    )
    vm = relax.VirtualMachine(ex, device)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.artifact_path, "params"), trust_remote_code=True
    )

    print("Tokenizing...")
    inputs = tvm.nd.array(
        tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy(),
        device,
    )
    first_sampled_token = tvm.nd.array(np.array([[6234]]).astype("int32"), device)
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    kv_caches = vm["create_kv_cache"]()
    # skip warm up

    logits, kv_caches = vm["prefill"](inputs, seq_len_shape, kv_caches, const_params)
    logits, kv_caches = vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )
    device.sync()

    kv_caches = vm["create_kv_cache"]()
    print("Running inference...")
    start = time.time()
    logits, kv_caches = vm["prefill"](inputs, seq_len_shape, kv_caches, const_params)
    device.sync()
    encoding_end = time.time()
    logits, kv_caches = vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )
    device.sync()
    end = time.time()
    fcache_view = tvm.get_global_func("vm.builtin.attention_kv_cache_view")
    first_k_cache = fcache_view(kv_caches[0], ShapeTuple([7, 32, 128]))
    if args.debug_dump:
        print(f"output kv_cache[0]:\n{first_k_cache.numpy().transpose(1, 0, 2)}")
        print(f"output logits:\n{logits.numpy()}")
    print(
        f"Time elapsed: encoding {(encoding_end - start)} seconds, decoding {end - encoding_end} secs"
    )

    if args.profile:
        cmp_instrument = LibCompare(ex, device)
        vm.set_instrument(cmp_instrument)

        print("Profiling...")
        kv_caches = vm["create_kv_cache"]()

        logits, kv_caches = vm["prefill"](
            inputs, seq_len_shape, kv_caches, const_params
        )
        print("======================= Encoding Profiling =======================")
        print_as_table(
            sorted(
                cmp_instrument.time_eval_results.items(),
                key=lambda x: -(x[1][0] * x[1][1]),
            )
        )
        cmp_instrument.time_eval_results.clear()

        logits, kv_caches = vm["decode"](
            first_sampled_token, second_seq_len_shape, kv_caches, const_params
        )
        print("======================= Decoding Profiling =======================")
        print_as_table(
            sorted(
                cmp_instrument.time_eval_results.items(),
                key=lambda x: -(x[1][0] * x[1][1]),
            )
        )


if __name__ == "__main__":
    ARGS = _parse_args()
    # deploy_to_pipeline(ARGS)
    answer_prompt(ARGS)
