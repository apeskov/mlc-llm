# pylint: disable=invalid-name,missing-docstring
# Used as reference

import argparse
import json
import os
import time
import random
from typing import List, Tuple

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer, LlamaTokenizer  # type: ignore[import]
from tvm import relax
from tvm.relax.testing.lib_comparator import LibCompareVMInstrument
from tvm.runtime import ShapeTuple

from mlc_llm import utils

prompt_templates = {
    "dolly": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n\n\n"
             "### Instruction:\n{prompt}\n\n"
             "### Response:\n",
    "vicuna": "A chat between a curious user and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
              "USER\n {prompt}\n"
              "ASSISTANT\n",
}


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--seq-len", type=int, default=128)
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
            res = self.mod.time_evaluator(
                name,
                dev=self.device,
                number=100,
                repeat=3,
            )(*new_args).mean
            shapes = [arg.shape for arg in new_args]
            total_bytes = sum(
                arg.numpy().size * arg.numpy().itemsize for arg in new_args
            )
            self.time_eval_results[name] = (res, 1, shapes, total_bytes)
        else:
            record = self.time_eval_results[name]
            self.time_eval_results[name] = (
                record[0],
                record[1] + 1,
                record[2],
                record[3],
            )


def print_as_table(sorted_list: List[Tuple[str, Tuple[float, int]]]):
    print(
        "Name".ljust(50)
        + "Time (ms)".ljust(12)
        + "Count".ljust(8)
        + "Total time (ms)".ljust(18)
        + "Pct (%)".ljust(10)
        + "Memory (MB)".ljust(16)
        + "Bandwidth (GB/s)".ljust(18)
        + "Shape"
    )
    total_time = sum(record[1][0] * record[1][1] for record in sorted_list) * 1000
    for record in sorted_list:
        time_used = record[1][0] * 1000
        weighted_time = time_used * record[1][1]
        percentage = weighted_time / total_time * 100
        total_bytes = record[1][3]
        bandwidth = total_bytes / record[1][0] / (1024**3)

        print(
            record[0].ljust(50)
            + f"{time_used:.4f}".ljust(12)
            + str(record[1][1]).ljust(8)
            + f"{weighted_time:.4f}".ljust(18)
            + f"{percentage:.2f}".ljust(10)
            + f"{total_bytes / (1024 * 1024):.2f}".ljust(16)
            + f"{bandwidth:.4f}".format(bandwidth).ljust(18)
            + ", ".join(str(s) for s in record[1][2])
        )
    print(f"Total time: {total_time:.4f} ms")
    print()


def generate_input_seq(template, tokenizer, seq_len):    
    prompts = [
        "Could you please write the introduction to a steamy romance novel as long as possible. ",
        "It should be love story about boy and girl at wild west at 2042. ",
        "Thanks a lot in advance. ",
        "Thx. "
    ]             

    prompt = ''
    while True:
        prompt_is_updated = False
        for p in prompts: 
            full_p = template.format(prompt=prompt + p)
            new_prompt_size = tokenizer(full_p, return_tensors="np").input_ids.size
            if new_prompt_size <= seq_len:
                prompt = prompt + p
                prompt_is_updated = True
        if not prompt_is_updated:
            break

    full_prompt = tokenizer(template.format(prompt=prompt), return_tensors="pt").input_ids.to(torch.int32)   
    tokens = torch.full((1, 2*seq_len+1), tokenizer.pad_token_type_id).to(torch.int32).to("cuda")

    in_seq_len = full_prompt.shape[1]
    tokens[:,0:in_seq_len] = full_prompt
    
    return tokens


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
    
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.artifact_path, "params"), trust_remote_code=True
    )

    mod_name = args.model.split("-")[0]
    seq_len = args.seq_len

    tokens = generate_input_seq(prompt_templates[mod_name], tokenizer, seq_len)

    print(f"Running inference... seq_len: {seq_len}")
    
    # from torch.profiler import profile, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CUDA], with_modules=False, with_stack=True) as prof:
    if True:
        kv_caches = vm["create_kv_cache"]()
        cur_len = seq_len
        
        in_seq_shape = tvm.runtime.ShapeTuple([cur_len])
        in_seq = tvm.nd.from_dlpack(tokens[:, :cur_len])
        
        start = time.time()        
        logits, kv_caches = vm["prefill"](in_seq, in_seq_shape, kv_caches, const_params)

        # Pytorch implementation of argmax
        logits = torch.from_dlpack(logits)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(torch.int32)
        tokens[:, cur_len] = next_token
        next_token = tvm.nd.from_dlpack(next_token)

        for _ in range(seq_len):
            cur_len += 1
            cur_len_shape = tvm.runtime.ShapeTuple([cur_len])

            logits, kv_caches = vm["decode"](next_token, cur_len_shape, kv_caches, const_params)

            # Pytorch implementation of argmax
            logits = torch.from_dlpack(logits)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(torch.int32)
            tokens[:, cur_len] = next_token
            next_token = tvm.nd.from_dlpack(next_token)

        end = time.time()
        print(f"Time elapsed: {(end - start)} sec")

    if 'prof' in locals(): 
        prof.export_chrome_trace(f"./trace_{mod_name}_tvm.json")

    answer_text = tokenizer.decode(tokens.reshape(-1)[seq_len:2*seq_len+1]) 
    print("Answer:\n\n", answer_text)


if __name__ == "__main__":
    ARGS = _parse_args()
    answer_prompt(ARGS)
