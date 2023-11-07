# pylint: disable=missing-docstring,invalid-name
import argparse
import functools
import json
import math
import os
import shutil
from typing import Any, Dict, List, Optional, Set

import numpy as np

import tvm
from tvm import relax
from tvm import meta_schedule as ms 
import tvm.tir.tensor_intrin.cuda

from .quantization import quantization_schemes
from .relax_model import param_manager


supported_model_types = set(
    ["llama", "gpt_neox", "gpt_bigcode", "minigpt", "moss", "rwkv", "gptj", "chatglm", "mistral", "stablelm_epoch"]
)


def wrap_tqdm_counter(func, **tqdm_kwargs):
    # tqdm isn't a hard requirement, so return the original function
    # if it isn't available.
    try:
        from tqdm import tqdm
    except ImportError:
        return func

    pbar = tqdm(**tqdm_kwargs)

    @functools.wraps(func)
    def inner(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return inner


def argparse_postproc_common(args: argparse.Namespace) -> None:
    if hasattr(args, "device_name"):
        if args.device_name == "auto":
            if tvm.cuda().exist:
                args.device_name = "cuda"
            elif tvm.metal().exist:
                args.device_name = "metal"
            elif tvm.vulkan().exist:
                args.device_name = "vulkan"
            elif tvm.opencl().exist:
                args.device_name = "opencl"
            else:
                raise ValueError("Cannot auto deduce device-name, please set it")

    model_category_override = {
        "moss-moon-003-sft": "gptj",
        "moss-moon-003-base": "gptj",
        "rwkv-": "rwkv",
        "rwkv_world": "rwkv_world",
        "minigpt": "minigpt",
    }
    try:
        with open(os.path.join(args.model_path, "config.json"), encoding="utf-8") as i_f:
            config = json.load(i_f)
            args.model_category = config["model_type"]
        model_path_lower = args.model_path.lower()
        if "rwkv" in model_path_lower and "world" in model_path_lower:
            args.model_category = "rwkv_world"
    except Exception:
        args.model_category = ""
    model = args.model.lower()
    if "rwkv" in model and "world" in model:
        model = "rwkv_world"
    for prefix, override_category in model_category_override.items():
        if model.startswith(prefix):
            args.model_category = override_category
            break
    assert args.model_category is not None

    model_conv_templates = {
        "llama-2": "llama-2",
        "codellama-7b-instruct": "codellama_instruct",
        "codellama-13b-instruct": "codellama_instruct",
        "codellama-34b-instruct": "codellama_instruct",
        "codellama": "codellama_completion",
        "vicuna-": "vicuna_v1.1",
        "dolly-": "dolly",
        "stablelm-3b-": "stablelm-3b",
        "stablelm-": "stablelm",
        "redpajama-": "redpajama_chat",
        "minigpt": "minigpt",
        "moss-moon-003-sft": "moss",
        "moss-moon-003-base": "LM",
        "gpt-j-": "LM",
        "open_llama": "LM",
        "rwkv-": "rwkv",
        "rwkv_world": "rwkv_world",
        "gorilla-": "gorilla",
        "guanaco": "guanaco",
        "wizardlm-7b": "wizardlm_7b",  # first get rid of 7b
        "wizardlm-": "vicuna_v1.1",  # all others use vicuna template
        "wizardmath-": "wizard_coder_or_math",
        "wizardcoder-": "wizard_coder_or_math",
        "starcoder": "gpt_bigcode",
        "gpt_bigcode-santacoder": "gpt_bigcode",
        "stablecode-completion": "stablecode_completion",
        "stablecode-instruct": "stablecode_instruct",
        "chatglm2": "glm",
        "codegeex2": "glm",
    }

    for prefix, conv_template in model_conv_templates.items():
        if model.startswith(prefix):
            args.conv_template = conv_template
            break
    else:
        args.conv_template = f"{args.model_category}_default"

    if args.quantization not in quantization_schemes:
        raise ValueError(f'Quantization "{args.quantization}" is not supported.')
    args.quantization = quantization_schemes[args.quantization]


def debug_dump_script(mod, name, args: argparse.Namespace, show_meta=True):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    dump_path = os.path.join(args.artifact_path, "debug", name)
    with open(dump_path, "w", encoding="utf-8") as outfile:
        outfile.write(mod.script(show_meta=show_meta))
    print(f"Dump mod to {dump_path}")


def debug_dump_benchmark_script(
    mod: tvm.ir.IRModule,
    name: str,
    args: argparse.Namespace,
) -> None:
    """Extract model level benchmark workloads from relax model."""
    if not args.debug_dump:
        return

    from tvm.dlight.benchmark import (  # pylint: disable=import-error,import-outside-toplevel
        extract_all_func_info_from_relax,
    )

    dump_path = os.path.join(args.artifact_path, "debug", name + ".py")
    with open(dump_path, "w", encoding="utf-8") as outfile:
        outfile.write(
            "# Please save this file to dlight_bench/models and add\n"
            + f"# `from .{name} import *` to dlight_bench/models/__init__.py\n"
            + "from dlight_bench import DlightBench\n"
            + "from tvm.script import tir as T\n\n"
        )

        stmt = []
        try:
            relax_funcs, _ = extract_all_func_info_from_relax(mod)
        except NotImplementedError:
            return
        tvm_script_prefix = "# from tvm.script import tir as T"
        for relax_func_gv in relax_funcs:  # pylint: disable=consider-using-dict-items
            for prim_func_gv in relax_funcs[relax_func_gv]:
                # add global_symbol
                func_body = (
                    mod[prim_func_gv]
                    .with_attr("global_symbol", prim_func_gv.name_hint)
                    .script(name=prim_func_gv.name_hint)
                )
                # remove prefix
                if func_body.startswith(tvm_script_prefix + "\n"):
                    func_body = func_body[len(tvm_script_prefix) :]
                # print out
                outfile.write(func_body + "\n")
                # register
                stmt.append(
                    f"DlightBench.register_bench_workload({prim_func_gv.name_hint}, "
                    f"'{name}', '{prim_func_gv.name_hint}')"
                )
        outfile.write("\n" + "\n".join(stmt) + "\n")
    print(f"Dump benchmarking script to {dump_path}.")


def debug_load_script(name: str, args: argparse.Namespace):
    input_path = os.path.join(args.artifact_path, "debug", name)
    lib = {"__file__": input_path}
    with open(input_path, "rb") as i_f:
        exec(compile(i_f.read(), input_path, "exec"), lib, lib)  # pylint: disable=exec-used
    return lib["Module"]


def debug_dump_shader(ex: tvm.relax.Executable, name: str, args: argparse.Namespace):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    target_kind = args.target.kind.default_keys[0]
    suffix_map = {
        "webgpu": ".wgsl",
        "cuda": ".cu",
        "metal": ".mtl",
        "opencl": ".cl",
    }
    suffix = suffix_map.get(target_kind, ".txt")
    dump_path = os.path.join(args.artifact_path, "debug", name + suffix)
    source = ex.mod.imported_modules[0].imported_modules[0].get_source()
    with open(dump_path, "w", encoding="utf-8") as outfile:
        outfile.write(source)
    print(f"Dump shader to {dump_path}")


def convert_weights(
    mod_transform: tvm.IRModule,
    param_mgr: param_manager.ParamManager,
    model_params: List[Optional[tvm.nd.NDArray]],
    args: argparse.Namespace,
):
    # Save the number of parameters before we lower mod_transform, so
    # we can use them in the progress bar.
    transform_func = mod_transform["transform_params"]
    num_original_params = len(transform_func.params[0].struct_info.fields)
    num_transformed_params = len(transform_func.struct_info.ret.fields)

    # Remove the dataflow block inside the param transform function,
    # so that the LazyTransformParams pass can be applied.
    mod_transform = relax.transform.ToNonDataflow()(mod_transform)
    mod_transform = relax.transform.LazyTransformParams()(mod_transform)
    mod_transform = tvm.tir.transform.ForceNarrowIndexToInt32()(mod_transform)
    mod_transform = relax.transform.LegalizeOps()(mod_transform)

    debug_dump_script(mod_transform, "mod_convert_weights.py", args)

    target = detect_local_target()
    print(f"Automatically using target for weight quantization: {target}")
    device = tvm.device(target.kind.default_keys[0])
    device_cpu = tvm.cpu()

    loaded_params: List[tvm.nd.NDArray] = []
    loaded_idx_set: Set[int] = set()
    loaded_torch_bins: Set[str] = set()
    cached_relax_params: Dict[int, tvm.nd.NDArray] = {}
    cached_torch_params: Dict[str, Any] = {}

    get_item, set_item = param_mgr.get_param_loading_functions(
        model_params,
        loaded_params,
        loaded_idx_set,
        loaded_torch_bins,
        cached_relax_params,
        cached_torch_params,
        device,
        device_cpu,
    )

    get_item = wrap_tqdm_counter(
        get_item, desc="Get old param", position=0, unit="tensors", total=num_original_params
    )
    set_item = wrap_tqdm_counter(
        set_item, desc="Set new param", position=1, unit="tensors", total=num_transformed_params
    )

    tvm.register_func(func_name="get_item", f=get_item, override=True)
    tvm.register_func(func_name="set_item", f=set_item, override=True)

    if target.kind.name != "llvm":
        with tvm.target.Target(target):
            mod_transform = tvm.tir.transform.DefaultGPUSchedule()(mod_transform)

    ex = relax.build(mod_transform, target=target)
    vm = relax.vm.VirtualMachine(ex, device)
    print("Start computing and quantizing weights... This may take a while.")
    vm["transform_params"]()
    print("Finish computing and quantizing weights.")
    return loaded_params


def save_params(params: List[tvm.nd.NDArray], artifact_path: str, num_presharded: int = 1) -> None:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    assert len(params) % num_presharded == 0
    num_weights = len(params) // num_presharded

    meta_data = {}
    param_dict = {}
    meta_data["ParamSize"] = len(params)
    for i, nd in enumerate(params):
        if num_presharded == 1:
            param_name = f"param_{i}"
        else:
            expected_worker_id = i // num_weights
            orig_param_id = i % num_weights
            param_name = f"param_{orig_param_id}_shard-{expected_worker_id+1}-of-{num_presharded}"

        param_dict[param_name] = nd

    total_size_bytes = sum(math.prod(param.shape) * np.dtype(param.dtype).itemsize for param in params)
    total_size_gb = total_size_bytes / (1024 ** 3)
    print(f"Total param size: {total_size_gb} GB")
    tvmjs.dump_ndarray_cache(
        param_dict, f"{artifact_path}/params", meta_data=meta_data, encode_format="raw"
    )


def load_params(artifact_path: str, device) -> List[tvm.nd.NDArray]:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
    plist = []
    size = meta["ParamSize"]
    for i in range(size):
        plist.append(params[f"param_{i}"])
    return plist


def copy_tokenizer(args: argparse.Namespace) -> None:
    for filename in os.listdir(args.model_path):
        if filename in [
            "tokenizer.model",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "tokenizer_config.json",
        ]:
            shutil.copy(
                os.path.join(args.model_path, filename),
                os.path.join(args.artifact_path, "params"),
            )


def get_tokenizer_files(path) -> List[str]:
    tokenizer_set = {
        "tokenizer.model",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    }
    return [x for x in os.listdir(path) if x in tokenizer_set]


def _detect_local_metal_host():
    target_triple = tvm._ffi.get_global_func("tvm.codegen.llvm.GetDefaultTargetTriple")()
    process_triple = tvm._ffi.get_global_func("tvm.codegen.llvm.GetProcessTriple")()
    host_cpu = tvm._ffi.get_global_func("tvm.codegen.llvm.GetHostCPUName")()
    print(
        f"Host CPU dection:\n  Target triple: {target_triple}\n  Process triple: {process_triple}\n  Host CPU: {host_cpu}"
    )
    if target_triple.startswith("x86_64-"):
        return tvm.target.Target(
            {
                "kind": "llvm",
                "mtriple": "x86_64-apple-macos",
                "mcpu": host_cpu,
            }
        )
    # should start with "arm64-"
    return tvm.target.Target(
        {
            "kind": "llvm",
            "mtriple": "arm64-apple-macos",
            "mcpu": host_cpu,
        }
    )


def _detect_local_metal():
    dev = tvm.metal()
    if not dev.exist:
        return None

    return tvm.target.Target(
        {
            "kind": "metal",
            "max_shared_memory_per_block": 32768,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": 32,
        },
        host=_detect_local_metal_host(),
    )


def _detect_local_cuda():
    dev = tvm.cuda()
    if not dev.exist:
        return None
    return tvm.target.Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
            "registers_per_block": 65536,
            "arch": "sm_" + dev.compute_version.replace(".", ""),
        }
    )


def _detect_local_rocm():
    dev = tvm.rocm()
    if not dev.exist:
        return None
    return tvm.target.Target(
        {
            "kind": "rocm",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
        }
    )


def _detect_local_vulkan():
    dev = tvm.vulkan()
    if not dev.exist:
        return None
    return tvm.target.Target(
        {
            "kind": "vulkan",
            "max_threads_per_block": dev.max_threads_per_block,
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "thread_warp_size": dev.warp_size,
            "supports_float16": 1,
            "supports_int16": 1,
            "supports_int8": 1,
            "supports_16bit_buffer": 1,
        }
    )


def _detect_local_opencl():
    dev = tvm.opencl()
    if not dev.exist:
        return None
    return tvm.target.Target("opencl")


def detect_local_target():
    for method in [
        _detect_local_metal,
        _detect_local_rocm,
        _detect_local_cuda,
        _detect_local_vulkan,
        _detect_local_opencl,
    ]:
        target = method()
        if target is not None:
            return target

    print("Failed to detect local GPU, falling back to CPU as a target")
    return tvm.target.Target("llvm")


def parse_target(args: argparse.Namespace) -> None:
    if not hasattr(args, "target"):
        return
    if args.target == "auto":
        target = detect_local_target()
        if target.host is None:
            target = tvm.target.Target(
                target,
                host="llvm",  # TODO: detect host CPU
            )
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "cuda" or args.target == "cuda-multiarch":
        target = _detect_local_cuda()
        if target is None:
            raise ValueError("Cannot detect local CUDA GPU target!")
        multiarch = args.target == "cuda-multiarch"
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
        if multiarch:
            args.target_kind += "-multiarch"
    elif args.target.startswith("nvidia/jetson"):
        try:
            args.target = tvm.target.Target(args.target)
        except ValueError:
            raise ValueError("Cannot find configuration of given nvidia/jetson board target!")
        if not hasattr(args, "cc_path") or args.cc_path == "":
            args.cc_path = "/usr/bin/aarch64-linux-gnu-g++"
        from tvm.contrib.cc import (  # pylint: disable=import-outside-toplevel
            cross_compiler,
        )

        args.export_kwargs = {
            "fcompile": cross_compiler(
                args.cc_path,
            ),
        }
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "metal":
        target = _detect_local_metal()
        if target is None:
            print("Cannot detect local Apple Metal GPU target! Falling back...")
            target = tvm.target.Target(
                tvm.target.Target(
                    {
                        "kind": "metal",
                        "max_threads_per_block": 256,
                        "max_shared_memory_per_block": 32768,
                        "thread_warp_size": 1,
                    }
                ),
                host=_detect_local_metal_host(),
            )
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "metal_x86_64":
        from tvm.contrib import xcode  # pylint: disable=import-outside-toplevel

        args.target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "metal",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                }
            ),
            host="llvm -mtriple=x86_64-apple-darwin",
        )
        args.target_kind = "metal_x86_64"
        args.export_kwargs = {
            "fcompile": xcode.create_dylib,
            "sdk": "macosx",
            "arch": "x86_64",
        }
        args.lib_format = "dylib"
    elif args.target in ["iphone", "iphone-dylib", "iphone-tar"]:
        from tvm.contrib import tar, xcode  # pylint: disable=import-outside-toplevel

        if args.target == "iphone-dylib":
            args.export_kwargs = {
                "fcompile": xcode.create_dylib,
                "sdk": "iphoneos",
                "arch": "arm64",
            }
            args.lib_format = "dylib"
        else:
            args.export_kwargs = {"fcompile": tar.tar}
            args.lib_format = "tar"
            args.system_lib = True
            args.system_lib_prefix = f"{args.model}_{args.quantization}_".replace("-", "_")

        @tvm.register_func("tvm_callback_metal_compile")
        def compile_metal(src, target):
            if target.libs:
                return xcode.compile_metal(src, sdk=target.libs[0])
            return xcode.compile_metal(src)

        target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "metal",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                    "libs": ["iphoneos"],
                }
            ),
            host="llvm -mtriple=arm64-apple-darwin",
        )
        args.target = target
        args.target_kind = "iphone"
    elif args.target == "vulkan":
        target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "vulkan",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                    "supports_float16": 1,
                    "supports_int16": 1,
                    "supports_int8": 1,
                    "supports_8bit_buffer": 1,
                    "supports_16bit_buffer": 1,
                    "supports_storage_buffer_storage_class": 1,
                }
            ),
            host="llvm",
        )
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "opencl":
        target = tvm.target.Target(
            "opencl",
            host="llvm",
        )
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "webgpu":
        args.target = tvm.target.Target(
            "webgpu",
            host="llvm -mtriple=wasm32-unknown-unknown-wasm",
        )
        args.target_kind = "webgpu"
        args.lib_format = "wasm"
        args.system_lib = True
        if os.environ.get("TVM_HOME", "") == "":
            raise RuntimeError(
                "Please set TVM_HOME for webgpu build following scripts/prep_emcc_deps.sh"
            )
    elif args.target in ["android", "android-dylib"]:  # android-opencl
        from tvm.contrib import ndk, tar

        if args.target == "android-dylib":
            args.export_kwargs = {
                "fcompile": ndk.create_shared,
            }
            args.lib_format = "so"
        else:
            args.export_kwargs = {
                "fcompile": tar.tar,
            }
            args.lib_format = "tar"
            args.system_lib = True
            args.system_lib_prefix = f"{args.model}_{args.quantization}_".replace("-", "_")
        args.target = tvm.target.Target(
            "opencl",
            host="llvm -mtriple=aarch64-linux-android",  # TODO: Only support arm64 for now
        )
        args.target_kind = "android"
    elif args.target in ["mali"]:
        from tvm.contrib import ndk

        args.export_kwargs = {
            "fcompile": ndk.create_shared,
        }
        target = tvm.target.Target(
            "opencl -device=mali",
            host="llvm -mtriple=aarch64-linux-gnu",
        )
        args.target = target
        args.target_kind = "mali"
    else:
        args.target = tvm.target.Target(args.target, host="llvm")
        args.target_kind = args.target.kind.default_keys[0]

    if args.target_kind == "cuda-multiarch":
        from tvm.contrib import nvcc

        assert args.target.arch[3:] != ""
        if int(args.target.arch[3:]) >= 70:
            compute_versions = [70, 72, 75, 80, 86, 87, 89, 90]
        else:
            compute_versions = [60, 61, 62]

        args.target_kind = "cuda"

        @tvm.register_func("tvm_callback_cuda_compile", override=True)
        def tvm_callback_cuda_compile(code, target):  # pylint: disable=unused-argument
            """use nvcc to generate fatbin code for better optimization"""
            arch = []
            for compute_version in compute_versions:
                arch += ["-gencode", f"arch=compute_{compute_version},code=sm_{compute_version}"]
            ptx = nvcc.compile_cuda(code, target_format="fatbin", arch=arch)
            return ptx

    # use mingw to cross compile windows
    if hasattr(args, "llvm_mingw") and args.llvm_mingw != "":
        from tvm.contrib.cc import (  # pylint: disable=import-outside-toplevel
            cross_compiler,
        )

        args.export_kwargs = {
            "fcompile": cross_compiler(
                os.path.join(args.llvm_mingw, "bin", "x86_64-w64-mingw32-clang++"),
                output_format="dll",
            ),
        }
        args.target = args.target.with_host("llvm -mtriple=x86_64-w64-windows-gnu")
        args.lib_format = "dll"

    print(f"Target configured: {args.target}")


@ms.utils.derived_object
class MDS1ScheduleRule(ms.schedule_rule.PyScheduleRule):
    def __init__(self) -> None:
        super().__init__()

    def _initialize_with_tune_context(self, context) -> None:
        pass
    
    def is_acceptable(self, sch: tvm.tir.Schedule, block):
        """Check if provided block is gemm
        Trifial implementation. Check blpck name ends with "_matmul"
        Is not correct for general cases. 
        """
        b = sch.get(block)
        return "matmul" in b.name_hint

    def deduce_mpad_value(self, sch: tvm.tir.Schedule):
        """
        Define m pad value will be used in schedule.
        Read proper hint attribute or provide some heuristic value.
        """
        func = sch.mod[sch.func_working_on]
        if "metaschedule.hint.m_pad_value" in func.attrs.keys():
            m_pad_value = func.attrs["metaschedule.hint.m_pad_value"]
            if isinstance(m_pad_value, tvm.tir.IntImm):
                m_pad_value = m_pad_value.value
            return m_pad_value
        
        return 64  # default value 

    def apply(self, sch: tvm.tir.Schedule, block: tvm.tir.schedule.BlockRV):
        if not self.is_acceptable(sch, block):
            return [sch]

        m_pad = self.deduce_mpad_value(sch)
        sch = sch.copy()

        # padding prolog
        # assume order of loops is : B M N K
        sch.pad_einsum(block, padding=[1, m_pad, 16, 16])
        b_pad_a = sch.get_producers(block)[0]
        b_pad_o = sch.get_consumers(block)[0]

        # schedule implement matmul with weight layout [K, N]. Relax use [N, K] by default 
        if "NT_" in sch.get(block).name_hint:
            sch.transform_layout(block=block, buffer=("read", 1), index_map=lambda n, k: (k, n), pad_value=None, assume_injective_transform=True)

        # block 16x16x16
        lb, lm, ln, lk = sch.get_loops(block)
        lm, lm_b = sch.split(lm, factors=[None, 16])
        ln, ln_b = sch.split(ln, factors=[None, 16])
        lk, lk_b = sch.split(lk, factors=[None, 16])
        sch.reorder(lm, ln, lk, lm_b, ln_b, lk_b)
        b_wmma = sch.blockize(lm_b)

        lm_4, lm = sch.split(lm, factors=[None, m_pad//16])
        lm_factors = sch.sample_perfect_tile(loop=lm, n=3, max_innermost_factor=4)
        lm_3, lm_2, lm_1 = sch.split(lm, factors=lm_factors)
        ln_factors = sch.sample_perfect_tile(loop=ln, n=4, max_innermost_factor=4)
        ln_4, ln_3, ln_2, ln_1 = sch.split(ln, factors=ln_factors)
        lk_factors = sch.sample_perfect_tile(loop=lk, n=2, max_innermost_factor=4)
        lk_2, lk_1 = sch.split(lk, factors=lk_factors)
        sch.reorder(lm_4, ln_4, lm_3, ln_3, lm_2, ln_2, lk_2, lk_1, lm_1, ln_1)
        lnm_by = sch.fuse(lm_4, ln_4)
        sch.bind(lnm_by, thread_axis="blockIdx.y")
        lnm_bx = sch.fuse(lm_3, ln_3)
        sch.bind(lnm_bx, thread_axis="blockIdx.x")
        lnm_ty = sch.fuse(lm_2, ln_2)
        sch.bind(lnm_ty, thread_axis="threadIdx.y")


        # copy from/to shared on level of L1 block
        b_o_shared = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="shared.dyn")
        b_o_wmma = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="wmma.accumulator")
        sch.reverse_compute_at(b_o_wmma, loop=lnm_ty, preserve_unit_loops=True, index=-1)
        sch.reverse_compute_at(b_o_shared, loop=lnm_ty, preserve_unit_loops=True, index=-1)
        
        b_a_shared = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="shared.dyn")
        b_a_wmma = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="wmma.matrix_a")
        sch.compute_at(b_a_wmma, loop=lk_1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
        sch.compute_at(b_a_shared, loop=lk_2, preserve_unit_loops=True, index=-1)

        b_b_shared = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="shared.dyn")
        b_b_wmma = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="wmma.matrix_b")
        sch.compute_at(b_b_wmma, loop=lk_1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
        sch.compute_at(b_b_shared, loop=lk_2, preserve_unit_loops=True, index=-1)

        b_wmma_init = sch.decompose_reduction(block=b_wmma, loop=lk_2)

        # tensozise helper
        def blk_tensorize(blk, intrin_name):
            *_, lm, ln = sch.get_loops(blk)
            lm, lm_b = sch.split(lm, factors=[None, 16])
            ln, ln_b = sch.split(ln, factors=[None, 16])
            sch.reorder(lm, ln, lm_b, ln_b)
            blk_16x16 = sch.blockize(lm_b)
            # TODO: add bind to Ty???
            sch.tensorize(blk_16x16, intrin_name)

        # vectorize helper
        def blk_vectorize(blk, vec_size=4, cooperative=True):
            # 16x16 4*32*Ty
            # Ideally it should be 8 (128bit register containd 8 half floats) 
            ty_size = (lm_factors[-2] * ln_factors[-2])  # TODO: error "Stringifying is not supported for type: tir.Mul"
            tx_size = 32
            *_, lm, ln = sch.get_loops(blk) 
            lmn = sch.fuse(lm, ln)
            # lmn, lmn_ty, lmn_tx, lmn_v = sch.split(lmn, factors=[None, ty_size, tx_size, vec_size])
            lmn, lm_ty, ln_ty_2, lmn_tx, lmn_v = sch.split(lmn, factors=[None, lm_factors[-2], ln_factors[-2], tx_size, vec_size])
            sch.bind(lmn_tx, thread_axis="threadIdx.x")
            if cooperative:
                sch.bind(sch.fuse(lm_ty, ln_ty_2), thread_axis="threadIdx.y")
            sch.vectorize(lmn_v)

            # NB! significant impact. Looks like bank conflict. "buffer_index=0" for cache write, is it correct? 
            sch.storage_align(block=blk, buffer_index=0, axis=-2, factor=16, offset=8)   
        
        # tensorize compute
        sch.tensorize(b_wmma, "wmma_sync_16x16x16_f16f16f16")
        sch.tensorize(b_wmma_init, "wmma_fill_16x16x16_f16")

        # tensorize load/store WMMA regs
        blk_tensorize(b_o_wmma, "wmma_store_16x16x16_f16_shared_dyn")
        blk_tensorize(b_a_wmma, "wmma_load_16x16x16_f16_a_shared_dyn")
        blk_tensorize(b_b_wmma, "wmma_load_16x16x16_f16_b_shared_dyn")   # TODO: It accepts "wmma_load_16x16x16_f16_b_trans_shared_dyn" as well.. problem

        # vectorize load/store smem
        blk_vectorize(b_a_shared, vec_size=4)
        blk_vectorize(b_b_shared, vec_size=4)
        blk_vectorize(b_o_shared, vec_size=4, cooperative=False)

        # Padding epilog
        sch.compute_inline(b_pad_a)
        sch.reverse_compute_inline(b_pad_o)

        return [sch]


    def clone(self) -> ms.schedule_rule.ScheduleRule:
        return MDS1ScheduleRule()


def dtune_space_gen():
    # TODO: Deduce this from target
    ms_rule_kind = "cuda-tensorcore"  

    rules = ms.schedule_rule.schedule_rule.create(ms_rule_kind)
    rules = [r for r in rules if not isinstance(r, (ms.schedule_rule.MultiLevelTiling, ms.schedule_rule.MultiLevelTilingTensorCore))]
    rules.insert(1, MDS1ScheduleRule()) 

    postprocs = ms.postproc.Postproc.create(ms_rule_kind)
    postprocs = [p for p in postprocs if not isinstance(p, (ms.postproc.DisallowDynamicLoop, ms.postproc.VerifyGPUCode))]

    return ms.space_generator.PostOrderApply(sch_rules=rules, postprocs=postprocs, mutator_probs=ms_rule_kind)


def dtune_load_db(args):
    if args.tune_db_path == "no":
        return ms.database.MemoryDatabase()

    work_dir = os.path.join(args.artifact_path, "dtune") if args.tune_db_path == "auto" else args.tune_db_path
    
    if not os.path.exists(work_dir):
        return ms.database.MemoryDatabase()
    
    db = ms.database.JSONDatabase(work_dir=work_dir, allow_missing=False)

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
