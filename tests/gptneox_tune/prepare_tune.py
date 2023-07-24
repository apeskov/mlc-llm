import re
from pathlib import Path
from dataclasses import dataclass

import tvm

GPTNEOX_TARGET = tvm.target.Target("nvidia/nvidia-a10g")

def _extract_fuc_src(data: str):
    code_sym_p = r"[\w\"' _,.\-+*/%:;<>=#(){}[\]]"
    code_line_p = rf"^ +({code_sym_p})*\n"
    func_p = rf"^ +@T.prim_func\n({code_line_p})*"

    funcs = {}
    for m in re.finditer(func_p, data, flags=re.MULTILINE):
        func_tir = m.string[m.start():m.end()]
        m = re.search(r"def (\w+)", func_tir)
        name = m.group(1)
        funcs[name] = func_tir

    return funcs


def _convert_to_static(func_src: str, *, val=None):
    """
    Convert prim func souces to static version    
    """
    assert val is not None
    dyn_var_names = []
        
    def _remove_and_keep_name(m):
        dyn_var_names.extend(m.groups())
        return ""
    # find name of double and single dyn var declaration, 
    # like "n, m = T.int64(), T.int64()" and "n = T.int64()"
    func_src = re.sub(r" *(\w*), (\w*) = T\.int64\(\), T\.int64\(\)", _remove_and_keep_name, func_src)
    func_src = re.sub(r" *(\w*) = T\.int64\(\)", _remove_and_keep_name, func_src)
        
    # replace dyn var with value
    for var_name in dyn_var_names:
        func_src = re.sub(rf"([ ,[(]){var_name}([ ,)\]])", rf"\g<1>T.int64({val})\g<2>", func_src)

    return func_src


def _write_to_file(funs: dict, *, file_name=None):
    assert file_name is not None
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    HEAD = """
from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    """ 

    with open(file_name, "w") as of:
        print(HEAD, file=of)
        for name, func in funs.items():
            print(func, file=of)


def main(dyn_tir_file, static_tir_file):
    # Dynamic
    with open(dyn_tir_file, "r") as file:
        funcs = _extract_fuc_src(file.read())
    
    # Just dump as is
    _write_to_file(funcs, file_name="__tmp/tir_dyn.py")
    
    # Convert to static version m=128
    funcs_m128 = {name: _convert_to_static(func, val=128) for name, func in funcs.items()}
    _write_to_file(funcs_m128, file_name="__tmp/tir_dyn_m128.py")

    # Convert to static version m=64
    funcs_m64 = {name: _convert_to_static(func, val=64) for name, func in funcs.items()}
    _write_to_file(funcs_m64, file_name="__tmp/tir_dyn_m64.py")

    # Convert to static version m=32
    funcs_m32 = {name: _convert_to_static(func, val=32) for name, func in funcs.items()}
    _write_to_file(funcs_m32, file_name="__tmp/tir_dyn_m32.py")

    # Convert to static version m=1    
    funcs_m1 = {name: _convert_to_static(func, val=1) for name, func in funcs.items()}
    _write_to_file(funcs_m1, file_name="__tmp/tir_dyn_m1.py")

    # Static 
    with open(static_tir_file, "r") as file:
        funcs = _extract_fuc_src(file.read())

    # Just dump as is
    _write_to_file(funcs, file_name="__tmp/tir_static.py")


# @dataclass
# class LLMTuneContext:    
#     mod: tvm.IRModule
#     mod_fixed_dyn: tvm.IRModule
#     num_dyn_vars: int
#     fixed_dyn_vars: list    # [32, 32]
#     kind: str              # "dynamic", "static"
#     tvm_version: str       # "tvm-main" "relax"
#     converter: Callable[[ms.Trace], ms.Trace]
    

def get_mod_to_tune(kind, *, filter=None):
    if kind == "static":
        from __tmp.tir_static import Module as StaticMod
        mod =  StaticMod
    elif kind == "dynamic":
        from __tmp.tir_dyn import Module as DynMod
        mod = DynMod
    elif kind == "dynamic_m128":
        from __tmp.tir_dyn_m128 import Module as DynM128Mod
        mod = DynM128Mod
    elif kind == "dynamic_m64":
        from __tmp.tir_dyn_m64 import Module as DynM64Mod
        mod = DynM64Mod
    elif kind == "dynamic_m32":
        from __tmp.tir_dyn_m32 import Module as DynM32Mod
        mod = DynM32Mod
    elif kind == "dynamic_m1":
        from __tmp.tir_dyn_m32 import Module as DynM1Mod
        mod = DynM1Mod
    else:
        assert False, "Unknown"

    if filter is not None:
        funcs = {gv.name_hint:mod[gv] for gv in mod.get_global_vars() if filter(gv.name_hint)}
        mod = tvm.IRModule(funcs)

    return mod
        


if __name__ == "__main__":
    dump_path = "../../dist/dolly-v2-12b-q0f16/debug"
    main(dyn_tir_file=f"{dump_path}/mod_tir_dynamic.py", static_tir_file=f"{dump_path}/mod_tir_static.py")
