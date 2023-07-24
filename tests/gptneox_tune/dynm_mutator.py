from typing import Any
from tvm.tir.schedule import Trace, Instruction, InstructionKind, BlockRV
from tvm.script import tir as T
import tvm.runtime

class DynMMutator():
    """ 
    Converter of static Trace to dynamic version 
    
    1. Find GetLoops of base block
    2. Inject PadEinsum
    3. Keep block which implement in and out padding
    4. Keep padded loop
    5. Make split of padded loop with None at the begin
    6. make compute inline for pad blocks

    Example of argument combination:
    q_matmul        : num_of_producers=2, num_of_producers_to_inline=1  A x B, A is padded 
    nt_matmul_1_dyn : num_of_producers=1, num_of_producers_to_inline=1  A x A, A is padded 
    nt_matmul_2_dyn : num_of_producers=2, num_of_producers_to_inline=2  A x B, boath A and B is padded 
    """    
    def __init__(self, base_block_name: str, pad_factors: list[int], *, padded_mask_in=[1, 0], padded_mask_out=[1]) -> None:
        self._base_block_name = base_block_name
        self._pad_factors = pad_factors
        self._padded_mask_in = padded_mask_in
        self._padded_mask_out = padded_mask_out

    def __call__(self, trace: Trace) -> Trace:
        """
        Mutate to Dyn version
        """
        def map_to_attr(attr):
            if isinstance(attr, tvm.runtime.String):
                # Tensor Cores specific (WMMA)
                # Injection of pad changes naming of some shared cache block read and write. 
                attr = str(attr)
                if attr.endswith("_reindex_shared.dyn_wmma.accumulator_o"):
                    attr = attr.replace("_reindex_shared.dyn_wmma.accumulator_o", "_reindex_pad_shared.dyn_wmma.accumulator_o")
                if attr.endswith("_reindex_shared.dyn_wmma.matrix_a_o"):
                    attr = attr.replace("_reindex_shared.dyn_wmma.matrix_a_o", "_reindex_pad_shared.dyn_wmma.matrix_a_o")
            return attr

        rv_map = {}
        def map_to(arg):
            return rv_map[arg] if arg in rv_map else arg
        
        def just_copy(inst):
            return Instruction(
                inst.kind,
                [map_to(i) for i in inst.inputs],
                [map_to_attr(a) for a in inst.attrs],
                [map_to(o) for o in inst.outputs]
            )

        def process_SampleCategorical(inst: Instruction):
            decision = int(trace.decisions[inst])
            val = inst.attrs[0][decision]
            rv_map[inst.outputs[0]] = val
            return []

        rv_padded_loops = []
        rv_to_inline = []
        rv_to_rev_inline = []
        def process_SamplePerfectTile(inst: Instruction):
            decision = [int(des) for des in trace.decisions[inst]]

            # if inst.inputs[0] in rv_padded_loops:
            decision[0] = None
            
            for rv, val in zip(inst.outputs, decision):
                rv_map[rv] = T.int64(val) if val is not None else None

            return []

        rv_base = None
        def process_GetBlock(inst: Instruction):
            nonlocal rv_base
            # Looking for base block
            if rv_base is None and inst.attrs[0] == self._base_block_name:
                rv_base = inst.outputs[0]
            
            return [just_copy(inst)]
        
        def process_GetLoops(inst: Instruction):
            nonlocal rv_base
            # Looking for GetLoops fro base block
            if inst.inputs[0] == rv_base and len(rv_padded_loops) == 0:
                pad = Instruction(
                    InstructionKind.get("PadEinsum"),
                    [rv_base], [[T.int64(val) for val in self._pad_factors]], []
                )
                p_in = Instruction(
                    InstructionKind.get("GetProducers"),
                    [rv_base], [], [BlockRV() for _ in range(len(self._padded_mask_in))]
                )            
                p_out = Instruction(
                    InstructionKind.get("GetConsumers"),
                    [rv_base], [], [BlockRV() for _ in range(len(self._padded_mask_out))]
                )

                # keep blocks to inline
                for idx, flg in enumerate(self._padded_mask_in):
                    if flg == 1:
                        rv_to_inline.append(p_in.outputs[idx])
                for idx, flg in enumerate(self._padded_mask_out):
                    if flg == 1:
                        rv_to_rev_inline.append(p_out.outputs[0])

                get_loop = just_copy(inst)
                # keep padded loop to extend splits
                for idx, pad_factor in enumerate(self._pad_factors):
                    if pad_factor != 1:
                        rv_padded_loops.append(get_loop.outputs[idx])

                return [pad, p_in, p_out, get_loop]
            else:
                return [just_copy(inst)]
        
        def process_EnterPostproc(inst: Instruction):
            inline_instr = []
            for rv in rv_to_inline:
                inline_instr.append(Instruction(
                    InstructionKind.get("ComputeInline"),
                    [rv], [], []
                ))
            for rv in rv_to_rev_inline:
                inline_instr.append(Instruction(
                    InstructionKind.get("ReverseComputeInline"),
                    [rv], [], []
                ))
            return inline_instr + [just_copy(inst)]

        processing_funcs ={
            "SamplePerfectTile": process_SamplePerfectTile,
            "SampleCategorical": process_SampleCategorical,
            "GetBlock": process_GetBlock,
            "GetLoops": process_GetLoops,
            "EnterPostproc": process_EnterPostproc,
        }

        new_insts = []
        for inst in trace.insts:
            if inst.kind.name in processing_funcs:
                for inst_ in processing_funcs[inst.kind.name](inst):
                    new_insts.append(inst_)
            else:
                new_insts.append(just_copy(inst))

        return Trace(new_insts, {})


class AsIs():
    def __init__(self) -> None:
        pass

    def __call__(self, trace: Trace) -> Trace:
        return trace