from tvm.script import ir as I
from tvm.script import tir as T
from tvm import relax


@I.ir_module
class ImplCollection_A100:
    @T.prim_func
    def matmul_1_15360_5120_40(A: T.Buffer((T.int64(1), T.int64(5120)), "float16"), B_pack: T.Buffer((T.int64(640), T.int64(15360)), "int32"), scales: T.Buffer((T.int64(40), T.int64(15360)), "float16"), zeros_pack: T.Buffer((T.int64(40), T.int64(1920)), "int32"), C: T.Buffer((T.int64(1), T.int64(15360)), "float16")):
        T.func_attr({"global_symbol": "matmul_1_15360_5120_40", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(960), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(960), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="wmma.accumulator")
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(5120), T.int64(15360)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(5120), T.int64(15360)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(T.int64(1), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(960), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(960), ax0_0_1_ax1_0_1_fused + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(40)):
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                                        with T.block("A_reindex_shared.dyn"):
                                            v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(128))
                                            v1 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(128))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_shared_dyn[v0, v1] = T.if_then_else(v0 < T.int64(1), A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(64)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(16))
                                        v1 = T.axis.spatial(T.int64(15360), ax0_0_1_ax1_0_1_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(16))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(4)):
                                with T.block("A_reindex_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0)
                                    v1_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(4) + ax1_0)
                                    T.reads(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(4) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(960), ax0_0_1_ax1_0_1_fused + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(960), ax0_0_1_ax1_0_1_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(4) + ax2_0_2)
                                    T.reads(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16))
                for ax2 in range(T.int64(1)):
                    for ax0_ax1_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(T.int64(1), T.int64(1)):
                            with T.block("C_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o, v2_o, v3_o = T.axis.remap("SSS", [ax0_0_1_ax1_0_1_fused, ax2_1, ax3])
                                v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                T.writes(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                A_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(16), 2), C_1.strides[0], "row_major")
                    for ax0_ax1_ax3_ax4_ax5_fused_0 in range(T.int64(8)):
                        for ax0_ax1_ax3_ax4_ax5_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                            for ax0_ax1_ax3_ax4_ax5_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                with T.block("C_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1, v2 = T.axis.remap("SS", [ax0_0_1_ax1_0_1_fused, ax2])
                                    v3 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v4 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) // T.int64(16))
                                    v5 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) % T.int64(16))
                                    T.where(((ax0_ax1_ax3_ax4_ax5_fused_0 + ax0_ax1_ax3_ax4_ax5_fused_1) * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) % T.int64(256) // T.int64(16) < T.int64(1))
                                    T.reads(C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5])
                                    T.writes(C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)])
                                    C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)] = C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5]


    @T.prim_func
    def matmul_1_5120_5120_40(A: T.Buffer((T.int64(1), T.int64(5120)), "float16"), B_pack: T.Buffer((T.int64(640), T.int64(5120)), "int32"), scales: T.Buffer((T.int64(40), T.int64(5120)), "float16"), zeros_pack: T.Buffer((T.int64(40), T.int64(640)), "int32"), C: T.Buffer((T.int64(1), T.int64(5120)), "float16")):
        T.func_attr({"global_symbol": "matmul_1_5120_5120_40", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="wmma.accumulator")
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(5120), T.int64(5120)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(5120), T.int64(5120)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(T.int64(2), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(1024), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(40), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(4) + ax0_0_2_ax1_0_2_fused + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(40)):
                        for ax0_ax1_fused_0 in range(T.int64(2)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                                        with T.block("A_reindex_shared.dyn"):
                                            v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(128))
                                            v1 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(128))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_shared_dyn[v0, v1] = T.if_then_else(v0 < T.int64(1), A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(64)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(64))
                                        v1 = T.axis.spatial(T.int64(5120), ax0_0_0_ax1_0_0_fused * T.int64(2560) + ax0_0_1_ax1_0_1_fused * T.int64(64) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(64))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(4)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(2)):
                                with T.block("A_reindex_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0)
                                    v1_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(2) + ax1_0)
                                    T.reads(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(2) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(4) + ax0_0_2_ax1_0_2_fused + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(4) + ax0_0_2_ax1_0_2_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(2) + ax2_0_2)
                                    T.reads(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16))
                for ax2 in range(T.int64(1)):
                    for ax0_ax1_fused in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(T.int64(1), T.int64(1)):
                            with T.block("C_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(4) + ax0_ax1_fused)
                                v2_o, v3_o = T.axis.remap("SS", [ax2_1, ax3])
                                v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                T.writes(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                A_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(16), 2), C_1.strides[0], "row_major")
                    for ax0_ax1_ax3_ax4_ax5_fused_0 in range(T.int64(1)):
                        for ax0_ax1_ax3_ax4_ax5_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                            for ax0_ax1_ax3_ax4_ax5_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax3_ax4_ax5_fused_3 in T.vectorized(T.int64(8)):
                                    with T.block("C_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(4) + (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(1024) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(256) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(8) + ax0_ax1_ax3_ax4_ax5_fused_3) // T.int64(256))
                                        v2 = T.axis.spatial(T.int64(1), ax2)
                                        v3 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v4 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(1024) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(256) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(8) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(256) // T.int64(16))
                                        v5 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(1024) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(256) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(8) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(16))
                                        T.where((((ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_1) * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) * T.int64(8) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(256) // T.int64(16) < T.int64(1))
                                        T.reads(C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5])
                                        T.writes(C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)])
                                        C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)] = C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5]


    @T.prim_func
    def matmul_1_20480_5120_40(A: T.Buffer((T.int64(1), T.int64(5120)), "float16"), B_pack: T.Buffer((T.int64(640), T.int64(20480)), "int32"), scales: T.Buffer((T.int64(40), T.int64(20480)), "float16"), zeros_pack: T.Buffer((T.int64(40), T.int64(2560)), "int32"), C: T.Buffer((T.int64(1), T.int64(20480)), "float16")):
        T.func_attr({"global_symbol": "matmul_1_20480_5120_40", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="wmma.accumulator")
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(5120), T.int64(20480)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(5120), T.int64(20480)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(T.int64(1), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(1280), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(1280), ax0_0_1_ax1_0_1_fused + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(80)):
                        for ax0_ax1_fused_0 in range(T.int64(16)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                                        with T.block("A_reindex_shared.dyn"):
                                            v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(64) + ax0_ax1_fused_1 * T.int64(64) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(64))
                                            v1 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(64) + ax0_ax1_fused_1 * T.int64(64) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(64))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_shared_dyn[v0, v1] = T.if_then_else(v0 < T.int64(1), A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(32)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(16))
                                        v1 = T.axis.spatial(T.int64(20480), ax0_0_1_ax1_0_1_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(16))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(2)):
                                with T.block("A_reindex_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0)
                                    v1_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(4) + ax2_0_1 * T.int64(2) + ax1_0)
                                    T.reads(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(4) + ax2_0_1 * T.int64(2) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(1280), ax0_0_1_ax1_0_1_fused + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(1280), ax0_0_1_ax1_0_1_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(320), ax2_0_0 * T.int64(4) + ax2_0_1 * T.int64(2) + ax2_0_2)
                                    T.reads(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16))
                for ax2 in range(T.int64(1)):
                    for ax0_ax1_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(T.int64(1), T.int64(1)):
                            with T.block("C_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o, v2_o, v3_o = T.axis.remap("SSS", [ax0_0_1_ax1_0_1_fused, ax2_1, ax3])
                                v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                T.writes(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                A_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(16), 2), C_1.strides[0], "row_major")
                    for ax0_ax1_ax3_ax4_ax5_fused_0 in range(T.int64(2)):
                        for ax0_ax1_ax3_ax4_ax5_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                            for ax0_ax1_ax3_ax4_ax5_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax3_ax4_ax5_fused_3 in T.vectorized(T.int64(4)):
                                    with T.block("C_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1, v2 = T.axis.remap("SS", [ax0_0_1_ax1_0_1_fused, ax2])
                                        v3 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v4 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) // T.int64(16))
                                        v5 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(16))
                                        T.where((((ax0_ax1_ax3_ax4_ax5_fused_0 + ax0_ax1_ax3_ax4_ax5_fused_1) * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(256) // T.int64(16) < T.int64(1))
                                        T.reads(C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5])
                                        T.writes(C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)])
                                        C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)] = C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5]


    @T.prim_func
    def matmul_1_5120_20480_160(A: T.Buffer((T.int64(1), T.int64(20480)), "float16"), B_pack: T.Buffer((T.int64(2560), T.int64(5120)), "int32"), scales: T.Buffer((T.int64(160), T.int64(5120)), "float16"), zeros_pack: T.Buffer((T.int64(160), T.int64(640)), "int32"), C: T.Buffer((T.int64(1), T.int64(5120)), "float16")):
        T.func_attr({"global_symbol": "matmul_1_5120_20480_160", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="wmma.accumulator")
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(16), T.int64(20480)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(20480), T.int64(5120)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(16), T.int64(20480)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(20480), T.int64(5120)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(T.int64(320), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(80)):
                        for ax0_ax1_fused_0 in range(T.int64(16)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                                        with T.block("A_reindex_shared.dyn"):
                                            v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(256))
                                            v1 = T.axis.spatial(T.int64(20480), ax2_0_0 * T.int64(256) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(256))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_shared_dyn[v0, v1] = T.if_then_else(v0 < T.int64(1), A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(128)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(20480), ax2_0_0 * T.int64(256) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(16))
                                        v1 = T.axis.spatial(T.int64(5120), ax0_0_0_ax1_0_0_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(16))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(8)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(2)):
                                with T.block("A_reindex_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0)
                                    v1_o = T.axis.spatial(T.int64(1280), ax2_0_0 * T.int64(16) + ax2_0_1 * T.int64(2) + ax1_0)
                                    T.reads(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(1280), ax2_0_0 * T.int64(16) + ax2_0_1 * T.int64(2) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(1280), ax2_0_0 * T.int64(16) + ax2_0_1 * T.int64(2) + ax2_0_2)
                                    T.reads(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16))
                for ax2 in range(T.int64(1)):
                    for ax0_ax1_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(T.int64(1), T.int64(1)):
                            with T.block("C_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o, v2_o, v3_o = T.axis.remap("SSS", [ax0_0_0_ax1_0_0_fused, ax2_1, ax3])
                                v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                T.writes(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                A_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(16), 2), C_1.strides[0], "row_major")
                    for ax0_ax1_ax3_ax4_ax5_fused_0 in range(T.int64(2)):
                        for ax0_ax1_ax3_ax4_ax5_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                            for ax0_ax1_ax3_ax4_ax5_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax3_ax4_ax5_fused_3 in T.vectorized(T.int64(4)):
                                    with T.block("C_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1, v2 = T.axis.remap("SS", [ax0_0_0_ax1_0_0_fused, ax2])
                                        v3 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v4 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) // T.int64(16))
                                        v5 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(16))
                                        T.where((((ax0_ax1_ax3_ax4_ax5_fused_0 + ax0_ax1_ax3_ax4_ax5_fused_1) * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(256) // T.int64(16) < T.int64(1))
                                        T.reads(C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5])
                                        T.writes(C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)])
                                        C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)] = C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5]



def _find_impl(M, N, K, G):
    impl_provider = ImplCollection_A100
    
    if M is not None:
        res = impl_provider[f"matmul_{M}_{N}_{K}_{G}"]
    else:
        res = get_f16u4f16_matmul(N, K, G)  # unoptimized stub 
    
    return res


def get_f16u4f16_matmul(N, K, G):
    assert K % 8 == 0,  "must be packable as a whole byte"

    # We need to calculate the group size based on the fact ceil(K / group_size) == G
    # we assume K is divisible by G, otherwise we need to
    # pass in group_size explicitly
    assert K % G == 0
    group_size = K // G

    func_name = f"matmul_dynm_{N}_{K}_{G}"

    @T.prim_func
    def matmul(
            a: T.handle,
            B_pack: T.Buffer((T.int64(K//8), T.int64(N)), "int32"),
            scales: T.Buffer((T.int64(G), T.int64(N)), "float16"),
            zeros_pack: T.Buffer((T.int64(G), T.int64(N//8),), "int32"),
            c: T.handle
    ):
        m = T.int64()
        A = T.match_buffer(a, (m, T.int64(K)), "float16")
        C = T.match_buffer(c, (m, T.int64(N)), "float16")
        B = T.alloc_buffer((T.int64(K), T.int64(N)), dtype="float16")

        T.func_attr({"global_symbol": func_name, "tir.noalias": True})

        zeros = T.alloc_buffer((T.int64(G), T.int64(N)), dtype="int32")
        for g, n in T.grid(T.int64(G), T.int64(N)):
            with T.block("zeros_decode"):
                vg, vn = T.axis.remap("SS", [g, n])
                zeros[vg, vn] = (zeros_pack[vg, vn // 8] >> (vn % 8 * 4) & 0xF) + T.int32(1)

        for k, n in T.grid(T.int64(K), T.int64(N)):
            with T.block("B_decode"):
                vk, vn= T.axis.remap("SS", [k, n])
                B[vk, vn] = T.cast((B_pack[vk // 8, vn] >> (vk % 8 * 4) & 0xF) - zeros[vk // group_size, vn], "float16") * scales[vk // group_size, vn]

        for i, j, k in T.grid(m, T.int64(N), T.int64(K)):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] += A[vi, vk] * B[vk, vj]

    return matmul


def fill_with_impls(bb: relax.BlockBuilder, config):
    for N, K, G in config:
        # static version
        func = _find_impl(1, N, K, G)
        bb.add_func(func, f"q_matmul_1_{N}_{K}_{G}")
        # dynamic M version
        func = _find_impl(None, N, K, G)
        bb.add_func(func, f"q_matmul_m_{N}_{K}_{G}")