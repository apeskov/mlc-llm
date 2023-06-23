from tvm.script import tir as T
from tvm import relax


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


def get_f16f16f16_matmul(N, K, G):
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
            B: T.Buffer((T.int64(K), T.int64(N)), "float16"),
            scales: T.Buffer((T.int64(G), T.int64(N)), "float16"),
            zeros_pack: T.Buffer((T.int64(G), T.int64(N//8),), "int32"),
            c: T.handle
    ):
        m = T.int64()
        A = T.match_buffer(a, (m, T.int64(K)), "float16")
        C = T.match_buffer(c, (m, T.int64(N)), "float16")

        T.func_attr({"global_symbol": func_name, "tir.noalias": True})

        for i, j, k in T.grid(m, T.int64(N), T.int64(K)):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] += A[vi, vk] * B[vk, vj]

    return matmul


def fill_with_impls(bb: relax.BlockBuilder, config):
    for N, K, G in config:
        func = get_f16u4f16_matmul(N, K, G)
        # func = get_f16f16f16_matmul(N, K, G)
        bb.add_func(func, f"q_matmul_{N}_{K}_{G}")
