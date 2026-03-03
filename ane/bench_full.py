#!/usr/bin/env python3
"""Full benchmark: ANE vs CPU (Accelerate) vs MPS (Metal GPU) matmul.

Tests at various dimensions to find the ANE crossover point.
"""
import ctypes, sys, os, time
import numpy as np

# Try importing torch for MPS comparison
try:
    import torch
    HAS_TORCH = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False

# --- Load bridge ---
BRIDGE = os.path.join(os.path.dirname(__file__), "libane_bridge.dylib")
lib = ctypes.CDLL(BRIDGE)

lib.ane_bridge_init.restype = ctypes.c_int
lib.ane_bridge_compile.restype = ctypes.c_void_p
lib.ane_bridge_compile.argtypes = [
    ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
]
lib.ane_bridge_eval.restype = ctypes.c_bool
lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
lib.ane_bridge_write_input.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.ane_bridge_read_output.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.ane_bridge_free.argtypes = [ctypes.c_void_p]


def gen_dyn_matmul_mil(ic, oc, seq):
    sp = seq + oc
    MIL_HDR = (
        'program(1.3)\n'
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
        '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
        '{"coremltools-version", "9.0"}})]\n{\n'
    )
    p = "mm"
    lines = [MIL_HDR]
    lines.append(f"    func main<ios18>(tensor<fp32, [1, {ic}, 1, {sp}]> x) {{\n")
    lines.append('        string to16 = const()[name=string("to16"), val=string("fp16")];\n')
    lines.append(f'        tensor<fp16, [1,{ic},1,{sp}]> xh = cast(dtype=to16,x=x)[name=string("cin")];\n')
    lines.append(f'        tensor<int32, [4]> {p}_ba = const()[name=string("{p}_ba"), val=tensor<int32, [4]>([0,0,0,0])];\n')
    lines.append(f'        tensor<int32, [4]> {p}_sa = const()[name=string("{p}_sa"), val=tensor<int32, [4]>([1,{ic},1,{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,{ic},1,{seq}]> {p}_act = slice_by_size(x=xh,begin={p}_ba,size={p}_sa)[name=string("{p}_act")];\n')
    lines.append(f'        tensor<int32, [4]> {p}_bw = const()[name=string("{p}_bw"), val=tensor<int32, [4]>([0,0,0,{seq}])];\n')
    lines.append(f'        tensor<int32, [4]> {p}_sw = const()[name=string("{p}_sw"), val=tensor<int32, [4]>([1,{ic},1,{oc}])];\n')
    lines.append(f'        tensor<fp16, [1,{ic},1,{oc}]> {p}_wt = slice_by_size(x=xh,begin={p}_bw,size={p}_sw)[name=string("{p}_wt")];\n')
    lines.append(f'        tensor<int32, [4]> {p}_ra = const()[name=string("{p}_ra"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,1,{ic},{seq}]> {p}_a2 = reshape(shape={p}_ra,x={p}_act)[name=string("{p}_a2")];\n')
    lines.append(f'        tensor<int32, [4]> {p}_pm = const()[name=string("{p}_pm"), val=tensor<int32, [4]>([0,1,3,2])];\n')
    lines.append(f'        tensor<fp16, [1,1,{seq},{ic}]> {p}_a3 = transpose(perm={p}_pm,x={p}_a2)[name=string("{p}_a3")];\n')
    lines.append(f'        tensor<int32, [4]> {p}_rw = const()[name=string("{p}_rw"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];\n')
    lines.append(f'        tensor<fp16, [1,1,{ic},{oc}]> {p}_W = reshape(shape={p}_rw,x={p}_wt)[name=string("{p}_W")];\n')
    lines.append(f'        bool bF = const()[name=string("bF"), val=bool(false)];\n')
    lines.append(f'        tensor<fp16, [1,1,{seq},{oc}]> {p}_yh = matmul(transpose_x=bF,transpose_y=bF,x={p}_a3,y={p}_W)[name=string("{p}_yh")];\n')
    lines.append(f'        tensor<fp16, [1,1,{oc},{seq}]> {p}_yt = transpose(perm={p}_pm,x={p}_yh)[name=string("{p}_yt")];\n')
    lines.append(f'        tensor<int32, [4]> {p}_ro = const()[name=string("{p}_ro"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,{oc},1,{seq}]> {p}_y = reshape(shape={p}_ro,x={p}_yt)[name=string("{p}_y")];\n')
    lines.append(f'        string to32 = const()[name=string("to32"), val=string("fp32")];\n')
    lines.append(f'        tensor<fp32, [1,{oc},1,{seq}]> y = cast(dtype=to32,x={p}_y)[name=string("cout")];\n')
    lines.append(f'    }} -> (y);\n}}\n')
    return "".join(lines)


def compile_kernel(ic, oc, seq):
    sp = seq + oc
    mil = gen_dyn_matmul_mil(ic, oc, seq)
    mil_bytes = mil.encode("utf-8")
    in_sz = (ctypes.c_size_t * 1)(ic * sp * 4)
    out_sz = (ctypes.c_size_t * 1)(oc * seq * 4)
    kern = lib.ane_bridge_compile(mil_bytes, len(mil_bytes), None, 0, 1, in_sz, 1, out_sz)
    return kern


def bench(ic, oc, seq, n_iters=300, warmup=30):
    sp = seq + oc
    np.random.seed(42)
    act = np.random.randn(ic, seq).astype(np.float32) * 0.01
    W = np.random.randn(ic, oc).astype(np.float32) * 0.01
    inp = np.zeros((ic, sp), dtype=np.float32)
    inp[:, :seq] = act
    inp[:, seq:seq+oc] = W
    inp_flat = inp.flatten()
    in_bytes = ic * sp * 4
    out_bytes = oc * seq * 4
    out = np.zeros(oc * seq, dtype=np.float32)

    # --- ANE ---
    kern = compile_kernel(ic, oc, seq)
    if not kern:
        return None, None, None

    for _ in range(warmup):
        lib.ane_bridge_write_input(kern, 0, inp_flat.ctypes.data, in_bytes)
        lib.ane_bridge_eval(kern)
    t0 = time.time()
    for _ in range(n_iters):
        lib.ane_bridge_write_input(kern, 0, inp_flat.ctypes.data, in_bytes)
        lib.ane_bridge_eval(kern)
        lib.ane_bridge_read_output(kern, 0, out.ctypes.data, out_bytes)
    ane_ms = (time.time() - t0) / n_iters * 1000
    lib.ane_bridge_free(kern)

    # --- CPU (numpy/Accelerate) ---
    act_t = act.T.copy()  # [seq, ic]
    for _ in range(warmup):
        _ = act_t @ W
    t0 = time.time()
    for _ in range(n_iters):
        _ = act_t @ W
    cpu_ms = (time.time() - t0) / n_iters * 1000

    # --- MPS (Metal GPU via PyTorch) ---
    mps_ms = None
    if HAS_TORCH:
        a_mps = torch.from_numpy(act_t).to("mps")
        w_mps = torch.from_numpy(W).to("mps")
        for _ in range(warmup):
            _ = a_mps @ w_mps
            torch.mps.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            _ = a_mps @ w_mps
            torch.mps.synchronize()
        mps_ms = (time.time() - t0) / n_iters * 1000

    return ane_ms, cpu_ms, mps_ms


def main():
    rc = lib.ane_bridge_init()
    assert rc == 0

    print("=" * 80)
    print("ANE vs CPU (Accelerate) vs MPS (Metal GPU) — Matmul Benchmark")
    print(f"MPS available: {HAS_TORCH}")
    print("=" * 80)

    tests = [
        # (name, ic, oc, seq)
        # Felix-LM dimensions
        ("Felix S0: 64→64, seq=512",        64,   64,   512),
        ("Felix S0: 64→256, seq=512",        64,   256,  512),
        ("Felix S1: 128→128, seq=512",       128,  128,  512),
        ("Felix S1: 128→512, seq=512",       128,  512,  512),
        ("Felix S2: 512→128, seq=512",       512,  128,  512),
        # GPT-2 small
        ("GPT2-sm: 768→768, seq=256",        768,  768,  256),
        ("GPT2-sm: 768→3072, seq=256",       768,  3072, 256),
        ("GPT2-sm: 3072→768, seq=256",       3072, 768,  256),
        # GPT-2 medium
        ("GPT2-md: 1024→1024, seq=256",      1024, 1024, 256),
        ("GPT2-md: 1024→4096, seq=256",      1024, 4096, 256),
        # Stories110M (what ANE repo targets)
        ("Stories: 768→768, seq=256",         768,  768,  256),
        ("Stories: 768→2048, seq=256",        768,  2048, 256),
        ("Stories: 2048→768, seq=256",        2048, 768,  256),
    ]

    header = f"{'Operation':<38} {'ANE':>8} {'CPU':>8}"
    if HAS_TORCH:
        header += f" {'MPS':>8} {'ANE/MPS':>8}"
    header += f" {'ANE/CPU':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, ic, oc, seq in tests:
        ane_ms, cpu_ms, mps_ms = bench(ic, oc, seq)
        if ane_ms is None:
            print(f"{name:<38} COMPILE FAILED")
            continue
        line = f"{name:<38} {ane_ms:>7.3f}  {cpu_ms:>7.3f}"
        if HAS_TORCH and mps_ms is not None:
            line += f"  {mps_ms:>7.3f}  {ane_ms/mps_ms:>7.2f}x"
        line += f"  {ane_ms/cpu_ms:>7.2f}x"
        print(line)

    print(f"\nNote: times in milliseconds. ANE/CPU < 1.0 means ANE is faster.")
    print("Done!")


if __name__ == "__main__":
    main()
