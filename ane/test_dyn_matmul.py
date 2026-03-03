#!/usr/bin/env python3
"""Test ANE bridge: compile + eval a dynamic matmul kernel.

Generates MIL matching gen_dyn_matmul_mil() from mil_dynamic.h exactly.
Input:  [1, IC, 1, SEQ+OC] fp32  — act[0:SEQ], W[SEQ:SEQ+OC]
Output: [1, OC, 1, SEQ] fp32     — act @ W
"""
import ctypes, sys, os, time
import numpy as np

# --- Config ---
IC = 128   # input channels (like DIM in felix)
OC = 128   # output channels
SEQ = 64   # sequence length (small for testing)

SP = SEQ + OC  # spatial dimension of input

# --- Load bridge ---
BRIDGE = os.path.join(os.path.dirname(__file__), "libane_bridge.dylib")
lib = ctypes.CDLL(BRIDGE)

# Declare function signatures
lib.ane_bridge_init.restype = ctypes.c_int
lib.ane_bridge_compile.restype = ctypes.c_void_p
lib.ane_bridge_compile.argtypes = [
    ctypes.c_char_p, ctypes.c_size_t,     # mil_text, mil_len
    ctypes.c_void_p, ctypes.c_size_t,     # weight_data, weight_len (NULL/0 for dynamic)
    ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),  # n_inputs, input_sizes
    ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),  # n_outputs, output_sizes
]
lib.ane_bridge_eval.restype = ctypes.c_bool
lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
lib.ane_bridge_write_input.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
]
lib.ane_bridge_read_output.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
]
lib.ane_bridge_free.argtypes = [ctypes.c_void_p]


def gen_dyn_matmul_mil(ic, oc, seq):
    """Generate MIL text exactly matching gen_dyn_matmul_mil() from mil_dynamic.h"""
    sp = seq + oc

    MIL_HDR = (
        'program(1.3)\n'
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
        '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
        '{"coremltools-version", "9.0"}})]\n{\n'
    )

    lines = [MIL_HDR]
    lines.append(f"    func main<ios18>(tensor<fp32, [1, {ic}, 1, {sp}]> x) {{\n")
    lines.append('        string to16 = const()[name=string("to16"), val=string("fp16")];\n')
    lines.append(f'        tensor<fp16, [1,{ic},1,{sp}]> xh = cast(dtype=to16,x=x)[name=string("cin")];\n')

    # Inline gen_dyn_matmul with prefix="mm", act_sp_off=0, w_sp_off=seq, input_var="xh"
    prefix = "mm"
    act_sp_off = 0
    w_sp_off = seq
    input_var = "xh"

    # Slice activations
    lines.append(f'        tensor<int32, [4]> {prefix}_ba = const()[name=string("{prefix}_ba"), val=tensor<int32, [4]>([0,0,0,{act_sp_off}])];\n')
    lines.append(f'        tensor<int32, [4]> {prefix}_sa = const()[name=string("{prefix}_sa"), val=tensor<int32, [4]>([1,{ic},1,{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,{ic},1,{seq}]> {prefix}_act = slice_by_size(x={input_var},begin={prefix}_ba,size={prefix}_sa)[name=string("{prefix}_act")];\n')

    # Slice weight
    lines.append(f'        tensor<int32, [4]> {prefix}_bw = const()[name=string("{prefix}_bw"), val=tensor<int32, [4]>([0,0,0,{w_sp_off}])];\n')
    lines.append(f'        tensor<int32, [4]> {prefix}_sw = const()[name=string("{prefix}_sw"), val=tensor<int32, [4]>([1,{ic},1,{oc}])];\n')
    lines.append(f'        tensor<fp16, [1,{ic},1,{oc}]> {prefix}_wt = slice_by_size(x={input_var},begin={prefix}_bw,size={prefix}_sw)[name=string("{prefix}_wt")];\n')

    # Reshape act: [1,ic,1,seq] → [1,1,ic,seq] → transpose → [1,1,seq,ic]
    lines.append(f'        tensor<int32, [4]> {prefix}_ra = const()[name=string("{prefix}_ra"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,1,{ic},{seq}]> {prefix}_a2 = reshape(shape={prefix}_ra,x={prefix}_act)[name=string("{prefix}_a2")];\n')
    lines.append(f'        tensor<int32, [4]> {prefix}_pm = const()[name=string("{prefix}_pm"), val=tensor<int32, [4]>([0,1,3,2])];\n')
    lines.append(f'        tensor<fp16, [1,1,{seq},{ic}]> {prefix}_a3 = transpose(perm={prefix}_pm,x={prefix}_a2)[name=string("{prefix}_a3")];\n')

    # Reshape weight: [1,ic,1,oc] → [1,1,ic,oc]
    lines.append(f'        tensor<int32, [4]> {prefix}_rw = const()[name=string("{prefix}_rw"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];\n')
    lines.append(f'        tensor<fp16, [1,1,{ic},{oc}]> {prefix}_W = reshape(shape={prefix}_rw,x={prefix}_wt)[name=string("{prefix}_W")];\n')

    # matmul: [1,1,seq,ic] @ [1,1,ic,oc] → [1,1,seq,oc]
    lines.append(f'        bool bF = const()[name=string("bF"), val=bool(false)];\n')
    lines.append(f'        tensor<fp16, [1,1,{seq},{oc}]> {prefix}_yh = matmul(transpose_x=bF,transpose_y=bF,x={prefix}_a3,y={prefix}_W)[name=string("{prefix}_yh")];\n')

    # Transpose back + reshape: [1,1,seq,oc] → [1,1,oc,seq] → [1,oc,1,seq]
    lines.append(f'        tensor<fp16, [1,1,{oc},{seq}]> {prefix}_yt = transpose(perm={prefix}_pm,x={prefix}_yh)[name=string("{prefix}_yt")];\n')
    lines.append(f'        tensor<int32, [4]> {prefix}_ro = const()[name=string("{prefix}_ro"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,{oc},1,{seq}]> {prefix}_y = reshape(shape={prefix}_ro,x={prefix}_yt)[name=string("{prefix}_y")];\n')

    # Cast back to fp32
    lines.append(f'        string to32 = const()[name=string("to32"), val=string("fp32")];\n')
    lines.append(f'        tensor<fp32, [1,{oc},1,{seq}]> y = cast(dtype=to32,x=mm_y)[name=string("cout")];\n')
    lines.append(f'    }} -> (y);\n}}\n')

    return "".join(lines)


def main():
    print(f"=== ANE Dynamic Matmul Test ===")
    print(f"IC={IC}, OC={OC}, SEQ={SEQ}, SP={SP}")

    # Init bridge
    rc = lib.ane_bridge_init()
    if rc != 0:
        print(f"FAIL: ane_bridge_init returned {rc}")
        sys.exit(1)
    print("ane_bridge_init: OK")

    # Generate MIL
    mil = gen_dyn_matmul_mil(IC, OC, SEQ)
    mil_bytes = mil.encode("utf-8")

    print(f"\nMIL text ({len(mil_bytes)} bytes):")
    # Print first few lines for debugging
    for i, line in enumerate(mil.split("\n")[:5]):
        print(f"  {line}")
    print(f"  ... ({len(mil.split(chr(10)))} lines total)")

    # Input: [1, IC, 1, SP] fp32 = IC * SP * 4 bytes
    # Output: [1, OC, 1, SEQ] fp32 = OC * SEQ * 4 bytes
    input_bytes = IC * SP * 4
    output_bytes = OC * SEQ * 4
    print(f"\nInput size: {input_bytes} bytes, Output size: {output_bytes} bytes")

    # Compile (no weight blob for dynamic matmul)
    input_sizes = (ctypes.c_size_t * 1)(input_bytes)
    output_sizes = (ctypes.c_size_t * 1)(output_bytes)

    print("Compiling...")
    t0 = time.time()
    kern = lib.ane_bridge_compile(
        mil_bytes, len(mil_bytes),
        None, 0,  # no weights
        1, input_sizes,
        1, output_sizes
    )
    t1 = time.time()

    if not kern:
        print(f"FAIL: compile returned NULL ({t1-t0:.3f}s)")
        # Write MIL to file for debugging
        with open("/tmp/failed_mil.txt", "w") as f:
            f.write(mil)
        print("MIL written to /tmp/failed_mil.txt")
        sys.exit(1)

    print(f"Compile + load: OK ({t1-t0:.3f}s)")

    # Prepare test data
    # act: [IC, SEQ] random, W: [IC, OC] identity-ish
    np.random.seed(42)
    act = np.random.randn(IC, SEQ).astype(np.float32) * 0.1
    W = np.eye(IC, OC, dtype=np.float32)  # identity for easy verification

    # Pack into input surface: [IC, SP] where sp[0:SEQ]=act, sp[SEQ:SEQ+OC]=W
    inp = np.zeros((IC, SP), dtype=np.float32)
    inp[:, :SEQ] = act
    inp[:, SEQ:SEQ+OC] = W
    inp_flat = inp.flatten()

    # Write input
    lib.ane_bridge_write_input(kern, 0, inp_flat.ctypes.data, input_bytes)

    # Eval
    print("Evaluating...")
    t0 = time.time()
    ok = lib.ane_bridge_eval(kern)
    t1 = time.time()
    if not ok:
        print(f"FAIL: eval returned false ({t1-t0:.6f}s)")
        lib.ane_bridge_free(kern)
        sys.exit(1)
    print(f"Eval: OK ({t1-t0:.6f}s)")

    # Read output
    out = np.zeros((OC, SEQ), dtype=np.float32)
    lib.ane_bridge_read_output(kern, 0, out.ctypes.data, output_bytes)

    # Verify: y = act.T @ W → [SEQ, OC], but output layout is [OC, SEQ]
    # The MIL does: transpose(act) → [seq,ic], matmul with W[ic,oc] → [seq,oc]
    # Then transposes back → [oc,seq]
    expected = act.T @ W  # [SEQ, IC] @ [IC, OC] = [SEQ, OC]
    expected = expected.T  # [OC, SEQ] - matches output layout

    max_err = np.max(np.abs(out - expected))
    mean_err = np.mean(np.abs(out - expected))
    print(f"\nVerification (act.T @ W):")
    print(f"  Max error:  {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    print(f"  Output[0,:5]: {out[0,:5]}")
    print(f"  Expected[0,:5]: {expected[0,:5]}")

    if max_err < 0.05:  # fp16 tolerance
        print("\n✓ PASS — ANE matmul matches CPU reference!")
    else:
        print(f"\n✗ FAIL — error too large (fp16 tolerance=0.05)")

    # Benchmark
    print("\nBenchmarking 100 evals...")
    t0 = time.time()
    for _ in range(100):
        lib.ane_bridge_write_input(kern, 0, inp_flat.ctypes.data, input_bytes)
        lib.ane_bridge_eval(kern)
    t1 = time.time()
    print(f"  100 evals in {t1-t0:.3f}s = {(t1-t0)/100*1000:.3f} ms/eval")

    # Cleanup
    lib.ane_bridge_free(kern)
    print("\nDone!")


if __name__ == "__main__":
    main()
