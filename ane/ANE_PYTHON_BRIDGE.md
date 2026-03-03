# Calling Apple Neural Engine from Python: A Working Guide

**Status**: Working on macOS 15+ (Apple Silicon M1/M2/M3/M4)
**Date**: March 2026
**Based on**: [maderix/ANE](https://github.com/maderix/ANE) training pipeline

## TL;DR

We got Python calling the Apple Neural Engine (ANE) for matrix multiplication via a C bridge library, using Apple's private `_ANEInMemoryModel` APIs. The key breakthrough was discovering that passing `nil` vs `@{}` (empty dictionary) to `modelWithMILText:weights:optionsPlist:` produces completely different behavior — `nil` silently fails, `@{}` works.

**Result**: ANE matmul is **2-8x faster than MPS (Metal GPU)** for typical transformer dimensions, and runs on completely separate hardware — meaning you can run ANE compute in parallel with GPU work.

## The Bug That Blocked Everything

### Symptom
When calling `modelWithMILText:weights:optionsPlist:` for kernels without const weights (dynamic matmul pattern), the call either:
- Returns `nil` (descriptor creation fails), or
- Returns a descriptor but `compileWithQoS:` fails with `InvalidMILProgram`

### Root Cause
The private ANE API treats `nil` and `@{}` (empty NSDictionary) differently for the `weights:` parameter:

```objc
// BROKEN — fails silently or produces InvalidMILProgram
id desc = objc_msgSend(g_D, @selector(modelWithMILText:weights:optionsPlist:),
    milData, nil, nil);

// WORKS — compiles and runs correctly
id desc = objc_msgSend(g_D, @selector(modelWithMILText:weights:optionsPlist:),
    milData, @{}, nil);
```

This single-character difference (`nil` → `@{}`) was the only thing preventing the Python bridge from working. The maderix/ANE training pipeline's `compile_kern_mil_w()` in `io.h` always receives a dictionary (even if empty `@{}`) because callers pass `@{}` for no-weight kernels. The bridge library had a conditional that converted empty dictionaries to `nil`, breaking the flow.

### Fix
In `ane_bridge.m`, line 96:
```objc
// Before (broken):
milData, wdict.count > 0 ? wdict : nil, nil);

// After (working):
milData, wdict.count > 0 ? wdict : @{}, nil);
```

## Architecture

```
Python (PyTorch/numpy)
    |
    | ctypes FFI
    v
libane_bridge.dylib (Objective-C, ARC)
    |
    | objc_msgSend to private classes
    v
AppleNeuralEngine.framework (private)
    |
    | IOSurface shared memory
    v
ANE Hardware (16-core Neural Engine)
```

### Components

1. **MIL Text** — Apple's Model Intermediate Language, a text-based IR that the ANE compiler consumes
2. **Bridge Library** (`libane_bridge.dylib`) — C-callable wrapper around private ANE Objective-C APIs
3. **IOSurface** — macOS shared-memory mechanism for zero-copy data transfer to/from ANE
4. **Private APIs**: `_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel`, `_ANERequest`, `_ANEIOSurfaceObject`

### Dynamic Weight Pattern

Instead of baking weights into the compiled model (which would require recompilation when weights change during training), we use the "dynamic matmul" pattern from maderix/ANE:

- Pack activations AND weights into a single input IOSurface
- Layout: `[1, IC, 1, SEQ+OC]` where `sp[0:SEQ]` = activations, `sp[SEQ:SEQ+OC]` = weights
- The MIL program slices these apart, does the matmul, and writes results to the output IOSurface
- Weights can change every step without recompiling the ANE kernel

## Quick Start

### 1. Build the Bridge

```bash
cd bridge/
xcrun clang -O2 -shared -o libane_bridge.dylib ane_bridge.m \
    -framework Foundation -framework IOSurface -fobjc-arc
```

### 2. Run the Test

```bash
python3 test_dyn_matmul.py
```

Expected output:
```
=== ANE Dynamic Matmul Test ===
IC=128, OC=128, SEQ=64, SP=192
ane_bridge_init: OK
Compiling...
Compile + load: OK (0.059s)
Evaluating...
Eval: OK (0.000261s)

Verification (act.T @ W):
  Max error:  0.000122
  Mean error: 0.000015

PASS — ANE matmul matches CPU reference!
```

### 3. Run the Benchmark

```bash
# Needs PyTorch for MPS comparison:
python3 bench_full.py
```

## Benchmark Results (Mac Mini M4, 16GB)

Matmul latency including IOSurface data transfer (ms):

| Operation | ANE | CPU (Accelerate) | MPS (Metal) | ANE vs MPS |
|-----------|-----|-------------------|-------------|------------|
| 64x64, seq=512 | 0.087 | 0.007 | 0.658 | **7.6x faster** |
| 64x256, seq=512 | 0.123 | 0.016 | 0.680 | **5.5x faster** |
| 128x128, seq=512 | 0.119 | 0.016 | 0.724 | **6.1x faster** |
| 128x512, seq=512 | 0.191 | 0.053 | 0.430 | **2.3x faster** |
| 512x128, seq=512 | 0.149 | 0.051 | 0.678 | **4.6x faster** |
| 768x768, seq=256 | 0.327 | 0.242 | 0.508 | **1.6x faster** |
| 768x3072, seq=256 | 1.143 | 2.091 | 0.924 | 0.8x (MPS wins) |
| 1024x4096, seq=256 | 1.783 | 2.983 | 1.556 | 0.9x (tie) |

### Key Takeaways

1. **ANE beats MPS by 2-8x** at dimensions typical of small-medium transformers (64-512)
2. **CPU (Accelerate/AMX) beats ANE** at small dimensions due to ~0.1ms IOSurface overhead
3. **ANE beats CPU** at large dimensions (768+) where compute dominates overhead
4. **ANE runs independently** — it's separate hardware from GPU and CPU, enabling parallelism
5. **Compilation is one-time** — ~60ms to compile a kernel, then reused for entire training run

### When to Use ANE vs MPS vs CPU

- **ANE**: Best when you can batch multiple matmuls per dispatch, or when MPS is busy with other work
- **MPS**: Best for very large matmuls (3072+) and when you need the full CUDA-like programming model
- **CPU (Accelerate)**: Best for small matmuls where IOSurface overhead would dominate

## MIL Text Format

The MIL (Model Intermediate Language) must follow a specific format. Here's the working pattern for a dynamic matmul:

```python
def gen_dyn_matmul_mil(ic, oc, seq):
    """Generate MIL for: output[oc,seq] = act[seq,ic] @ W[ic,oc]"""
    sp = seq + oc
    mil = f"""program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}},
 {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}},
 {{"coremltools-version", "9.0"}})]
{{
    func main<ios18>(tensor<fp32, [1, {ic}, 1, {sp}]> x) {{
        // ... slice, reshape, transpose, matmul, cast ...
    }} -> (y);
}}"""
```

Critical format requirements:
- Must start with `program(1.3)` and include `buildInfo`
- Function signature: `func main<ios18>(...)`
- Return syntax: `} -> (output_var);`
- All tensors must be 4D: `[batch, channels, height, width]`
- Every operation needs a `name=string("...")` attribute
- Type annotations are required on every variable

See `test_dyn_matmul.py` for the complete working MIL generator.

## Python Bridge API

```python
import ctypes
lib = ctypes.CDLL("libane_bridge.dylib")

# Initialize (call once)
lib.ane_bridge_init()  # returns 0 on success

# Compile a kernel (one-time cost ~60ms)
kern = lib.ane_bridge_compile(
    mil_bytes, len(mil_bytes),   # MIL text as UTF-8 bytes
    None, 0,                      # weight_data, weight_len (None for dynamic)
    1, input_sizes,               # n_inputs, array of byte sizes
    1, output_sizes               # n_outputs, array of byte sizes
)

# Write input data (IOSurface memcpy)
lib.ane_bridge_write_input(kern, 0, data.ctypes.data, nbytes)

# Execute on ANE (~0.1ms)
lib.ane_bridge_eval(kern)

# Read output data
lib.ane_bridge_read_output(kern, 0, output.ctypes.data, nbytes)

# Cleanup
lib.ane_bridge_free(kern)
```

## Known Limitations

1. **Private APIs** — Could break with any macOS update. No stability guarantees.
2. **fp16 precision** — ANE operates in fp16 internally. Input/output can be fp32 (cast in MIL).
3. **4D tensors only** — All ANE tensors must be exactly 4D `[B,C,H,W]`.
4. **IOSurface overhead** — ~0.05-0.1ms per read/write. Dominates at small sizes.
5. **No gradient support** — Must implement backward pass as separate MIL kernels.
6. **Compilation cost** — ~60ms per unique kernel shape. Cache compiled kernels.
7. **Limited operations** — Not all ops are ANE-accelerated. `matmul`, `conv`, `softmax`, basic elementwise work. Complex ops may fall back to CPU.

## Files

- `ane_bridge.h` — C header for the bridge API
- `ane_bridge.m` — Objective-C bridge implementation (the fix is on line 96)
- `test_dyn_matmul.py` — Working test: compile + eval + verify a dynamic matmul
- `bench_ane_vs_cpu.py` — ANE vs CPU benchmark
- `bench_full.py` — Full benchmark: ANE vs CPU vs MPS

## Credits

- [maderix/ANE](https://github.com/maderix/ANE) — The original ANE training pipeline that proved this is possible
- The dynamic weight pattern (packing weights into input IOSurface) is from maderix's `training_dynamic/` pipeline
