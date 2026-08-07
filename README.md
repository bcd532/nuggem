# nuggem
> (new gemm / nugget gemm)

Single-precision GEMM (general matrix multiply) in C++, computing C = A * B for row-major matrices.

Design follows the standard high-performance approach: a 6x8 register-blocked microkernel using AVX2 and FMA, fed by tiles packed k-major into contiguous buffers so the kernel streams cache-friendly loads instead of striding through the source matrices. B is packed NR values per depth, A is packed MR values per depth so the kernel can broadcast straight out of the buffer. Correctness is checked against an independent triple loop, and timing is compared against single-threaded OpenBLAS in the same run.

## Current state, read this first

Packed A is actually wired into the kernel now instead of being computed and ignored. Results are correct (max diff vs the triple loop is 0), but it made things slower, not faster, so this is not worth pulling down yet.

The problem is loop order. pack_A_tile sits inside the `ic` loop, under the `jc` loop, so every A panel gets repacked once per column tile, N/8 times over a full call. That packing traffic ends up costing more than the FMA work it was supposed to help.

One run at 1020x1024x1068, single thread:

```
CBLAS:  19.25 ms/call, 115.9 GFLOP/s
NUGGEM: 142.45 ms/call,  15.7 GFLOP/s
```

The fix is to hoist the A packing out and pack each panel once, with the B tile loop on the inside. That is the next thing I am doing.

Other rough edges while I am in here:

- No edge handling. M has to be a multiple of 6 and N a multiple of 8 or the kernel runs off the end of the tile.
- Kc is just K, so there is no k blocking yet and a large K will not stay in L2.
- The microkernel accumulates with load-add-store, so C has to be zeroed before the call.

## Requirements

An x86-64 CPU with AVX2 and FMA support.

A BLAS with the CBLAS interface, for the benchmark comparison in main. It calls `openblas_set_num_threads` to pin the reference to one thread, so that means OpenBLAS specifically, plus its headers.

## Build

```
g++ -O2 -mavx2 -mfma nuggem_kernel.cpp -o nuggem -lblas
```

Some distros (Arch among them) put the headers in their own directory and name the library differently:

```
g++ -O2 -mavx2 -mfma -I/usr/include/openblas nuggem_kernel.cpp -o nuggem -lopenblas
```

MIT
