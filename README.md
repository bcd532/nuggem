# nuggem (new gemm / nugget gemm)

Single-precision GEMM (general matrix multiply) in C, computing C = A * B for row-major matrices.

Design follows the standard high-performance approach: a 6x8 register-blocked microkernel using AVX2 and FMA, fed by B tiles packed k-major into a contiguous buffer so the kernel streams cache-friendly loads instead of striding through the source matrix. A scalar microkernel is included as a reference for validating the vectorized path.

## Requirements

An x86-64 CPU with AVX2 and FMA support.

## Build

```
gcc -O2 -mavx2 -mfma nuggem_kernel.c -o nuggem
```

MIT
