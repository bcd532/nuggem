# nuggem
> (new gemm / nugget gemm)

Single-precision GEMM (general matrix multiply) in C++, computing `C = A * B` for row-major matrices.

A 6x8 register-blocked microkernel using AVX2 and FMA, fed by tiles packed k-major into contiguous buffers so the kernel streams cache-friendly loads instead of striding through the source matrices. B is packed NR values per depth, A is packed MR values per depth so the kernel can broadcast straight out of the buffer. K is blocked at Kc = 128.

## Requirements

- An x86-64 CPU with AVX2 and FMA support.
- A C++17 compiler.
- OpenBLAS plus headers, for the benchmark comparison in the example. It calls `openblas_set_num_threads` to pin the reference to one thread.

## Build

```
make
make run
```

The Makefile finds OpenBLAS through `pkg-config` and falls back to `-lopenblas`. Binaries land in `build/`.

## API

```c++
#include "nuggem.hpp"

namespace nuggem {
    void ng_sgemm_cpu(int M, int N, int K,
                      const float *__restrict__ A,
                      const float *__restrict__ B,
                      float *__restrict__ C,
                      int ldA, int ldB, int ldC);
}
```

| Parameter | Meaning |
| --- | --- |
| `M`, `N`, `K` | `C` is M×N, `A` is M×K, `B` is K×N |
| `A`, `B` | Row-major inputs |
| `C` | Row-major output, fully overwritten |
| `ldA`, `ldB`, `ldC` | Elements per row in storage; `ldA >= K`, `ldB >= N`, `ldC >= N` |

`C` is zeroed by the implementation before accumulation, so it does not need to be cleared by the caller.

Constraints: `M % 6 == 0` and `N % 8 == 0`. There is no edge handling, so other sizes read and write past the tile.

`nuggem::detail` holds the microkernel and the packing routines. It is internal and not part of the API.

### Example

```c++
#include "nuggem.hpp"
#include <vector>

int M = 1020, N = 1024, K = 1068;
std::vector<float> A(M*K), B(K*N), C(M*N);
// fill A and B ...

nuggem::ng_sgemm_cpu(M, N, K, A.data(), B.data(), C.data(), K, N, N);
```

`examples/cpu_example.cpp` checks the result against an independent triple loop, times it against single-threaded OpenBLAS, and appends both to `bench.csv`.

## Layout

```
include/nuggem.hpp        public API + microkernel
src/cpu/nuggem_cpu.cpp    CPU backend
src/cuda/                 placeholder
examples/cpu_example.cpp  correctness check + benchmark
```

MIT
