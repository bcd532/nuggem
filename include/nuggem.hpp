#ifndef NUGGEM_HPP
#define NUGGEM_HPP

#include <immintrin.h>


namespace nuggem{

// C = A * B, single precision, row-major.
//   A is M×K, B is K×N, C is M×N.
//   ldA,ldB,ldC = elements per row in storage (ldA>=K, ldB>=N, ldC>=N).
// Overwrite semantics: C is fully written (implementation zeroes it first).
// Current CPU backend constraints: M % 6 == 0, N % 8 == 0 (no edge handling yet).
    void ng_sgemm_cpu(int M, int N, int K,
        const float *__restrict__ A,
        const float *__restrict__ B,
        float *__restrict__ C,
        int ldA, int ldB, int ldC);


namespace detail{

template<int MR, int NR>
void micro_kernel(
        int Kc,
        const float* A, const float* packed_B,
        float* C, int ldC
        ){
    __m256 acc[MR];
    for (int i = 0; i < MR; i++){
        acc[i] = _mm256_setzero_ps();
    }

    for (int k = 0; k < Kc; k++){
        __m256 b = _mm256_loadu_ps(packed_B + k*NR);
        for (int i = 0; i < MR; i++)
            acc[i] = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k*MR + i), b, acc[i]);
    }

    for (int i = 0; i < MR; i++)
        _mm256_storeu_ps(C + i*ldC, _mm256_add_ps(_mm256_loadu_ps(C + i*ldC), acc[i]));


}

}

}

#endif
