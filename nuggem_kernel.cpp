#include <immintrin.h>
#include <stdlib.h>
#include <cblas.h>
#include "nuggem_kernel.hpp"

#define MR 6
#define NR 8


void pack_B_tile(
    float *B,                               // Original B matrix (row major)
    int ldB,                                // Leading dimension of B (usually N)
    int row_start,                          // Starting row (depth) of the tile in B
    int col_start,                          // Starting column (N) of the tile in B
    int Kc,                                 // Depths to pack ( K dimension )
    int Nr,                                 // Number of columns to pack ( N dimension )
    float *packed                           // Packed B ( k-major: Nr values per depth, contiguous )
    ){

    for (int depth = 0; depth < Kc; depth++){
        for (int col = 0; col < Nr; col++) 
            packed[depth * Nr + col] = B[(row_start + depth) * ldB + (col_start + col)];
    }

}

void pack_A_tile(
    float *A,                               // Original A matrix (row major)
    int ldA,                                // Leading dimension of A (usually K)
    int row_start,                          // Starting row (M) of the tile in A
    int col_start,                          // Starting column (K / depth) of the tile in A
    int Kc,                                 // Depths to pack ( K dimension )
    float *packed                           // Packed A ( intended k-major: MR values per depth )
    ){

    for (int depth = 0; depth < Kc; depth++){
        for (int m = 0; m < MR; m++) 
            packed[depth * MR + m] = A[(row_start + m) * ldA + (col_start + depth)];
    }

}

/* This function loops over tiles of C, packs B for each tile, and then calls the micro kernel to do the actual computation
 * */
void nuggem(
            int M, int N, int K,            // Matrix dimensions: C(MxN) = A(MxK) * B(KxN)
            float *__restrict__ A,              // Row-major, leading dimension K
            float *__restrict__ B,              // Row-major, leading dimension N
            float *__restrict__ C,              // Row-major, leading dimension N
            int ldA, int ldB, int ldC       // Leading dimensions (usually K,N,N)
            ){
    int Kc = K;
    float *packed_B = (float*)malloc(Kc * NR * sizeof(float));
    float *packed_A = (float*)malloc(MR * Kc * sizeof(float));

    for (int jc = 0; jc < N; jc += NR){
        pack_B_tile(B, ldB,0,jc,Kc,NR,packed_B);
        for (int ic = 0; ic < M; ic += MR){
            float *c = C + ic * ldC + jc;
            pack_A_tile(A,ldA,ic,0,Kc,packed_A);
            micro_kernel<MR,NR>(Kc, packed_A,packed_B, c, ldC);

        }

    }

    free(packed_B); free(packed_A);

}
