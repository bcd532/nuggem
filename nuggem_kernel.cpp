#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cblas.h>
#include <vector>


#define MR 6
#define NR 8



// dev test functions
void random_fill(float *a, int x,int y, int mrand){
    for(int r = 0; r < x; r++)
        for (int c = 0; c < y; c++)
            a[r*y+c] = rand() % mrand;
}

double now_sec(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void zero_fill(float *a, int n, int r, int c){
    for (int vr = 0; vr < r; vr++)
        for (int vc = 0; vc < c; vc++)
            a[vr*n+vc] = 0.0f;
}

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

/* Compute a 6x8 block of C ( 6 rows and 8 columns )
 * Using 6 rows of A (each row Kc elements long)
 * packed_B holds the 8-wide tile k-major: NR values per depth, laid out contiguously
 * */
void micro_kernel_6x8(
        int Kc,                             // How many depths are packed ( the K dimension of this tile )
        float *A,                           // Pointer to 6 rows of A, positioned at the right K offset
        float *packed_B,                    // The contiguous packed buffer
        float *C,                           // Pointer to 6x8 block of C to update
        int ldC,                            // Leading dimension of C for strided storing
        int Nr                              // Columns per depth in packed_B ( packing stride )
        ){
    

    // init accumulators
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();

      // inner loop over K ( steps by 1 )
    for (int k = 0; k < Kc; k++){

        // load the Nr packed B values for depth k ( contiguous )
        __m256 b = _mm256_loadu_ps(packed_B + k * Nr);

        // broadcast each A element across all 8 lanes THEN FMA into its C row
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k*MR + 0), b, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k*MR + 1), b, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k*MR + 2), b, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k*MR + 3), b, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k*MR + 4), b, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k*MR + 5), b, c5);
    }

    // load-add-store so repeated tile calls accumulate ( C must be pre-zeroed )
    _mm256_storeu_ps(C + 0 * ldC, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldC), c0));
    _mm256_storeu_ps(C + 1 * ldC, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldC), c1));
    _mm256_storeu_ps(C + 2 * ldC, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldC), c2));
    _mm256_storeu_ps(C + 3 * ldC, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldC), c3));
    _mm256_storeu_ps(C + 4 * ldC, _mm256_add_ps(_mm256_loadu_ps(C + 4 * ldC), c4));
    _mm256_storeu_ps(C + 5 * ldC, _mm256_add_ps(_mm256_loadu_ps(C + 5 * ldC), c5));
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
            micro_kernel_6x8(Kc, packed_A,packed_B, c, ldC, NR);

        }

    }

    free(packed_B); free(packed_A);

}



int main(){
    srand(time(NULL));
    openblas_set_num_threads(1);

    int M = 1020;
    int N = 1024;
    int K = 1068;
    int ldA = K;
    int ldB = N;
    int ldC = N;

    // RAII buffers: sized here, zero-initialized, freed automatically at scope exit.
    // C(...) with one size arg value-initializes to 0.0f, so it doubles as calloc.
    std::vector<float> A(M*K), B(K*N), C(M*N), C_ref(M*N), C_blas_ref(M*N);

    random_fill(A.data(), M,K, 4);
    random_fill(B.data(), K,N, 4);

    // correctness reference (independent triple loop), built once
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++){
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i*ldA + k] * B[k * ldB + j];
            C_ref[i * ldC + j] = s;
        }

    // correctness of nuggem, on a clean C, BEFORE any timing clobbers it
    std::fill(C.begin(), C.end(), 0.0f);
    nuggem(M,N,K, A.data(), B.data(), C.data(), ldA,ldB,ldC);

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++){
        float d = C[i] - C_ref[i];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
    }
    printf("M=%d N=%d K=%d   Max difference (nuggem vs triple-loop): %f\n", M, N, K, max_diff);

    // nuggem timing 
    int reps = 20;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++){
        std::fill(C.begin(), C.end(), 0.0f);
        nuggem(M,N,K, A.data(), B.data(), C.data(), ldA,ldB,ldC);
    }
    double t1= now_sec();

    // OpenBLAS timing
    double t2= now_sec();
    for (int r = 0; r < reps; r++){
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A.data(), ldA,
                B.data(), ldB,
                0.0f, C_blas_ref.data(), ldC);
    }
    double t3=now_sec();

    volatile float sink = C[0] + C_blas_ref[0];
    (void)sink;

    double sec_per_call2 = (t3-t2) / reps;
    double gflops2 = (2.0 * M * N * K) / sec_per_call2 / 1e9;
    printf("CBLAS:  %dx%dx%d: %.2f ms/call, %.1f GFLOP/s\n",
        M,N,K,sec_per_call2 * 1e3, gflops2);

    double sec_per_call = (t1-t0) / reps;
    double gflops = (2.0 * M * N * K) / sec_per_call / 1e9;
    printf("NUGGEM: %dx%dx%d: %.2f ms/call, %.1f GFLOP/s\n",
        M,N,K,sec_per_call * 1e3, gflops);

    // 
    return 0;
}
