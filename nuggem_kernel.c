#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cblas.h>

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
        int ldA,
        int Nr                              // Columns per depth in packed_B ( packing stride )
        ){
    

    // init accumulators
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();

    // pointers to the 6 rows of A we deal with 
    float *A0 = A;
    float *A1 = A + ldA;
    float *A2 = A + 2 * ldA;
    float *A3 = A + 3 * ldA;
    float *A4 = A + 4 * ldA;
    float *A5 = A + 5 * ldA;

    // inner loop over K ( steps by 1 )
    for (int k = 0; k < Kc; k++){

        // load the Nr packed B values for depth k ( contiguous )
        __m256 b = _mm256_loadu_ps(packed_B + k * Nr);

        // broadcast each A element across all 8 lanes THEN FMA into its C row
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A0 + k), b, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A1 + k), b, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A2 + k), b, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A3 + k), b, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A4 + k), b, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A5 + k), b, c5);
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
            float *restrict A,              // Row-major, leading dimension K
            float *restrict B,              // Row-major, leading dimension N 
            float *restrict C,              // Row-major, leading dimension N 
            int ldA, int ldB, int ldC       // Leading dimensions (usually K,N,N)
            ){
    int Kc = K;
    float *packed_B = malloc(Kc * NR * sizeof(float));
  
    for (int jc = 0; jc < N; jc += NR){
        pack_B_tile(B, ldB,0,jc,Kc,NR,packed_B); 
        for (int ic = 0; ic < M; ic += MR){
            float *a = A + ic * ldA;
            float *c = C + ic * ldC + jc;

            micro_kernel_6x8(Kc, a,packed_B, c, ldC, ldA, NR);

        }
        
    }

    free(packed_B);

}


void micro_kernel_scalar(
        int Mr,                             // Rows of the C block ( caller's tile height )
        int Nr,                             // Columns of the C block / packed per depth
        int Kc,                             // Depths to pack ( K dimension )
        int ldC,                            // Leading dimension of C
        int ldA,                            // Leading dimension of A 
        float *A,                           // Pointer to A matrix 
        float *packedB,                     // Pointer to packed B matrix 
        float *C                            // Pointer to C block
        ){
    // init accumulators ( VLA sized by the caller's block, so no initializer -> memset )
    float c[Mr][Nr];
    memset(c, 0, sizeof c);

    // loop over K, step by 1
    for (int k = 0; k < Kc; k++){
        
        // for each row of the C block
        for (int nr = 0; nr < Mr; nr++){

            // we are at offset k in row nr of A
            float a_val = A[nr * ldA + k];

            // for each column of the C block
            for (int nc = 0; nc < Nr; nc++){
                float b_val = packedB[k * Nr + nc];
                
                // accumulate
                c[nr][nc] += a_val * b_val;
            }
        }
    }

    // Store back to C
    for (int nr = 0; nr < Mr; nr++){
        for (int nc = 0; nc < Nr; nc++){
            C[nr * ldC + nc] += c[nr][nc];
        }
    }
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

    
    float *A    = malloc(M * K * sizeof(float));
    float *B    = malloc(K * N * sizeof(float));
    float *C    = calloc(M * N, sizeof(float));
    float *C_ref = calloc(M * N, sizeof(float));
    float *C_blas_ref = calloc(M * N, sizeof(float));

    random_fill(A, M,K, 4);
    random_fill(B, K,N, 4);

    memset(C, 0, M*N*sizeof(float));
    nuggem(M,N,K,A,B,C,ldA,ldB,ldC);

    int reps = 20;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++){
        memset(C,0, M*N*sizeof(float));
        nuggem(M,N,K,A,B,C,ldA,ldB,ldC);
        
    }
    double t1= now_sec();

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++){
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i*ldA + k] * B[k * ldB + j];
            C_ref[i * ldC + j] = s;
        }
    

    double t2= now_sec();
    for (int r = 0; r < reps; r++){
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, ldA,
                B, ldB,
                0.0f, C, ldC);
    }
    double t3=now_sec();

    double sec_per_call2 = (t3-t2) / reps;
    double gflops2 = (2.0 * M * N * K) / sec_per_call2 / 1e9;
    printf("%dx%dx%dx: %.2f ms/call, %.1f GFLOP/s\n",
        M,N,K,sec_per_call2 * 1e3, gflops2);
    

    double sec_per_call = (t1-t0) / reps;
    double gflops = (2.0 * M * N * K) / sec_per_call / 1e9;
    printf("%dx%dx%dx: %.2f ms/call, %.1f GFLOP/s\n",
        M,N,K,sec_per_call * 1e3, gflops);

    volatile float sink = C[0];
    (void)sink;

    float max_diff = 0.0f;
    
    for (int i = 0; i < M * N; i++){
        float d = C[i] - C_ref[i];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
    }
    printf("M=%d N=%d K=%d   Max difference: %f\n", M, N, K, max_diff);


    
    free(A); free(B); free(C); free(C_ref);

    return 0;
}
