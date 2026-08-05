#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MR 6
#define NR 8

// dev test functions
void random_fill(float *a, int n, int mrand){
    for(int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            a[r*n+c] = rand() % mrand;
}

void zero_fill(float *a, int n, int r, int c){
    for (int vr = 0; vr < r; vr++)
        for (int vc = 0; vc < c; vc++)
            a[vr*n+vc] = 0.0f;
}


/* This function loops over tiles of C, packs B for each tile, and then calls the micro kernel to do the actual computation
 * */
void nuggem(
            int M, int N, int K,            // Matrix dimensions: C(MxN) = A(MxN) * B(MxN)
            float *restrict A,              // Row-major, leading dimension K
            float *restrict B,              // Row-major, leading dimension N 
            float *restrict C,              // Row-major, leading dimension N 
            int ldA, int ldB, int ldC       // Leading dimensions (usually K,N,N)
            ){ 

}

/* Main function to compute a 6x8 block of C ( 6 rows and 8 columns )
 * Using 6 rows of A (each row Kc elements long)
 * 8 columns of B ( each column Kc elements long; packed contiguously)
 * */
void micro_kernel_6x8(
        int Kc,                             // How many rows of B are packed ( the K dimension of this tile: 16,32,64, A multiple of 8 for AVX
        float *A,                           // Pointer to 6 rows of A, positioned at the right K offset
        float *packed_B,                    // The contiguous packed buffer
        float *C,                           // Pointer to 6x8 block of C to update
        int ldC,                            // Leading dimension of C for strided storing
        int ldA
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

    // inner loop over K ( steps by 8 )
    for (int k = 0; k < Kc; k += 8){

        // load the 6 rows of A for K block
        __m256 a0 = _mm256_loadu_ps(A0);
        __m256 a1 = _mm256_loadu_ps(A1);
        __m256 a2 = _mm256_loadu_ps(A2);
        __m256 a3 = _mm256_loadu_ps(A3);
        __m256 a4 = _mm256_loadu_ps(A4);
        __m256 a5 = _mm256_loadu_ps(A5);


        // advance packedB to the next 8 K elements
        for (int nc = 0; nc < 8; nc++){
            __m256 b = _mm256_loadu_ps(packed_B + nc * Kc);

            // FMA all 6 rows
            c0 = _mm256_fmadd_ps(a0, b, c0);
            c1 = _mm256_fmadd_ps(a1, b, c1);
            c2 = _mm256_fmadd_ps(a2, b, c2);
            c3 = _mm256_fmadd_ps(a3, b, c3);
            c4 = _mm256_fmadd_ps(a4, b, c4);
            c5 = _mm256_fmadd_ps(a5, b, c5);
        }
        A0 += 8; A1 += 8;
        A2 += 8; A3 += 8;
        A4 += 8; A5 += 8;

        packed_B += 8;
    }
        _mm256_storeu_ps(C + 0 * ldC, c0);
        _mm256_storeu_ps(C + 1 * ldC, c1);
        _mm256_storeu_ps(C + 2 * ldC, c2);
        _mm256_storeu_ps(C + 3 * ldC, c3);
        _mm256_storeu_ps(C + 4 * ldC, c4);
        _mm256_storeu_ps(C + 5 * ldC, c5);
    
}

void pack_B_tile(
    float *B,                               // Original B matrix (row major)
    int ldB,                                // Leading dimension of B (usually N)
    int row_start,                          // Starting row of the tile in B
    int col_start,                          // Starting column of the tile in B
    int Kc,                                 // Number of rows to pack ( K dimension )
    int Nr,                                 // Numer of columns to pack ( C dimension )
    float *packed                           // Packed B
    ){

    for (int col = 0; col < Nr; col++){
        for (int row = 0; row < Kc; row++) 
            packed[col * Kc + row] = B[(row_start + row) * ldB + (col_start + col)];
    }

}

void micro_kernel_scalar(
        int Kc,                             // Number of rows to pack ( K dimension )
        int ldC,                            // Leading dimension of C
        int ldA,                            // Leading dimension of A 
        float *A,                           // Pointer to A matrix 
        float *packedB,                     // Pointer to packed B matrix 
        float *C                            // Pointer to packed C matrix
        ){
    // init accumulators
    float c[6][8] = {0};

    // loop over K, step by 8 
    for (int k = 0; k < Kc; k++){
        
        // for each of the 8 columns of C and B
        for (int nr = 0; nr < 6; nr++){

            // packed b for this column starts at PackedB + nc * Kc
            // we are at offset K in that column 
            float a_val = A[nr * ldA + k];

            // for each of the 8 rows of C and B 
            for (int nc = 0; nc < 8;nc++){
                float b_val = packedB[nc * Kc + k];
                
                // accumulate
                c[nr][nc] += a_val * b_val;
            }
        }
    }

    // Store back to C
    for (int nr = 0; nr < 6; nr++){
        for (int nc = 0; nc < 8; nc++){
            C[nr * ldC + nc] += c[nr][nc];
        }
    }
}


int main(){
    srand(time(NULL));
    int Kc = 8; int ldA = Kc, ldC = 8;

    int dim_size = 1024;
    float *A = malloc(dim_size * dim_size * sizeof(float));
    float *B = malloc(dim_size * dim_size * sizeof(float));
    float *C_scalar = malloc(dim_size * dim_size * sizeof(float));
    float *packedB = malloc(Kc * ldC * sizeof(float));
    float *C_avx = malloc(dim_size * dim_size * sizeof(float));

    random_fill(A, dim_size, 4);
    random_fill(B, dim_size, 4);

    pack_B_tile(B, dim_size,0,0,Kc,ldC,packedB); 

    micro_kernel_scalar(Kc,ldC, ldA, A, packedB, C_scalar);
    
    micro_kernel_6x8(Kc,A,packedB,C_avx,ldC,ldA);

    printf("RESULT C SCALAR:\n");
    for (int r = 0; r < 6; r++){
        for (int c = 0; c < 8; c++){
            printf("%8.1f ", C_scalar[r*8+c]);
        }
    printf("\n");
    }
    
    printf("RESULT C AVX:\n");
    for (int r = 0; r < 6; r++){
        for (int c = 0; c < 8; c++){
            printf("%8.1f ", C_avx[r*8+c]);
        }
    printf("\n");
    }
    
    float max_diff = 0;
    for (int i = 0; i < 6 * 8; i++) {
        float diff = C_scalar[i] - C_avx[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        printf("%8.1f ", diff);
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("\nMax difference: %f\n", max_diff);


    
    free(A); free(B); free(C_scalar); free(packedB); free(C_avx);

    return 0;
}
