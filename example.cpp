#include <ctime>
#include <cblas.h>
#include <vector>
#include <fstream>
#include "nuggem_kernel.hpp"
#include <stdlib.h>
#include <iostream>


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
    nuggem(M,N,K, A.data(), B.data(), C.data(), ldA,ldB,ldC, 32);

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
        nuggem(M,N,K, A.data(), B.data(), C.data(), ldA,ldB,ldC,128);
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
    std::time_t now = std::time(nullptr);

    bool need_header = true;
    {
        std::ifstream check("bench.csv");
        if (check.good() && check.peek() != std::ifstream::traits_type::eof())
        need_header = false;
    }


    std::ofstream out("bench.csv", std::ios::app); 
    if (need_header){
        out << "impl,M,N,K,gflops,timestamps\n";
    }
    out << "CBLAS," << M << "," << N << "," << K << "," << gflops2 << "," << (long)now << "\n";   
    
    double sec_per_call = (t1-t0) / reps;
    double gflops = (2.0 * M * N * K) / sec_per_call / 1e9;
    out << "NUGGEM," << M << "," << N << "," << K << "," << gflops << "," << (long)now <<"\n";
    
    

    std::cout << "Results sent to 'bench.csv'";
    return 0;
}
