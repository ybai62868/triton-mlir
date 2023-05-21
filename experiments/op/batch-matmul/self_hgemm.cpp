#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace std;


#define CHECK_CUDA(call) { gpuAssert((call), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


int main(void)
{
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int batch_count = 4;


    __half* h_a =  (__half*)malloc(sizeof(__half) * m * k * batch_count);
    __half* h_b =  (__half*)malloc(sizeof(__half) * k * n * batch_count);
    __half* h_c =  (__half*)malloc(sizeof(__half) * m * n * batch_count);


    for ( int batch = 0; batch < batch_count; batch++ ) {
        for ( int i = 0; i < m; i++ ) {
            for ( int j = 0;j < n; j++) {
                int index = batch * m * n + i * n + j;
                h_a[index] = index * index + 1.0f;
                h_b[index] = index + 1.0f;
                h_c[index] = 0.0;
            }
        }
    }

    __half* d_a;
    __half* d_b;
    __half* d_c;
    CHECK_CUDA(cudaMalloc(&d_a, sizeof(__half) * m * k * batch_count));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(__half) * k * n * batch_count));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(__half) * m * n * batch_count));

    CHECK_CUDA(cudaMemcpy(h_a, d_a, sizeof(__half) * m * k * batch_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b, d_b, sizeof(__half) * k * n * batch_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_c, d_c, sizeof(__half) * m * n * batch_count, cudaMemcpyDeviceToHost));

    cublasHandle_t handle;
    cublasCreate(&handle);

    __half alpha = 1.0f;
    __half beta = 1.0f;
    int warm_up = 5;
    int iteration = 10;
    for ( int i = 0; i < warm_up; i++ ) {
        cublasHgemmStridedBatched(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  (const half*) d_a,
                                  k,
                                  m * k,
                                  (const half*) d_b,
                                  n,
                                  k * n,
                                  &beta,
                                  d_c, n,
                                  m*n,
                                  batch_count);

    }

    float total_time = 0.0;
    for ( int i = 0; i < iteration; i++ ) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start, 0);
        cublasHgemmStridedBatched(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  (const half*) d_a,
                                  k,
                                  m * k,
                                  (const half*) d_b,
                                  n,
                                  k * n,
                                  &beta,
                                  d_c, n,
                                  m*n,
                                  batch_count);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, end);
        total_time += elapsed_time;
        std::cout << "Iteration " << i + 1 << ": " << elapsed_time << " ms" << std::endl;
    }
    float average_time = total_time / 10;
    std::cout << "Average time: " << average_time << " ms" << std::endl;
    long long computation = 2.0 * m * n * k * batch_count;
    // std::cout << "TFLOPs: " << computation << std::endl;
    std::cout << "TFLOPS: " << computation * 1e-9 / average_time << std::endl;
    cudaMemcpy(h_c, d_c, sizeof(__half) * m * n * batch_count, cudaMemcpyDeviceToHost);


    cublasDestroy(handle);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}