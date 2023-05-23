#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

template <typename T, typename S>
void allocate_memory(int batch_count, int m, int n, int k, T **A, T **B, S **C) {
    cudaMallocManaged(A, batch_count * m * k * sizeof(T));
    cudaMallocManaged(B, batch_count * k * n * sizeof(T));
    cudaMallocManaged(C, batch_count * m * n * sizeof(S));
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

template <typename T, typename S>
int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
                   int m, int n, int k, T *A, T *B, S *C, int lda, int ldb, int ldc,
                   S *alpha, S *beta, int algo) {
    cudaDataType_t AType, BType, CType, ComputeType;
    if (std::is_same<T, float>::value) {
        AType = BType = CType = ComputeType = CUDA_R_32F;
    } else if (std::is_same<T, __half>::value) {
        AType = BType = CType = ComputeType = CUDA_R_16F;
    } else if (std::is_same<T, int8_t>::value) {
        AType = BType = CUDA_R_8I;
        CType = ComputeType = CUDA_R_32I;
    } else {
        printf("Not supported data type.");
        return -1;
    }
    cublasStatus_t status;
    status = cublasGemmEx(handle,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          alpha,
                          A,
                          AType,
                          lda,
                          B,
                          BType,
                          ldb,
                          beta,
                          C,
                          CType,
                          ldc,
                          ComputeType,
                          static_cast<cublasGemmAlgo_t>(algo));
    
    if (status == CUBLAS_STATUS_SUCCESS)
        return 1;
    else
        return -1;
}

template <typename T, typename S>
void test_gemm(cublasHandle_t handle, int m, int n, int k, T *A, T *B, S *C,
               S *alpha, S *beta, int algo, int iteration) {
    float total_time = 0;
    for (int i = 0; i < iteration; ++i) {
        struct timeval start, end;
        cudaDeviceSynchronize();
        cudaProfilerStart();
        gettimeofday(&start, NULL);
        int success = cublas_gemm_ex(handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     n,
                                     m,
                                     k,
                                     B,
                                     A,
                                     C,
                                     n,
                                     k,
                                     n,
                                     alpha,
                                     beta,
                                     static_cast<cublasGemmAlgo_t>(algo));
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        cudaProfilerStop();
        if (success > 0 && i > 0)
            total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
    if (total_time > 0)
        printf("algo %d: %.3f ms\n", algo, total_time / (iteration - 1));
}

int main() {
    int batch_count = 16;
    int m = 4096, n = 4096, k = 4096;
    printf("shape: (%d, %d, %d) x (%d, %d, %d)\n", batch_count, m, k, batch_count, k, n);
    // without tensor core
    int start_algo = CUBLAS_GEMM_DEFAULT;
    int end_algo = CUBLAS_GEMM_ALGO23;

    // with tensor core
    int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;
    int iteration = 10;

    float *fA, *fB, *fC;
    __half *hA, *hB, *hC;
    float f_alpha = 1, f_beta = 0;
    __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
    int32_t i_alpha = 1, i_beta = 0;
    allocate_memory(batch_count, m, n, k, &fA, &fB, &fC);
    allocate_memory(batch_count, m, n, k, &hA, &hB, &hC);
    for (int i = 0; i < m * k; ++i) {
        fA[i] = float(i % 255 - 127) / 127;
        hA[i] = __float2half_rn(fA[i]);
    } 
    for (int i = 0; i < k * n; ++i) {
        fB[i] = float(i % 255 - 127) / 127;
        hB[i] = __float2half_rn(fB[i]);
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
    for (int algo = start_algo; algo <= end_algo; ++algo)
        test_gemm(handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, algo, iteration);
    for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
        test_gemm(handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, algo, iteration);
    

    printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
    for (int algo = start_algo; algo <= end_algo; ++algo)
        test_gemm(handle, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, iteration);
    for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
        test_gemm(handle, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, iteration);

    
    free_memory(fA, fB, fC);
    free_memory(hA, hB, hC);
    return 0;
}