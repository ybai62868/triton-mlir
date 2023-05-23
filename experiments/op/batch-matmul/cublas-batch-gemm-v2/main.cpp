#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 3 // 矩阵维度
#define BATCH 2 // 批次大小

int main(void)
{
    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;
    cublasHandle_t handle;
    float *d_A, *d_B, *d_C;
    float alpha = 1.0f, beta = 0.0f;
    int lda = N, ldb = N, ldc = N;
    int strideA = N * N, strideB = N * N, strideC = N * N;
    int batchCount = BATCH;
    int sizeA = strideA * batchCount, sizeB = strideB * batchCount, sizeC = strideC * batchCount;
    float A[N * N * BATCH] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}; // 输入矩阵A
    float B[N * N * BATCH] = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}; // 输入矩阵B
    float C[N * N * BATCH] = {0.0f}; // 输出矩阵C

    // 分配设备内存
    cudaStatus = cudaMalloc((void**)&d_A, sizeA * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&d_B, sizeB * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&d_C, sizeC * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }

    // 将输入数据从主机内存复制到设备内存
    cudaStatus = cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }
    cudaStatus = cudaMemcpy(d_B, B, sizeB * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }

    // 创建cublas句柄
    cublasStatus = cublasCreate(&handle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("cublasCreate failed: %d\n", cublasStatus);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }

    // 执行矩阵乘法操作
    cublasStatus = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, CUDA_R_16F, N, strideB,
        d_A, CUDA_R_16F, N, strideA,
        &beta,
        d_C, CUDA_R_16F, N, strideC,
        batchCount,
        CUDA_R_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("cublasGemmStrideBatchedEx failed: %d\n", cublasStatus);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        return 1;
    }

    // 将输出数据从设备内存复制到主机内存
    cudaStatus = cudaMemcpy(C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        return 1;
    }

    // 打印输出矩阵C
    printf("Output matrix C:\n");
    for (int i = 0; i < BATCH; i++) {
        printf("Batch %d:\n", i);
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                printf("%f ", C[i * strideC + j * N + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // 释放内存和句柄
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}