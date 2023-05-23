#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

#define CHECK_CUDA(call)                                                       \
  { gpuAssert((call), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

cublasGemmAlgo_t algoList[] = {
    CUBLAS_GEMM_DEFAULT,
    CUBLAS_GEMM_ALGO0,
    CUBLAS_GEMM_ALGO1,
    CUBLAS_GEMM_ALGO2,
    CUBLAS_GEMM_ALGO3,
    CUBLAS_GEMM_ALGO4,
    CUBLAS_GEMM_ALGO5,
    CUBLAS_GEMM_ALGO6,
    CUBLAS_GEMM_ALGO7,
    CUBLAS_GEMM_ALGO8,
    CUBLAS_GEMM_ALGO9,
    CUBLAS_GEMM_ALGO10,
    CUBLAS_GEMM_ALGO11,
    CUBLAS_GEMM_ALGO12,
    CUBLAS_GEMM_ALGO13,
    CUBLAS_GEMM_ALGO14,
    CUBLAS_GEMM_ALGO15,
    CUBLAS_GEMM_ALGO16,
    CUBLAS_GEMM_ALGO17,
    CUBLAS_GEMM_DFALT_TENSOR_OP,
    CUBLAS_GEMM_ALGO0_TENSOR_OP,
    CUBLAS_GEMM_ALGO1_TENSOR_OP,
    CUBLAS_GEMM_ALGO2_TENSOR_OP,
    CUBLAS_GEMM_ALGO3_TENSOR_OP,
    CUBLAS_GEMM_ALGO4_TENSOR_OP,
    CUBLAS_GEMM_ALGO18,
    CUBLAS_GEMM_ALGO19,
    CUBLAS_GEMM_ALGO20,
    CUBLAS_GEMM_ALGO21,
    CUBLAS_GEMM_ALGO22,
    CUBLAS_GEMM_ALGO23,
    CUBLAS_GEMM_ALGO5_TENSOR_OP,
    CUBLAS_GEMM_ALGO6_TENSOR_OP,
    CUBLAS_GEMM_ALGO7_TENSOR_OP,
    CUBLAS_GEMM_ALGO8_TENSOR_OP,
    CUBLAS_GEMM_ALGO9_TENSOR_OP,
    CUBLAS_GEMM_ALGO10_TENSOR_OP,
    CUBLAS_GEMM_ALGO11_TENSOR_OP,
    CUBLAS_GEMM_ALGO12_TENSOR_OP,
    CUBLAS_GEMM_ALGO13_TENSOR_OP,
    CUBLAS_GEMM_ALGO14_TENSOR_OP,
    CUBLAS_GEMM_ALGO15_TENSOR_OP,
};

int main(void) {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int batch_count = 4;

  __half *h_a = (__half *)malloc(sizeof(__half) * m * k * batch_count);
  __half *h_b = (__half *)malloc(sizeof(__half) * k * n * batch_count);
  __half *h_c = (__half *)malloc(sizeof(__half) * m * n * batch_count);

  for (int batch = 0; batch < batch_count; batch++) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        int index = batch * m * n + i * n + j;
        h_a[index] = index * index + 1.0f;
        h_b[index] = index + 1.0f;
        h_c[index] = 0.0;
      }
    }
  }

  __half *d_a;
  __half *d_b;
  __half *d_c;
  CHECK_CUDA(cudaMalloc(&d_a, sizeof(__half) * m * k * batch_count));
  CHECK_CUDA(cudaMalloc(&d_b, sizeof(__half) * k * n * batch_count));
  CHECK_CUDA(cudaMalloc(&d_c, sizeof(__half) * m * n * batch_count));

  CHECK_CUDA(cudaMemcpy(h_a, d_a, sizeof(__half) * m * k * batch_count,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_b, d_b, sizeof(__half) * k * n * batch_count,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_c, d_c, sizeof(__half) * m * n * batch_count,
                        cudaMemcpyDeviceToHost));

  cublasHandle_t handle;
  cublasCreate(&handle);

  // __half alpha = 1.0f;
  // __half beta = 1.0f;
  __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
  int warm_up = 25;
  int iteration = 100;

  int start_algo = CUBLAS_GEMM_DEFAULT;
  int end_algo = CUBLAS_GEMM_ALGO23;

  int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;


  for (const auto &algo : algoList) {
    for (int i = 0; i < warm_up; i++) {
      cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, CUDA_R_16F, k,
          m * k, d_b, CUDA_R_16F, n, k * n, &beta, d_c, CUDA_R_16F, n, m * n,
          batch_count, CUDA_R_16F, static_cast<cublasGemmAlgo_t>(algo));
    }
  }
  // for ( int algo = start_algo; algo <= end_algo; algo++ ) {
    float min_time = 0xffff;
    cublasGemmAlgo_t algo_index;
  for (const auto &algo : algoList) {
    float total_time = 0.0;
    for (int i = 0; i < iteration; i++) {

      cudaEvent_t start, end;
      cudaEventCreate(&start);
      cudaEventCreate(&end);

      cudaEventRecord(start, 0);
      cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, CUDA_R_16F, k,
          m * k, d_b, CUDA_R_16F, n, k * n, &beta, d_c, CUDA_R_16F, n, m * n,
          batch_count, CUDA_R_16F, static_cast<cublasGemmAlgo_t>(algo));
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      float elapsed_time;
      cudaEventElapsedTime(&elapsed_time, start, end);
      total_time += elapsed_time;
      // std::cout << "algo:" << algo << " "
      //           << "Iteration " << i + 1 << ": " << elapsed_time << " ms"
      //           << std::endl;
    }
    float current_time = total_time / iteration;
    std::cout << "algo:" << algo << " " << current_time << " ms" << std::endl;
    if( current_time < min_time ) {
      min_time = current_time;
      algo_index = algo;
    }
  }
  std::cout << "best:" << algo_index << " " << min_time << " ms" << std::endl;
  
  // float average_time = total_time / 10;
  // std::cout << "Average time: " << average_time << " ms" << std::endl;
  long long computation = 2.0 * m * n * k * batch_count;
  // // std::cout << "TFLOPs: " << computation << std::endl;
  std::cout << "BEST TFLOPS: " << computation * 1e-9 / min_time << std::endl;
  cudaMemcpy(h_c, d_c, sizeof(__half) * m * n * batch_count,
  cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}