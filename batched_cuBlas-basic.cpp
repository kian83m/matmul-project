#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>


const char *dgemm_desc = "System cuBLAS batched BASIC.";

void batched_gemm_kernel( 
    const int M, 
    const int N, 
    const double **A0, 
    const double **B0, 
    double **C0
)
{
    // Create a handle for cuBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return;
    }

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    size_t size = M * M * sizeof(double);
    
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);


    for (int i = 0; i < N; i++){
        cudaMemcpy(d_A, A0[i], size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B0[i], size, cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, size);


        // Perform matrix multiplication using cuBLAS
        const double alpha = 1.0;
        const double beta = 0.0;

        status = cublasDgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            M, M, M,
                            &alpha,
                            d_B, M,
                            d_A, M,
                            &beta,
                            d_C, M);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            fprintf(stderr, "CUBLAS matrix multiplication failed\n");
            return;
        }

        cudaMemcpy(C0[i], d_C, size, cudaMemcpyDeviceToHost);
    }

    // Free device memory and destroy the handle
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}