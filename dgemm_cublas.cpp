
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>


const char *dgemm_desc = "System cuBLAS dgemm.";

void square_dgemm(int M, const double *A, const double *B, double *C)
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
    if (cudaMalloc((void **)&d_A, size) != cudaSuccess ||
        cudaMalloc((void **)&d_B, size) != cudaSuccess ||
        cudaMalloc((void **)&d_C, size) != cudaSuccess)
    {
        fprintf(stderr, "Device memory allocation failed\n");
        cublasDestroy(handle);
        return;
    }

    // Copy matrices from host to device memory
    if (cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "Host to device memory copy failed\n");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        return;
    }

    // Perform matrix multiplication using cuBLAS
    const double alpha = 1.0;
    const double beta = 0.0;

    // Since matrices are in row-major order, the program sees A^T, B^T, and stores C^T. 
    // Hence, switch the sequence of d_A and d_B as input such that B^T * A^T = C^T
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
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        return;
    }

    // Copy the result from device to host memory
    if (cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "Device to host memory copy failed\n");
    }

    // Free device memory and destroy the handle
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}