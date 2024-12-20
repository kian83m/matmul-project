#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

const char *dgemm_desc = "Optimized cuBLAS batched GEMM.";

void batched_gemm_kernel(
    const int M,
    const int N,
    const double **A0,  // Host array of pointers to host memory
    const double **B0,  // Host array of pointers to host memory
    double **C0         // Host array of pointers to host memory
) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0;
    const double beta = 0.0;
    size_t size = M * M * sizeof(double);

    // Allocate host memory for arrays of device pointers
    double **h_A = (double**)malloc(N * sizeof(double*));
    double **h_B = (double**)malloc(N * sizeof(double*));
    double **h_C = (double**)malloc(N * sizeof(double*));

    // Allocate device memory for arrays of pointers
    double **d_A, **d_B, **d_C;
    cudaMalloc((void**)&d_A, N * sizeof(double *));
    cudaMalloc((void**)&d_B, N * sizeof(double *));
    cudaMalloc((void**)&d_C, N * sizeof(double *));

    // Allocate device memory for all matrices in a contiguous block
    double *d_A_contig, *d_B_contig, *d_C_contig;
    cudaMalloc((void**)&d_A_contig, N * size);
    cudaMalloc((void**)&d_B_contig, N * size);
    cudaMalloc((void**)&d_C_contig, N * size);

    // Set up device pointers
    for (int i = 0; i < N; ++i) {
        h_A[i] = d_A_contig + i * M * M;
        h_B[i] = d_B_contig + i * M * M;
        h_C[i] = d_C_contig + i * M * M;
    }

    // Copy the device pointers to device memory
    cudaMemcpy(d_A, h_A, N * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * sizeof(double *), cudaMemcpyHostToDevice);

    // Copy all A and B matrices individually
    for (int i = 0; i < N; ++i) {
        cudaMemcpy(d_A_contig + i * M * M, A0[i], size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_contig + i * M * M, B0[i], size, cudaMemcpyHostToDevice);
    }

    // Initialize C to zero
    cudaMemset(d_C_contig, 0, N * size);

    // Perform the batched GEMM correctly with A, then B
    cublasStatus_t status = cublasDgemmBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, M, M,
        &alpha,
        (const double**)d_B, M,
        (const double**)d_A, M,
        &beta,
        d_C, M,
        N
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasDgemmBatched failed\n");
        exit(EXIT_FAILURE);
    }

    // Copy the results back to host
    for (int i = 0; i < N; ++i) {
        cudaMemcpy(C0[i], d_C_contig + i * M * M, size, cudaMemcpyDeviceToHost);
    }

    // Clean up
    cudaFree(d_A_contig);
    cudaFree(d_B_contig);
    cudaFree(d_C_contig);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cublasDestroy(handle);
}



// #include <cstdio>
// #include <cstdlib>
// #include <cublas_v2.h>
// #include <cuda_runtime.h>

// const char *dgemm_desc = "Optimized cuBLAS batched GEMM.";

// void batched_gemm_kernel(
//     const int M,
//     const int N,
//     const double **A0,  // Host array of host pointers
//     const double **B0,  // Host array of host pointers
//     double **C0         // Host array of host pointers
// ) {
//     // Create cuBLAS handle
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     const double alpha = 1.0;
//     const double beta = 0.0;
//     size_t size = M * M * sizeof(double);

//     // Allocate pinned host memory for A, B, and C
//     double **h_A = (double**)malloc(N * sizeof(double*));
//     double **h_B = (double**)malloc(N * sizeof(double*));
//     double **h_C = (double**)malloc(N * sizeof(double*));

//     // Allocate device memory for A, B, and C
//     double **d_A, **d_B, **d_C;
//     cudaMalloc((void**)&d_A, N * sizeof(double *));
//     cudaMalloc((void**)&d_B, N * sizeof(double *));
//     cudaMalloc((void**)&d_C, N * sizeof(double *));

//     // Allocate device memory for all matrices in a contiguous block
//     double *d_A_contig, *d_B_contig, *d_C_contig;
//     cudaMalloc((void**)&d_A_contig, N * size);
//     cudaMalloc((void**)&d_B_contig, N * size);
//     cudaMalloc((void**)&d_C_contig, N * size);

//     // Set up device pointers
//     for (int i = 0; i < N; ++i) {
//         h_A[i] = d_A_contig + i * M * M;
//         h_B[i] = d_B_contig + i * M * M;
//         h_C[i] = d_C_contig + i * M * M;
//     }

//     // Copy the device pointers to device memory
//     cudaMemcpy(d_A, h_A, N * sizeof(double *), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, N * sizeof(double *), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_C, h_C, N * sizeof(double *), cudaMemcpyHostToDevice);

//     // Copy all A and B matrices in one go
//     cudaMemcpy(d_A_contig, A0[0], N * size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B_contig, B0[0], N * size, cudaMemcpyHostToDevice);
//     // Initialize C to zero
//     cudaMemset(d_C_contig, 0, N * size);

//     // Perform the batched GEMM
//     cublasStatus_t status = cublasDgemmBatched(
//         handle,
//         CUBLAS_OP_N, CUBLAS_OP_N,
//         M, M, M,
//         &alpha,
//         d_B, M,
//         d_A, M,
//         &beta,
//         d_C, M,
//         N
//     );
//     if (status != CUBLAS_STATUS_SUCCESS) {
//         fprintf(stderr, "cublasDgemmBatched failed\n");
//         exit(EXIT_FAILURE);
//     }

//     // Copy the results back to host
//     cudaMemcpy(C0[0], d_C_contig, N * size, cudaMemcpyDeviceToHost);

//     // Clean up
//     cudaFree(d_A_contig);
//     cudaFree(d_B_contig);
//     cudaFree(d_C_contig);
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
//     free(h_A);
//     free(h_B);
//     free(h_C);
//     cublasDestroy(handle);
// }
