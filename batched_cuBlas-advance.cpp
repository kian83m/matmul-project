#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

const char *dgemm_desc = "Optimized cuBLAS batched GEMM.";

void batched_gemm_kernel(
    const int M,
    const int N,
    const double **A0,  // Host array of host pointers
    const double **B0,  // Host array of host pointers
    double **C0         // Host array of host pointers
) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0;
    const double beta = 0.0;
    size_t size = M * M * sizeof(double);

    // Allocate pinned host memory for A, B, and C
    double **h_A = (double**)malloc(N * sizeof(double*));
    double **h_B = (double**)malloc(N * sizeof(double*));
    double **h_C = (double**)malloc(N * sizeof(double*));

    // Allocate device memory for A, B, and C
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

    // Copy all A and B matrices in one go
    cudaMemcpy(d_A_contig, A0[0], N * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_contig, B0[0], N * size, cudaMemcpyHostToDevice);
    // Initialize C to zero
    cudaMemset(d_C_contig, 0, N * size);

    // Perform the batched GEMM
    cublasStatus_t status = cublasDgemmBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, M, M,
        &alpha,
        d_A, M,
        d_B, M,
        &beta,
        d_C, M,
        N
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasDgemmBatched failed\n");
        exit(EXIT_FAILURE);
    }

    // Copy the results back to host
    cudaMemcpy(C0[0], d_C_contig, N * size, cudaMemcpyDeviceToHost);

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

// const char *dgemm_desc = "System cuBLAS batched ADVANCED.";

// void batched_gemm_kernel(
//     const int M,
//     const int N,
//     const double **A0,  // Host array of device pointers
//     const double **B0,  // Host array of device pointers
//     double **C0         // Host array of device pointers
// ) {
//     // Create cublas handle
//     cudaError_t err;
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     const double alpha = 1.0;
//     const double beta = 0.0;
//     size_t size = M * M * sizeof(double);

//     double **d_A, **d_B, **d_C;
//     cudaMalloc((void**)&d_A, N * sizeof(double *));
//     cudaMalloc((void**)&d_B, N * sizeof(double *));
//     cudaMalloc((void**)&d_C, N * sizeof(double *));

//     // Allocate host arrays for the device pointers
//     double *h_A[N], *h_B[N], *h_C[N];
//     for (int i = 0; i < N; ++i) {
//         // Allocate memory for each matrix on the device
//         cudaMalloc((void**)&h_A[i], size);
//         cudaMalloc((void**)&h_B[i], size);
//         cudaMalloc((void**)&h_C[i], size);

//         // Copy the data from the host to the device
//         cudaMemcpy(h_A[i], A0[i], size, cudaMemcpyHostToDevice);
//         cudaMemcpy(h_B[i], B0[i], size, cudaMemcpyHostToDevice);
//         cudaMemcpy(h_C[i], C0[i], size, cudaMemcpyHostToDevice);
//     }

//     // Copy the host arrays of device pointers to the device
//     cudaMemcpy(d_A, h_A, N * sizeof(double *), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, N * sizeof(double *), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_C, h_C, N * sizeof(double *), cudaMemcpyHostToDevice);


//     // Perform the batched GEMM
//     cublasStatus_t status;
//     status = cublasDgemmBatched(
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

//     double *C_host[N];
//     for (int i = 0; i < N; ++i) {
//         C_host[i] = (double *)malloc(size);  // Allocate host memory for each matrix
//     }

//     // Copy the device-side pointer array `d_C` to a host array
//     double *h_d_C[N];
//     cudaMemcpy(h_d_C, d_C, N * sizeof(double *), cudaMemcpyDeviceToHost);

//     // Copy each matrix from the device back to the host
//     for (int i = 0; i < N; ++i) {
//         cudaMemcpy(C_host[i], h_d_C[i], size, cudaMemcpyDeviceToHost);
//     }

//     for (int i = 0; i < N; i++){
//         C0[i] = C_host[i];
//     }



//     // Clean up
//     for (int i = 0; i < N; ++i) {
//         cudaFree(h_A[i]);
//         cudaFree(h_B[i]);
//         cudaFree(h_C[i]);
//     }

//     for (int i = 0; i < N; ++i) {
//         cudaFree(h_d_C[i]);
//     }
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     cublasDestroy(handle);
// }
