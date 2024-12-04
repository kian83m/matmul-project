const char *dgemm_desc = "Our fancy basic gpu dgemm";

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define blocksize 16
#define CEIL_DIV(a, b) ((a + b - 1) / b)

__global__
void square_dgemm_kernel(const int M, const double *A0, const double *B0, double *C0)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    // C[x][y] = 0;
    if(x < M && y < M){
        // C0[x * M + y] = 0.0;
        // // calculating C[x][y] as inner product of row x of A and col y of B (all zero-indexing)
        // for(int k = 0; k < M; k++){
        //     C0[x * M + y] += A0[x * M + k] * B0[y + k * M];
        // }
        C0[y * M + x] = 0.0;
        // calculating C[x][y] as inner product of row x of A and col y of B (all zero-indexing)
        for(int k = 0; k < M; k++){
            C0[y * M + x] += A0[k * M + x] * B0[k + y * M];
        }
    }
}

double *A, *B, *C;

void square_dgemm(const int M, const double *A0, const double *B0, double *C0)
{
    cudaMalloc(&A, M * M * sizeof(double));
    cudaMalloc(&B, M * M * sizeof(double));
    cudaMalloc(&C, M * M * sizeof(double));
    cudaMemcpy(A, A0, M * M * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B, B0, M * M * sizeof(double), cudaMemcpyHostToDevice);
    dim3 numBlocks(CEIL_DIV(M, blocksize), CEIL_DIV(M, blocksize));
    dim3 threadsPerBlock(blocksize, blocksize);
    square_dgemm_kernel<<<numBlocks, threadsPerBlock>>>(M, A, B, C);
    cudaDeviceSynchronize();
    cudaMemcpy(C0, C, M * M * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}