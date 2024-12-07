const char *dgemm_desc = "Our fancy blazing fast cornell university blas gpu dgemm";

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define BLOCKSIZE 32
#define CEIL_DIV(a, b) ((a + b - 1) / b)

__global__
void square_dgemm_kernel(const int M, const double *A0, const double *B0, double *C0)
{
    // threads with neighboring threadIdx access same row in A and neighboring cols in B
    const int x = threadIdx.x / BLOCKSIZE + BLOCKSIZE * blockIdx.x;
    const int y = threadIdx.x % BLOCKSIZE + BLOCKSIZE * blockIdx.y;

    if(x < M && y < M){
        // row major order version
        double sum = 0.0;
        // calculating C[x][y] as inner product of row x of A and col y of B (all zero-indexing)
        for(int k = 0; k < M; k++){
            // C0[x * M + y] += A0[x * M + k] * B0[y + k * M];
            sum += A0[x * M + k] * B0[k * M + y];
        }
        C0[x * M + y] = sum;
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

    dim3 numBlocks(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
    dim3 threadsPerBlock(BLOCKSIZE * BLOCKSIZE);
    square_dgemm_kernel<<<numBlocks, threadsPerBlock>>>(M, A, B, C);

    cudaDeviceSynchronize();
    cudaMemcpy(C0, C, M * M * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}