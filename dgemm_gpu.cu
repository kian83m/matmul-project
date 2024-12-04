const char *dgemm_desc = "Our fancy basic gpu dgemm";

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define blocksize 16
#define CEIL_DIV(a, b) ((a + b - 1) / b)

__global__
void square_dgemm_kernel(const int M, const double *A, const double *B, double *C)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    // C[x][y] = 0;
    if(x < M && y < M){
        C[x * M + y] = 0.0;
        // calculating C[x][y] as inner product of row x of A and col y of B (all zero-indexing)
        for(int k = 0; k < M; k++){
            C[x * M + y] += A[x * M + k] * B[y + k * M];
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    dim3 numBlocks(CEIL_DIV(M, blocksize), CEIL_DIV(M, blocksize));
    dim3 threadsPerBlock(blocksize, blocksize);
    square_dgemm_kernel<<<numBlocks, threadsPerBlock>>>(M, A, B, C);
}