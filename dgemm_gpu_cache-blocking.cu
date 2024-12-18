const char *dgemm_desc = "Our fancy blazing fast cornell university blas gpu dgemm";

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define BLOCKSIZE 32
#define CEIL_DIV(a, b) ((a + b - 1) / b)

__global__ void square_dgemm_kernel(const int M, const double *A0, const double *B0, double *C0)
{
    // For each thread block, we can fit tiles of size 32 by 32 of A, B
    // (16KB in total) to the shared memory
    // Can fit 164/17 = 9 blocks on each SM
    // No need to store a tile of C!
    // Each thread can hold the cumulative sum as local variable
    // and update C0 once after all computation.
    // The only tunable parameter is BLOCKSIZE for now
    __shared__ double tile_A[BLOCKSIZE][BLOCKSIZE];
    __shared__ double tile_B[BLOCKSIZE][BLOCKSIZE];
    // __shared__ double tile_C [BLOCKSIZE][BLOCKSIZE];

    // threadIdx ranges from 0 to BLOCKSIZE^2 - 1,
    // so threadRow and threadCol ranges from 0 to BLOCKSIZE - 1
    // and threads of consecutive threadIdx.x have same threadRow and consecutive threadCol
    const int threadRow = threadIdx.x / BLOCKSIZE;
    const int threadCol = threadIdx.x % BLOCKSIZE;

    // blockRow, blockCol ranges from 0 to ceil(M / BLOCKSIZE) - 1
    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;

    // (cRow, cCol) is the element this thread is responsible for in C0.
    const int cRow = blockRow * BLOCKSIZE + threadRow;
    const int cCol = blockCol * BLOCKSIZE + threadCol;

    // each thread setting an element to 0 in C, then __syncthreads(); row major order
    //  tile_C[threadRow][threadCol] = 0;
    //  __syncthreads();

    double tmp = 0.0;

    for (int tileIdx = 0; tileIdx < M; tileIdx += BLOCKSIZE)
    {
        // each thread loads an element from A, B
        // for the incomplete final tiles, load first few rows of A and first few columns of B
        // each thread loads A[cRow][tileIdx + threadCol] and B[tileIdx + threadRow][cCol]
        double aVal = 0.0, bVal = 0.0;
        // Load A if in range
        if (cRow < M && (tileIdx + threadCol) < M)
        {
            aVal = A0[cRow * M + (tileIdx + threadCol)];
        }
        // Load B if in range, not memory coalesced indexing
        if (cCol < M && (tileIdx + threadRow) < M)
        {
            bVal = B0[(tileIdx + threadRow) * M + cCol];
        }
        tile_A[threadRow][threadCol] = aVal;
        tile_B[threadRow][threadCol] = bVal;
        // Synchronize to ensure all threads have loaded and tile is fully updated
        __syncthreads();
        // C[cRow][cCol] = \sum A[cRow][:] * B[:][cCol]
        // In each tile, each thread computes
        // C[cRow][cCol] += \sum A[cRow][tileIdx : tileIdx + BLOCKSIZE - 1] *
        // B[tileIdx : tileIdx + BLOCKSIZE - 1][cCol]
        for (int k = 0; k < BLOCKSIZE; k++)
        {
            tmp += tile_A[threadRow][k] * tile_B[k][threadCol];
        }
        // corresponds to C0[cRow][cCol]
        // tile_C[threadRow][threadCol] += tmp;

        // Synchronize before loading the next tile
        __syncthreads();
    }
    // __syncthreads();
    // C0[cRow * M + cCol] = tile_C[threadRow][threadCol];
    // Write the result back to C if within bounds
    if (cRow < M && cCol < M)
    {
        C0[cRow * M + cCol] = tmp;
    }
    // // threads with neighboring threadIdx access same row in A and neighboring cols in B
    // const int x = threadIdx.x / BLOCKSIZE + BLOCKSIZE * blockIdx.x;
    // const int y = threadIdx.x % BLOCKSIZE + BLOCKSIZE * blockIdx.y;

    // if(x < M && y < M){
    //     double sum = 0.0;
    //     // calculating C[x][y] as inner product of row x of A and col y of B (all zero-indexing)
    //     for(int k = 0; k < M; k++){
    //         // row major order
    //         sum += A0[x * M + k] * B0[k * M + y];
    //     }
    //     C0[x * M + y] = sum;
    // }
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