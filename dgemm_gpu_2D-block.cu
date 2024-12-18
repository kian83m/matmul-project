#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define CEIL_DIV(a, b) ((a + b - 1) / b)

// Tile sizes
#define BM 64 // Block size in M dimension
#define BN 64 // Block size in N dimension
#define BK 16 // Block size in K dimension

// Thread tile sizes (each thread computes an 8x8 tile of C)
#define TM 4
#define TN 4

// The kernel computes C = A * B for MxM matrices.
__global__ void square_dgemm_kernel_2d_blocktiling(const int M, const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C)
{
  // Each block computes a BMxBN tile of C
  // Each thread computes a TMxTN tile of C

  __shared__ double As[BM * BK]; // Shared memory for A tile
  __shared__ double Bs[BK * BN]; // Shared memory for B tile

  // Determine which block of the output C we're working on
  const int blockRow = blockIdx.x; // Along M dimension
  const int blockCol = blockIdx.y; // Along N dimension

  // Number of threads in each dimension of the threadblock grid:
  // Each block is (BM/TM) by (BN/TN) threads, i.e. each dimension: BM/TM and BN/TN
  // For BM=64, BN=64, TM=8, TN=8 => 8x8=64 threads per block
  const int THREADS_PER_BLOCK_M = BM / TM;
  const int THREADS_PER_BLOCK_N = BN / TN;

  // 2D thread index within the block
  const int tRow = threadIdx.y; // 0 to THREADS_PER_BLOCK_M-1
  const int tCol = threadIdx.x; // 0 to THREADS_PER_BLOCK_N-1

  // Global starting row and column for the sub-block of C this block computes
  const int globalRowStart = blockRow * BM;
  const int globalColStart = blockCol * BN;

  // Each thread computes a TMxTN tile:
  // The top-left corner of this thread's tile in the block space:
  const int threadBaseRow = tRow * TM; // local row offset within the block
  const int threadBaseCol = tCol * TN; // local col offset within the block

  // If we consider indexing into C:
  // C-sub-block rows: [globalRowStart .. globalRowStart+BM-1]
  // C-sub-block cols: [globalColStart .. globalColStart+BN-1]

  // Initialize a register cache for the partial results
  double threadResults[TM * TN] = {0.0};

  // We loop over K in steps of BK
  // We will move along the K dimension: 0, BK, 2*BK, ...
  for (int kBlock = 0; kBlock < M; kBlock += BK)
  {
    // Load the A tile of size BMxBK from global memory to shared memory
    // Each block of threads (BMxBN) is handled by (BM/TM * BN/TN) threads.
    // We must have each thread load multiple elements.

    // Total elements in A tile: BM * BK
    // Total elements in B tile: BK * BN
    // Each thread should load (BM*BK)/(THREADS_PER_BLOCK_M*THREADS_PER_BLOCK_N)
    // and similarly for B.

    // Flatten thread index within the block of threads:
    int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

    // Load A tile (BMxBK)
    {
      int elemsPerThreadA = (BM * BK) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
      for (int i = 0; i < elemsPerThreadA; i++)
      {
        int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
        if (index < BM * BK)
        {
          int rowA = index / BK; // which row in the tile
          int colA = index % BK; // which col in the tile
          int globalArow = globalRowStart + rowA;
          int globalAcol = kBlock + colA;
          double valA = 0.0;
          if (globalArow < M && globalAcol < M)
          {
            valA = A[globalArow * M + globalAcol];
          }
          As[rowA * BK + colA] = valA;
        }
      }
    }

    // Load B tile (BKxBN)
    {
      int elemsPerThreadB = (BK * BN) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
      for (int i = 0; i < elemsPerThreadB; i++)
      {
        int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
        if (index < BK * BN)
        {
          int rowB = index / BN;
          int colB = index % BN;
          int globalBrow = kBlock + rowB;
          int globalBcol = globalColStart + colB;
          double valB = 0.0;
          if (globalBrow < M && globalBcol < M)
          {
            valB = B[globalBrow * M + globalBcol];
          }
          Bs[rowB * BN + colB] = valB;
        }
      }
    }

    __syncthreads();

    // Now, perform the multiplication for the BK dimension
    // For each dotIdx in [0..BK-1], we do:
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
    {
      // Load the necessary elements from As and Bs into registers
      double regM[TM];
      double regN[TN];

      // Load a column of length TM from As for this thread
      for (int i = 0; i < TM; i++)
      {
        int aRow = threadBaseRow + i;
        regM[i] = As[aRow * BK + dotIdx];
      }

      // Load a row of length TN from Bs for this thread
      for (int j = 0; j < TN; j++)
      {
        int bCol = threadBaseCol + j;
        regN[j] = Bs[dotIdx * BN + bCol];
      }

      // Compute outer product and accumulate into threadResults
      for (int i = 0; i < TM; i++)
      {
        for (int j = 0; j < TN; j++)
        {
          threadResults[i * TN + j] += regM[i] * regN[j];
        }
      }
    }

    __syncthreads();
  }

  // Write results back to C
  for (int i = 0; i < TM; i++)
  {
    int cRow = globalRowStart + threadBaseRow + i;
    for (int j = 0; j < TN; j++)
    {
      int cCol = globalColStart + threadBaseCol + j;
      if (cRow < M && cCol < M)
      {
        C[cRow * M + cCol] = threadResults[i * TN + j];
      }
    }
  }
}

double *dA, *dB, *dC;

void square_dgemm(const int M, const double *A0, const double *B0, double *C0)
{
  cudaMalloc(&dA, M * M * sizeof(double));
  cudaMalloc(&dB, M * M * sizeof(double));
  cudaMalloc(&dC, M * M * sizeof(double));
  cudaMemcpy(dA, A0, M * M * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B0, M * M * sizeof(double), cudaMemcpyHostToDevice);

  // Each block computes a BMxBN tile. The grid covers MxM
  dim3 numBlocks(CEIL_DIV(M, BM), CEIL_DIV(M, BN));
  // Each block has (BM/TM) by (BN/TN) threads
  dim3 threadsPerBlock(BN / TN, BM / TM);

  square_dgemm_kernel_2d_blocktiling<<<numBlocks, threadsPerBlock>>>(M, dA, dB, dC);
  cudaDeviceSynchronize();
  cudaMemcpy(C0, dC, M * M * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

const char *dgemm_desc = "Optimized DGEMM with 2D blocktiling (Kernel 5)";
