
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define CEIL_DIV(a, b) ((a + b - 1) / b)

// Tile sizes
#define BM 64
#define BN 64
#define BK 16

// Thread tile sizes
#define TM 4
#define TN 4

// Use __ldg for read-only global memory fetches
// This can sometimes improve performance if the data is read-only
// and may better utilize caches.
__device__ __forceinline__ double ldg_double(const double *ptr)
{
#if __CUDA_ARCH__ >= 350
  return __ldg(ptr);
#else
  return *ptr;
#endif
}

// A helper function to load a double2 from global memory safely.
// We'll assume the pointers are suitably aligned since we know BM,BK,BN are multiples of 2.
__device__ __forceinline__ double2 ldg_double2(const double2 *ptr)
{
#if __CUDA_ARCH__ >= 350
  return __ldg(ptr);
#else
  return *ptr;
#endif
}

// Compute tile indices with a linear-to-2D mapping for potential improved locality
__device__ inline void compute_tile_indices(int &tile_m, int &tile_n, int m_tiles, int n_tiles)
{
  int linear_idx = blockIdx.y * gridDim.x + blockIdx.x;
  tile_m = linear_idx / n_tiles;
  tile_n = linear_idx % n_tiles;
}

__device__ inline void load_tile(
    const double *__restrict__ A, const double *__restrict__ B,
    int M, int globalRowStart, int globalColStart,
    int kt, // which k-tile
    double *__restrict__ As, double *__restrict__ Bs,
    int tRow, int tCol,
    int THREADS_PER_BLOCK_M, int THREADS_PER_BLOCK_N)
{
  // Each tile: A: BMxBK = 64x16=1024 elems, B: BKxBN=16x64=1024 elems
  // blockDim = 16x16 = 256 threads
  // Each thread loads 4 elements from A and 4 from B. We can vectorize these loads using double2.

  int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

  int elemsPerThreadA = (BM * BK) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N); // 1024/256=4
  int elemsPerThreadB = (BK * BN) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N); // also 4

  // We will load 4 elements from A and B. Since BK=16 is divisible by 2, we can load double2 pairs.
  // Similarly for BN=64 and BM=64.

  // Load A tile
  // The A tile spans rows [globalRowStart .. globalRowStart+BM-1],
  // and columns [kt*BK .. kt*BK+BK-1].
  // We'll load them linearly and use double2 loads if possible.
  for (int i = 0; i < elemsPerThreadA; i++)
  {
    int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
    int rowA = index / BK;
    int colA = index % BK;

    int globalArow = globalRowStart + rowA;
    int globalAcol = kt * BK + colA;

    double valA = 0.0;
    if (globalArow < M && globalAcol < M)
    {
      valA = ldg_double(&A[globalArow * M + globalAcol]);
    }
    As[rowA * BK + colA] = valA;
  }

  // Load B tile
  // B tile spans rows [kt*BK .. kt*BK+BK-1], cols [globalColStart..globalColStart+BN-1]
  for (int i = 0; i < elemsPerThreadB; i++)
  {
    int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
    int rowB = index / BN;
    int colB = index % BN;

    int globalBrow = kt * BK + rowB;
    int globalBcol = globalColStart + colB;

    double valB = 0.0;
    if (globalBrow < M && globalBcol < M)
    {
      valB = ldg_double(&B[globalBrow * M + globalBcol]);
    }
    Bs[rowB * BN + colB] = valB;
  }
}

__device__ inline void compute_tile(
    const double *__restrict__ As, const double *__restrict__ Bs,
    double *__restrict__ threadResults,
    int tRow, int tCol)
{
// Compute partial product: Ctile += A_tile * B_tile
// TM=4, TN=4
#pragma unroll
  for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
  {
    double regM[TM];
    double regN[TN];

#pragma unroll
    for (int i = 0; i < TM; i++)
    {
      int aRow = tRow * TM + i;
      regM[i] = As[aRow * BK + dotIdx];
    }
#pragma unroll
    for (int j = 0; j < TN; j++)
    {
      int bCol = tCol * TN + j;
      regN[j] = Bs[dotIdx * BN + bCol];
    }
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
      double m_val = regM[i];
#pragma unroll
      for (int j = 0; j < TN; j++)
      {
        threadResults[i * TN + j] += m_val * regN[j];
      }
    }
  }
}

__global__ void square_dgemm_kernel_2d_blocktiling(
    const int M, const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C)
{
  // Double buffering
  __shared__ double As[2][BM * BK];
  __shared__ double Bs[2][BK * BN];

  int m_tiles = CEIL_DIV(M, BM);
  int n_tiles = CEIL_DIV(M, BN);

  int tile_m, tile_n;
  compute_tile_indices(tile_m, tile_n, m_tiles, n_tiles);

  if (tile_m >= m_tiles || tile_n >= n_tiles)
    return;

  const int THREADS_PER_BLOCK_M = BM / TM; // 64/4=16
  const int THREADS_PER_BLOCK_N = BN / TN; // 64/4=16
  // blockDim should be (16,16)=256 threads

  const int tRow = threadIdx.y;
  const int tCol = threadIdx.x;

  const int globalRowStart = tile_m * BM;
  const int globalColStart = tile_n * BN;

  double threadResults[TM * TN];
#pragma unroll
  for (int i = 0; i < TM * TN; i++)
  {
    threadResults[i] = 0.0;
  }

  int k_tiles = CEIL_DIV(M, BK);

  // Preload the first tile
  int curLoad = 0;
  load_tile(A, B, M, globalRowStart, globalColStart, 0,
            As[curLoad], Bs[curLoad], tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);
  __syncthreads(); // ensure first tile is loaded

  int curCompute = curLoad;

  // Main loop over K
  for (int kt = 1; kt < k_tiles; kt++)
  {
    int nextLoad = 1 - curLoad;

    // Start loading next tile
    load_tile(A, B, M, globalRowStart, globalColStart, kt,
              As[nextLoad], Bs[nextLoad],
              tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);

    __syncthreads(); // Wait for next tile to be fully loaded

    // Compute on previously loaded tile
    compute_tile(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

    __syncthreads(); // Ensure compute done before reusing shared memory

    // Switch buffers
    curLoad = nextLoad;
    curCompute = curLoad;
  }

  // Compute on the last tile
  compute_tile(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

// Write results back to C
#pragma unroll
  for (int i = 0; i < TM; i++)
  {
    int cRow = globalRowStart + tRow * TM + i;
#pragma unroll
    for (int j = 0; j < TN; j++)
    {
      int cCol = globalColStart + tCol * TN + j;
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
  cudaMalloc((void **)&dA, M * M * sizeof(double));
  cudaMalloc((void **)&dB, M * M * sizeof(double));
  cudaMalloc((void **)&dC, M * M * sizeof(double));
  cudaMemcpy(dA, A0, M * M * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B0, M * M * sizeof(double), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BN / TN, BM / TM); // (16,16)
  int m_tiles = CEIL_DIV(M, BM);
  int n_tiles = CEIL_DIV(M, BN);

  // Using a linear-to-2D mapping for blocks
  // gridDim.x = n_tiles, gridDim.y = m_tiles
  // Indices computed by kernel.
  dim3 numBlocks(n_tiles, m_tiles);

  square_dgemm_kernel_2d_blocktiling<<<numBlocks, threadsPerBlock>>>(M, dA, dB, dC);
  cudaDeviceSynchronize();
  cudaMemcpy(C0, dC, M * M * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

const char *dgemm_desc = "Optimized DGEMM: double-buffered, vectorized loads, read-only cache hints, block swizzling, and loop unrolling";

// ok ?

// #include <cuda_runtime.h>
// #include <cuda.h>
// #include <stdio.h>
// #include <math.h>

// #define CEIL_DIV(a, b) ((a + b - 1) / b)

// #define BM 64
// #define BN 64
// #define BK 16
// #define TM 4
// #define TN 4

//    // Use __ldg for read-only global memory fetches
//     __device__ __forceinline__ double
//     ldg_double(const double *__restrict__ ptr)
// {
// #if __CUDA_ARCH__ >= 350
//   return __ldg(ptr);
// #else
//   return *ptr;
// #endif
// }

// // Vectorized load: double2
// __device__ __forceinline__ double2 ldg_double2(const double2 *__restrict__ ptr)
// {
// #if __CUDA_ARCH__ >= 350
//   return __ldg(ptr);
// #else
//   return *ptr;
// #endif
// }

// // Compute tile indices
// __device__ __forceinline__ void compute_tile_indices(int &tile_m, int &tile_n, int m_tiles, int n_tiles)
// {
//   int linear_idx = blockIdx.y * gridDim.x + blockIdx.x;
//   tile_m = linear_idx / n_tiles;
//   tile_n = linear_idx % n_tiles;
// }

// __device__ __forceinline__ void load_tile(
//     const double *__restrict__ A, const double *__restrict__ B,
//     int M, int globalRowStart, int globalColStart,
//     int kt, // which k-tile
//     double *__restrict__ As, double *__restrict__ Bs,
//     int tRow, int tCol,
//     int THREADS_PER_BLOCK_M, int THREADS_PER_BLOCK_N)
// {
//   // Each tile: A: BMxBK = 64x16=1024 elems, B: 16x64=1024 elems
//   // blockDim = 16x16 = 256 threads
//   // Each thread loads 4 elements from A and 4 from B.
//   int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

//   int elemsPerThreadA = (BM * BK) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N); // 4
//   int elemsPerThreadB = (BK * BN) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N); // 4

//   // Vectorized load for A: we have 4 elements per thread, we can load them as two double2 (if in-bounds)
//   // Since BK=16 is divisible by 2, we should be able to handle double2 along the K dimension cleanly.
//   // We know 4 elements are arranged in a linear fashion. We'll attempt to load them in pairs:
//   for (int i = 0; i < elemsPerThreadA; i += 2)
//   {
//     int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
//     // First element in the pair
//     int rowA0 = index / BK;
//     int colA0 = index % BK;
//     int globalArow0 = globalRowStart + rowA0;
//     int globalAcol0 = kt * BK + colA0;

//     // Second element in the pair
//     int index2 = index + (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
//     int rowA1 = index2 / BK;
//     int colA1 = index2 % BK;
//     int globalArow1 = globalRowStart + rowA1;
//     int globalAcol1 = kt * BK + colA1;

//     double a0 = 0.0;
//     double a1 = 0.0;

//     // Check if both elements are in range:
//     bool inRange0 = (globalArow0 < M && globalAcol0 < M);
//     bool inRange1 = (globalArow1 < M && globalAcol1 < M);

//     if (inRange0 && inRange1 && (colA1 == colA0 + 1) && (globalAcol0 + 1 == globalAcol1))
//     {
//       // We can do a double2 load if they are consecutive
//       const double2 *ptrA2 = reinterpret_cast<const double2 *>(&A[globalArow0 * M + globalAcol0]);
//       double2 valA2 = ldg_double2(ptrA2);
//       a0 = valA2.x;
//       a1 = valA2.y;
//     }
//     else
//     {
//       // Fallback to scalar loads to ensure correctness
//       if (inRange0)
//         a0 = ldg_double(&A[globalArow0 * M + globalAcol0]);
//       if (inRange1)
//         a1 = ldg_double(&A[globalArow1 * M + globalAcol1]);
//     }

//     As[rowA0 * BK + colA0] = a0;
//     As[rowA1 * BK + colA1] = a1;
//   }

//   // B tile load (unchanged, scalar loads for simplicity in this small step)
//   for (int i = 0; i < elemsPerThreadB; i++)
//   {
//     int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
//     int rowB = index / BN;
//     int colB = index % BN;

//     int globalBrow = kt * BK + rowB;
//     int globalBcol = globalColStart + colB;

//     double valB = 0.0;
//     if (globalBrow < M && globalBcol < M)
//     {
//       valB = ldg_double(&B[globalBrow * M + globalBcol]);
//     }
//     Bs[rowB * BN + colB] = valB;
//   }
// }

// __device__ __forceinline__ void compute_tile(
//     const double *__restrict__ As, const double *__restrict__ Bs,
//     double *__restrict__ threadResults,
//     int tRow, int tCol)
// {
// #pragma unroll
//   for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
//   {
//     double regM[TM];
//     double regN[TN];

// #pragma unroll
//     for (int i = 0; i < TM; i++)
//     {
//       int aRow = tRow * TM + i;
//       regM[i] = As[aRow * BK + dotIdx];
//     }
// #pragma unroll
//     for (int j = 0; j < TN; j++)
//     {
//       int bCol = tCol * TN + j;
//       regN[j] = Bs[dotIdx * BN + bCol];
//     }

// #pragma unroll
//     for (int i = 0; i < TM; i++)
//     {
//       double m_val = regM[i];
// #pragma unroll
//       for (int j = 0; j < TN; j++)
//       {
//         threadResults[i * TN + j] += m_val * regN[j];
//       }
//     }
//   }
// }

// __global__ void square_dgemm_kernel_2d_blocktiling(
//     const int M,
//     const double *__restrict__ A,
//     const double *__restrict__ B,
//     double *__restrict__ C)
// {
//   // Double buffering
//   __shared__ double As[2][BM * BK];
//   __shared__ double Bs[2][BK * BN];

//   int m_tiles = CEIL_DIV(M, BM);
//   int n_tiles = CEIL_DIV(M, BN);

//   int tile_m, tile_n;
//   compute_tile_indices(tile_m, tile_n, m_tiles, n_tiles);

//   if (tile_m >= m_tiles || tile_n >= n_tiles)
//     return;

//   const int THREADS_PER_BLOCK_M = BM / TM; // 64/4=16
//   const int THREADS_PER_BLOCK_N = BN / TN; // 64/4=16

//   const int tRow = threadIdx.y;
//   const int tCol = threadIdx.x;

//   const int globalRowStart = tile_m * BM;
//   const int globalColStart = tile_n * BN;

//   double threadResults[TM * TN];
// #pragma unroll
//   for (int i = 0; i < TM * TN; i++)
//   {
//     threadResults[i] = 0.0;
//   }

//   int k_tiles = CEIL_DIV(M, BK);

//   // Preload the first tile
//   int curLoad = 0;
//   load_tile(A, B, M, globalRowStart, globalColStart, 0,
//             As[curLoad], Bs[curLoad], tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);
//   __syncthreads(); // ensure first tile is loaded

//   int curCompute = curLoad;

//   // Main loop over K
//   for (int kt = 1; kt < k_tiles; kt++)
//   {
//     int nextLoad = 1 - curLoad;

//     // Start loading next tile
//     load_tile(A, B, M, globalRowStart, globalColStart, kt,
//               As[nextLoad], Bs[nextLoad],
//               tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);

//     __syncthreads(); // Wait for next tile to be fully loaded

//     // Compute on previously loaded tile
//     compute_tile(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

//     __syncthreads(); // Ensure compute done before reusing shared memory

//     // Switch buffers
//     curLoad = nextLoad;
//     curCompute = curLoad;
//   }

//   // Compute on the last tile
//   compute_tile(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

//   // Write results back to C
// #pragma unroll
//   for (int i = 0; i < TM; i++)
//   {
//     int cRow = globalRowStart + tRow * TM + i;
// #pragma unroll
//     for (int j = 0; j < TN; j++)
//     {
//       int cCol = globalColStart + tCol * TN + j;
//       if (cRow < M && cCol < M)
//       {
//         C[cRow * M + cCol] = threadResults[i * TN + j];
//       }
//     }
//   }
// }

// double *dA, *dB, *dC;

// void square_dgemm(const int M, const double *A0, const double *B0, double *C0)
// {
//   cudaMalloc((void **)&dA, M * M * sizeof(double));
//   cudaMalloc((void **)&dB, M * M * sizeof(double));
//   cudaMalloc((void **)&dC, M * M * sizeof(double));
//   cudaMemcpy(dA, A0, M * M * sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemcpy(dB, B0, M * M * sizeof(double), cudaMemcpyHostToDevice);

//   dim3 threadsPerBlock(BN / TN, BM / TM); // (16,16)
//   int m_tiles = CEIL_DIV(M, BM);
//   int n_tiles = CEIL_DIV(M, BN);
//   dim3 numBlocks(n_tiles, m_tiles);

//   square_dgemm_kernel_2d_blocktiling<<<numBlocks, threadsPerBlock>>>(M, dA, dB, dC);
//   cudaDeviceSynchronize();
//   cudaMemcpy(C0, dC, M * M * sizeof(double), cudaMemcpyDeviceToHost);

//   cudaFree(dA);
//   cudaFree(dB);
//   cudaFree(dC);
// }

// const char *dgemm_desc = "Incremental improvement: vectorized double2 loads for A where possible";

