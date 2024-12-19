// #include <cuda_runtime.h>
// #include <cuda.h>
// #include <stdio.h>
// #include <math.h>

// #define CEIL_DIV(a, b) ((a + b - 1) / b)

// // Tile sizes
// #define BM 64 // Block size in M dimension
// #define BN 64 // Block size in N dimension
// #define BK 16 // Block size in K dimension

// // Thread tile sizes (each thread computes a 4x4 tile of C)
// #define TM 4
// #define TN 4

// __global__ void square_dgemm_kernel_2d_blocktiling(const int M,
//                                                    const double *__restrict__ A,
//                                                    const double *__restrict__ B,
//                                                    double *__restrict__ C)
// {
//   __shared__ double As[BM * BK]; // Shared memory for A tile
//   __shared__ double Bs[BK * BN]; // Shared memory for B tile

//   const int blockRow = blockIdx.x; // Along M dimension
//   const int blockCol = blockIdx.y; // Along N dimension

//   const int THREADS_PER_BLOCK_M = BM / TM;
//   const int THREADS_PER_BLOCK_N = BN / TN;

//   const int tRow = threadIdx.y;
//   const int tCol = threadIdx.x;

//   const int globalRowStart = blockRow * BM;
//   const int globalColStart = blockCol * BN;

//   const int threadBaseRow = tRow * TM;
//   const int threadBaseCol = tCol * TN;

//   double threadResults[TM * TN] = {0.0};

//   // Vectorized loads: each double2 loads two doubles.
//   // Total threads per block:
//   int totalThreads = THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N;
//   int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

//   // For A (BM x BK):
//   // Total elements = BM*BK. double2 load covers 2 elements.
//   int aVectorLoads = (BM * BK) / 2 / totalThreads;
//   // For B (BK x BN):
//   int bVectorLoads = (BK * BN) / 2 / totalThreads;

//   for (int kBlock = 0; kBlock < M; kBlock += BK)
//   {
//     // We define tile pointers for convenience
//     const double *A_tile = &A[globalRowStart * M + kBlock];
//     const double *B_tile = &B[kBlock * M + globalColStart];

//     {
//       double2 *AsVec = (double2 *)As;
//       for (int i = 0; i < aVectorLoads; i++)
//       {
//         int vecIndex = linearThreadIdx + i * totalThreads;
//         int elemIndex = vecIndex * 2;
//         if (elemIndex < BM * BK)
//         {
//           int rowA = elemIndex / BK;
//           int colA = elemIndex % BK;
//           int globalArow = globalRowStart + rowA;
//           int globalAcol = kBlock + colA;

//           double val0 = 0.0, val1 = 0.0;
//           // Check if both elements fit in range
//           if (globalArow < M && globalAcol < M)
//             val0 = A[globalArow * M + globalAcol];

//           // Second element of the pair
//           int nextColA = globalAcol + 1;
//           if ((colA + 1 < BK) && globalArow < M && nextColA < M)
//             val1 = A[globalArow * M + nextColA];

//           AsVec[vecIndex] = make_double2(val0, val1);
//         }
//       }
//     }

//     {
//       double2 *BsVec = (double2 *)Bs;
//       for (int i = 0; i < bVectorLoads; i++)
//       {
//         int vecIndex = linearThreadIdx + i * totalThreads;
//         int elemIndex = vecIndex * 2;
//         if (elemIndex < BK * BN)
//         {
//           int rowB = elemIndex / BN;
//           int colB = elemIndex % BN;
//           int globalBrow = kBlock + rowB;
//           int globalBcol = globalColStart + colB;

//           double val0 = 0.0, val1 = 0.0;
//           if (globalBrow < M && globalBcol < M)
//             val0 = B[globalBrow * M + globalBcol];

//           int nextColB = globalBcol + 1;
//           if ((colB + 1 < BN) && globalBrow < M && nextColB < M)
//             val1 = B[globalBrow * M + nextColB];

//           BsVec[vecIndex] = make_double2(val0, val1);
//         }
//       }
//     }

//     __syncthreads();

//     // Compute partial results for this K-block
//     for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
//     {
//       double regM[TM];
//       double regN[TN];

//       for (int i = 0; i < TM; i++)
//       {
//         int aRow = threadBaseRow + i;
//         regM[i] = As[aRow * BK + dotIdx];
//       }

//       for (int j = 0; j < TN; j++)
//       {
//         int bCol = threadBaseCol + j;
//         regN[j] = Bs[dotIdx * BN + bCol];
//       }

//       for (int i = 0; i < TM; i++)
//       {
//         for (int j = 0; j < TN; j++)
//         {
//           threadResults[i * TN + j] += regM[i] * regN[j];
//         }
//       }
//     }

//     __syncthreads();
//   }

//   // Store results to C
//   for (int i = 0; i < TM; i++)
//   {
//     int cRow = globalRowStart + threadBaseRow + i;
//     for (int j = 0; j < TN; j++)
//     {
//       int cCol = globalColStart + threadBaseCol + j;
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
//   cudaMalloc(&dA, M * M * sizeof(double));
//   cudaMalloc(&dB, M * M * sizeof(double));
//   cudaMalloc(&dC, M * M * sizeof(double));
//   cudaMemcpy(dA, A0, M * M * sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemcpy(dB, B0, M * M * sizeof(double), cudaMemcpyHostToDevice);

//   dim3 numBlocks(CEIL_DIV(M, BM), CEIL_DIV(M, BN));
//   dim3 threadsPerBlock(BN / TN, BM / TM);

//   square_dgemm_kernel_2d_blocktiling<<<numBlocks, threadsPerBlock>>>(M, dA, dB, dC);
//   cudaDeviceSynchronize();
//   cudaMemcpy(C0, dC, M * M * sizeof(double), cudaMemcpyDeviceToHost);

//   cudaFree(dA);
//   cudaFree(dB);
//   cudaFree(dC);
// }

// const char *dgemm_desc = "Optimized DGEMM with partial vectorization (Step-by-step fix)";


// #include <cuda_runtime.h>
// #include <cuda.h>
// #include <stdio.h>
// #include <math.h>

// #define CEIL_DIV(a, b) ((a + b - 1) / b)

// // Tile sizes
// #define BM 64
// #define BN 64
// #define BK 16

// // Each thread computes a 4x4 tile
// #define TM 4
// #define TN 4

// __global__ void square_dgemm_kernel_2d_blocktiling(const int M,
//                                                    const double *__restrict__ A,
//                                                    const double *__restrict__ B,
//                                                    double *__restrict__ C)
// {
//   __shared__ double As[BM * BK]; // Shared memory for A tile
//   __shared__ double Bs[BK * BN]; // Shared memory for B tile

//   const int blockRow = blockIdx.x;
//   const int blockCol = blockIdx.y;

//   // Threads per block in M and N directions
//   const int THREADS_PER_BLOCK_M = BM / TM; // 64/4 =16
//   const int THREADS_PER_BLOCK_N = BN / TN; // 64/4 =16

//   const int tRow = threadIdx.y; // 0..15
//   const int tCol = threadIdx.x; // 0..15

//   const int globalRowStart = blockRow * BM;
//   const int globalColStart = blockCol * BN;

//   const int threadBaseRow = tRow * TM;
//   const int threadBaseCol = tCol * TN;

//   double threadResults[TM * TN] = {0.0};

//   const int totalThreads = THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N; // 256
//   const int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

//   // Number of double2 loads per thread for A and B
//   const int aVectorLoads = ((BM * BK) / 2 + totalThreads - 1) / totalThreads;
//   const int bVectorLoads = ((BK * BN) / 2 + totalThreads - 1) / totalThreads;

//   // Iterate over the K dimension in steps of BK
//   for (int kBlock = 0; kBlock < M; kBlock += BK)
//   {
//     // Pointers to the sub-block in global memory
//     const double *A_tile = A + (globalRowStart * M + kBlock);
//     const double *B_tile = B + (kBlock * M + globalColStart);

//     // Load A tile into shared memory using double2
//     for (int i = 0; i < aVectorLoads; i++)
//     {
//       int vecIndex = linearThreadIdx + i * totalThreads;
//       if (vecIndex < (BM * BK) / 2)
//       {
//         int elemIndex = vecIndex * 2;
//         int rowA = elemIndex / BK; // which row in A tile
//         int colA = elemIndex % BK; // which column in A tile

//         // First element
//         double val0 = 0.0;
//         int gRowA = globalRowStart + rowA;
//         int gColA = kBlock + colA;
//         if (gRowA < M && gColA < M)
//           val0 = A_tile[rowA * M + colA];

//         // Second element
//         double val1 = 0.0;
//         if (colA + 1 < BK)
//         {
//           int gColA2 = kBlock + (colA + 1);
//           if (gRowA < M && gColA2 < M)
//             val1 = A_tile[rowA * M + (colA + 1)];
//         }

//         As[elemIndex] = val0;
//         As[elemIndex + 1] = val1;
//       }
//     }

//     // Load B tile into shared memory using double2
//     for (int i = 0; i < bVectorLoads; i++)
//     {
//       int vecIndex = linearThreadIdx + i * totalThreads;
//       if (vecIndex < (BK * BN) / 2)
//       {
//         int elemIndex = vecIndex * 2;
//         int rowB = elemIndex / BN; // which row in B tile
//         int colB = elemIndex % BN; // which column in B tile

//         double val0 = 0.0;
//         int gRowB = kBlock + rowB;
//         int gColB = globalColStart + colB;
//         if (gRowB < M && gColB < M)
//           val0 = B_tile[rowB * M + colB];

//         double val1 = 0.0;
//         if (colB + 1 < BN)
//         {
//           int gColB2 = globalColStart + (colB + 1);
//           if (gRowB < M && gColB2 < M)
//             val1 = B_tile[rowB * M + (colB + 1)];
//         }

//         Bs[elemIndex] = val0;
//         Bs[elemIndex + 1] = val1;
//       }
//     }

//     __syncthreads();

//     // Compute partial results for this K-block
//     // dotIdx: which column in A and row in B
//     for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
//     {
//       for (int i = 0; i < TM; i++)
//       {
//         double aVal = As[(threadBaseRow + i) * BK + dotIdx];
//         for (int j = 0; j < TN; j++)
//         {
//           double bVal = Bs[dotIdx * BN + (threadBaseCol + j)];
//           threadResults[i * TN + j] += aVal * bVal;
//         }
//       }
//     }

//     __syncthreads();
//   }

//   // Store results to global memory (scalar store for correctness)
//   for (int i = 0; i < TM; i++)
//   {
//     int cRow = globalRowStart + threadBaseRow + i;
//     if (cRow < M)
//     {
//       for (int j = 0; j < TN; j++)
//       {
//         int cCol = globalColStart + threadBaseCol + j;
//         if (cCol < M)
//         {
//           C[cRow * M + cCol] = threadResults[i * TN + j];
//         }
//       }
//     }
//   }
// }

// double *dA, *dB, *dC;

// void square_dgemm(const int M, const double *A0, const double *B0, double *C0)
// {
//   cudaMalloc(&dA, M * M * sizeof(double));
//   cudaMalloc(&dB, M * M * sizeof(double));
//   cudaMalloc(&dC, M * M * sizeof(double));

//   cudaMemcpy(dA, A0, M * M * sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemcpy(dB, B0, M * M * sizeof(double), cudaMemcpyHostToDevice);

//   dim3 numBlocks(CEIL_DIV(M, BM), CEIL_DIV(M, BN));
//   dim3 threadsPerBlock(BN / TN, BM / TM); // (64/4=16, 64/4=16) -> (16,16)

//   square_dgemm_kernel_2d_blocktiling<<<numBlocks, threadsPerBlock>>>(M, dA, dB, dC);
//   cudaDeviceSynchronize();

//   cudaMemcpy(C0, dC, M * M * sizeof(double), cudaMemcpyDeviceToHost);

//   cudaFree(dA);
//   cudaFree(dB);
//   cudaFree(dC);
// }

// const char *dgemm_desc = "Corrected DGEMM with vectorized global loads using double2 and scalar computation.";

// Good First Step

// #include <cuda_runtime.h>
// #include <cuda.h>
// #include <stdio.h>
// #include <math.h>

// #define CEIL_DIV(a, b) ((a + b - 1) / b)

// // Tile sizes
// #define BM 64 // Block size in M dimension
// #define BN 64 // Block size in N dimension
// #define BK 16 // Block size in K dimension

// // Thread tile sizes (each thread computes a TMxTN tile of C)
// #define TM 4
// #define TN 4

// __global__ void square_dgemm_kernel_2d_blocktiling(const int M, const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C)
// {
//   __shared__ double As[BM * BK]; // now viewed as (BK x BM) in col-major form: As[k*BM + m]
//   __shared__ double Bs[BK * BN]; // viewed as (BK x BN): Bs[k*BN + n]

//   const int blockRow = blockIdx.x; // block index along M dimension
//   const int blockCol = blockIdx.y; // block index along N dimension

//   // Threads per block configuration:
//   const int THREADS_PER_BLOCK_M = BM / TM; // = 64/4 = 16
//   const int THREADS_PER_BLOCK_N = BN / TN; // = 64/4 = 16

//   // Total threads per block = 16*16 = 256 threads
//   // (But original code used BM=64, BN=64, TM=8, TN=8 => 8x8=64 threads.
//   // Adjusting TM,TN as per the original code:
//   // The user code states TM=4, TN=4 => THREADS_PER_BLOCK_M=16, THREADS_PER_BLOCK_N=16 => 256 threads.)
//   // If you prefer fewer threads, adjust TM,TN accordingly.

//   const int tRow = threadIdx.y;
//   const int tCol = threadIdx.x;

//   // Global start indices for this block tile
//   const int globalRowStart = blockRow * BM;
//   const int globalColStart = blockCol * BN;

//   // This thread's starting tile offset
//   const int threadBaseRow = tRow * TM;
//   const int threadBaseCol = tCol * TN;

//   // Flatten thread index for linear mapping of loads
//   const int THREADS_PER_BLOCK = THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N; // = 256
//   const int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

//   // Precompute how many elements each thread loads from A and B per iteration
//   const int elemsA = BM * BK; // total elements in the A tile
//   const int elemsB = BK * BN; // total elements in the B tile

//   // Each thread loads a contiguous chunk of these
//   // Using double2 vectorization: each double2 covers 2 elements
//   // Make sure (elemsA and elemsB) are even. They are 64*16=1024 for both, which is even.
//   const int double2_elemsA = elemsA / 2;
//   const int double2_elemsB = elemsB / 2;

//   const int elemsPerThreadA = (double2_elemsA + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//   const int elemsPerThreadB = (double2_elemsB + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

//   double threadResults[TM * TN] = {0.0};

//   for (int kBlock = 0; kBlock < M; kBlock += BK)
//   {
//     //==========================
//     // Load A tile into shared memory (transposed)
//     //==========================
//     {
//       double2 *A2 = (double2 *)A; // reinterpret for vector loads
//       double2 *As2 = (double2 *)As;
//       int A_tile_double2_offset = (globalRowStart * M + kBlock);
//       // Note: We are loading a BMxBK block from A starting at (globalRowStart,kBlock)
//       // Flattened: row major in global mem: For i in [0,BM), j in [0,BK):
//       // globalAindex = (globalRowStart + i)*M + (kBlock + j)
//       // We want to store transposed in SMEM: As[k*BM + i]
//       // We'll just load row-major from global and store to As in col-major form.

//       for (int load_i = 0; load_i < elemsPerThreadA; load_i++)
//       {
//         int idx2 = linearThreadIdx + load_i * THREADS_PER_BLOCK;
//         if (idx2 < double2_elemsA)
//         {
//           // double2 index back to element index:
//           int elem_idx = idx2 * 2;  // each double2 covers 2 elements
//           int irow = elem_idx / BK; // i in [0,BM)
//           int icol = elem_idx % BK; // j in [0,BK)
//           int globalArow = globalRowStart + irow;
//           int globalAcol = kBlock + icol;
//           double2 val2 = {0.0, 0.0};

//           if (globalArow < M && globalAcol + 1 < M)
//           {
//             // load two elements at once
//             int globalAIdx2 = (globalArow * M + globalAcol) / 2;
//             // This doesn't match directly if we do double2 indexing simply.
//             // Let's load element-wise instead for clarity:
//             // We'll just do two single loads for clarity. Vectorization can be done if pointers are aligned.
//             double val0 = (globalArow < M && globalAcol < M) ? A[globalArow * M + globalAcol] : 0.0;
//             double val1 = (globalArow < M && (globalAcol + 1) < M) ? A[globalArow * M + globalAcol + 1] : 0.0;
//             val2.x = val0;
//             val2.y = val1;
//           }
//           else
//           {
//             // Handle boundary
//             if (globalArow < M && globalAcol < M)
//             {
//               val2.x = A[globalArow * M + globalAcol];
//             }
//             // val2.y remains 0 if out of range
//           }

//           // Now we must store them transposed in As:
//           // We have two consecutive elements: (irow, icol) and possibly (irow, icol+1)
//           // Actually, we must be careful here because we took two consecutive global elements.
//           // We are vector-loading horizontally along K dimension. It's simpler to consider we load along row:
//           // To simplify, let's just store element by element:
//           // We'll store val2.x and val2.y separately:
//           As[icol * BM + irow] = val2.x; // transpose: (irow,icol)->(icol,irow)
//           if (icol + 1 < BK)
//           {
//             As[(icol + 1) * BM + irow] = val2.y;
//           }
//         }
//       }
//     }

//     //==========================
//     // Load B tile into shared memory
//     //==========================
//     {
//       double2 *B2 = (double2 *)B;
//       double2 *Bs2 = (double2 *)Bs;
//       // Similar approach for B
//       // B tile: size BKxBN at (kBlock, globalColStart)
//       // globalBindex = (kBlock+i)*M + (globalColStart + j)
//       // We'll store similarly as Bs[k*BN + n]

//       for (int load_i = 0; load_i < elemsPerThreadB; load_i++)
//       {
//         int idx2 = linearThreadIdx + load_i * THREADS_PER_BLOCK;
//         if (idx2 < double2_elemsB)
//         {
//           int elem_idx = idx2 * 2;
//           int irow = elem_idx / BN; // i in [0,BK)
//           int icol = elem_idx % BN; // j in [0,BN)
//           int globalBrow = kBlock + irow;
//           int globalBcol = globalColStart + icol;
//           double val0 = 0.0, val1 = 0.0;
//           if (globalBrow < M && globalBcol < M)
//             val0 = B[globalBrow * M + globalBcol];
//           if (globalBrow < M && (globalBcol + 1) < M)
//             val1 = B[globalBrow * M + globalBcol + 1];

//           Bs[irow * BN + icol] = val0;
//           if (icol + 1 < BN)
//           {
//             Bs[irow * BN + (icol + 1)] = val1;
//           }
//         }
//       }
//     }

//     __syncthreads();

//     //==========================
//     // Compute partial results for this block
//     //==========================
//     for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
//     {
//       double regM[TM];
//       double regN[TN];
//       for (int i = 0; i < TM; i++)
//       {
//         int aRow = threadBaseRow + i;
//         // As is transposed: As[k*BM + m], with k=dotIdx, m=aRow
//         regM[i] = As[dotIdx * BM + aRow];
//       }
//       for (int j = 0; j < TN; j++)
//       {
//         int bCol = threadBaseCol + j;
//         // Bs[k*BN + n], k=dotIdx, n=bCol
//         regN[j] = Bs[dotIdx * BN + bCol];
//       }
//       for (int i = 0; i < TM; i++)
//       {
//         for (int j = 0; j < TN; j++)
//         {
//           threadResults[i * TN + j] += regM[i] * regN[j];
//         }
//       }
//     }

//     __syncthreads();
//   }

//   //==========================
//   // Write results back to global memory
//   //==========================
//   for (int i = 0; i < TM; i++)
//   {
//     int cRow = globalRowStart + threadBaseRow + i;
//     for (int j = 0; j < TN; j++)
//     {
//       int cCol = globalColStart + threadBaseCol + j;
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
//   cudaMalloc(&dA, M * M * sizeof(double));
//   cudaMalloc(&dB, M * M * sizeof(double));
//   cudaMalloc(&dC, M * M * sizeof(double));
//   cudaMemcpy(dA, A0, M * M * sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemcpy(dB, B0, M * M * sizeof(double), cudaMemcpyHostToDevice);

//   dim3 numBlocks(CEIL_DIV(M, BM), CEIL_DIV(M, BN));
//   dim3 threadsPerBlock(BN / TN, BM / TM);
//   // For BM=64, TM=4 => BM/TM=16; BN=64, TN=4 => BN/TN=16; so threads=(16,16)=256 threads per block.

//   square_dgemm_kernel_2d_blocktiling<<<numBlocks, threadsPerBlock>>>(M, dA, dB, dC);
//   cudaDeviceSynchronize();
//   cudaMemcpy(C0, dC, M * M * sizeof(double), cudaMemcpyDeviceToHost);

//   cudaFree(dA);
//   cudaFree(dB);
//   cudaFree(dC);
// }

// const char *dgemm_desc = "Optimized DGEMM with 2D blocktiling, transposed shared memory, and vectorized loads.";


// #include <cuda_runtime.h>
// #include <cuda.h>
// #include <stdio.h>
// #include <math.h>

// #define CEIL_DIV(a, b) ((a + b - 1) / b)

// // Tile sizes
// #define BM 64
// #define BN 64
// #define BK 16

// // Each thread computes a 4x4 tile
// #define TM 4
// #define TN 4

// __shared__ __align__(16) double As[BM * BK]; // Shared memory for A tile
// __shared__ __align__(16) double Bs[BK * BN]; // Shared memory for B tile

// __global__ void square_dgemm_kernel_2d_blocktiling(const int M,
//                                                    const double *__restrict__ A,
//                                                    const double *__restrict__ B,
//                                                    double *__restrict__ C)
// {
//   // Use blockIdx.y for rows and blockIdx.x for columns, as in the references
//   const int blockRow = blockIdx.y; 
//   const int blockCol = blockIdx.x;

//   // Dimensions of the thread block and sub-tiles
//   const int THREADS_PER_BLOCK_M = BM / TM; // 64/4 =16
//   const int THREADS_PER_BLOCK_N = BN / TN; // 64/4 =16

//   const int tRow = threadIdx.y; // 0..15
//   const int tCol = threadIdx.x; // 0..15

//   const int globalRowStart = blockRow * BM;
//   const int globalColStart = blockCol * BN;

//   const int threadBaseRow = tRow * TM; // Within-block row offset
//   const int threadBaseCol = tCol * TN; // Within-block column offset

//   double threadResults[TM * TN] = {0.0};

//   const int totalThreads = THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N; // 256
//   const int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

//   // Number of double2 loads per tile
//   const int A_double2_tiles = (BM * BK) / 2; // 64*16=1024 doubles; /2=512 double2
//   const int B_double2_tiles = (BK * BN) / 2; // 16*64=1024 doubles; /2=512 double2
//   const int aVectorLoads = (A_double2_tiles + totalThreads - 1) / totalThreads;
//   const int bVectorLoads = (B_double2_tiles + totalThreads - 1) / totalThreads;

//   // Iterate over K dimension in steps of BK
//   for (int kBlock = 0; kBlock < M; kBlock += BK)
//   {
//     // Global pointers to the current A and B sub-blocks
//     const double *A_tile = A + (globalRowStart * M + kBlock);
//     const double *B_tile = B + (kBlock * M + globalColStart);

//     // Load A tile into shared memory using double2
//     {
//       double2 *As_vec = reinterpret_cast<double2 *>(As);
//       for (int i = 0; i < aVectorLoads; i++)
//       {
//         int vecIndex = linearThreadIdx + i * totalThreads;
//         if (vecIndex < A_double2_tiles)
//         {
//           int elemIndex = vecIndex * 2;
//           int rowA = elemIndex / BK; // which row in A tile
//           int colA = elemIndex % BK; // which column in A tile

//           double val0 = 0.0;
//           double val1 = 0.0;
//           int gRowA = globalRowStart + rowA;
//           int gColA = kBlock + colA;
//           if (gRowA < M && gColA < M)
//             val0 = A_tile[rowA * M + colA];

//           if (colA + 1 < BK) {
//             int gColA2 = kBlock + (colA + 1);
//             if (gRowA < M && gColA2 < M)
//               val1 = A_tile[rowA * M + (colA + 1)];
//           }

//           As_vec[vecIndex] = make_double2(val0, val1);
//         }
//       }
//     }

//     // Load B tile into shared memory using double2
//     {
//       double2 *Bs_vec = reinterpret_cast<double2 *>(Bs);
//       for (int i = 0; i < bVectorLoads; i++)
//       {
//         int vecIndex = linearThreadIdx + i * totalThreads;
//         if (vecIndex < B_double2_tiles)
//         {
//           int elemIndex = vecIndex * 2;
//           int rowB = elemIndex / BN; // which row in B tile
//           int colB = elemIndex % BN; // which column in B tile

//           double val0 = 0.0;
//           double val1 = 0.0;
//           int gRowB = kBlock + rowB;
//           int gColB = globalColStart + colB;

//           if (gRowB < M && gColB < M)
//             val0 = B_tile[rowB * M + colB];

//           if (colB + 1 < BN) {
//             int gColB2 = globalColStart + (colB + 1);
//             if (gRowB < M && gColB2 < M)
//               val1 = B_tile[rowB * M + (colB + 1)];
//           }

//           Bs_vec[vecIndex] = make_double2(val0, val1);
//         }
//       }
//     }

//     __syncthreads();

//     // Compute partial results
//     double2 *As_vec = reinterpret_cast<double2 *>(As);
//     double2 *Bs_vec = reinterpret_cast<double2 *>(Bs);

//     int BK_half = BK / 2; // =8
//     int BN_half = BN / 2; // =32

//     for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
//     {
//       int dotPair = dotIdx / 2;
//       int dotElem = dotIdx % 2;

//       for (int i = 0; i < TM; i++)
//       {
//         int aRow = threadBaseRow + i;
//         double2 aVec = As_vec[aRow * BK_half + dotPair];
//         double aVal = (dotElem == 0) ? aVec.x : aVec.y;

//         for (int j = 0; j < TN; j++)
//         {
//           int bRow = dotIdx;
//           int bCol = threadBaseCol + j;
//           int bPair = bCol / 2;
//           int bElem = bCol % 2;

//           double2 bVec = Bs_vec[bRow * BN_half + bPair];
//           double bVal = (bElem == 0) ? bVec.x : bVec.y;

//           threadResults[i * TN + j] += aVal * bVal;
//         }
//       }
//     }

//     __syncthreads();
//   }

//   // Store results to global memory (scalar store for correctness verification)
//   for (int i = 0; i < TM; i++)
//   {
//     int cRow = globalRowStart + threadBaseRow + i;
//     if (cRow < M)
//     {
//       for (int j = 0; j < TN; j++)
//       {
//         int cCol = globalColStart + threadBaseCol + j;
//         if (cCol < M)
//         {
//           C[cRow * M + cCol] = threadResults[i * TN + j];
//         }
//       }
//     }
//   }
// }

// double *dA, *dB, *dC;

// void square_dgemm(const int M, const double *A0, const double *B0, double *C0)
// {
//   cudaMalloc(&dA, M * M * sizeof(double));
//   cudaMalloc(&dB, M * M * sizeof(double));
//   cudaMalloc(&dC, M * M * sizeof(double));

//   cudaMemcpy(dA, A0, M * M * sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemcpy(dB, B0, M * M * sizeof(double), cudaMemcpyHostToDevice);

//   dim3 numBlocks(CEIL_DIV(M, BM), CEIL_DIV(M, BN));
//   dim3 threadsPerBlock(BN / TN, BM / TM); // (64/4=16, 64/4=16) -> (16,16)

//   square_dgemm_kernel_2d_blocktiling<<<numBlocks, threadsPerBlock>>>(M, dA, dB, dC);
//   cudaDeviceSynchronize();

//   cudaMemcpy(C0, dC, M * M * sizeof(double), cudaMemcpyDeviceToHost);

//   cudaFree(dA);
//   cudaFree(dB);
//   cudaFree(dC);
// }

// const char *dgemm_desc = "Revised DGEMM with vectorized loads for A and B and scalar stores, ensuring correct results.";


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

// A function to load a BMxBK tile from A and a BKxBN tile from B into shared memory
__device__ inline void load_tile(
    const double *A, const double *B,
    int M, int globalRowStart, int globalColStart,
    int kt, // which k-tile
    double *As, double *Bs,
    int tRow, int tCol,
    int THREADS_PER_BLOCK_M, int THREADS_PER_BLOCK_N)
{
  int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

  // Number of elements in each tile:
  // A tile: BM x BK = 64 x 16 = 1024 elements
  // B tile: BK x BN = 16 x 64 = 1024 elements
  // Each block: (BM/TM) x (BN/TN) = (64/4) x (64/4) = 16 x 16 = 256 threads.
  // Each thread loads (1024/256)=4 elements from A and 4 from B.

  int elemsPerThreadA = (BM * BK) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
  int elemsPerThreadB = (BK * BN) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);

  // Load A tile: covers rows [globalRowStart..globalRowStart+BM-1], cols [kt*BK..kt*BK+BK-1]
  for (int i = 0; i < elemsPerThreadA; i++) {
    int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
    int rowA = index / BK;
    int colA = index % BK;

    int globalArow = globalRowStart + rowA;
    int globalAcol = kt * BK + colA;

    double valA = 0.0;
    if (globalArow < M && globalAcol < M) {
      valA = A[globalArow * M + globalAcol];
    }
    As[rowA * BK + colA] = valA;
  }

  // Load B tile: covers rows [kt*BK..kt*BK+BK-1], cols [globalColStart..globalColStart+BN-1]
  for (int i = 0; i < elemsPerThreadB; i++) {
    int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
    int rowB = index / BN;
    int colB = index % BN;

    int globalBrow = kt * BK + rowB;
    int globalBcol = globalColStart + colB;

    double valB = 0.0;
    if (globalBrow < M && globalBcol < M) {
      valB = B[globalBrow * M + globalBcol];
    }
    Bs[rowB * BN + colB] = valB;
  }
}

__device__ inline void compute_tile(
    double *As, double *Bs,
    double *threadResults,
    int tRow, int tCol)
{
  // Compute partial product: Ctile += A_tile * B_tile
  // TM=4, TN=4
  for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
    double regM[TM];
    double regN[TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      int aRow = tRow * TM + i;
      regM[i] = As[aRow * BK + dotIdx];
    }
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      int bCol = tCol * TN + j;
      regN[j] = Bs[dotIdx * BN + bCol];
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        threadResults[i * TN + j] += regM[i] * regN[j];
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

  // Simple 2D mapping of blocks to tiles
  int tile_m = blockIdx.y;
  int tile_n = blockIdx.x;

  if (tile_m >= m_tiles || tile_n >= n_tiles)
    return;

  const int THREADS_PER_BLOCK_M = BM / TM; // 64/4=16
  const int THREADS_PER_BLOCK_N = BN / TN; // 64/4=16
  // blockDim would be (THREADS_PER_BLOCK_N, THREADS_PER_BLOCK_M) = (16,16) = 256 threads

  const int tRow = threadIdx.y; 
  const int tCol = threadIdx.x; 

  const int globalRowStart = tile_m * BM;
  const int globalColStart = tile_n * BN;

  double threadResults[TM * TN];
  #pragma unroll
  for (int i = 0; i < TM * TN; i++) {
    threadResults[i] = 0.0;
  }

  int k_tiles = CEIL_DIV(M, BK);

  // Preload the first tile
  int curLoad = 0;
  load_tile(A, B, M, globalRowStart, globalColStart, 0,
            As[curLoad], Bs[curLoad], tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);
  __syncthreads(); // ensure first tile is loaded

  int curCompute = curLoad;

  for (int kt = 1; kt < k_tiles; kt++) {
    int nextLoad = 1 - curLoad;

    // Load next tile
    load_tile(A, B, M, globalRowStart, globalColStart, kt,
              As[nextLoad], Bs[nextLoad], tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);

    __syncthreads(); // Wait for load to complete

    // Compute on previously loaded tile
    compute_tile(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

    __syncthreads(); // Ensure compute done before reusing buffers

    // Switch buffers
    curLoad = nextLoad;
    curCompute = curLoad;
  }

  // Compute on the last tile
  compute_tile(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

  // Write back results
  for (int i = 0; i < TM; i++) {
    int cRow = globalRowStart + tRow * TM + i;
    for (int j = 0; j < TN; j++) {
      int cCol = globalColStart + tCol * TN + j;
      if (cRow < M && cCol < M) {
        C[cRow * M + cCol] = threadResults[i * TN + j];
      }
    }
  }
}

double *dA, *dB, *dC;

void square_dgemm(const int M, const double *A0, const double *B0, double *C0) {
  cudaMalloc(&dA, M * M * sizeof(double));
  cudaMalloc(&dB, M * M * sizeof(double));
  cudaMalloc(&dC, M * M * sizeof(double));
  cudaMemcpy(dA, A0, M * M * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B0, M * M * sizeof(double), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BN / TN, BM / TM); // (16,16)
  int m_tiles = CEIL_DIV(M, BM);
  int n_tiles = CEIL_DIV(M, BN);

  dim3 numBlocks(n_tiles, m_tiles);

  square_dgemm_kernel_2d_blocktiling<<<numBlocks, threadsPerBlock>>>(M, dA, dB, dC);
  cudaDeviceSynchronize();
  cudaMemcpy(C0, dC, M * M * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

const char *dgemm_desc = "Double-buffered DGEMM kernel with correct indexing and synchronization";