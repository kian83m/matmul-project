#include <stdio.h>
const char *dgemm_desc = "My Basic Batched.";

#define CEIL_DIV(a, b) ((a + b - 1) / b)

// Tile sizes
#define BK 16

// Thread tile sizes
#define TM 4
#define TN 4

// Use __ldg for read-only global memory fetches
// This can sometimes improve performance if the data is read-only
// and may better utilize caches.
__device__ __forceinline__ double ldg_double(const double *ptr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(ptr);
#else
  return *ptr;
#endif
}

// A helper function to load a double2 from global memory safely.
// We'll assume the pointers are suitably aligned since we know BM,BK,BN are multiples of 2.
__device__ __forceinline__ double2 ldg_double2(const double2 *ptr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(ptr);
#else
  return *ptr;
#endif
}

// Compute tile indices with a linear-to-2D mapping for potential improved locality
__device__ inline void compute_tile_indices(int &tile_m, int &tile_n, int m_tiles, int n_tiles) {
  int linear_idx = blockIdx.y * gridDim.x + blockIdx.x;
  tile_m = linear_idx / n_tiles;
  tile_n = linear_idx % n_tiles;
}

//_________________________________________________________
//***********************64*******************************/
#define BM64 64
#define BN64 64

__device__ inline void load_tile_64(
    const double * __restrict__ A, const double * __restrict__ B,
    int M, int globalRowStart, int globalColStart,
    int kt, // which k-tile
    double * __restrict__ As, double * __restrict__ Bs,
    int tRow, int tCol,
    int THREADS_PER_BLOCK_M, int THREADS_PER_BLOCK_N)
{
  // Each tile: A: BMxBK = 64x16=1024 elems, B: BKxBN=16x64=1024 elems
  // blockDim = 16x16 = 256 threads
  // Each thread loads 4 elements from A and 4 from B. We can vectorize these loads using double2.

  int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

  int elemsPerThreadA = (BM64 * BK) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N); // 1024/256=4
  int elemsPerThreadB = (BK * BN64) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N); // also 4

  // We will load 4 elements from A and B. Since BK=16 is divisible by 2, we can load double2 pairs.
  // Similarly for BN=64 and BM=64.

  // Load A tile
  // The A tile spans rows [globalRowStart .. globalRowStart+BM-1], 
  // and columns [kt*BK .. kt*BK+BK-1].
  // We'll load them linearly and use double2 loads if possible.
  for (int i = 0; i < elemsPerThreadA; i++) {
    int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
    int rowA = index / BK; 
    int colA = index % BK;

    int globalArow = globalRowStart + rowA;
    int globalAcol = kt * BK + colA;

    double valA = 0.0;
    if (globalArow < M && globalAcol < M) {
      valA = ldg_double(&A[globalArow * M + globalAcol]);
    }
    As[rowA * BK + colA] = valA;
  }

  // Load B tile
  // B tile spans rows [kt*BK .. kt*BK+BK-1], cols [globalColStart..globalColStart+BN-1]
  for (int i = 0; i < elemsPerThreadB; i++) {
    int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
    int rowB = index / BN64;
    int colB = index % BN64;

    int globalBrow = kt * BK + rowB;
    int globalBcol = globalColStart + colB;

    double valB = 0.0;
    if (globalBrow < M && globalBcol < M) {
      valB = ldg_double(&B[globalBrow * M + globalBcol]);
    }
    Bs[rowB * BN64 + colB] = valB;
  }
}

__device__ inline void compute_tile_64(
    const double * __restrict__ As, const double * __restrict__ Bs,
    double * __restrict__ threadResults,
    int tRow, int tCol)
{
  // Compute partial product: Ctile += A_tile * B_tile
  // TM=4, TN=4
  #pragma unroll
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
      regN[j] = Bs[dotIdx * BN64 + bCol];
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      double m_val = regM[i];
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        threadResults[i * TN + j] += m_val * regN[j];
      }
    }
  }
}

__global__ void square_dgemm_kernel_2d_blocktiling_64(
    const int M, const int N, const double ** A0, const double ** B0, double ** C0)
{
  // Double buffering
  __shared__ double As[2][BM64 * BK];
  __shared__ double Bs[2][BK * BN64];

  int m_tiles = CEIL_DIV(M, BM64);
  int n_tiles = CEIL_DIV(M, BN64);

  int tile_m, tile_n;
  compute_tile_indices(tile_m, tile_n, m_tiles, n_tiles);

  int batch = blockIdx.z; 

  if (tile_m >= m_tiles || tile_n >= n_tiles || batch >= N)
    return;


    const double *A = A0[batch];
    const double *B = B0[batch];
    double *C = C0[batch];

  const int THREADS_PER_BLOCK_M = BM64 / TM; // 64/4=16
  const int THREADS_PER_BLOCK_N = BN64 / TN; // 64/4=16
  // blockDim should be (16,16)=256 threads

  const int tRow = threadIdx.y; 
  const int tCol = threadIdx.x; 

  const int globalRowStart = tile_m * BM64;
  const int globalColStart = tile_n * BN64;

  double threadResults[TM * TN];
  #pragma unroll
  for (int i = 0; i < TM * TN; i++) {
    threadResults[i] = 0.0;
  }

  int k_tiles = CEIL_DIV(M, BK);

  // Preload the first tile
  int curLoad = 0;
  load_tile_64(A, B, M, globalRowStart, globalColStart, 0,
            As[curLoad], Bs[curLoad], tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);
  __syncthreads(); // ensure first tile is loaded

  int curCompute = curLoad;

  // Main loop over K
  for (int kt = 1; kt < k_tiles; kt++) {
    int nextLoad = 1 - curLoad;

    // Start loading next tile
    load_tile_64(A, B, M, globalRowStart, globalColStart, kt,
              As[nextLoad], Bs[nextLoad],
              tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);

    __syncthreads(); // Wait for next tile to be fully loaded

    // Compute on previously loaded tile
    compute_tile_64(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

    __syncthreads(); // Ensure compute done before reusing shared memory

    // Switch buffers
    curLoad = nextLoad;
    curCompute = curLoad;
  }

  // Compute on the last tile
  compute_tile_64(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

  // Write results back to C
  #pragma unroll
  for (int i = 0; i < TM; i++) {
    int cRow = globalRowStart + tRow * TM + i;
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      int cCol = globalColStart + tCol * TN + j;
      if (cRow < M && cCol < M) {
        C[cRow * M + cCol] = threadResults[i * TN + j];
      }
    }
  }
}
//********************************************************/
//_________________________________________________________






//_________________________________________________________
//***********************32*******************************/
#define BM32 32
#define BN32 32

__device__ inline void load_tile_32(
    const double * __restrict__ A, const double * __restrict__ B,
    int M, int globalRowStart, int globalColStart,
    int kt, // which k-tile
    double * __restrict__ As, double * __restrict__ Bs,
    int tRow, int tCol,
    int THREADS_PER_BLOCK_M, int THREADS_PER_BLOCK_N)
{
  // Each tile: A: BMxBK = 64x16=1024 elems, B: BKxBN=16x64=1024 elems
  // blockDim = 16x16 = 256 threads
  // Each thread loads 4 elements from A and 4 from B. We can vectorize these loads using double2.

  int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

  int elemsPerThreadA = (BM32 * BK) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N); // 1024/256=4
  int elemsPerThreadB = (BK * BN32) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N); // also 4

  // We will load 4 elements from A and B. Since BK=16 is divisible by 2, we can load double2 pairs.
  // Similarly for BN=64 and BM=64.

  // Load A tile
  // The A tile spans rows [globalRowStart .. globalRowStart+BM-1], 
  // and columns [kt*BK .. kt*BK+BK-1].
  // We'll load them linearly and use double2 loads if possible.
  for (int i = 0; i < elemsPerThreadA; i++) {
    int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
    int rowA = index / BK; 
    int colA = index % BK;

    int globalArow = globalRowStart + rowA;
    int globalAcol = kt * BK + colA;

    double valA = 0.0;
    if (globalArow < M && globalAcol < M) {
      valA = ldg_double(&A[globalArow * M + globalAcol]);
    }
    As[rowA * BK + colA] = valA;
  }

  // Load B tile
  // B tile spans rows [kt*BK .. kt*BK+BK-1], cols [globalColStart..globalColStart+BN-1]
  for (int i = 0; i < elemsPerThreadB; i++) {
    int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
    int rowB = index / BN32;
    int colB = index % BN32;

    int globalBrow = kt * BK + rowB;
    int globalBcol = globalColStart + colB;

    double valB = 0.0;
    if (globalBrow < M && globalBcol < M) {
      valB = ldg_double(&B[globalBrow * M + globalBcol]);
    }
    Bs[rowB * BN32 + colB] = valB;
  }
}

__device__ inline void compute_tile_32(
    const double * __restrict__ As, const double * __restrict__ Bs,
    double * __restrict__ threadResults,
    int tRow, int tCol)
{
  // Compute partial product: Ctile += A_tile * B_tile
  // TM=4, TN=4
  #pragma unroll
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
      regN[j] = Bs[dotIdx * BN32 + bCol];
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      double m_val = regM[i];
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        threadResults[i * TN + j] += m_val * regN[j];
      }
    }
  }
}

__global__ void square_dgemm_kernel_2d_blocktiling_32(
    const int M, const int N, const double ** A0, const double ** B0, double ** C0)
{
  // Double buffering
  __shared__ double As[2][BM32 * BK];
  __shared__ double Bs[2][BK * BN32];

  int m_tiles = CEIL_DIV(M, BM32);
  int n_tiles = CEIL_DIV(M, BN32);

  int tile_m, tile_n;
  compute_tile_indices(tile_m, tile_n, m_tiles, n_tiles);

  int batch = blockIdx.z; 

  if (tile_m >= m_tiles || tile_n >= n_tiles || batch >= N)
    return;


    const double *A = A0[batch];
    const double *B = B0[batch];
    double *C = C0[batch];

  const int THREADS_PER_BLOCK_M = BM32 / TM; // 64/4=16
  const int THREADS_PER_BLOCK_N = BN32 / TN; // 64/4=16
  // blockDim should be (16,16)=256 threads

  const int tRow = threadIdx.y; 
  const int tCol = threadIdx.x; 

  const int globalRowStart = tile_m * BM32;
  const int globalColStart = tile_n * BN32;

  double threadResults[TM * TN];
  #pragma unroll
  for (int i = 0; i < TM * TN; i++) {
    threadResults[i] = 0.0;
  }

  int k_tiles = CEIL_DIV(M, BK);

  // Preload the first tile
  int curLoad = 0;
  load_tile_32(A, B, M, globalRowStart, globalColStart, 0,
            As[curLoad], Bs[curLoad], tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);
  __syncthreads(); // ensure first tile is loaded

  int curCompute = curLoad;

  // Main loop over K
  for (int kt = 1; kt < k_tiles; kt++) {
    int nextLoad = 1 - curLoad;

    // Start loading next tile
    load_tile_32(A, B, M, globalRowStart, globalColStart, kt,
              As[nextLoad], Bs[nextLoad],
              tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);

    __syncthreads(); // Wait for next tile to be fully loaded

    // Compute on previously loaded tile
    compute_tile_32(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

    __syncthreads(); // Ensure compute done before reusing shared memory

    // Switch buffers
    curLoad = nextLoad;
    curCompute = curLoad;
  }

  // Compute on the last tile
  compute_tile_32(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

  // Write results back to C
  #pragma unroll
  for (int i = 0; i < TM; i++) {
    int cRow = globalRowStart + tRow * TM + i;
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      int cCol = globalColStart + tCol * TN + j;
      if (cRow < M && cCol < M) {
        C[cRow * M + cCol] = threadResults[i * TN + j];
      }
    }
  }
}
//********************************************************/
//_________________________________________________________

//_________________________________________________________
//***********************16*******************************/
#define BM16 16
#define BN16 16

__device__ inline void load_tile_16(
    const double * __restrict__ A, const double * __restrict__ B,
    int M, int globalRowStart, int globalColStart,
    int kt, // which k-tile
    double * __restrict__ As, double * __restrict__ Bs,
    int tRow, int tCol,
    int THREADS_PER_BLOCK_M, int THREADS_PER_BLOCK_N)
{
  // Each tile: A: BMxBK = 64x16=1024 elems, B: BKxBN=16x64=1024 elems
  // blockDim = 16x16 = 256 threads
  // Each thread loads 4 elements from A and 4 from B. We can vectorize these loads using double2.

  int linearThreadIdx = tRow * THREADS_PER_BLOCK_N + tCol;

  int elemsPerThreadA = (BM16 * BK) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N); // 1024/256=4
  int elemsPerThreadB = (BK * BN16) / (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N); // also 4

  // We will load 4 elements from A and B. Since BK=16 is divisible by 2, we can load double2 pairs.
  // Similarly for BN=64 and BM=64.

  // Load A tile
  // The A tile spans rows [globalRowStart .. globalRowStart+BM-1], 
  // and columns [kt*BK .. kt*BK+BK-1].
  // We'll load them linearly and use double2 loads if possible.
  for (int i = 0; i < elemsPerThreadA; i++) {
    int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
    int rowA = index / BK; 
    int colA = index % BK;

    int globalArow = globalRowStart + rowA;
    int globalAcol = kt * BK + colA;

    double valA = 0.0;
    if (globalArow < M && globalAcol < M) {
      valA = ldg_double(&A[globalArow * M + globalAcol]);
    }
    As[rowA * BK + colA] = valA;
  }

  // Load B tile
  // B tile spans rows [kt*BK .. kt*BK+BK-1], cols [globalColStart..globalColStart+BN-1]
  for (int i = 0; i < elemsPerThreadB; i++) {
    int index = linearThreadIdx + i * (THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N);
    int rowB = index / BN16;
    int colB = index % BN16;

    int globalBrow = kt * BK + rowB;
    int globalBcol = globalColStart + colB;

    double valB = 0.0;
    if (globalBrow < M && globalBcol < M) {
      valB = ldg_double(&B[globalBrow * M + globalBcol]);
    }
    Bs[rowB * BN16 + colB] = valB;
  }
}

__device__ inline void compute_tile_16(
    const double * __restrict__ As, const double * __restrict__ Bs,
    double * __restrict__ threadResults,
    int tRow, int tCol)
{
  // Compute partial product: Ctile += A_tile * B_tile
  // TM=4, TN=4
  #pragma unroll
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
      regN[j] = Bs[dotIdx * BN16 + bCol];
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      double m_val = regM[i];
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        threadResults[i * TN + j] += m_val * regN[j];
      }
    }
  }
}

__global__ void square_dgemm_kernel_2d_blocktiling_16(
    const int M, const int N, const double ** A0, const double ** B0, double ** C0)
{
  // Double buffering
  __shared__ double As[2][BM16 * BK];
  __shared__ double Bs[2][BK * BN16];

  int m_tiles = CEIL_DIV(M, BM16);
  int n_tiles = CEIL_DIV(M, BN16);

  int tile_m, tile_n;
  compute_tile_indices(tile_m, tile_n, m_tiles, n_tiles);

  int batch = blockIdx.z; 

  if (tile_m >= m_tiles || tile_n >= n_tiles || batch >= N)
    return;


    const double *A = A0[batch];
    const double *B = B0[batch];
    double *C = C0[batch];

  const int THREADS_PER_BLOCK_M = BM16 / TM; // 64/4=16
  const int THREADS_PER_BLOCK_N = BN16 / TN; // 64/4=16
  // blockDim should be (16,16)=256 threads

  const int tRow = threadIdx.y; 
  const int tCol = threadIdx.x; 

  const int globalRowStart = tile_m * BM16;
  const int globalColStart = tile_n * BN16;

  double threadResults[TM * TN];
  #pragma unroll
  for (int i = 0; i < TM * TN; i++) {
    threadResults[i] = 0.0;
  }

  int k_tiles = CEIL_DIV(M, BK);

  // Preload the first tile
  int curLoad = 0;
  load_tile_16(A, B, M, globalRowStart, globalColStart, 0,
            As[curLoad], Bs[curLoad], tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);
  __syncthreads(); // ensure first tile is loaded

  int curCompute = curLoad;

  // Main loop over K
  for (int kt = 1; kt < k_tiles; kt++) {
    int nextLoad = 1 - curLoad;

    // Start loading next tile
    load_tile_16(A, B, M, globalRowStart, globalColStart, kt,
              As[nextLoad], Bs[nextLoad],
              tRow, tCol, THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_N);

    __syncthreads(); // Wait for next tile to be fully loaded

    // Compute on previously loaded tile
    compute_tile_16(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

    __syncthreads(); // Ensure compute done before reusing shared memory

    // Switch buffers
    curLoad = nextLoad;
    curCompute = curLoad;
  }

  // Compute on the last tile
  compute_tile_16(As[curCompute], Bs[curCompute], threadResults, tRow, tCol);

  // Write results back to C
  #pragma unroll
  for (int i = 0; i < TM; i++) {
    int cRow = globalRowStart + tRow * TM + i;
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      int cCol = globalColStart + tCol * TN + j;
      if (cRow < M && cCol < M) {
        C[cRow * M + cCol] = threadResults[i * TN + j];
      }
    }
  }
}
//********************************************************/
//_________________________________________________________



// Host function that launches the kernel
// We assume that A, B, C are device arrays of pointers to device matrices.
void batched_gemm_kernel
( 
    const int M, 
    const int N, 
    const double **A0, 
    const double **B0, 
    double **C0
)
{


    dim3 blockDim;
    dim3 gridDim;

    // dim3 blockDim(16, 16);
    // dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
    //              (M + blockDim.y - 1) / blockDim.y,
    //              N);

    size_t matrix_size = M * M * sizeof(double);

    // Allocate device memory for matrices
    double *d_A_data, *d_B_data, *d_C_data;
    cudaMalloc((void **)&d_A_data, N * matrix_size);
    cudaMalloc((void **)&d_B_data, N * matrix_size);
    cudaMalloc((void **)&d_C_data, N * matrix_size);

    const double **d_A_array, **d_B_array;
          double **d_C_array;
    cudaMalloc((void **)&d_A_array, N * sizeof(double *));
    cudaMalloc((void **)&d_B_array, N * sizeof(double *));
    cudaMalloc((void **)&d_C_array, N * sizeof(double *));

    // Allocate host memory for device pointers
    double **h_A_list = (double **)malloc(N * sizeof(double *));
    double **h_B_list = (double **)malloc(N * sizeof(double *));
    double **h_C_list = (double **)malloc(N * sizeof(double *));

    // Copy input matrices to device memory
    for (int i = 0; i < N; i++) {
        cudaMemcpy(d_A_data + i * M * M, A0[i], matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_data + i * M * M, B0[i], matrix_size, cudaMemcpyHostToDevice);
    }

    // Prepare device pointers for the batch
    for (int i = 0; i < N; i++) {
        h_A_list[i] = d_A_data + i * M * M;
        h_B_list[i] = d_B_data + i * M * M;
        h_C_list[i] = d_C_data + i * M * M;
    }

    // Copy the array of device pointers to the GPU
    cudaMemcpy(d_A_array, h_A_list, N * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array, h_B_list, N * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array, h_C_list, N * sizeof(double *), cudaMemcpyHostToDevice);

    // Launch the kernel
    
    // Decide which kernel to launch based on M
    if (M > 94){
        // 64x64 Tile Configuration
        blockDim = dim3(16, 16);
        gridDim = dim3(CEIL_DIV(M, BM64), CEIL_DIV(M, BN64), N);

        square_dgemm_kernel_2d_blocktiling_64<<<gridDim, blockDim>>>(M, N, d_A_array, d_B_array, d_C_array);
        // printf("Launched 64x64 tile kernel.\n");
    } 
    else if(M > 40){
        // 32x32 Tile Configuration
        blockDim = dim3(8, 8);
        gridDim = dim3(CEIL_DIV(M, BM32), CEIL_DIV(M, BN32), N);

        square_dgemm_kernel_2d_blocktiling_32<<<gridDim, blockDim>>>(M, N, d_A_array, d_B_array, d_C_array);
        // printf("Launched 32x32 tile kernel.\n");
    } 
    else{
        // 16x16 Tile Configuration
        blockDim = dim3(4, 4);
        gridDim = dim3(CEIL_DIV(M, BM16), CEIL_DIV(M, BN16), N);

        square_dgemm_kernel_2d_blocktiling_16<<<gridDim, blockDim>>>(M, N, d_A_array, d_B_array, d_C_array);
        // printf("Launched 16x16 tile kernel.\n");
    }

    cudaDeviceSynchronize();

    // Copy the result back to the host
    for (int i = 0; i < N; i++) {
        cudaMemcpy(C0[i], d_C_data + i * M * M, matrix_size, cudaMemcpyDeviceToHost);
    }

    // Free device memory
    cudaFree(d_A_data);
    cudaFree(d_B_data);
    cudaFree(d_C_data);
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);

    // Free host memory
    free(h_A_list);
    free(h_B_list);
    free(h_C_list);
}
