const char* dgemm_desc = "My awesome dgemm.";

//best verrsion? timing 17,18
#include <immintrin.h> // Include AVX-512 intrinsics
#include <omp.h>
#include <stdlib.h>    // For aligned memory allocation if needed

#define BLOCK_SIZE 112
#define M_BLOCK_SIZE 16
#define N_BLOCK_SIZE 16

#define min(a,b) (((a)<(b))?(a):(b))

/* Optimized do_block with enhanced general case handling using AVX-512 masks */
static void do_block(const int lda, const int ldb, const int ldc,
                     const int M_block, const int N_block, const int K_block,
                     const double *restrict A_block, const double *restrict B_block, double *restrict C_block)
{
    if (M_block == M_BLOCK_SIZE && N_block == N_BLOCK_SIZE)
    {
        // Optimized kernel for 16x16 block using AVX-512

        // Arrays to hold the C registers (16 columns, each with two __m512d for 16 elements)
        __m512d c[16][2];

        // Load C columns
        for (int j = 0; j < 16; ++j)
        {
            c[j][0] = _mm512_loadu_pd(&C_block[j * ldc + 0]); // Lower 8 elements
            c[j][1] = _mm512_loadu_pd(&C_block[j * ldc + 8]); // Upper 8 elements
        }

        for (int k = 0; k < K_block; ++k)
        {
            // Load A column
            __m512d a_vec0 = _mm512_loadu_pd(&A_block[k * lda + 0]);
            __m512d a_vec1 = _mm512_loadu_pd(&A_block[k * lda + 8]);

            // Perform FMA for each column j
            for (int j = 0; j < 16; ++j)
            {
                double b_scalar = B_block[k + j * ldb];
                __m512d b_val = _mm512_set1_pd(b_scalar);
                c[j][0] = _mm512_fmadd_pd(a_vec0, b_val, c[j][0]);
                c[j][1] = _mm512_fmadd_pd(a_vec1, b_val, c[j][1]);
            }
        }

        // Store the updated C columns
        for (int j = 0; j < 16; ++j)
        {
            _mm512_storeu_pd(&C_block[j * ldc + 0], c[j][0]);
            _mm512_storeu_pd(&C_block[j * ldc + 8], c[j][1]);
        }
    }
    else if (M_block == M_BLOCK_SIZE)
    {
        // Vectorized over M dimension using AVX-512
        for (int j = 0; j < N_block; ++j)
        {
            __m512d c_vec0 = _mm512_loadu_pd(&C_block[j * ldc + 0]);
            __m512d c_vec1 = _mm512_loadu_pd(&C_block[j * ldc + 8]);
            for (int k = 0; k < K_block; ++k)
            {
                __m512d a_vec0 = _mm512_loadu_pd(&A_block[k * lda + 0]);
                __m512d a_vec1 = _mm512_loadu_pd(&A_block[k * lda + 8]);
                double b_scalar = B_block[k + j * ldb];
                __m512d b_val = _mm512_set1_pd(b_scalar);
                c_vec0 = _mm512_fmadd_pd(a_vec0, b_val, c_vec0);
                c_vec1 = _mm512_fmadd_pd(a_vec1, b_val, c_vec1);
            }
            _mm512_storeu_pd(&C_block[j * ldc + 0], c_vec0);
            _mm512_storeu_pd(&C_block[j * ldc + 8], c_vec1);
        }
    }
    else
    {
        // Enhanced general case with hierarchical vectorization and masking

        for (int j = 0; j < N_block; ++j)
        {
            int i = 0;

            // Process blocks of 16 elements
            for (; i <= M_block - 16; i += 16)
            {
                // Load C in two vectors
                __m512d c0 = _mm512_loadu_pd(&C_block[i + j * ldc + 0]);
                __m512d c1 = _mm512_loadu_pd(&C_block[i + j * ldc + 8]);

                // Accumulate over K
                for (int k = 0; k < K_block; ++k)
                {
                    __m512d a0 = _mm512_loadu_pd(&A_block[i + k * lda + 0]);
                    __m512d a1 = _mm512_loadu_pd(&A_block[i + k * lda + 8]);
                    double b_scalar = B_block[k + j * ldb];
                    __m512d b_val = _mm512_set1_pd(b_scalar);
                    c0 = _mm512_fmadd_pd(a0, b_val, c0);
                    c1 = _mm512_fmadd_pd(a1, b_val, c1);
                }

                // Store the results
                _mm512_storeu_pd(&C_block[i + j * ldc + 0], c0);
                _mm512_storeu_pd(&C_block[i + j * ldc + 8], c1);
            }

            // Process remaining 8 elements
            for (; i <= M_block - 8; i += 8)
            {
                __m512d c_vec = _mm512_loadu_pd(&C_block[i + j * ldc]);
                for (int k = 0; k < K_block; ++k)
                {
                    __m512d a_vec = _mm512_loadu_pd(&A_block[i + k * lda]);
                    double b_scalar = B_block[k + j * ldb];
                    __m512d b_val = _mm512_set1_pd(b_scalar);
                    c_vec = _mm512_fmadd_pd(a_vec, b_val, c_vec);
                }
                _mm512_storeu_pd(&C_block[i + j * ldc], c_vec);
            }

            // Handle remaining elements with AVX-512 masking
            if (i < M_block)
            {
                int remaining = M_block - i;
                __mmask8 mask = (1 << remaining) - 1; // Mask for remaining elements (up to 8)

                // Load C with mask
                __m512d c_partial = _mm512_mask_load_pd(_mm512_setzero_pd(), mask, &C_block[i + j * ldc]);
                for (int k = 0; k < K_block; ++k)
                {
                    __m512d a_partial = _mm512_mask_load_pd(_mm512_setzero_pd(), mask, &A_block[i + k * lda]);
                    double b_scalar = B_block[k + j * ldb];
                    __m512d b_val = _mm512_set1_pd(b_scalar);
                    c_partial = _mm512_fmadd_pd(a_partial, b_val, c_partial);
                }
                _mm512_mask_store_pd(&C_block[i + j * ldc], mask, c_partial);
            }
        }
    }
}

/* This routine performs a dgemm operation
 * C := C + A * B
 * where A, B, and C are M-by-M matrices stored in column-major format.
 */
void square_dgemm(const int M, const double *restrict A, const double *restrict B, double *restrict C)
{
    // Define a threshold for small matrices
    const int SMALL_SIZE_THRESHOLD = 64;

    // Loop order changed to KJI for better cache performance
    for (int k = 0; k < M; k += BLOCK_SIZE)
    {
        // Block dimension for K
        int K_block = min(BLOCK_SIZE, M - k);

        // Determine if the current block should be parallelized
        // Parallelize only if M is larger than the threshold
        // #pragma omp parallel for collapse(2) schedule(guided) if(M > SMALL_SIZE_THRESHOLD)
        #pragma omp parallel num_threads(1)
        for (int j = 0; j < M; j += N_BLOCK_SIZE)
        {
            for (int i = 0; i < M; i += M_BLOCK_SIZE)
            {
                // Block dimensions for M and N
                int M_block = min(M_BLOCK_SIZE, M - i);
                int N_block = min(N_BLOCK_SIZE, M - j);

                // Calculate the starting pointers for A, B, and C
                const double *A_block = &A[i + k * M];
                const double *B_block = &B[k + j * M];
                double *C_block = &C[i + j * M];

                // Perform individual block dgemm
                do_block(M, M, M, M_block, N_block, K_block, A_block, B_block, C_block);
            }
        }
    }
}
