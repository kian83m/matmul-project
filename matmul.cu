#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifndef COMPILER
#  define COMPILER "unknown"
#endif
#ifndef FLAGS
#  define FLAGS "unknown"
#endif

// #define DEBUG_RUN

extern const char* dgemm_desc;
// extern void square_dgemm();
void square_dgemm(int M, const double *A, const double *B, double *C);

/*
  We try to run enough iterations to get reasonable timings.  The matrices
  are multiplied at least MIN_RUNS times.  If that doesn't take MIN_SECS
  seconds, then we double the number of iterations and try again.

  You may want to modify these to speed debugging...
*/
#define MIN_RUNS 4
/* #define MIN_SECS 1.0 */
#define MIN_SECS 0.25

/*
  Note the strange sizes...  You'll see some interesting effects
  around some of the powers-of-two.
*/
const int test_sizes[] = {
    // 80, 96, 112, 128, 144, 192, 200, 208,
    31, 32, 96, 97, 127, 128, 129, 191, 192, 229,
#if defined(DEBUG_RUN)
# define MAX_SIZE 229u
// # define MAX_SIZE 208u
#else
    255, 256, 257, 319, 320, 321, 417, 479, 480, 511, 512, 639, 640,
    767, 768, 769, 1023, 1024, 1025, 1525, 1526, 1527,
    2024, 2025, 2525, 2526, 2527, 3000, 4000, 5000
# define MAX_SIZE 5000u
#endif
};

#define N_SIZES (sizeof (test_sizes) / sizeof (int))


/* --
 * Initialize A to random numbers (A is MAX_SIZE * MAX_SIZE)
 */
void matrix_init(double *A)
{
    for (int i = 0; i < MAX_SIZE*MAX_SIZE; ++i) 
        A[i] = drand48();
}


/* --
 * Zero out C (which is MAX_SIZE * MAX_SIZE)
 */
void matrix_clear(double *C)
{
    memset(C, 0, MAX_SIZE * MAX_SIZE * sizeof(double));
}

/* --
 * Check that C = A*B to within roundoff error.
 *
 * We use the fact that dot products satisfy the error bound
 *
 *   float(sum a_i * b_i) = sum a_i * b_i * (1 + delta_i)
 *
 * where delta_i <= n * epsilon.  In order to check your matrix
 * multiply, we compute each element in turn and make sure that
 * your product is within three times the given error bound.
 * We make it three times because there are three sources of
 * error:
 *
 *  - the roundoff error in your multiply
 *  - the roundoff error in our multiply
 *  - the roundoff error in computing the error bound
 *
 *  That last source of error is not so significant, but that's a
 *  story for another day.
 */
void diff_dgemm(const int M, const double *C_user, const double *C_ground_truth)
{
    FILE* fp_our  = fopen("dump_our.txt", "w");
    FILE* fp_ref  = fopen("dump_ref.txt", "w");
    FILE* fp_diff = fopen("dump_diff.txt", "w");

    if (!fp_our || !fp_ref || !fp_diff) {
        fprintf(stderr, "Error opening output files in diff_dgemm.\n");
        exit(-1);
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double user_value = C_user[i * M + j];
            double ground_truth_value = C_ground_truth[i * M + j];
            double diff = user_value - ground_truth_value;

            fprintf(fp_our,  " %g", user_value);
            fprintf(fp_ref,  " %g", ground_truth_value);
            fprintf(fp_diff, " % 0.0e", diff);
        }
        fprintf(fp_our, "\n");
        fprintf(fp_ref, "\n");
        fprintf(fp_diff, "\n");
    }

    fclose(fp_diff);
    fclose(fp_ref);
    fclose(fp_our);
}
// void diff_dgemm(const int M, const double *A, const double *B, double *C)
// {
//     FILE* fp_our  = fopen("dump_our.txt", "w");
//     FILE* fp_ref  = fopen("dump_ref.txt", "w");
//     FILE* fp_diff = fopen("dump_diff.txt", "w");
//     matrix_clear(C);
//     square_dgemm(M, A, B, C);
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < M; ++j) {
//             double dotprod = 0;
//             for (int k = 0; k < M; ++k) {
//                 // double prod = A[k*M + i] * B[j*M + k];
//                 double prod = A[i*M + k] * B[k*M + j];
//                 dotprod += prod;
//             }
//             // fprintf(fp_our,  " %g", C[j*M+i]);
//             fprintf(fp_our,  " %g", C[i*M+j]);
//             fprintf(fp_ref,  " %g", dotprod);
//             // fprintf(fp_diff, " % 0.0e", C[j*M+i]-dotprod);
//             fprintf(fp_diff, " % 0.0e", C[i*M+j]-dotprod);
//         }
//         fprintf(fp_our, "\n");
//         fprintf(fp_ref, "\n");
//         fprintf(fp_diff, "\n");
//     }
//     fclose(fp_diff);
//     fclose(fp_ref);
//     fclose(fp_our);
// }

// GPU kernel to compute matrix multiplication (C = A * B)
__global__ void gpu_mmm_ground_truth(const double* A, const double* B, double* C, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= M) return;

    double sum = 0.0;
    for (int k = 0; k < M; ++k) {
        sum += A[row * M + k] * B[k * M + col];
    }

    C[row * M + col] = sum;
}

// CUDA kernel to calculate error bounds and validate results
__global__ void calculate_error_bound(
    const double* A, const double* B, const double* C, double* error_bounds, 
    int M, double epsilon) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= M) return;

    double dotprod = 0.0;
    double errorbound = 0.0;

    for (int k = 0; k < M; ++k) {
        double prod = A[row * M + k] * B[k * M + col];
        dotprod += prod;
        errorbound += fabs(prod);
    }

    error_bounds[row * M + col] = errorbound * (M * epsilon);
}

// void validate_dgemm(int M, const double* A, const double* B, double* C) {
//     double *d_A, *d_B, *d_C, *d_error_bounds;
//     size_t size = M * M * sizeof(double);

//     // Allocate device memory
//     cudaMalloc((void**)&d_A, size);
//     cudaMalloc((void**)&d_B, size);
//     cudaMalloc((void**)&d_C, size);
//     cudaMalloc((void**)&d_error_bounds, size);

//     // Copy matrices from host to device
//     cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

//     // Compute ground truth using GPU
//     gpu_mmm_ground_truth<<<dim3((M + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(d_A, d_B, d_C, M);
//     cudaDeviceSynchronize();

//     // Launch error bound kernel
//     calculate_error_bound<<<dim3((M + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(d_A, d_B, d_C, d_error_bounds, M, DBL_EPSILON);
//     cudaDeviceSynchronize();

//     // Validation
//     for(int row = 0; row < M; row++){
//         for(int col = 0; col < M; col++){
//             double computed_value = d_C[row * M + col];
//             double dotprod = C[row * M + col];
//             double err = fabs(computed_value - dotprod);
//             if (err > 3.0 * d_error_bounds[row * M + col]) {
//                 printf("Validation failed at (%d, %d):\n", row, col);
//                 printf("Computed: %g, Expected: %g, Error: %g, Bound: %g\n",
//                     computed_value, dotprod, err, 3.0 * d_error_bounds[row * M + col]);
//                 diff_dgemm(M, A, B, C);
//             }
//         }
//     }

//     // Copy results back to host
//     cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
//     cudaFree(d_error_bounds);
// }

void validate_dgemm(int M, const double* A, const double* B, double* C) {
    // First, compute C using your own implementation
    matrix_clear(C);
    square_dgemm(M, A, B, C);

    double *d_A, *d_B, *d_C_ground_truth, *d_error_bounds;
    size_t size = M * M * sizeof(double);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C_ground_truth, size);
    cudaMalloc((void**)&d_error_bounds, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Compute ground truth using GPU
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gpu_mmm_ground_truth<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C_ground_truth, M);
    cudaDeviceSynchronize();

    // Launch error bound kernel
    calculate_error_bound<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C_ground_truth, d_error_bounds, M, DBL_EPSILON);
    cudaDeviceSynchronize();

    // Allocate host memory to receive data from device
    double* h_C_ground_truth = (double*)malloc(size);
    double* h_error_bounds = (double*)malloc(size);

    // Copy results from device to host
    cudaMemcpy(h_C_ground_truth, d_C_ground_truth, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_error_bounds, d_error_bounds, size, cudaMemcpyDeviceToHost);

    // Validation
    for(int row = 0; row < M; row++){
        for(int col = 0; col < M; col++){
            double computed_value = h_C_ground_truth[row * M + col]; // Ground truth
            double user_value = C[row * M + col]; // Your computed value
            double err = fabs(computed_value - user_value);
            double error_bound = h_error_bounds[row * M + col];
            if (err > 3.0 * error_bound) {
                printf("Validation failed at (%d, %d):\n", row, col);
                printf("Computed: %g, Expected: %g, Error: %g, Bound: %g\n",
                    user_value, computed_value, err, 3.0 * error_bound);
                // diff_dgemm(M, A, B, C);
                diff_dgemm(M, C, h_C_ground_truth);
                exit(-1);
            }
        }
    }

    // Free host and device memory
    free(h_C_ground_truth);
    free(h_error_bounds);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_ground_truth);
    cudaFree(d_error_bounds);
}


/* --
 * Compute a MFlop/s rate for C += A*B.
 *
 * The code runs the multiplication repeatedly in a loop MIN_RUNS times,
 * then doubles the loop time if it did not take MIN_SECS to perform the
 * run.  This helps us get around the limits of timer resolution.
 */
double time_dgemm(const int M, const double *A, const double *B, double *C)
{
    double secs = -1.0;
    double mflops_sec;
    int num_iterations = MIN_RUNS;
    while (secs < MIN_SECS) {
        matrix_clear(C);
        double start = omp_get_wtime();
        for (int i = 0; i < num_iterations; ++i) {
            square_dgemm(M, A, B, C);
        }
        double finish = omp_get_wtime();
        double mflops = 2.0 * num_iterations * M * M * M / 1.0e6;
        secs = finish-start;
        mflops_sec = mflops / secs;
        num_iterations *= 2;
    }
    return mflops_sec;
}

int main(int argc, char** argv) {

    if (argc > 2) {
        fprintf(stderr, "Usage: matmul [csv]\n");
        exit(2);
    }
    
    FILE* fp;
    if (argc == 1) {
        const char* exename = argv[0];
        const char* s = exename + strlen(exename);
        for (; s != exename && *s != '-' && *s != '/'; --s);
        char* fname = (char*) malloc(strlen(s) + strlen("timing.csv") + 1);
        strcpy(fname, "timing");
        strcat(fname, s);
        strcat(fname, ".csv");
        fp = fopen(fname, "w");
        free(fname);
    } else 
        fp = fopen(argv[1], "w");
    
    if (!fp) {
        fprintf(stderr, "Could not open '%s' for output\n", argv[1]);
        exit(3);
    }

    double* A = (double*) malloc(MAX_SIZE * MAX_SIZE * sizeof(double));
    double* B = (double*) malloc(MAX_SIZE * MAX_SIZE * sizeof(double));
    double* C = (double*) malloc(MAX_SIZE * MAX_SIZE * sizeof(double));

    matrix_init(A);
    matrix_init(B);

    printf("Compiler:\t%s\nOptions:\t%s\nDescription:\t%s\n\n",
           COMPILER, FLAGS, dgemm_desc);

    fprintf(fp, "size,mflop\n");
    for (int i = 0; i < N_SIZES; ++i) {
        const int M = test_sizes[i];
        validate_dgemm(M, A, B, C);
        fprintf(fp, "%u,%lg\n", M, time_dgemm(M, A, B, C));
    }

    free(A);
    free(B);
    free(C);

    fclose(fp);
    return 0;
}
