#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#ifndef COMPILER
#define COMPILER "unknown"
#endif
#ifndef FLAGS
#define FLAGS "unknown"
#endif

extern void batched_gemm_kernel(const int M, const int N, const double **A0, const double **B0, double **C0);
extern const char *dgemm_desc;

#define MIN_RUNS 7
#define MIN_SECS 0.8

// #define DEBUG_RUN

// const int test_sizes[] = {
//     31,
//     32,
//     96,
//     97,
//     127,
//     128,
//     129,
//     191,192,229,
//     255,256,257,
//     319,320,321,
// #if !defined(DEBUG_RUN)
//     // 417,479,480,
//     // 511,512,639,640,
//     // 767,768,769,
//     // 1023,
//     // 1024,1025,
//     // 1525,1526,1527,
//     // 2024,
//     // 2025,2525,
//     // 2526,2527,
//     // 3000,
//     // 4000,5000
// #endif
// };

// const int test_sizes[] = {
//     128,
//     128,
//     128,
//     128,
//     128,
//     128,
//     128,
//     128,
//     128,
//     128,
//     128,
//     128,
//     128,
//     128,
//     128,
//     128
// };

// const int batched_sizes[] = {
//     31,
//     32,
//     96,
//     97,
//     127,
//     128,
//     129,
//     191,192,229,
//     255,256,257,
//     319,320,321
// };

const int test_sizes[] = {
31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121,
126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 201,
206, 211, 216, 221, 226, 231, 236, 241, 246, 251, 256, 261, 266, 271, 276, 281,
286, 291, 296, 301, 306, 311, 316, 321, 326, 331, 336, 341, 346, 351, 356
};

const int batched_sizes[] = {
31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121,
126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 201,
206, 211, 216, 221, 226, 231, 236, 241, 246, 251, 256, 261, 266, 271, 276, 281,
286, 291, 296, 301, 306, 311, 316, 321, 326, 331, 336, 341, 346, 351, 356
};


#define N_SIZES (sizeof(test_sizes) / sizeof(int))
#define MAX_SIZE 356u
#define DEFAULT_BATCH_SIZE 15
#define MAX_BATCH_SIZE 356u


// GPU kernel to compute ground truth
__global__ void gpu_batched_mmm_ground_truth(
    double **A_list,
    double **B_list,
    double **C_list, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int i = blockIdx.z; // which matrix in the batch
    if (i < N && row < M && col < M)
    {
        const double *A = A_list[i];
        const double *B = B_list[i];
        double *C = C_list[i];

        double sum = 0.0;
        for (int k = 0; k < M; ++k)
        {
            sum += A[row * M + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

// CUDA kernel to calculate error bounds
__global__ void gpu_batched_calculate_error_bound(
    double **A_list,
    double **B_list,
    double **C_ref_list,
    double *error_bounds_list,
    int M, int N, double epsilon)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.z;

    if (i < N && row < M && col < M)
    {
        const double *A = A_list[i];
        const double *B = B_list[i];

        double dotprod = 0.0;
        double errorbound = 0.0;

        for (int k = 0; k < M; ++k)
        {
            double prod = A[row * M + k] * B[k * M + col];
            dotprod += prod;
            errorbound += fabs(prod);
        }

        error_bounds_list[i * M * M + row * M + col] = errorbound * (M * epsilon);
    }
}

void diff_batched_dgemm(const int M, const int N, double **C, double **C_ref)
{
    FILE *fp_our = fopen("dump_our_batched.txt", "w");
    FILE *fp_ref = fopen("dump_ref_batched.txt", "w");
    FILE *fp_diff = fopen("dump_diff_batched.txt", "w");

    if (!fp_our || !fp_ref || !fp_diff)
    {
        fprintf(stderr, "Error opening output files in diff_batched_dgemm.\n");
        exit(-1);
    }

    for (int i = 0; i < N; i++)
    {
        fprintf(fp_our, "Matrix %d\n", i);
        fprintf(fp_ref, "Matrix %d\n", i);
        fprintf(fp_diff, "Matrix %d\n", i);
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < M; c++)
            {
                double user_val = C[i][r * M + c];
                double ref_val = C_ref[i][r * M + c];
                double diff = user_val - ref_val;

                fprintf(fp_our, " %g", user_val);
                fprintf(fp_ref, " %g", ref_val);
                fprintf(fp_diff, " % 0.0e", diff);
            }
            fprintf(fp_our, "\n");
            fprintf(fp_ref, "\n");
            fprintf(fp_diff, "\n");
        }
    }

    fclose(fp_diff);
    fclose(fp_ref);
    fclose(fp_our);
}

void validate_batched_gemm(int M, int N, double **A, double **B, double **C)
{
    // Clear C and run the user's batched kernel
    for (int i = 0; i < N; i++)
        memset(C[i], 0, M * M * sizeof(double));
    batched_gemm_kernel(M, N, (const double **)A, (const double **)B, C);

    double **C_ref = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
        C_ref[i] = (double *)malloc(M * M * sizeof(double));

    double **h_A_list = (double **)malloc(N * sizeof(double *));
    double **h_B_list = (double **)malloc(N * sizeof(double *));
    double **h_C_ref_list = (double **)malloc(N * sizeof(double *));

    size_t mat_size = M * M * sizeof(double);

    double *d_A_data, *d_B_data, *d_C_ref_data;
    cudaMalloc((void **)&d_A_data, N * mat_size);
    cudaMalloc((void **)&d_B_data, N * mat_size);
    cudaMalloc((void **)&d_C_ref_data, N * mat_size);

    for (int i = 0; i < N; i++)
    {
        cudaMemcpy(d_A_data + i * M * M, A[i], mat_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_data + i * M * M, B[i], mat_size, cudaMemcpyHostToDevice);
    }

    double **d_A_array, **d_B_array, **d_C_ref_array;
    cudaMalloc((void **)&d_A_array, N * sizeof(double *));
    cudaMalloc((void **)&d_B_array, N * sizeof(double *));
    cudaMalloc((void **)&d_C_ref_array, N * sizeof(double *));

    for (int i = 0; i < N; i++)
    {
        h_A_list[i] = d_A_data + i * M * M;
        h_B_list[i] = d_B_data + i * M * M;
        h_C_ref_list[i] = d_C_ref_data + i * M * M;
    }

    cudaMemcpy(d_A_array, h_A_list, N * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array, h_B_list, N * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ref_array, h_C_ref_list, N * sizeof(double *), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   N);

    gpu_batched_mmm_ground_truth<<<numBlocks, threadsPerBlock>>>(d_A_array, d_B_array, d_C_ref_array, M, N);
    cudaDeviceSynchronize();

    double *d_error_bounds_list;
    cudaMalloc((void **)&d_error_bounds_list, N * mat_size);
    double *h_error_bounds_list = (double *)malloc(N * mat_size);

    gpu_batched_calculate_error_bound<<<numBlocks, threadsPerBlock>>>(d_A_array, d_B_array, d_C_ref_array, d_error_bounds_list, M, N, DBL_EPSILON);
    cudaDeviceSynchronize();

    cudaMemcpy(h_error_bounds_list, d_error_bounds_list, N * mat_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        cudaMemcpy(C_ref[i], h_C_ref_list[i], mat_size, cudaMemcpyDeviceToHost);

    // Validation
    for (int i = 0; i < N; i++)
    {
        for (int row = 0; row < M; row++)
        {
            for (int col = 0; col < M; col++)
            {
                double computed_value = C_ref[i][row * M + col];
                double user_value = C[i][row * M + col];
                double err = fabs(computed_value - user_value);
                double error_bound = h_error_bounds_list[i * M * M + row * M + col];
                if (err > 3.0 * error_bound)
                {
                    printf("Validation failed for batch %d at (%d, %d):\n", i, row, col);
                    printf("Computed: %g, Expected: %g, Error: %g, Bound: %g\n",
                           user_value, computed_value, err, 3.0 * error_bound);
                    diff_batched_dgemm(M, N, C, C_ref);
                    exit(-1);
                }
            }
        }
    }

    // Cleanup
    free(h_A_list);
    free(h_B_list);
    free(h_C_ref_list);
    for (int i = 0; i < N; i++)
        free(C_ref[i]);
    free(C_ref);
    free(h_error_bounds_list);

    cudaFree(d_A_data);
    cudaFree(d_B_data);
    cudaFree(d_C_ref_data);
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_ref_array);
    cudaFree(d_error_bounds_list);
}

/*
 * Measure time for batched_gemm_kernel.
 */
double time_batched_gemm(int M, int N, double **A, double **B, double **C)
{
    double secs = -1.0;
    double mflops_sec;
    int num_iterations = MIN_RUNS;
    while (secs < MIN_SECS) {
        for (int i = 0; i < N; i++)
            memset(C[i], 0, M * M * sizeof(double));
        double start = omp_get_wtime();
        for (int i = 0; i < num_iterations; ++i) {
            batched_gemm_kernel(M, N, (const double **)A, (const double **)B, C);
        }
        double finish = omp_get_wtime();
        double ops = 2.0 * num_iterations * N * (double)M * (double)M * (double)M / 1.0e6;
        secs = finish - start;
        mflops_sec = ops / secs;
        num_iterations *= 2;
    }
    return mflops_sec;
}

int main(int argc, char **argv)
{
    if (argc > 2) {
        fprintf(stderr, "Usage: batched_matmul [csv]\n");
        exit(2);
    }

    FILE *fp;
    if (argc == 1) {
        const char *exename = argv[0];
        const char *s = exename + strlen(exename);
        for (; s != exename && *s != '-' && *s != '/'; --s);
        char *fname = (char *)malloc(strlen(s) + strlen("timing_batched.csv") + 1);
        strcpy(fname, "timing_batched");
        strcat(fname, s);
        strcat(fname, ".csv");
        fp = fopen(fname, "w");
        free(fname);
    } else {
        fp = fopen(argv[1], "w");
    }

    if (!fp) {
        fprintf(stderr, "Could not open '%s' for output\n", argv[1]);
        exit(3);
    }


    printf("Compiler:\t%s\nOptions:\t%s\nTag: %s\n\n",
           COMPILER, FLAGS, dgemm_desc);

    fprintf(fp, "M,batch_size,mflop\n");

    for (int idx = 0; idx < N_SIZES; ++idx) {
        int M = test_sizes[idx];
        int N = batched_sizes[idx];
        printf("M: %d, N: %d \n", M, N);

        double **A = (double **)malloc(N * sizeof(double *));
        double **B = (double **)malloc(N * sizeof(double *));
        double **C = (double **)malloc(N * sizeof(double *));

        for (int i = 0; i < N; i++) {
            A[i] = (double *)malloc(M * M * sizeof(double));
            B[i] = (double *)malloc(M * M * sizeof(double));
            C[i] = (double *)malloc(M * M * sizeof(double));
        }

        srand48(0);
        for (int i = 0; i < N; i++) {
            for (int el = 0; el < M*M; el++) {
                A[i][el] = drand48();
                B[i][el] = drand48();
            }
        }

        validate_batched_gemm(M, N, A, B, C);
        double perf = time_batched_gemm(M, N, A, B, C);
        fprintf(fp, "%d,%d,%lg\n", M, N, perf);

        for (int i = 0; i < N; i++) {
            free(A[i]);
            free(B[i]);
            free(C[i]);
        }
        free(A);
        free(B);
        free(C);
    }

    fclose(fp);
    return 0;
}