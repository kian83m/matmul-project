const char *dgemm_desc = "Basic, three-loop dgemm.";

void square_dgemm(const int M,
                  const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < M; ++j)
        // // Original version
        // {
        //     double cij = C[j * M + i];
        //     for (k = 0; k < M; ++k)
        //         cij += A[k * M + i] * B[j * M + k];
        //     C[j * M + i] = cij;
        // }

        // new row major order version
        {
            // double cij = C[i * M + j];
            double cij = 0;
            for (k = 0; k < M; ++k)
                cij += A[i * M + k] * B[k * M + j];
            C[i * M + j] = cij;
        }
    }
}
