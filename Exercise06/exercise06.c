#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 1000000
#define STRIP_SIZE 256

int main() {
    /* Allocate arrays dynamically to avoid stack overflow */
    double *A = (double *)malloc((size_t)N * sizeof(double));
    double *B = (double *)malloc((size_t)N * sizeof(double));
    double *C = (double *)malloc((size_t)N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for size N = %d\n", N);
        free(A);
        free(B);
        free(C);
        return 1;
    }

    /* Initialize arrays deterministically */
    for (int i = 0; i < N; i++) {
        A[i] = (double)(i % 1000) * 0.5;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    /* Query number of threads that will participate */
    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
        }
    }

    printf("=================================================================\n");
    printf("Exercise 06: Strip-Mined Array Multiplication (C = A * B)\n");
    printf("Array Size (N):     %d elements\n", N);
    printf("Strip Size:         %d elements (%zu bytes per strip)\n",
           STRIP_SIZE, STRIP_SIZE * sizeof(double));
    printf("Total Strips:       %d (%d full strips + 1 partial strip of %d)\n",
           (N + STRIP_SIZE - 1) / STRIP_SIZE,
           N / STRIP_SIZE,
           N % STRIP_SIZE);
    printf("OpenMP Threads:     %d\n", num_threads);
    printf("=================================================================\n");

    /* Time only the strip-mined computation */
    double tstart = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int start = 0; start < N; start += STRIP_SIZE) {
        int end = start + STRIP_SIZE;
        if (end > N) {
            end = N;
        }

        #pragma omp simd
        for (int i = start; i < end; i++) {
            C[i] = A[i] * B[i];
        }
    }

    double tstop = omp_get_wtime();
    double tcalc = tstop - tstart;

    printf("Computation Time:   %.6f seconds (%.3f ms)\n", tcalc, tcalc * 1000.0);

    /* Validate correctness against expected values */
    int errors = 0;
    for (int i = 0; i < N; i++) {
        double expected = (double)(i % 1000);
        if (fabs(C[i] - expected) > 1e-9) {
            if (errors < 5) {
                printf("Mismatch at index %d: expected %.4f, got %.4f\n",
                       i, expected, C[i]);
            }
            errors++;
        }
    }

    /* Print sample outputs demonstrating element-wise calculation */
    printf("\nSample Elements Check:\n");
    int sample_indices[] = {0, 100, 500000, 999935, 999936, N - 1};
    int num_samples = (int)(sizeof(sample_indices) / sizeof(sample_indices[0]));
    for (int s = 0; s < num_samples; s++) {
        int idx = sample_indices[s];
        printf("  Index %6d: A[%6d]=%7.2f, B[%6d]=%4.2f => C[%6d]=%7.2f (expected %7.2f)\n",
               idx, idx, A[idx], idx, B[idx], idx, C[idx], (double)(idx % 1000));
    }

    if (errors == 0) {
        printf("\nValidation Result:  SUCCESS (All %d elements match expected results)\n", N);
    } else {
        printf("\nValidation Result:  FAILED (%d mismatches detected)\n", errors);
    }
    printf("=================================================================\n");

    /* Free dynamically allocated memory */
    free(A);
    free(B);
    free(C);

    return (errors == 0) ? 0 : 1;
}
