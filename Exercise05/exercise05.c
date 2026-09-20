#include <omp.h>
#include <stdio.h>


int fib(int n) {
    int i, j;
    if (n < 2)
        return n;

    #pragma omp task shared(i) firstprivate(n)
    i = fib(n - 1);

    #pragma omp task shared(j) firstprivate(n)
    j = fib(n - 2);

    #pragma omp taskwait

    return i + j;
}

int main() {
    int n = 30;
    int result;
    int nthreads;
    double tstart, tstop, tcalc;

    printf("Computing Fibonacci(%d) using OpenMP task parallelization\n", n);
    printf("------------------------------------------------------------\n");

    tstart = omp_get_wtime();

    #pragma omp parallel shared(result, nthreads)
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
            result = fib(n);
        }
    }

    tstop = omp_get_wtime();
    tcalc = tstop - tstart;

    printf("fib(%d)      = %d\n", n, result);
    printf("Threads      = %d\n", nthreads);
    printf("Time elapsed = %.6f seconds\n", tcalc);

    return 0;
}
