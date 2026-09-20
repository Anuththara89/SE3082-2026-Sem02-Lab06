#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define NROWS 1000
#define NCOLS 1000
#define TOTAL_POINTS (NROWS * NCOLS)
#define MAX_ITER 2000

#define R_MIN -2.0
#define R_MAX  0.5
#define I_MIN  0.0
#define I_MAX  1.125

/*
 * Standard Mandelbrot iteration for a single point c = cr + i*ci.
 * Returns number of iterations executed (0 <= iter <= MAX_ITER).
 */
static inline int mandelbrot_point(double cr, double ci, int max_iter) {
    double zr = 0.0;
    double zi = 0.0;
    int iter = 0;

    while ((zr * zr + zi * zi) <= 4.0 && iter < max_iter) {
        double temp = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = temp;
        iter++;
    }

    return iter;
}

/* Serial reference computation for validation */
void compute_serial(int *grid) {
    for (int r = 0; r < NROWS; r++) {
        double ci = I_MIN + (I_MAX - I_MIN) * (double)r / (double)(NROWS - 1);
        for (int c = 0; c < NCOLS; c++) {
            double cr = R_MIN + (R_MAX - R_MIN) * (double)c / (double)(NCOLS - 1);
            grid[r * NCOLS + c] = mandelbrot_point(cr, ci, MAX_ITER);
        }
    }
}

/* OpenMP parallel computation using schedule(static) */
void compute_omp_static(int *grid) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < NROWS; r++) {
        double ci = I_MIN + (I_MAX - I_MIN) * (double)r / (double)(NROWS - 1);
        for (int c = 0; c < NCOLS; c++) {
            double cr = R_MIN + (R_MAX - R_MIN) * (double)c / (double)(NCOLS - 1);
            grid[r * NCOLS + c] = mandelbrot_point(cr, ci, MAX_ITER);
        }
    }
}

/* OpenMP parallel computation using schedule(dynamic) */
void compute_omp_dynamic(int *grid) {
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < NROWS; r++) {
        double ci = I_MIN + (I_MAX - I_MIN) * (double)r / (double)(NROWS - 1);
        for (int c = 0; c < NCOLS; c++) {
            double cr = R_MIN + (R_MAX - R_MIN) * (double)c / (double)(NCOLS - 1);
            grid[r * NCOLS + c] = mandelbrot_point(cr, ci, MAX_ITER);
        }
    }
}

/* OpenMP parallel computation using schedule(guided) */
void compute_omp_guided(int *grid) {
    #pragma omp parallel for schedule(guided)
    for (int r = 0; r < NROWS; r++) {
        double ci = I_MIN + (I_MAX - I_MIN) * (double)r / (double)(NROWS - 1);
        for (int c = 0; c < NCOLS; c++) {
            double cr = R_MIN + (R_MAX - R_MIN) * (double)c / (double)(NCOLS - 1);
            grid[r * NCOLS + c] = mandelbrot_point(cr, ci, MAX_ITER);
        }
    }
}

/* Validate test grid against sequential reference grid */
int validate_grid(const int *test_grid, const int *ref_grid, int total) {
    int mismatches = 0;
    for (int i = 0; i < total; i++) {
        if (test_grid[i] != ref_grid[i]) {
            mismatches++;
        }
    }
    return mismatches;
}

/* Calculate area estimate from points inside the set */
double estimate_area(const int *grid, int total) {
    int inside_count = 0;
    for (int i = 0; i < total; i++) {
        if (grid[i] == MAX_ITER) {
            inside_count++;
        }
    }
    double box_area = (R_MAX - R_MIN) * (I_MAX - I_MIN);
    /* Multiply by 2.0 because we compute the upper symmetric half */
    return 2.0 * box_area * ((double)inside_count / (double)total);
}

int main(void) {
    /* Allocate grids dynamically to prevent stack overflow */
    int *ref_grid  = (int *)malloc((size_t)TOTAL_POINTS * sizeof(int));
    int *test_grid = (int *)malloc((size_t)TOTAL_POINTS * sizeof(int));

    if (ref_grid == NULL || test_grid == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for grid size %dx%d\n", NROWS, NCOLS);
        free(ref_grid);
        free(test_grid);
        return 1;
    }

    /* Query active thread count */
    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
        }
    }

    printf("=================================================================================\n");
    printf("Exercise 07: Area of Mandelbrot Set - OpenMP Loop Scheduling Comparison\n");
    printf("Grid Dimensions:    %d x %d (%d total points)\n", NROWS, NCOLS, TOTAL_POINTS);
    printf("Coordinate Box:     Real [%.2f, %.2f], Imag [%.2f, %.3f]\n", R_MIN, R_MAX, I_MIN, I_MAX);
    printf("Maximum Iterations: %d\n", MAX_ITER);
    printf("Active Threads:     %d\n", num_threads);
    printf("=================================================================================\n");

    /* Step 1: Compute sequential reference for validation */
    compute_serial(ref_grid);
    double reference_area = estimate_area(ref_grid, TOTAL_POINTS);

    /* Prepare schedules to test */
    struct {
        const char *name;
        void (*func)(int *);
    } schedules[] = {
        {"static",  compute_omp_static},
        {"dynamic", compute_omp_dynamic},
        {"guided",  compute_omp_guided}
    };
    int num_schedules = (int)(sizeof(schedules) / sizeof(schedules[0]));

    printf("Threads | Schedule | Time (seconds) | Mismatches | Status  | Area Estimate\n");
    printf("---------------------------------------------------------------------------------\n");

    int all_passed = 1;

    for (int s = 0; s < num_schedules; s++) {
        /* Clear test grid before running */
        memset(test_grid, 0, (size_t)TOTAL_POINTS * sizeof(int));

        /* Timed parallel region: measure ONLY the computation */
        double tstart = omp_get_wtime();
        schedules[s].func(test_grid);
        double tstop = omp_get_wtime();
        double elapsed = tstop - tstart;

        /* Validation outside the timed region */
        int mismatches = validate_grid(test_grid, ref_grid, TOTAL_POINTS);
        double computed_area = estimate_area(test_grid, TOTAL_POINTS);

        const char *status = (mismatches == 0) ? "PASSED" : "FAILED";
        if (mismatches != 0) {
            all_passed = 0;
        }

        printf("%7d | %-8s | %14.6f | %10d | %-7s | %13.6f\n",
               num_threads,
               schedules[s].name,
               elapsed,
               mismatches,
               status,
               computed_area);
    }

    printf("---------------------------------------------------------------------------------\n");
    printf("Sequential Reference Area Estimate: %.6f\n", reference_area);
    printf("Overall Result: %s\n", all_passed ? "ALL SCHEDULES PASSED" : "SOME SCHEDULES FAILED");
    printf("=================================================================================\n");

    /* Free dynamically allocated memory */
    free(ref_grid);
    free(test_grid);

    return all_passed ? 0 : 1;
}
