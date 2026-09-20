#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define NPART 500       /* 4 * 5^3 = 500 atoms in FCC lattice */
#define NSTEPS 50       /* Simulation steps */
#define DT 0.005        /* Integration time step */
#define RHO 0.8442      /* Liquid argon density (reduced units) */
#define RCUT 2.5        /* Potential cutoff radius */

typedef struct {
    double *rx, *ry, *rz;
    double *vx, *vy, *vz;
    double *fx, *fy, *fz;
    double side;
    double pot;
    double vir;
    double ekin;
} System;

/* Allocate system arrays */
int allocate_system(System *s, int npart) {
    s->rx = (double *)malloc((size_t)npart * sizeof(double));
    s->ry = (double *)malloc((size_t)npart * sizeof(double));
    s->rz = (double *)malloc((size_t)npart * sizeof(double));
    s->vx = (double *)malloc((size_t)npart * sizeof(double));
    s->vy = (double *)malloc((size_t)npart * sizeof(double));
    s->vz = (double *)malloc((size_t)npart * sizeof(double));
    s->fx = (double *)malloc((size_t)npart * sizeof(double));
    s->fy = (double *)malloc((size_t)npart * sizeof(double));
    s->fz = (double *)malloc((size_t)npart * sizeof(double));

    if (!s->rx || !s->ry || !s->rz || !s->vx || !s->vy || !s->vz ||
        !s->fx || !s->fy || !s->fz) {
        return 0;
    }
    return 1;
}

/* Free system arrays */
void free_system(System *s) {
    free(s->rx); free(s->ry); free(s->rz);
    free(s->vx); free(s->vy); free(s->vz);
    free(s->fx); free(s->fy); free(s->fz);
}

/* Copy system state */
void copy_system(System *dst, const System *src, int npart) {
    dst->side = src->side;
    dst->pot  = src->pot;
    dst->vir  = src->vir;
    dst->ekin = src->ekin;
    for (int i = 0; i < npart; i++) {
        dst->rx[i] = src->rx[i];
        dst->ry[i] = src->ry[i];
        dst->rz[i] = src->rz[i];
        dst->vx[i] = src->vx[i];
        dst->vy[i] = src->vy[i];
        dst->vz[i] = src->vz[i];
        dst->fx[i] = src->fx[i];
        dst->fy[i] = src->fy[i];
        dst->fz[i] = src->fz[i];
    }
}

/* Initialize FCC lattice and deterministic velocities */
void init_fcc(System *s, int npart) {
    int m = 5; /* 4 * m^3 = 500 atoms */
    s->side = cbrt((double)npart / RHO);
    double a = s->side / (double)m;
    int p = 0;

    for (int ix = 0; ix < m; ix++) {
        for (int iy = 0; iy < m; iy++) {
            for (int iz = 0; iz < m; iz++) {
                double x = ix * a;
                double y = iy * a;
                double z = iz * a;

                s->rx[p] = x;         s->ry[p] = y;         s->rz[p] = z;         p++;
                s->rx[p] = x + 0.5*a; s->ry[p] = y + 0.5*a; s->rz[p] = z;         p++;
                s->rx[p] = x + 0.5*a; s->ry[p] = y;         s->rz[p] = z + 0.5*a; p++;
                s->rx[p] = x;         s->ry[p] = y + 0.5*a; s->rz[p] = z + 0.5*a; p++;
            }
        }
    }

    /* Initialize velocities deterministically */
    double sumv_x = 0.0, sumv_y = 0.0, sumv_z = 0.0;
    for (int i = 0; i < npart; i++) {
        s->vx[i] = sin((double)i * 0.1);
        s->vy[i] = cos((double)i * 0.1);
        s->vz[i] = sin((double)i * 0.2);
        sumv_x += s->vx[i];
        sumv_y += s->vy[i];
        sumv_z += s->vz[i];
    }
    /* Ensure zero net momentum */
    for (int i = 0; i < npart; i++) {
        s->vx[i] -= sumv_x / npart;
        s->vy[i] -= sumv_y / npart;
        s->vz[i] -= sumv_z / npart;
        s->fx[i] = 0.0;
        s->fy[i] = 0.0;
        s->fz[i] = 0.0;
    }
    s->pot = 0.0;
    s->vir = 0.0;
    s->ekin = 0.0;
}

/* Step 1: domove - Update positions and partial velocities */
void domove(System *s, int npart) {
    double side = s->side;
    for (int i = 0; i < npart; i++) {
        s->rx[i] += s->vx[i] * DT + 0.5 * s->fx[i] * DT * DT;
        s->ry[i] += s->vy[i] * DT + 0.5 * s->fy[i] * DT * DT;
        s->rz[i] += s->vz[i] * DT + 0.5 * s->fz[i] * DT * DT;

        /* Periodic boundary conditions */
        if (s->rx[i] < 0.0)  s->rx[i] += side;
        if (s->rx[i] >= side) s->rx[i] -= side;
        if (s->ry[i] < 0.0)  s->ry[i] += side;
        if (s->ry[i] >= side) s->ry[i] -= side;
        if (s->rz[i] < 0.0)  s->rz[i] += side;
        if (s->rz[i] >= side) s->rz[i] -= side;

        /* Half-step velocity update */
        s->vx[i] += 0.5 * s->fx[i] * DT;
        s->vy[i] += 0.5 * s->fy[i] * DT;
        s->vz[i] += 0.5 * s->fz[i] * DT;
    }
}

/* Step 2 (Serial Reference): forces_serial */
void forces_serial(System *s, int npart) {
    double side = s->side;
    double side_half = 0.5 * side;
    double rcut2 = RCUT * RCUT;
    double pot = 0.0;
    double vir = 0.0;

    for (int i = 0; i < npart; i++) {
        s->fx[i] = 0.0;
        s->fy[i] = 0.0;
        s->fz[i] = 0.0;
    }

    for (int i = 0; i < npart - 1; i++) {
        for (int j = i + 1; j < npart; j++) {
            double dx = s->rx[i] - s->rx[j];
            double dy = s->ry[i] - s->ry[j];
            double dz = s->rz[i] - s->rz[j];

            /* Minimum image convention */
            if (dx >  side_half) dx -= side;
            if (dx < -side_half) dx += side;
            if (dy >  side_half) dy -= side;
            if (dy < -side_half) dy += side;
            if (dz >  side_half) dz -= side;
            if (dz < -side_half) dz += side;

            double r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < rcut2) {
                double r2inv = 1.0 / r2;
                double r6inv = r2inv * r2inv * r2inv;
                double r12inv = r6inv * r6inv;

                /* Lennard-Jones potential: 4 * (r^-12 - r^-6) */
                double e = 4.0 * (r12inv - r6inv);
                pot += e;

                /* Virial and force magnitude */
                double f_factor = 48.0 * r2inv * (r12inv - 0.5 * r6inv);
                vir += f_factor * r2;

                double fxi = f_factor * dx;
                double fyi = f_factor * dy;
                double fzi = f_factor * dz;

                s->fx[i] += fxi; s->fy[i] += fyi; s->fz[i] += fzi;
                s->fx[j] -= fxi; s->fy[j] -= fyi; s->fz[j] -= fzi;
            }
        }
    }
    s->pot = pot;
    s->vir = vir;
}

/* Step 2 (OpenMP Parallel): forces_omp with reduction and critical */
void forces_omp(System *s, int npart) {
    double side = s->side;
    double side_half = 0.5 * side;
    double rcut2 = RCUT * RCUT;
    double pot = 0.0;
    double vir = 0.0;

    for (int i = 0; i < npart; i++) {
        s->fx[i] = 0.0;
        s->fy[i] = 0.0;
        s->fz[i] = 0.0;
    }

    #pragma omp parallel for schedule(static) \
        default(none) \
        shared(s, npart, side, side_half, rcut2) \
        reduction(+:pot, vir)
    for (int i = 0; i < npart - 1; i++) {
        for (int j = i + 1; j < npart; j++) {
            double dx = s->rx[i] - s->rx[j];
            double dy = s->ry[i] - s->ry[j];
            double dz = s->rz[i] - s->rz[j];

            /* Minimum image convention */
            if (dx >  side_half) dx -= side;
            if (dx < -side_half) dx += side;
            if (dy >  side_half) dy -= side;
            if (dy < -side_half) dy += side;
            if (dz >  side_half) dz -= side;
            if (dz < -side_half) dz += side;

            double r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < rcut2) {
                double r2inv = 1.0 / r2;
                double r6inv = r2inv * r2inv * r2inv;
                double r12inv = r6inv * r6inv;

                /* 2 Reduction variables: pot and vir */
                double e = 4.0 * (r12inv - r6inv);
                pot += e;

                double f_factor = 48.0 * r2inv * (r12inv - 0.5 * r6inv);
                vir += f_factor * r2;

                double fxi = f_factor * dx;
                double fyi = f_factor * dy;
                double fzi = f_factor * dz;

                /*
                 * Critical section protecting force array updates:
                 * Thread A (at index i_A) and Thread B (at index i_B) may
                 * simultaneously interact with particle j. Without this critical
                 * directive, concurrent writes to fx[j], fy[j], fz[j] cause a race.
                 */
                #pragma omp critical
                {
                    s->fx[i] += fxi; s->fy[i] += fyi; s->fz[i] += fzi;
                    s->fx[j] -= fxi; s->fy[j] -= fyi; s->fz[j] -= fzi;
                }
            }
        }
    }
    s->pot = pot;
    s->vir = vir;
}

/* Step 3: mkekin - Finalize velocity update and calculate kinetic energy */
void mkekin(System *s, int npart) {
    double ekin = 0.0;
    for (int i = 0; i < npart; i++) {
        s->vx[i] += 0.5 * s->fx[i] * DT;
        s->vy[i] += 0.5 * s->fy[i] * DT;
        s->vz[i] += 0.5 * s->fz[i] * DT;
        ekin += 0.5 * (s->vx[i]*s->vx[i] + s->vy[i]*s->vy[i] + s->vz[i]*s->vz[i]);
    }
    s->ekin = ekin;
}

/* Step 4: velavg - Compute average temperature */
double velavg(const System *s, int npart) {
    return (2.0 * s->ekin) / (3.0 * (double)npart);
}

int main(void) {
    System s_ser, s_par;

    if (!allocate_system(&s_ser, NPART) || !allocate_system(&s_par, NPART)) {
        fprintf(stderr, "Error: Memory allocation failed for system size %d\n", NPART);
        return 1;
    }

    init_fcc(&s_ser, NPART);
    copy_system(&s_par, &s_ser, NPART);

    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
        }
    }

    printf("=================================================================================\n");
    printf("Exercise 08: Molecular Dynamics (MolDyn) Parallelization with OpenMP\n");
    printf("Particles:           %d (Argon FCC Lattice)\n", NPART);
    printf("Simulation Steps:    %d\n", NSTEPS);
    printf("Time Step (dt):      %.4f\n", DT);
    printf("Potential Cutoff:    %.2f (Lennard-Jones 12-6)\n", RCUT);
    printf("OpenMP Active Threads: %d\n", num_threads);
    printf("=================================================================================\n");

    /* Sequential Reference Run */
    printf("\nRunning Sequential Reference Simulation (%d steps)...\n", NSTEPS);
    double tstart_ser = omp_get_wtime();
    for (int step = 1; step <= NSTEPS; step++) {
        domove(&s_ser, NPART);
        forces_serial(&s_ser, NPART);
        mkekin(&s_ser, NPART);
    }
    double tstop_ser = omp_get_wtime();
    double time_ser = tstop_ser - tstart_ser;

    double temp_ser = velavg(&s_ser, NPART);
    double etot_ser = s_ser.pot + s_ser.ekin;

    printf("Sequential Run Completed in: %.6f seconds\n", time_ser);
    printf("  Potential Energy (pot): %15.8f\n", s_ser.pot);
    printf("  Virial           (vir): %15.8f\n", s_ser.vir);
    printf("  Kinetic Energy  (ekin): %15.8f\n", s_ser.ekin);
    printf("  Total Energy    (etot): %15.8f\n", etot_ser);
    printf("  Temperature     (temp): %15.8f\n", temp_ser);

    /* OpenMP Parallel Run */
    printf("\nRunning OpenMP Parallel Simulation with %d threads (%d steps)...\n", num_threads, NSTEPS);
    double tstart_par = omp_get_wtime();
    for (int step = 1; step <= NSTEPS; step++) {
        domove(&s_par, NPART);
        forces_omp(&s_par, NPART);
        mkekin(&s_par, NPART);
    }
    double tstop_par = omp_get_wtime();
    double time_par = tstop_par - tstart_par;

    double temp_par = velavg(&s_par, NPART);
    double etot_par = s_par.pot + s_par.ekin;

    printf("Parallel Run Completed in:   %.6f seconds\n", time_par);
    printf("  Potential Energy (pot): %15.8f\n", s_par.pot);
    printf("  Virial           (vir): %15.8f\n", s_par.vir);
    printf("  Kinetic Energy  (ekin): %15.8f\n", s_par.ekin);
    printf("  Total Energy    (etot): %15.8f\n", etot_par);
    printf("  Temperature     (temp): %15.8f\n", temp_par);

    /* Numerical Validation */
    double diff_pot  = fabs(s_ser.pot - s_par.pot);
    double diff_vir  = fabs(s_ser.vir - s_par.vir);
    double diff_ekin = fabs(s_ser.ekin - s_par.ekin);
    double diff_etot = fabs(etot_ser - etot_par);

    const double tol = 1e-7;
    int pass = (diff_pot < tol && diff_vir < tol && diff_ekin < tol && diff_etot < tol);

    printf("\n=================================================================================\n");
    printf("Numerical Validation (Parallel vs. Sequential):\n");
    printf("  |Diff Potential Energy|: %e (Tolerance: %e)\n", diff_pot, tol);
    printf("  |Diff Virial|:           %e (Tolerance: %e)\n", diff_vir, tol);
    printf("  |Diff Kinetic Energy|:   %e (Tolerance: %e)\n", diff_ekin, tol);
    printf("  |Diff Total Energy|:     %e (Tolerance: %e)\n", diff_etot, tol);
    printf("  Validation Result:       %s\n", pass ? "PASSED (Numerically Consistent)" : "FAILED");
    printf("=================================================================================\n");

    free_system(&s_ser);
    free_system(&s_par);

    return pass ? 0 : 1;
}
