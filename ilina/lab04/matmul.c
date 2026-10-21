#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static void calc_ijk_serial(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < N; k++)
                acc += A[i * N + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

static void calc_ijk_parallel(const double *A, const double *B, double *C, int N, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < N; k++)
                acc += A[i * N + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

static void calc_ikj_serial(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a_val = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a_val * B[k * N + j];
        }
    }
}

static void calc_ikj_parallel(const double *A, const double *B, double *C, int N, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a_val = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a_val * B[k * N + j];
        }
    }
}

static void calc_blocked_parallel(const double *A, const double *B, double *C, int N, int b, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += b) {
        for (int jj = 0; jj < N; jj += b) {
            for (int kk = 0; kk < N; kk += b) {
                for (int i = ii; i < ii + b && i < N; i++) {
                    for (int k = kk; k < kk + b && k < N; k++) {
                        double a_val = A[i * N + k];
                        for (int j = jj; j < jj + b && j < N; j++)
                            C[i * N + j] += a_val * B[k * N + j];
                    }
                }
            }
        }
    }
}

static void check_result(const double *C, const double *Ref, int N) {
    double max_err = 0.0;
    for (int i = 0; i < N * N; i++) {
        double diff = fabs(C[i] - Ref[i]);
        if (diff > max_err) max_err = diff;
    }
    printf("  maxdiff = %.2e (%s)\n", max_err, max_err < 1e-9 ? "OK" : "FAIL");
}

static double get_median(double *arr, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (arr[i] > arr[j]) { double t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
    return arr[n / 2];
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s N b threads version\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int b = atoi(argv[2]);
    int threads = atoi(argv[3]);
    const char *version = argv[4];
    size_t sz = (size_t)N * N * sizeof(double);

    double *A = malloc(sz), *B = malloc(sz), *C = malloc(sz), *Ref = malloc(sz);
    if (!A || !B || !C || !Ref) return 1;

    for (int i = 0; i < N * N; i++) {
        A[i] = (rand() % 100) * 0.01;
        B[i] = (rand() % 100) * 0.01;
    }

    memset(Ref, 0, sz);
    calc_ijk_serial(A, B, Ref, N);

    double times[3];
    for (int r = 0; r < 3; r++) {
        memset(C, 0, sz);
        double t0 = omp_get_wtime();
        if (strcmp(version, "ijk_seq") == 0) calc_ijk_serial(A, B, C, N);
        else if (strcmp(version, "ijk_omp") == 0) calc_ijk_parallel(A, B, C, N, threads);
        else if (strcmp(version, "ikj_seq") == 0) calc_ikj_serial(A, B, C, N);
        else if (strcmp(version, "ikj_omp") == 0) calc_ikj_parallel(A, B, C, N, threads);
        else if (strcmp(version, "blocked_omp") == 0) calc_blocked_parallel(A, B, C, N, b, threads);
        times[r] = omp_get_wtime() - t0;
    }

    double t_med = get_median(times, 3);
    int is_seq = (strcmp(version, "ijk_seq") == 0 || strcmp(version, "ikj_seq") == 0);
    
    printf("N=%d b=%d threads=%d version=%s time=%.4f s\n", N, b, is_seq ? 1 : threads, version, t_med);
    check_result(C, Ref, N);

    free(A); free(B); free(C); free(Ref);
    return 0;
}