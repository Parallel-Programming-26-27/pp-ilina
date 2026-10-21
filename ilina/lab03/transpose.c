#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Вариант В
#define DEFAULT_N 3500
#define DEFAULT_B 32

static void transpose_rows(double *A, double *B, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[j * N + i] = A[i * N + j];
}

static void transpose_cols(double *A, double *B, int N) {
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            B[j * N + i] = A[i * N + j];
}

static void transpose_blocked(double *A, double *B, int N, int b) {
    for (int i0 = 0; i0 < N; i0 += b) {
        for (int j0 = 0; j0 < N; j0 += b) {
            int i_max = (i0 + b < N) ? i0 + b : N;
            int j_max = (j0 + b < N) ? j0 + b : N;
            for (int i = i0; i < i_max; i++)
                for (int j = j0; j < j_max; j++)
                    B[j * N + i] = A[i * N + j];
        }
    }
}

static void transpose_rows_omp(double *A, double *B, int N, int threads, int sched_type, int chunk) {
    omp_set_num_threads(threads);
    if (sched_type == 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) B[j * N + i] = A[i * N + j];
    } else {
        #pragma omp parallel for schedule(static, chunk)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) B[j * N + i] = A[i * N + j];
    }
}

static void transpose_blocked_omp(double *A, double *B, int N, int b, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < N; i0 += b) {
        for (int j0 = 0; j0 < N; j0 += b) {
            int i_max = (i0 + b < N) ? i0 + b : N;
            int j_max = (j0 + b < N) ? j0 + b : N;
            for (int i = i0; i < i_max; i++)
                for (int j = j0; j < j_max; j++)
                    B[j * N + i] = A[i * N + j];
        }
    }
}

static int check_correctness(double *A, double *B, int N) {
    int errors = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (B[j * N + i] != A[i * N + j]) errors++;
    return errors;
}

static double median(double *arr, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (arr[i] > arr[j]) {
                double tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            }
    return arr[n / 2];
}

static void report(int N, int b, int threads, const char *version, double t, double bw, int err) {
    printf("N=%d b=%d threads=%d version=%s time=%.4f s BW=%.2f GB/s check=%dK\n", 
           N, b, threads, version, t, bw, err);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    int b = (argc > 2) ? atoi(argv[2]) : DEFAULT_B;
    int threads = (argc > 3) ? atoi(argv[3]) : 8;
    const char *version = (argc > 4) ? argv[4] : "rows";

    size_t sz = (size_t)N * N * sizeof(double);
    double *A = (double *)malloc(sz);
    double *B = (double *)malloc(sz);
    double *B_ref = (double *)malloc(sz);
    if (!A || !B || !B_ref) return 1;

    for (int i = 0; i < N * N; i++) A[i] = (double)(i % 100) / 10.0;

    double times[5];
    for (int k = 0; k < 5; k++) {
        double t0 = omp_get_wtime();
        if (strcmp(version, "rows") == 0) transpose_rows(A, B, N);
        else if (strcmp(version, "cols") == 0) transpose_cols(A, B, N);
        else if (strcmp(version, "blocked") == 0) transpose_blocked(A, B, N, b);
        else if (strcmp(version, "rows_omp") == 0) transpose_rows_omp(A, B, N, threads, 0, 0);
        else if (strcmp(version, "rows_omp_s1") == 0) transpose_rows_omp(A, B, N, threads, 1, 1);
        else if (strcmp(version, "rows_omp_s8") == 0) transpose_rows_omp(A, B, N, threads, 1, 8);
        else if (strcmp(version, "blocked_omp") == 0) transpose_blocked_omp(A, B, N, b, threads);
        times[k] = omp_get_wtime() - t0;
    }

    double t_med = median(times, 5);
    transpose_rows(A, B_ref, N);
    int errors = check_correctness(A, B, N);
    double bw = (2.0 * N * N * sizeof(double)) / t_med / 1e9;

    report(N, b, threads, version, t_med, bw, errors);

    free(A); free(B); free(B_ref);
    return 0;
}