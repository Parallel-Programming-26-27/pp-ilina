#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static const double PI_REF = 3.14159265358979323846;
static double leibniz_serial(long long n)
{
    double sum = 0.0;
    long long i;
    for (i = 0; i < n; i++) {
        double sign = (i & 1) ? -1.0 : 1.0;   /* (-1)^i */
        sum += sign / (2.0 * i + 1.0);
    }
    return 4.0 * sum;
}

static double leibniz_omp(long long n)
{
    double sum = 0.0;
    long long i;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (i = 0; i < n; i++) {
        double sign = (i & 1) ? -1.0 : 1.0;
        sum += sign / (2.0 * i + 1.0);
    }
    return 4.0 * sum;
}

static void report(const char *label, double pi, double t, int nt)
{
    printf("%-16s | pi=%.12f | ошибка=%.3e | время=%.4f с | потоков=%d\n",
           label, pi, fabs(pi - PI_REF), t, nt);
}

int main(int argc, char **argv)
{
    long long n       = (argc > 1) ? atoll(argv[1]) : 1000000000LL;
    int       threads = (argc > 2) ? atoi(argv[2])  : omp_get_max_threads();

    if (n <= 0 || threads <= 0) {
        fprintf(stderr, "Использование: %s [n] [потоки]\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(threads);

    printf("Ряд Лейбница, n = %lld, потоков = %d\n", n, threads);
    printf("-------------------------------------------------------------\n");

    double start = omp_get_wtime();
    double pi_seq = leibniz_serial(n);
    double time_seq = omp_get_wtime() - start;
    report("последовательная", pi_seq, time_seq, 1);

    start = omp_get_wtime();
    double pi_par = leibniz_omp(n);
    double time_par = omp_get_wtime() - start;
    report("OpenMP", pi_par, time_par, threads);

    printf("-------------------------------------------------------------\n");
    printf("Ускорение S = T1 / Tp = %.3f\n", time_seq / time_par);
    return 0;
}