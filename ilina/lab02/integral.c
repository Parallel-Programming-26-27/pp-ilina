/* ==== ВАРИАНТ B: f(x) = cos(x^2), отрезок [0; 1] ==== */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>



static double func(double x)
{
    return cos(x * x);
}

#define A_DEF  0.0
#define B_DEF  1.0

static double midpoint_serial(double a, double b, long long n)
{
    double h   = (b - a) / (double)n;
    double sum = 0.0;
    long long   i;
    for (i = 0; i < n; i++)
        sum += func(a + ((double)i + 0.5) * h);
    return h * sum;
}

static double midpoint_omp(double a, double b, long long n)
{
    double h   = (b - a) / (double)n;
    double sum = 0.0;
    long long   i;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (i = 0; i < n; i++)
        sum += func(a + ((double)i + 0.5) * h);
    return h * sum;
}

static double integrate_to_eps(double a, double b, double eps,
                               int use_omp, long long *n_out)
{
    long long   n    = 1024;
    double prev = use_omp ? midpoint_omp(a, b, n)
                          : midpoint_serial(a, b, n);
    double cur  = prev;

    while (1) {
        n *= 2;
        cur = use_omp ? midpoint_omp(a, b, n)
                      : midpoint_serial(a, b, n);
        if (fabs(cur - prev) < eps) break;
        prev = cur;
        if (n > (1LL << 40)) break;
    }
    *n_out = n;
    return cur;
}

static void report(const char *label, double I, double t, int nt)
{
    printf("%-14s | I = %.12f | время = %.4f с | потоков = %d\n",
           label, I, t, nt);
}

int main(int argc, char **argv)
{
    double a   = (argc > 1) ? atof(argv[1])  : A_DEF;
    double b   = (argc > 2) ? atof(argv[2])  : B_DEF;
    long long   n   = (argc > 3) ? atoll(argv[3]) : 100000000LL;
    int    nt  = (argc > 4) ? atoi(argv[4])  : omp_get_max_threads();
    double eps = (argc > 5) ? atof(argv[5])  : 0.0;

    if (n <= 0 || nt <= 0) {
        fprintf(stderr, "Использование: %s [a] [b] [n] [потоки] [eps]\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(nt);

    printf("Метод средних прямоугольников\n");
    printf("a = %.4f, b = %.4f, n = %lld, потоков = %d\n", a, b, n, nt);
    printf("-------------------------------------------------------------\n");

    double t0    = omp_get_wtime();
    double i_seq = midpoint_serial(a, b, n);
    double t_seq = omp_get_wtime() - t0;
    report("последоват.", i_seq, t_seq, 1);

    t0           = omp_get_wtime();
    double i_par = midpoint_omp(a, b, n);
    double t_par = omp_get_wtime() - t0;
    report("OpenMP", i_par, t_par, nt);

    printf("-------------------------------------------------------------\n");
    printf("Ускорение S = T1 / Tp = %.3f\n\n", t_seq / t_par);

    if (eps > 0.0) {
        printf("Вычисление с точностью eps = %.1e\n", eps);
        long long n_seq = 0, n_par = 0;

        t0 = omp_get_wtime();
        double I_seq_eps = integrate_to_eps(a, b, eps, 0, &n_seq);
        double t_seq_eps = omp_get_wtime() - t0;

        t0 = omp_get_wtime();
        double I_par_eps = integrate_to_eps(a, b, eps, 1, &n_par);
        double t_par_eps = omp_get_wtime() - t0;

        printf("последоват. | n = %lld | I = %.12f | время = %.4f с\n",
               n_seq, I_seq_eps, t_seq_eps);
        printf("OpenMP      | n = %lld | I = %.12f | время = %.4f с\n",
               n_par, I_par_eps, t_par_eps);
    }

    return 0;
}