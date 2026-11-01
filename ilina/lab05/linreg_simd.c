#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef struct {
    int N, d, K;
    double lambda, eta;
} Params;

static unsigned long long rng_state = 12345ULL;

double rnd(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rng_state >> 11) / 9007199254740992.0;
}

static inline double soft_thresh(double z, double alpha) {
    double s = fabs(z) - alpha;
    if (s < 0.0) s = 0.0;
    return copysign(s, z);
}

void apply_update(double *w, double *b, const double *grad, double gb, const Params *p) {
    double inv_N = 1.0 / p->N;
    for (int j = 0; j < p->d; j++) {
        w[j] = soft_thresh(w[j] - p->eta * grad[j] * inv_N, p->eta * p->lambda);
    }
    *b -= p->eta * gb * inv_N;
}

void apply_update_simd(double *w, double *b, const double *grad, double gb, const Params *p) {
    double inv_N = 1.0 / p->N;
    #pragma omp simd
    for (int j = 0; j < p->d; j++) {
        w[j] = soft_thresh(w[j] - p->eta * grad[j] * inv_N, p->eta * p->lambda);
    }
    *b -= p->eta * gb * inv_N;
}

void train_serial(const double *X, const double *y, double *w, double *b, double *grad, const Params *p) {
    for (int iter = 0; iter < p->K; iter++) {
        memset(grad, 0, p->d * sizeof(double));
        double gb = 0.0;
        for (int i = 0; i < p->N; i++) {
            const double *xi = X + (size_t)i * p->d;
            double pred = *b;
            for (int j = 0; j < p->d; j++) pred += xi[j] * w[j];
            double err = pred - y[i];
            for (int j = 0; j < p->d; j++) grad[j] += err * xi[j];
            gb += err;
        }
        apply_update(w, b, grad, gb, p);
    }
}

void train_simd(const double *X, const double *y, double *w, double *b, double *grad, const Params *p) {
    for (int iter = 0; iter < p->K; iter++) {
        memset(grad, 0, p->d * sizeof(double));
        double gb = 0.0;
        for (int i = 0; i < p->N; i++) {
            const double *xi = X + (size_t)i * p->d;
            double pred = *b;
            #pragma omp simd reduction(+:pred)
            for (int j = 0; j < p->d; j++) pred += xi[j] * w[j];
            
            double err = pred - y[i];
            
            #pragma omp simd
            for (int j = 0; j < p->d; j++) grad[j] += err * xi[j];
            gb += err;
        }
        apply_update_simd(w, b, grad, gb, p);
    }
}

void train_omp(const double *X, const double *y, double *w, double *b, double *grad, double *part, double *partb, int stride, const Params *p) {
    int T = omp_get_max_threads();
    for (int iter = 0; iter < p->K; iter++) {
        memset(part, 0, (size_t)T * stride * sizeof(double));
        memset(partb, 0, (size_t)T * 8 * sizeof(double));
        double bb = *b;

        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            double *local_grad = part + (size_t)t * stride;
            double local_gb = 0.0;

            #pragma omp for schedule(static)
            for (int i = 0; i < p->N; i++) {
                const double *xi = X + (size_t)i * p->d;
                double pred = bb;
                for (int j = 0; j < p->d; j++) pred += xi[j] * w[j];
                double err = pred - y[i];
                for (int j = 0; j < p->d; j++) local_grad[j] += err * xi[j];
                local_gb += err;
            }
            partb[t * 8] = local_gb;
        }
        
        memset(grad, 0, p->d * sizeof(double));
        double gb = 0.0;
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < p->d; j++) grad[j] += part[(size_t)t * stride + j];
            gb += partb[t * 8];
        }
        apply_update(w, b, grad, gb, p);
    }
}

void train_omp_simd(const double *X, const double *y, double *w, double *b, double *grad, double *part, double *partb, int stride, const Params *p) {
    int T = omp_get_max_threads();
    for (int iter = 0; iter < p->K; iter++) {
        memset(part, 0, (size_t)T * stride * sizeof(double));
        memset(partb, 0, (size_t)T * 8 * sizeof(double));
        double bb = *b;

        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            double *local_grad = part + (size_t)t * stride;
            double local_gb = 0.0;

            #pragma omp for schedule(static)
            for (int i = 0; i < p->N; i++) {
                const double *xi = X + (size_t)i * p->d;
                double pred = bb;
                #pragma omp simd reduction(+:pred)
                for (int j = 0; j < p->d; j++) pred += xi[j] * w[j];
                
                double err = pred - y[i];
                
                #pragma omp simd
                for (int j = 0; j < p->d; j++) local_grad[j] += err * xi[j];
                local_gb += err;
            }
            partb[t * 8] = local_gb;
        }
        
        memset(grad, 0, p->d * sizeof(double));
        double gb = 0.0;
        for (int t = 0; t < T; t++) {
            #pragma omp simd
            for (int j = 0; j < p->d; j++) grad[j] += part[(size_t)t * stride + j];
            gb += partb[t * 8];
        }
        apply_update_simd(w, b, grad, gb, p);
    }
}

double calc_loss(const double *X, const double *y, const double *w, double b, const Params *p) {
    double s = 0.0;
    for (int i = 0; i < p->N; i++) {
        const double *xi = X + (size_t)i * p->d;
        double pred = b;
        for (int j = 0; j < p->d; j++) pred += xi[j] * w[j];
        double err = pred - y[i];
        s += err * err;
    }
    double l1 = 0.0;
    for (int j = 0; j < p->d; j++) l1 += fabs(w[j]);
    return s / (2.0 * p->N) + p->lambda * l1;
}

int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 8) {
        printf("Usage: %s N d K lambda eta threads version\n", argv[0]);
        return 1;
    }
    
    Params p;
    p.N = atoi(argv[1]);
    p.d = atoi(argv[2]);
    p.K = atoi(argv[3]);
    p.lambda = atof(argv[4]);
    p.eta = atof(argv[5]);
    int threads = atoi(argv[6]);
    const char *version = argv[7];

    omp_set_num_threads(threads);

    int stride = (p.d + 7) / 8 * 8;
    
    double *X = malloc((size_t)p.N * p.d * sizeof(double));
    double *y = malloc((size_t)p.N * sizeof(double));
    double *w_true = malloc(p.d * sizeof(double));
    double *w = malloc(p.d * sizeof(double));
    double *w_ref = malloc(p.d * sizeof(double));
    double *grad = malloc(p.d * sizeof(double));
    
    double *part = malloc((size_t)threads * stride * sizeof(double));
    double *partb = malloc((size_t)threads * 8 * sizeof(double));

    for (int j = 0; j < p.d; j++) {
        double v = 0.1 + 0.9 * rnd();
        w_true[j] = (j % 4 == 0) ? 0.0 : (rnd() < 0.5 ? -v : v);
    }
    
    for (int i = 0; i < p.N; i++) {
        double sum = 0.5;
        for (int j = 0; j < p.d; j++) {
            double x_val = 2.0 * rnd() - 1.0;
            X[(size_t)i * p.d + j] = x_val;
            sum += x_val * w_true[j];
        }
        y[i] = sum + 0.1 * (2.0 * rnd() - 1.0);
    }

    double b_ref = 0.0, b = 0.0;
    memset(w_ref, 0, p.d * sizeof(double));
    train_serial(X, y, w_ref, &b_ref, grad, &p);
    double loss_ref = calc_loss(X, y, w_ref, b_ref, &p);

    double times[3];
    for (int r = 0; r < 3; r++) {
        memset(w, 0, p.d * sizeof(double));
        b = 0.0;
        double t0 = omp_get_wtime();
        
        if (strcmp(version, "seq") == 0) train_serial(X, y, w, &b, grad, &p);
        else if (strcmp(version, "simd") == 0) train_simd(X, y, w, &b, grad, &p);
        else if (strcmp(version, "omp") == 0) train_omp(X, y, w, &b, grad, part, partb, stride, &p);
        else if (strcmp(version, "omp_simd") == 0) train_omp_simd(X, y, w, &b, grad, part, partb, stride, &p);
        
        times[r] = omp_get_wtime() - t0;
    }
    
    qsort(times, 3, sizeof(double), cmp_d);
    double t_med = times[1];

    double final_loss = calc_loss(X, y, w, b, &p);
    double dw = 0.0;
    int zeros = 0;
    for (int j = 0; j < p.d; j++) {
        double diff = fabs(w[j] - w_ref[j]);
        if (diff > dw) dw = diff;
        if (fabs(w[j]) < 1e-6) zeros++;
    }
    
    int ok = (dw < 1e-8) && (fabs(final_loss - loss_ref) < 1e-8);
    int used_threads = (strstr(version, "omp") != NULL) ? threads : 1;

    printf("N=%d d=%d K=%d lambda=%g eta=%g threads=%d version=%s time=%.6f s loss=%.6f dw=%.2e zeros=%d check=%s\n",
           p.N, p.d, p.K, p.lambda, p.eta, used_threads, version, t_med, final_loss, dw, zeros, ok ? "OK" : "FAIL");

    free(X); free(y); free(w_true); free(w); free(w_ref); free(grad); free(part); free(partb);
    return 0;
}