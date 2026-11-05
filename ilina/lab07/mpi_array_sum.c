#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define REPS 10000

int main(int argc, char **argv) {
    int rank, size;
    int N = 0;
    long long expected_sum = 0;
    long long local_sum = 0;
    long long gather_sum = 0;
    long long reduce_sum = 0;
    long long allreduce_sum = 0;
    
    long long *array = NULL;
    long long *local_array = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;
    long long *gathered = NULL;
    
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) N = atoi(argv[1]);
    else if (rank == 0) {
        printf("Использование: %s <N> [agg]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base_chunk = N / size;
    int remainder = N % size;
    
    sendcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base_chunk + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }
    
    int local_n = sendcounts[rank];
    local_array = (long long *)malloc(local_n * sizeof(long long));

    if (rank == 0) {
        array = (long long *)malloc(N * sizeof(long long));
        for (int i = 0; i < N; i++) array[i] = i + 1;
        expected_sum = (long long)N * (N + 1) / 2;
        gathered = (long long *)malloc(size * sizeof(long long));
    }

    MPI_Scatterv(array, sendcounts, displs, MPI_LONG_LONG_INT,
                 local_array, local_n, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; i++) {
        local_sum += local_array[i];
    }

    if (argc > 2 && strcmp(argv[2], "agg") == 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        start = MPI_Wtime();
        for (int r = 0; r < REPS; r++) {
            MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, gathered, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                gather_sum = 0;
                for (int i = 0; i < size; i++) gather_sum += gathered[i];
            }
        }
        end = MPI_Wtime();
        if (rank == 0) {
            printf("P=%d method=Gather+sum repeats=%d time=%.9f s correct=%s\n",
                   size, REPS, (end - start) / REPS, gather_sum == expected_sum ? "да" : "нет");
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Тест Reduce
        start = MPI_Wtime();
        for (int r = 0; r < REPS; r++) {
            MPI_Reduce(&local_sum, &reduce_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        end = MPI_Wtime();
        if (rank == 0) {
            printf("P=%d method=Reduce repeats=%d time=%.9f s correct=%s\n",
                   size, REPS, (end - start) / REPS, reduce_sum == expected_sum ? "да" : "нет");
        }
    } 
    else {
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();

        MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, gathered, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (int i = 0; i < size; i++) gather_sum += gathered[i];
        }

        MPI_Reduce(&local_sum, &reduce_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sum, &allreduce_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();

        if (rank == 0) {
            printf("N=%d, P=%d\n", N, size);
            printf("Ожидаемая сумма: %lld\n", expected_sum);
            printf("Gather:          %lld\n", gather_sum);
            printf("Reduce:          %lld\n", reduce_sum);
            printf("Allreduce:       %lld\n", allreduce_sum);
            printf("Время: %.6f с\n", end - start);
            
            if (gather_sum == expected_sum && reduce_sum == expected_sum && allreduce_sum == expected_sum)
                printf("Проверка: Корректно!\n");
            else
                printf("Проверка: ОШИБКА!\n");
        }
    }

    free(local_array);
    free(sendcounts);
    free(displs);
    if (rank == 0) {
        free(array);
        free(gathered);
    }
    
    MPI_Finalize();
    return 0;
}