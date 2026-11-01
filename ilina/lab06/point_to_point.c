#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_MESSAGE 100
#define TAG_REPLY   200
#define TAG_TEST    300

void basic_exchange(int rank, int size) {
    if (size < 2) return;
    char msg[] = "Hello from Rank 0!";
    char reply[] = "Message received";
    char buf[256];
    MPI_Status status;

    if (rank == 0) {
        printf("[Rank 0] Sending message to Rank 1...\n");
        MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
        
        MPI_Recv(buf, 256, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, &status);
        printf("[Rank 0] Received reply: '%s'\n", buf);
    } else if (rank == 1) {
        MPI_Recv(buf, 256, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
        printf("[Rank 1] Received message: '%s'\n", buf);
        
        printf("[Rank 1] Sending reply to Rank 0...\n");
        MPI_Send(reply, strlen(reply) + 1, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    }
}

void multi_exchange(int rank, int size) {
    char msg[] = "Hello from Master";
    char reply[] = "ACK";
    char buf[256];
    MPI_Status status;

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, i, TAG_MESSAGE, MPI_COMM_WORLD);
        }
        for (int i = 1; i < size; i++) {
            MPI_Recv(buf, 256, MPI_CHAR, i, TAG_REPLY, MPI_COMM_WORLD, &status);
            printf("[Rank 0] Got reply from Rank %d: '%s'\n", status.MPI_SOURCE, buf);
        }
    } else {
        MPI_Recv(buf, 256, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
        printf("[Rank %d] Received: '%s'\n", rank, buf);
        MPI_Send(reply, strlen(reply) + 1, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    }
}

void benchmark(int rank, int size) {
    if (size < 2) return;
    int sizes[] = {1, 1024, 65536, 1048576}; // 1 B, 1 KB, 64 KB, 1 MB
    int num_sizes = 4;
    int reps = 100;
    
    if (rank == 0) printf("\n--- Ping-Pong Benchmark (Rank 0 <-> Rank 1) ---\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int sz = sizes[s];
        char *send_buf = malloc(sz);
        char *recv_buf = malloc(sz);
        memset(send_buf, 'A', sz);
        
        // Прогрев
        if (rank == 0) {
            MPI_Send(send_buf, sz, MPI_CHAR, 1, TAG_TEST, MPI_COMM_WORLD);
            MPI_Recv(recv_buf, sz, MPI_CHAR, 1, TAG_TEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(recv_buf, sz, MPI_CHAR, 0, TAG_TEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(recv_buf, sz, MPI_CHAR, 0, TAG_TEST, MPI_COMM_WORLD);
        }

        double start = MPI_Wtime();
        
        if (rank == 0) {
            for (int r = 0; r < reps; r++) {
                MPI_Send(send_buf, sz, MPI_CHAR, 1, TAG_TEST, MPI_COMM_WORLD);
                MPI_Recv(recv_buf, sz, MPI_CHAR, 1, TAG_TEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            double end = MPI_Wtime();
            printf("Size: %8d bytes | Avg time: %.6f s\n", sz, (end - start) / (2.0 * reps));
        } else if (rank == 1) {
            for (int r = 0; r < reps; r++) {
                MPI_Recv(recv_buf, sz, MPI_CHAR, 0, TAG_TEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(recv_buf, sz, MPI_CHAR, 0, TAG_TEST, MPI_COMM_WORLD);
            }
        }
        
        free(send_buf);
        free(recv_buf);
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Error: This program requires at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank < 2) {
        if (rank == 0) printf("=== Basic Exchange (0 <-> 1) ===\n");
        basic_exchange(rank, size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("\n=== Multi-Process Exchange (0 <-> All) ===\n");
    multi_exchange(rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    
    benchmark(rank, size);

    MPI_Finalize();
    return 0;
}