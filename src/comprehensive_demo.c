// Matrix multiplication using Bulk synchronous parallel (BSP) strategy.
// Divide computation into supersteps, collective communication.

// A x B = result. A(N * K), B(K * M), C(N * M); Number of process = P
// Collective communication times ~ log(P) due to tree-based communication  algorithm.
// T_tot = alpha * log(P) + beta * (M*K + K*M) / P
// alpha is the communication overhead, beta is the overhead of computation & data parallelization.

// Main advantage: Predictable performance, easier to debug and maintain Almost the most robust, unless the memory is limited or the computation time is unpredictable/prolonged.

// Potential bottleneck: have to wait for the lowest process to complete computing. May be slowed down when the time cost of each task is unpredicatble and much longer than that of MPI index communication.

#include "cblas.h"
#include "info_op.h"
#include "matrix_op.h"

#include "mpi.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    printf("]> Demostration program: comprehensive example of matrix multiply.\n");
    printf("]> Algorithm options: \n\tMPI (including option of |Rough Simple sequential|CBLAS calculation| for worker indexes) \n\tRough Simple sequential \n\t CBLAS sequential.\n");
    printf("]> Other options: \n\tMPI communication |blocking|non-blocking| \n\tMPI index debug info |on|off|");
    printf("]> Edit CMakeLists.txt compilation macro definitions for tuning all the options.");    
    printf("=======================================================");
    int n_matrix_A_row, n_matrix_A_col, n_matrix_B_row, n_matrix_B_col;
    if (argc == 1) {
        n_matrix_A_row = 1000;
        n_matrix_A_col = 500;
        n_matrix_B_row = n_matrix_A_col;
        n_matrix_B_col = 2000;
    } else if (argc == 5) {
        n_matrix_A_row = atoi(argv[1]);
        n_matrix_A_col = atoi(argv[2]);
        n_matrix_B_row = atoi(argv[3]);
        n_matrix_B_col = atoi(argv[4]);
        if (n_matrix_A_col != n_matrix_B_row) {
            fprintf(stderr,
                    "Error: n_matrix_A_col should be equal to n_matrix_B_row\n");
            return 1;
        }
    } else {
        fprintf(stderr,
                "Usage: %s [n_matrix_A_row n_matrix_A_col n_matrix_B_row "
                "n_matrix_B_col]\n",
                argv[0]);
        return 1;
    }

    // default np = n cores of you hardware, use -np N to specify the number of processes
    int num_processor;
    int myrank;
    double swtime, ewtime, swtime_cblas, ewtime_cblas, swtime_seq, ewtime_seq;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processor);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0) {
        double** matrix_A = generate_matrix(n_matrix_A_row, n_matrix_A_col);
        double** matrix_B = generate_matrix(n_matrix_B_row, n_matrix_B_col);
        double** matrix_result = make_matrix(n_matrix_A_row, n_matrix_B_col);
        double** matrix_result_cblas = make_matrix(n_matrix_A_row, n_matrix_B_col);

        double* L_A = linearlize_matrix(matrix_A, n_matrix_A_row, n_matrix_A_col);
        int n_avg_task_row = n_matrix_A_row / (num_processor - 1);
        int n_remainder_task_row = n_matrix_A_row % (num_processor - 1);

        swtime = MPI_Wtime();

        MPI_Bcast(linearlize_matrix(matrix_B, n_matrix_B_row, n_matrix_B_col),
                  n_matrix_B_row * n_matrix_B_col, MPI_DOUBLE, myrank,
                  MPI_COMM_WORLD);

        int row_offset = 0;
        MPI_Request* send_requests = (MPI_Request*)malloc((num_processor - 1) * sizeof(MPI_Request));
        for (int worker_rank = 1; worker_rank < num_processor; worker_rank++) {
            int n_row_send = worker_rank <= n_remainder_task_row ? n_avg_task_row + 1 : n_avg_task_row;
#if NON_BLOCKING_COMMUNICATION
            // non-blocking send
            MPI_Isend(L_A + row_offset * n_matrix_A_col, n_row_send * n_matrix_A_col,
                      MPI_DOUBLE, worker_rank, worker_rank, MPI_COMM_WORLD,
                      &send_requests[worker_rank - 1]);
            debug_proc(myrank, "Allocating rows to process %d\n", worker_rank);
#else
            MPI_Send(L_A + row_offset * n_matrix_A_col, n_row_send * n_matrix_A_col,
                     MPI_DOUBLE, worker_rank, worker_rank, MPI_COMM_WORLD);
            debug_proc(myrank, "Allocated rows to process %d\n", worker_rank);
#endif
            row_offset += n_row_send;
        }
#if NON_BLOCKING_COMMUNICATION
        MPI_Waitall(num_processor - 1, send_requests, MPI_STATUSES_IGNORE);
#endif

        row_offset = 0;
        MPI_Request* recv_requests = (MPI_Request*)malloc((num_processor - 1) * sizeof(MPI_Request));
        for (int worker_rank = 1; worker_rank < num_processor; worker_rank++) {
            int n_row_recv = worker_rank <= n_remainder_task_row ? n_avg_task_row + 1 : n_avg_task_row;
#if NON_BLOCKING_COMMUNICATION
            // non-blocking receive, the faster worker result get earlier
            MPI_Irecv(matrix_result[row_offset], n_row_recv * n_matrix_B_col,
                      MPI_DOUBLE, worker_rank, worker_rank, MPI_COMM_WORLD,
                      &recv_requests[worker_rank - 1]);
            debug_proc(myrank, "Receiving result from process %d\n", worker_rank);
#else
            MPI_Recv(matrix_result[row_offset], n_row_recv * n_matrix_B_col,
                     MPI_DOUBLE, worker_rank, worker_rank, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            debug_proc(myrank, "Received result from process %d\n", worker_rank);
#endif
            row_offset += n_row_recv;
        }
#if NON_BLOCKING_COMMUNICATION
        MPI_Waitall(num_processor - 1, recv_requests, MPI_STATUSES_IGNORE);
#endif
        // conclude
        ewtime = MPI_Wtime();
        printf("MPI Result matrix showcase:\n");
        print_matrix_less(matrix_result, n_matrix_A_row, n_matrix_B_col, 5, 5);
        printf("[Time taken] MPI: %f seconds\n", ewtime - swtime);
#if ADD_RSSEQ_COMPARE
        swtime_seq = MPI_Wtime();
        double* L_result_seq = matrix_multiply_matrix_linear(
            L_A, linearlize_matrix(matrix_B, n_matrix_B_row, n_matrix_B_col),
            n_matrix_A_row, n_matrix_A_col, n_matrix_B_col);
        ewtime_seq = MPI_Wtime();
        printf("[Time taken] Rough simple sequential: %f seconds\n",
               ewtime_seq - swtime_seq);
#else
        printf("\n");
#endif
        // cblas test and contrast
        swtime_cblas = MPI_Wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_matrix_A_row,
                    n_matrix_B_col, n_matrix_A_col, 1.0, matrix_A[0],
                    n_matrix_A_col, matrix_B[0], n_matrix_B_col, 0.0,
                    matrix_result_cblas[0], n_matrix_B_col);
        ewtime_cblas = MPI_Wtime();
        printf("[Time taken] CBLAS: %f seconds\n", ewtime_cblas - swtime_cblas);
        printf("CBLAS result matrix showcase:\n");
        print_matrix_less(matrix_result_cblas, n_matrix_A_row, n_matrix_B_col, 5,
                          5);
        printf("Checking result from MPI by CBLAS result:\n");
        check_matrix_equal(matrix_result, matrix_result_cblas, n_matrix_A_row,
                           n_matrix_B_col);
    }
    if (myrank != 0) {
        double* L_B = (double*)malloc(n_matrix_B_row * n_matrix_B_col * sizeof(double));
        MPI_Bcast(L_B, n_matrix_B_row * n_matrix_B_col, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int n_avg_task_row = n_matrix_A_row / (num_processor - 1);
        int n_remainder_task_row = n_matrix_A_row % (num_processor - 1);
        int n_row_recv = myrank <= n_remainder_task_row ? n_avg_task_row + 1 : n_avg_task_row;

        double* L_A_recv = (double*)malloc(n_row_recv * n_matrix_A_col * sizeof(double));
        MPI_Recv(L_A_recv, n_row_recv * n_matrix_A_col, MPI_DOUBLE, 0, myrank,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        debug_proc(myrank, "Received rows from process 0\n");
        // **PERFORMANCE-CRITICAL** point here. Read "matrix_op.h" for details
#if MPI_COMPUTATION_USE_BLAS
        double* L_result = matrix_multiply_matrix_linear_cblas(L_A_recv, L_B, n_row_recv, n_matrix_A_col, n_matrix_B_col);
#else
        double* L_result =  matrix_multiply_matrix_linear(L_A_recv, L_B, n_row_recv, n_matrix_A_col, n_matrix_B_col);
#endif
        debug_proc(myrank, "Computed result for %d rows\n", n_row_recv);
        // send the result back to process 0, blocking send in case L_result be changed before the non-blocking send is completed
        MPI_Send(L_result, n_row_recv * n_matrix_B_col, MPI_DOUBLE, 0, myrank, MPI_COMM_WORLD);
        debug_proc(myrank, "Sent result back to process 0\n");
    }
    MPI_Finalize();
    return 0;
}
