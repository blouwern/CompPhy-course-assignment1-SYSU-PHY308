#pragma once
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

double get_timer() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void debug_proc(int rank, char* message, ...) {
#if REPORT_DEBUG_INFO
    va_list args;
    va_start(args, message);
    printf("|%12.6f|%2d|", get_timer(), rank);
    vprintf(message, args);
    fflush(stdout);
    va_end(args);
#else
    return;
#endif
}

void print_matrix(double** matrix, int n_row, int n_col) {
    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_matrix_less(double** matrix, int n_row, int n_col, int n_show_row, int n_show_col) {
    if(n_show_row > n_row || n_show_col > n_col){
        printf("Error: matrix showcase size (%d,%d) outrange the actual matrix (%d,%d)", n_show_row, n_show_col, n_row, n_col);
        return;
    }
    printf("---------------------------------------------\n");
    printf("The top-left %d x %d block of the matrix:\n", n_show_row, n_show_col);
    for (int i = 0; i < n_row && i < n_show_row; i++) {
        printf("|");
        for (int j = 0; j < n_col && j < n_show_col; j++) {
            printf("%.6f ", matrix[i][j]);
        }
        printf("|\n");
    }
}

void print_L_matrix_less(double* matrix_L, int n_row, int n_col, int n_show_row, int n_show_col) {
    if(n_show_row > n_row || n_show_col > n_col){
        printf("Error: matrix showcase size (%d,%d) outrange the actual matrix (%d,%d)", n_show_row, n_show_col, n_row, n_col);
        return;
    }
    printf("---------------------------------------------\n");
    printf("The top-left %d x %d block of the matrix:\n", n_show_row, n_show_col);
    for (int i = 0; i < n_row && i < n_show_row; i++) {
        printf("|");
        for (int j = 0; j < n_col && j < n_show_col; j++) {
            printf("%.6f ", matrix_L[i * n_row + j]);
        }
        printf("|\n");
    }
}
