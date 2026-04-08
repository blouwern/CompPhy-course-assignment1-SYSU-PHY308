#include "cblas.h"
#include "info_op.h"
#include "matrix_op.h"

#include <stdio.h>

int main(int argc, char* argv[]) {
    const char* module_name = "CBLAS sequential";
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
    double** matrix_A = generate_matrix(n_matrix_A_row, n_matrix_A_col);
    double** matrix_B = generate_matrix(n_matrix_B_row, n_matrix_B_col);
    double** matrix_result = make_matrix(n_matrix_A_row, n_matrix_B_col);
    double swtime = get_timer();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_matrix_A_row,
                n_matrix_B_col, n_matrix_A_col, 1.0, matrix_A[0], n_matrix_A_col,
                matrix_B[0], n_matrix_B_col, 0.0, matrix_result[0],
                n_matrix_B_col);
    double ewtime = get_timer();
    printf("Computation completed.\n");
    printf("Result matrix showcase:\n");
    print_matrix_less(matrix_result, n_matrix_A_row, n_matrix_B_col, 5, 5);
    printf("[Time taken]<%s> : %f seconds\n", module_name, ewtime - swtime);
    free(matrix_A[0]);
    free(matrix_A);
    free(matrix_B[0]);
    free(matrix_B);
    free(matrix_result[0]);
    free(matrix_result);

    return 0;
}
