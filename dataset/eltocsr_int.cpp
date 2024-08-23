#include <stdio.h>
#include <stdlib.h>

#define VDATA_SIZE 16

using namespace std;

typedef struct v_datatype {
    int data[VDATA_SIZE];
} v_dt;

typedef struct {
    int num_nodes;
    int num_edges;
    v_dt *row_ptr;
    v_dt *col_idx;
    int *values;
} CSRMatrix;

void generate_graph_from_file(CSRMatrix *A, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    int src, dst;

    A->num_nodes = 4847571;
    A->num_edges = 68993773;

    int row_ptr_size = (A->num_nodes + VDATA_SIZE) / VDATA_SIZE;  // Adjust the size for row_ptr
    A->row_ptr = (v_dt *)malloc(row_ptr_size * sizeof(v_dt));
    A->col_idx = (v_dt *)malloc((A->num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    A->values = (int *)malloc(A->num_edges * sizeof(int));

    // Initialize row_ptr to zero
    for (int i = 0; i < row_ptr_size; i++) {
        for (int j = 0; j < VDATA_SIZE; j++) {
            A->row_ptr[i].data[j] = 0;
        }
    }

    // Count the number of edges per node
    rewind(file);
    while (fscanf(file, "%d %d", &src, &dst) != EOF) {
        A->row_ptr[(src + 1) / VDATA_SIZE].data[(src + 1) % VDATA_SIZE]++;
    }

    // Accumulate the row_ptr values
    for (int i = 1; i <= A->num_nodes; i++) {
        A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] += A->row_ptr[(i - 1) / VDATA_SIZE].data[(i - 1) % VDATA_SIZE];
    }

    // Temporary array to track current position in col_idx
    int *current_pos = (int *)malloc(A->num_nodes * sizeof(int));
    for (int i = 0; i < A->num_nodes; i++) {
        current_pos[i] = A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
    }

    // Fill col_idx and values
    rewind(file);
    while (fscanf(file, "%d %d", &src, &dst) != EOF) {
        int pos = current_pos[src]++;
        A->col_idx[pos / VDATA_SIZE].data[pos % VDATA_SIZE] = dst;
        A->values[pos] = 1;  // Edge weights are set to 1
    }

    // Clean up
    free(current_pos);
    fclose(file);
}

void save_csr_to_file(CSRMatrix *A, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    // Write num_nodes and num_edges
    fwrite(&A->num_nodes, sizeof(int), 1, file);
    fwrite(&A->num_edges, sizeof(int), 1, file);

    // Write row_ptr, col_idx, and values arrays
    int row_ptr_size = (A->num_nodes + VDATA_SIZE) / VDATA_SIZE;
    fwrite(A->row_ptr, sizeof(v_dt), row_ptr_size, file);
    fwrite(A->col_idx, sizeof(v_dt), (A->num_edges + VDATA_SIZE - 1) / VDATA_SIZE, file);
    fwrite(A->values, sizeof(int), A->num_edges, file);

    fclose(file);
}

int main() {
    CSRMatrix A;
    generate_graph_from_file(&A, "/home/kdg6245/graph/dataset/soc-LiveJournal1.txt");
    save_csr_to_file(&A, "csr_matrix_LiveJournal1_int.bin");

    for (int i = 0; i < 50; i++) {
        printf("Node %d col_idx %d\n", i, A.col_idx[0].data[i]);
    }

    // Free the allocated memory
    free(A.row_ptr);
    free(A.col_idx);
    free(A.values);

    return 0;
}
