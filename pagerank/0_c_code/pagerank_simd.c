#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define ALPHA 0.85
#define MAX_ITER 1000
#define EPSILON 1e-6
#define NUM_NODES 100
#define NUM_EDGES 1000

typedef struct {
    int num_nodes;
    int num_edges;
    int *row_ptr;
    int *col_idx;
    double *values;
} CSRMatrix;

typedef struct {
    int node;
    int in_degree;
    double rank;
} NodeData;

void generate_random_graph(CSRMatrix *A, int num_nodes, int num_edges, int *in_degree) {
    A->num_nodes = num_nodes;
    A->num_edges = num_edges;
    A->row_ptr = (int *)malloc((num_nodes + 1) * sizeof(int));
    A->col_idx = (int *)malloc(num_edges * sizeof(int));
    A->values = (double *)malloc(num_edges * sizeof(double));

    int *out_degree = (int *)calloc(num_nodes, sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < num_edges; i++) {
        int src = rand() % num_nodes;
        int dst = rand() % num_nodes;
        A->col_idx[i] = dst;
        out_degree[src]++;
        in_degree[dst]++;
        A->values[i] = 1.0; // Edge weights are set to 1.0
    }

    A->row_ptr[0] = 0;
    for (int i = 1; i <= num_nodes; i++) {
        A->row_ptr[i] = A->row_ptr[i - 1] + out_degree[i - 1];
    }

    // Print out-degree and in-degree for each node
    //for (int i = 0; i < num_nodes; i++) {
    //    printf("Node %d: %d outgoing edges, %d incoming edges\n", i, out_degree[i], in_degree[i]);
    //}

    free(out_degree);
}

void pageRank_CSR(const CSRMatrix *A, double *r) {
    int num_nodes = A->num_nodes;
    double *r_new = (double *)malloc(num_nodes * sizeof(double));
    double base_score = (1.0 - ALPHA) / num_nodes;

    for (int i = 0; i < num_nodes; i++) {
        r[i] = 1.0 / num_nodes;
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {

        for (int i = 0; i < num_nodes; i++) {
            r_new[i] = base_score;
        }

        #pragma omp parallel for
        for (int u = 0; u < num_nodes; u++) {
            //int out_degree_u = A->row_ptr[u + 1] - A->row_ptr[u];

            // 버퍼 생성
            int buffer_start = A->row_ptr[u];
            int buffer_end = A->row_ptr[u + 1];
            int buffer_size = buffer_end - buffer_start;
            int out_degree_u = buffer_end - buffer_start;
            int col_idx_buffer[10];

            // col_idx 버퍼를 10개씩 채움
            for (int b = 0; b < buffer_size; b += 10) {
                int chunk_size = (b + 10 > buffer_size) ? buffer_size - b : 10;
                for (int k = 0; k < chunk_size; k++) {
                    col_idx_buffer[k] = A->col_idx[buffer_start + b + k];
                }

                // SIMD 병렬 처리
                #pragma omp simd
                for (int k = 0; k < chunk_size; k++) {
                    int v = col_idx_buffer[k];
                    #pragma omp atomic
                    r_new[v] += ALPHA * r[u] / out_degree_u;
                }
            }
        }

        double diff = 0.0;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < num_nodes; i++) {
            diff += fabs(r_new[i] - r[i]);
        }
        if (diff < EPSILON) {
            break;
        }

        memcpy(r, r_new, num_nodes * sizeof(double));
    }

    free(r_new);
}

int compare_by_rank(const void *a, const void *b) {
    NodeData *nodeA = (NodeData *)a;
    NodeData *nodeB = (NodeData *)b;
    if (nodeB->rank > nodeA->rank) return 1;
    if (nodeB->rank < nodeA->rank) return -1;
    return 0;
}

int main() {
    CSRMatrix A;
    int *in_degree = (int *)calloc(NUM_NODES, sizeof(int));
    generate_random_graph(&A, NUM_NODES, NUM_EDGES, in_degree);

    double *r = (double *)malloc(A.num_nodes * sizeof(double));
    
    // PageRank 계산
    pageRank_CSR(&A, r);

    NodeData *nodes = (NodeData *)malloc(A.num_nodes * sizeof(NodeData));
    for (int i = 0; i < A.num_nodes; i++) {
        nodes[i].node = i;
        nodes[i].in_degree = in_degree[i];
        nodes[i].rank = r[i];
    }

    // PageRank 순으로 정렬하여 출력
    qsort(nodes, A.num_nodes, sizeof(NodeData), compare_by_rank);
    printf("Sorted by rank:\n");
    for (int i = 0; i < A.num_nodes; i++) {
        printf("Node %d: in-degree %d, rank %f\n", nodes[i].node, nodes[i].in_degree, nodes[i].rank);
    }

    // 메모리 해제
    free(A.row_ptr);
    free(A.col_idx);
    free(A.values);
    free(r);
    free(in_degree);
    free(nodes);

    return 0;
}
