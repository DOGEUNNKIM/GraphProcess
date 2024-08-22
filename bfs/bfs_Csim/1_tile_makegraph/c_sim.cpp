#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include "ap_fixed.h"
#include "hls_stream.h"
#include <queue>

#define VDATA_SIZE 8
#define NUM_NODES 10000
#define NUM_EDGES 160000
#define TILE_SIZE 512
#define START_VERTEX 900
    
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE


typedef struct v_datatype { int64_t data[VDATA_SIZE]; } v_dt;

typedef struct {
    int64_t num_nodes;
    int64_t num_edges;
    v_dt *row_ptr;
    v_dt *col_idx;
    int64_t *values;
} CSRMatrix;

typedef struct {
    int64_t node;
    int64_t in_degree;
    int64_t rank;
} NodeData;


CSRMatrix A;


void bfs(int64_t frontier[], int64_t Vprop[], int64_t row_ptr[], int64_t col_idx[], int64_t start, int64_t NumNode, int64_t NumTile) {
    int64_t value = 0;
    int64_t beforeIDX = 0;
    int64_t currIDX = 1;
    int64_t nextNumIDX = 1;

    frontier[0] = start;
    Vprop[start] = 1;

    for (beforeIDX = 0; beforeIDX < NumNode; beforeIDX = currIDX) {
        currIDX = nextNumIDX;
        value++;

        for (int64_t tile = 0; tile < NumTile; tile++) {
            // Vprop을 프리패치하는 부분
            for (int64_t idx = beforeIDX; idx < currIDX; idx++) {
                int64_t old_Vprop_idx = frontier[idx];
                int64_t row_ptr_start = row_ptr[old_Vprop_idx];
                int64_t row_ptr_end = row_ptr[old_Vprop_idx + 1];
                for (int64_t ptr = row_ptr_start; ptr < row_ptr_end; ptr++) {
                    int64_t col_idx_value = col_idx[ptr];

                    if (Vprop[col_idx_value] == -1 || Vprop[col_idx_value] > value + 1) {
                        Vprop[col_idx_value] = value + 1;
                        frontier[nextNumIDX] = col_idx_value;
                        nextNumIDX++;
                    }
                }
            }
        }
        printf("frontier: ");
        for(int64_t i = 0;i<NumNode;i++){
            printf("%ld, ", frontier[i]);
        }
        printf("beforeIDX = %ld,currIDX = %ld, nextNumIDX = %ld\n",beforeIDX,currIDX,nextNumIDX);
    }
}

void bfs_hls(const v_dt *col,  // Read-Only Vector 1 from hbm -> col index
        const v_dt *row,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
        int64_t *frontier,      // Output Result to hbm -> bfs score before
        v_dt *Vprop       // Output Result to hbm -> bfs score next
        ) {

    int64_t value = -1;
    int64_t beforeIDX = 0;
    int64_t currIDX = 1;
    int64_t nextNumIDX = 1;

int64_t row_ptr_buffer[2];
int64_t col_idx_value;
int64_t *Vprop_buffer = (int64_t *)calloc(TILE_SIZE, sizeof(int64_t));
//int64_t Vprop_buffer[TILE_SIZE];
    for (beforeIDX = 0; beforeIDX < NUM_NODES; beforeIDX = currIDX) {
        if(beforeIDX !=0 && currIDX == nextNumIDX){
            break;
        }
        currIDX = nextNumIDX;
        value++;

        for (int64_t tile = 0; tile < NUM_TILES; tile++) {
            for(int64_t k = 0; k < TILE_SIZE; k += VDATA_SIZE){
                for(int64_t i = 0;i<VDATA_SIZE ;i++){
                      Vprop_buffer[k+i] = Vprop[(tile*TILE_SIZE + k+i) / VDATA_SIZE].data[(tile*TILE_SIZE + k+i) % VDATA_SIZE];
                }
            }

            for (int64_t idx = beforeIDX; idx < currIDX; idx++) {
                int64_t old_Vprop_idx = frontier[idx];
                row_ptr_buffer[0] = row[(tile*NUM_NODES+ old_Vprop_idx)  / VDATA_SIZE].data[(tile*NUM_NODES+ old_Vprop_idx) % VDATA_SIZE];
                row_ptr_buffer[1] = row[(tile*NUM_NODES+ old_Vprop_idx + 1)  / VDATA_SIZE].data[(tile*NUM_NODES+ old_Vprop_idx + 1) % VDATA_SIZE];
                for (int64_t ptr = row_ptr_buffer[0]; ptr < row_ptr_buffer[1]; ptr ++ ) {
                    col_idx_value = col[ptr  / VDATA_SIZE].data[ptr % VDATA_SIZE] - TILE_SIZE*tile;
                    if (Vprop_buffer[col_idx_value] == -1 || Vprop_buffer[col_idx_value] > value + 1) {
                       Vprop_buffer[col_idx_value] = value + 1;
                       #pragma HLS dependence variable=Vprop_buffer false
                       frontier[nextNumIDX] = col_idx_value + TILE_SIZE*tile;
                       nextNumIDX++;
                    }
                }
            }
            for(int64_t kk = 0; kk < TILE_SIZE; kk += VDATA_SIZE){
                for(int64_t ii = 0;ii<VDATA_SIZE ;ii++){
                    Vprop[(tile*TILE_SIZE + kk+ii)  / VDATA_SIZE].data[(tile*TILE_SIZE + kk+ii) % VDATA_SIZE] =Vprop_buffer[kk+ii];
                }
            }
        }
    }
}


typedef struct {
    int64_t src;
    int64_t dst;
} Edge;

int64_t edge_exists(Edge *edges, int64_t edge_count, int64_t src, int64_t dst) {
    for (int64_t i = 0; i < edge_count; i++) {
        if (edges[i].src == src && edges[i].dst == dst) {
            return 1;
        }
    }
    return 0;
}

void generate_random_graph(CSRMatrix *A, int64_t num_nodes, int64_t num_edges, int64_t *in_degree) {
    A->num_nodes = num_nodes;
    A->num_edges = num_edges;
    A->row_ptr = (v_dt *)malloc(((num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    A->col_idx = (v_dt *)malloc((num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    A->values = (int64_t *)malloc(num_edges * sizeof(int64_t));

    int64_t *out_degree = (int64_t *)calloc(num_nodes, sizeof(int64_t));
    Edge *edges = (Edge *)malloc(num_edges * sizeof(Edge));
    int64_t edge_count = 0;

    srand(time(NULL));
    while (edge_count < num_edges) {
        int64_t src = rand() % num_nodes;
        int64_t dst = rand() % num_nodes;
        if (!edge_exists(edges, edge_count, src, dst)) {
            edges[edge_count].src = src;
            edges[edge_count].dst = dst;
            out_degree[src]++;
            in_degree[dst]++;
            A->values[edge_count] = 1.0f; // Edge weights are set to 1.0
            edge_count++;
        }
    }

    // Initialize row_ptr
    A->row_ptr[0].data[0] = 0.0f;
    for (int64_t i = 1; i <= num_nodes; i++) {
        A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] = A->row_ptr[(i - 1) / VDATA_SIZE].data[(i - 1) % VDATA_SIZE] + out_degree[i - 1];
    }

    // Temporary array to keep track of positions in col_idx
    int64_t *current_pos = (int64_t *)malloc(num_nodes * sizeof(int64_t));
    for (int64_t i = 0; i < num_nodes; i++) {
        current_pos[i] = A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
    }

    // Fill col_idx based on row_ptr
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t src = edges[i].src;
        int64_t dst = edges[i].dst;
        int64_t pos = current_pos[src]++;
        A->col_idx[pos / VDATA_SIZE].data[pos % VDATA_SIZE] = dst;
    }

    //printf("row_ptr\n");
    //for (int64_t i = 0; i < num_nodes + 1; i++) {
    //    printf("%ld ", A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]);
    //}
    //printf("\n");
    //
    //printf("col_idx\n");
    //for (int64_t i = 0; i < num_edges; i++) {
    //    printf("%ld ", A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE]);
    //}
    //printf("\n");

    free(out_degree);
    free(edges);
    free(current_pos);
}

void tile_CSRMatrix_func(const CSRMatrix *A, CSRMatrix *T, int64_t tile_size) {
    int64_t num_nodes = A->num_nodes;
    int64_t num_tiles = (num_nodes + tile_size - 1) / tile_size;

    T->num_nodes = num_nodes;
    T->num_edges = A->num_edges;
    T->row_ptr = (v_dt *)malloc(((num_nodes *num_tiles  + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt) * num_tiles);
    T->col_idx = (v_dt *)malloc((A->num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    T->values = (int64_t *)malloc(A->num_edges * sizeof(int64_t));

    // Create num_tiles csr (row ptr & col index)
    CSRMatrix *tile_CSRMatrix = (CSRMatrix *)malloc(num_tiles * sizeof(CSRMatrix));

    for (int64_t t = 0; t < num_tiles; t++) {
        tile_CSRMatrix[t].num_nodes = num_nodes;
        tile_CSRMatrix[t].num_edges = 0; // This will be adjusted later
        tile_CSRMatrix[t].row_ptr = (v_dt *)malloc(((num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
        tile_CSRMatrix[t].col_idx = NULL; // This will be allocated later
        tile_CSRMatrix[t].values = NULL; // This will be allocated later
    }
    
    // Count the edges for each tile
    for (int64_t i = 0; i < A->num_edges; i++) {
        int64_t index = A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE] / tile_size;
        tile_CSRMatrix[index].num_edges++;
    }
    for (int64_t i = 0; i < num_tiles; i++) {
        tile_CSRMatrix[i].col_idx = (v_dt *)malloc(((tile_CSRMatrix[i].num_edges) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
        tile_CSRMatrix[i].values = (int64_t *)malloc((tile_CSRMatrix[i].num_edges) * sizeof(int64_t));
        tile_CSRMatrix[i].row_ptr[0].data[0] = 0; // Initialize the first element of row_ptr
    }
    int64_t col_idx_ptr[num_tiles] = {0};
    for (int64_t i = 0; i < A->num_edges; i++) {
        int64_t index = A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE] / tile_size;
        tile_CSRMatrix[index].col_idx[col_idx_ptr[index] / VDATA_SIZE].data[col_idx_ptr[index] % VDATA_SIZE] = A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE];
        col_idx_ptr[index]++;
    }

    // Populate row_ptr for each tile
    for (int64_t t = 0; t < num_tiles; t++) {
        for (int64_t i = 0; i <= num_nodes; i++) {
            tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] = 0;
        }
    }

    for (int64_t i = 0; i < num_nodes; i++) {
        int64_t row_start = A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
        int64_t row_end = A->row_ptr[(i + 1) / VDATA_SIZE].data[(i + 1) % VDATA_SIZE];

        for (int64_t j = row_start; j < row_end; j++) {
            int64_t col = A->col_idx[j / VDATA_SIZE].data[j % VDATA_SIZE];
            int64_t tile_index = col / tile_size;

            tile_CSRMatrix[tile_index].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]++;
        }
    }

    for (int64_t t = 0; t < num_tiles; t++) {
        int64_t cumulative_sum = 0;
        for (int64_t i = 0; i <= num_nodes; i++) {
            int64_t temp = tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
            tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] = cumulative_sum;
            cumulative_sum += temp;
        }
    }


    //printf("tile row_ptr\n");
    for (int64_t t = 0; t < num_tiles; t++) {
        for (int64_t i = 0; i < tile_CSRMatrix[t].num_nodes+1; i++) {
            //printf("%ld ", tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]);
            if(t > 0){
              tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] += tile_CSRMatrix[t-1].row_ptr[num_nodes / VDATA_SIZE].data[num_nodes % VDATA_SIZE];
            }
        }
        //printf("\n");
    }
    for (int64_t t = 0; t < num_tiles; t++) {
        for (int64_t i = 1; i <= num_nodes; i++) {
             T->row_ptr[(t*num_nodes + i) / VDATA_SIZE].data[(t*num_nodes + i) % VDATA_SIZE] =  tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
        }
    }

    int64_t j=0;

    for (int64_t t = 0; t < num_tiles; t++) {
        if(t>0){
          j += (tile_CSRMatrix[t-1].num_edges);
        }
        for (int64_t i = 0; i < tile_CSRMatrix[t].num_edges; i++) {
             T->col_idx[(j + i) / VDATA_SIZE].data[(j + i) % VDATA_SIZE] =  tile_CSRMatrix[t].col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE];
        }
    }
    T->row_ptr[0].data[0] = 0;

    // Free allocated memory for tile_CSRMatrix
    for (int64_t t = 0; t < num_tiles; t++) {
        free(tile_CSRMatrix[t].row_ptr);
        free(tile_CSRMatrix[t].col_idx);
        free(tile_CSRMatrix[t].values);
    }
    free(tile_CSRMatrix);
}

//BFS code
void bfs_CSR(const CSRMatrix *A, int64_t start_node, int64_t *distances) {
    int64_t num_nodes = A->num_nodes;
    std::queue<int> q;
    bool *visited = (bool *)malloc(num_nodes * sizeof(bool));
    std::memset(visited, 0, num_nodes * sizeof(bool));

    // Initialize distances array
    for (int64_t i = 0; i < num_nodes; i++) {
        distances[i] = -1;  // -1 indicates that the node has not been visited yet
    }

    // Start BFS from the start_node
    visited[start_node] = true;
    distances[start_node] = 0;
    q.push(start_node);
    int64_t iter =0;

    while (!q.empty()) {
        int64_t u = q.front();
        q.pop();
        iter ++;
    //printf("iter = %ld\n", iter);
        int64_t buffer_start = A->row_ptr[u / VDATA_SIZE].data[u % VDATA_SIZE];
        int64_t buffer_end = A->row_ptr[(u + 1) / VDATA_SIZE].data[(u + 1) % VDATA_SIZE];
        int64_t buffer_size = buffer_end - buffer_start;
        int64_t col_idx_buffer[10];

        for (int64_t b = 0; b < buffer_size; b += 10) {
            int64_t chunk_size = (b + 10 > buffer_size) ? buffer_size - b : 10;
            for (int64_t k = 0; k < chunk_size; k++) {
                col_idx_buffer[k] = A->col_idx[(buffer_start + b + k) / VDATA_SIZE].data[(buffer_start + b + k) % VDATA_SIZE];
            }

            for (int64_t k = 0; k < chunk_size; k++) {
                int64_t v = col_idx_buffer[k];
                if (!visited[v]) {
                    visited[v] = true;
                    distances[v] = distances[u] + 1;
                    q.push(v);
                }
            }
        }
    }

    free(visited);
}


//hls::stream<int,100> queue;

int main() {

    int64_t *in_degree = (int64_t *)calloc(NUM_NODES, sizeof(int64_t));
    generate_random_graph(&A, NUM_NODES, NUM_EDGES, in_degree);

    int64_t *r = (int64_t *)malloc(A.num_nodes * sizeof(int64_t));

    // BFS 계산
    bfs_CSR(&A, START_VERTEX ,r);
    
    int64_t *frontier = (int64_t *)malloc(NUM_NODES * sizeof(int64_t));

    v_dt *Vprop = (v_dt *)malloc((size_t)((NUM_NODES+VDATA_SIZE-1)/VDATA_SIZE) * sizeof(v_dt));
    
    for (int64_t i = 0; i < NUM_NODES; i++) {
        frontier[i] = -1;
        Vprop[i / VDATA_SIZE].data[i % VDATA_SIZE] = -1;
    }

    frontier[0] = START_VERTEX;
    Vprop[START_VERTEX / VDATA_SIZE].data[START_VERTEX % VDATA_SIZE]= 0;

    CSRMatrix T;
    tile_CSRMatrix_func(&A, &T, TILE_SIZE);
    printf("Start FPGA\n");
    bfs_hls(T.col_idx, T.row_ptr, frontier, Vprop);
    printf("Finish FPGA\n");
    NodeData *nodes = (NodeData *)malloc(A.num_nodes * sizeof(NodeData));
    for (int64_t i = 0; i < A.num_nodes; i++) {
        nodes[i].node = i;
        nodes[i].in_degree = in_degree[i];
        nodes[i].rank = r[i];
    }
    // BFS 결과 확인
    for (int64_t i = 0; i < A.num_nodes; i++) {
        //std::cout << "cpu: " << nodes[i].rank << std::endl;
        //std::cout << "hls: " << Vprop[i / VDATA_SIZE].data[i % VDATA_SIZE] << std::endl;
        if(nodes[i].rank != Vprop[i / VDATA_SIZE].data[i % VDATA_SIZE]){
            printf("TEST FAIL\n");
            free(A.row_ptr);
            free(A.col_idx);
            free(A.values);
            free(r);
            free(in_degree);
            free(nodes);
            return 0;
        }
    }
    printf("TEST PASS\n");


    // 메모리 해제
    free(A.row_ptr);
    free(A.col_idx);
    free(A.values);
    free(r);
    free(in_degree);
    free(nodes);

    return 0;
}

