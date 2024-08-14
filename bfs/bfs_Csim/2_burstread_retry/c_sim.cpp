#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include "ap_fixed.h"
#include "hls_stream.h"
#include <queue>

#define VDATA_SIZE 16
#define NUM_NODES 1024
#define NUM_EDGES 16000
#define TILE_SIZE 512
#define START_VERTEX 1
    
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE


typedef struct v_datatype { int data[VDATA_SIZE]; } v_dt;
typedef struct r_f_datatype { int data[2]; } row_dt;

typedef struct {
    int num_nodes;
    int num_edges;
    v_dt *row_ptr;
    v_dt *col_idx;
    int *values;
} CSRMatrix;

typedef struct {
    int node;
    int in_degree;
    int rank;
} NodeData;


CSRMatrix A;


void bfs(int frontier[], int Vprop[], int row_ptr[], int col_idx[], int start, int NumNode, int NumTile) {
    int value = 0;
    int beforeIDX = 0;
    int currIDX = 1;
    int nextNumIDX = 1;

    frontier[0] = start;
    Vprop[start] = 1;

    for (beforeIDX = 0; beforeIDX < NumNode; beforeIDX = currIDX) {
        currIDX = nextNumIDX;
        value++;

        for (int tile = 0; tile < NumTile; tile++) {
            // Vprop을 프리패치하는 부분
            for (int idx = beforeIDX; idx < currIDX; idx++) {
                int old_Vprop_idx = frontier[idx];
                int row_ptr_start = row_ptr[old_Vprop_idx];
                int row_ptr_end = row_ptr[old_Vprop_idx + 1];
                for (int ptr = row_ptr_start; ptr < row_ptr_end; ptr++) {
                    int col_idx_value = col_idx[ptr];

                    if (Vprop[col_idx_value] == -1 || Vprop[col_idx_value] > value + 1) {
                        Vprop[col_idx_value] = value + 1;
                        frontier[nextNumIDX] = col_idx_value;
                        nextNumIDX++;
                    }
                }
            }
        }
        printf("frontier: ");
        for(int i = 0;i<NumNode;i++){
            printf("%d, ", frontier[i]);
        }
        printf("beforeIDX = %d,currIDX = %d, nextNumIDX = %d\n",beforeIDX,currIDX,nextNumIDX);
    }
}

void bfs_hls(const v_dt *col,  // Read-Only Vector 1 from hbm -> col index
        const v_dt *row,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
        int *frontier,      // Output Result to hbm -> bfs score before
        v_dt *Vprop       // Output Result to hbm -> bfs score next
        ) {

#pragma HLS INTERFACE m_axi port = col offset = slave bundle = gmem0 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = row offset = slave bundle = gmem1 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = frontier offset = slave bundle = gmem3 depth = 4096
#pragma HLS INTERFACE m_axi port = Vprop offset = slave bundle = gmem4 max_widen_bitwidth=512 depth = 4096

#pragma HLS INTERFACE s_axilite port=col bundle=control
#pragma HLS INTERFACE s_axilite port=row bundle=control
#pragma HLS INTERFACE s_axilite port=frontier bundle=control
#pragma HLS INTERFACE s_axilite port=Vprop bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int value = -1;
    int beforeIDX = 0;
    int currIDX = 1;
    int nextNumIDX = 1;

int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
int col_idx_value;
int Vprop_buffer[TILE_SIZE];
#pragma HLS array_partition variable=Vprop_buffer factor=16 cyclic

    for (beforeIDX = 0; beforeIDX < NUM_NODES; beforeIDX = currIDX) {
        if(beforeIDX !=0 && currIDX == nextNumIDX){
            break;
        }
        currIDX = nextNumIDX;
        value++;

        for (int tile = 0; tile < NUM_TILES; tile++) {
            for(int k = 0; k < TILE_SIZE; k += VDATA_SIZE){
            #pragma HLS pipeline II=1
                for(int i = 0;i<VDATA_SIZE ;i++){
                      Vprop_buffer[k+i] = Vprop[(tile*TILE_SIZE + k+i)>>4].data[(tile*TILE_SIZE + k+i)%16];
                }
            }

            for (int idx = beforeIDX; idx < currIDX; idx++) {
                int old_Vprop_idx = frontier[idx];
                row_ptr_buffer[0] = row[(tile*NUM_NODES+ old_Vprop_idx) >> 4].data[(tile*NUM_NODES+ old_Vprop_idx)%16];
                row_ptr_buffer[1] = row[(tile*NUM_NODES+ old_Vprop_idx + 1) >> 4].data[(tile*NUM_NODES+ old_Vprop_idx + 1)%16];

                for (int ptr = row_ptr_buffer[0]; ptr < row_ptr_buffer[1]; ptr ++ ) {
                    col_idx_value = col[ptr >> 4].data[ptr%16] - TILE_SIZE*tile;
                    if (Vprop_buffer[col_idx_value] == -1 || Vprop_buffer[col_idx_value] > value + 1) {
                       Vprop_buffer[col_idx_value] = value + 1;
                       frontier[nextNumIDX] = col_idx_value + TILE_SIZE*tile;
                       nextNumIDX++;
                    }

                }
            }
            for(int kk = 0; kk < TILE_SIZE; kk += VDATA_SIZE){
            #pragma HLS pipeline II=1
                for(int ii = 0;ii<VDATA_SIZE ;ii++){
                    Vprop[(tile*TILE_SIZE + kk+ii)>>4].data[(tile*TILE_SIZE + kk+ii)%16] =Vprop_buffer[kk+ii];
                }
            }
        }
    }
}


typedef struct {
    int src;
    int dst;
} Edge;

int edge_exists(Edge *edges, int edge_count, int src, int dst) {
    for (int i = 0; i < edge_count; i++) {
        if (edges[i].src == src && edges[i].dst == dst) {
            return 1;
        }
    }
    return 0;
}

void generate_random_graph(CSRMatrix *A, int num_nodes, int num_edges, int *in_degree) {
    A->num_nodes = num_nodes;
    A->num_edges = num_edges;
    A->row_ptr = (v_dt *)malloc(((num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    A->col_idx = (v_dt *)malloc((num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    A->values = (int *)malloc(num_edges * sizeof(int));

    int *out_degree = (int *)calloc(num_nodes, sizeof(int));
    Edge *edges = (Edge *)malloc(num_edges * sizeof(Edge));
    int edge_count = 0;

    srand(time(NULL));
    while (edge_count < num_edges) {
        int src = rand() % num_nodes;
        int dst = rand() % num_nodes;
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
    for (int i = 1; i <= num_nodes; i++) {
        A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] = A->row_ptr[(i - 1) / VDATA_SIZE].data[(i - 1) % VDATA_SIZE] + out_degree[i - 1];
    }

    // Temporary array to keep track of positions in col_idx
    int *current_pos = (int *)malloc(num_nodes * sizeof(int));
    for (int i = 0; i < num_nodes; i++) {
        current_pos[i] = A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
    }

    // Fill col_idx based on row_ptr
    for (int i = 0; i < num_edges; i++) {
        int src = edges[i].src;
        int dst = edges[i].dst;
        int pos = current_pos[src]++;
        A->col_idx[pos / VDATA_SIZE].data[pos % VDATA_SIZE] = dst;
    }

    //printf("row_ptr\n");
    //for (int i = 0; i < num_nodes + 1; i++) {
    //    printf("%d ", A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]);
    //}
    //printf("\n");
    //
    //printf("col_idx\n");
    //for (int i = 0; i < num_edges; i++) {
    //    printf("%d ", A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE]);
    //}
    //printf("\n");

    free(out_degree);
    free(edges);
    free(current_pos);
}

void tile_CSRMatrix_func(const CSRMatrix *A, CSRMatrix *T, int tile_size) {
    int num_nodes = A->num_nodes;
    int num_tiles = (num_nodes + tile_size - 1) / tile_size;

    T->num_nodes = num_nodes;
    T->num_edges = A->num_edges;
    T->row_ptr = (v_dt *)malloc(((num_nodes *num_tiles  + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt) * num_tiles);
    T->col_idx = (v_dt *)malloc((A->num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    T->values = (int *)malloc(A->num_edges * sizeof(int));

    // Create num_tiles csr (row ptr & col index)
    CSRMatrix *tile_CSRMatrix = (CSRMatrix *)malloc(num_tiles * sizeof(CSRMatrix));

    for (int t = 0; t < num_tiles; t++) {
        tile_CSRMatrix[t].num_nodes = num_nodes;
        tile_CSRMatrix[t].num_edges = 0; // This will be adjusted later
        tile_CSRMatrix[t].row_ptr = (v_dt *)malloc(((num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
        tile_CSRMatrix[t].col_idx = NULL; // This will be allocated later
        tile_CSRMatrix[t].values = NULL; // This will be allocated later
    }
    
    // Count the edges for each tile
    for (int i = 0; i < A->num_edges; i++) {
        int index = A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE] / tile_size;
        tile_CSRMatrix[index].num_edges++;
    }
    for (int i = 0; i < num_tiles; i++) {
        tile_CSRMatrix[i].col_idx = (v_dt *)malloc(((tile_CSRMatrix[i].num_edges) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
        tile_CSRMatrix[i].values = (int *)malloc((tile_CSRMatrix[i].num_edges) * sizeof(int));
        tile_CSRMatrix[i].row_ptr[0].data[0] = 0; // Initialize the first element of row_ptr
    }
    int col_idx_ptr[num_tiles] = {0};
    for (int i = 0; i < A->num_edges; i++) {
        int index = A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE] / tile_size;
        tile_CSRMatrix[index].col_idx[col_idx_ptr[index] / VDATA_SIZE].data[col_idx_ptr[index] % VDATA_SIZE] = A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE];
        col_idx_ptr[index]++;
    }

    // Populate row_ptr for each tile
    for (int t = 0; t < num_tiles; t++) {
        for (int i = 0; i <= num_nodes; i++) {
            tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] = 0;
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        int row_start = A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
        int row_end = A->row_ptr[(i + 1) / VDATA_SIZE].data[(i + 1) % VDATA_SIZE];

        for (int j = row_start; j < row_end; j++) {
            int col = A->col_idx[j / VDATA_SIZE].data[j % VDATA_SIZE];
            int tile_index = col / tile_size;

            tile_CSRMatrix[tile_index].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]++;
        }
    }

    for (int t = 0; t < num_tiles; t++) {
        int cumulative_sum = 0;
        for (int i = 0; i <= num_nodes; i++) {
            int temp = tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
            tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] = cumulative_sum;
            cumulative_sum += temp;
        }
    }


    //printf("tile row_ptr\n");
    for (int t = 0; t < num_tiles; t++) {
        for (int i = 0; i < tile_CSRMatrix[t].num_nodes+1; i++) {
            //printf("%d ", tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]);
            if(t > 0){
              tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] += tile_CSRMatrix[t-1].row_ptr[num_nodes / VDATA_SIZE].data[num_nodes % VDATA_SIZE];
            }
        }
        //printf("\n");
    }
    for (int t = 0; t < num_tiles; t++) {
        for (int i = 1; i <= num_nodes; i++) {
             T->row_ptr[(t*num_nodes + i) / VDATA_SIZE].data[(t*num_nodes + i) % VDATA_SIZE] =  tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
        }
    }

    int j=0;

    for (int t = 0; t < num_tiles; t++) {
        if(t>0){
          j += (tile_CSRMatrix[t-1].num_edges);
        }
        for (int i = 0; i < tile_CSRMatrix[t].num_edges; i++) {
             T->col_idx[(j + i) / VDATA_SIZE].data[(j + i) % VDATA_SIZE] =  tile_CSRMatrix[t].col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE];
        }
    }
    T->row_ptr[0].data[0] = 0;

    // Free allocated memory for tile_CSRMatrix
    for (int t = 0; t < num_tiles; t++) {
        free(tile_CSRMatrix[t].row_ptr);
        free(tile_CSRMatrix[t].col_idx);
        free(tile_CSRMatrix[t].values);
    }
    free(tile_CSRMatrix);
}

//BFS code
void bfs_CSR(const CSRMatrix *A, int start_node, int *distances) {
    int num_nodes = A->num_nodes;
    std::queue<int> q;
    bool *visited = (bool *)malloc(num_nodes * sizeof(bool));
    std::memset(visited, 0, num_nodes * sizeof(bool));

    // Initialize distances array
    for (int i = 0; i < num_nodes; i++) {
        distances[i] = -1;  // -1 indicates that the node has not been visited yet
    }

    // Start BFS from the start_node
    visited[start_node] = true;
    distances[start_node] = 0;
    q.push(start_node);
    int iter =0;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        iter ++;
    //printf("iter = %d\n", iter);
        int buffer_start = A->row_ptr[u / VDATA_SIZE].data[u % VDATA_SIZE];
        int buffer_end = A->row_ptr[(u + 1) / VDATA_SIZE].data[(u + 1) % VDATA_SIZE];
        int buffer_size = buffer_end - buffer_start;
        int col_idx_buffer[10];

        for (int b = 0; b < buffer_size; b += 10) {
            int chunk_size = (b + 10 > buffer_size) ? buffer_size - b : 10;
            for (int k = 0; k < chunk_size; k++) {
                col_idx_buffer[k] = A->col_idx[(buffer_start + b + k) / VDATA_SIZE].data[(buffer_start + b + k) % VDATA_SIZE];
            }

            for (int k = 0; k < chunk_size; k++) {
                int v = col_idx_buffer[k];
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


int main() {
    //int check = 1;
    //for (int i = 0; i < 10 ; i ++){
    //    queue.write(i);
    //}
    //while (check != 0) {
    //   if ( !queue.empty()){
    //      int k;
    //      queue.read_nb(k);
    //      
    //      check = 1;
    //      printf("k=%d\n",k);
    //   }
    //     
    //   check = (check << 1);
    //}
    int *in_degree = (int *)calloc(NUM_NODES, sizeof(int));
    generate_random_graph(&A, NUM_NODES, NUM_EDGES, in_degree);

    int *r = (int *)malloc(A.num_nodes * sizeof(int));

    // BFS 계산
    bfs_CSR(&A, START_VERTEX ,r);
    
    int frontier[NUM_NODES];
    v_dt Vprop[(NUM_NODES+VDATA_SIZE-1)/VDATA_SIZE];
    for (int i = 0; i < NUM_NODES; i++) {
        frontier[i] = -1;
        Vprop[i>>4].data[i%16] = -1;
    }

    frontier[0] = START_VERTEX;
    Vprop[START_VERTEX>>4].data[START_VERTEX%16]= 0;

    CSRMatrix T;
    tile_CSRMatrix_func(&A, &T, TILE_SIZE);
    printf("Start FPGA\n");
    bfs_hls(T.col_idx, T.row_ptr, frontier, Vprop);
    printf("Finish FPGA\n");
    NodeData *nodes = (NodeData *)malloc(A.num_nodes * sizeof(NodeData));
    for (int i = 0; i < A.num_nodes; i++) {
        nodes[i].node = i;
        nodes[i].in_degree = in_degree[i];
        nodes[i].rank = r[i];
    }
    // BFS 결과 확인
    for (int i = 0; i < A.num_nodes; i++) {
        std::cout << "cpu: " << nodes[i].rank << std::endl;
        std::cout << "hls: " << Vprop[i>>4].data[i%16] << std::endl;
        if(nodes[i].rank != Vprop[i>>4].data[i%16]){
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

