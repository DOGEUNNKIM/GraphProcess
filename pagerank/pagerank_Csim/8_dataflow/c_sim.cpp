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
//#define ALPHA 0.85
#define MAX_ITER 1000
//#define EPSILON 1e-6
#define NUM_NODES 4096
#define NUM_EDGES 4096*16
#define TILE_SIZE 512
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE


float ALPHA = 0.85;
float EPSILON = 1e-6;

float a = 1;
float base_score = (a - ALPHA) / NUM_NODES;

typedef struct v_datatype { int data[VDATA_SIZE]; } v_dt;
typedef struct v_f_datatype { float data[VDATA_SIZE]; } v_dt_f;
typedef struct r_f_datatype { int data[2]; } row_dt;

typedef struct {
    int num_nodes;
    int num_edges;
    v_dt_f *row_ptr;
    v_dt *col_idx;
    float *values;
} CSRMatrix_f;

typedef struct {
    int num_nodes;
    int num_edges;
    v_dt *row_ptr;
    v_dt *col_idx;
    float *values;
} CSRMatrix;

typedef struct {
    int node;
    int in_degree;
    float rank;
} NodeData;


CSRMatrix_f A;

void InitVprop(hls::stream<float>& init_vprop){
    int start_iter;
    Init1: for(int tile =0 ; tile < NUM_TILES ; tile ++){//start one tile
//#pragma HLS UNROLL factor = 16
        int tile_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
        Init2: for(int i = 0 ; i < TILE_SIZE ; i ++){
#pragma HLS pipeline II=1
            if ( i < tile_size ) {
                //out2[tile*TILE_SIZE + i] = base_score;
                init_vprop.write(base_score);
            }
        }
    }  
}

void PrefetchGraph(hls::stream<float>& prefetch_data, hls::stream<row_dt>& prefetch_row_ptr, hls::stream<float>& prefetch_out_deg,
                   const v_dt *in2, const v_dt_f *in3, float *out1) {
    int start_iter;
    float out_degree; 
    float src_pagerank;
    float out_degree_buffer[2];
#pragma HLS array_partition variable=out_degree_buffer complete
    int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
    row_dt row_stream;
    int start_row0 = in2[0].data[0];
    int start_row1 = in2[0].data[1];
    int start_out0 = in3[0].data[0];
    int start_out1 = in3[0].data[1];


    Prefetch1: for(int tile = 0; tile < NUM_TILES; tile++) { // start one tile
//#pragma HLS UNROLL factor = 16
        Prefetch2: for(int src = 0; src < NUM_NODES; src++) { // start one tile line
#pragma HLS LOOP_FLATTEN
#pragma HLS pipeline II=1
            // row_ptr
            if(tile ==0 && src ==0 ){
                row_ptr_buffer[0] = start_row0;
                row_ptr_buffer[1] = start_row1;
            } else {
                row_ptr_buffer[0] = row_ptr_buffer[1];
                row_ptr_buffer[1] = in2[(src + tile * NUM_NODES + 1) >> 4].data[(src + tile * NUM_NODES + 1) % 16];
            }
            row_stream.data[0] = row_ptr_buffer[0] ;
            row_stream.data[1] = row_ptr_buffer[1] ;
            prefetch_row_ptr.write(row_stream);

            // src
            src_pagerank = out1[src];
            prefetch_data.write(src_pagerank);

            // out_deg
            if(tile ==0 && src ==0 ){
                out_degree_buffer[0] = start_out0;
                out_degree_buffer[1] = start_out1;
            }else{
                out_degree_buffer[0] = out_degree_buffer[1];
                out_degree_buffer[1] = in3[(src + 1) >> 4].data[(src + 1) % 16];
            }
            out_degree = out_degree_buffer[1] - out_degree_buffer[0];
            prefetch_out_deg.write(out_degree);
        }
    }
}

void ProcessingElement(hls::stream<float>& init_vprop, hls::stream<float>& finish_vprop,
                       hls::stream<float>& prefetch_data, hls::stream<row_dt>& prefetch_row_ptr, hls::stream<float>& prefetch_out_deg,
                       const v_dt *in1, float *out2){
    float score_buffer[TILE_SIZE];
#pragma HLS array_partition variable=score_buffer complete
    float src_pagerank;
    float out_degree;
    row_dt row_stream;
    int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
    int delta_row_ptr;
    int col_idx;
    int buffer_start;
    int start_iter;
    float attr;

    Process1: for(int tile = 0 ; tile < NUM_TILES ; tile ++){//start one tile
//#pragma HLS UNROLL factor = 16
        int tile_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
        Process2: for(int i = 0 ; i < TILE_SIZE ; i ++){//prefetch one tile data
#pragma HLS pipeline II=1
            if(i < tile_size ){
                score_buffer[i] = init_vprop.read();
            }
        }
        Process3: for(int src = 0 ; src < NUM_NODES ; src ++){//calculate one tile
            //read
            src_pagerank = prefetch_data.read();
            out_degree = prefetch_out_deg.read();
            row_stream = prefetch_row_ptr.read();
            row_ptr_buffer[0] = row_stream.data[0];
            row_ptr_buffer[1] = row_stream.data[1];
            delta_row_ptr = row_ptr_buffer[1] - row_ptr_buffer[0];
            buffer_start = row_ptr_buffer[0];
            if( delta_row_ptr > static_cast<float>(0)){
                attr = ALPHA * src_pagerank / out_degree;
            }

            Process4: for(int i = 0 ; i < delta_row_ptr ; i ++){//calculate one tile line
#pragma HLS pipeline II=1
                col_idx = in1[(buffer_start + i)/16].data[(buffer_start + i)%16];
                score_buffer[col_idx - TILE_SIZE*tile] += attr;
            }
        }
        Process5: for(int i = 0 ; i < TILE_SIZE ; i ++){//prefetch one tile data
            if(i < tile_size ){
#pragma HLS pipeline II=1
                out2[TILE_SIZE*tile + i] = score_buffer[i];
                finish_vprop.write(score_buffer[i]);
            }
        }
    }
    
}


float IterationCheck(hls::stream<float>& finish_vprop,
                        float *out1){
    float diff;
    diff = 0;
    float before_vertex;
    float after_vertex;
    Iter1: for(int tile = 0; tile < NUM_TILES; tile ++){//start one tile
//#pragma HLS UNROLL factor = 16
        int tile_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
        Iter2: for(int i = 0 ; i < TILE_SIZE ; i ++){//get data
#pragma HLS pipeline II=1
            if(i < tile_size ){
                before_vertex = out1[tile* TILE_SIZE +i];
                after_vertex = finish_vprop.read();
                diff += ( (after_vertex-before_vertex) * (after_vertex-before_vertex));
            }
        }
    }
    return diff;
}

extern "C"{
void pagerank(const v_dt *in1,  // Read-Only Vector 1 from hbm -> col index
              const v_dt *in2,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
              const v_dt_f *in3,// Read-Only Vector 2 from hbm -> row ptr
              float *out1,      // Output Result to hbm -> pagerank score before
              float *out2       // Output Result to hbm -> pagerank score next
              ) {

#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem0 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem1 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = in3 offset = slave bundle = gmem2 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = out1 offset = slave bundle = gmem3 
#pragma HLS INTERFACE m_axi port = out2 offset = slave bundle = gmem4 

#pragma HLS INTERFACE s_axilite port=in1 bundle=control
#pragma HLS INTERFACE s_axilite port=in2 bundle=control
#pragma HLS INTERFACE s_axilite port=in3 bundle=control
#pragma HLS INTERFACE s_axilite port=out1 bundle=control
#pragma HLS INTERFACE s_axilite port=out2 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

hls::stream<float,1400> finish_vprop; 
hls::stream<float,1400> init_vprop; 

hls::stream<float,1400> prefetch_data;
hls::stream<float,1400> prefetch_out_deg;
hls::stream<row_dt,1400> prefetch_row_ptr; 

float diff = 100;

main1: for(int iter = 0; iter < MAX_ITER; iter ++){
    if ( iter%2 == 0 ){
        InitVprop(init_vprop);
        PrefetchGraph( prefetch_data, prefetch_row_ptr, prefetch_out_deg, in2, in3, out1);
        ProcessingElement( init_vprop, finish_vprop, prefetch_data, prefetch_row_ptr, prefetch_out_deg, in1, out2);
        diff = IterationCheck( finish_vprop, out1);
    }else{
        InitVprop(init_vprop);
        PrefetchGraph( prefetch_data, prefetch_row_ptr, prefetch_out_deg, in2, in3, out2);
        ProcessingElement( init_vprop, finish_vprop, prefetch_data, prefetch_row_ptr, prefetch_out_deg, in1, out1);
        diff = IterationCheck( finish_vprop, out2);
    }
    
    if(diff < EPSILON){
        printf("FPGA iter = %d\n", iter);
        break;
        
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

void generate_random_graph(CSRMatrix_f *A, int num_nodes, int num_edges, int *in_degree) {
    A->num_nodes = num_nodes;
    A->num_edges = num_edges;
    A->row_ptr = (v_dt_f *)malloc(((num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt_f));
    A->col_idx = (v_dt *)malloc((num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    A->values = (float *)malloc(num_edges * sizeof(float));

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

void tile_CSRMatrix_func(const CSRMatrix_f *A, CSRMatrix *T, int tile_size) {
    int num_nodes = A->num_nodes;
    int num_tiles = (num_nodes + tile_size - 1) / tile_size;

    T->num_nodes = num_nodes;
    T->num_edges = A->num_edges;
    T->row_ptr = (v_dt *)malloc(((num_nodes *num_tiles  + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt) * num_tiles);
    T->col_idx = (v_dt *)malloc((A->num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    T->values = (float *)malloc(A->num_edges * sizeof(float));

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
        tile_CSRMatrix[i].values = (float *)malloc((tile_CSRMatrix[i].num_edges) * sizeof(float));
        tile_CSRMatrix[i].row_ptr[0].data[0] = 0; // Initialize the first element of row_ptr
    }
    int col_idx_ptr[num_tiles] = {0};
    for (int i = 0; i < A->num_edges; i++) {
        int index = A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE] / tile_size;
        tile_CSRMatrix[index].col_idx[col_idx_ptr[index] / VDATA_SIZE].data[col_idx_ptr[index] % VDATA_SIZE] = A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE];
        col_idx_ptr[index]++;
    }

    //printf("col_idx===========================\n");
    //for (int index = 0; index < num_tiles; index++) {
    //    for (int i = 0; i < tile_CSRMatrix[index].num_edges; i++) {
    //        printf("%d ", tile_CSRMatrix[index].col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE]);
    //    }
    //    printf("\n");
    //}
    //printf("\n");

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

    //printf("row_ptr\n");
    //for (int i = 0; i < num_nodes *num_tiles  + 1; i++) {
    //    printf("%d ", T->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]);
    //}
    //printf("\n");
    //
    //printf("col_idx\n");
    //for (int i = 0; i < NUM_EDGES; i++) {
    //    printf("%d ", T->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE]);
    //}
    //printf("\n");

    T->row_ptr[0].data[0] = 0;
    // Free allocated memory for tile_CSRMatrix
    for (int t = 0; t < num_tiles; t++) {
        free(tile_CSRMatrix[t].row_ptr);
        free(tile_CSRMatrix[t].col_idx);
        free(tile_CSRMatrix[t].values);
    }
    free(tile_CSRMatrix);
}

//pageRank code
void pageRank_CSR(const CSRMatrix_f *A, float *r) {
    int num_nodes = A->num_nodes;
    float *r_new = (float *)malloc(num_nodes * sizeof(float));
    float a =1;
    float base_score = (a - ALPHA) / num_nodes;

    for (int i = 0; i < num_nodes; i++) {
        r[i] = 1.0 / num_nodes;
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
//printf("iter = %d\n", iter);
        for (int i = 0; i < num_nodes; i++) {
            r_new[i] = base_score;
        }

        for (int u = 0; u < num_nodes; u++) {
            int buffer_start = A->row_ptr[u / VDATA_SIZE].data[u % VDATA_SIZE];
            int buffer_end = A->row_ptr[(u + 1) / VDATA_SIZE].data[(u + 1) % VDATA_SIZE];
            int buffer_size = buffer_end - buffer_start;
            int out_degree_u = buffer_end - buffer_start;
            int col_idx_buffer[10];

            for (int b = 0; b < buffer_size; b += 10) {
                int chunk_size = (b + 10 > buffer_size) ? buffer_size - b : 10;
                for (int k = 0; k < chunk_size; k++) {
                    col_idx_buffer[k] = A->col_idx[(buffer_start + b + k) / VDATA_SIZE].data[(buffer_start + b + k) % VDATA_SIZE];
                }

                for (int k = 0; k < chunk_size; k++) {
                    int v = col_idx_buffer[k];
                    r_new[v] += ALPHA * r[u] / out_degree_u;
                }
            }
        }

        float diff = 0.0;
        for (int i = 0; i < num_nodes; i++) {
            diff += (r_new[i] - r[i])*(r_new[i] - r[i]) ;
        }
        if (diff < EPSILON) {
            printf("iter = %d\n", iter);
            break;
        }

        memcpy(r, r_new, num_nodes * sizeof(float));
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
    int *in_degree = (int *)calloc(NUM_NODES, sizeof(int));
    generate_random_graph(&A, NUM_NODES, NUM_EDGES, in_degree);

    float *r = (float *)malloc(A.num_nodes * sizeof(float));
    
    // PageRank 계산
    pageRank_CSR(&A, r);
    
    float score1[NUM_NODES];
    float score2[NUM_NODES];

    for(int i = 0; i < NUM_NODES; i++){
        score1[i] = 1.0 / NUM_NODES;
        score2[i] = 0;
    }
    CSRMatrix T;
    tile_CSRMatrix_func(&A, &T, TILE_SIZE);
    printf("Start FPGA\n");
    pagerank(T.col_idx, T.row_ptr, A.row_ptr, score1, score2);
    printf("Finish FPGA\n");
    NodeData *nodes = (NodeData *)malloc(A.num_nodes * sizeof(NodeData));
    for (int i = 0; i < A.num_nodes; i++) {
        nodes[i].node = i;
        nodes[i].in_degree = in_degree[i];
        nodes[i].rank = r[i];
    }

    float sum = 0.0;
    for (int i = 0; i < NUM_NODES; i++) {
        sum += (score2[i] - nodes[i].rank)*(score2[i] - nodes[i].rank);
    }
    std::cout << "Diff: " << static_cast<float>(sum) << std::endl;
    
    if(sum < EPSILON*100){
        printf("TEST PASS\n");    
    }

    for (int i = 0; i < A.num_nodes; i++) {
        std::cout << "Value1: " << static_cast<float>(nodes[i].rank) << std::endl;
        std::cout << "Value2: " << static_cast<float>(score2[i]) << std::endl;
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

