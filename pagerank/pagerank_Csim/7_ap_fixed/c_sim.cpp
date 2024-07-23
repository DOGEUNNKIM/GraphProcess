#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include "ap_fixed.h"


#define VDATA_SIZE 16
//#define ALPHA 0.85
#define MAX_ITER 1000
//#define EPSILON 1e-6
#define NUM_NODES 100
#define NUM_EDGES 1000
#define BUFFER_SIZE 16
#define TILE_SIZE 16
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE

//#define my_float float

typedef ap_fixed<32,12> my_float;


my_float ALPHA = 0.85;
my_float EPSILON = 1e-6;

typedef struct v_datatype { int data[VDATA_SIZE]; } v_dt;
typedef struct v_f_datatype { my_float data[VDATA_SIZE]; } v_dt_f;

typedef struct {
    int num_nodes;
    int num_edges;
    v_dt_f *row_ptr;
    v_dt *col_idx;
    my_float *values;
} CSRMatrix_f;

typedef struct {
    int num_nodes;
    int num_edges;
    v_dt *row_ptr;
    v_dt *col_idx;
    my_float *values;
} CSRMatrix;

typedef struct {
    int node;
    int in_degree;
    my_float rank;
} NodeData;


CSRMatrix_f A;

// use 1 dsp
extern "C"{
void pagerank(const v_dt *in1,  // Read-Only Vector 1 from hbm -> col index
              const v_dt *in2,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
              const v_dt_f *in3,// Read-Only Vector 2 from hbm -> row ptr 
              my_float *out1,      // Output Result to hbm -> pagerank score before
              my_float *out2       // Output Result to hbm -> pagerank score next
              ) {

#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem0 max_widen_bitwidth=512 
#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port = in3 offset = slave bundle = gmem2 max_widen_bitwidth=512
// random read -> cannot burst read
#pragma HLS INTERFACE m_axi port = out1 offset = slave bundle = gmem3 
#pragma HLS INTERFACE m_axi port = out2 offset = slave bundle = gmem4 

// set constant
my_float base_score;
#pragma HLS bind_op variable=base_score op=fsub impl=fabric
#pragma HLS bind_op variable=base_score op=fdiv impl=fabric
my_float a =1;
base_score = (a - ALPHA) / NUM_NODES;

int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
my_float out_degree_buffer[2];
#pragma HLS array_partition variable=out_degree_buffer complete
my_float score_buffer[TILE_SIZE];
#pragma HLS array_partition variable=score_buffer complete

int col_idx_buffer[BUFFER_SIZE];
#pragma HLS array_partition variable=col_idx_buffer factor=16 cyclic


for (int iter = 0; iter < MAX_ITER; iter++) {
  for (int i = 0; i < NUM_NODES; i ++) {
#pragma HLS pipeline II=1
    out2[i] = base_score;
  }
  //cal score_new
  for (int tile = 0; tile < NUM_TILES; tile ++) {
//#pragma HLS pipeline rewind
    //prefatch score_buffer
    int score_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
    
    for(int k = 0; k < TILE_SIZE; k++){
#pragma HLS pipeline II=1
      if ( k < score_size ) {
        score_buffer[k] = out2[tile*TILE_SIZE + k];
      }
    }

    for (int u = 0; u < NUM_NODES; u++) {
//#pragma HLS pipeline rewind
      //prefatch row_ptr
      row_ptr_buffer[0]= in2[(u+tile*NUM_NODES)/16].data[(u+tile*NUM_NODES)%16];
      row_ptr_buffer[1]= in2[(u+tile*NUM_NODES+1)/16].data[(u+tile*NUM_NODES+1)%16];
      int size_ = row_ptr_buffer[1] - row_ptr_buffer[0];
      
      out_degree_buffer[0]= in3[u/16].data[u%16];
      out_degree_buffer[1]= in3[(u+1)/16].data[(u+1)%16]; 
      my_float out_degree_u;  
#pragma HLS bind_op variable=out_degree_u op=fsub impl=fabric
      out_degree_u = out_degree_buffer[1] - out_degree_buffer[0];
      int buffer_start = row_ptr_buffer[0];
      //push
      for (int b = 0; b < NUM_NODES; b += BUFFER_SIZE) {
        if(b < size_){
#pragma HLS pipeline //rewind
          //prefatch col_idx_buffer
          int chunk_size = (b + BUFFER_SIZE > size_) ? size_ - b : BUFFER_SIZE;
  
          for (int k = 0; k < BUFFER_SIZE; k++) {
#pragma HLS unroll
            col_idx_buffer[k] = in1[(b + buffer_start + k)/16].data[(b + buffer_start + k)%16];
          }
          
          //SIMD parallel compute
          for (int l = 0; l < BUFFER_SIZE; l++) {
#pragma HLS unroll 
            if( l < chunk_size) {
            int idx = col_idx_buffer[l] - TILE_SIZE*tile;
            score_buffer[idx] = score_buffer[idx] + (ALPHA *   out1[u] / out_degree_u );
            }
          }
        }
      }
    }
    
    //return score_buffer
    for(int k = 0; k < TILE_SIZE; k++){
#pragma HLS pipeline II=1
      if ( k < score_size ) {
        out2[tile*TILE_SIZE + k] = score_buffer[k];
      }
    }
  }
  
  //check converge & update score
  
  my_float diff = 0;
  for (int i = 0; i < NUM_NODES; i += 1) {
#pragma HLS pipeline II=1
    diff += (out1[i] - out2[i]) * (out1[i] - out2[i]);
    out1[i] = out2[i];
  }


  if (diff < EPSILON ){
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
    A->values = (my_float *)malloc(num_edges * sizeof(my_float));

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
    T->values = (my_float *)malloc(A->num_edges * sizeof(my_float));

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
        tile_CSRMatrix[i].values = (my_float *)malloc((tile_CSRMatrix[i].num_edges) * sizeof(my_float));
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

    // Free allocated memory for tile_CSRMatrix
    for (int t = 0; t < num_tiles; t++) {
        free(tile_CSRMatrix[t].row_ptr);
        free(tile_CSRMatrix[t].col_idx);
        free(tile_CSRMatrix[t].values);
    }
    free(tile_CSRMatrix);
}


void pageRank_CSR(const CSRMatrix_f *A, my_float *r) {
    int num_nodes = A->num_nodes;
    my_float *r_new = (my_float *)malloc(num_nodes * sizeof(my_float));
    my_float a =1;
    my_float base_score = (a - ALPHA) / num_nodes;

    for (int i = 0; i < num_nodes; i++) {
        r[i] = 1.0 / num_nodes;
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
printf("iter = %d\n", iter);
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

        my_float diff = 0.0;
        for (int i = 0; i < num_nodes; i++) {
            diff += (r_new[i] - r[i])*(r_new[i] - r[i]) ;
        }
        if (diff < EPSILON) {
            break;
        }

        memcpy(r, r_new, num_nodes * sizeof(my_float));
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

    my_float *r = (my_float *)malloc(A.num_nodes * sizeof(my_float));
    
    // PageRank 계산
    pageRank_CSR(&A, r);
    
    my_float score1[NUM_NODES];
    my_float score2[NUM_NODES];

    for(int i = 0; i < NUM_NODES; i++){
        score1[i] = 1.0 / NUM_NODES;
        score2[i] = 0;
    }
    CSRMatrix T;
    tile_CSRMatrix_func(&A, &T, TILE_SIZE);
    
    pagerank(T.col_idx, T.row_ptr, A.row_ptr, score1, score2);


    NodeData *nodes = (NodeData *)malloc(A.num_nodes * sizeof(NodeData));
    for (int i = 0; i < A.num_nodes; i++) {
        nodes[i].node = i;
        nodes[i].in_degree = in_degree[i];
        nodes[i].rank = r[i];
    }

    my_float sum = 0.0;
    for (int i = 0; i < NUM_NODES; i++) {
        sum += (score2[i] - nodes[i].rank)*(score2[i] - nodes[i].rank);
    }
    std::cout << "Diff: " << static_cast<float>(sum) << std::endl;
    
    

    // PageRank 순으로 정렬하여 출력
    //qsort(nodes, A.num_nodes, sizeof(NodeData), compare_by_rank);
    //printf("Sorted by rank:\n");
    sum = 0.0;
    for (int i = 0; i < A.num_nodes; i++) {
        //float rank_as_int = std::stof(nodes[i].rank.to_string());
        //printf("Node %d: in-degree %d, rank %f\n", nodes[i].node, nodes[i].in_degree, rank_as_int);
        sum += nodes[i].rank;
        std::cout << "Value1: " << static_cast<float>(nodes[i].rank) << std::endl;
        std::cout << "Value2: " << static_cast<float>(score2[i]) << std::endl;
    }
    //float sum_as_int = std::stof(sum.to_string());
    //printf("sum %f\n", sum_as_int);


    // 메모리 해제
    free(A.row_ptr);
    free(A.col_idx);
    free(A.values);
    free(r);
    free(in_degree);
    free(nodes);

    return 0;
}

