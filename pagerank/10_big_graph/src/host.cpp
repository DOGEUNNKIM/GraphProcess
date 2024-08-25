/**********
Copyright (c) 2020, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
#include <cstring>
#include <iostream>
#include <chrono>
#include <math.h>
#include <omp.h>
#include <vector>
#include <time.h>
#include <queue>
#include <fstream>
#include <stdint.h>
// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"


#define VDATA_SIZE 8
#define ALPHA 0.85f
#define MAX_ITER 100
#define EPSILON 1e-6f
#define TILE_SIZE 524288  // 4MB
#define UNROLL_SIZE 8
//facebook
//#define NUM_NODES_ 4039
//#define NUM_EDGES 88234
//pocker
//#define NUM_NODES_ 1632803
//#define NUM_EDGES 30622564
//LiveJournal1
#define NUM_NODES_ 4847571
#define NUM_EDGES 68993773

#define NUM_TILES (NUM_NODES_ + TILE_SIZE - 1) / TILE_SIZE

using namespace std;

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
    float rank;
} NodeData;

typedef struct {
    int64_t src;
    int64_t dst;
} Edge;

// Function to load CSRMatrix from a binary file
void load_csr_matrix(CSRMatrix *A, const char* filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    infile.read(reinterpret_cast<char*>(&A->num_nodes), sizeof(int64_t  ));
    infile.read(reinterpret_cast<char*>(&A->num_edges), sizeof(int64_t  ));

    A->row_ptr = (v_dt*)malloc(((A->num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    A->col_idx = (v_dt*)malloc((A->num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    A->values = (int64_t  *)malloc(A->num_edges * sizeof(int64_t  ));

    infile.read(reinterpret_cast<char*>(A->row_ptr), ((A->num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    infile.read(reinterpret_cast<char*>(A->col_idx), (A->num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    infile.read(reinterpret_cast<char*>(A->values), A->num_edges * sizeof(int64_t  ));

    infile.close();
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
extern "C"{
void pagerank(const v_dt *in1,  // Read-Only Vector 1 from hbm -> col index
              const v_dt *in2,  // Read-Only Vector 2 from hbm -> row ptr
              const v_dt *in3,// Read-Only Vector 2 from hbm -> row ptr 
              float *out1,      // Output Result to hbm -> pagerank score
              float *out2,      // Output Result to hbm -> pagerank score
              int64_t NUM_NODES
              ) {


  // set constant
  float base_score = (1.0 - ALPHA) / NUM_NODES;
  
  int64_t row_ptr_buffer[2];
  #pragma HLS array_partition variable=row_ptr_buffer complete
  float out_degree_buffer[2];
  #pragma HLS array_partition variable=out_degree_buffer complete
  int64_t col_idx_buffer[UNROLL_SIZE];
  #pragma HLS array_partition variable=col_idx_buffer factor=VDATA_SIZE cyclic
  float score_buffer[TILE_SIZE];
  #pragma HLS array_partition variable=score_buffer factor=VDATA_SIZE cyclic
  
  
  int64_t out_degree_u;
  int64_t buffer_start;
  int64_t size_;
  float src_pagerank;

  for (int64_t iter = 0; iter < MAX_ITER; iter++) {
  if( (iter %2) == 0){
    for (int64_t i = 0; i < NUM_NODES; i ++) {
      #pragma HLS pipeline II=1
      out2[i] = base_score;
    }
    //push score_new
    for (int64_t tile = 0; tile < NUM_TILES; tile ++) {
      int64_t score_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
      for(int64_t k = 0; k < TILE_SIZE; k++){
        #pragma HLS pipeline II=1
        if ( k < score_size ) {
          score_buffer[k] = out2[tile*TILE_SIZE + k];
        }
      }
      for (int64_t u = 0; u < NUM_NODES; u++) {
        //prefatch row_ptr
        row_ptr_buffer[0]= in2[(u+tile*NUM_NODES)/VDATA_SIZE].data[(u+tile*NUM_NODES)%VDATA_SIZE];
        row_ptr_buffer[1]= in2[(u+tile*NUM_NODES+1)/VDATA_SIZE].data[(u+tile*NUM_NODES+1)%VDATA_SIZE];
        size_ = row_ptr_buffer[1] - row_ptr_buffer[0];
        buffer_start = row_ptr_buffer[0];
        src_pagerank =  out1[u];
        
        out_degree_buffer[0]= in3[u/VDATA_SIZE].data[u%VDATA_SIZE];
        out_degree_buffer[1]= in3[(u+1)/VDATA_SIZE].data[(u+1)%VDATA_SIZE]; 
        out_degree_u = out_degree_buffer[1] - out_degree_buffer[0];
        //push
        if(out_degree_u != 0){
        for (int64_t b = 0; b < size_; b += UNROLL_SIZE) {
          //prefatch col_idx_buffer
          int64_t chunk_size = (b + UNROLL_SIZE > size_) ? size_ - b : UNROLL_SIZE;
          for (int64_t k = 0; k < UNROLL_SIZE; k++) {
            #pragma HLS unroll
            if( k < chunk_size) {
                col_idx_buffer[k]= in1[(b + buffer_start + k)/VDATA_SIZE].data[(b + buffer_start + k)%VDATA_SIZE];
            }
          }
          //SIMD parallel compute
          for (int64_t l = 0; l < UNROLL_SIZE; l++) {
            #pragma HLS unroll 
            if( l < chunk_size) {
              int64_t idx = col_idx_buffer[l] - TILE_SIZE*tile;
              score_buffer[idx] = score_buffer[idx] + (ALPHA * src_pagerank / out_degree_u );
            }
          }
        }
        }//push_one_vertex
      }//push_one_tile
      for(int64_t k = 0; k < TILE_SIZE; k++){
        #pragma HLS pipeline II=1
        if ( k < score_size ) {
          out2[tile*TILE_SIZE + k] = score_buffer[k];
        }
      }
    }//push_all_tile
  }else{
    for (int64_t i = 0; i < NUM_NODES; i ++) {
      #pragma HLS pipeline II=1
      out1[i] = base_score;
    }
    //push score_new
    for (int64_t tile = 0; tile < NUM_TILES; tile ++) {
      int64_t score_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
      for(int64_t k = 0; k < TILE_SIZE; k++){
        #pragma HLS pipeline II=1
        if ( k < score_size ) {
          score_buffer[k] = out1[tile*TILE_SIZE + k];
        }
      }
      for (int64_t u = 0; u < NUM_NODES; u++) {
        //prefatch row_ptr
        row_ptr_buffer[0]= in2[(u+tile*NUM_NODES)/VDATA_SIZE].data[(u+tile*NUM_NODES)%VDATA_SIZE];
        row_ptr_buffer[1]= in2[(u+tile*NUM_NODES+1)/VDATA_SIZE].data[(u+tile*NUM_NODES+1)%VDATA_SIZE];
        size_ = row_ptr_buffer[1] - row_ptr_buffer[0];
        buffer_start = row_ptr_buffer[0];
        src_pagerank =  out2[u];
        
        out_degree_buffer[0]= in3[u/VDATA_SIZE].data[u%VDATA_SIZE];
        out_degree_buffer[1]= in3[(u+1)/VDATA_SIZE].data[(u+1)%VDATA_SIZE]; 
        out_degree_u = out_degree_buffer[1] - out_degree_buffer[0];
        if(out_degree_u != 0){
        //push
        for (int64_t b = 0; b < size_; b += UNROLL_SIZE) {
          //prefatch col_idx_buffer
          int64_t chunk_size = (b + UNROLL_SIZE > size_) ? size_ - b : UNROLL_SIZE;
          for (int64_t k = 0; k < UNROLL_SIZE; k++) {
            #pragma HLS unroll
            if( k < chunk_size) {
                col_idx_buffer[k]= in1[(b + buffer_start + k)/VDATA_SIZE].data[(b + buffer_start + k)%VDATA_SIZE];
            }
          }
          //SIMD parallel compute
          for (int64_t l = 0; l < UNROLL_SIZE; l++) {
            #pragma HLS unroll 
            if( l < chunk_size) {
              int64_t idx = col_idx_buffer[l] - TILE_SIZE*tile;
              score_buffer[idx] = score_buffer[idx] + (ALPHA * src_pagerank / out_degree_u );
            }
          }
        }
        }//push_one_vertex
      }//push_one_tile
      for(int64_t k = 0; k < TILE_SIZE; k++){
        #pragma HLS pipeline II=1
        if ( k < score_size ) {
          out1[tile*TILE_SIZE + k] = score_buffer[k];
        }
      }
    }//push_all_tile
  }//iter change
  
  float diff = 0.0f;
  
  for (int64_t i = 0; i < NUM_NODES; i += 1) {
    #pragma HLS pipeline II=1
    float temp = ((out2[i] - out1[i]) > 0 )? 
                  (out2[i] - out1[i]):
                  (out1[i] - out2[i]);
    diff += temp;
  }
  
  //if converge then break 
  if (diff < EPSILON) {
      break;
  }
  //printf("FPGA diff = %f\n", diff);
  //printf("FPGA iter = %ld\n", iter);

  // update score
  if( (iter %2) == 0){
    // update score
    for (int64_t i = 0; i < NUM_NODES; i ++ ) {
      #pragma HLS pipeline II=1
      out1[i] = out2[i];
    }
  }else{
    for (int64_t i = 0; i < NUM_NODES; i ++ ) {
      #pragma HLS pipeline II=1
      out2[i] = out1[i];
    }
  }
  }


}
}

void pageRank_CSR(const CSRMatrix *A, float *r) {
    int64_t num_nodes = A->num_nodes;
    float *r_new = (float *)malloc(num_nodes * sizeof(float));
    float base_score = (1 - ALPHA) / num_nodes;

    for (int64_t i = 0; i < num_nodes; i++) {
        r[i] = 1.0 / num_nodes;
    }
    

    for (int64_t iter = 0; iter < MAX_ITER; iter++) {

        for (int64_t i = 0; i < num_nodes; i++) {
            r_new[i] = base_score;
        }

        for (int64_t u = 0; u < num_nodes; u++) {
            int64_t buffer_start = A->row_ptr[u / VDATA_SIZE].data[u % VDATA_SIZE];
            int64_t buffer_end = A->row_ptr[(u + 1) / VDATA_SIZE].data[(u + 1) % VDATA_SIZE];
            int64_t buffer_size = buffer_end - buffer_start;
            int64_t out_degree_u = buffer_end - buffer_start;
            
            int64_t col_idx_buffer[10];

            for (int64_t b = 0; b < buffer_size; b += 10) {
                int64_t chunk_size = (b + 10 > buffer_size) ? buffer_size - b : 10;
                for (int64_t k = 0; k < chunk_size; k++) {
                    col_idx_buffer[k] = A->col_idx[(buffer_start + b + k) / VDATA_SIZE].data[(buffer_start + b + k) % VDATA_SIZE];
                }

                //#pragma omp simd
                for (int64_t k = 0; k < chunk_size; k++) {
                    int64_t v = col_idx_buffer[k];
                    //#pragma omp atomic
                    if(out_degree_u != 0){
                        r_new[v] += ALPHA * r[u] / out_degree_u;
                    }
                }
            }
        }

        float diff = 0.0f;
        for (int64_t i = 0; i < num_nodes; i++) {
            float temp = ((r_new[i] - r[i]) > 0 )? 
                        (r_new[i] - r[i]):
                        (r[i] - r_new[i]);
            diff += temp ;
        }
        
        //printf("CPU iter = %ld\n", iter);
        //printf("CPU diff = %f\n", diff);
        
        if (diff < EPSILON) {
            break;
        }
        memcpy(r, r_new, (int64_t)num_nodes * sizeof(float));
    }

    free(r_new);
}


int main(int argc, char **argv) {
    
  std::string xclbin_file_name = argv[1];
  CSRMatrix A;
  printf("START LOAD\n");
  //load_csr_matrix(&A, "/home/kdg6245/graph/dataset/csr_matrix_facebook_int64.bin");
  //load_csr_matrix(&A, "/home/kdg6245/graph/dataset/csr_matrix_pokec.bin");
  load_csr_matrix(&A, "/home/kdg6245/graph/dataset/csr_matrix_LiveJournal1.bin");
  printf("FINISH LOAD\n");
  float *cpu_result = (float *)malloc(A.num_nodes * sizeof(float));

  auto cpu_begin = std::chrono::high_resolution_clock::now();
  // PageRank 계산
  printf("START CPU\n");
  pageRank_CSR(&A, cpu_result);
  printf("FINISH CPU\n");
  auto cpu_end = std::chrono::high_resolution_clock::now();

  NodeData *nodes = (NodeData *)malloc(A.num_nodes * sizeof(NodeData));
  float sum = 0.0; 
  //for (int64_t i = 0; i < 10; i++) {
  //  cout << cpu_result[i] << endl;
  //}
  for (int64_t i = 0; i < A.num_nodes; i++) {
      nodes[i].node = i;
      nodes[i].rank = cpu_result[i];
      sum += nodes[i].rank;
  }
  printf("sum %f\n", sum);
  
  printf("START PREPROCESSING\n");
  CSRMatrix T;
  tile_CSRMatrix_func(&A, &T, TILE_SIZE);
  printf("FINISH PREPROCESSING\n");

  float *FPGA_result_1 = (float *)malloc((int64_t)A.num_nodes * sizeof(float));
  for (int i = 0; i < A.num_nodes; ++i) {
    FPGA_result_1[i] = 1/A.num_nodes;
  }
  float *FPGA_result_2 = (float *)malloc((int64_t)A.num_nodes * sizeof(float));
  for (int i = 0; i < A.num_nodes; ++i) {
    FPGA_result_2[i] = 0.0f;
  }
  
  printf("START FPGA SIM\n");
  auto fpga_sim_begin = std::chrono::high_resolution_clock::now();
  pagerank(T.col_idx,T.row_ptr, A.row_ptr,FPGA_result_1,FPGA_result_2, NUM_NODES_);
  auto fpga_sim_end = std::chrono::high_resolution_clock::now();
  printf("FINISH FPGA SIM\n");

  //The host code assumes there is a single device and opening a device by
  //device index 0. If there are multiple devices then this device index needed
  //to be adjusted. The user can get the list of the devices and device indices
  //by xbtuil examine command.
  unsigned int device_index = 0;
  std::cout << "Open the device " << device_index << std::endl;
  auto device = xrt::device(device_index);
  
  std::cout << "Load the xclbin " << xclbin_file_name << std::endl;
  auto uuid = device.load_xclbin(xclbin_file_name);

  size_t row_ptr_bytes = sizeof(int64_t) * (A.num_nodes + 1);
  size_t row_ptr_process_bytes = sizeof(int64_t) * (A.num_nodes*(int64_t)NUM_TILES  + 1);
  size_t col_idx_process_bytes = sizeof(int64_t) * A.num_edges;
  size_t score_bytes = sizeof(float) * A.num_nodes;

  auto krnl = xrt::kernel(device, uuid, "pagerank");

  std::cout << "Allocate Buffer in Global Memory\n";
  auto bo0 = xrt::bo(device, col_idx_process_bytes, krnl.group_id(0));
  auto bo1 = xrt::bo(device, row_ptr_process_bytes, krnl.group_id(1));
  auto bo2 = xrt::bo(device, row_ptr_bytes, krnl.group_id(2));
  auto bo_score_1 = xrt::bo(device, score_bytes, krnl.group_id(3));
  auto bo_score_2 = xrt::bo(device, score_bytes, krnl.group_id(4));
  // Map the contents of the buffer object into host memory
  auto bo0_map = bo0.map<int64_t *>();
  auto bo1_map = bo1.map<int64_t *>();
  auto bo2_map = bo2.map<int64_t *>();
  auto bo_score_map_1 = bo_score_1.map<float *>();
  auto bo_score_map_2 = bo_score_2.map<float *>();
  std::fill(bo0_map, bo0_map + (int64_t)A.num_edges, 0);
  std::fill(bo1_map, bo1_map + (int64_t)A.num_nodes*(int64_t)NUM_TILES + 1, 0);
  std::fill(bo2_map, bo2_map + (int64_t)A.num_nodes + 1, 0.0f);

  std::fill(bo_score_map_1, bo_score_map_1 + (int64_t)A.num_nodes, 1.0/A.num_nodes);
  std::fill(bo_score_map_2, bo_score_map_2 + (int64_t)A.num_nodes, 0.0);

  // Create the test data
  vector<float> bufReference(A.num_nodes);

  for (int64_t i = 0; i < (int64_t)A.num_edges; ++i) {
    bo0_map[i] = T.col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  for (int64_t i = 0; i < (int64_t)A.num_nodes*NUM_TILES + 1; ++i) {
    bo1_map[i] = T.row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  for (int64_t i = 0; i < (int64_t)A.num_nodes; ++i) {
    bo2_map[i] = A.row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  for (int64_t i = 0; i < (int64_t)A.num_nodes; ++i) {
    bo_score_map_1[i] = 1/A.num_nodes;
  }
  for (int64_t i = 0; i < (int64_t)A.num_nodes; ++i) {
    bo_score_map_2[i] = 0;
  }
  for (int64_t i = 0; i < (int64_t)A.num_nodes; ++i) {
    bufReference[i] = nodes[i].rank;
  }

  auto fpga_begin = std::chrono::high_resolution_clock::now();

  // Synchronize buffer content with device side
  std::cout << "synchronize input buffer data to device global memory\n";


  //////////////////////////////////////////////////////////////////////////////
  auto host_to_fpga_start = std::chrono::high_resolution_clock::now();


  bo0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_score_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_score_2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  auto host_to_fpga_end = std::chrono::high_resolution_clock::now();
  /////////////////////////////////////////////////////////////////////////////


  std::cout << "synchronize input buffer data to device global memory finish\n";
  auto fpga_cal_begin = std::chrono::high_resolution_clock::now();
  printf("START FPGA HW\n");
  auto run = krnl(bo0, bo1, bo2,bo_score_1, bo_score_2, NUM_NODES_);
  run.wait();
  printf("FINISH FPGA HW\n");
  auto fpga_cal_end = std::chrono::high_resolution_clock::now();

  // Get the output;
  std::cout << "Get the output data from the device" << std::endl;
  
  //////////////////////////////////////////////////////////////////////////////
  auto fpga_to_host_start = std::chrono::high_resolution_clock::now();

  bo_score_1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_score_2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  auto fpga_to_host_end = std::chrono::high_resolution_clock::now();
  /////////////////////////////////////////////////////////////////////////////


  auto fpga_end = std::chrono::high_resolution_clock::now();

  
  std::chrono::duration<double> fpga_cal_duration = fpga_cal_end - fpga_cal_begin;
  std::cout << "FPGA IN Time:                 " << fpga_cal_duration.count() << " s" << std::endl;

  std::chrono::duration<double> host_to_fpga_duration = host_to_fpga_end -host_to_fpga_start;
  std::cout << "host to fpga PCIe Time:       " << host_to_fpga_duration.count() << " s" << std::endl;

  std::chrono::duration<double> fpga_to_host_duration = fpga_to_host_end -fpga_to_host_start;
  std::cout << "fpga to host PCIe Time:       " << fpga_to_host_duration.count() << " s" << std::endl;
  
  std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;
  std::cout << "FPGA Time:                    " << fpga_duration.count() << " s" << std::endl;
  
  std::chrono::duration<double> fpga_duration_sim = fpga_sim_end - fpga_sim_begin;
  std::cout << "FPGA simulation Time:                    " << fpga_duration_sim.count() << " s" << std::endl;

  std::chrono::duration<double> cpu_duration = cpu_end - cpu_begin;
  std::cout << "CPU Time:                     " << cpu_duration.count() << " s" << std::endl;

  std::cout << "FPGA / CPU Speedup:                 " << cpu_duration.count() / fpga_duration.count() << " x" << std::endl;
  std::cout << "FPGA / SIM Speedup:                 " << cpu_duration.count() / fpga_duration.count() << " x" << std::endl;
  
  //// 메모리 해제
  //Check result
  for (int i = 0; i < 20; i++) {
      std::cout << "CPU  result     " << static_cast<float>(bufReference[i]) << std::endl;
      std::cout << "FPGA simulation " << static_cast<float>(FPGA_result_2[i]) << std::endl;
      std::cout << "FPGA hardware   " << static_cast<float>(bo_score_map_1[i]) << std::endl;
  }
  free(A.row_ptr);
  free(A.col_idx);
  free(A.values);
  free(cpu_result);
  free(nodes);
  free(T.row_ptr);
  free(T.col_idx);
  free(FPGA_result_1);
  free(FPGA_result_2);

  sum = 0.0;
  for (int64_t i = 0; i < A.num_nodes; i++) {
      sum += bo_score_map_1[i];
  }
  printf("sum %f\n", sum);


  // Validate our results
  
  auto compare_begin = std::chrono::high_resolution_clock::now();

  float diff = 0;
  for (int64_t i = 0; i < A.num_nodes; i++) {
      diff += abs(bo_score_map_1[i] - bufReference[i]) ;
  }
  printf("diff %f\n", diff);
  
  if (diff > EPSILON*100)
    throw std::runtime_error("Score does not match reference");
  
  std::cout << "TEST PASSED\n";
  
  auto compare_end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> compare_duration = compare_end - compare_begin;
  std::cout << "Compare Time:                 " << compare_duration.count() << " s" << std::endl;

  return 0;
}
