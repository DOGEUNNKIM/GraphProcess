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

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

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
    float *values;
} CSRMatrix;

typedef struct {
    int node;
    int in_degree;
    float rank;
} NodeData;

using namespace std;

void generate_random_graph(CSRMatrix *A, int num_nodes, int num_edges, int *in_degree) {
    A->num_nodes = num_nodes;
    A->num_edges = num_edges;
    A->row_ptr = (int *)malloc((num_nodes + 1) * sizeof(int));
    A->col_idx = (int *)malloc(num_edges * sizeof(int));
    A->values = (float *)malloc(num_edges * sizeof(float));

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

void pageRank_CSR(const CSRMatrix *A, float *r) {
    int num_nodes = A->num_nodes;
    float *r_new = (float *)malloc(num_nodes * sizeof(float));
    float base_score = (1.0 - ALPHA) / num_nodes;

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

        float diff = 0.0;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < num_nodes; i++) {
            diff += fabs(r_new[i] - r[i]);
        }
        if (diff < EPSILON) {
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


int main(int argc, char **argv) {
    
  std::string xclbin_file_name = argv[1];
  CSRMatrix A;
  int *in_degree = (int *)calloc(NUM_NODES, sizeof(int));

  generate_random_graph(&A, NUM_NODES, NUM_EDGES, in_degree);

  float *r = (float *)malloc(A.num_nodes * sizeof(float));

  auto cpu_begin = std::chrono::high_resolution_clock::now();
  // PageRank 계산
  pageRank_CSR(&A, r);

  auto cpu_end = std::chrono::high_resolution_clock::now();


  NodeData *nodes = (NodeData *)malloc(A.num_nodes * sizeof(NodeData));
  for (int i = 0; i < A.num_nodes; i++) {
      nodes[i].node = i;
      nodes[i].in_degree = in_degree[i];
      nodes[i].rank = r[i];
  }

  // PageRank 순으로 정렬하여 출력
  //qsort(nodes, A.num_nodes, sizeof(NodeData), compare_by_rank);
  printf("Sorted by rank:\n");
  float sum = 0.0;
  for (int i = 0; i < A.num_nodes; i++) {
      printf("Node %d: in-degree %d, rank %f\n", nodes[i].node, nodes[i].in_degree, nodes[i].rank);
      sum += nodes[i].rank;
  }
  
  printf("sum %f\n", sum);
  // The host code assumes there is a single device and opening a device by
  // device index 0. If there are multiple devices then this device index needed
  // to be adjusted. The user can get the list of the devices and device indices
  // by xbtuil examine command.
  unsigned int device_index = 0;
  std::cout << "Open the device" << device_index << std::endl;
  auto device = xrt::device(device_index);
  
  std::cout << "Load the xclbin " << xclbin_file_name << std::endl;
  auto uuid = device.load_xclbin(xclbin_file_name);

  size_t row_ptr_bytes = sizeof(int) * (A.num_nodes + 1);
  size_t col_idx_bytes = sizeof(int) * A.num_edges;
  size_t score_bytes = sizeof(float) * A.num_nodes;

  auto krnl = xrt::kernel(device, uuid, "pagerank");

  std::cout << "Allocate Buffer in Global Memory\n";
  auto bo0 = xrt::bo(device, col_idx_bytes, krnl.group_id(0));
  auto bo1 = xrt::bo(device, row_ptr_bytes, krnl.group_id(1));
  auto bo_score_1 = xrt::bo(device, score_bytes, krnl.group_id(2));
  auto bo_score_2 = xrt::bo(device, score_bytes, krnl.group_id(3));
  auto bo_row = xrt::bo(device, row_ptr_bytes, krnl.group_id(4));
  auto bo_col = xrt::bo(device, col_idx_bytes, krnl.group_id(5));
  // Map the contents of the buffer object into host memory
  auto bo0_map = bo0.map<int *>();
  auto bo1_map = bo1.map<int *>();
  auto bo_score_map_1 = bo_score_1.map<float *>();
  auto bo_score_map_2 = bo_score_2.map<float *>();
  auto bo_row_map = bo_row.map<int *>();
  auto bo_col_map = bo_col.map<int *>();
  std::fill(bo0_map, bo0_map + A.num_edges, 0);
  std::fill(bo1_map, bo1_map + A.num_nodes + 1, 0);

  std::fill(bo_score_map_1, bo_score_map_1 + A.num_nodes, 1.0/A.num_nodes);
  std::fill(bo_score_map_2, bo_score_map_2 + A.num_nodes, 0.0);

  std::fill(bo_row_map, bo_row_map + A.num_nodes + 1, 0);
  std::fill(bo_col_map, bo_col_map + A.num_edges, 0);

  // Create the test data
  vector<float> bufReference(A.num_nodes);
  for (int i = 0; i < A.num_nodes + 1; ++i) {
    bo1_map[i] = A.row_ptr[i];
  }
  for (int i = 0; i < A.num_edges; ++i) {
    bo0_map[i] = A.col_idx[i];
  }
  for (int i = 0; i < A.num_nodes; ++i) {
    bo_score_map_1[i] = 1.0/A.num_nodes;
  }
  for (int i = 0; i < A.num_nodes; ++i) {
    bo_score_map_2[i] = 0.0;
  }
  for (int i = 0; i < A.num_nodes; ++i) {
    bufReference[i] = nodes[i].rank;
  }

  auto fpga_begin = std::chrono::high_resolution_clock::now();

  // Synchronize buffer content with device side
  std::cout << "synchronize input buffer data to device global memory\n";


  //////////////////////////////////////////////////////////////////////////////
  auto host_to_fpga_start = std::chrono::high_resolution_clock::now();


  bo0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  bo_score_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_score_2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  auto host_to_fpga_end = std::chrono::high_resolution_clock::now();
  /////////////////////////////////////////////////////////////////////////////


  std::cout << "synchronize input buffer data to device global memory finish\n";
  auto fpga_cal_begin = std::chrono::high_resolution_clock::now();
  auto run = krnl(bo0, bo1, bo_score_1, bo_score_2, bo_row, bo_col);
  run.wait();
  std::cout << "finish run" << std::endl;
  auto fpga_cal_end = std::chrono::high_resolution_clock::now();

  // Get the output;
  std::cout << "Get the output data from the device" << std::endl;
  
  //////////////////////////////////////////////////////////////////////////////
  auto fpga_to_host_start = std::chrono::high_resolution_clock::now();

  bo_score_1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_row.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_col.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

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

  std::chrono::duration<double> cpu_duration = cpu_end - cpu_begin;
  std::cout << "CPU Time:                     " << cpu_duration.count() << " s" << std::endl;

  std::cout << "FPGA Speedup:                 " << cpu_duration.count() / fpga_duration.count() << " x" << std::endl;
  
  // 메모리 해제
  free(A.row_ptr);
  free(A.col_idx);
  free(A.values);
  free(r);
  free(in_degree);
  free(nodes);

  sum = 0.0;
  for (int i = 0; i < A.num_nodes; i++) {
      printf("Node rank %f\n", bo_score_map_1[i]);
      sum += bo_score_map_1[i];
  }
  printf("sum %f\n", sum);

  /*
  for (int i = 0; i < A.num_nodes+1; i++) {
      printf("bo_row_map %d\n", bo_row_map[i]);
  }

  for (int i = 0; i < A.num_nodes+1; i++) {
      printf("bo_row_map %d\n", bo1_map[i]);
  }
  
  for (int i = 400; i < 500; i++) {
      printf("bo_col_map %d\n", bo_col_map[i]);
  }
  printf("aaaaaaaaaaaaaaa\n" );
  for (int i = 400; i < 500; i++) {
      printf("bo0_map %d\n", bo0_map[i]);
  }*/



  // Validate our results
  
  auto compare_begin = std::chrono::high_resolution_clock::now();

  if (std::memcmp(&bo_row_map[0], &bo1_map[0], (NUM_NODES+1)))
    throw std::runtime_error("Row does not match reference");
  
  std::cout << "row PASSED\n";
  if (std::memcmp(&bo_col_map[0], &bo0_map[0], NUM_EDGES))
    throw std::runtime_error("Col does not match reference");

  std::cout << "col PASSED\n";

  sum = 0.0;
  for (int i = 0; i < A.num_nodes; i++) {
      sum += abs(bo_score_map_1[i] - bufReference[i]) ;
  }
  printf("diff %f\n", sum);

  if (sum > 0.002)
    throw std::runtime_error("Score does not match reference");

  std::cout << "TEST PASSED\n";
  
  auto compare_end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> compare_duration = compare_end - compare_begin;
  std::cout << "Compare Time:                 " << compare_duration.count() << " s" << std::endl;

  return 0;
}
