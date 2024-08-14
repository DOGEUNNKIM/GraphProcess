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


#define VDATA_SIZE 16
//#define ALPHA 0.85
#define MAX_ITER 1000
//#define EPSILON 1e-6
#define NUM_NODES 2048
#define NUM_EDGES 4096*16
#define TILE_SIZE 512
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE

float ALPHA = 0.85;
float EPSILON = 1e-6;


typedef struct v_datatype { int data[VDATA_SIZE]; } v_dt;
typedef struct v_f_datatype { float data[VDATA_SIZE]; } v_dt_f;

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

using namespace std;

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
    #pragma omp parallel
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
    #pragma omp parallel for
    for (int i = 1; i <= num_nodes; i++) {
        A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] = A->row_ptr[(i - 1) / VDATA_SIZE].data[(i - 1) % VDATA_SIZE] + out_degree[i - 1];
    }

    // Temporary array to keep track of positions in col_idx
    int *current_pos = (int *)malloc(num_nodes * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) {
        current_pos[i] = A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
    }

    // Fill col_idx based on row_ptr
    #pragma omp parallel for
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
    #pragma omp parallel for
    for (int t = 0; t < num_tiles; t++) {
        tile_CSRMatrix[t].num_nodes = num_nodes;
        tile_CSRMatrix[t].num_edges = 0; // This will be adjusted later
        tile_CSRMatrix[t].row_ptr = (v_dt *)malloc(((num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
        tile_CSRMatrix[t].col_idx = NULL; // This will be allocated later
        tile_CSRMatrix[t].values = NULL; // This will be allocated later
    }
    // Count the edges for each tile
    //#pragma omp parallel for
    for (int i = 0; i < A->num_edges; i++) {
        int index = A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE] / tile_size;
        tile_CSRMatrix[index].num_edges++;
    }
    //#pragma omp parallel for
    for (int i = 0; i < num_tiles; i++) {
        tile_CSRMatrix[i].col_idx = (v_dt *)malloc(((tile_CSRMatrix[i].num_edges) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
        tile_CSRMatrix[i].values = (float *)malloc((tile_CSRMatrix[i].num_edges) * sizeof(float));
        tile_CSRMatrix[i].row_ptr[0].data[0] = 0; // Initialize the first element of row_ptr
    }
    int col_idx_ptr[num_tiles] = {0};
    //#pragma omp parallel for
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
    //#pragma omp parallel for
    for (int t = 0; t < num_tiles; t++) {
        for (int i = 0; i <= num_nodes; i++) {
            tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] = 0;
        }
    }
    //#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) {
        int row_start = A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
        int row_end = A->row_ptr[(i + 1) / VDATA_SIZE].data[(i + 1) % VDATA_SIZE];

        for (int j = row_start; j < row_end; j++) {
            int col = A->col_idx[j / VDATA_SIZE].data[j % VDATA_SIZE];
            int tile_index = col / tile_size;

            tile_CSRMatrix[tile_index].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]++;
        }
    }
    //#pragma omp parallel for
    for (int t = 0; t < num_tiles; t++) {
        int cumulative_sum = 0;
        for (int i = 0; i <= num_nodes; i++) {
            int temp = tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
            tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] = cumulative_sum;
            cumulative_sum += temp;
        }
    }
    //printf("tile row_ptr\n");
    //#pragma omp parallel for
    for (int t = 0; t < num_tiles; t++) {
        for (int i = 0; i < tile_CSRMatrix[t].num_nodes+1; i++) {
            //printf("%d ", tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]);
            if(t > 0){
              tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] += tile_CSRMatrix[t-1].row_ptr[num_nodes / VDATA_SIZE].data[num_nodes % VDATA_SIZE];
            }
        }
        //printf("\n");
    }
    //#pragma omp parallel for
    for (int t = 0; t < num_tiles; t++) {
        for (int i = 1; i <= num_nodes; i++) {
             T->row_ptr[(t*num_nodes + i) / VDATA_SIZE].data[(t*num_nodes + i) % VDATA_SIZE] =  tile_CSRMatrix[t].row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
        }
    }

    int j=0;
    //#pragma omp parallel for
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

void pageRank_CSR(const CSRMatrix_f *A, float *r) {
    int num_nodes = A->num_nodes;
    float *r_new = (float *)malloc(num_nodes * sizeof(float));
    float one = 1;
    float base_score = (one - ALPHA) / num_nodes;

    for (int i = 0; i < num_nodes; i++) {
        r[i] = 1.0 / num_nodes;
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {

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

                //#pragma omp simd
                for (int k = 0; k < chunk_size; k++) {
                    int v = col_idx_buffer[k];
                    //#pragma omp atomic
                    r_new[v] += ALPHA * r[u] / out_degree_u;
                }
            }
        }

        float diff = 0.0;
        for (int i = 0; i < num_nodes; i++) {
            diff += (r_new[i] - r[i])*(r_new[i] - r[i]) ;
        }
        if (diff < EPSILON) {
            printf("CPU iter = %d\n", iter);
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
  CSRMatrix_f A;
  int *in_degree = (int *)calloc(NUM_NODES, sizeof(int));

  generate_random_graph(&A, NUM_NODES, NUM_EDGES, in_degree);

  float *r = (float *)malloc(A.num_nodes * sizeof(float));

  auto cpu_begin = std::chrono::high_resolution_clock::now();
  //Do PageRank
  pageRank_CSR(&A, r);

  auto cpu_end = std::chrono::high_resolution_clock::now();

  //PREPROCESS 
  CSRMatrix T;
  tile_CSRMatrix_func(&A, &T, TILE_SIZE);

  NodeData *nodes = (NodeData *)malloc(A.num_nodes * sizeof(NodeData));
  for (int i = 0; i < A.num_nodes; i++) {
      nodes[i].node = i;
      nodes[i].in_degree = in_degree[i];
      nodes[i].rank = r[i];
  }
    
  // The host code assumes there is a single device and opening a device by
  // device index 0. If there are multiple devices then this device index needed
  // to be adjusted. The user can get the list of the devices and device indices
  // by xbtuil examine command.
  unsigned int device_index = 0;
  std::cout << "Open the device" << device_index << std::endl;
  auto device = xrt::device(device_index);
  
  std::cout << "Load the xclbin " << xclbin_file_name << std::endl;
  auto uuid = device.load_xclbin(xclbin_file_name);

  size_t row_ptr_bytes = sizeof(float) * (A.num_nodes + 1);
  size_t row_ptr_process_bytes = sizeof(int) * (A.num_nodes*NUM_TILES  + 1);
  size_t col_idx_process_bytes = sizeof(int) * A.num_edges;
  size_t score_bytes = sizeof(float) * A.num_nodes;

  auto krnl = xrt::kernel(device, uuid, "pagerank");

  std::cout << "Allocate Buffer in Global Memory\n";
  auto bo0 = xrt::bo(device, col_idx_process_bytes, krnl.group_id(0));
  auto bo1 = xrt::bo(device, row_ptr_process_bytes, krnl.group_id(1));
  auto bo2 = xrt::bo(device, row_ptr_bytes, krnl.group_id(2));
  auto bo_score_1 = xrt::bo(device, score_bytes, krnl.group_id(3));
  auto bo_score_2 = xrt::bo(device, score_bytes, krnl.group_id(4));
  // Map the contents of the buffer object into host memory
  auto bo0_map = bo0.map<int *>();
  auto bo1_map = bo1.map<int *>();
  auto bo2_map = bo2.map<float *>();
  auto bo_score_map_1 = bo_score_1.map<float *>();
  auto bo_score_map_2 = bo_score_2.map<float *>();
  std::fill(bo0_map, bo0_map + A.num_edges, 0);
  std::fill(bo1_map, bo1_map + A.num_nodes*NUM_TILES + 1, 0);
  std::fill(bo2_map, bo2_map + A.num_nodes + 1, 0.0f);

  std::fill(bo_score_map_1, bo_score_map_1 + A.num_nodes, 1.0/A.num_nodes);
  std::fill(bo_score_map_2, bo_score_map_2 + A.num_nodes, 0.0);
  
  // Create the test data

  //
  // need check
  //

  vector<float> bufReference(A.num_nodes);

  float one =1;
  float zero = 0;
  
  for (int i = 0; i < A.num_edges; ++i) {
    bo0_map[i] = T.col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  for (int i = 0; i < A.num_nodes*NUM_TILES + 1; ++i) {
    bo1_map[i] = T.row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  for (int i = 0; i < A.num_nodes; ++i) {
    bo2_map[i] = A.row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  for (int i = 0; i < A.num_nodes; ++i) {
    bo_score_map_1[i] = one/A.num_nodes;
  }
  for (int i = 0; i < A.num_nodes; ++i) {
    bo_score_map_2[i] = zero;
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
  bo2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  bo_score_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_score_2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  auto host_to_fpga_end = std::chrono::high_resolution_clock::now();
  /////////////////////////////////////////////////////////////////////////////


  std::cout << "synchronize input buffer data to device global memory finish\n";
  auto fpga_cal_begin = std::chrono::high_resolution_clock::now();
  auto run = krnl(bo0, bo1, bo2,bo_score_1, bo_score_2);
  run.wait();
  std::cout << "finish run" << std::endl;
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


  //Check result
  //for (int i = 0; i < A.num_nodes; i++) {
  //    std::cout << "Node rank1: " << static_cast<float>(nodes[i].rank) << std::endl;
  //    std::cout << "Node rank2 " << static_cast<float>(bo_score_map_2[i]) << std::endl;
  //}
    

  // Validate our results
  
  auto compare_begin = std::chrono::high_resolution_clock::now();


  float diff = 0;
  for (int i = 0; i < A.num_nodes; i++) {
      diff += (bo_score_map_2[i] - bufReference[i])*(bo_score_map_2[i] - bufReference[i]) ;
  }
  std::cout << "Diff: " << static_cast<float>(diff) << std::endl;

  if (diff > EPSILON*100)
    throw std::runtime_error("Score does not match reference");

  std::cout << "TEST PASSED\n";
  
  auto compare_end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> compare_duration = compare_end - compare_begin;
  std::cout << "Compare Time:                 " << compare_duration.count() << " s" << std::endl;

  return 0;
}
