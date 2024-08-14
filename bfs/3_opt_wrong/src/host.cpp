#include <cstring>
#include <iostream>
#include <chrono>
#include <math.h>
#include <omp.h>
#include <vector>
#include <time.h>
#include <queue>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include "ap_fixed.h"

#define VDATA_SIZE 16
#define NUM_NODES 4096
#define NUM_EDGES 40000
#define BUFFER_SIZE 16
#define TILE_SIZE 512
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE
#define START_VERTEX 52

using namespace std;

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

    //printf("col_idx\n");
    //for (int i = 0; i < num_edges; i++) {
    //    printf("%d ", A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE]);
    //}
    //printf("\n");

    free(out_degree);
    free(edges);
    free(current_pos);
}

void tile_CSRMatrixunc(const CSRMatrix *A, CSRMatrix *T, int tile_size) {
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


int main(int argc, char **argv) {
    
  std::string xclbin_file_name = argv[1];
  CSRMatrix A;
  int *in_degree = (int *)calloc(NUM_NODES, sizeof(int));

  generate_random_graph(&A, NUM_NODES, NUM_EDGES, in_degree);

  int *r = (int *)malloc(A.num_nodes * sizeof(int));

  auto cpu_begin = std::chrono::high_resolution_clock::now();

  //Do PageRank
  bfs_CSR(&A, START_VERTEX ,r);

  auto cpu_end = std::chrono::high_resolution_clock::now();

  //PREPROCESS 
  CSRMatrix T;
  tile_CSRMatrixunc(&A, &T, TILE_SIZE);

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

  size_t row_ptr_process_bytes = sizeof(int) * (A.num_nodes*NUM_TILES  + 1);
  size_t col_idx_process_bytes = sizeof(int) * A.num_edges;
  size_t node_bytes = sizeof(int) * A.num_nodes;

  auto krnl = xrt::kernel(device, uuid, "bfs");

  std::cout << "Allocate Buffer in Global Memory\n";
  auto col = xrt::bo(device, col_idx_process_bytes, krnl.group_id(0));
  auto row = xrt::bo(device, row_ptr_process_bytes, krnl.group_id(1));
  auto frontier = xrt::bo(device, node_bytes, krnl.group_id(2));
  auto Vprop = xrt::bo(device, node_bytes, krnl.group_id(3));

  // Map the contents of the buffer object into host memory
  auto col_map = col.map<int *>();
  auto row_map = row.map<int *>();
  auto frontier_map = frontier.map<int *>();
  auto Vprop_map = Vprop.map<int *>();

  std::fill(col_map, col_map + A.num_edges, 0);
  std::fill(row_map, row_map + A.num_nodes*NUM_TILES + 1, 0);
  std::fill(frontier_map, frontier_map + A.num_nodes, -1);
  std::fill(Vprop_map, Vprop_map + A.num_nodes, -1);
  
  // Create the test data
  vector<int> bufReference(A.num_nodes);
  
  for (int i = 0; i < A.num_edges; ++i) {
    col_map[i] = T.col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  for (int i = 0; i < A.num_nodes*NUM_TILES + 1; ++i) {
    row_map[i] = T.row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  frontier_map[0] = START_VERTEX;
  Vprop_map[START_VERTEX] = 0;

  for (int i = 0; i < A.num_nodes; ++i) {
    bufReference[i] = nodes[i].rank;
  }

  auto fpga_begin = std::chrono::high_resolution_clock::now();

  // Synchronize buffer content with device side
  std::cout << "synchronize input buffer data to device global memory\n";


  //////////////////////////////////////////////////////////////////////////////
  auto host_to_fpga_start = std::chrono::high_resolution_clock::now();


  col.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  row.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  frontier.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  Vprop.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  auto host_to_fpga_end = std::chrono::high_resolution_clock::now();
  /////////////////////////////////////////////////////////////////////////////


  std::cout << "synchronize input buffer data to device global memory finish\n";
  auto fpga_cal_begin = std::chrono::high_resolution_clock::now();
  auto run = krnl(col, row, frontier, Vprop);
  run.wait();
  std::cout << "finish run" << std::endl;
  auto fpga_cal_end = std::chrono::high_resolution_clock::now();

  // Get the output;
  std::cout << "Get the output data from the device" << std::endl;
  
  //////////////////////////////////////////////////////////////////////////////
  auto fpga_to_host_start = std::chrono::high_resolution_clock::now();

  Vprop.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

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


  //Check FPGA result
  auto compare_begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < A.num_nodes; i++) {
      if (Vprop_map[i] > bufReference[i]){
          throw std::runtime_error("Score does not match reference");
      }
  }
  auto compare_end = std::chrono::high_resolution_clock::now();
  std::cout << "TEST PASSED\n";

  std::chrono::duration<double> compare_duration = compare_end - compare_begin;
  std::cout << "Compare Time:                 " << compare_duration.count() << " s" << std::endl;

  return 0;
}
