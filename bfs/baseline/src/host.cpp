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
#define NUM_NODES 4039
#define NUM_EDGES 88234
#define START_VERTEX 0

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
    int64_t rank;
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

int64_t edge_exists(Edge *edges, int64_t edge_count, int64_t src, int64_t dst) {
    for (int64_t i = 0; i < edge_count; i++) {
        if (edges[i].src == src && edges[i].dst == dst) {
            return 1;
        }
    }
    return 0;
}

void generate_random_graph(CSRMatrix *A, int64_t num_nodes, int64_t num_edges, int64_t *in_degree) {
    printf("make graph\n");
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
    A->row_ptr[0].data[0] = 0;
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
    std::queue<int64_t > q;
    bool *visited = (bool *)malloc(num_nodes * sizeof(bool));
    std::memset(visited, 0, num_nodes * sizeof(bool));

    // Initialize distances array
    for (int64_t  i = 0; i < num_nodes; i++) {
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
        //#pragma omp parallel for
        for (int64_t  b = 0; b < buffer_size; b += 10) {
            int64_t chunk_size = (b + 10 > buffer_size) ? buffer_size - b : 10;
            for (int64_t  k = 0; k < chunk_size; k++) {
                col_idx_buffer[k] = A->col_idx[(buffer_start + b + k) / VDATA_SIZE].data[(buffer_start + b + k) % VDATA_SIZE];
            }

            for (int64_t  k = 0; k < chunk_size; k++) {
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
void bfs_hls(const v_dt *col,  // Read-Only Vector 1 from hbm -> col index
        const v_dt *row,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
        int64_t *frontier,      // Output Result to hbm -> bfs score before
        int *Vprop       // Output Result to hbm -> bfs score next
        ) {

    int64_t value = -1;
    int64_t beforeIDX = 0;
    int64_t currIDX = 1;
    int64_t nextNumIDX = 1;

    for (beforeIDX = 0; beforeIDX < NUM_NODES; beforeIDX = currIDX) {
        if((beforeIDX !=0) && (currIDX == nextNumIDX)){
            break;
        }
        currIDX = nextNumIDX;
        value = value + 1;
        for (int64_t idx = beforeIDX; idx < currIDX; idx++) {
            int64_t old_Vprop_idx = frontier[idx];
            int64_t row_ptr_start = row[(old_Vprop_idx) / VDATA_SIZE].data[(old_Vprop_idx)%VDATA_SIZE];
            int64_t row_ptr_end = row[(old_Vprop_idx + 1) / VDATA_SIZE].data[(old_Vprop_idx + 1)%VDATA_SIZE];
            for (int64_t ptr = row_ptr_start; ptr < row_ptr_end; ptr++) {
                int64_t col_idx_value = col[ptr / VDATA_SIZE].data[ptr%VDATA_SIZE];
                if (Vprop[col_idx_value] == -1 || Vprop[col_idx_value] > value + 1) {
                    Vprop[col_idx_value] = (value + 1);
                    frontier[nextNumIDX] = col_idx_value;
                    nextNumIDX++;
                }
            }
        }
    }
}


int main(int  argc, char **argv) {
    
  std::string xclbin_file_name = argv[1];
  CSRMatrix A;

  printf("START LOAD GRAPH\n");
  //load_csr_matrix(&A, "/home/kdg6245/graph/dataset/csr_matrix_facebook_int64.bin");
  int64_t *indeg = (int64_t  *)malloc(NUM_NODES * sizeof(int64_t ));
  generate_random_graph(&A, NUM_NODES, NUM_EDGES,indeg );
  printf("FINISH LOAD GRAPH\n");    


  int64_t *r = (int64_t  *)malloc(A.num_nodes * sizeof(int64_t ));

  //auto cpu_begin = std::chrono::high_resolution_clock::now();

  //Do BFS
  printf("START CPU BFS\n");
  bfs_CSR(&A, START_VERTEX ,r);
  printf("FINISH CPU BFS\n");

  //auto cpu_end = std::chrono::high_resolution_clock::now();

  //PREPROCESS 
  CSRMatrix T;
  T = A;
  //printf("START PREPROCESS GRAPH\n");
  //tile_CSRMatrix_func(&A, &T, TILE_SIZE);
  //printf("FINISH PREPROCESS GRAPH\n");

  NodeData *nodes = (NodeData *)malloc((int64_t)A.num_nodes * sizeof(NodeData));
  for (int64_t  i = 0; i < A.num_nodes; i++) {
      nodes[i].node = i;
      nodes[i].rank = r[i];
  }

////
//
    int64_t *frontier1 = (int64_t *)malloc(A.num_nodes * sizeof(int64_t));
    int *Vprop1 = (int*)malloc(A.num_nodes * sizeof(int));
    for (int64_t  i = 0; i < NUM_NODES; i++) {
        frontier1[i] = -1;
        Vprop1[i] = -1;
    }
    frontier1[0] = START_VERTEX;
    Vprop1[START_VERTEX] = 0;

  auto cpu_begin = std::chrono::high_resolution_clock::now();
  bfs_hls(T.col_idx, T.row_ptr, frontier1, Vprop1);

  auto cpu_end = std::chrono::high_resolution_clock::now();




//
////
  // The host code assumes there is a single device and opening a device by
  // device index 0. If there are multiple devices then this device index needed
  // to be adjusted. The user can get the list of the devices and device indices
  // by xbtuil examine command.
  unsigned int device_index = 0;
  std::cout << "Open the device" << device_index << std::endl;
  auto device = xrt::device(device_index);
  
  std::cout << "Load the xclbin " << xclbin_file_name << std::endl;
  auto uuid = device.load_xclbin(xclbin_file_name);
  
  int64_t row_ptr_process_bytes = sizeof(int64_t ) * (A.num_nodes  + 1);
  int64_t col_idx_process_bytes = sizeof(int64_t ) * (A.num_edges+1);
  int64_t node_bytes = sizeof(int64_t) * A.num_nodes;
  int vprop_bytes = sizeof(int) * A.num_nodes;

  auto krnl = xrt::kernel(device, uuid, "bfs");

  std::cout << "Allocate Buffer in Global Memory\n";
  auto col = xrt::bo(device, col_idx_process_bytes, krnl.group_id(0));
  printf("col done\n");
  auto row = xrt::bo(device, row_ptr_process_bytes, krnl.group_id(1));
  printf("row done\n");
  auto frontier = xrt::bo(device, node_bytes, krnl.group_id(2));
  printf("frontier done\n");
  auto Vprop2 = xrt::bo(device, vprop_bytes, krnl.group_id(3));
  printf("Vprop2 done\n");

  std::cout << "Map Buffer in Global Memory\n";
  // Map the contents of the buffer object into host memory
  auto col_map = col.map<int64_t*>();
  auto row_map = row.map<int64_t*>();
  auto frontier_map = frontier.map<int64_t*>();
  auto Vprop_map = Vprop2.map<int*>();

  std::cout << "Fill Buffer in Global Memory\n";
  std::fill(col_map, col_map + A.num_edges, 0);
  std::fill(row_map, row_map + A.num_nodes + 1, 0);
  std::fill(frontier_map, frontier_map + A.num_nodes,  (int64_t)-1);
  std::fill(Vprop_map, Vprop_map + A.num_nodes, -1);
  
  // Create the test data
  vector<int64_t> bufReference(A.num_nodes);
  std::cout << "Fill Buffer in Global Memory\n";
  for (int64_t  i = 0; i < A.num_edges; ++i) {
    col_map[i] = T.col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  for (int64_t i = 0; i < (int64_t)(A.num_nodes + 1); ++i) {
    row_map[i] = T.row_ptr[i / VDATA_SIZE ].data[i % VDATA_SIZE];
  }
  for (int64_t  i = 0; i < A.num_nodes; ++i) {
    frontier_map[i] = -1;
  }
  for (int64_t  i = 0; i < A.num_nodes; ++i) {
    Vprop_map[i] = -1;
  }
  
  frontier_map[0] = START_VERTEX;
  Vprop_map[START_VERTEX] = 0;

  for (int64_t  i = 0; i < A.num_nodes; ++i) {
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
  Vprop2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  auto host_to_fpga_end = std::chrono::high_resolution_clock::now();
  /////////////////////////////////////////////////////////////////////////////


  std::cout << "synchronize input buffer data to device global memory finish\n";
  auto fpga_cal_begin = std::chrono::high_resolution_clock::now();
  std::cout << "START FPGA" << std::endl;
  auto run = krnl(col, row, frontier, Vprop2);
  run.wait();
  std::cout << "FINISH FPGA" << std::endl;
  auto fpga_cal_end = std::chrono::high_resolution_clock::now();

  // Get the output;
  std::cout << "Get the output data from the device" << std::endl;
  
  //////////////////////////////////////////////////////////////////////////////
  auto fpga_to_host_start = std::chrono::high_resolution_clock::now();
  
  col.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  row.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  frontier.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  Vprop2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

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
  free(indeg);
  free(nodes);
  
  auto compare_begin = std::chrono::high_resolution_clock::now();

  printf("CHECK FRONTIER\n");
  for (int64_t i = 0; i < A.num_nodes; i++) {
      if (frontier_map[i] != frontier1[i]){
          cout << i << endl;
        throw std::runtime_error("frontier_map does not match reference");
      }
  }
  printf("CHECK VPROP\n");
  for (int64_t i = 0; i < 300; i++) {
    printf("Vprop_map[%ld] = %d\n", frontier_map[i],Vprop_map[frontier_map[i]]);
  }

  for (int64_t i = 0; i < A.num_nodes; i++) {
      if (Vprop_map[frontier_map[i]] != Vprop1[frontier1[i]]){
          printf("i = %ld\n",i);
          printf("FPGA = %d, CPU = %d\n",Vprop_map[frontier_map[i]],Vprop1[frontier_map[i]]);
          throw std::runtime_error("Score does not match reference");
      }
  }
  auto compare_end = std::chrono::high_resolution_clock::now();
  std::cout << "TEST PASSED\n";

  std::chrono::duration<double> compare_duration = compare_end - compare_begin;
  std::cout << "Compare Time:                 " << compare_duration.count() << " s" << std::endl;

  return 0;
}
