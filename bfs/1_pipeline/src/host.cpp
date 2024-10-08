#include <cstring>
#include <iostream>
#include <chrono>
#include <math.h>
#include <omp.h>
#include <vector>
#include <time.h>
#include <queue>
#include <fstream>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include "ap_fixed.h"

#define VDATA_SIZE 16
#define NUM_NODES 4847571
#define TILE_SIZE 4847571
#define NUM_EDGES 68993773
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE
#define START_VERTEX 0

using namespace std;

typedef struct v_datatype { int data[VDATA_SIZE]; } v_dt;

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

//int edge_exists(Edge *edges, int edge_count, int src, int dst) {
//    for (int i = 0; i < edge_count; i++) {
//        if (edges[i].src == src && edges[i].dst == dst) {
//            return 1;
//        }
//    }
//    return 0;
//}
//
//void generate_random_graph(CSRMatrix *A, int num_nodes, int num_edges, int *in_degree) {
//    printf("make graph\n");
//    A->num_nodes = num_nodes;
//    A->num_edges = num_edges;
//    A->row_ptr = (v_dt *)malloc(((num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
//    A->col_idx = (v_dt *)malloc((num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
//    A->values = (int *)malloc(num_edges * sizeof(int));
//
//    int *out_degree = (int *)calloc(num_nodes, sizeof(int));
//    Edge *edges = (Edge *)malloc(num_edges * sizeof(Edge));
//    int edge_count = 0;
//
//    srand(time(NULL));
//    #pragma omp parallel
//    while (edge_count < num_edges) {
//        int src = rand() % num_nodes;
//        int dst = rand() % num_nodes;
//        if (!edge_exists(edges, edge_count, src, dst)) {
//            edges[edge_count].src = src;
//            edges[edge_count].dst = dst;
//            out_degree[src]++;
//            in_degree[dst]++;
//            A->values[edge_count] = 1.0f; // Edge weights are set to 1.0
//            edge_count++;
//        }
//    }
//
//    // Initialize row_ptr
//    A->row_ptr[0].data[0] = 0.0f;
//    #pragma omp parallel for
//    for (int i = 1; i <= num_nodes; i++) {
//        A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE] = A->row_ptr[(i - 1) / VDATA_SIZE].data[(i - 1) % VDATA_SIZE] + out_degree[i - 1];
//    }
//
//    // Temporary array to keep track of positions in col_idx
//    int *current_pos = (int *)malloc(num_nodes * sizeof(int));
//    #pragma omp parallel for
//    for (int i = 0; i < num_nodes; i++) {
//        current_pos[i] = A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
//    }
//
//    // Fill col_idx based on row_ptr
//    #pragma omp parallel for
//    for (int i = 0; i < num_edges; i++) {
//        int src = edges[i].src;
//        int dst = edges[i].dst;
//        int pos = current_pos[src]++;
//        A->col_idx[pos / VDATA_SIZE].data[pos % VDATA_SIZE] = dst;
//    }
//
//    //printf("row_ptr\n");
//    //for (int i = 0; i < num_nodes + 1; i++) {
//    //    printf("%d ", A->row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]);
//    //}
//    //printf("\n");
//
//    //printf("col_idx\n");
//    //for (int i = 0; i < num_edges; i++) {
//    //    printf("%d ", A->col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE]);
//    //}
//    //printf("\n");
//
//    free(out_degree);
//    free(edges);
//    free(current_pos);
//}


int *frontt = (int *)malloc((size_t)NUM_NODES * sizeof(int));


// Function to load CSRMatrix from a binary file
void load_csr_matrix(CSRMatrix *A, const char* filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    infile.read(reinterpret_cast<char*>(&A->num_nodes), sizeof(int));
    infile.read(reinterpret_cast<char*>(&A->num_edges), sizeof(int));

    A->row_ptr = (v_dt*)malloc(((A->num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    A->col_idx = (v_dt*)malloc((A->num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    A->values = (int*)malloc(A->num_edges * sizeof(int));

    infile.read(reinterpret_cast<char*>(A->row_ptr), ((A->num_nodes + 1) + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    infile.read(reinterpret_cast<char*>(A->col_idx), (A->num_edges + VDATA_SIZE - 1) / VDATA_SIZE * sizeof(v_dt));
    infile.read(reinterpret_cast<char*>(A->values), A->num_edges * sizeof(int));

    infile.close();
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
    //#pragma omp parallel for
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
        tile_CSRMatrix[i].values = (int *)malloc((tile_CSRMatrix[i].num_edges) * sizeof(int));
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
        //printf("i = %d\n",i);
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

//BFS code
void bfs_CSR(const CSRMatrix *A, int start_node, int *distances) {
    printf("Do CPU BFS\n");
    int num_nodes = A->num_nodes;
    std::queue<int> q;
    bool *visited = (bool *)malloc((size_t)num_nodes * sizeof(bool));
    std::memset(visited, 0, (size_t)num_nodes * sizeof(bool));

    // Initialize distances array
    for (size_t i = 0; i < num_nodes; i++) {
        distances[i] = -1;  // -1 indicates that the node has not been visited yet
    }

    // Start BFS from the start_node
    visited[start_node] = true;
    distances[start_node] = 0;
    q.push(start_node);
    int iter =0;
    int index = 1;
    
    frontt[0] = start_node;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        iter ++;
    //printf("iter = %d\n", iter);
        int buffer_start = A->row_ptr[u / VDATA_SIZE].data[u % VDATA_SIZE];
        int buffer_end = A->row_ptr[(u + 1) / VDATA_SIZE].data[(u + 1) % VDATA_SIZE];
        int buffer_size = buffer_end - buffer_start;
        int col_idx_buffer[10];
        //#pragma omp parallel for
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
                    frontt[index] = v;
                    index ++;


                }
            }
        }
    }

    free(visited);
}

extern "C"{
void bfs_hls(const v_dt *col,  // Read-Only Vector 1 from hbm -> col index
        const v_dt *row,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
        int *frontier,      // Output Result to hbm -> bfs score before
        int *Vprop,       // Output Result to hbm -> bfs score next
        int _NUM_NODES_
        ) {


    int value = -1;
    int beforeIDX = 0;
    int currIDX = 1;
    int nextNumIDX = 1;

    for (beforeIDX = 0; beforeIDX < _NUM_NODES_; beforeIDX = currIDX) {
        if(beforeIDX !=0 && currIDX == nextNumIDX){
            break;
        }
        currIDX = nextNumIDX;
        value++;

        for (int tile = 0; tile < NUM_TILES; tile++) {
            // Vprop을 프리패치하는 부분
            for (int idx = beforeIDX; idx < currIDX; idx++) {
                int old_Vprop_idx = frontier[idx];
                int row_ptr_start = row[((size_t)tile*_NUM_NODES_+ old_Vprop_idx) / VDATA_SIZE].data[((size_t)tile*_NUM_NODES_+ old_Vprop_idx) % VDATA_SIZE];
                int row_ptr_end   = row[((size_t)tile*_NUM_NODES_+ old_Vprop_idx + 1) / VDATA_SIZE].data[((size_t)tile*_NUM_NODES_+ old_Vprop_idx + 1) % VDATA_SIZE];
                for (int ptr = row_ptr_start; ptr < row_ptr_end; ptr++) {
                    int col_idx_value = col[ptr / VDATA_SIZE].data[ptr % VDATA_SIZE];          
                    if (Vprop[col_idx_value] == -1 || Vprop[col_idx_value] > (value + 1)) {
                        Vprop[col_idx_value] = value + 1;
                        frontier[nextNumIDX] = col_idx_value;
                        nextNumIDX++; 
                    }
                }
            }
        }
    }
}
}



int main(int argc, char **argv) {
    
  std::string xclbin_file_name = argv[1];
  CSRMatrix A;
  printf("START LOAD GRAPH\n");
  load_csr_matrix(&A, "/home/kdg6245/graph/dataset/csr_matrix_LiveJournal1_int.bin");
  printf("FINISH LOAD GRAPH\n");
  int *r = (int *)malloc((size_t)A.num_nodes * sizeof(int));
  for(int i=0;i<NUM_NODES;i++){
    frontt[i]=-1;
  }


  auto cpu_begin = std::chrono::high_resolution_clock::now();

  bfs_CSR(&A, START_VERTEX ,r);

  auto cpu_end = std::chrono::high_resolution_clock::now();

  CSRMatrix T;
  tile_CSRMatrixunc(&A, &T, TILE_SIZE);

  NodeData *nodes = (NodeData *)malloc(A.num_nodes * sizeof(NodeData));
  for (size_t i = 0; i < A.num_nodes; i++) {
      nodes[i].node = i;
      nodes[i].rank = r[i];
  }

  unsigned int device_index = 0;
  std::cout << "Open the device" << device_index << std::endl;
  auto device = xrt::device(device_index);
  
  std::cout << "Load the xclbin " << xclbin_file_name << std::endl;
  auto uuid = device.load_xclbin(xclbin_file_name);

  size_t row_ptr_process_bytes = sizeof(int) * ((size_t)A.num_nodes*NUM_TILES  + 1);
  size_t col_idx_process_bytes = sizeof(int) * (size_t)A.num_edges;
  size_t node_bytes = sizeof(int) * (size_t)A.num_nodes;
  printf("row_ptr_process_bytes = %ld\n",row_ptr_process_bytes );
  printf("col_idx_process_bytes = %ld\n",col_idx_process_bytes );
  printf("node_bytes = %ld\n",node_bytes );

  auto krnl = xrt::kernel(device, uuid, "bfs");

  std::cout << "Allocate Buffer in Global Memory\n";
  auto col = xrt::bo(device, col_idx_process_bytes, krnl.group_id(0));
  auto row = xrt::bo(device, row_ptr_process_bytes, krnl.group_id(1));
  auto frontier = xrt::bo(device, node_bytes, krnl.group_id(2));
  auto Vprop = xrt::bo(device, node_bytes, krnl.group_id(3));

  auto col_map = col.map<int *>();
  auto row_map = row.map<int *>();
  auto frontier_map = frontier.map<int *>();
  auto Vprop_map = Vprop.map<int *>();

  std::fill(col_map, col_map + (size_t)A.num_edges, 0);
  std::fill(row_map, row_map + (size_t)(A.num_nodes*(size_t)NUM_TILES) + 1, 0);
  std::fill(frontier_map, frontier_map + (size_t)A.num_nodes, -1);
  std::fill(Vprop_map, Vprop_map + (size_t)A.num_nodes, -1);
  
  vector<int> bufReference(A.num_nodes);
  size_t i=0;
  for (i = 0; i < (size_t)A.num_edges; ++i) {
    col_map[i] = T.col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  printf("col i =%ld\n",i);
  for (i = 0; i < (size_t)row_ptr_process_bytes/4; ++i) {
    row_map[i] = T.row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE];
  }
  printf("row i =%ld\n",i);
  for (i = 0; i < (size_t)A.num_nodes + 1; ++i) {
    frontier_map[i] = -1;
  }
  printf("frontier i =%ld\n",i);
  for (i = 0; i < (size_t)A.num_nodes + 1; ++i) {
    Vprop_map[i] = -1;
  }
  printf("Vprop i =%ld\n",i);
  frontier_map[0] = START_VERTEX;
  Vprop_map[START_VERTEX] = 0;

  for (int i = 0; i < (size_t)A.num_nodes; ++i) {
    bufReference[i] = nodes[i].rank;
  }
  
  auto fpga_begin = std::chrono::high_resolution_clock::now();

  std::cout << "synchronize input buffer data to device global memory\n";
  
  
  auto host_to_fpga_start = std::chrono::high_resolution_clock::now();
  
  
  col.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  row.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  frontier.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  Vprop.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  auto host_to_fpga_end = std::chrono::high_resolution_clock::now();
  
  
  std::cout << "synchronize input buffer data to device global memory finish\n";
  auto fpga_cal_begin = std::chrono::high_resolution_clock::now();
  auto run = krnl(col, row, frontier, Vprop, NUM_NODES );
  run.wait();
  std::cout << "finish run" << std::endl;
  auto fpga_cal_end = std::chrono::high_resolution_clock::now();
  
  std::cout << "Get the output data from the device" << std::endl;
  
  auto fpga_to_host_start = std::chrono::high_resolution_clock::now();
  
  col.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  row.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  frontier.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  Vprop.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  
  auto fpga_to_host_end = std::chrono::high_resolution_clock::now();
  
  
  for (size_t i = 0; i < (size_t)A.num_edges; ++i) {
    if(col_map[i] != T.col_idx[i / VDATA_SIZE].data[i % VDATA_SIZE]){
        printf("col\n");
    }
  }
  for (size_t i = 0; i < (size_t)row_ptr_process_bytes/4; ++i) {
    if(row_map[i] != T.row_ptr[i / VDATA_SIZE].data[i % VDATA_SIZE]){
        printf("row\n");
    }
  }
  printf("frontier1, %ld\n",(size_t)A.num_nodes * sizeof(int));
  int *frontier1 = (int*)malloc((size_t)A.num_nodes * sizeof(int));
  printf("vprop1, %ld\n",(size_t)A.num_nodes * sizeof(int));
  int *Vprop1 = (int*)malloc((size_t)A.num_nodes * sizeof(int));
  for (size_t  i = 0; i < NUM_NODES; i++) {
      frontier1[i] = -1;
      Vprop1[i] = -1;
  }
  frontier1[0] = START_VERTEX;
  Vprop1[START_VERTEX] = 0;
  bfs_hls(T.col_idx, T.row_ptr, frontier1, Vprop1,NUM_NODES);
  //for (int i = 4400337; i < 4400367; ++i) {
  //      printf("frontt[%d] = %d, frontier1[%d] = %d\n",i,frontt[i],i,frontier1[i]);
  //  
  //}
  for (int i = 0; i < 300; ++i) {
        printf("frontier_map[%d] = %d,frontier1[%d] = %d\n"
        ,i, frontier_map[i]
        ,i, frontier1[i]);
  }

  for (int i = 0; i < A.num_nodes; i++) {
      if (frontier1[i] != frontt[i]){
        printf("i= %d\n",i);
          throw std::runtime_error("frontier1 does not match reference");
      }
  }


  for (int i = 0; i < A.num_nodes; i++) {
      if (Vprop1[i] != bufReference[i]){
          throw std::runtime_error("Score does not match reference");
      }
  }
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
  
  free(A.row_ptr);
  free(A.col_idx);
  free(A.values);
  free(r);
  free(nodes);
  free(frontt);


  auto compare_begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < A.num_nodes; i++) {
      if (Vprop_map[i] != bufReference[i]){
          throw std::runtime_error("Score does not match reference");
      }
  }
  auto compare_end = std::chrono::high_resolution_clock::now();
  std::cout << "TEST PASSED\n";

  std::chrono::duration<double> compare_duration = compare_end - compare_begin;
  std::cout << "Compare Time:                 " << compare_duration.count() << " s" << std::endl;

  return 0;
}
