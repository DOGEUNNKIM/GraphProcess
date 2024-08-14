#include <iostream>
#include "ap_fixed.h"
#include "hls_stream.h"

#define VDATA_SIZE 16
#define NUM_NODES 1024*64
#define TILE_SIZE 1024*32
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE

typedef struct v_datatype { int data[VDATA_SIZE]; } v_dt;
typedef struct r_f_datatype { int data[2]; } row_dt;


extern "C"{
void bfs(const v_dt *col,  // Read-Only Vector 1 from hbm -> col index
        const v_dt *row,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
        int *frontier,      // Output Result to hbm -> bfs score before
        int *Vprop       // Output Result to hbm -> bfs score next
        ) {

#pragma HLS INTERFACE m_axi port = col offset = slave bundle = gmem0 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = row offset = slave bundle = gmem1 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = frontier offset = slave bundle = gmem3 
#pragma HLS INTERFACE m_axi port = Vprop offset = slave bundle = gmem4 

#pragma HLS INTERFACE s_axilite port=col bundle=control
#pragma HLS INTERFACE s_axilite port=row bundle=control
#pragma HLS INTERFACE s_axilite port=frontier bundle=control
#pragma HLS INTERFACE s_axilite port=Vprop bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int value = -1;
    int beforeIDX = 0;
    int currIDX = 1;
    int nextNumIDX = 1;

    for (beforeIDX = 0; beforeIDX < NUM_NODES; beforeIDX = currIDX) {
        if(beforeIDX !=0 && currIDX == nextNumIDX){
            break;
        }
        currIDX = nextNumIDX;
        value++;
        for (int tile = 0; tile < NUM_TILES; tile++) {
            // Vprop을 프리패치하는 부분
            for (int idx = beforeIDX; idx < currIDX; idx++) {
                int old_Vprop_idx = frontier[idx];
                int row_ptr_start = row[(tile*NUM_NODES+ old_Vprop_idx) >> 4].data[(tile*NUM_NODES+ old_Vprop_idx)%16];
                int row_ptr_end = row[(tile*NUM_NODES+ old_Vprop_idx + 1) >> 4].data[(tile*NUM_NODES+ old_Vprop_idx + 1)%16];
                for (int ptr = row_ptr_start; ptr < row_ptr_end; ptr++) {
                    int col_idx_value = col[ptr >> 4].data[ptr%16];

                    if (Vprop[col_idx_value] == -1 || Vprop[col_idx_value] > value + 1) {
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