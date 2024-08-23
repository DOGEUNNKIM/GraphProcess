#include <iostream>
#include <stdint.h>
#include "ap_fixed.h"
#include "hls_stream.h"

#define VDATA_SIZE 8
#define NUM_NODES 4039

typedef struct v_datatype { int64_t data[VDATA_SIZE]; } v_dt;


extern "C"{
void bfs(const v_dt *col,  // Read-Only Vector 1 from hbm -> col index
        const v_dt *row,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
        int64_t *frontier,      // Output Result to hbm -> bfs score before
        int *Vprop       // Output Result to hbm -> bfs score next

        ) {

#pragma HLS INTERFACE m_axi port = col      offset = slave bundle = gmem0 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = row      offset = slave bundle = gmem1 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = frontier offset = slave bundle = gmem3 depth = 4096
#pragma HLS INTERFACE m_axi port = Vprop    offset = slave bundle = gmem4 depth = 4096

#pragma HLS INTERFACE s_axilite port=col        bundle=control
#pragma HLS INTERFACE s_axilite port=row        bundle=control
#pragma HLS INTERFACE s_axilite port=frontier   bundle=control
#pragma HLS INTERFACE s_axilite port=Vprop      bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    int value = -1;
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
}