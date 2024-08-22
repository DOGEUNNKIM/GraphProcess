#include <iostream>
#include "ap_fixed.h"
#include "hls_stream.h"

#define VDATA_SIZE 8
#define TILE_SIZE 4847571
#define NUM_NODES 4847571
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE

typedef struct v_datatype { int64_t data[VDATA_SIZE]; } v_dt;


extern "C"{
void bfs(const v_dt *col,  // Read-Only Vector 1 from hbm -> col index
        const v_dt *row,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
        int64_t *frontier,      // Output Result to hbm -> bfs score before
        v_dt *Vprop       // Output Result to hbm -> bfs score next
        ) {

#pragma HLS INTERFACE m_axi port = col offset = slave bundle = gmem0 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = row offset = slave bundle = gmem1 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = frontier offset = slave bundle = gmem2 depth = 4096
#pragma HLS INTERFACE m_axi port = Vprop offset = slave bundle = gmem3  depth = 4096

#pragma HLS INTERFACE s_axilite port=col bundle=control
#pragma HLS INTERFACE s_axilite port=row bundle=control
#pragma HLS INTERFACE s_axilite port=frontier bundle=control
#pragma HLS INTERFACE s_axilite port=Vprop bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int64_t value = -1;
    int64_t beforeIDX = 0;
    int64_t currIDX = 1;
    int64_t nextNumIDX = 1;

int64_t row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
int64_t col_idx_value;
int64_t Vprop_buffer[TILE_SIZE];
//#pragma HLS bind_storage variable=Vprop_buffer type =RAM_1P impl=URAM
#pragma HLS array_partition variable=Vprop_buffer factor=16 cyclic

    for (beforeIDX = 0; beforeIDX < NUM_NODES; beforeIDX = currIDX) {
        if(beforeIDX !=0 && currIDX == nextNumIDX){
            break;
        }
        currIDX = nextNumIDX;
        value++;

        for (int64_t tile = 0; tile < NUM_TILES; tile++) {
            for(int64_t k = 0; k < TILE_SIZE; k += VDATA_SIZE){
                for(int64_t i = 0;i<VDATA_SIZE ;i++){
                      Vprop_buffer[k+i] = Vprop[(tile*TILE_SIZE + k+i) / VDATA_SIZE].data[(tile*TILE_SIZE + k+i) % VDATA_SIZE];
                }
            }

            for (int64_t idx = beforeIDX; idx < currIDX; idx++) {
                int64_t old_Vprop_idx = frontier[idx];
                row_ptr_buffer[0] = row[(tile*NUM_NODES+ old_Vprop_idx)  / VDATA_SIZE].data[(tile*NUM_NODES+ old_Vprop_idx) % VDATA_SIZE];
                row_ptr_buffer[1] = row[(tile*NUM_NODES+ old_Vprop_idx + 1)  / VDATA_SIZE].data[(tile*NUM_NODES+ old_Vprop_idx + 1) % VDATA_SIZE];
                for (int64_t ptr = row_ptr_buffer[0]; ptr < row_ptr_buffer[1]; ptr ++ ) {
                    col_idx_value = col[ptr  / VDATA_SIZE].data[ptr % VDATA_SIZE] - TILE_SIZE*tile;
                    if (Vprop_buffer[col_idx_value] == -1 || Vprop_buffer[col_idx_value] > value + 1) {
                       Vprop_buffer[col_idx_value] = value + 1;
                       #pragma HLS dependence variable=Vprop_buffer false
                       frontier[nextNumIDX] = col_idx_value + TILE_SIZE*tile;
                       nextNumIDX++;
                    }
                }
            }
            for(int64_t kk = 0; kk < TILE_SIZE; kk += VDATA_SIZE){
                for(int64_t ii = 0;ii<VDATA_SIZE ;ii++){
                    Vprop[(tile*TILE_SIZE + kk+ii)  / VDATA_SIZE].data[(tile*TILE_SIZE + kk+ii) % VDATA_SIZE] =Vprop_buffer[kk+ii];
                }
            }
        }
    }
}
}