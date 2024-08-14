#include <iostream>
#include "ap_fixed.h"
#include "hls_stream.h"

#define VDATA_SIZE 16
#define NUM_NODES 1024*64
#define TILE_SIZE 1024*32

//BRAM에는 1024*64가 최대임 
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE

typedef struct v_datatype { int data[VDATA_SIZE]; } v_dt;
typedef struct r_f_datatype { int data[2]; } row_dt;


extern "C"{
void bfs(const v_dt *col,  // Read-Only Vector 1 from hbm -> col index
        const v_dt *row,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
        int *frontier,      // Output Result to hbm -> bfs score before
        v_dt *Vprop       // Output Result to hbm -> bfs score next
        ) {

#pragma HLS INTERFACE m_axi port = col offset = slave bundle = gmem0 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = row offset = slave bundle = gmem1 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = frontier offset = slave bundle = gmem3 depth = 4096
#pragma HLS INTERFACE m_axi port = Vprop offset = slave bundle = gmem4 max_widen_bitwidth=512 depth = 4096

#pragma HLS INTERFACE s_axilite port=col bundle=control
#pragma HLS INTERFACE s_axilite port=row bundle=control
#pragma HLS INTERFACE s_axilite port=frontier bundle=control
#pragma HLS INTERFACE s_axilite port=Vprop bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int value = -1;
    int beforeIDX = 0;
    int currIDX = 1;
    int nextNumIDX = 1;

int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
int col_idx_value;
int Vprop_buffer[TILE_SIZE];
#pragma HLS array_partition variable=Vprop_buffer factor=16 cyclic

    for (beforeIDX = 0; beforeIDX < NUM_NODES; beforeIDX = currIDX) {
        if(beforeIDX !=0 && currIDX == nextNumIDX){
            break;
        }
        currIDX = nextNumIDX;
        value++;

        for (int tile = 0; tile < NUM_TILES; tile++) {
            for(int k = 0; k < TILE_SIZE; k += VDATA_SIZE){
            #pragma HLS pipeline II=1
                for(int i = 0;i<VDATA_SIZE ;i++){
                      Vprop_buffer[k+i] = Vprop[(tile*TILE_SIZE + k+i)>>4].data[(tile*TILE_SIZE + k+i)%16];
                }
            }

            for (int idx = beforeIDX; idx < currIDX; idx++) {
                int old_Vprop_idx = frontier[idx];
                row_ptr_buffer[0] = row[(tile*NUM_NODES+ old_Vprop_idx) >> 4].data[(tile*NUM_NODES+ old_Vprop_idx)%16];
                row_ptr_buffer[1] = row[(tile*NUM_NODES+ old_Vprop_idx + 1) >> 4].data[(tile*NUM_NODES+ old_Vprop_idx + 1)%16];
                for (int ptr = row_ptr_buffer[0]; ptr < row_ptr_buffer[1]; ptr ++ ) {
                    col_idx_value = col[ptr >> 4].data[ptr%16] - TILE_SIZE*tile;
                    if (Vprop_buffer[col_idx_value] == -1 || Vprop_buffer[col_idx_value] > value + 1) {
                       Vprop_buffer[col_idx_value] = value + 1;
                       #pragma HLS dependence variable=Vprop_buffer false
                       frontier[nextNumIDX] = col_idx_value + TILE_SIZE*tile;
                       nextNumIDX++;
                    }
                }
            }
            for(int kk = 0; kk < TILE_SIZE; kk += VDATA_SIZE){
            #pragma HLS pipeline II=1
                for(int ii = 0;ii<VDATA_SIZE ;ii++){
                    Vprop[(tile*TILE_SIZE + kk+ii)>>4].data[(tile*TILE_SIZE + kk+ii)%16] =Vprop_buffer[kk+ii];
                }
            }
        }
    }
}
}