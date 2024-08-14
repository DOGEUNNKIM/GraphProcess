#include <iostream>
#include "ap_fixed.h"

#define VDATA_SIZE 16
#define MAX_ITER 1000
#define NUM_NODES 2048
#define TILE_SIZE 512
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE

typedef struct v_datatype { int data[VDATA_SIZE]; } v_dt;
typedef struct v_f_datatype { float data[VDATA_SIZE]; } v_dt_f;



// use 1 dsp
extern "C"{
void pagerank(const v_dt *in1,  // Read-Only Vector 1 from hbm -> col index
              const v_dt *in2,  // Read-Only Vector 2 from hbm -> row ptr preprocessed
              const v_dt_f *in3,// Read-Only Vector 2 from hbm -> row ptr 
              float *out1,      // Output Result to hbm -> pagerank score before
              float *out2,       // Output Result to hbm -> pagerank score next
              int *iter_count
              ) {

#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem0 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem1 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = in3 offset = slave bundle = gmem2 max_widen_bitwidth=512 depth = 4096
#pragma HLS INTERFACE m_axi port = out1 offset = slave bundle = gmem3 
#pragma HLS INTERFACE m_axi port = out2 offset = slave bundle = gmem4 
#pragma HLS INTERFACE m_axi port = iter_count offset = slave bundle = gmem5 

#pragma HLS INTERFACE s_axilite port=in1 bundle=control
#pragma HLS INTERFACE s_axilite port=in2 bundle=control
#pragma HLS INTERFACE s_axilite port=in3 bundle=control
#pragma HLS INTERFACE s_axilite port=out1 bundle=control
#pragma HLS INTERFACE s_axilite port=out2 bundle=control
#pragma HLS INTERFACE s_axilite port=iter_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control 


float ALPHA = 0.85;
float EPSILON = 1e-6;

// set constant
float base_score;
float a = 1;
base_score = (a - ALPHA) / NUM_NODES;

int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
float out_degree_buffer[2];
#pragma HLS array_partition variable=out_degree_buffer complete
float score_buffer[TILE_SIZE];
#pragma HLS array_partition variable=score_buffer factor=32 cyclic

int col_idx_buffer[VDATA_SIZE];
#pragma HLS array_partition variable=col_idx_buffer factor=16 cyclic

int iter;
for (iter = 0; iter < MAX_ITER; iter++) {
  for (int i = 0; i < NUM_NODES; i ++) {
#pragma HLS pipeline II=1
    out2[i] = base_score;
  }
  //cal score_new
  for (int tile = 0; tile < NUM_TILES; tile ++) {
//#pragma HLS pipeline rewind
    //prefatch score_buffer
    int score_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
    
    for(int k = 0; k < TILE_SIZE; k++){
#pragma HLS pipeline II=1
      if ( k < score_size ) {
        score_buffer[k] = out2[tile*TILE_SIZE + k];
      }
    }

    for (int u = 0; u < NUM_NODES; u++) {
//#pragma HLS pipeline rewind
      //prefatch row_ptr
      row_ptr_buffer[0]= in2[(u+tile*NUM_NODES)/16].data[(u+tile*NUM_NODES)%16];
      row_ptr_buffer[1]= in2[(u+tile*NUM_NODES+1)/16].data[(u+tile*NUM_NODES+1)%16];
      int size_ = row_ptr_buffer[1] - row_ptr_buffer[0];
	    float src_pagerank = out1[u];
      
      out_degree_buffer[0]= in3[u/16].data[u%16];
      out_degree_buffer[1]= in3[(u+1)/16].data[(u+1)%16]; 
      float out_degree_u;
      out_degree_u = out_degree_buffer[1] - out_degree_buffer[0];
      int buffer_start = row_ptr_buffer[0];
      //push
      for (int b = 0; b < NUM_NODES; b += VDATA_SIZE) {
        if(b < size_){
#pragma HLS pipeline
          //prefatch col_idx_buffer
          int chunk_size = (b + VDATA_SIZE > size_) ? size_ - b : VDATA_SIZE;
  
          for (int k = 0; k < VDATA_SIZE; k++) {
#pragma HLS unroll
            col_idx_buffer[k] = in1[(b + buffer_start + k)/16].data[(b + buffer_start + k)%16];
          }
          
          //SIMD parallel compute
          for (int l = 0; l < VDATA_SIZE; l++) {
#pragma HLS unroll 
#pragma HLS dependence variable=score_buffer false                
            if( l < chunk_size) {
            int idx = col_idx_buffer[l] - TILE_SIZE*tile;
            score_buffer[idx] = score_buffer[idx] + (ALPHA * src_pagerank / out_degree_u );
            }
          }
        }
      }
    }
    
    //return score_buffer
    for(int k = 0; k < TILE_SIZE; k++){
#pragma HLS pipeline II=1
      if ( k < score_size ) {
        out2[tile*TILE_SIZE + k] = score_buffer[k];
      }
    }
  }
  
  //check converge & update score
  
  float diff = 0;
  for (int i = 0; i < NUM_NODES; i += 1) {
#pragma HLS pipeline II=1
    diff += (out1[i] - out2[i]) * (out1[i] - out2[i]);
    out1[i] = out2[i];
  }


  if (diff < EPSILON ){
    break;
  }
iter_count[0] = iter;
}


}
}
