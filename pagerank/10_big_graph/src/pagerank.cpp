#include <iostream>

#define VDATA_SIZE 8
#define ALPHA 0.85f
#define MAX_ITER 100
#define EPSILON 1e-6f
#define TILE_SIZE 524288  // 4MB
#define UNROLL_SIZE 8
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE

typedef struct v_datatype { int64_t data[VDATA_SIZE]; } v_dt;

extern "C"{
void pagerank(const v_dt *in1,  // Read-Only Vector 1 from hbm -> col index
              const v_dt *in2,  // Read-Only Vector 2 from hbm -> row ptr
              const v_dt *in3,// Read-Only Vector 2 from hbm -> row ptr 
              float *out1,      // Output Result to hbm -> pagerank score
              float *out2,      // Output Result to hbm -> pagerank score
              int64_t NUM_NODES
              ) {

#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem0 max_widen_bitwidth=512 
#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port = in3 offset = slave bundle = gmem2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port = out1 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = out2 offset = slave bundle = gmem4 

  // set constant
  float base_score = (1.0 - ALPHA) / NUM_NODES;
  
  int64_t row_ptr_buffer[2];
  #pragma HLS array_partition variable=row_ptr_buffer complete
  float out_degree_buffer[2];
  #pragma HLS array_partition variable=out_degree_buffer complete
  int64_t col_idx_buffer[UNROLL_SIZE];
  #pragma HLS array_partition variable=col_idx_buffer factor=VDATA_SIZE cyclic
  float score_buffer[TILE_SIZE];
  #pragma HLS array_partition variable=score_buffer factor=VDATA_SIZE cyclic
  
  
  int64_t out_degree_u;
  int64_t buffer_start;
  int64_t size_;
  float src_pagerank;

  for (int64_t iter = 0; iter < MAX_ITER; iter++) {
  if( (iter %2) == 0){
    for (int64_t i = 0; i < NUM_NODES; i ++) {
      #pragma HLS pipeline II=1
      out2[i] = base_score;
    }
    //push score_new
    for (int64_t tile = 0; tile < NUM_TILES; tile ++) {
      int64_t score_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
      for(int64_t k = 0; k < TILE_SIZE; k++){
        #pragma HLS pipeline II=1
        if ( k < score_size ) {
          score_buffer[k] = out2[tile*TILE_SIZE + k];
        }
      }
      for (int64_t u = 0; u < NUM_NODES; u++) {
        //prefatch row_ptr
        row_ptr_buffer[0]= in2[(u+tile*NUM_NODES)/VDATA_SIZE].data[(u+tile*NUM_NODES)%VDATA_SIZE];
        row_ptr_buffer[1]= in2[(u+tile*NUM_NODES+1)/VDATA_SIZE].data[(u+tile*NUM_NODES+1)%VDATA_SIZE];
        size_ = row_ptr_buffer[1] - row_ptr_buffer[0];
        buffer_start = row_ptr_buffer[0];
        src_pagerank =  out1[u];
        
        out_degree_buffer[0]= in3[u/VDATA_SIZE].data[u%VDATA_SIZE];
        out_degree_buffer[1]= in3[(u+1)/VDATA_SIZE].data[(u+1)%VDATA_SIZE]; 
        out_degree_u = out_degree_buffer[1] - out_degree_buffer[0];
        //push
        if(out_degree_u != 0){
        for (int64_t b = 0; b < size_; b += UNROLL_SIZE) {
          //prefatch col_idx_buffer
          int64_t chunk_size = (b + UNROLL_SIZE > size_) ? size_ - b : UNROLL_SIZE;
          for (int64_t k = 0; k < UNROLL_SIZE; k++) {
            #pragma HLS unroll
            if( k < chunk_size) {
                col_idx_buffer[k]= in1[(b + buffer_start + k)/VDATA_SIZE].data[(b + buffer_start + k)%VDATA_SIZE];
            }
          }
          //SIMD parallel compute
          for (int64_t l = 0; l < UNROLL_SIZE; l++) {
            #pragma HLS unroll 
            if( l < chunk_size) {
              int64_t idx = col_idx_buffer[l] - TILE_SIZE*tile;
              score_buffer[idx] = score_buffer[idx] + (ALPHA * src_pagerank / out_degree_u );
            }
          }
        }
        }//push_one_vertex
      }//push_one_tile
      for(int64_t k = 0; k < TILE_SIZE; k++){
        #pragma HLS pipeline II=1
        if ( k < score_size ) {
          out2[tile*TILE_SIZE + k] = score_buffer[k];
        }
      }
    }//push_all_tile
  }else{
    for (int64_t i = 0; i < NUM_NODES; i ++) {
      #pragma HLS pipeline II=1
      out1[i] = base_score;
    }
    //push score_new
    for (int64_t tile = 0; tile < NUM_TILES; tile ++) {
      int64_t score_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
      for(int64_t k = 0; k < TILE_SIZE; k++){
        #pragma HLS pipeline II=1
        if ( k < score_size ) {
          score_buffer[k] = out1[tile*TILE_SIZE + k];
        }
      }
      for (int64_t u = 0; u < NUM_NODES; u++) {
        //prefatch row_ptr
        row_ptr_buffer[0]= in2[(u+tile*NUM_NODES)/VDATA_SIZE].data[(u+tile*NUM_NODES)%VDATA_SIZE];
        row_ptr_buffer[1]= in2[(u+tile*NUM_NODES+1)/VDATA_SIZE].data[(u+tile*NUM_NODES+1)%VDATA_SIZE];
        size_ = row_ptr_buffer[1] - row_ptr_buffer[0];
        buffer_start = row_ptr_buffer[0];
        src_pagerank =  out2[u];
        
        out_degree_buffer[0]= in3[u/VDATA_SIZE].data[u%VDATA_SIZE];
        out_degree_buffer[1]= in3[(u+1)/VDATA_SIZE].data[(u+1)%VDATA_SIZE]; 
        out_degree_u = out_degree_buffer[1] - out_degree_buffer[0];
        if(out_degree_u != 0){
        //push
        for (int64_t b = 0; b < size_; b += UNROLL_SIZE) {
          //prefatch col_idx_buffer
          int64_t chunk_size = (b + UNROLL_SIZE > size_) ? size_ - b : UNROLL_SIZE;
          for (int64_t k = 0; k < UNROLL_SIZE; k++) {
            #pragma HLS unroll
            if( k < chunk_size) {
                col_idx_buffer[k]= in1[(b + buffer_start + k)/VDATA_SIZE].data[(b + buffer_start + k)%VDATA_SIZE];
            }
          }
          //SIMD parallel compute
          for (int64_t l = 0; l < UNROLL_SIZE; l++) {
            #pragma HLS unroll 
            if( l < chunk_size) {
              int64_t idx = col_idx_buffer[l] - TILE_SIZE*tile;
              score_buffer[idx] = score_buffer[idx] + (ALPHA * src_pagerank / out_degree_u );
            }
          }
        }
        }//push_one_vertex
      }//push_one_tile
      for(int64_t k = 0; k < TILE_SIZE; k++){
        #pragma HLS pipeline II=1
        if ( k < score_size ) {
          out1[tile*TILE_SIZE + k] = score_buffer[k];
        }
      }
    }//push_all_tile
  }//iter change
  
  float diff = 0.0f;
  
  for (int64_t i = 0; i < NUM_NODES; i += 1) {
    #pragma HLS pipeline II=1
    float temp = ((out2[i] - out1[i]) > 0 )? 
                  (out2[i] - out1[i]):
                  (out1[i] - out2[i]);
    diff += temp;
  }
  
  //if converge then break 
  if (diff < EPSILON) {
      break;
  }

  // update score
  if( (iter %2) == 0){
    // update score
    for (int64_t i = 0; i < NUM_NODES; i ++ ) {
      #pragma HLS pipeline II=1
      out1[i] = out2[i];
    }
  }else{
    for (int64_t i = 0; i < NUM_NODES; i ++ ) {
      #pragma HLS pipeline II=1
      out2[i] = out1[i];
    }
  }
  }


}
}