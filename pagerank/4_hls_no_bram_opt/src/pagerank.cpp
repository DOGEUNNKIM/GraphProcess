#include <iostream>

#define VDATA_SIZE 16
#define ALPHA 0.85
#define MAX_ITER 1000
#define EPSILON 1e-6
#define NUM_NODES 100
#define NUM_EDGES 1000
#define BUFFER_SIZE 16

typedef struct v_datatype { unsigned int data[VDATA_SIZE]; } v_dt;
typedef struct v_datatype_d { float data[VDATA_SIZE]; } v_dt_d;

extern "C"{
void pagerank(const v_dt *in1,  // Read-Only Vector 1 from hbm -> col index
              const v_dt *in2,  // Read-Only Vector 2 from hbm -> row ptr
              float *out1,      // Output Result to hbm -> pagerank score
              float *out2,      // Output Result to hbm -> pagerank score
              v_dt *out_row,        // Output Result to hbm -> row ptr
              v_dt *out_col        // Output Result to hbm -> col index
              ) {

#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem0 max_widen_bitwidth=512 
#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem1 max_widen_bitwidth=512

// random read -> cannot burst read
#pragma HLS INTERFACE m_axi port = out1 offset = slave bundle = gmem2 
#pragma HLS INTERFACE m_axi port = out2 offset = slave bundle = gmem3 

#pragma HLS INTERFACE m_axi port = out_row offset = slave bundle = gmem4 max_widen_bitwidth=512 
#pragma HLS INTERFACE m_axi port = out_col offset = slave bundle = gmem5 max_widen_bitwidth=512
// set constant

float base_score = (1.0 - ALPHA) / NUM_NODES;

int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
int col_idx_buffer[BUFFER_SIZE];
#pragma HLS array_partition variable=col_idx_buffer factor=16 cyclic

int row_ptr[NUM_NODES+1];
#pragma HLS array_partition variable=row_ptr factor=16 cyclic
int col_idx[NUM_EDGES];
#pragma HLS array_partition variable=col_idx factor=16 cyclic

int out_degree_u;
int buffer_start;

iter:
for (int iter = 0; iter < MAX_ITER; iter++) {
  if( (iter%2) == 0){
    for (int i = 0; i < NUM_NODES; i ++) {
      #pragma HLS pipeline II=1
      out2[i] = base_score;
    }
    //cal score_new
    for (int u = 0; u < NUM_NODES; u++) {
      //prefatch row_ptr
      row_ptr_buffer[0]= in2[u/16].data[u%16];
      row_ptr_buffer[1]= in2[(u+1)/16].data[(u+1)%16];
      out_degree_u = row_ptr_buffer[1] - row_ptr_buffer[0];
      buffer_start = row_ptr_buffer[0];

      row_ptr[u] = row_ptr_buffer[0];
      row_ptr[u+1] = row_ptr_buffer[1];
  
      //push
      for (int b = 0; b < out_degree_u; b += BUFFER_SIZE) {
        //prefatch col_idx_buffer
        int chunk_size = (b + BUFFER_SIZE > out_degree_u) ? out_degree_u - b : BUFFER_SIZE;

        //for (int k = 0; k < chunk_size; k++) {
        for (int k = 0; k < BUFFER_SIZE; k++) {
          #pragma HLS unroll
          col_idx_buffer[k]= in1[(b + buffer_start + k)/16].data[(b + buffer_start + k)%16];
        }
        
        //for (int k = 0; k < chunk_size; k++) {
        for (int k = 0; k < BUFFER_SIZE; k++) {
          col_idx[b + buffer_start + k] = col_idx_buffer[k];
        }
        //SIMD parallel compute
        for (int k = 0; k < chunk_size; k++) {
          int v = col_idx_buffer[k];
          out2[v] += (ALPHA * out1[u] / out_degree_u);
        }
      }
    }
  }else{
    for (int i = 0; i < NUM_NODES; i ++) {
      #pragma HLS pipeline II=1
      out1[i] = base_score;
    }
    //cal score_new
    for (int u = 0; u < NUM_NODES; u++) {
      //prefatch row_ptr
      
      row_ptr_buffer[0]= in2[u/16].data[u%16];
      row_ptr_buffer[1]= in2[(u+1)/16].data[(u+1)%16];
      out_degree_u = row_ptr_buffer[1] - row_ptr_buffer[0];
      buffer_start = row_ptr_buffer[0];
      
      
      row_ptr[u] = row_ptr_buffer[0];
      row_ptr[u+1] = row_ptr_buffer[1];
  
      //push
      for (int b = 0; b < out_degree_u; b += BUFFER_SIZE) {
        //prefatch col_idx_buffer
        int chunk_size = (b + BUFFER_SIZE > out_degree_u) ? out_degree_u - b : BUFFER_SIZE;

        for (int k = 0; k < BUFFER_SIZE; k++) {
          #pragma HLS unroll
          col_idx_buffer[k]= in1[(b + buffer_start + k)/16].data[(b + buffer_start + k)%16];
        }
        
        for (int k = 0; k < chunk_size; k++) {
          col_idx[b + buffer_start + k] = col_idx_buffer[k];
        }
        //SIMD parallel compute
        for (int k = 0; k < chunk_size; k++) {
          int v = col_idx_buffer[k];
          out1[v] += (ALPHA * out2[u] / out_degree_u);
        }

      }
    }
  }
  
  float diff = 0.0;
  
  for (int i = 0; i < NUM_NODES; i += 1) {
    #pragma HLS pipeline II=1
    float temp = (out2[i] - out1[i]);
    diff += temp * temp;
  }
  
  //if converge then break 
  if (diff < EPSILON) {
      break;
  }

  // update score
  if( (iter %2) == 0){
    // update score
    for (int i = 0; i < NUM_NODES; i ++ ) {
      #pragma HLS pipeline II=1
      out1[i] = out2[i];
    }
  }else{
    for (int i = 0; i < NUM_NODES; i ++ ) {
      #pragma HLS pipeline II=1
      out2[i] = out1[i];
    }
  }
}


for (int i = 0; i < NUM_NODES+1; i += 16) {
	#pragma HLS pipeline II=1
	for (int k = 0; k < 16; k += 1) {
		#pragma HLS unroll
    out_row[i/16].data[k] = row_ptr[i+k] ;
  }
}

for (int i = 0; i < NUM_EDGES; i += 16) {
	#pragma HLS pipeline II=1
	for (int k = 0; k < 16; k += 1) {
		#pragma HLS unroll
    out_col[i/16].data[k] = col_idx[i+k] ;
  }
}

}
}