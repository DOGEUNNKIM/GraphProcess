#include <iostream>

#define VDATA_SIZE 16
#define ALPHA 0.85f
#define MAX_ITER 1000
#define EPSILON 1e-6f
#define NUM_NODES 100
#define NUM_EDGES 1000
#define BUFFER_SIZE 16

typedef struct v_datatype { unsigned int data[VDATA_SIZE]; } v_dt;

extern "C"{
void pagerank(const v_dt *in1,  // Read-Only Vector 1 from hbm -> col index
              const v_dt *in2,  // Read-Only Vector 2 from hbm -> row ptr
              float *out1,      // Output Result to hbm -> pagerank score
              float *out2       // Output Result to hbm -> pagerank score
              ) {
#pragma HLS ALLOCATION instances=fmul limit=0 operation
#pragma HLS ALLOCATION instances=fdiv limit=0 operation
#pragma HLS ALLOCATION instances=fadd limit=0 operation
#pragma HLS ALLOCATION instances=fsub limit=0 operation
#pragma HLS ALLOCATION instances=fcmp limit=0 operation

#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem0 max_widen_bitwidth=512 
#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem1 max_widen_bitwidth=512

// random read -> cannot burst read
#pragma HLS INTERFACE m_axi port = out1 offset = slave bundle = gmem2 
#pragma HLS INTERFACE m_axi port = out2 offset = slave bundle = gmem3 

// set constant
float base_score;
base_score = (1.0f - ALPHA) / NUM_NODES;


unsigned int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
unsigned int col_idx_buffer[BUFFER_SIZE];
#pragma HLS array_partition variable=col_idx_buffer factor=16 cyclic


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
  
      //int out_degree_u = row_ptr_buffer[1] - row_ptr_buffer[0];
      int buffer_start = row_ptr_buffer[0];
  
      //push
      for (int b = 0; b < (row_ptr_buffer[1] - row_ptr_buffer[0]); b += BUFFER_SIZE) {
        //prefatch col_idx_buffer
        int chunk_size = (b + BUFFER_SIZE > (row_ptr_buffer[1] - row_ptr_buffer[0])) ? (row_ptr_buffer[1] - row_ptr_buffer[0]) - b : BUFFER_SIZE;

        for (int k = 0; k < BUFFER_SIZE; k++) {
          #pragma HLS unroll
          col_idx_buffer[k] = in1[(b + buffer_start + k)/16].data[(b + buffer_start + k)%16];
        }
        
        //SIMD parallel compute
        for (int k = 0; k < chunk_size; k++) {

          float out2_temp = (chunk_size > BUFFER_SIZE) ? 0.0f : out2[col_idx_buffer[k]];
          float out1_temp = (chunk_size > BUFFER_SIZE) ? 0.0f : out1[u];

          float result;
          result = out2_temp + (ALPHA * out1_temp / (row_ptr_buffer[1] -row_ptr_buffer[0]) );

          out2[col_idx_buffer[k]] = (chunk_size > BUFFER_SIZE) ? 0 :result;
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
  
      int out_degree_u = row_ptr_buffer[1] - row_ptr_buffer[0];
      int buffer_start = row_ptr_buffer[0];
  
      //push
      for (int b = 0; b < out_degree_u; b += BUFFER_SIZE) {
        //prefatch col_idx_buffer
        int chunk_size = (b + BUFFER_SIZE > out_degree_u) ? out_degree_u - b : BUFFER_SIZE;

        for (int k = 0; k < BUFFER_SIZE; k++) {
          #pragma HLS unroll
          col_idx_buffer[k]= in1[(b + buffer_start + k)/16].data[(b + buffer_start + k)%16];
        }
        
        //SIMD parallel compute
        for (int k = 0; k < chunk_size; k++) {
          float out1_temp_1 = (chunk_size > BUFFER_SIZE) ? 0.0f : out1[col_idx_buffer[k]];
          float out2_temp_1 = (chunk_size > BUFFER_SIZE) ? 0.0f : out2[u];

          float result_1;
          result_1 = out1_temp_1 + (ALPHA * out2_temp_1 / (row_ptr_buffer[1] -row_ptr_buffer[0]) );

          out1[col_idx_buffer[k]] = (chunk_size > BUFFER_SIZE) ? 0 :result_1;
        }
      }
    }
  }
  //check converge 
  
  float diff = 0;

  for (int i = 0; i < NUM_NODES; i += 1) {
    #pragma HLS pipeline II=1

    float out2_1 = (NUM_NODES < i) ? 0.0f : out2[i];
    float out1_1 = (NUM_NODES < i) ? 0.0f : out1[i];
    
    diff += (out2_1 - out1_1) * (out2_1 - out1_1);
  }


  //if (diff < 1 ){
  //  break;
  //}
  
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

}
}