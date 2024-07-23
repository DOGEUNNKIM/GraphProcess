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
              v_dt_d *out        // Output Result to hbm -> pagerank score
              ) {

#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem0 max_widen_bitwidth=512 
#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem2 max_widen_bitwidth=512


#pragma HLS RESOURCE variable=out core=AddSub
#pragma HLS RESOURCE variable=out core=Mul_LUT


// set constant

float base_score = (1.0 - ALPHA) / NUM_NODES;

#pragma HLS RESOURCE variable=base_score core=AddSub
#pragma HLS RESOURCE variable=base_score core=Mul_LUT

//set bram
float score[NUM_NODES];
#pragma HLS array_partition variable=score factor=16 cyclic
float score_new[NUM_NODES];
#pragma HLS array_partition variable=score_new factor=16 cyclic


#pragma HLS RESOURCE variable=score_new core=AddSub
#pragma HLS RESOURCE variable=score_new core=Mul_LUT
#pragma HLS RESOURCE variable=score core=AddSub
#pragma HLS RESOURCE variable=score core=Mul_LUT

int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
int col_idx_buffer[BUFFER_SIZE];
#pragma HLS array_partition variable=col_idx_buffer factor=16 cyclic


//init score
for (int i = 0; i < NUM_NODES; i ++) {
    score[i] = 1.0/NUM_NODES;
}

for (int iter = 0; iter < MAX_ITER; iter++) {
  //init score_new
  for (int i = 0; i < NUM_NODES; i ++) {
    score_new[i] = base_score;
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
      prefatch_col_idx_buffer_unroll:
      for (int k = 0; k < chunk_size; k++) {
        col_idx_buffer[k]= in1[(b + buffer_start + k)/16].data[(b + buffer_start + k)%16];
      }
      //SIMD parallel compute
      for (int k = 0; k < chunk_size; k++) {
        int v = col_idx_buffer[k];
        score_new[v] += (ALPHA * score[u] / out_degree_u);
      }
    }
  }

  //check converge 
  float diff = 0.0;
  
  #pragma HLS RESOURCE variable=diff core=AddSub
  #pragma HLS RESOURCE variable=diff core=Mul_LUT
  for (int i = 0; i < NUM_NODES; i += 1) {
    diff = diff + (score_new[i ] - score[i]) * (score_new[i ] - score[i ]);
  }
  
  //if converge then break 
  if (diff < EPSILON) {
      break;
  }

  // update score
  update_score_pipeline:
  for (int i = 0; i < NUM_NODES; i ++) {
    score[i] = score_new[i];
  }
}

// export pagerank score
export_pagerank_score_pipeline:
for (int i = 0; i < NUM_NODES; i += 16) {
	#pragma HLS pipeline II=1
	for (int k = 0; k < 16; k += 1) {
		#pragma HLS unroll
    out[i/16].data[k] = score_new[i+k];
  }
}



}
}