#include <iostream>
#include "ap_fixed.h"
#include "hls_stream.h"

#define VDATA_SIZE 16
#define MAX_ITER 1000
#define NUM_NODES 2048
#define TILE_SIZE 512
#define NUM_TILES (NUM_NODES + TILE_SIZE - 1) / TILE_SIZE


typedef struct v_datatype { int data[VDATA_SIZE]; } v_dt;
typedef struct v_f_datatype { float data[VDATA_SIZE]; } v_dt_f;
typedef struct r_f_datatype { int data[2]; } row_dt;

float ALPHA = 0.85;
float EPSILON = 1e-6;

float a = 1;
float base_score = (a - ALPHA) / NUM_NODES;

void InitVprop(hls::stream<float>& init_vprop){
    int start_iter;
    Init1: for(int tile =0 ; tile < NUM_TILES ; tile ++){//start one tile
#pragma HLS UNROLL factor = 16
        int tile_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
        Init2: for(int i = 0 ; i < TILE_SIZE ; i ++){
#pragma HLS pipeline II=1
            if ( i < tile_size ) {
                //out2[tile*TILE_SIZE + i] = base_score;
                init_vprop.write(base_score);
            }
        }
    }  
}

void PrefetchGraph(hls::stream<float>& prefetch_data, hls::stream<row_dt>& prefetch_row_ptr, hls::stream<float>& prefetch_out_deg,
                   const v_dt *in2, const v_dt_f *in3, float *out1) {
    int start_iter;
    float out_degree; 
    float src_pagerank;
    float out_degree_buffer[2];
#pragma HLS array_partition variable=out_degree_buffer complete
    int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
    row_dt row_stream;
    int start_row0 = in2[0].data[0];
    int start_row1 = in2[0].data[1];
    int start_out0 = in3[0].data[0];
    int start_out1 = in3[0].data[1];


    Prefetch1: for(int tile = 0; tile < NUM_TILES; tile++) { // start one tile
#pragma HLS UNROLL factor = 16
        Prefetch2: for(int src = 0; src < NUM_NODES; src++) { // start one tile line
#pragma HLS LOOP_FLATTEN
#pragma HLS pipeline II=1
            // row_ptr
            if(tile ==0 && src ==0 ){
                row_ptr_buffer[0] = start_row0;
                row_ptr_buffer[1] = start_row1;
            } else {
                row_ptr_buffer[0] = row_ptr_buffer[1];
                row_ptr_buffer[1] = in2[(src + tile * NUM_NODES + 1) >> 4].data[(src + tile * NUM_NODES + 1) % 16];
            }
            row_stream.data[0] = row_ptr_buffer[0] ;
            row_stream.data[1] = row_ptr_buffer[1] ;
            prefetch_row_ptr.write(row_stream);

            // src
            src_pagerank = out1[src];
            prefetch_data.write(src_pagerank);

            // out_deg
            if(tile ==0 && src ==0 ){
                out_degree_buffer[0] = start_out0;
                out_degree_buffer[1] = start_out1;
            }else{
                out_degree_buffer[0] = out_degree_buffer[1];
                out_degree_buffer[1] = in3[(src + 1) >> 4].data[(src + 1) % 16];
            }
            out_degree = out_degree_buffer[1] - out_degree_buffer[0];
            prefetch_out_deg.write(out_degree);
        }
    }
}

void ProcessingElement(hls::stream<float>& init_vprop, hls::stream<float>& finish_vprop,
                       hls::stream<float>& prefetch_data, hls::stream<row_dt>& prefetch_row_ptr, hls::stream<float>& prefetch_out_deg,
                       const v_dt *in1, float *out2){
    float score_buffer[TILE_SIZE];
    float src_pagerank;
    float out_degree;
    row_dt row_stream;
    int row_ptr_buffer[2];
#pragma HLS array_partition variable=row_ptr_buffer complete
    int delta_row_ptr;
    int col_idx;
    int buffer_start;
    int start_iter;
    float attr;

    Process1: for(int tile = 0 ; tile < NUM_TILES ; tile ++){//start one tile
#pragma HLS UNROLL factor = 16
        int tile_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
        Process2: for(int i = 0 ; i < TILE_SIZE ; i ++){//prefetch one tile data
#pragma HLS pipeline II=1
            if(i < tile_size ){
                score_buffer[i] = init_vprop.read();
            }
        }
        Process3: for(int src = 0 ; src < NUM_NODES ; src ++){//calculate one tile
            //read
            src_pagerank = prefetch_data.read();
            out_degree = prefetch_out_deg.read();
            row_stream = prefetch_row_ptr.read();
            row_ptr_buffer[0] = row_stream.data[0];
            row_ptr_buffer[1] = row_stream.data[1];
            delta_row_ptr = row_ptr_buffer[1] - row_ptr_buffer[0];
            buffer_start = row_ptr_buffer[0];
            if( delta_row_ptr > static_cast<float>(0)){
                attr = ALPHA * src_pagerank / out_degree;
            }

            Process4: for(int i = 0 ; i < delta_row_ptr ; i ++){//calculate one tile line
#pragma HLS pipeline II=1
#pragma HLS dependence variable=score_buffer false
                col_idx = in1[(buffer_start + i) >> 4].data[(buffer_start + i)%16];
                score_buffer[col_idx - TILE_SIZE*tile] += attr;
            }
        }
        Process5: for(int i = 0 ; i < TILE_SIZE ; i ++){//prefetch one tile data
            if(i < tile_size ){
#pragma HLS pipeline II=1
                out2[TILE_SIZE*tile + i] = score_buffer[i];
                finish_vprop.write(score_buffer[i]);
            }
        }
    }
}


float IterationCheck(hls::stream<float>& finish_vprop,
                        float *out1){
    float diff[TILE_SIZE];
#pragma HLS array_partition variable=diff complete
    float diff_all=0;
    for(int i = 0;i<TILE_SIZE;i++){
        diff[i] = 0;
    }
    
    float before_vertex;
    float after_vertex;
    Iter1: for(int tile = 0; tile < NUM_TILES; tile ++){//start one tile
#pragma HLS UNROLL factor = 16
        int tile_size = ( (tile + 1) * TILE_SIZE > NUM_NODES) ? NUM_NODES - TILE_SIZE*tile : TILE_SIZE;
        Iter2: for(int i = 0 ; i < TILE_SIZE ; i ++){//get data
#pragma HLS pipeline II=1
            if(i < tile_size ){
                before_vertex = out1[tile* TILE_SIZE +i];
                after_vertex = finish_vprop.read();
                diff[i] += ( (after_vertex-before_vertex) * (after_vertex-before_vertex));
                //#pragma HLS dependence variable=diff false
            }
        }
    }
    Iter3: for(int i = 0;i<TILE_SIZE;i++){
#pragma HLS pipeline
        diff_all += diff[i];
    }

    return diff_all;
}

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

#pragma HLS DATAFLOW

hls::stream<float,NUM_NODES> finish_vprop; 
hls::stream<float,NUM_NODES> init_vprop; 

hls::stream<float,NUM_NODES*NUM_TILES> prefetch_data;
hls::stream<float,NUM_NODES*NUM_TILES> prefetch_out_deg;
hls::stream<row_dt,NUM_NODES*NUM_TILES> prefetch_row_ptr; 

float diff = 100;
int iter =0;
main1: for(iter = 0; iter < MAX_ITER; iter ++){
    if ( iter%2 == 0 ){
        InitVprop(init_vprop);
        PrefetchGraph( prefetch_data, prefetch_row_ptr, prefetch_out_deg, in2, in3, out1);
        ProcessingElement( init_vprop, finish_vprop, prefetch_data, prefetch_row_ptr, prefetch_out_deg, in1, out2);
        diff = IterationCheck( finish_vprop, out1);
    }else{
        InitVprop(init_vprop);
        PrefetchGraph( prefetch_data, prefetch_row_ptr, prefetch_out_deg, in2, in3, out2);
        ProcessingElement( init_vprop, finish_vprop, prefetch_data, prefetch_row_ptr, prefetch_out_deg, in1, out1);
        diff = IterationCheck( finish_vprop, out2);
    }
    
    if(diff < EPSILON){
        break;
        
    }
    
}
iter_count[0] = iter;

}
}