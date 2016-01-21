#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
#define CUDA_CHECK_KERNEL {cudaError_t error = cudaGetLastError(); if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

#define BLOCKSIZE 1024
#define N 10

__global__ void count_combinations(int64_t *preset, uint64_t *found_combinations){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;

	int8_t cols[N];

// for testing Use only
/*
  int64_t bv = preset[4 * idx];
  int64_t bh = preset[4 * idx + 1];
  int64_t bu = preset[4 * idx + 2];
  int64_t bd = preset[4 * idx + 3];
*/
  int64_t bh = 0;
  int64_t bu = 0;
  int64_t bd = 0;
  
  //TODO Algo starts here

  //blockvectors for a new position
    int64_t tbh; 
    int64_t tbu; 
    int64_t tbd; 

    int64_t free_vector; //has 1 in positions which are free in the
                         //current col (opposite bh)

    int32_t cc = 0; //current column
   
    for(int32_t i=0;i<N;i++){
        cols[i] = -1; //this value means currently not set
    }

    while(1){
        if(cols[cc] > -1){
            //delete the old placement in the blockvectors (if any)
            tbh = 1 << cols[cc];
            tbu = 1 << (cols[cc] + cc);
            tbd = 1 << (cols[cc] + N - 1 - cc);
            
            bh -= tbh;
            bu -= tbu;
            bd -= tbd;
        }

        cols[cc]++;

        //find the next new placement in the current col
        free_vector = ~(bh | (bu >> cc) | (bd >> (N -1 - cc)));
        while( (cols[cc] < N) && 
                !((free_vector >> cols[cc]) & 1) ){
            cols[cc]++;
        } //postcondition: cols[cc] is at free position or equal to N

        if(cols[cc] >= N){ //track back
            if(cc == 0){ //stop
                break;
            }
            cols[cc] = -1;
            cc--;
        } else {
            //set the new placement in the blockvectors
            tbh = 1 << cols[cc];
            tbu = 1 << (cols[cc] + cc);
            tbd = 1 << (cols[cc] + N - 1 - cc);

            bh += tbh;
            bu += tbu;
            bd += tbd;
            
            if(cc == N - 1){
                found_combinations[idx]++;
            } else {
                cc++;
            }
        }
    }
    return;
}
  




int main(int argc, char ** argv){
    if(argc < 2){
        printf("Not enough arguments\n");
        return 1;
    }
    //int32_t N = atoi(argv[1]);


		//prefer Use of L1-Cache
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1););

	
	int blocksize=BLOCKSIZE;
	dim3 dimBlock(blocksize);
	dim3 dimGrid(1);
	
	//TODO
	int size = blocksize;
	
	uint64_t * found_combinations_host = (uint64_t *) malloc(blocksize*sizeof(uint64_t));
	for(int i = 0; i < blocksize; i++){
		found_combinations_host[i] = 0;
	}
	
	
	int64_t preset_host[blocksize*4];
	int64_t *preset_device;
	uint64_t *found_combinations_device;
	
	CUDA_CHECK(cudaMalloc((void**)&preset_device,size*4*sizeof(int64_t)));
	CUDA_CHECK(cudaMalloc((void**)&found_combinations_device,size*sizeof(uint64_t)));
	
	CUDA_CHECK(cudaMemcpy(preset_device,preset_host,size*4*sizeof(int64_t),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(found_combinations_device,found_combinations_host,size*sizeof(uint64_t),cudaMemcpyHostToDevice));
	
	
	//TODO
	count_combinations<<<dimGrid,dimBlock>>>(preset_device, found_combinations_device);
	CUDA_CHECK_KERNEL
	
	
	
	CUDA_CHECK(cudaMemcpy(found_combinations_host,found_combinations_device,size*sizeof(uint64_t),cudaMemcpyDeviceToHost));
	
	uint64_t found_combinations = 0;
	for(int i = 0; i < size; i++){
		found_combinations += found_combinations_host[i];
	}
	printf("\nFound Combinations: %" PRIu64 "\n", found_combinations);
	
	
	CUDA_CHECK(cudaFree(found_combinations_device));
	CUDA_CHECK(cudaFree(preset_device));
	
}
